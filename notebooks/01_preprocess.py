# Databricks notebook source
# COMMAND ----------
## Use case: An agent that answers questions about Eurovision.
from datetime import datetime

import arxiv
import polars as pl
import wikipediaapi
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from eurovision_voting_bloc_party.config import get_env, load_config
from eurovision_voting_bloc_party.utils import (
    TABLE_NAME,
    TABLE_NAME_ARXIV,
    TABLE_NAME_WIKI,
    load_eurovision_data_from_kaggle,
    prepare_eurovision_tabular_data,
    write_to_delta_table,
)

# COMMAND ----------
# Create Spark session
# (make sure to have Databricks Connect set up before doing this, with compute attached)
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("project_config.yaml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} ready")

# COMMAND ----------
# Section 1: Load Kaggle CSV and write to Delta table
kaggle_dict = {
    type: load_eurovision_data_from_kaggle(type)
    for type in ["contest", "country", "song"]
}

eurovision_dataset = prepare_eurovision_tabular_data(kaggle_dict)

# Create DataFrame and write to Delta table
df = spark.createDataFrame(eurovision_dataset.to_arrow())
write_to_delta_table(df, CATALOG, SCHEMA, TABLE_NAME)


# COMMAND ----------
# Verify the data - read back the table and show some records
def read_delta_table(catalog: str, schema: str, table_name: str) -> pl.DataFrame:
    df = spark.table(f"{catalog}.{schema}.{table_name}")

    logger.info(f"Table: {catalog}.{schema}.{table_name}")
    logger.info(f"Total records: {df.count()}")
    logger.info("Schema:")
    df.printSchema()

    logger.info("Sample records:")
    df.head(5)


read_delta_table(CATALOG, SCHEMA, TABLE_NAME)


# COMMAND ----------
# Section 2: Wikipedia ingestion
years = [str(year) for year in range(1956, 2024)]
wiki = wikipediaapi.Wikipedia(user_agent="EurovisionVotingBlocParty/1.0", language="en")


def fetch_wikipedia_page(year: str) -> str:
    page = wiki.page(f"Eurovision_Song_Contest_{year}")
    if page.exists():
        return {
            "year": year,
            "title": page.title,
            "text": page.text,
            "summary": page.summary,
        }


wikipedia_data = [fetch_wikipedia_page(year) for year in years]

wikipedia_spark_df = spark.createDataFrame(wikipedia_data)

write_to_delta_table(wikipedia_spark_df, CATALOG, SCHEMA, TABLE_NAME_WIKI)
read_delta_table(CATALOG, SCHEMA, TABLE_NAME_WIKI)


# COMMAND ----------
# Section 3: ArXiv ingestion
def fetch_arxiv_data(query: str = "eurovision", max_results: int = 50) -> list:
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in search.results():
        paper = {
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": int(result.published.strftime("%Y%m%d%H%M")),
            "updated": result.updated.isoformat() if result.updated else None,
            "categories": ", ".join(result.categories),
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "ingestion_timestamp": datetime.now().isoformat(),
        }
        results.append(paper)
    logger.info(f"Fetched {len(results)} papers from arXiv for query '{query}'")
    return results


papers_list = fetch_arxiv_data(max_results=50)
logger.info("Sample paper:")
logger.info(f"Title: {papers_list[0]['title']}")
logger.info(f"Authors: {papers_list[0]['authors']}")
logger.info(f"arXiv ID: {papers_list[0]['arxiv_id']}")

arxiv_schema = StructType(
    [
        StructField("arxiv_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("authors", ArrayType(StringType()), True),  # Array of strings
        StructField("summary", StringType(), True),
        StructField("published", LongType(), True),
        StructField("updated", StringType(), True),
        StructField("categories", StringType(), True),
        StructField("pdf_url", StringType(), True),
        StructField("primary_category", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
    ]
)

arxiv_spark_df = spark.createDataFrame(papers_list, schema=arxiv_schema)

write_to_delta_table(arxiv_spark_df, CATALOG, SCHEMA, TABLE_NAME_ARXIV)
read_delta_table(CATALOG, SCHEMA, TABLE_NAME_ARXIV)

# COMMAND ----------
# Section 3: Experiment with LLMs
w = WorkspaceClient()

host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(api_key=token, base_url=f"{host.rstrip('/')}/serving-endpoints")
model_name = "databricks-llama-4-maverick"

query = f"""
  SELECT
      country,
      COUNT(*) as participations,
      SUM(CASE WHEN final_place = 1 THEN 1 ELSE 0 END) as wins,
      AVG(final_place) as avg_place
  FROM {CATALOG}.{SCHEMA}.{TABLE_NAME}
  WHERE final_place IS NOT NULL
  GROUP BY country
  HAVING COUNT(*) >= 5
  ORDER BY participations DESC, wins ASC
  """


summary_df = spark.sql(query)
summary_text = summary_df.toPandas().to_markdown(index=False)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant with access to "
                "Eurovision Song Contest data. "
                "Use the provided data to answer questions accurately. "
                "Be concise and direct. "
                "Answer only what is asked without explaining your reasoning. "
                "If the data is insufficient to answer, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Eurovision participation data:\n\n{summary_text}\n\n"
                "Which countries participated the most times but with "
                "the least success (fewest wins)? "
                "Tell a story about the number of times they participated, "
                "the amount of wins over the years, "
                "and even if they didn't win, how they placed. Make it witty."
            ),
        },
    ],
    max_tokens=500,
    temperature=0.7,
)

logger.info("Response:")
logger.info(response.choices[0].message.content)
logger.info(f"Tokens used: {response.usage.total_tokens}")
logger.info(f"Input tokens: {response.usage.prompt_tokens}")
logger.info(f"Output tokens: {response.usage.completion_tokens}")

# COMMAND ----------
