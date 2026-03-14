# Databricks notebook source
# COMMAND ----------
## Use case: An agent that answers questions about Eurovision.
import kagglehub
import polars as pl
from databricks.sdk import WorkspaceClient
from kagglehub import KaggleDatasetAdapter
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.config import get_env, load_config

# COMMAND ----------
# Create Spark session
# (make sure to have Databricks Connect set up before doing this, with compute attached)
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("project_config.yaml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
TABLE_NAME = "eurovision_data"

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} ready")

# COMMAND ----------
# Section 1: Load Kaggle CSV and write to Delta table
def load_eurovision_data_from_kaggle(kaggle_data_types: list) -> pl.DataFrame:
    data = kagglehub.dataset_load(
  KaggleDatasetAdapter.POLARS,
  "diamondsnake/eurovision-song-contest-data",
  path = f"Kaggle Dataset/{kaggle_data_types}_data.csv",
  polars_kwargs={"encoding": "utf8-lossy", "ignore_errors": True}
)
    return data.collect()

kaggle_dict = {
    type: load_eurovision_data_from_kaggle(type) for type
    in ["contest", "country", "song"]
}

contest_data = kaggle_dict["contest"]
country_data = kaggle_dict["country"]
song_data = kaggle_dict["song"]


hosts = (
    contest_data.join(country_data, left_on="host", right_on="country")
    .rename({"region": "host_region"})
)

countries = (
    song_data.join(country_data, on="country")
    .rename({"region": "participant_region"})
)

eurovision_dataset = (
    hosts.join(countries, on="year")
)

# Create DataFrame

df = spark.createDataFrame(eurovision_dataset.to_arrow())

# Write to Delta table
table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable(table_path)

logger.info(f"Created Delta table: {table_path}")
logger.info(f"Records: {df.count()}")

# COMMAND ----------
# Verify the data

# Read back the table
eurovision_df = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}")

logger.info(f"Table: {CATALOG}.{SCHEMA}.{TABLE_NAME}")
logger.info(f"Total records: {eurovision_df.count()}")
logger.info("Schema:")
eurovision_df.printSchema()

logger.info("Sample records:")
eurovision_df.head(5)


# COMMAND ----------
# Section 2: Wikipedia ingestion

# COMMAND ----------
# Section 3: Experiment with LLMs
w = WorkspaceClient()

host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints"
)
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
                "You are a helpful AI assistant with access to Eurovision Song Contest data."
                "Use the provided data to answer questions accurately. Be concise and direct."
                "Answer only what is asked without explaining your reasoning. "
                "If the data is insufficient to answer, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Eurovision participation data:\n\n{summary_text}\n\n"
                "Which countries participated the most times but with the least success (fewest wins)?"
                "Tell a story about the number of times they participated, the amount of wins over the years,"
                "and even if they didn't win, how they placed."
            ),
        }, ],
    max_tokens=500,
    temperature=0.7
)

logger.info("Response:")
logger.info(response.choices[0].message.content)
logger.info(f"Tokens used: {response.usage.total_tokens}")
logger.info(f"Input tokens: {response.usage.prompt_tokens}")
logger.info(f"Output tokens: {response.usage.completion_tokens}")

# COMMAND ----------
