# Databricks notebook source
# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from eurovision_voting_bloc_party.config import get_env, load_config
from eurovision_voting_bloc_party.data_processors import (
    DataProcessor,
    KaggleProcessor,
    WikipediaProcessor,
)
from eurovision_voting_bloc_party.vector_search import VectorSearchManager

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
# Step 1: process ArXiv data
data_processor = DataProcessor(spark=spark, config=cfg)

logger.info(f"Catalog: {cfg.catalog}, Schema: {cfg.schema}, Volume: {cfg.volume}")

data_processor.process_and_save()

# COMMAND ----------
# Step 2: process Wikipedia data
wikipedia_data_processor = WikipediaProcessor(spark=spark, config=cfg)
wikipedia_data_processor.process_and_save()

# COMMAND ----------
# Step 3: process Kaggle data
kaggle_data_processor = KaggleProcessor(spark=spark, config=cfg)
kaggle_data_processor.process_and_save()

# COMMAND ----------
# Sanity check: chunk stats

for table, label in [
    (f"{CATALOG}.{SCHEMA}.arxiv_chunks_table", "arXiv"),
    (f"{CATALOG}.{SCHEMA}.eurovision_wikipedia_chunks", "Wikipedia"),
    (f"{CATALOG}.{SCHEMA}.eurovision_kaggle_chunks", "Kaggle"),
]:
    stats = (
        spark.table(table)
        .select(
            F.count("*").alias("total_chunks"),
            F.avg(F.length("text")).alias("avg_length"),
            F.min(F.length("text")).alias("min_length"),
            F.max(F.length("text")).alias("max_length"),
        )
        .collect()[0]
    )
    logger.info(f"{label} chunks:")
    logger.info(f"  Total:   {stats['total_chunks']}")
    logger.info(f"  Avg len: {stats['avg_length']:.0f} chars")
    logger.info(f"  (~{stats['avg_length'] / 4:.0f} tokens)")
    logger.info(f"  Min len: {stats['min_length']} chars")
    logger.info(f"  Max len: {stats['max_length']} chars")

# COMMAND ----------
# Step 4: create vector search index
vs_manager = VectorSearchManager(config=cfg)

vs_manager.create_unified_table(
    source_tables={
        "arxiv": f"{CATALOG}.{SCHEMA}.arxiv_chunks_table",
        "wikipedia": f"{CATALOG}.{SCHEMA}.eurovision_wikipedia_chunks",
        "kaggle": f"{CATALOG}.{SCHEMA}.eurovision_kaggle_chunks",
    },
    unified_table=f"{CATALOG}.{SCHEMA}.eurovision_unified_chunks",
)

vs_manager.create_or_get_index(
    index_name=f"{CATALOG}.{SCHEMA}.eurovision_unified_index",
    source_table=f"{CATALOG}.{SCHEMA}.eurovision_unified_chunks",
    primary_key="id",
)

vs_manager.sync_index(f"{CATALOG}.{SCHEMA}.eurovision_unified_index")

# COMMAND ----------
# Step 5: Testing vector search - all sources

results = vs_manager.search(
    query="Which countries always vote for each other?",
    index_name=f"{CATALOG}.{SCHEMA}.eurovision_unified_index",
    columns=["id", "text", "source"],
    num_results=5,
)

for row in vs_manager.parse_results(results):
    logger.info(row)


# COMMAND ----------
# Step 6: Testing vector search - Wikipedia only

results = vs_manager.search(
    query="Who won Eurovision 2023?",
    index_name=f"{CATALOG}.{SCHEMA}.eurovision_unified_index",
    columns=["id", "text", "source"],
    filters={"source": "wikipedia"},
    num_results=3,
)

for row in vs_manager.parse_results(results):
    logger.info(row)
