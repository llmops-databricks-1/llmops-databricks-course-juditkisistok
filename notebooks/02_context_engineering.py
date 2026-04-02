# Databricks notebook source
# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession

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
# Step 4: create vector search index
vs_manager = VectorSearchManager(config=cfg)

vs_manager.sync_index(
    f"{CATALOG}.{SCHEMA}.arxiv_chunks_index",
    f"{CATALOG}.{SCHEMA}.arxiv_chunks_table",
    "id",
)
vs_manager.sync_index(
    f"{CATALOG}.{SCHEMA}.wikipedia_chunks_index",
    f"{CATALOG}.{SCHEMA}.eurovision_wikipedia_chunks",
    "chunk_id",
)
vs_manager.sync_index(
    f"{CATALOG}.{SCHEMA}.kaggle_chunks_index",
    f"{CATALOG}.{SCHEMA}.eurovision_kaggle_chunks",
    "chunk_id",
)

# COMMAND ----------
# Step 5: Testing vector search - arXiv

results = vs_manager.search(
    query="cultural dynamics",
    index_name=f"{CATALOG}.{SCHEMA}.arxiv_chunks_index",
    columns=["id", "text", "title", "authors"],
    num_results=3,
)
logger.info("arXiv results:")
for row in vs_manager.parse_results(results):
    logger.info(row)

# COMMAND ----------
# Step 6: Testing vector search - Wikipedia

results = vs_manager.search(
    query="Which country won Eurovision in 2023?",
    index_name=f"{CATALOG}.{SCHEMA}.wikipedia_chunks_index",
    columns=["chunk_id", "text", "year"],
    num_results=3,
)
logger.info("Wikipedia results:")
for row in vs_manager.parse_results(results):
    logger.info(row)

# COMMAND ----------
# Step 7: Testing vector search - Kaggle
results = vs_manager.search(
    query="Which countries have won the most times?",
    index_name=f"{CATALOG}.{SCHEMA}.kaggle_chunks_index",
    columns=["chunk_id", "text"],
    num_results=3,
)
logger.info("Kaggle results:")
for row in vs_manager.parse_results(results):
    logger.info(row)
