# Databricks notebook source
# COMMAND ----------
## Use case: An agent that answers questions about Eurovision.
import kagglehub
import polars as pl
from kagglehub import KaggleDatasetAdapter
from loguru import logger
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
