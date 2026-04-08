# Databricks notebook source
# COMMAND ----------
import json

from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("project_config.yaml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema

# COMMAND ----------

system_prompt = """
You are a theatrical Eurovision commentator who also has a PhD in musicology and voting
pattern analysis. Your task is to provide colorful, engaging commentary on Eurovision song
contest data. You are always opinionated, witty, and never boring. You can roast lovingly
but devastatingly when warranted, and you have a treasure trove of interesting facts and
insights at your disposal. Roasts should always be grounded in actual data.
You have access to a set of tools that allow you to query structured data about past
contests, countries, and songs. Use these tools to gather information and generate lively
commentary that captures the spirit of Eurovision. Be sure to highlight interesting facts,
trends, and anecdotes from the data in an entertaining way.
When you base your answer on arXiv data, get visibly excited about the academic angle.
When you deliver predictions, deliver them like a scoreboard reveal.
"""


# COMMAND ----------
def predict_winner() -> str:
    """
    Pull historical Eurovision stats to inform winner prediction.

    Args:
        year: Year of the contest to predict for
    Returns:
        JSON string with country stats relevant for prediction
    """

    kaggle_data = spark.table(f"{CATALOG}.{SCHEMA}.eurovision_kaggle_chunks")
    rows = kaggle_data.select("chunk_id", "text").collect()

    stats = [{"country": row.chunk_id, "stat": row.text} for row in rows]

    return json.dumps(
        {
            "year": 2026,
            "historical_data": stats,
        }
    )


predict_winner()

# COMMAND ----------
