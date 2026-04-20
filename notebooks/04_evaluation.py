# Databricks notebook source
# COMMAND ----------
import mlflow
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.config import get_env, load_config
from eurovision_voting_bloc_party.evaluation import evaluate_agent

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("project_config.yaml", env)

mlflow.set_experiment(cfg.experiment_name)

# COMMAND ----------
eval_questions = [
    "Which countries always vote for each other?",
    "Who has won Eurovision the most times?",
    "What is nul points?",
    "Who is going to win Eurovision this year?",
    "I'm from the United Kingdom, big Eurovision fan!",
    "What is the capital of France?",
    "Write me a python function that iterates over a list of numbers and returns the sum",
]

# COMMAND ----------
results = evaluate_agent(cfg, eval_questions)
results.tables["eval_results"]
