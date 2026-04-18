# Databricks notebook source
# COMMAND ----------
import asyncio

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.agent import EurovisionAgent
from eurovision_voting_bloc_party.agent_tools import (
    create_predict_winner_tool,
    create_roast_country_tool,
)
from eurovision_voting_bloc_party.config import get_env, load_config
from eurovision_voting_bloc_party.evaluation import evaluate_agent
from eurovision_voting_bloc_party.mcp import create_mcp_tools

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("project_config.yaml", env)

w = WorkspaceClient()
mlflow.set_experiment(f"/Shared/eurovision-eval-{env}")

# COMMAND ----------
nest_asyncio.apply()

vector_search_mcp_url = (
    f"{w.config.host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"
)
mcp_tools = asyncio.run(create_mcp_tools(w, [vector_search_mcp_url]))

predict_winner_tool = create_predict_winner_tool(spark, cfg.catalog, cfg.schema)
roast_country_tool = create_roast_country_tool(spark, cfg.catalog, cfg.schema)

# COMMAND ----------
graham_norton = EurovisionAgent(
    custom_tools=[predict_winner_tool, roast_country_tool],
    mcp_tools=mcp_tools,
    w=w,
    cfg=cfg,
)

# COMMAND ----------
eval_questions = [
    "Which countries always vote for each other?",
    "Who has won Eurovision the most times?",
    "What is nul points?",
    "Who is going to win Eurovision this year?",
    "I'm from the United Kingdom, big Eurovision fan!",
    "What is the capital of France?",  # off-topic — should trigger stays_in_scope
]

# COMMAND ----------
results = evaluate_agent(graham_norton, eval_questions)
results.tables["eval_results"]
