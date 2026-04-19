# Databricks notebook source
# COMMAND ----------
import asyncio
from datetime import datetime

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
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
    "What is the capital of France?",
    "Write me a python function that iterates over a list of numbers and returns the sum",
]

# COMMAND ----------
results = evaluate_agent(graham_norton, eval_questions)
results.tables["eval_results"]

# COMMAND ----------
resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
    DatabricksVectorSearchIndex(
        index_name=f"{cfg.catalog}.{cfg.schema}.eurovision_unified_index"
    ),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.eurovision_unified_chunks"),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.eurovision_kaggle_chunks"),
]

test_request = {
    "input": [{"role": "user", "content": "Which countries always vote for each other?"}]
}

model_name = f"{cfg.catalog}.{cfg.schema}.eurovision_agent"
ts = datetime.now().strftime("%Y-%m-%d")

# COMMAND ----------
with mlflow.start_run(run_name=f"eurovision-agent-{ts}"):
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../src/eurovision_voting_bloc_party/agent.py",
        resources=resources,
        model_config={
            "catalog": cfg.catalog,
            "schema": cfg.schema,
            "volume": cfg.volume,
            "llm_endpoint": cfg.llm_endpoint,
            "vector_search_endpoint": cfg.vector_search_endpoint,
            "embedding_endpoint": cfg.embedding_endpoint,
        },
    )

# COMMAND ----------
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
)

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)

logger.info(
    f"Registered {model_name} version {registered_model.version}",
    "with alias 'latest-model'",
)
