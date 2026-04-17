# Databricks notebook source
# COMMAND ----------
import asyncio

import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.agent import EurovisionAgent
from eurovision_voting_bloc_party.agent_tools import (
    create_predict_winner_tool,
    create_roast_country_tool,
)
from eurovision_voting_bloc_party.config import get_env, load_config
from eurovision_voting_bloc_party.mcp import create_mcp_tools

spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("project_config.yaml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema

w = WorkspaceClient()
vector_search_mcp_url = f"{w.config.host}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"

# COMMAND ----------

nest_asyncio.apply()

mcp_tools = asyncio.run(create_mcp_tools(w, [vector_search_mcp_url]))
logger.info(f"Loaded {len(mcp_tools)} MCP tools")
for tool in mcp_tools:
    logger.info(f"Tool: {tool.name}, Spec: {tool.spec}")


# COMMAND ----------
predict_winner_tool = create_predict_winner_tool(spark, CATALOG, SCHEMA)
roast_country_tool = create_roast_country_tool(spark, CATALOG, SCHEMA)

graham_norton = EurovisionAgent(
    custom_tools=[predict_winner_tool, roast_country_tool],
    mcp_tools=mcp_tools,
    w=w,
    cfg=cfg,
)

# COMMAND ----------
# general question
request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Which countries always vote for each other?"}],
    custom_inputs={"session_id": "demo-session-1"},
)
response = graham_norton.predict(request)
logger.info(response.output[0].model_dump()["content"][0]["text"])

# COMMAND ----------
# nerdy fact
request = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": "Give me a nerdy Eurovision fact from academic research",
        }
    ],
    custom_inputs={"session_id": "demo-session-1"},
)
response = graham_norton.predict(request)
logger.info(response.output[0].model_dump()["content"][0]["text"])

# COMMAND ----------
request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Who is going to win Eurovision this year?"}],
    custom_inputs={"session_id": "demo-session-1"},
)
response = graham_norton.predict(request)
logger.info(response.output[0].model_dump()["content"][0]["text"])

# COMMAND ----------
request = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "I'm from the United Kingdom, big Eurovision fan!"}
    ],
    custom_inputs={"session_id": "demo-session-1"},
)
response = graham_norton.predict(request)
logger.info(response.output[0].model_dump()["content"][0]["text"])
