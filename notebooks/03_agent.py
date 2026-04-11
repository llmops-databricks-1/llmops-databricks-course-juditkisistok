# Databricks notebook source
# COMMAND ----------
import asyncio

import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger
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
response = graham_norton.ask("Which countries always vote for each other?")
logger.info(response)

# COMMAND ----------
# nerdy fact
response = graham_norton.ask("Give me a nerdy Eurovision fact from academic research")
logger.info(response)

# COMMAND ----------
response = graham_norton.ask("Who is going to win Eurovision this year?")
logger.info(response)
# COMMAND ----------
response = graham_norton.ask("I'm from the United Kingdom, big Eurovision fan!")
logger.info(response)
