# Databricks notebook source
# COMMAND ----------
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.agent import EurovisionAgent
from eurovision_voting_bloc_party.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("project_config.yaml", env)

# COMMAND ----------
graham_norton = EurovisionAgent(
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
