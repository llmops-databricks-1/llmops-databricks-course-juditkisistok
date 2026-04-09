# Databricks notebook source
# COMMAND ----------
import asyncio
import json

import nest_asyncio
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.config import get_env, load_config
from eurovision_voting_bloc_party.mcp import ToolInfo, create_mcp_tools

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
def roast_country(country_name: str) -> str:
    """
    Fetch Eurovision stats for a country to fuel a loving but devastating roast.
    Args:
        country_name: Name of the country to roast
    Returns:
        JSON string with the country's Eurovision history
    """

    kaggle_data = spark.table(f"{CATALOG}.{SCHEMA}.eurovision_kaggle_chunks")
    country_id = country_name.strip().lower().replace(" ", "_")

    rows = kaggle_data.filter(F.col("chunk_id") == country_id).select("text").collect()

    if not rows:
        return json.dumps(
            {
                "error": f"""No data found for country: {country_name}.
                                Maybe they never qualified?"""
            }
        )

    return json.dumps(
        {
            "country": country_name,
            "stats": rows[0].text,
        }
    )


roast_country("Sweden")

# COMMAND ----------
predict_winner_spec = {
    "type": "function",
    "function": {
        "name": "predict_winner",
        "description": "Fetch historical Eurovision country stats to predict this year's "
        "winner. Use this when someone asks who will win, who should win, or "
        "wants a prediction.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

roast_country_spec = {
    "type": "function",
    "function": {
        "name": "roast_country",
        "description": "Fetch a country's Eurovision history to deliver a loving but "
        "devastating roast. Use this when someone mentions they're from a country, "
        "asks about a specific country, or explicitly asks for a roast.",
        "parameters": {
            "type": "object",
            "properties": {
                "country_name": {
                    "type": "string",
                    "description": "The country to roast, e.g. 'United Kingdom', "
                    "'Norway', 'France'",
                }
            },
            "required": ["country_name"],
        },
    },
}

predict_winner_tool = ToolInfo(
    name="predict_winner",
    spec=predict_winner_spec,
    exec_fn=predict_winner,
)

roast_country_tool = ToolInfo(
    name="roast_country",
    spec=roast_country_spec,
    exec_fn=roast_country,
)

# COMMAND ----------

nest_asyncio.apply()

w = WorkspaceClient()
vector_search_mcp_url = f"{w.config.host}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"

mcp_tools = asyncio.run(create_mcp_tools(w, [vector_search_mcp_url]))
logger.info(f"Loaded {len(mcp_tools)} MCP tools")
for tool in mcp_tools:
    logger.info(f"Tool: {tool.name}, Spec: {tool.spec}")

# COMMAND ----------


class EurovisionAgent:
    def __init__(self, system_prompt: str, custom_tools: list[ToolInfo], mcp_tools: list):
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in custom_tools + mcp_tools}
        self.client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/serving-endpoints",
        )

    def _get_tool_specs(self) -> list[dict]:
        return [tool.spec for tool in self.tools.values()]

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        try:
            return str(self.tools[tool_name].exec_fn(**args))
        except Exception as e:
            logger.error(f"Tool error: {e}")

        tool = self.tools[tool_name]
        return tool.exec_fn(**args)

    def ask(self, question: str, max_iterations: int = 10) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=cfg.llm_endpoint,
                messages=messages,
                tools=self._get_tool_specs(),
            )

            message = response.choices[0].message

            if message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )
                for tc in message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    logger.info(f"Executing tool: {tc.function.name} with args: {args}")
                    tool_result = self._execute_tool(tc.function.name, args)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
                    )
            else:
                return message.content
        return "Max iterations reached without a final answer."


# COMMAND ----------
graham_norton = EurovisionAgent(
    system_prompt=system_prompt,
    custom_tools=[predict_winner_tool, roast_country_tool],
    mcp_tools=mcp_tools,
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

# COMMAND ----------
