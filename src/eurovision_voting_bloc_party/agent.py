import asyncio
import contextlib
import json
import os
import warnings
from collections.abc import Generator
from datetime import datetime
from typing import Any
from uuid import uuid4

import backoff
import mlflow
import nest_asyncio
import openai
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
from mlflow.pyfunc import PythonModelContext, ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from pyspark.sql import SparkSession

from eurovision_voting_bloc_party.agent_tools import (
    create_predict_winner_tool,
    create_roast_country_tool,
)
from eurovision_voting_bloc_party.config import ProjectConfig
from eurovision_voting_bloc_party.mcp import create_mcp_tools


class EurovisionAgent(ResponsesAgent):
    SYSTEM_PROMPT = """
        You are Graham Norton hosting Eurovision — dry, witty, and delightfully sarcastic,
        but with a secret PhD in musicology and voting pattern analysis that occasionally
        slips out. You've seen it all: the tactical voting, the novelty acts, the
        inexplicable douze points to a neighbouring country. You love Eurovision deeply
        but you're not going to pretend it's high art.

        When answering questions, be entertaining first, accurate second — but always
        accurate. When roasting a country, make it personal and grounded in their actual
        Eurovision record. When making predictions, deliver them like you're reading the
        scoreboard at 2am and you can't quite believe what you're seeing.

        When you stumble upon academic research about Eurovision, act mildly horrified
        that someone wrote a paper about this, then get genuinely excited about the
        findings.
        """

    def __init__(self, cfg: ProjectConfig | None = None):
        if cfg is None:
            return

        nest_asyncio.apply()
        self.cfg = cfg
        self.system_prompt = self.SYSTEM_PROMPT

        w = WorkspaceClient()
        spark = SparkSession.builder.getOrCreate()

        self.workspace_client = w
        self.model_serving_client = w.serving_endpoints.get_open_ai_client()

        mcp_url = f"{w.config.host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"
        mcp_tools = asyncio.run(create_mcp_tools(w, [mcp_url]))

        custom_tools = [
            create_predict_winner_tool(spark, cfg.catalog, cfg.schema),
            create_roast_country_tool(spark, cfg.catalog, cfg.schema),
        ]

        self.tools = {tool.name: tool for tool in mcp_tools + custom_tools}

    def _get_tool_specs(self) -> list[dict]:
        return [tool.spec for tool in self.tools.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        try:
            return str(self.tools[tool_name].exec_fn(**args))
        except Exception as e:
            error_string = f"Tool error: {e}"
            logger.error(error_string)
            return error_string

    def handle_tool_call(
        self, tool_call: dict, messages: list[dict]
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool call, append the result to messages, and return a stream event.
        """
        args = json.loads(tool_call["arguments"])
        result = self._execute_tool(tool_name=tool_call["name"], args=args)
        tool_call_output = self.create_function_call_output_item(
            call_id=tool_call["id"], output=result
        )
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=tool_call_output,
        )

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def call_llm(self, messages: list[dict]) -> Generator[dict, None, None]:
        """
        Call the LLM with the given messages and yield stream events as they come in.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="PydanticSerializationUnexpectedValue"
            )
            stream = self.model_serving_client.chat.completions.create(
                model=self.cfg.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self._get_tool_specs(),
                stream=True,
            )
            with mlflow.start_span(name="call_llm", span_type=SpanType.LLM) as span:
                last_chunk: dict[str, Any] = {}
                for chunk in stream:
                    chunk_dict = chunk.to_dict()
                    last_chunk = chunk_dict
                    yield chunk_dict
                span.set_outputs(
                    {
                        "model": last_chunk.get("model"),
                        "usage": last_chunk.get("usage"),
                    }
                )

    def _extract_output_items(
        self, events: list[ResponsesAgentStreamEvent]
    ) -> list[dict]:
        """Pull the final message items out of stream events."""
        items = []
        for event in events:
            if event.type != "response.output_item.done":
                continue
            item = event.item if isinstance(event.item, dict) else event.item.model_dump()
            if item.get("type") == "message":
                items.append(item)

        return items

    def _run_tool_loop(self, messages: list, max_iterations: int = 10) -> list:
        events = []
        for _ in range(max_iterations):
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                break
            elif last_msg.get("type") == "function_call":
                events.append(self.handle_tool_call(last_msg, messages))
            else:
                events.extend(
                    output_to_responses_items_stream(
                        chunks=self.call_llm(messages),
                        aggregator=messages,
                    )
                )
        else:
            events.append(
                ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        "Max iterations reached. Stopping.", str(uuid4())
                    ),
                )
            )
        return events

    @mlflow.trace(span_type=SpanType.CHAIN)
    def call_and_run_tools(
        self,
        request_input: list,
        session_id: str = None,
        request_id: str = None,
    ) -> list[ResponsesAgentStreamEvent]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(request_input)

        with contextlib.suppress(Exception):
            mlflow.update_current_trace(
                tags={
                    "git_sha": os.getenv("GIT_SHA", "local"),
                    "model_serving_endpoint": os.getenv(
                        "MODEL_SERVING_ENDPOINT_NAME", "local"
                    ),
                    "agent_type": "eurovision_agent",
                },
                metadata=({"mlflow.trace.session": session_id} if session_id else {}),
                client_request_id=request_id,
            )

        return self._run_tool_loop(messages)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        events = list(self.predict_stream(request))
        return ResponsesAgentResponse(
            output=self._extract_output_items(events),
            custom_outputs=request.custom_inputs,
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        custom_inputs = request.custom_inputs or {}
        session_id, request_id = (
            custom_inputs.get("session_id"),
            custom_inputs.get("request_id"),
        )

        request_input = [i.model_dump() for i in request.input]
        events = self.call_and_run_tools(
            request_input, session_id=session_id, request_id=request_id
        )
        yield from events

    def load_context(self, context: PythonModelContext) -> None:
        """Called by MLflow when loading the model from registry."""
        with contextlib.suppress(Exception):
            nest_asyncio.apply()

            model_config = context.model_config
            w = WorkspaceClient()
            spark = SparkSession.builder.getOrCreate()
            self.workspace_client = w
            self.model_serving_client = w.serving_endpoints.get_open_ai_client()
            self.system_prompt = self.SYSTEM_PROMPT

            self.cfg = ProjectConfig(
                catalog=model_config["catalog"],
                schema=model_config["schema"],
                volume=model_config.get("volume", ""),
                llm_endpoint=model_config["llm_endpoint"],
                vector_search_endpoint=model_config.get("vector_search_endpoint", ""),
                embedding_endpoint=model_config.get("embedding_endpoint", ""),
            )

            mcp_url = (
                f"{w.config.host}/api/2.0/mcp/vector-search"
                f"/{self.cfg.catalog}/{self.cfg.schema}"
            )

            custom_tools = [
                create_predict_winner_tool(
                    spark,
                    self.cfg.catalog,
                    self.cfg.schema,
                ),
                create_roast_country_tool(
                    spark,
                    self.cfg.catalog,
                    self.cfg.schema,
                ),
            ]
            mcp_tools = asyncio.run(create_mcp_tools(w, [mcp_url]))
            self.tools = {tool.name: tool for tool in mcp_tools + custom_tools}


def log_register_agent(
    cfg: ProjectConfig,
    git_sha: str,
    run_id: str,
    agent_code_path: str,
    model_name: str,
    evaluation_metrics: dict | None = None,
) -> mlflow.entities.model_registry.RegisteredModel:
    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
        DatabricksVectorSearchIndex(
            index_name=f"{cfg.catalog}.{cfg.schema}.eurovision_unified_index"
        ),
        DatabricksTable(
            table_name=f"{cfg.catalog}.{cfg.schema}.eurovision_unified_chunks"
        ),
        DatabricksTable(
            table_name=f"{cfg.catalog}.{cfg.schema}.eurovision_kaggle_chunks"
        ),
    ]

    model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "volume": cfg.volume,
        "llm_endpoint": cfg.llm_endpoint,
        "vector_search_endpoint": cfg.vector_search_endpoint,
        "embedding_endpoint": cfg.embedding_endpoint,
    }

    test_request = {
        "input": [
            {"role": "user", "content": "Which countries always vote for each other?"}
        ]
    }

    ts = datetime.now().strftime("%Y-%m-%d")

    with mlflow.start_run(
        run_name=f"eurovision-agent-{ts}",
        tags={"git_sha": git_sha, "run_id": run_id},
    ):
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
        )
        if evaluation_metrics:
            mlflow.log_metrics(evaluation_metrics)

    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
        tags={"git_sha": git_sha, "run_id": run_id},
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
    return registered_model


mlflow.models.set_model(EurovisionAgent())
