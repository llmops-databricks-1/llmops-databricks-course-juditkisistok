import json
import os
import warnings
from collections.abc import Generator
from typing import Any
from uuid import uuid4

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from eurovision_voting_bloc_party.config import ProjectConfig
from eurovision_voting_bloc_party.mcp import ToolInfo


class EurovisionAgent(ResponsesAgent):
    def __init__(
        self,
        w: WorkspaceClient,
        cfg: ProjectConfig,
        custom_tools: list[ToolInfo],
        mcp_tools: list[ToolInfo],
    ):
        self.tools = {tool.name: tool for tool in custom_tools + mcp_tools}
        self.cfg = cfg
        self.workspace_client = w
        self.model_serving_client = w.serving_endpoints.get_open_ai_client()

        self.system_prompt = """
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
