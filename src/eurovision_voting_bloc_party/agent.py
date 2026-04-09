import json

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from eurovision_voting_bloc_party.mcp import ToolInfo


class EurovisionAgent:
    def __init__(
        self, w: WorkspaceClient, cfg: dict, custom_tools: list[ToolInfo], mcp_tools: list
    ):
        self.tools = {tool.name: tool for tool in custom_tools + mcp_tools}
        self.client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/serving-endpoints",
        )
        self.cfg = cfg

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

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        try:
            return str(self.tools[tool_name].exec_fn(**args))
        except Exception as e:
            error_string = f"Tool error: {e}"
            logger.error(error_string)
            return error_string

    def ask(self, question: str, max_iterations: int = 10) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.cfg.llm_endpoint,
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
