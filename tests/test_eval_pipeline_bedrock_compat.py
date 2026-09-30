from __future__ import annotations

import copy
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline as ep  # noqa: E402
from tests._provider_fakes import OutcomeSequence  # noqa: E402


class _BedrockClient:
    def __init__(self, responses: list[dict]):
        self._sequence = OutcomeSequence(responses)
        self.calls = self._sequence.calls

    def converse(self, **kwargs):
        return copy.deepcopy(self._sequence.take(kwargs))


class _Sandbox:
    def __init__(self, result: str):
        self.result = result
        self.calls: list[tuple[str, Path]] = []

    def run(
        self,
        code: str,
        csv_path: Path,
        *,
        deadline: float | None = None,
    ) -> str:
        self.calls.append((code, csv_path))
        return self.result


class BedrockCompatibilityTests(unittest.TestCase):
    def test_query_bedrock_deepseek_request_and_response_contract(self):
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "first reason"}
                            }
                        },
                        {
                            "reasoningContent": {
                                "reasoningText": "second reason"
                            }
                        },
                        {"text": '{"answer": "direct answer"}'},
                    ],
                }
            },
            "usage": {"inputTokens": 11, "outputTokens": 7},
            "metrics": {"latencyMs": 123},
        }
        client = _BedrockClient([response])

        with patch.dict(
            os.environ,
            {"BEDROCK_DEEPSEEK_MODEL_ID": "", "BEDROCK_MAX_TOKENS": ""},
        ):
            answer, reasoning, metrics = ep.query_bedrock_deepseek(
                client,
                "bedrock-deepseek-v3.2",
                "a,b\n1,2\n",
                "What is b?",
            )

        self.assertEqual(answer, "direct answer")
        self.assertEqual(reasoning, "first reason\n\nsecond reason")
        self.assertEqual(metrics["input_tokens"], 11)
        self.assertEqual(metrics["output_tokens"], 7)
        self.assertIsNone(metrics["ttft_s"])
        self.assertGreaterEqual(metrics["total_latency_s"], 0)
        self.assertEqual(metrics["bedrock_latency_ms"], 123)

        self.assertEqual(len(client.calls), 1)
        request = client.calls[0]
        self.assertEqual(
            set(request),
            {"modelId", "system", "messages", "inferenceConfig"},
        )
        self.assertEqual(request["modelId"], "deepseek.v3.2")
        self.assertEqual(request["system"], [{"text": ep.DEEPSEEK_JSON_SYSTEM_PROMPT}])
        self.assertEqual(request["inferenceConfig"], {"maxTokens": 8_192})
        self.assertEqual(
            request["messages"],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": ep._build_user_content(
                                "a,b\n1,2\n",
                                "What is b?",
                            )
                        }
                    ],
                }
            ],
        )

    def test_query_bedrock_deepseek_tools_replays_tool_result_and_usage(self):
        tool_output = {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {"text": "inspect with Python"}
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "tool-1",
                        "name": "run_python",
                        "input": {"code": "print(42)"},
                    }
                },
            ],
        }
        final_output = {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": "answer from the tool result"
                    }
                },
                {"text": "final answer"},
            ],
        }
        client = _BedrockClient(
            [
                {
                    "output": {"message": tool_output},
                    "stopReason": "tool_use",
                    "usage": {"inputTokens": 3, "outputTokens": 2},
                    "metrics": {"latencyMs": 10},
                },
                {
                    "output": {"message": final_output},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 5, "outputTokens": 4},
                    "metrics": {"latencyMs": 20},
                },
            ]
        )
        sandbox = _Sandbox("42")
        csv_path = Path("unused.csv")

        with patch.dict(os.environ, {"BEDROCK_MAX_TOKENS": ""}):
            answer, reasoning, tool_log, metrics = ep.query_bedrock_deepseek_tools(
                client,
                "bedrock:custom.profile-id",
                "",
                "Compute the answer.",
                sandbox,
                csv_path,
                enabled_tools={"run_python"},
            )

        self.assertEqual(answer, "final answer")
        self.assertEqual(
            reasoning,
            "inspect with Python\n\nanswer from the tool result",
        )
        self.assertEqual(
            tool_log,
            [
                {
                    "tool": "run_python",
                    "input": "print(42)",
                    "step_thinking": "inspect with Python",
                }
            ],
        )
        self.assertEqual(sandbox.calls, [("print(42)", csv_path)])
        self.assertEqual(metrics["input_tokens"], 8)
        self.assertEqual(metrics["output_tokens"], 6)
        self.assertIsNone(metrics["ttft_s"])
        self.assertGreaterEqual(metrics["total_latency_s"], 0)
        self.assertEqual(metrics["bedrock_latency_ms"], 20)

        self.assertEqual(len(client.calls), 2)
        first_request, second_request = client.calls
        for request in client.calls:
            self.assertEqual(
                set(request),
                {
                    "modelId",
                    "system",
                    "messages",
                    "toolConfig",
                    "inferenceConfig",
                },
            )
            self.assertEqual(request["modelId"], "custom.profile-id")
            self.assertEqual(request["inferenceConfig"], {"maxTokens": 8_192})
            self.assertEqual(
                request["toolConfig"],
                {"tools": [ep._bedrock_tool("run_python")]},
            )
            tool_spec = request["toolConfig"]["tools"][0]["toolSpec"]
            self.assertEqual(tool_spec["name"], "run_python")
            self.assertEqual(
                tool_spec["inputSchema"]["json"]["required"],
                ["code"],
            )
            self.assertFalse(
                tool_spec["inputSchema"]["json"]["additionalProperties"]
            )

        initial_user_message = {
            "role": "user",
            "content": [
                {
                    "text": ep._build_user_content(
                        "",
                        "Compute the answer.",
                    )
                }
            ],
        }
        self.assertEqual(first_request["messages"], [initial_user_message])
        self.assertEqual(
            second_request["messages"],
            [
                initial_user_message,
                tool_output,
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "tool-1",
                                "content": [{"text": "42"}],
                            }
                        }
                    ],
                },
            ],
        )
        self.assertEqual(first_request["system"], second_request["system"])
        self.assertEqual(
            first_request["system"],
            [{"text": ep._build_system_prompt_tools({"run_python"})}],
        )


if __name__ == "__main__":
    unittest.main()
