from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline as ep  # noqa: E402
from tests._provider_fakes import OutcomeSequence  # noqa: E402


class _ToolCall:
    def __init__(self, call_id: str = "call-1"):
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(name="load_data", arguments="{}")

    def model_dump(self, **_kwargs):
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _ChatMessage:
    def __init__(self, content: str | None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = None
        self.model_extra = {}

    def model_dump(self, **_kwargs):
        message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        return message


def _chat_response(
    message: _ChatMessage,
    *,
    input_tokens: int = 1,
    output_tokens: int = 1,
):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        ),
    )


class _ChatCompletions:
    def __init__(self, responses):
        self._sequence = OutcomeSequence(responses)
        self.calls = self._sequence.calls

    def create(self, **kwargs):
        return self._sequence.take(kwargs)


class _Responses:
    def __init__(self, responses):
        self._sequence = OutcomeSequence(responses)
        self.calls = self._sequence.calls

    def create(self, **kwargs):
        return self._sequence.take(kwargs)


class _AnthropicMessages:
    def __init__(self, messages):
        def _snapshot(kwargs: dict) -> dict:
            captured = dict(kwargs)
            captured["messages"] = list(kwargs["messages"])
            return captured

        self._sequence = OutcomeSequence(messages, snapshot=_snapshot)
        self.calls = self._sequence.calls

    def create(self, **kwargs):
        return self._sequence.take(kwargs)


class _BedrockClient:
    def __init__(self, responses, *, close_error: BaseException | None = None):
        self._sequence = OutcomeSequence(responses)
        self.calls = self._sequence.calls
        self.closed = False
        self.close_error = close_error

    def converse(self, **kwargs):
        outcome = self._sequence.take(kwargs)
        return copy.deepcopy(outcome)

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class ToolLoopBudgetTests(unittest.TestCase):
    def test_default_limits_remain_bounded(self):
        self.assertEqual(ep.MAX_TOOL_ITER, 25)
        self.assertEqual(ep.MAX_TOOL_CALLS, 64)
        self.assertEqual(ep.MAX_TOOL_TOTAL_TOKENS, 500_000)
        self.assertEqual(ep.MAX_TOOL_RESULT_BYTES, 2_000_000)
        self.assertEqual(ep.MAX_TOOL_WALL_TIME_S, 600.0)
        self.assertEqual(ep.MAX_TOOL_PROVIDER_CALL_S, 300.0)
        self.assertLessEqual(
            ep.MAX_TOOL_PROVIDER_CALL_S,
            ep.MAX_TOOL_WALL_TIME_S,
        )

    def test_openai_chat_token_limit_stops_before_tool_or_salvage(self):
        tool_message = _ChatMessage(None, [_ToolCall()])
        completions = _ChatCompletions(
            [_chat_response(tool_message, input_tokens=3, output_tokens=2)]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with (
            patch.object(ep, "MAX_TOOL_TOTAL_TOKENS", 5),
            patch.object(
                ep,
                "_run_eval_tool",
                side_effect=AssertionError("tool must not run"),
            ),
        ):
            answer, _thinking, _tool_log, metrics = ep.query_openai_style_tools(
                client,
                "gpt-4o",
                "",
                "question",
                None,
                Path("unused.csv"),
                enabled_tools={"load_data"},
            )

        self.assertIn("token limit", answer)
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(metrics["input_tokens"], 3)
        self.assertEqual(metrics["output_tokens"], 2)
        self.assertEqual(metrics["tool_iterations"], 1)
        self.assertEqual(metrics["tool_calls"], 0)
        self.assertEqual(metrics["budget_stop_reason"], "tokens")

    def test_final_answer_is_kept_when_its_response_reaches_token_limit(self):
        completions = _ChatCompletions(
            [
                _chat_response(
                    _ChatMessage("final answer"),
                    input_tokens=3,
                    output_tokens=2,
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with patch.object(ep, "MAX_TOOL_TOTAL_TOKENS", 5):
            answer, _thinking, _tool_log, metrics = ep.query_openai_style_tools(
                client,
                "gpt-4o",
                "",
                "question",
                None,
                Path("unused.csv"),
                enabled_tools={"load_data"},
            )

        self.assertEqual(answer, "final answer")
        self.assertNotIn("budget_stop_reason", metrics)
        self.assertEqual(metrics["tool_iterations"], 1)

    def test_salvage_answer_is_kept_when_usage_reaches_token_limit(self):
        completions = _ChatCompletions(
            [
                _chat_response(
                    _ChatMessage(""),
                    input_tokens=1,
                    output_tokens=1,
                ),
                _chat_response(
                    _ChatMessage("final answer"),
                    input_tokens=2,
                    output_tokens=1,
                ),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with patch.object(ep, "MAX_TOOL_TOTAL_TOKENS", 5):
            answer, _thinking, _tool_log, metrics = (
                ep.query_openai_style_tools(
                    client,
                    "gpt-4o",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                )
            )

        self.assertEqual(answer, "final answer")
        self.assertEqual(metrics["input_tokens"], 3)
        self.assertEqual(metrics["output_tokens"], 2)
        self.assertEqual(metrics["tool_iterations"], 2)
        self.assertNotIn("budget_stop_reason", metrics)

    def test_openai_chat_batch_cannot_bypass_tool_call_limit(self):
        tool_message = _ChatMessage(
            None,
            [_ToolCall("call-1"), _ToolCall("call-2")],
        )
        salvage_message = _ChatMessage("final answer")
        completions = _ChatCompletions(
            [_chat_response(tool_message), _chat_response(salvage_message)]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with (
            patch.object(ep, "MAX_TOOL_CALLS", 1),
            patch.object(
                ep,
                "_parse_json_tool_input",
                return_value={},
            ) as parse_input,
            patch.object(ep, "_run_eval_tool", return_value="data") as run_tool,
        ):
            answer, _thinking, _tool_log, metrics = ep.query_openai_style_tools(
                client,
                "gpt-4o",
                "",
                "question",
                None,
                Path("unused.csv"),
                enabled_tools={"load_data"},
            )

        self.assertEqual(answer, "final answer")
        self.assertEqual(run_tool.call_count, 1)
        self.assertEqual(parse_input.call_count, 1)
        self.assertEqual(len(completions.calls), 2)
        salvage_request = completions.calls[1]
        tool_messages = [
            m for m in salvage_request["messages"] if m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 2)
        self.assertIn("tool-call limit", tool_messages[1]["content"])
        self.assertNotIn("tools", salvage_request)
        self.assertEqual(metrics["tool_iterations"], 2)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["budget_stop_reason"], "tool_calls")

    def test_openai_chat_caps_aggregate_tool_result_bytes(self):
        tool_message = _ChatMessage(
            None,
            [_ToolCall("call-1"), _ToolCall("call-2")],
        )
        completions = _ChatCompletions(
            [_chat_response(tool_message)]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with (
            patch.object(ep, "MAX_TOOL_RESULT_BYTES", 5),
            patch.object(ep, "_run_eval_tool", return_value="123456") as run_tool,
        ):
            answer, _thinking, _tool_log, metrics = ep.query_openai_style_tools(
                client,
                "gpt-4o",
                "",
                "question",
                None,
                Path("unused.csv"),
                enabled_tools={"load_data"},
            )

        self.assertIn("tool-result limit", answer)
        self.assertEqual(run_tool.call_count, 1)
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(metrics["tool_result_bytes"], 5)
        self.assertEqual(metrics["budget_stop_reason"], "tool_output")

    def test_exact_tool_result_limit_skips_next_parallel_tool(self):
        tool_message = _ChatMessage(
            None,
            [_ToolCall("call-1"), _ToolCall("call-2")],
        )
        completions = _ChatCompletions([_chat_response(tool_message)])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with (
            patch.object(ep, "MAX_TOOL_RESULT_BYTES", 5),
            patch.object(
                ep,
                "_run_eval_tool",
                return_value="12345",
            ) as run_tool,
        ):
            answer, _thinking, _tool_log, metrics = (
                ep.query_openai_style_tools(
                    client,
                    "gpt-4o",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                )
            )

        self.assertIn("tool-result limit", answer)
        self.assertEqual(run_tool.call_count, 1)
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(metrics["tool_result_bytes"], 5)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["budget_stop_reason"], "tool_output")

    def test_second_provider_failure_keeps_partial_metrics_and_tool_log(self):
        completions = _ChatCompletions(
            [
                _chat_response(
                    _ChatMessage(None, [_ToolCall()]),
                    input_tokens=3,
                    output_tokens=2,
                ),
                RuntimeError("second round failed"),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        answer, _thinking, tool_log, metrics = ep.query_openai_style_tools(
            client,
            "gpt-4o",
            "",
            "question",
            None,
            REPO_ROOT / "pyproject.toml",
            enabled_tools={"load_data"},
        )

        self.assertEqual(
            answer,
            "ERROR: provider_error: RuntimeError: second round failed",
        )
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(len(tool_log), 1)
        self.assertEqual(tool_log[0]["tool"], "load_data")
        self.assertEqual(metrics["input_tokens"], 3)
        self.assertEqual(metrics["output_tokens"], 2)
        self.assertEqual(metrics["tool_iterations"], 2)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertGreater(metrics["tool_result_bytes"], 0)
        self.assertNotIn("budget_stop_reason", metrics)
        self.assertEqual(
            metrics["provider_error"],
            {"type": "RuntimeError", "message": "second round failed"},
        )
        self.assertTrue(metrics["usage_incomplete"])

    def test_provider_failures_are_structured_across_other_tool_apis(self):
        cases = [
            (
                "responses",
                lambda: ep.query_openai_responses_tools(
                    SimpleNamespace(
                        responses=_Responses([RuntimeError("request failed")])
                    ),
                    "gpt-5-mini",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                ),
            ),
            (
                "anthropic",
                lambda: ep.query_anthropic_tools(
                    SimpleNamespace(
                        messages=_AnthropicMessages(
                            [RuntimeError("request failed")]
                        )
                    ),
                    "claude-sonnet-4-6",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                ),
            ),
            (
                "bedrock",
                lambda: ep.query_bedrock_deepseek_tools(
                    _BedrockClient([RuntimeError("request failed")]),
                    "bedrock:custom.profile-id",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                ),
            ),
        ]

        for provider, query in cases:
            with self.subTest(provider=provider):
                answer, _thinking, tool_log, metrics = query()
                self.assertEqual(
                    answer,
                    "ERROR: provider_error: RuntimeError: request failed",
                )
                self.assertEqual(tool_log, [])
                self.assertEqual(metrics["tool_iterations"], 1)
                self.assertEqual(metrics["tool_calls"], 0)
                self.assertNotIn("budget_stop_reason", metrics)
                self.assertEqual(
                    metrics["provider_error"],
                    {"type": "RuntimeError", "message": "request failed"},
                )
                self.assertTrue(metrics["usage_incomplete"])

    def test_salvage_failure_keeps_budget_reason_and_diagnostic(self):
        completions = _ChatCompletions(
            [
                _chat_response(
                    _ChatMessage(None, [_ToolCall()]),
                    input_tokens=3,
                    output_tokens=2,
                ),
                RuntimeError("salvage failed"),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with (
            patch.object(ep, "MAX_TOOL_ITER", 2),
            patch.object(ep, "_run_eval_tool", return_value="data"),
        ):
            answer, _thinking, _tool_log, metrics = (
                ep.query_openai_style_tools(
                    client,
                    "gpt-4o",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                )
            )

        self.assertIn("model-call limit", answer)
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(metrics["input_tokens"], 3)
        self.assertEqual(metrics["output_tokens"], 2)
        self.assertEqual(metrics["tool_iterations"], 2)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["budget_stop_reason"], "iterations")
        self.assertEqual(
            metrics["salvage_error"],
            {"type": "RuntimeError", "message": "salvage failed"},
        )
        self.assertTrue(metrics["usage_incomplete"])

    def test_responses_deadline_stops_after_running_current_tool(self):
        function_call = SimpleNamespace(
            type="function_call",
            name="load_data",
            arguments="{}",
            call_id="call-1",
        )
        response = SimpleNamespace(
            output=[function_call],
            output_text="",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        responses = _Responses([response])
        client = SimpleNamespace(responses=responses)
        now = [0.0]

        def _run_tool(*_args, **_kwargs):
            now[0] = 2.0
            return "data"

        with (
            patch.object(ep, "MAX_TOOL_WALL_TIME_S", 1.0),
            patch.object(ep.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(ep, "_run_eval_tool", side_effect=_run_tool),
        ):
            answer, _thinking, _tool_log, metrics = (
                ep.query_openai_responses_tools(
                    client,
                    "gpt-5-mini",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                )
            )

        self.assertIn("wall-time limit", answer)
        self.assertEqual(len(responses.calls), 1)
        self.assertEqual(metrics["tool_iterations"], 1)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["budget_stop_reason"], "deadline")

    def test_provider_timeout_at_deadline_keeps_partial_budget_metrics(self):
        now = [0.0]

        class _TimeoutCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **_kwargs):
                self.calls += 1
                now[0] = 2.0
                raise TimeoutError("provider timed out")

        completions = _TimeoutCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        with (
            patch.object(ep, "MAX_TOOL_WALL_TIME_S", 1.0),
            patch.object(ep.time, "monotonic", side_effect=lambda: now[0]),
        ):
            answer, _thinking, _tool_log, metrics = ep.query_openai_style_tools(
                client,
                "gpt-4o",
                "",
                "question",
                None,
                Path("unused.csv"),
                enabled_tools={"load_data"},
            )

        self.assertIn("wall-time limit", answer)
        self.assertEqual(completions.calls, 1)
        self.assertEqual(metrics["tool_iterations"], 1)
        self.assertEqual(metrics["budget_stop_reason"], "deadline")

    def test_anthropic_iteration_limit_reserves_one_salvage_call(self):
        tool_block = SimpleNamespace(
            type="tool_use",
            id="tool-1",
            name="load_data",
            input={},
        )
        tool_message = SimpleNamespace(
            content=[tool_block],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        final_message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="final answer")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        messages = _AnthropicMessages([tool_message, final_message])
        client = SimpleNamespace(messages=messages)

        with (
            patch.object(ep, "MAX_TOOL_ITER", 2),
            patch.object(ep, "_run_eval_tool", return_value="data"),
        ):
            answer, _thinking, _tool_log, metrics = ep.query_anthropic_tools(
                client,
                "claude-sonnet-4-6",
                "",
                "question",
                None,
                Path("unused.csv"),
                enabled_tools={"load_data"},
            )

        self.assertEqual(answer, "final answer")
        self.assertEqual(len(messages.calls), 2)
        self.assertEqual(messages.calls[0]["tool_choice"], {"type": "auto"})
        self.assertEqual(messages.calls[1]["tool_choice"], {"type": "none"})
        self.assertEqual(metrics["tool_iterations"], 2)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["budget_stop_reason"], "iterations")

    def test_bedrock_iteration_limit_salvage_has_no_tool_config(self):
        tool_output = {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tool-1",
                        "name": "load_data",
                        "input": {},
                    }
                }
            ],
        }
        final_output = {
            "role": "assistant",
            "content": [{"text": "final answer"}],
        }
        client = _BedrockClient(
            [
                {
                    "output": {"message": tool_output},
                    "stopReason": "tool_use",
                    "usage": {"inputTokens": 1, "outputTokens": 1},
                    "metrics": {"latencyMs": 1},
                },
                {
                    "output": {"message": final_output},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1},
                    "metrics": {"latencyMs": 2},
                },
            ]
        )

        with (
            patch.object(ep, "MAX_TOOL_ITER", 2),
            patch.object(ep, "_run_eval_tool", return_value="data"),
        ):
            answer, _thinking, _tool_log, metrics = (
                ep.query_bedrock_deepseek_tools(
                    client,
                    "bedrock:custom.profile-id",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                )
            )

        self.assertEqual(answer, "final answer")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("toolConfig", client.calls[0])
        self.assertNotIn("toolConfig", client.calls[1])
        self.assertEqual(metrics["tool_iterations"], 2)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["budget_stop_reason"], "iterations")
        self.assertEqual(metrics["bedrock_latency_ms"], 2)

    def test_bedrock_uses_remaining_time_for_late_provider_call(self):
        tool_output = {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tool-1",
                        "name": "load_data",
                        "input": {},
                    }
                }
            ],
        }
        final_output = {
            "role": "assistant",
            "content": [{"text": "final answer"}],
        }
        base_client = _BedrockClient(
            [
                {
                    "output": {"message": tool_output},
                    "stopReason": "tool_use",
                    "usage": {"inputTokens": 1, "outputTokens": 1},
                    "metrics": {"latencyMs": 1},
                }
            ]
        )
        late_client = _BedrockClient(
            [
                {
                    "output": {"message": final_output},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1},
                    "metrics": {"latencyMs": 2},
                }
            ]
        )
        requested_timeouts: list[float] = []
        now = [0.0]

        def _client_factory(timeout_s: float):
            requested_timeouts.append(timeout_s)
            return late_client

        def _run_tool(*_args, **_kwargs):
            now[0] = 5.0
            return "data"

        with (
            patch.object(ep, "MAX_TOOL_WALL_TIME_S", 10.0),
            patch.object(ep, "MAX_TOOL_PROVIDER_CALL_S", 6.0),
            patch.object(ep.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(ep, "_run_eval_tool", side_effect=_run_tool),
        ):
            answer, _thinking, _tool_log, metrics = (
                ep.query_bedrock_deepseek_tools(
                    base_client,
                    "bedrock:custom.profile-id",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                    client_factory=_client_factory,
                )
            )

        self.assertEqual(answer, "final answer")
        self.assertEqual(len(base_client.calls), 1)
        self.assertEqual(len(late_client.calls), 1)
        self.assertEqual(requested_timeouts, [5.0])
        self.assertTrue(late_client.closed)
        self.assertEqual(metrics["tool_iterations"], 2)
        self.assertEqual(metrics["tool_calls"], 1)

    def test_bedrock_temporary_client_close_error_does_not_hide_answer(self):
        final_output = {
            "role": "assistant",
            "content": [{"text": "final answer"}],
        }
        temporary_client = _BedrockClient(
            [
                {
                    "output": {"message": final_output},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1},
                    "metrics": {"latencyMs": 2},
                }
            ],
            close_error=RuntimeError("close failed"),
        )

        with (
            patch.object(ep, "MAX_TOOL_WALL_TIME_S", 5.0),
            patch.object(ep, "MAX_TOOL_PROVIDER_CALL_S", 6.0),
            patch.object(ep.time, "monotonic", return_value=0.0),
        ):
            answer, _thinking, _tool_log, metrics = (
                ep.query_bedrock_deepseek_tools(
                    _BedrockClient([]),
                    "bedrock:custom.profile-id",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                    client_factory=lambda _timeout: temporary_client,
                )
            )

        self.assertEqual(answer, "final answer")
        self.assertTrue(temporary_client.closed)
        self.assertNotIn("provider_error", metrics)

    def test_bedrock_rechecks_deadline_after_temporary_client_creation(self):
        temporary_client = _BedrockClient([])
        now = [0.0]

        def _client_factory(_timeout_s: float):
            now[0] = 6.0
            return temporary_client

        with (
            patch.object(ep, "MAX_TOOL_WALL_TIME_S", 5.0),
            patch.object(ep, "MAX_TOOL_PROVIDER_CALL_S", 6.0),
            patch.object(ep.time, "monotonic", side_effect=lambda: now[0]),
        ):
            answer, _thinking, _tool_log, metrics = (
                ep.query_bedrock_deepseek_tools(
                    _BedrockClient([]),
                    "bedrock:custom.profile-id",
                    "",
                    "question",
                    None,
                    Path("unused.csv"),
                    enabled_tools={"load_data"},
                    client_factory=_client_factory,
                )
            )

        self.assertIn("wall-time limit", answer)
        self.assertEqual(temporary_client.calls, [])
        self.assertTrue(temporary_client.closed)
        self.assertEqual(metrics["budget_stop_reason"], "deadline")

    def test_actual_clients_disable_retries_and_clamp_request_timeout(self):
        class _Client:
            def __init__(self):
                self.options = None

            def with_options(self, **kwargs):
                self.options = kwargs
                return self

        client = _Client()
        budget = ep._ToolLoopBudget()
        self.assertIs(ep._budgeted_api_client(client, budget), client)
        self.assertEqual(client.options["max_retries"], 0)
        self.assertGreater(client.options["timeout"], 0)
        self.assertLessEqual(
            client.options["timeout"],
            ep.MAX_TOOL_PROVIDER_CALL_S,
        )

    def test_run_python_receives_session_deadline_without_mutating_timeout(self):
        class _Sandbox:
            def __init__(self):
                self.timeout = 30
                self.seen_deadline = None

            def run(self, _code, _csv_path, *, deadline=None):
                self.seen_deadline = deadline
                return "ok"

        sandbox = _Sandbox()
        budget = ep._ToolLoopBudget()
        budget.deadline_at = 123.5
        result = ep._run_budgeted_eval_tool(
            "run_python",
            {"code": "print('ok')"},
            sandbox,
            Path("unused.csv"),
            [],
            None,
            budget,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(sandbox.seen_deadline, 123.5)
        self.assertEqual(sandbox.timeout, 30)


if __name__ == "__main__":
    unittest.main()
