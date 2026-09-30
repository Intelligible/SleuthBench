from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline as ep  # noqa: E402
from tests._provider_fakes import OutcomeSequence  # noqa: E402


def _usage():
    return SimpleNamespace(input_tokens=3, output_tokens=2, prompt_tokens=3, completion_tokens=2)


class _AnthropicStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.message


class _AnthropicMessages:
    def __init__(self, message):
        self.message = message
        self.stream_calls = []
        self.create_calls = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        return _AnthropicStream(self.message)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.message


class _SequencedAnthropicMessages:
    def __init__(self, messages):
        def _snapshot(kwargs: dict) -> dict:
            captured = dict(kwargs)
            captured["messages"] = list(kwargs["messages"])
            return captured

        self._sequence = OutcomeSequence(messages, snapshot=_snapshot)
        self.create_calls = self._sequence.calls

    def create(self, **kwargs):
        return self._sequence.take(kwargs)


class _ChatCompletions:
    def __init__(self, responses):
        self._sequence = OutcomeSequence(responses)
        self.calls = self._sequence.calls

    def create(self, **kwargs):
        return self._sequence.take(kwargs)


class _ToolCall:
    def __init__(self, call_id="call-1", name="load_data", arguments="{}"):
        self.id = call_id
        self.type = "function"
        self.function = SimpleNamespace(name=name, arguments=arguments)

    def model_dump(self, **_kwargs):
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


def _chat_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=_usage())


class AnthropicCompatibilityTests(unittest.TestCase):
    def test_current_sonnet_and_fable_models_accept_direct_text(self):
        cases = (
            ("claude-sonnet-5", True),
            ("claude-sonnet-4-6", True),
            ("claude-fable-5", False),
        )
        for model, sends_thinking_switch in cases:
            with self.subTest(model=model):
                message = SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="direct answer")],
                    usage=_usage(),
                )
                messages = _AnthropicMessages(message)
                client = SimpleNamespace(messages=messages)

                answer, thinking, _metrics = ep.query_anthropic(client, model, "", "question")

                self.assertEqual(answer, "direct answer")
                self.assertIsNone(thinking)
                request = messages.stream_calls[0]
                self.assertEqual(request["tool_choice"], {"type": "auto"})
                self.assertEqual("thinking" in request, sends_thinking_switch)
                if sends_thinking_switch:
                    self.assertEqual(request["thinking"], {"type": "adaptive"})

    def test_anthropic_salvage_preserves_thinking_and_disables_more_tools(self):
        thinking_block = SimpleNamespace(type="thinking", thinking="", signature="sig")
        tool_block = SimpleNamespace(type="tool_use", id="tool-1", name="load_data", input={})
        history = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [thinking_block, tool_block]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "data"}]},
        ]
        tools = [{"name": "load_data", "input_schema": {"type": "object"}}]

        for model, sends_thinking_switch in (
            ("claude-sonnet-5", True),
            ("claude-fable-5", False),
        ):
            with self.subTest(model=model):
                final_message = SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="final answer")],
                    usage=_usage(),
                )
                messages = _AnthropicMessages(final_message)
                client = SimpleNamespace(messages=messages)

                answer, _input_tokens, _output_tokens = ep._force_final_answer_anthropic(
                    client, model, "system", history, tools
                )

                self.assertEqual(answer, "final answer")
                request = messages.create_calls[0]
                self.assertEqual(request["tool_choice"], {"type": "none"})
                self.assertIs(request["tools"], tools)
                self.assertIs(request["messages"][1]["content"][0], thinking_block)
                self.assertEqual("thinking" in request, sends_thinking_switch)
                self.assertEqual(request["output_config"], {"effort": "low"})

    def test_anthropic_tool_loop_replays_complete_thinking_blocks(self):
        cases = (
            ("claude-sonnet-5", True),
            ("claude-sonnet-4-6", True),
            ("claude-fable-5", False),
        )
        with patch.object(ep, "_run_eval_tool", return_value="x\n1\n"):
            for model, sends_thinking_switch in cases:
                with self.subTest(model=model):
                    thinking_block = SimpleNamespace(
                        type="thinking", thinking="", signature=f"sig-{model}"
                    )
                    tool_block = SimpleNamespace(
                        type="tool_use", id="tool-1", name="load_data", input={}
                    )
                    tool_message = SimpleNamespace(
                        content=[thinking_block, tool_block],
                        usage=_usage(),
                    )
                    final_message = SimpleNamespace(
                        content=[SimpleNamespace(type="text", text="final answer")],
                        usage=_usage(),
                    )
                    messages = _SequencedAnthropicMessages([tool_message, final_message])
                    client = SimpleNamespace(messages=messages)

                    answer, _thinking, _tool_log, _metrics = ep.query_anthropic_tools(
                        client,
                        model,
                        "",
                        "question",
                        None,
                        Path("unused.csv"),
                        enabled_tools={"load_data"},
                    )

                    self.assertEqual(answer, "final answer")
                    self.assertEqual(len(messages.create_calls), 2)
                    for request in messages.create_calls:
                        self.assertEqual(request["tool_choice"], {"type": "auto"})
                        self.assertEqual("thinking" in request, sends_thinking_switch)
                    replayed_content = messages.create_calls[1]["messages"][1]["content"]
                    self.assertIs(replayed_content[0], thinking_block)
                    self.assertIs(replayed_content[1], tool_block)

    def test_older_sonnet_does_not_receive_unsupported_adaptive_mode(self):
        self.assertFalse(ep.supports_thinking("claude-sonnet-4-5"))
        self.assertEqual(ep._anthropic_thinking_kwargs("claude-sonnet-4-5"), {})


class BedrockConfigurationTests(unittest.TestCase):
    def test_max_tokens_uses_family_default_when_env_is_unset_or_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                ep._bedrock_max_tokens("bedrock-deepseek-v3.2"),
                8_192,
            )

            os.environ["BEDROCK_MAX_TOKENS"] = ""
            self.assertEqual(
                ep._bedrock_max_tokens("bedrock-deepseek-v3.2"),
                8_192,
            )

    def test_max_tokens_uses_valid_env_value(self):
        with patch.dict(
            os.environ,
            {"BEDROCK_MAX_TOKENS": "12345"},
            clear=False,
        ):
            self.assertEqual(
                ep._bedrock_max_tokens("bedrock-deepseek-v3.2"),
                12_345,
            )

    def test_max_tokens_rejects_invalid_env_value_with_clear_error(self):
        with patch.dict(
            os.environ,
            {"BEDROCK_MAX_TOKENS": "not-an-integer"},
            clear=False,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "^BEDROCK_MAX_TOKENS='not-an-integer' is not a valid integer$",
            ) as raised:
                ep._bedrock_max_tokens("bedrock-deepseek-v3.2")

        self.assertTrue(raised.exception.__suppress_context__)


class DeepSeekCompatibilityTests(unittest.TestCase):
    def test_tool_thinking_capability_is_version_gated(self):
        self.assertTrue(
            ep._family("deepseek-v4-flash").get("supports_thinking_tools", False)
        )
        for model in ("deepseek-v3.2", "deepseek-chat"):
            with self.subTest(model=model):
                self.assertFalse(
                    ep._family(model).get("supports_thinking_tools", False)
                )

    def test_v4_without_tools_uses_supported_json_thinking_request(self):
        for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
            with self.subTest(model=model):
                message = SimpleNamespace(
                    content='{"answer": "ok"}',
                    reasoning_content="reasoning",
                    model_extra={},
                )
                completions = _ChatCompletions([_chat_response(message)])
                client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

                answer, reasoning, _metrics = ep.query_deepseek(client, model, "", "question")

                self.assertEqual(answer, "ok")
                self.assertEqual(reasoning, "reasoning")
                request = completions.calls[0]
                self.assertEqual(request["response_format"], {"type": "json_object"})
                self.assertEqual(request["reasoning_effort"], "high")
                self.assertEqual(request["extra_body"], {"thinking": {"type": "enabled"}})

    def test_v4_without_tools_retries_one_documented_empty_json_response(self):
        empty_message = SimpleNamespace(content="", reasoning_content="first", model_extra={})
        final_message = SimpleNamespace(
            content='{"answer": "retried"}',
            reasoning_content="second",
            model_extra={},
        )
        completions = _ChatCompletions(
            [_chat_response(empty_message), _chat_response(final_message)]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        answer, reasoning, metrics = ep.query_deepseek(
            client, "deepseek-v4-flash", "", "question"
        )

        self.assertEqual(answer, "retried")
        self.assertEqual(reasoning, "second")
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(metrics["input_tokens"], 6)
        self.assertEqual(metrics["output_tokens"], 4)

    def test_v4_tool_loop_omits_tool_choice_and_replays_reasoning(self):
        with patch.object(ep, "_run_eval_tool", return_value="x\n1\n"):
            for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
                with self.subTest(model=model):
                    tool_call = _ToolCall()
                    tool_message = SimpleNamespace(
                        content=None,
                        reasoning_content="reason-1",
                        model_extra={},
                        tool_calls=[tool_call],
                    )
                    final_message = SimpleNamespace(
                        content="final answer",
                        reasoning_content="reason-2",
                        model_extra={},
                        tool_calls=None,
                    )
                    completions = _ChatCompletions(
                        [_chat_response(tool_message), _chat_response(final_message)]
                    )
                    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

                    answer, _thinking, _tool_log, _metrics = ep.query_deepseek_tools(
                        client,
                        model,
                        "",
                        "question",
                        None,
                        Path("unused.csv"),
                        enabled_tools={"load_data"},
                    )

                    self.assertEqual(answer, "final answer")
                    self.assertEqual(len(completions.calls), 2)
                    for request in completions.calls:
                        self.assertNotIn("tool_choice", request)
                        self.assertEqual(request["reasoning_effort"], "high")
                        self.assertEqual(request["extra_body"], {"thinking": {"type": "enabled"}})

                    replayed = completions.calls[1]["messages"][2]
                    self.assertEqual(replayed["role"], "assistant")
                    self.assertEqual(replayed["content"], "")
                    self.assertEqual(replayed["reasoning_content"], "reason-1")
                    self.assertEqual(replayed["tool_calls"], [tool_call.model_dump()])

    def test_generic_model_uses_non_thinking_tool_request_shape(self):
        final_message = SimpleNamespace(
            content="final answer",
            reasoning_content=None,
            model_extra={},
            tool_calls=None,
            model_dump=lambda **_kwargs: {
                "role": "assistant",
                "content": "final answer",
            },
        )
        completions = _ChatCompletions([_chat_response(final_message)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        answer, thinking, _tool_log, _metrics = ep.query_deepseek_tools(
            client,
            "deepseek-chat",
            "",
            "question",
            None,
            Path("unused.csv"),
            enabled_tools={"load_data"},
        )

        self.assertEqual(answer, "final answer")
        self.assertIsNone(thinking)
        request = completions.calls[0]
        self.assertEqual(request["tool_choice"], "auto")
        self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("extra_body", request)

    def test_v4_salvage_keeps_tool_reasoning_and_thinking_mode(self):
        history = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "reason-1",
                "tool_calls": [_ToolCall().model_dump()],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "data"},
        ]
        final_message = SimpleNamespace(content="final answer")
        completions = _ChatCompletions([_chat_response(final_message)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        answer, _input_tokens, _output_tokens = ep._force_final_answer_openai_chat(
            client,
            "deepseek-v4-flash",
            history,
            thinking_mode=True,
        )

        self.assertEqual(answer, "final answer")
        request = completions.calls[0]
        self.assertNotIn("tool_choice", request)
        self.assertNotIn("tools", request)
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(request["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(request["max_tokens"], ep.DEEPSEEK_SALVAGE_MAX_TOKENS)
        replayed = request["messages"][1]
        self.assertEqual(replayed["content"], "")
        self.assertEqual(replayed["reasoning_content"], "reason-1")

    def test_generic_salvage_strips_thinking_only_fields(self):
        history = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "reason-1",
                "tool_calls": [_ToolCall().model_dump()],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "data"},
        ]
        final_message = SimpleNamespace(content="final answer")
        completions = _ChatCompletions([_chat_response(final_message)])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        answer, _input_tokens, _output_tokens = ep._force_final_answer_openai_chat(
            client,
            "deepseek-chat",
            history,
            thinking_mode=False,
        )

        self.assertEqual(answer, "final answer")
        request = completions.calls[0]
        self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("extra_body", request)
        self.assertNotIn("tools", request)
        self.assertNotIn("tool_choice", request)
        self.assertEqual(request["max_tokens"], ep.SALVAGE_MAX_TOKENS)
        self.assertNotIn("reasoning_content", request["messages"][1])


if __name__ == "__main__":
    unittest.main()
