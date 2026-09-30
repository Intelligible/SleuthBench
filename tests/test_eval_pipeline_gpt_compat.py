from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline as ep  # noqa: E402
from tests._provider_fakes import OutcomeSequence  # noqa: E402


class _RecordingEndpoint:
    def __init__(self, responses):
        self._sequence = OutcomeSequence(responses)
        self.calls = self._sequence.calls

    def create(self, **kwargs):
        return self._sequence.take(kwargs)


_RecordingChatCompletions = _RecordingEndpoint
_RecordingResponses = _RecordingEndpoint


class _ChatMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None
        self.reasoning_content = None
        self.model_extra = {}

    def model_dump(self, **_kwargs):
        return {"role": "assistant", "content": self.content}


def _chat_usage(prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _stream_chunk(content=None, usage=None):
    choices = (
        [SimpleNamespace(delta=SimpleNamespace(content=content))]
        if content is not None
        else []
    )
    return SimpleNamespace(choices=choices, usage=usage)


class AnswerPayloadParsingTests(unittest.TestCase):
    def test_extracts_only_a_string_answer_from_an_object(self):
        self.assertEqual(
            ep._parse_answer_payload('{"answer": "ok"}'),
            "ok",
        )

    def test_non_object_json_falls_back_to_stripped_raw_text(self):
        for raw, expected in (
            ("  []\n", "[]"),
            ("\tnull ", "null"),
            (" 42 ", "42"),
            (" true ", "true"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(ep._parse_answer_payload(raw), expected)

    def test_non_string_or_missing_answer_falls_back_to_raw_object(self):
        for raw in (
            '{"answer": null}',
            '{"answer": []}',
            '{"answer": 42}',
            '{"other": "value"}',
        ):
            with self.subTest(raw=raw):
                self.assertEqual(ep._parse_answer_payload(raw), raw)


class GPTCompatibilityTests(unittest.TestCase):
    def test_supported_tools_define_provider_order(self):
        self.assertEqual(
            ep.SUPPORTED_TOOLS,
            ("load_data", "run_python"),
        )
        self.assertEqual(tuple(ep._TOOL_SPECS), ep.SUPPORTED_TOOLS)
        self.assertEqual(
            ep._provider_tools(
                set(ep.SUPPORTED_TOOLS),
                lambda name: {"name": name},
            ),
            [{"name": name} for name in ep.SUPPORTED_TOOLS],
        )

    def test_query_openai_uses_chat_completions_stream_and_json_schema(self):
        stream = [
            _stream_chunk('{"answer":'),
            _stream_chunk('"ok"}'),
            _stream_chunk(usage=_chat_usage(11, 4)),
        ]
        completions = _RecordingChatCompletions([stream])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        answer, thinking, metrics = ep.query_openai(
            client,
            "gpt-4o",
            "feature,target\n1,2\n",
            "What is the answer?",
        )

        self.assertEqual(answer, "ok")
        self.assertIsNone(thinking)
        self.assertEqual(metrics["input_tokens"], 11)
        self.assertEqual(metrics["output_tokens"], 4)
        self.assertIsNotNone(metrics["ttft_s"])

        request = completions.calls[0]
        self.assertEqual(request["model"], "gpt-4o")
        self.assertEqual(request["max_tokens"], 1024)
        self.assertNotIn("max_completion_tokens", request)
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertEqual(
            request["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": ep.ANSWER_SCHEMA,
                },
            },
        )
        self.assertEqual(
            request["messages"][0],
            {"role": "system", "content": ep.SYSTEM_PROMPT},
        )
        self.assertIn("What is the answer?", request["messages"][1]["content"])

    def test_non_reasoning_gpt_tools_use_chat_request_without_thinking_fields(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=_ChatMessage("final answer"))],
            usage=_chat_usage(7, 3),
        )
        completions = _RecordingChatCompletions([response])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        )

        answer, thinking, tool_log, metrics = ep.query_openai_style_tools(
            client,
            "gpt-4o",
            "",
            "question",
            None,
            Path("unused.csv"),
            enabled_tools={"load_data"},
            thinking_mode=False,
        )

        self.assertFalse(ep._uses_responses_api("gpt-4o"))
        self.assertEqual(answer, "final answer")
        self.assertIsNone(thinking)
        self.assertEqual(tool_log, [])
        self.assertEqual(metrics["input_tokens"], 7)
        self.assertEqual(metrics["output_tokens"], 3)

        request = completions.calls[0]
        self.assertEqual(request["model"], "gpt-4o")
        self.assertEqual(request["max_tokens"], 2048)
        self.assertEqual(request["tool_choice"], "auto")
        self.assertNotIn("max_completion_tokens", request)
        self.assertNotIn("reasoning_effort", request)
        self.assertNotIn("extra_body", request)
        self.assertEqual(request["tools"][0]["type"], "function")
        self.assertEqual(
            request["tools"][0]["function"]["name"],
            "load_data",
        )

    def test_reasoning_gpt_responses_api_request_and_usage(self):
        reasoning_item = SimpleNamespace(
            type="reasoning",
            summary=[SimpleNamespace(text="reasoning summary")],
        )
        response = SimpleNamespace(
            output=[reasoning_item],
            output_text="reasoned answer",
            usage=SimpleNamespace(input_tokens=13, output_tokens=5),
        )
        responses = _RecordingResponses([response])
        client = SimpleNamespace(responses=responses)

        answer, thinking, tool_log, metrics = ep.query_openai_responses_tools(
            client,
            "gpt-5-mini",
            "",
            "question",
            None,
            Path("unused.csv"),
            enabled_tools={"load_data"},
        )

        self.assertTrue(ep._uses_responses_api("gpt-5-mini"))
        self.assertEqual(
            ep._openai_tokens_kwarg("gpt-5-mini", 123),
            {"max_completion_tokens": 123},
        )
        self.assertEqual(answer, "reasoned answer")
        self.assertEqual(thinking, "reasoning summary")
        self.assertEqual(tool_log, [])
        self.assertEqual(metrics["input_tokens"], 13)
        self.assertEqual(metrics["output_tokens"], 5)

        request = responses.calls[0]
        self.assertEqual(request["model"], "gpt-5-mini")
        self.assertEqual(request["tool_choice"], "auto")
        self.assertEqual(
            request["reasoning"],
            {"effort": ep.OPENAI_REASONING_EFFORT, "summary": "auto"},
        )
        self.assertEqual(
            request["max_output_tokens"],
            ep.OPENAI_RESPONSES_MAX_OUTPUT,
        )
        self.assertEqual(request["tools"][0]["type"], "function")
        self.assertEqual(request["tools"][0]["name"], "load_data")
        self.assertEqual(request["input"][0]["role"], "user")
        self.assertIn("question", request["input"][0]["content"])


if __name__ == "__main__":
    unittest.main()
