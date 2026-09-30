from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_pipeline as ep  # noqa: E402


OVERLAPPING_PREFIX_CASES = (
    ("claude-opus-4-8", "claude-opus", "claude-opus-4-8-20260724"),
    ("gpt-5", "gpt-", "gpt-5.4"),
    ("deepseek-v4-", "deepseek-", "deepseek-v4-pro"),
)


ROUTING_CASES = (
    {
        "model": "claude-opus-4-8-20260724",
        "provider": "anthropic",
        "tokens_kwarg": "max_tokens",
        "thinking_mode": "adaptive",
        "supports_thinking": True,
        "supports_thinking_tools": False,
        "uses_responses_api": False,
    },
    {
        "model": "claude-opus-4-5",
        "provider": "anthropic",
        "tokens_kwarg": "max_tokens",
        "thinking_mode": None,
        "supports_thinking": False,
        "supports_thinking_tools": False,
        "uses_responses_api": False,
    },
    {
        "model": "gpt-5.4",
        "provider": "openai",
        "tokens_kwarg": "max_completion_tokens",
        "thinking_mode": None,
        "supports_thinking": False,
        "supports_thinking_tools": False,
        "uses_responses_api": True,
    },
    {
        "model": "gpt-4o",
        "provider": "openai",
        "tokens_kwarg": "max_tokens",
        "thinking_mode": None,
        "supports_thinking": False,
        "supports_thinking_tools": False,
        "uses_responses_api": False,
    },
    {
        "model": "deepseek-v4-pro",
        "provider": "deepseek",
        "tokens_kwarg": "max_tokens",
        "thinking_mode": None,
        "supports_thinking": False,
        "supports_thinking_tools": True,
        "uses_responses_api": False,
    },
    {
        "model": "deepseek-chat",
        "provider": "deepseek",
        "tokens_kwarg": "max_tokens",
        "thinking_mode": None,
        "supports_thinking": False,
        "supports_thinking_tools": False,
        "uses_responses_api": False,
    },
)


class ModelFamilyRoutingTests(unittest.TestCase):
    def test_specific_prefixes_precede_overlapping_generic_prefixes(self):
        positions = {
            prefix: index
            for index, (prefix, _metadata) in enumerate(ep.MODEL_FAMILIES)
        }

        for specific, generic, model in OVERLAPPING_PREFIX_CASES:
            with self.subTest(specific=specific, generic=generic):
                self.assertTrue(model.startswith(specific))
                self.assertTrue(model.startswith(generic))
                self.assertLess(positions[specific], positions[generic])
                matched_prefix = next(
                    prefix
                    for prefix, _metadata in ep.MODEL_FAMILIES
                    if model.startswith(prefix)
                )
                self.assertEqual(matched_prefix, specific)

    def test_overlapping_families_route_with_expected_capabilities(self):
        for case in ROUTING_CASES:
            model = case["model"]
            with self.subTest(model=model):
                family = ep._family(model)

                self.assertEqual(ep.detect_provider(model), case["provider"])
                self.assertEqual(family["tokens_kwarg"], case["tokens_kwarg"])
                self.assertEqual(
                    family.get("thinking_mode"),
                    case["thinking_mode"],
                )
                self.assertEqual(
                    ep.supports_thinking(model),
                    case["supports_thinking"],
                )
                self.assertEqual(
                    bool(family.get("supports_thinking_tools", False)),
                    case["supports_thinking_tools"],
                )
                self.assertEqual(
                    ep._uses_responses_api(model),
                    case["uses_responses_api"],
                )

    def test_bedrock_deepseek_names_stay_on_bedrock_families(self):
        cases = (
            (
                "bedrock-deepseek-v3.2",
                "bedrock-deepseek-v3.2",
                "deepseek.v3.2",
            ),
            (
                "bedrock:us.deepseek.r1-v1:0",
                "bedrock:",
                None,
            ),
        )

        for model, expected_prefix, expected_model_id in cases:
            with self.subTest(model=model):
                matched_prefix, family = next(
                    (prefix, metadata)
                    for prefix, metadata in ep.MODEL_FAMILIES
                    if model.startswith(prefix)
                )

                self.assertEqual(matched_prefix, expected_prefix)
                self.assertEqual(ep.detect_provider(model), "bedrock")
                self.assertNotIn("tokens_kwarg", family)
                self.assertEqual(family["max_tokens"], 8_192)
                self.assertFalse(ep.supports_thinking(model))
                self.assertFalse(
                    bool(family.get("supports_thinking_tools", False))
                )
                self.assertFalse(ep._uses_responses_api(model))
                if expected_model_id is not None:
                    self.assertEqual(
                        family["bedrock_model_id"],
                        expected_model_id,
                    )


if __name__ == "__main__":
    unittest.main()
