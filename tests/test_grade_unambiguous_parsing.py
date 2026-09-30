from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import grade_pipeline as gp  # noqa: E402


class StrictNumericGradingTests(unittest.TestCase):
    def test_relative_tolerance_helper_is_zero_safe(self):
        self.assertEqual(
            gp._relative_tolerance_check(100.0, 104.0, 0.05),
            (0.04, True),
        )
        self.assertEqual(
            gp._relative_tolerance_check(0.0, 0.06, 0.05),
            (0.06, False),
        )

    def test_canonical_numeric_answers_still_grade_deterministically(self):
        self.assertEqual(
            gp.grade_numeric_value(0.15, "`0.15`.")[0],
            "CORRECT",
        )
        self.assertEqual(
            gp.grade_numeric_value(0.15, "0.30")[0],
            "INCORRECT",
        )

    def test_negated_or_multiple_numeric_answers_defer_to_judge(self):
        answers = (
            "No threshold exists; 0.15 is only the feature mean.",
            "The candidates are 0.15 and 0.30.",
            "0.15 is not the answer.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertIsNone(gp.grade_numeric_value(0.15, answer))


class UnambiguousColumnValueGradingTests(unittest.TestCase):
    def test_leading_pair_with_neutral_explanation_is_allowed(self):
        answers = (
            "**Longitude, -122.25**\nThe effect later falls to -3.8.",
            "Longitude, -122.25.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                result = gp.grade_column_value(
                    "Longitude, -122.23",
                    answer,
                )
                self.assertEqual(result[0], "CORRECT")

    def test_negated_or_alternative_pairs_defer_to_judge(self):
        answers = (
            "It is not Longitude, -122.23; the answer is Latitude, 37.7.",
            "Longitude, -122.23 is not the answer.",
            "Longitude, -122.23; Latitude, 37.7",
            "Longitude, -122.23 but Latitude, 37.7 is the answer.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertIsNone(
                    gp.grade_column_value("Longitude, -122.23", answer)
                )

    def test_grade_entry_uses_judge_for_negated_expected_value(self):
        create = Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"grade":"INCORRECT",'
                                '"reasoning":"The expected value was negated."}'
                            )
                        )
                    )
                ]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        entry = {
            "answer_format": "value",
            "question": "What is the threshold?",
            "expected_answer": 0.15,
            "model_answer": (
                "No threshold exists; 0.15 is only the feature mean."
            ),
        }

        grade, reasoning = gp.grade_entry(client, "gpt-test", entry)

        self.assertEqual(grade, "INCORRECT")
        self.assertIn("negated", reasoning)
        create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
