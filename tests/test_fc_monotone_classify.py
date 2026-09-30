from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phenomena.fc_monotone_classify import PHENOMENON  # noqa: E402
from phenomena.fc_monotone_classify import inject, validate  # noqa: E402


class _DeterministicRng:
    def permutation(self, values):
        if isinstance(values, (int, np.integer)):
            return np.arange(values)
        return np.asarray(values)[::-1]


class MonotoneClassifyValidationTests(unittest.TestCase):
    def setUp(self):
        repeated = np.tile(np.arange(10, dtype=float) + 0.25, 2)
        self.df = pd.DataFrame({
            "row_id": np.arange(20),
            "id": np.arange(100, 120),
            "serial_number": np.arange(200, 220),
            "declared_id": repeated,
            "feature_a": repeated,
            "feature_b": np.roll(repeated, 1),
            "feature_c": np.roll(repeated, 2),
            "target": np.arange(20, dtype=float) + 0.125,
        })

    def test_requests_id_columns_from_dataset_summary(self):
        self.assertEqual(PHENOMENON.summary_fields, ("id_no",))

    def test_injector_excludes_declared_and_detected_ids(self):
        rho_values = iter([0.8, 0.8, 0.8, 0.1])
        with patch(
            "phenomena.fc_monotone_classify.spearman_bin_rho",
            side_effect=lambda *_: next(rho_values),
        ):
            with patch(
                "phenomena.fc_monotone_classify.has_direction_reversal",
                return_value=True,
            ):
                _, metadata = inject(
                    self.df,
                    {
                        "outcome_col": "target",
                        "effect_strength": 3.0,
                        "id_no_cols": ["declared_id"],
                    },
                    _DeterministicRng(),
                )

        effects = metadata["effects"]
        self.assertEqual(effects["injected_feature"], "feature_a")
        self.assertEqual(
            effects["id_no_cols"],
            ["declared_id", "id", "row_id", "serial_number"],
        )

    def test_validator_rejects_equal_or_lower_rho(self):
        effects = {
            "injected_feature": "feature_a",
            "id_no_cols": ["declared_id"],
        }
        tied_scores = {
            "feature_a": 0.1,
            "feature_b": 0.1,
            "feature_c": 0.3,
        }

        with patch(
            "phenomena.fc_monotone_classify.spearman_bin_rho",
            side_effect=lambda feature, _outcome: tied_scores[feature.name],
        ):
            tied_result = validate(self.df, effects, "target")

        lower_scores = {**tied_scores, "feature_b": 0.099}
        with patch(
            "phenomena.fc_monotone_classify.spearman_bin_rho",
            side_effect=lambda feature, _outcome: lower_scores[feature.name],
        ):
            lower_result = validate(self.df, effects, "target")

        self.assertFalse(tied_result.passed)
        self.assertEqual(tied_result.checks[0].name, "dominance_lowest_rho")
        self.assertFalse(lower_result.passed)
        self.assertEqual(lower_result.checks[0].name, "dominance_lowest_rho")

    def test_validator_rejects_legacy_id_injected_feature(self):
        result = validate(
            self.df,
            {"injected_feature": "row_id"},
            "target",
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            result.checks[0].name,
            "injected_feature_not_identifier",
        )


if __name__ == "__main__":
    unittest.main()
