from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phenomena import dq_missing_label_target_dependent  # noqa: E402
from phenomena._base import (  # noqa: E402
    CheckDetail,
    ValidationResult,
    last_failed_detail,
    row_ids,
)
from shared.metrics import (  # noqa: E402
    binned_importance,
    eligible_numeric_columns,
    has_direction_reversal,
    heteroskedasticity_score,
    nonmonotone_score,
    plateau_score,
    safe_std,
    spearman_bin_rho,
)


class SharedMetricHelperTests(unittest.TestCase):
    def test_eligible_numeric_columns_preserves_order_and_filters(self):
        frame = pd.DataFrame({
            "excluded": [0, 1, 2, 3],
            "low_cardinality": [1, 1, 1, 2],
            "eligible": [0.0, 1.0, 2.0, 3.0],
            "text": ["a", "b", "c", "d"],
        })

        self.assertEqual(
            eligible_numeric_columns(
                frame,
                exclude={"excluded"},
                min_unique=3,
            ),
            ["eligible"],
        )

    def test_safe_std_retains_sample_std_and_handles_degenerate_series(self):
        values = pd.Series([1.0, 2.0, 3.0])

        self.assertEqual(safe_std(values), float(values.std()))
        self.assertEqual(safe_std(pd.Series([2.0, 2.0])), 1.0)
        self.assertEqual(safe_std(pd.Series([2.0])), 1.0)
        self.assertEqual(safe_std(pd.Series([2.0]), fallback=7.0), 7.0)

    def test_quantile_metrics_keep_their_distinct_fallbacks(self):
        feature = pd.Series([1.0, 1.0, 1.0, 1.0])
        outcome = pd.Series([0.0, 1.0, 2.0, 3.0])

        self.assertEqual(nonmonotone_score(feature, outcome), 0.0)
        self.assertEqual(spearman_bin_rho(feature, outcome), 1.0)
        self.assertFalse(has_direction_reversal(feature, outcome))
        self.assertEqual(binned_importance(feature, outcome), 0.0)
        self.assertEqual(heteroskedasticity_score(feature, outcome), 1.0)
        self.assertEqual(plateau_score(feature, outcome), 0.0)


class PhenomenonHelperTests(unittest.TestCase):
    def test_last_failed_detail_returns_the_last_failure(self):
        result = ValidationResult(
            passed=False,
            checks=[
                CheckDetail("first", False, 0.0, 1.0, "first failure"),
                CheckDetail("pass", True, 1.0, 1.0, "passed"),
                CheckDetail("last", False, 0.0, 1.0, "last failure"),
            ],
        )

        self.assertEqual(last_failed_detail(result), "last failure")
        self.assertEqual(
            last_failed_detail(ValidationResult(passed=False), "fallback"),
            "fallback",
        )

    def test_row_ids_uses_stable_ids_or_positions(self):
        positions = np.array([2, 0])

        self.assertEqual(
            row_ids(pd.DataFrame({"row_id": [30, 10, 20]}), positions),
            [20, 30],
        )
        self.assertEqual(
            row_ids(pd.DataFrame({"value": [3, 1, 2]}), positions),
            [0, 2],
        )

    def test_target_dependent_validator_reuses_injected_group_stat(self):
        frame = pd.DataFrame({
            "category": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "target": list(range(95, 105)) + list(range(10)) + list(range(10, 20)),
        })
        effects = {
            "feature": "category",
            "injected_label": "A",
            "direction": "high",
            "n_injected_rows": 10,
            "p_thresh": 1.0,
            "margin": 0.0,
            "min_gap_factor": 0.0,
        }

        with patch.object(
            dq_missing_label_target_dependent,
            "_group_stat",
            wraps=dq_missing_label_target_dependent._group_stat,
        ) as group_stat:
            result = dq_missing_label_target_dependent.validate(
                frame,
                effects,
                "target",
            )

        self.assertTrue(result.passed)
        self.assertEqual(group_stat.call_count, 3)


if __name__ == "__main__":
    unittest.main()
