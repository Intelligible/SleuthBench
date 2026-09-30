from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phenomena.dq_bad_row_indicator import inject, validate  # noqa: E402

PARAMS = {
    "outcome_col": "target",
    "bad_fraction": 0.1,
    "corruption_cap_std": 2.0,
}


class BadRowIndicatorTests(unittest.TestCase):
    def test_binary_targets_flip_zero_rows_to_one(self):
        base = pd.DataFrame({
            "feature": np.arange(100),
            "target": [1] * 5 + [0] * 95,
        })

        injected, metadata = inject(base, PARAMS, np.random.default_rng(1729))
        effects = metadata["effects"]
        bad_indices = effects["bad_indices"]
        indicator = effects["indicator_col"]

        self.assertEqual(effects["corruption_mode"], "binary_zero_to_one")
        self.assertEqual(len(bad_indices), 10)
        self.assertTrue((base.iloc[bad_indices]["target"] == 0).all())
        self.assertTrue((injected.iloc[bad_indices]["target"] == 1).all())
        self.assertEqual(set(injected["target"].unique()), {0, 1})
        self.assertEqual(int(injected[indicator].sum()), 10)

        result = validate(injected, effects, "target")
        self.assertTrue(result.passed)
        outcome_check = next(
            check for check in result.checks
            if check.name == "outcome_difference"
        )
        self.assertEqual(outcome_check.threshold, 0.05)

    def test_binary_bool_dtype_is_preserved(self):
        base = pd.DataFrame({
            "feature": np.arange(100),
            "target": [True] * 5 + [False] * 95,
        })

        injected, _ = inject(base, PARAMS, np.random.default_rng(42))

        self.assertEqual(injected["target"].dtype, base["target"].dtype)

    def test_binary_injection_rejects_too_few_zero_rows(self):
        base = pd.DataFrame({
            "feature": np.arange(100),
            "target": [0] * 5 + [1] * 95,
        })

        with self.assertRaisesRegex(ValueError, "zero-label rows"):
            inject(base, PARAMS, np.random.default_rng(42))

    def test_binary_injection_rejects_no_clean_class_separation(self):
        base = pd.DataFrame({
            "feature": np.arange(100),
            "target": [0] * 10 + [1] * 90,
        })

        with self.assertRaisesRegex(ValueError, "effect size > 0.05"):
            inject(base, PARAMS, np.random.default_rng(42))

    def test_nonbinary_target_keeps_continuous_corruption(self):
        base = pd.DataFrame({
            "feature": np.arange(100),
            "target": np.full(100, 0.25),
        })

        injected, metadata = inject(base, PARAMS, np.random.default_rng(42))

        self.assertEqual(
            metadata["effects"]["corruption_mode"],
            "continuous_positive_shift",
        )
        self.assertTrue(validate(injected, metadata["effects"], "target").passed)

    def test_outcome_difference_threshold_is_strict(self):
        exact = pd.DataFrame({
            "target": [-1.0, 0.0, 1.0, 0.05],
            "flag": [0, 0, 0, 1],
        })
        above = exact.copy()
        above.loc[3, "target"] = 0.050001

        exact_result = validate(exact, {"indicator_col": "flag"}, "target")
        above_result = validate(above, {"indicator_col": "flag"}, "target")

        self.assertFalse(exact_result.passed)
        self.assertTrue(above_result.passed)
        self.assertEqual(exact_result.checks[-1].threshold, 0.05)


if __name__ == "__main__":
    unittest.main()
