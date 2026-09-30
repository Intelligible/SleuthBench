from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phenomena import dq_unreliable_feature  # noqa: E402


class _DeterministicRng:
    def permutation(self, values):
        return np.arange(values)

    def normal(self, _loc, scale):
        return np.zeros(len(scale))


class IdentifierGuardTests(unittest.TestCase):
    def setUp(self):
        repeated = np.tile(np.arange(10, dtype=float) + 0.25, 2)
        self.df = pd.DataFrame({
            "row_id": np.arange(20),
            "serial_number": np.arange(100, 120),
            "declared_id": repeated,
            "feature_a": repeated,
            "feature_b": np.roll(repeated, 1),
            "feature_c": np.roll(repeated, 2),
            "target": np.arange(20, dtype=float) + 0.125,
        })

    def test_unreliable_feature_excludes_and_rejects_ids(self):
        self.assertEqual(
            dq_unreliable_feature.PHENOMENON.summary_fields,
            ("id_no",),
        )
        with patch(
            "phenomena.dq_unreliable_feature.heteroskedasticity_score",
            return_value=3.0,
        ):
            _, metadata = dq_unreliable_feature.inject(
                self.df,
                {
                    "outcome_col": "target",
                    "id_no_cols": ["declared_id"],
                },
                _DeterministicRng(),
            )

        self.assertEqual(metadata["effects"]["unreliable_feature"], "feature_a")
        self.assertEqual(
            set(metadata["effects"]["id_no_cols"]),
            {"declared_id", "row_id", "serial_number"},
        )

        result = dq_unreliable_feature.validate(
            self.df,
            {"unreliable_feature": "row_id"},
            "target",
        )
        self.assertFalse(result.passed)
        self.assertEqual(
            result.checks[0].name,
            "unreliable_feature_not_identifier",
        )


if __name__ == "__main__":
    unittest.main()
