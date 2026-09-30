from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phenomena import dq_conditional_bad_rows  # noqa: E402


class ConditionalBadRowsTests(unittest.TestCase):
    def test_standardize_requests_writable_copy(self) -> None:
        frame = pd.DataFrame(
            {
                "first": [1.0, np.nan, 3.0],
                "second": [4.0, 5.0, 6.0],
            }
        )
        original_to_numpy = pd.DataFrame.to_numpy

        def readonly_unless_copied(self, *args, **kwargs):
            values = original_to_numpy(self, *args, **kwargs)
            if not kwargs.get("copy", False):
                values.setflags(write=False)
            return values

        with patch.object(pd.DataFrame, "to_numpy", readonly_unless_copied):
            standardized = dq_conditional_bad_rows._standardize(
                frame,
                ["first", "second"],
            )

        self.assertTrue(np.isfinite(standardized).all())
        np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
