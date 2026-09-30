from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import answer_pipeline  # noqa: E402
import phenomena_pipeline  # noqa: E402
import validate_pipeline  # noqa: E402
from io_utils import load_csv, save_csv  # noqa: E402
from phenomena import PHENOMENA, TEMPLATE_TO_PHENOMENON  # noqa: E402
from phenomena._base import AnswerUnavailable  # noqa: E402
from phenomena.fc_nonmonotone_peak import (  # noqa: E402
    PHENOMENON,
    compute_answer_v0,
    estimate_observed_peak,
    is_integer_valued,
    validate,
    verify_observed_peak,
)

TEMPLATE_ID = "fc_nonmonotone_peak_v0"


def _frame(
    target: np.ndarray,
    *,
    x: np.ndarray | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if x is None:
        x = np.linspace(0.0, 100.0, len(target))
    return pd.DataFrame({
        "peak_feature": x,
        "noise_a": rng.normal(size=len(x)),
        "noise_b": rng.normal(size=len(x)),
        "target": target,
    })


def _integer_frame(centre: float, *, seed: int = 11, n: int = 1200) -> pd.DataFrame:
    """Integer feature 1..12 with a smooth inverted-U centred at ``centre``."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 13, size=n).astype(float)
    target = 100.0 * (1.0 - ((x - centre) / 6.0) ** 2) + rng.normal(0.0, 3.0, n)
    return _frame(target, x=x, seed=seed)


# Reference instances on the tracked 1000-row tables (seed 42). The SHA-256 is
# of table.csv with CRLF line endings; the gold is the observed-peak answer on
# that table. Regenerating with the same seed must reproduce all of them byte
# for byte.
_REFERENCE_INSTANCES = {
    "bike_sharing_1000": (
        "99c7dba9f40b3d5a95b1553a2f2094cda3082494bd6d90588e86f258eb073fc4",
        'mnth, 7',
    ),
    "california_housing_1000": (
        "60050b1386f6dab8e697775d28dfd1c4be9464c702f2f9248faf7a39bcad5141",
        'HouseAge, 30',
    ),
    "kc_housing_1000": (
        "9c7b9684c8deb8cf11491bb3a999d48d25dc88229dcda980807d416589fabb04",
        'yr_built, 1993',
    ),
    "metro_interstate_traffic_volume_1000": (
        "e0b2b2cceb2f95f61532c19703da2ee6ed0a7eed19c92c4f6c39644d428187e9",
        'month, 7',
    ),
    "steel_industry_data_1000": (
        "d563ee376ee73b3f8b3e31f2fd55167ebed336ca1061ff5a4a06280ebd140a7a",
        'NSM, 39636',
    ),
}


def _crlf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(data).hexdigest()


def _legacy_manifest(frame: pd.DataFrame, root: Path, *, injected_centre: float) -> Path:
    instance = root / "dataset" / "seed_42" / "fc_nonmonotone_peak"
    instance.mkdir(parents=True)
    save_csv(frame, instance / "table.csv")
    manifest = {
        "dataset_name": "dataset",
        "seed": 42,
        "target": "target",
        "phenomenon": {
            "injector_type": "fc_nonmonotone_peak",
            "effects": {
                "peak_feature": "peak_feature",
                "injected_centre": injected_centre,
                "amplitude": 20.0,
                "half_range": 50.0,
                "id_no_cols": [],
            },
        },
        "qa_pairs": [
            {
                "template_id": TEMPLATE_ID,
                "question": f"question for {TEMPLATE_ID}",
                "category": "Feature Contribution",
                "slot_assignments": {"OUTCOME_COL": "target"},
                "answer": "stale, 1",
            }
        ],
        "validation": {"passed": True, "checks": []},
    }
    manifest_path = instance / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class NonmonotonePeakTests(unittest.TestCase):
    def test_registry_binding(self):
        self.assertIs(PHENOMENA["fc_nonmonotone_peak"], PHENOMENON)
        self.assertIs(TEMPLATE_TO_PHENOMENON[TEMPLATE_ID], PHENOMENON)
        self.assertIs(PHENOMENON.compute_answers[TEMPLATE_ID], compute_answer_v0)
        self.assertEqual(PHENOMENON.summary_fields, ("id_no",))

    def test_stage2_validator_checks_the_injected_centre(self):
        rng = np.random.default_rng(101)
        x = np.linspace(0.0, 100.0, 600)
        target = 20.0 * (1.0 - ((x - 50.0) / 50.0) ** 2)
        target += rng.normal(0.0, 0.5, len(x))
        result = validate(
            _frame(target, x=x),
            {"peak_feature": "peak_feature", "injected_centre": 50.0},
            "target",
        )

        self.assertTrue(result.passed)
        self.assertEqual(
            [check.name for check in result.checks],
            [
                "peak_bin_unique",
                "peak_not_at_edge",
                "injected_centre_within_observed_peak_bin",
                "dominance_nonmonotone_score",
            ],
        )

    def test_verifier_is_stage2_checks_with_observed_peak(self):
        rng = np.random.default_rng(101)
        x = np.linspace(0.0, 100.0, 600)
        target = 0.2 * x + 20.0 * (1.0 - ((x - 40.0) / 50.0) ** 2)
        target += rng.normal(0.0, 0.5, len(x))
        result, gold = verify_observed_peak(
            _frame(target, x=x),
            {"peak_feature": "peak_feature", "injected_centre": 40.0},
            "target",
        )

        self.assertTrue(result.passed, msg=[c.detail for c in result.checks])
        self.assertEqual(
            [check.name for check in result.checks],
            [
                "peak_bin_unique",
                "peak_not_at_edge",
                "observed_peak_in_peak_bin",
                "dominance_nonmonotone_score",
            ],
        )
        # No grading-tolerance / precision checks in the verifier.
        names = {c.name for c in result.checks}
        for banned in ("bootstrap", "precision", "resolution", "agreement"):
            self.assertFalse(any(banned in n for n in names), names)
        self.assertIsNotNone(gold)

    def test_regenerates_reference_tables_and_golds_byte_for_byte(self):
        summaries = REPO_ROOT / "data/standardized/summaries"
        for dataset, (sha256, gold) in _REFERENCE_INSTANCES.items():
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                with redirect_stdout(StringIO()):
                    dirs = phenomena_pipeline.build_instance(
                        summaries / f"{dataset}.json",
                        42,
                        REPO_ROOT / "templates",
                        template_filter=[TEMPLATE_ID],
                        output_root=Path(tmp),
                    )
                self.assertEqual(len(dirs), 1)
                instance = dirs[0]
                manifest_path = instance / "manifest.json"
                table_path = instance / "table.csv"
                self.assertEqual(_crlf_sha256(table_path), sha256)

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    [qa["template_id"] for qa in manifest["qa_pairs"]],
                    [TEMPLATE_ID],
                )
                with redirect_stdout(StringIO()):
                    validate_pipeline.run(
                        Path(tmp),
                        dataset_filter=None,
                        injector_filter=None,
                        force=False,
                        manifest_paths=[manifest_path],
                    )
                    answer_pipeline.process_instance(manifest_path)
                saved = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertTrue(saved["validation"]["passed"])
                self.assertEqual(saved["qa_pairs"][0]["answer"], gold)
                self.assertEqual(
                    saved["phenomenon"]["effects"],
                    manifest["phenomenon"]["effects"],
                )
                self.assertEqual(_crlf_sha256(table_path), sha256)

    def test_gold_is_the_observed_peak_not_the_injected_centre(self):
        rng = np.random.default_rng(101)
        x = np.linspace(0.0, 100.0, 600)
        target = 0.2 * x + 20.0 * (1.0 - ((x - 40.0) / 50.0) ** 2)
        target += rng.normal(0.0, 0.5, len(x))
        frame = _frame(target, x=x)
        effects = {
            "peak_feature": "peak_feature",
            "injected_centre": 40.0,
            "id_no_cols": [],
        }
        effects_before = copy.deepcopy(effects)

        answer = compute_answer_v0(frame, {"OUTCOME_COL": "target"}, effects)
        gold = estimate_observed_peak(frame, "peak_feature", "target")

        self.assertFalse(gold.integer_feature)
        self.assertEqual(answer, gold.answer)
        self.assertNotEqual(answer, "peak_feature, 40.0")
        # The analytic vertex of the final curve is 52.5, not 40.
        self.assertAlmostEqual(gold.raw_estimate, 52.5, delta=2.0)
        self.assertEqual(effects, effects_before)

    def test_integer_feature_gold_rounds_to_nearest_integer(self):
        frame = _integer_frame(4.3)
        gold = estimate_observed_peak(frame, "peak_feature", "target")

        self.assertTrue(gold.integer_feature)
        self.assertAlmostEqual(gold.raw_estimate, 4.3, delta=0.15)
        self.assertEqual(gold.value, 4.0)
        self.assertEqual(gold.answer, "peak_feature, 4")
        self.assertEqual(
            compute_answer_v0(frame, {"OUTCOME_COL": "target"}, {
                "peak_feature": "peak_feature", "injected_centre": 4.0,
            }),
            "peak_feature, 4",
        )

    def test_integer_feature_near_midpoint_still_rounds_to_one_integer(self):
        frame = _integer_frame(4.5)
        gold = estimate_observed_peak(frame, "peak_feature", "target")

        self.assertTrue(gold.integer_feature)
        self.assertAlmostEqual(gold.raw_estimate, 4.5, delta=0.15)
        self.assertIn(gold.value, (4.0, 5.0))
        self.assertIn(gold.answer, ("peak_feature, 4", "peak_feature, 5"))
        self.assertNotIn(" or ", gold.answer)

    def test_integer_valued_float_column_counts_as_integer(self):
        frame = _integer_frame(7.0)
        frame["peak_feature"] = frame["peak_feature"].astype(float)
        self.assertEqual(frame["peak_feature"].dtype, np.float64)
        self.assertTrue(is_integer_valued(frame["peak_feature"]))
        self.assertFalse(is_integer_valued(pd.Series([1.0, 2.5, 3.0])))
        gold = estimate_observed_peak(frame, "peak_feature", "target")
        self.assertEqual(gold.answer, "peak_feature, 7")

    def test_continuous_feature_keeps_decimal_gold(self):
        rng = np.random.default_rng(43)
        x = np.linspace(0.001, 1.001, 600)
        target = 10.0 * (1.0 - ((x - 0.4137) / 0.5) ** 2)
        target += rng.normal(0.0, 0.05, len(x))
        gold = estimate_observed_peak(_frame(target, x=x), "peak_feature", "target")

        self.assertFalse(gold.integer_feature)
        self.assertAlmostEqual(gold.value, 0.4137, delta=0.03)

    def test_unavailable_gold_clears_the_answer_without_failing_the_stage(self):
        rng = np.random.default_rng(103)
        x = np.linspace(0.0, 100.0, 600)
        # Monotone increasing: no interior peak, so the verifier's edge check fails.
        target = 0.5 * x + rng.normal(0.0, 0.5, len(x))
        frame = _frame(target, x=x)

        with self.assertRaises(AnswerUnavailable):
            compute_answer_v0(frame, {"OUTCOME_COL": "target"}, {
                "peak_feature": "peak_feature",
                "injected_centre": 50.0,
            })

        with tempfile.TemporaryDirectory() as tmp:
            # Manifest whose stored validation is PASS; the answer stage trusts
            # it, clears the stale answer and reports NO_GOLD without failing.
            manifest_path = _legacy_manifest(frame, Path(tmp), injected_centre=50.0)
            table_path = manifest_path.parent / "table.csv"
            table_before = table_path.read_bytes()
            original = json.loads(manifest_path.read_text(encoding="utf-8"))

            with redirect_stdout(StringIO()):
                rows = answer_pipeline.process_instance(manifest_path)

            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            statuses = {row[2]: row[-1] for row in rows}

            self.assertTrue(saved["validation"]["passed"])
            self.assertIsNone(saved["qa_pairs"][0]["answer"])
            self.assertEqual(statuses[TEMPLATE_ID], "NO_GOLD")
            self.assertEqual(
                saved["phenomenon"]["effects"],
                original["phenomenon"]["effects"],
            )
            self.assertEqual(
                [qa["question"] for qa in saved["qa_pairs"]],
                [qa["question"] for qa in original["qa_pairs"]],
            )
            self.assertEqual(table_before, table_path.read_bytes())

    def test_round_trip_csv_controls_gold_without_changing_csv(self):
        rng = np.random.default_rng(43)
        x = np.linspace(0.001, 1.001, 600)
        target = 10.0 * (1.0 - ((x - 0.4137) / 0.5) ** 2)
        target += rng.normal(0.0, 0.05, len(x))
        frame = _frame(target, x=x)
        effects = {"peak_feature": "peak_feature", "injected_centre": 0.4}

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "table.csv"
            save_csv(frame.round(4), csv_path)
            before = csv_path.read_bytes()
            answer = compute_answer_v0(
                load_csv(csv_path),
                {"OUTCOME_COL": "target"},
                effects,
            )

            self.assertEqual(before, csv_path.read_bytes())
            self.assertEqual(
                answer,
                compute_answer_v0(
                    load_csv(csv_path),
                    {"OUTCOME_COL": "target"},
                    effects,
                ),
            )


if __name__ == "__main__":
    unittest.main()
