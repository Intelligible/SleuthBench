from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phenomena.dq_bad_row_indicator_v1 import (  # noqa: E402
    PHENOMENON,
    _continuous_proposals,
    _ResidualProfile,
    compute_answer_v1,
    inject,
    validate,
)

FAST_PARAMS = {
    "outcome_col": "target",
    "bad_fraction": 0.05,
    "n_estimators": 32,
    "n_splits": 3,
}


def _placebo_positions(
    rng: np.random.Generator,
    n_rows: int,
    n_positive: int,
) -> np.ndarray:
    return np.sort(rng.choice(n_rows, size=n_positive, replace=False))


def _with_placebo_flags(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Add two unrelated binary columns with the injection's prevalence."""
    result = frame.copy()
    rng = np.random.default_rng(seed)
    n_positive = int(len(result) * FAST_PARAMS["bad_fraction"])
    for name in ("placebo_flag_a", "placebo_flag_b"):
        result[name] = 0
        result.loc[
            _placebo_positions(rng, len(result), n_positive),
            name,
        ] = 1
    return result


def _regression_frame(n_rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(104729)
    x_linear = rng.uniform(-4.0, 4.0, size=n_rows)
    x_curve = rng.uniform(-2.0, 2.0, size=n_rows)
    x_noise = rng.normal(size=n_rows)
    target = (
        80.0
        + 9.0 * x_linear
        + 4.0 * np.square(x_curve)
        + 1.5 * x_noise
        + rng.normal(0.0, 0.25, size=n_rows)
    )
    frame = pd.DataFrame(
        {
            "row_id": np.arange(10_000, 10_000 + n_rows),
            "x_linear": x_linear,
            "x_curve": x_curve,
            "x_noise": x_noise,
            "target": target,
        }
    )
    return _with_placebo_flags(frame, seed=130363)


def _binary_frame(n_rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(15485863)
    x_primary = rng.normal(size=n_rows)
    x_secondary = rng.normal(size=n_rows)
    x_noise = rng.normal(size=n_rows)
    # Keep the clean positive class rare, as in AI4I.  Flipping confident
    # negatives then produces both a large residual signal and a meaningful
    # directed target-lift signal for the validator's binary-target gate.
    target = (2.5 * x_primary + 0.7 * x_secondary > 4.0).astype(int)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(20_000, 20_000 + n_rows),
            "x_primary": x_primary,
            "x_secondary": x_secondary,
            "x_noise": x_noise,
            "target": target,
        }
    )
    return _with_placebo_flags(frame, seed=32452843)


class BadRowIndicatorV1TemplateTests(unittest.TestCase):
    def test_template_states_the_indicator_semantics_without_uniqueness_hint(self):
        template_path = (
            REPO_ROOT
            / "templates"
            / "data_quality"
            / "dq_bad_row_indicator_v1.json"
        )
        template = json.loads(template_path.read_text(encoding="utf-8"))

        self.assertEqual(template["template_id"], "dq_bad_row_indicator_v1")
        self.assertEqual(template["answer_format"], "column_name")
        self.assertEqual(template["slots"]["BAD_FRACTION"]["default"], 0.05)
        self.assertEqual(
            template["phenomena"][0]["injector"],
            "dq_bad_row_indicator_v1",
        )
        for field in ("business_question", "ds_question"):
            question = template[field].lower()
            self.assertIn("0/1", question)
            self.assertIn("1", question)
            self.assertIn("otherwise similar records", question)
            self.assertNotIn("only 0/1", question)


class BadRowIndicatorV1Tests(unittest.TestCase):
    def _assert_mechanical_contract(
        self,
        source: pd.DataFrame,
        injected: pd.DataFrame,
        metadata: dict,
    ) -> tuple[str, np.ndarray]:
        effects = metadata["effects"]
        indicator = effects["indicator_col"]
        placebo_columns = effects["placebo_cols"]
        n_bad = max(3, int(round(len(source) * FAST_PARAMS["bad_fraction"])))

        self.assertEqual(metadata["type"], "dq_bad_row_indicator_v1")
        self.assertIn(indicator, injected.columns)
        self.assertNotIn(indicator, source.columns)
        self.assertEqual(set(injected[indicator].unique()), {0, 1})
        self.assertEqual(int(injected[indicator].sum()), n_bad)
        self.assertEqual(effects["n_bad_rows"], n_bad)
        self.assertGreaterEqual(len(placebo_columns), 1)
        self.assertTrue(
            all(int(injected[column].sum()) == n_bad for column in placebo_columns)
        )

        # The two input placebos and the injector's placebos remain in the
        # anonymised output, so the answer is not the only low-prevalence bit.
        same_prevalence_binary = []
        for column in injected.columns:
            if column in {"target", "row_id"}:
                continue
            numeric = pd.to_numeric(injected[column], errors="coerce")
            if (
                numeric.notna().all()
                and set(numeric.unique()) == {0, 1}
                and int(numeric.sum()) == n_bad
            ):
                same_prevalence_binary.append(column)
        self.assertGreaterEqual(len(same_prevalence_binary), 6)

        flagged = np.flatnonzero(injected[indicator].to_numpy() == 1)
        changed = np.flatnonzero(
            ~np.isclose(
                source["target"].to_numpy(dtype=float),
                injected["target"].to_numpy(dtype=float),
            )
        )
        np.testing.assert_array_equal(changed, flagged)
        self.assertEqual(sorted(effects["bad_indices"]), flagged.tolist())

        self.assertGreaterEqual(
            float(injected["target"].min()),
            float(source["target"].min()),
        )
        self.assertLessEqual(
            float(injected["target"].max()),
            float(source["target"].max()),
        )
        self.assertEqual(injected["target"].dtype, source["target"].dtype)
        pd.testing.assert_series_equal(injected["row_id"], source["row_id"])

        # Apart from target corruption and the newly planted flags, injection
        # only anonymises the original feature columns; it must not alter them.
        generated = {indicator, *placebo_columns}
        payload_columns = [
            column
            for column in injected.columns
            if column not in {"target", "row_id"} | generated
        ]
        original_columns = [
            column for column in source.columns if column not in {"target", "row_id"}
        ]
        self.assertEqual(len(payload_columns), len(original_columns))
        unmatched = list(payload_columns)
        for original in original_columns:
            for candidate in unmatched:
                try:
                    pd.testing.assert_series_equal(
                        injected[candidate],
                        source[original],
                        check_names=False,
                    )
                except AssertionError:
                    continue
                unmatched.remove(candidate)
                break
            else:
                self.fail(f"No anonymised output column preserves {original!r}")

        support = effects["target_support"]
        self.assertEqual(support["original_min"], float(source["target"].min()))
        self.assertEqual(support["original_max"], float(source["target"].max()))

        result = validate(injected, effects, "target")
        self.assertTrue(
            result.passed,
            msg="; ".join(check.detail for check in result.checks),
        )
        return indicator, flagged

    def test_regression_injection_is_support_bounded_and_observable(self):
        source = _regression_frame()
        injected, metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(42),
        )

        self._assert_mechanical_contract(source, injected, metadata)

    def test_regression_float32_target_preserves_dtype(self):
        source = _regression_frame()
        source["target"] = source["target"].astype("float32")

        injected, metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(42),
        )

        self._assert_mechanical_contract(source, injected, metadata)
        self.assertEqual(injected["target"].dtype, source["target"].dtype)

    def test_continuous_proposal_rechecks_float32_cast_at_support_boundary(self):
        rounded_high = np.float32(128.0)
        previous_float32 = np.nextafter(
            rounded_high,
            np.float32(-np.inf),
            dtype=np.float32,
        )
        allowed_high = float(previous_float32) + 0.75 * (
            float(rounded_high) - float(previous_float32)
        )
        raw_proposal = allowed_high

        self.assertLessEqual(raw_proposal, allowed_high)
        self.assertGreater(float(np.float32(raw_proposal)), allowed_high)

        profile = _ResidualProfile(
            predictions=np.asarray([0.0]),
            signed_residuals=np.asarray([0.0]),
            residual_scores=np.asarray([0.0]),
            raw_features=(),
            target_is_binary=False,
        )
        proposals = _continuous_proposals(
            target=np.asarray([-1.0]),
            profile=profile,
            pool=np.asarray([0]),
            allowed_low=0.0,
            allowed_high=allowed_high,
            amplitude=raw_proposal,
            integer_target=False,
            target_dtype=np.dtype("float32"),
        )

        self.assertEqual(proposals, {})

    def test_binary_injection_flips_conditionally_expected_zero_rows(self):
        source = _binary_frame()
        injected, metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(42),
        )

        _, flagged = self._assert_mechanical_contract(
            source,
            injected,
            metadata,
        )
        self.assertTrue((source.iloc[flagged]["target"] == 0).all())
        self.assertTrue((injected.iloc[flagged]["target"] == 1).all())
        self.assertEqual(set(injected["target"].unique()), {0, 1})

    def test_binary_bool_target_preserves_labels_and_dtype(self):
        source = _binary_frame()
        source["target"] = source["target"].astype(bool)

        injected, metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(42),
        )

        _, flagged = self._assert_mechanical_contract(
            source,
            injected,
            metadata,
        )
        self.assertEqual(injected["target"].dtype, source["target"].dtype)
        self.assertEqual(set(injected["target"].unique()), {False, True})
        self.assertTrue((~source.iloc[flagged]["target"]).all())
        self.assertTrue(injected.iloc[flagged]["target"].all())

    def test_binary_nonzero_one_labels_preserve_original_classes(self):
        source = _binary_frame()
        source["target"] = source["target"].map({0: -1, 1: 1}).astype("int64")

        injected, metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(42),
        )

        _, flagged = self._assert_mechanical_contract(
            source,
            injected,
            metadata,
        )
        self.assertEqual(injected["target"].dtype, source["target"].dtype)
        self.assertEqual(set(injected["target"].unique()), {-1, 1})
        self.assertTrue((source.iloc[flagged]["target"] == -1).all())
        self.assertTrue((injected.iloc[flagged]["target"] == 1).all())

    def test_injection_is_deterministic_for_the_same_rng_seed(self):
        source = _regression_frame()

        first_df, first_metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(1729),
        )
        second_df, second_metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(1729),
        )

        pd.testing.assert_frame_equal(first_df, second_df)
        self.assertEqual(
            json.dumps(first_metadata, sort_keys=True),
            json.dumps(second_metadata, sort_keys=True),
        )

    def test_validator_rejects_tampering_support_breach_and_tied_flag(self):
        source = _regression_frame()
        injected, metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(271828),
        )
        effects = metadata["effects"]
        indicator = effects["indicator_col"]

        tampered_flag = injected.copy()
        tampered_flag[indicator] = tampered_flag[effects["placebo_cols"][0]]
        self.assertFalse(validate(tampered_flag, effects, "target").passed)

        support_breach = injected.copy()
        support_breach.loc[support_breach.index[0], "target"] = (
            float(source["target"].max()) + 1.0
        )
        self.assertFalse(validate(support_breach, effects, "target").passed)

        ambiguous = injected.copy()
        ambiguous["same_prevalence_copy"] = ambiguous[indicator]
        self.assertFalse(validate(ambiguous, effects, "target").passed)

        mismatched_effects = dict(effects)
        mismatched_effects["bad_indices"] = np.flatnonzero(
            injected[effects["placebo_cols"][0]].to_numpy() == 1
        ).tolist()
        self.assertFalse(validate(injected, mismatched_effects, "target").passed)

    def test_answer_and_registry_contract_return_indicator_name(self):
        source = _regression_frame()
        injected, metadata = inject(
            source,
            FAST_PARAMS,
            np.random.default_rng(65537),
        )
        effects = metadata["effects"]

        self.assertEqual(
            compute_answer_v1(injected, {"OUTCOME_COL": "target"}, effects),
            effects["indicator_col"],
        )
        self.assertEqual(PHENOMENON.name, "dq_bad_row_indicator_v1")
        self.assertIs(
            PHENOMENON.compute_answers["dq_bad_row_indicator_v1"],
            compute_answer_v1,
        )


if __name__ == "__main__":
    unittest.main()
