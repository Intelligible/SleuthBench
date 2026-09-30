"""Bad-rows indicator phenomenon.

Injects a 0/1 column whose 1-values mark rows with corrupted outcomes. Both
the injector and the validator require flagged rows to have a higher mean
outcome than clean rows by more than 0.05 clean standard deviations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import coerce_numeric

from ._base import CheckDetail, InjectionRejected, Phenomenon, ValidationResult


_CANDIDATE_NAMES = [
    "region", "channel", "segment", "cohort", "tier",
    "variant", "cluster", "period", "quarter", "sample",
    "trial", "run", "split", "bucket", "partition",
    "group", "class", "phase", "batch", "slot",
]

_OUTCOME_DIFFERENCE_THRESHOLD = 0.05


def _indicator_exists_check(
    df: pd.DataFrame,
    indicator_col: str,
) -> CheckDetail:
    """Return the indicator-column presence check."""
    present = indicator_col in df.columns
    return CheckDetail(
        name="indicator_exists",
        passed=present,
        metric=1.0 if present else 0.0,
        threshold=1.0,
        detail=(
            f"Indicator column '{indicator_col}' "
            f"{'found' if present else 'not found in data'}"
        ),
    )


def _select_bad_indices(
    outcome_values: pd.Series,
    n_bad: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Select rows to corrupt and report whether the target is binary 0/1."""
    observed = set(outcome_values.dropna().unique().tolist())
    is_binary = observed == {0.0, 1.0}

    if is_binary:
        zero_positions = np.flatnonzero(outcome_values.to_numpy() == 0.0)
        if len(zero_positions) < n_bad:
            raise InjectionRejected(
                f"Need {n_bad} zero-label rows for binary corruption, "
                f"found {len(zero_positions)}"
            )
        return rng.choice(zero_positions, size=n_bad, replace=False), True

    return rng.choice(len(outcome_values), size=n_bad, replace=False), False


def _outcome_difference_check(
    outcome: pd.Series,
    indicator: pd.Series,
) -> CheckDetail:
    """Apply the shared flagged-vs-clean validation rule."""
    flagged = outcome[indicator == 1]
    clean = outcome[indicator == 0]

    if len(flagged) == 0 or len(clean) == 0:
        return CheckDetail(
            name="outcome_difference", passed=False,
            metric=0.0, threshold=_OUTCOME_DIFFERENCE_THRESHOLD,
            detail=f"Flagged = {len(flagged)} rows, clean = {len(clean)} rows",
        )

    mean_diff = flagged.mean() - clean.mean()
    clean_std = clean.std()
    if clean_std == 0:
        clean_std = 1.0
    effect_size = mean_diff / clean_std
    diff_ok = bool(effect_size > _OUTCOME_DIFFERENCE_THRESHOLD)
    return CheckDetail(
        name="outcome_difference", passed=diff_ok,
        metric=effect_size, threshold=_OUTCOME_DIFFERENCE_THRESHOLD,
        detail=f"Flagged-clean mean diff = {effect_size:.4f} std devs",
    )


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Plant a binary "bad rows" flag column whose 1-values mark rows with
    corrupted outcomes.

    For a binary 0/1 target, chooses zero-label rows and flips their target to
    1. For any other numeric target, chooses rows uniformly at random and adds
    a positive half-normal shift, clipped at ``corruption_cap_std`` times the
    outcome standard deviation. Integer-valued non-binary targets are rounded
    back to integers. A new 0/1 indicator column marks the corrupted rows and
    is inserted at a random position with a plausible-sounding name. The final
    flagged-vs-clean effect must pass the same threshold used by the validator.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    bad_fraction = float(params.get("bad_fraction", 0.1))
    corruption_cap_std = float(params.get("corruption_cap_std", 2.0))

    n = len(df)
    n_bad = max(1, int(n * bad_fraction))
    outcome_values = df[outcome_col].astype(float)
    bad_indices, is_binary = _select_bad_indices(outcome_values, n_bad, rng)

    existing = set(df.columns)
    candidates = [n for n in _CANDIDATE_NAMES if n not in existing]
    indicator_col = candidates[rng.integers(0, len(candidates))] + "_flag"
    insert_pos = rng.integers(0, max(1, len(df.columns)))
    if insert_pos == len(df.columns):
        insert_pos = max(0, len(df.columns) - 1)
    df.insert(insert_pos, indicator_col, 0)
    df.iloc[bad_indices, df.columns.get_loc(indicator_col)] = 1

    if is_binary:
        positive_value = df.loc[outcome_values == 1.0, outcome_col].iloc[0]
        new_values = np.full(n_bad, positive_value)
        corruption_mode = "binary_zero_to_one"
    else:
        outcome_std = outcome_values.std()
        if outcome_std == 0:
            outcome_std = 1.0

        corruption = np.abs(rng.normal(0, outcome_std * 0.5, n_bad))
        corruption = np.minimum(corruption, outcome_std * corruption_cap_std)
        new_values = outcome_values.iloc[bad_indices].values + corruption
        non_null = outcome_values.dropna()
        if len(non_null) > 0 and np.all(np.isclose(non_null.values, np.rint(non_null.values))):
            new_values = np.rint(new_values).astype(float)
        corruption_mode = "continuous_positive_shift"
    df.iloc[bad_indices, df.columns.get_loc(outcome_col)] = new_values

    difference_check = _outcome_difference_check(
        coerce_numeric(df[outcome_col]), df[indicator_col]
    )
    if not difference_check.passed:
        raise InjectionRejected(
            "Injected bad rows failed outcome-difference validation: "
            f"{difference_check.detail}; require effect size > "
            f"{_OUTCOME_DIFFERENCE_THRESHOLD}"
        )

    effects = {
        "indicator_col": indicator_col,
        "n_bad_rows": int(n_bad),
        "bad_fraction": bad_fraction,
        "corruption_cap_std": corruption_cap_std,
        "corruption_mode": corruption_mode,
        "bad_indices": bad_indices.tolist(),
    }

    return df, {
        "type": "dq_bad_row_indicator",
        "params": params,
        "effects": effects,
    }


def validate(
    df: pd.DataFrame,
    effects: dict,
    target_col: str,
    *,
    indicator_check: CheckDetail | None = None,
) -> ValidationResult:
    indicator_col = effects["indicator_col"]
    if indicator_check is None:
        indicator_check = _indicator_exists_check(df, indicator_col)
    checks = [indicator_check]
    if not indicator_check.passed:
        return ValidationResult(passed=False, checks=checks)

    difference_check = _outcome_difference_check(
        coerce_numeric(df[target_col]), df[indicator_col]
    )
    checks.append(difference_check)
    return ValidationResult(passed=difference_check.passed, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the indicator column name that identifies bad rows."""
    return effects["indicator_col"]


PHENOMENON = Phenomenon(
    name="dq_bad_row_indicator",
    inject=inject,
    validate=validate,
    compute_answers={
        "dq_bad_row_indicator_v0": compute_answer_v0,
    },
)
