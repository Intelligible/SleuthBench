"""Noise-feature phenomenon.

Permutes one existing numeric feature while preserving its marginal
distribution. The ground-truth answer is the permuted column. Validation uses
binned predictive importance and requires it to rank below every other
eligible feature.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import (
    binned_importance,
    coerce_numeric,
    eligible_numeric_columns,
    id_like_cols,
)

from ._base import CheckDetail, InjectionRejected, Phenomenon, ValidationResult


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Replace one eligible numeric feature with a permutation of its values.

    Excludes the target, identifier columns, and explicit ``exclude_cols``.
    Eligible candidates are seed-shuffled and tried in that order. If a
    permutation does not reduce importance, the next candidate is tried.
    """
    df = df.copy()
    target_col = params["target_col"]

    id_no_cols = (
        set(params.get("id_no_cols", []) or [])
        | id_like_cols(df)
    )
    id_no_cols.discard(target_col)

    exclude_cols = set(params.get("exclude_cols", []) or [])
    exclude_cols.update(id_no_cols)
    exclude_cols.add(target_col)

    numeric_cols = eligible_numeric_columns(
        df,
        exclude=exclude_cols,
        min_unique=5,
    )

    if len(numeric_cols) < 3:
        raise InjectionRejected(
            f"Need at least 3 eligible numeric features, found {len(numeric_cols)}"
        )

    target = df[target_col].astype(float)

    pre_importance = {
        c: binned_importance(df[c].astype(float), target)
        for c in numeric_cols
    }

    candidates = sorted(pre_importance, key=pre_importance.get, reverse=True)
    order = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in order]

    last_error = None

    for chosen_feature in candidates:
        original_importance = pre_importance[chosen_feature]

        original_values = df[chosen_feature].values.copy()
        trial_values = rng.permutation(original_values)

        noise_importance = binned_importance(
            pd.Series(trial_values, dtype=float), target
        )
        if noise_importance >= original_importance:
            last_error = (
                f"Permuted importance ({noise_importance:.4f}) did not drop "
                f"below original ({original_importance:.4f}) for '{chosen_feature}'"
            )
            continue

        df[chosen_feature] = trial_values

        effects = {
            "noise_feature": str(chosen_feature),
            "original_importance": float(original_importance),
            "noise_importance": float(noise_importance),
            "id_no_cols": sorted(id_no_cols),
        }

        return df, {
            "type": "fc_noise_feature",
            "params": params,
            "effects": effects,
        }

    raise InjectionRejected(
        f"No candidate survived validation. Last error: {last_error}"
    )


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    noise_col = effects["noise_feature"]
    id_no_cols = (
        set(effects.get("id_no_cols", []) or [])
        | id_like_cols(df)
    )
    id_no_cols.discard(target_col)
    checks = []

    if noise_col in id_no_cols:
        checks.append(CheckDetail(
            name="noise_feature_not_identifier", passed=False,
            metric=0.0, threshold=1.0,
            detail=f"Noise feature '{noise_col}' is identifier-like",
        ))
        return ValidationResult(passed=False, checks=checks)

    outcome_vals = coerce_numeric(df[target_col])

    noise_importance = binned_importance(
        coerce_numeric(df[noise_col]), outcome_vals
    )

    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {target_col, noise_col},
        min_unique=5,
    )

    for col in numeric_cols:
        other_importance = binned_importance(df[col].astype(float), outcome_vals)
        if other_importance <= noise_importance:
            checks.append(CheckDetail(
                name="dominance_lowest_importance", passed=False,
                metric=other_importance, threshold=noise_importance,
                detail=f"Feature '{col}' has importance ({other_importance:.4f}) <= "
                       f"noise '{noise_col}' ({noise_importance:.4f})",
            ))
            return ValidationResult(passed=False, checks=checks)

    checks.append(CheckDetail(
        name="dominance_lowest_importance", passed=True,
        metric=noise_importance, threshold=0.0,
        detail=f"Noise importance {noise_importance:.4f}",
    ))
    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the feature that is pure noise (permuted column)."""
    return effects["noise_feature"]


PHENOMENON = Phenomenon(
    name="fc_noise_feature",
    inject=inject,
    validate=validate,
    compute_answers={
        "fc_noise_feature_v0": compute_answer_v0,
    },
    summary_fields=("id_no",),
)
