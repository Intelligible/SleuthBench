"""Heteroskedastic uncertainty phenomenon.

Adds feature-dependent target noise whose variance increases across one
numeric feature. The ground-truth answer is that feature, and validation
requires it to have the strongest eligible heteroskedasticity score.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import (
    coerce_numeric,
    decimal_places,
    eligible_numeric_columns,
    heteroskedasticity_score,
    id_like_cols,
    safe_std,
)

from ._base import CheckDetail, InjectionRejected, Phenomenon, ValidationResult


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Plant feature-dependent target noise for one eligible numeric feature.

    Requires at least three non-identifier numeric features. Eligible
    candidates are seed-shuffled and tried in that order. Noise variance
    increases with the normalised feature value, and a candidate must produce
    a post-injection variance ratio of at least 2.0.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    noise_multiplier = float(params.get("noise_multiplier", "3.0"))
    id_no_cols = set(params.get("id_no_cols", []) or []) | id_like_cols(df)
    id_no_cols.discard(outcome_col)

    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {outcome_col},
        min_unique=5,
    )

    if len(numeric_cols) < 3:
        raise InjectionRejected(
            f"Need at least 3 eligible numeric features, found {len(numeric_cols)}"
        )

    outcome_vals = df[outcome_col].astype(float)
    outcome_std = safe_std(outcome_vals)
    outcome_dp = decimal_places(outcome_vals)

    pre_scores = {
        c: heteroskedasticity_score(df[c].astype(float), outcome_vals)
        for c in numeric_cols
    }
    candidates = sorted(pre_scores, key=pre_scores.get)
    order = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in order]

    last_error = None

    for chosen_feature in candidates:
        feature_vals = df[chosen_feature].astype(float)

        feat_min = float(feature_vals.min())
        feat_max = float(feature_vals.max())
        feat_range = feat_max - feat_min
        if feat_range == 0:
            last_error = f"Feature '{chosen_feature}' has zero range"
            continue
        normalised = (feature_vals - feat_min) / feat_range

        noise_std_per_row = (0.2 + noise_multiplier * normalised) * outcome_std
        noise = rng.normal(0, noise_std_per_row)
        trial_outcome = (outcome_vals + noise).round(outcome_dp)

        injected_score = heteroskedasticity_score(feature_vals, trial_outcome)
        if injected_score < 2.0:
            last_error = (
                f"Heteroskedasticity too weak for '{chosen_feature}': "
                f"variance ratio = {injected_score:.4f} < 2.0"
            )
            continue

        df[outcome_col] = trial_outcome

        effects = {
            "unreliable_feature": str(chosen_feature),
            "noise_multiplier": float(noise_multiplier),
            "heteroskedasticity_score": float(injected_score),
            "id_no_cols": sorted(id_no_cols),
        }

        return df, {
            "type": "dq_unreliable_feature",
            "params": params,
            "effects": effects,
        }

    raise InjectionRejected(
        f"No candidate survived validation. Last error: {last_error}"
    )


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    feature = effects["unreliable_feature"]
    id_no_cols = set(effects.get("id_no_cols", []) or []) | id_like_cols(df)
    id_no_cols.discard(target_col)
    feature_vals = coerce_numeric(df[feature])
    outcome_vals = coerce_numeric(df[target_col])
    checks = []

    if feature in id_no_cols:
        checks.append(CheckDetail(
            name="unreliable_feature_not_identifier", passed=False,
            metric=0.0, threshold=1.0,
            detail=f"Injected feature '{feature}' is identifier-like",
        ))
        return ValidationResult(passed=False, checks=checks)

    injected_score = heteroskedasticity_score(feature_vals, outcome_vals)

    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {target_col, feature},
        min_unique=5,
    )

    for other_col in numeric_cols:
        other_score = heteroskedasticity_score(df[other_col].astype(float), outcome_vals)
        if other_score >= injected_score:
            checks.append(CheckDetail(
                name="dominance_heteroskedasticity", passed=False,
                metric=other_score, threshold=injected_score,
                detail=f"Feature '{other_col}' has score ({other_score:.4f}) >= "
                       f"injected '{feature}' ({injected_score:.4f})",
            ))
            return ValidationResult(passed=False, checks=checks)

    checks.append(CheckDetail(
        name="dominance_heteroskedasticity", passed=True,
        metric=injected_score, threshold=0.0,
        detail=f"Injected score {injected_score:.4f}",
    ))
    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the feature with the most unreliable relationship with the outcome."""
    return effects["unreliable_feature"]


PHENOMENON = Phenomenon(
    name="dq_unreliable_feature",
    inject=inject,
    validate=validate,
    compute_answers={
        "dq_unreliable_feature_v0": compute_answer_v0,
    },
    summary_fields=("id_no",),
)
