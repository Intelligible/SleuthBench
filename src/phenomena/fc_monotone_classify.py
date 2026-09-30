"""Monotone-vs-non-monotone classification phenomenon.

Adds a V-shaped target effect to a previously monotone numeric feature. The
ground-truth answer is the name of the injected feature. Injection acceptance
requires post-injection ``|rho| < 0.5`` and a direction reversal in the binned
target means; validation requires a strictly lower absolute binned Spearman
correlation than every peer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import (
    coerce_numeric,
    decimal_places,
    eligible_numeric_columns,
    has_direction_reversal,
    id_like_cols,
    safe_std,
    spearman_bin_rho,
)

from ._base import CheckDetail, InjectionRejected, Phenomenon, ValidationResult


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Plant a V-shape in one currently monotone numeric feature.

    Candidates have pre-injection binned Spearman ``|rho| >= 0.6`` and are
    tried in seed-shuffled order. Acceptance requires post-injection
    ``|rho| < 0.5`` and a direction reversal in the binned target means.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    effect_strength = float(params.get("effect_strength", "3.0"))
    id_no_cols = (
        set(params.get("id_no_cols", []) or [])
        | id_like_cols(df)
    )
    id_no_cols.discard(outcome_col)

    numeric_cols = []
    for c in eligible_numeric_columns(
        df,
        exclude=id_no_cols | {outcome_col},
        min_unique=10,
    ):
        if df[c].nunique() == len(df) and df[c].is_monotonic_increasing:
            continue
        numeric_cols.append(c)

    if len(numeric_cols) < 3:
        raise InjectionRejected(
            f"Need at least 3 eligible numeric features, found {len(numeric_cols)}"
        )

    outcome_vals = df[outcome_col].astype(float)

    pre_rhos = {}
    for col in numeric_cols:
        pre_rhos[col] = spearman_bin_rho(df[col].astype(float), outcome_vals)

    monotone_candidates = [
        col for col in numeric_cols if abs(pre_rhos[col]) >= 0.6
    ]
    if len(monotone_candidates) < 1:
        raise InjectionRejected(
            "No features with |rho| >= 0.6 to inject into; "
            "dataset has no clearly monotone features"
        )

    order = rng.permutation(len(monotone_candidates))
    last_error = None

    for idx in order:
        chosen_feature = monotone_candidates[idx]
        feature_vals = df[chosen_feature].astype(float)

        midpoint = float(np.percentile(feature_vals.dropna(), 50))
        feature_dp = decimal_places(df[chosen_feature])
        midpoint = round(midpoint, feature_dp)

        feat_min = float(feature_vals.min())
        feat_max = float(feature_vals.max())
        half_range = (feat_max - feat_min) / 2.0
        if half_range == 0:
            last_error = f"Feature '{chosen_feature}' has zero range"
            continue

        outcome_std = safe_std(outcome_vals)
        amplitude = effect_strength * outcome_std

        norm_dist = (feature_vals - midpoint) / half_range
        valley = 1.0 - np.abs(norm_dist)
        trial_outcome = outcome_vals - amplitude * valley

        outcome_dp = decimal_places(outcome_vals)
        trial_outcome = trial_outcome.round(outcome_dp)

        injected_rho = spearman_bin_rho(feature_vals, trial_outcome)
        if abs(injected_rho) >= 0.5:
            last_error = (
                f"Injected feature '{chosen_feature}' is still monotone: "
                f"|rho| = {abs(injected_rho):.4f} >= 0.5"
            )
            continue

        if not has_direction_reversal(feature_vals, trial_outcome):
            last_error = (
                f"No direction reversal detected in bin means for '{chosen_feature}'"
            )
            continue

        df[outcome_col] = trial_outcome

        effects = {
            "injected_feature": str(chosen_feature),
            "midpoint": float(midpoint),
            "amplitude": float(amplitude),
            "half_range": float(half_range),
            "id_no_cols": sorted(id_no_cols),
        }

        return df, {
            "type": "fc_monotone_classify",
            "params": params,
            "effects": effects,
        }

    raise InjectionRejected(
        f"No monotone candidate survived validation. "
        f"Last error: {last_error}"
    )


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    feature = effects["injected_feature"]
    id_no_cols = (
        set(effects.get("id_no_cols", []) or [])
        | id_like_cols(df)
    )
    id_no_cols.discard(target_col)
    outcome_vals = coerce_numeric(df[target_col])
    feature_vals = coerce_numeric(df[feature])
    checks = []

    if feature in id_no_cols:
        checks.append(CheckDetail(
            name="injected_feature_not_identifier", passed=False,
            metric=0.0, threshold=1.0,
            detail=f"Injected feature '{feature}' is identifier-like",
        ))
        return ValidationResult(passed=False, checks=checks)

    injected_rho = abs(spearman_bin_rho(feature_vals, outcome_vals))

    numeric_cols = []
    for c in eligible_numeric_columns(
        df,
        exclude=id_no_cols | {target_col, feature},
        min_unique=10,
    ):
        if df[c].nunique() == len(df) and df[c].is_monotonic_increasing:
            continue
        numeric_cols.append(c)

    for col in numeric_cols:
        other_rho = abs(spearman_bin_rho(df[col].astype(float), outcome_vals))
        if other_rho <= injected_rho:
            checks.append(CheckDetail(
                name="dominance_lowest_rho", passed=False,
                metric=other_rho, threshold=injected_rho,
                detail=f"Feature '{col}' has |rho| = {other_rho:.4f} <= "
                       f"injected '{feature}' |rho| = {injected_rho:.4f}",
            ))
            return ValidationResult(passed=False, checks=checks)

    checks.append(CheckDetail(
        name="dominance_lowest_rho", passed=True,
        metric=injected_rho, threshold=0.0,
        detail=f"Injected |rho| = {injected_rho:.4f}",
    ))
    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the feature with the most non-monotone relationship.

    Answer format: "feature_name"
    """
    return effects["injected_feature"]


PHENOMENON = Phenomenon(
    name="fc_monotone_classify",
    inject=inject,
    validate=validate,
    compute_answers={
        "fc_monotone_classify_v0": compute_answer_v0,
    },
    summary_fields=("id_no",),
)
