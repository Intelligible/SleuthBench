"""Threshold / plateau phenomenon.

Adds a mean-centred hockey-stick effect that rises to a threshold and then
plateaus. The ground-truth answer is the feature and threshold. Validation
requires the injected feature to have the dominant eligible plateau score.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import (
    coerce_numeric,
    decimal_places,
    eligible_numeric_columns,
    id_like_cols,
    plateau_score,
    safe_std,
)

from ._base import CheckDetail, InjectionRejected, Phenomenon, ValidationResult


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Plant a threshold / plateau shape in one eligible numeric feature.

    Requires at least two non-identifier features with ten unique values.
    Eligible candidates are seed-shuffled and tried in that order. Thresholds
    are drawn from the 30th-60th percentiles, and a candidate must reach a
    post-injection plateau score of at least 3.0.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    effect_strength = float(params.get("effect_strength", "2.0"))
    id_no_cols = (
        set(params.get("id_no_cols", []) or [])
        | id_like_cols(df)
    )
    id_no_cols.discard(outcome_col)

    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {outcome_col},
        min_unique=10,
    )

    if len(numeric_cols) < 2:
        raise InjectionRejected(
            f"Need at least 2 eligible numeric features, found {len(numeric_cols)}"
        )

    outcome_vals = df[outcome_col].astype(float)
    outcome_std = safe_std(outcome_vals)
    amplitude = effect_strength * outcome_std
    outcome_dp = decimal_places(outcome_vals)

    pre_scores = {
        c: plateau_score(df[c].astype(float), outcome_vals)
        for c in numeric_cols
    }
    candidates = sorted(pre_scores, key=pre_scores.get)
    order = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in order]

    last_error = None

    for chosen_feature in candidates:
        feature_vals = df[chosen_feature].astype(float)

        pct = rng.uniform(0.30, 0.60)
        threshold_raw = float(np.percentile(feature_vals.dropna(), pct * 100))
        feature_dp = decimal_places(df[chosen_feature])
        threshold = round(threshold_raw, feature_dp)

        feat_min = float(feature_vals.min())
        feat_max = float(feature_vals.max())
        feat_range = feat_max - feat_min
        if feat_range == 0:
            last_error = f"Feature '{chosen_feature}' has zero range"
            continue

        normalised = (feature_vals - feat_min) / feat_range
        threshold_norm = (threshold - feat_min) / feat_range

        effect = np.minimum(normalised, threshold_norm)
        eff_min = float(effect.min())
        eff_max = float(effect.max())
        eff_range = eff_max - eff_min
        if eff_range > 0:
            effect = (effect - eff_min) / eff_range
        effect = effect - effect.mean()
        trial_outcome = (outcome_vals + amplitude * effect).round(outcome_dp)

        injected_score = plateau_score(feature_vals, trial_outcome)
        if injected_score < 3.0:
            last_error = (
                f"Plateau shape too weak for '{chosen_feature}': "
                f"score = {injected_score:.4f} < 3.0"
            )
            continue

        df[outcome_col] = trial_outcome

        effects = {
            "threshold_feature": str(chosen_feature),
            "threshold_value": float(threshold),
            "amplitude": float(amplitude),
            "id_no_cols": sorted(id_no_cols),
        }

        return df, {
            "type": "fc_threshold_value",
            "params": params,
            "effects": effects,
        }

    raise InjectionRejected(
        f"No candidate survived validation. Last error: {last_error}"
    )


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    feature = effects["threshold_feature"]
    checks = []
    id_no_cols = (
        set(effects.get("id_no_cols", []) or [])
        | id_like_cols(df)
    )
    id_no_cols.discard(target_col)

    feature_is_eligible = feature not in id_no_cols
    checks.append(CheckDetail(
        name="threshold_feature_not_identifier",
        passed=feature_is_eligible,
        metric=1.0 if feature_is_eligible else 0.0,
        threshold=1.0,
        detail=(
            f"Threshold feature '{feature}' is not identifier-like"
            if feature_is_eligible
            else f"Threshold feature '{feature}' is identifier-like"
        ),
    ))
    if not feature_is_eligible:
        return ValidationResult(passed=False, checks=checks)

    feature_vals = coerce_numeric(df[feature])
    outcome_vals = coerce_numeric(df[target_col])

    try:
        bins = pd.qcut(feature_vals, q=6, duplicates="drop")
    except ValueError:
        checks.append(CheckDetail(
            name="plateau_shape", passed=False, metric=0.0, threshold=4.0,
            detail=f"Cannot bin feature '{feature}' into quantiles",
        ))
        return ValidationResult(passed=False, checks=checks)

    bin_means = outcome_vals.groupby(bins, observed=True).mean().sort_index()
    if len(bin_means) < 4:
        checks.append(CheckDetail(
            name="plateau_shape", passed=False,
            metric=len(bin_means), threshold=4.0,
            detail=f"Feature '{feature}' produced fewer than 4 bins",
        ))
        return ValidationResult(passed=False, checks=checks)

    mid = len(bin_means) // 2
    first_range = abs(float(bin_means.iloc[mid - 1] - bin_means.iloc[0]))
    second_range = abs(float(bin_means.iloc[-1] - bin_means.iloc[mid]))
    shape_ok = first_range > second_range
    checks.append(CheckDetail(
        name="plateau_shape", passed=shape_ok,
        metric=first_range, threshold=second_range,
        detail=f"First half range = {first_range:.4f}, second half range = {second_range:.4f}",
    ))
    if not shape_ok:
        return ValidationResult(passed=False, checks=checks)

    injected_score = plateau_score(feature_vals, outcome_vals)
    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {target_col, feature},
        min_unique=10,
    )

    for other_col in numeric_cols:
        other_score = plateau_score(df[other_col].astype(float), outcome_vals)
        if other_score >= injected_score:
            checks.append(CheckDetail(
                name="dominance_plateau_score", passed=False,
                metric=other_score, threshold=injected_score,
                detail=f"Feature '{other_col}' has plateau score ({other_score:.4f}) >= "
                       f"injected '{feature}' ({injected_score:.4f})",
            ))
            return ValidationResult(passed=False, checks=checks)

    checks.append(CheckDetail(
        name="dominance_plateau_score", passed=True,
        metric=injected_score, threshold=0.0,
        detail=f"Injected score {injected_score:.4f}",
    ))
    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the feature with the threshold effect and its plateau value.

    Answer format: "feature_name, threshold_value"
    """
    return f"{effects['threshold_feature']}, {effects['threshold_value']}"


PHENOMENON = Phenomenon(
    name="fc_threshold_value",
    inject=inject,
    validate=validate,
    compute_answers={
        "fc_threshold_value_v0": compute_answer_v0,
    },
    summary_fields=("id_no",),
)
