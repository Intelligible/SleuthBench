"""Categorical target-outlier phenomenon.

Shifts the target for one populous category. The ground-truth answer is that
category. Validation requires its target gap to be significant, strongest,
and clearly dominant among categories of the selected feature.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import (
    clean_categorical_features,
    coerce_numeric,
    decimal_places,
    welch_t_test,
)

from ._base import (
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
    last_failed_detail,
)


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Shift the target for one eligible categorical group.

    Clean features must have 2-30 categories, and the selected category must
    contain at least ``max(10, ceil(0.05 * n))`` rows. Candidate groups are
    seed-shuffled, both shift directions are tried, and validation must pass.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    categorical_cols = list(params.get("categorical_cols", []) or [])
    shift_strength = float(params.get("shift_strength", 2.0))
    min_cat = int(params.get("min_categories", 2))
    max_cat = int(params.get("max_categories", 30))
    p_thresh = float(params.get("p_thresh", 0.01))
    margin = float(params.get("margin", 1.0))

    n = len(df)
    min_count = max(10, math.ceil(0.05 * n))

    y_series = coerce_numeric(df[outcome_col])
    y = y_series.to_numpy(dtype=float)
    finite = np.isfinite(y)
    if int(np.unique(y[finite]).size) <= 2:
        raise InjectionRejected("target is binary/degenerate; shift injector needs a numeric target")
    sigma_y = float(np.std(y[finite], ddof=1))
    if sigma_y <= 0 or not np.isfinite(sigma_y):
        raise InjectionRejected("target has zero / invalid spread")
    y_dp = decimal_places(y_series)

    features = clean_categorical_features(
        df,
        categorical_cols,
        min_cat,
        max_cat,
        exclude_missing_like=True,
    )
    if not features:
        raise InjectionRejected("no clean categorical feature (2..30 labels, no missing-like values)")

    # Candidate (feature, category) pairs with enough rows.
    pairs: list[tuple[str, str]] = []
    for feature in features:
        vc = df[feature].astype(str).value_counts()
        for cat, cnt in vc.items():
            if int(cnt) >= min_count and (n - int(cnt)) >= 2:
                pairs.append((feature, str(cat)))
    if not pairs:
        raise InjectionRejected(f"no category has >= {min_count} rows in any clean feature")

    order = rng.permutation(len(pairs))
    directions = [1, -1] if rng.integers(0, 2) == 0 else [-1, 1]
    last_error = "no attempts"
    for pi in order:
        feature, category = pairs[pi]
        cat_mask = (df[feature].astype(str) == category).to_numpy()
        mean_before = float(np.nanmean(y[cat_mask]))
        for direction in directions:
            new_y = y.copy()
            new_y[cat_mask] = new_y[cat_mask] + direction * shift_strength * sigma_y

            t_stat, p_value = welch_t_test(new_y[cat_mask], new_y[~cat_mask])
            effects = {
                "feature": feature,
                "category": category,
                "direction": int(direction),
                "shift_strength": float(shift_strength),
                "n_category_rows": int(cat_mask.sum()),
                "target_mean_category_before": mean_before,
                "target_mean_category_after": float(np.nanmean(new_y[cat_mask])),
                "target_mean_rest_after": float(np.nanmean(new_y[~cat_mask])),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "p_thresh": float(p_thresh),
                "margin": float(margin),
                "min_count": int(min_count),
            }

            trial = df.copy()
            trial[outcome_col] = pd.Series(new_y, index=df.index).round(y_dp)
            result = validate(trial, effects, outcome_col)
            if result.passed:
                return trial, {
                    "type": "dq_categorical_target_outlier",
                    "params": params,
                    "effects": effects,
                }
            last_error = last_failed_detail(result)

    raise InjectionRejected(f"No shifted category was a dominant outlier. Last error: {last_error}")


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    feature = effects["feature"]
    category = str(effects["category"])
    p_thresh = float(effects.get("p_thresh", 0.01))
    margin = float(effects.get("margin", 1.0))
    checks: list[CheckDetail] = []

    if feature not in df.columns:
        checks.append(CheckDetail(
            name="feature_present", passed=False, metric=0.0, threshold=1.0,
            detail=f"Feature '{feature}' missing from the table",
        ))
        return ValidationResult(passed=False, checks=checks)

    y = coerce_numeric(df[target_col]).to_numpy(dtype=float)
    labels = df[feature].astype(str)
    if category not in set(labels.unique()):
        checks.append(CheckDetail(
            name="category_present", passed=False, metric=0.0, threshold=1.0,
            detail=f"Category '{category}' not found in '{feature}'",
        ))
        return ValidationResult(passed=False, checks=checks)

    # Per-category |t| against the rest; the planted category must top the list.
    abs_t: dict[str, float] = {}
    p_by_cat: dict[str, float] = {}
    for c in labels.unique():
        mask = (labels == c).to_numpy()
        t, p = welch_t_test(y[mask], y[~mask])
        abs_t[c] = abs(t)
        p_by_cat[c] = p

    t_star = abs_t[category]
    p_star = p_by_cat[category]
    others = [v for k, v in abs_t.items() if k != category]
    second = max(others) if others else 0.0

    is_max = t_star >= max(abs_t.values())
    checks.append(CheckDetail(
        name="strongest_outlier", passed=bool(is_max), metric=float(t_star),
        threshold=float(second),
        detail=f"|t| for '{category}' = {t_star:.4f}; strongest other = {second:.4f}",
    ))
    if not is_max:
        return ValidationResult(passed=False, checks=checks)

    sig_ok = p_star < p_thresh
    checks.append(CheckDetail(
        name="significant", passed=bool(sig_ok), metric=float(p_star), threshold=float(p_thresh),
        detail=f"Welch p for '{category}' = {p_star:.2e} (require < {p_thresh})",
    ))
    if not sig_ok:
        return ValidationResult(passed=False, checks=checks)

    dom_ok = t_star >= second + margin
    checks.append(CheckDetail(
        name="dominance_margin", passed=bool(dom_ok), metric=float(t_star - second),
        threshold=float(margin),
        detail=f"|t| gap over second-best = {t_star - second:.4f} (require >= {margin})",
    ))
    if not dom_ok:
        return ValidationResult(passed=False, checks=checks)

    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Gold answer: the category whose target values were shifted to stand out."""
    return effects["category"]


PHENOMENON = Phenomenon(
    name="dq_categorical_target_outlier",
    inject=inject,
    validate=validate,
    compute_answers={
        "dq_categorical_target_outlier_v0": compute_answer_v0,
    },
    summary_fields=("categorical",),
)
