"""Target-dependent categorical-bucket phenomenon.

Inspired by datasets where missing feature values are represented as ordinary
categories, this injector anonymises one categorical feature and adds a neutral
bucket whose membership depends on the target. Because every label is neutral,
the bucket can be identified only from its distinctive target distribution.

The ground-truth answer is the bucket's neutral label. Validation requires it
to be a significant, correctly signed, dominant category-level target outlier.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import (
    clean_categorical_features,
    coerce_numeric,
    two_proportion_z_test,
    welch_t_test,
)

from ._base import (
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
    last_failed_detail,
    row_ids,
)


def _is_binary(y: np.ndarray) -> bool:
    return int(np.unique(y[np.isfinite(y)]).size) == 2


def _group_stat(group: np.ndarray, rest: np.ndarray, is_binary: bool) -> tuple[float, float]:
    """Target-gap test statistic for ``group`` vs ``rest``; sign follows the mean gap.

    Welch two-sample t-test for numeric targets, two-proportion z-test for
    two-valued targets, which must be coded 0/1.  Returns ``(stat, p_value)``.
    """
    if is_binary:
        return two_proportion_z_test(group, rest)
    return welch_t_test(group, rest)


def _neutral_labels(k: int) -> list[str]:
    """Excel-style neutral labels: A, B, ..., Z, AA, AB, ... (k of them)."""
    labels: list[str] = []
    i = 0
    while len(labels) < k:
        n, s = i, ""
        while True:
            s = chr(ord("A") + n % 26) + s
            n = n // 26 - 1
            if n < 0:
                break
        labels.append(s)
        i += 1
    return labels


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Anonymise a categorical feature and add a target-selected neutral bucket.

    Iterates over the clean categorical features (no NaN, cardinality within
    the configured bounds, default 2..30) in a seed-shuffled order and, for
    each feature, tries both directions in a seed-chosen order. Every attempt
    renames each level to a neutral pseudo-label and pulls the
    ``n_inject = max(min_inject, ceil(q * n))`` rows with the largest ("high")
    or smallest ("low") finite target values into one extra neutral "withheld"
    bucket label.  Neutral labels are assigned in a shuffled order so the
    bucket's letter is not predictable.  The target is untouched.  The first
    (feature, direction) pair that :func:`validate` confirms to be a
    significant, correctly-signed, dominant category-level target outlier is
    accepted.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    categorical_cols = list(params.get("categorical_cols", []) or [])
    q = float(params.get("q", 0.10))
    min_inject = int(params.get("min_inject", 10))
    min_cat = int(params.get("min_categories", 2))
    max_cat = int(params.get("max_categories", 30))
    p_thresh = float(params.get("p_thresh", 0.01))
    margin = float(params.get("margin", 1.0))
    min_gap_factor = float(params.get("min_gap_factor", 0.5))

    n = len(df)
    n_inject = max(min_inject, math.ceil(q * n))

    y_series = coerce_numeric(df[outcome_col])
    y = y_series.to_numpy(dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(y))
    if len(finite_idx) < n_inject + 2:
        raise InjectionRejected(f"too few finite-target rows to form a bucket of {n_inject}")
    is_binary = _is_binary(y)
    sigma_y = float(np.std(y[finite_idx], ddof=1))

    features = clean_categorical_features(
        df,
        categorical_cols,
        min_cat,
        max_cat,
        exclude_missing_like=False,
    )
    if not features:
        raise InjectionRejected("no clean categorical feature (2..30 labels, no NaN)")

    # Deterministic target ordering (stable ties by original row position).
    ordered = finite_idx[np.argsort(y[finite_idx], kind="stable")]

    feat_order = rng.permutation(len(features))
    directions = ["high", "low"] if rng.integers(0, 2) == 0 else ["low", "high"]
    last_error = "no attempts"
    for fi in feat_order:
        feature = features[fi]
        orig = df[feature].astype(str).to_numpy()
        for direction in directions:
            sel = ordered[-n_inject:] if direction == "high" else ordered[:n_inject]
            bucket_mask = np.zeros(n, dtype=bool)
            bucket_mask[sel] = True

            # Groups = original levels still present outside the bucket, plus the
            # bucket. Assign neutral labels in a shuffled order so the bucket's
            # label carries no positional hint.
            present = sorted(set(orig[~bucket_mask].tolist()))
            groups = present + ["__bucket__"]
            neutral = _neutral_labels(len(groups))
            perm = rng.permutation(len(groups))
            assign = {g: neutral[int(perm[j])] for j, g in enumerate(groups)}
            injected_label = assign["__bucket__"]

            new_col = pd.Series(orig, index=df.index).map(
                {g: assign[g] for g in present}
            ).astype(object)
            new_col[bucket_mask] = injected_label

            trial = df.copy()
            trial[feature] = new_col

            lab_mask = (trial[feature].astype(str) == injected_label).to_numpy()
            stat, p_value = _group_stat(y[lab_mask], y[~lab_mask], is_binary)
            effects = {
                "feature": feature,
                "target": outcome_col,
                "injected_label": injected_label,
                "direction": direction,
                "q": float(q),
                "n_injected_rows": int(n_inject),
                "row_ids": row_ids(df, sel),
                "label_mapping": {
                    **{str(g): assign[g] for g in present},
                    "__withheld_bucket__": injected_label,
                },
                "mean_target_injected_label": float(np.nanmean(y[lab_mask])),
                "mean_target_rest": float(np.nanmean(y[~lab_mask])),
                "target_gap": float(np.nanmean(y[lab_mask]) - np.nanmean(y[~lab_mask])),
                "t_stat": float(stat),
                "p_value": float(p_value),
                "p_thresh": float(p_thresh),
                "margin": float(margin),
                "min_gap_factor": float(min_gap_factor),
                "target_sd": float(sigma_y),
                "is_binary": bool(is_binary),
            }
            result = validate(trial, effects, outcome_col)
            if result.passed:
                return trial, {
                    "type": "dq_missing_label_target_dependent",
                    "params": params,
                    "effects": effects,
                }
            last_error = last_failed_detail(result)

    raise InjectionRejected(f"No target-tail bucket passed validation. Last error: {last_error}")


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    feature = effects["feature"]
    label = str(effects["injected_label"])
    direction = str(effects["direction"])
    n_injected = int(effects.get("n_injected_rows", 0))
    p_thresh = float(effects.get("p_thresh", 0.01))
    margin = float(effects.get("margin", 1.0))
    min_gap_factor = float(effects.get("min_gap_factor", 0.5))
    checks: list[CheckDetail] = []

    if feature not in df.columns:
        checks.append(CheckDetail(
            name="feature_present", passed=False, metric=0.0, threshold=1.0,
            detail=f"Feature '{feature}' missing from the table",
        ))
        return ValidationResult(passed=False, checks=checks)

    labels = df[feature].astype(str)
    uniques = labels.unique().tolist()

    if label not in set(uniques):
        checks.append(CheckDetail(
            name="label_present", passed=False, metric=0.0, threshold=1.0,
            detail=f"Bucket label '{label}' not found in '{feature}'",
        ))
        return ValidationResult(passed=False, checks=checks)
    lab_mask = (labels == label).to_numpy()
    count_ok = int(lab_mask.sum()) == n_injected
    checks.append(CheckDetail(
        name="bucket_intact", passed=bool(count_ok), metric=float(lab_mask.sum()),
        threshold=float(n_injected),
        detail=f"Bucket '{label}' has {int(lab_mask.sum())} rows (expected {n_injected})",
    ))
    if not count_ok:
        return ValidationResult(passed=False, checks=checks)

    y = coerce_numeric(df[target_col]).to_numpy(dtype=float)
    is_binary = _is_binary(y)
    mean_label = float(np.nanmean(y[lab_mask]))
    mean_rest = float(np.nanmean(y[~lab_mask]))
    stat, p_value = _group_stat(y[lab_mask], y[~lab_mask], is_binary)

    sig_ok = p_value < p_thresh
    checks.append(CheckDetail(
        name="significant", passed=bool(sig_ok), metric=float(p_value), threshold=float(p_thresh),
        detail=f"Target-gap p for '{label}' = {p_value:.2e} (require < {p_thresh})",
    ))
    if not sig_ok:
        return ValidationResult(passed=False, checks=checks)

    dir_ok = (mean_label > mean_rest) if direction == "high" else (mean_label < mean_rest)
    checks.append(CheckDetail(
        name="direction_match", passed=bool(dir_ok), metric=float(mean_label - mean_rest),
        threshold=0.0,
        detail=f"mean('{label}')={mean_label:.4g} vs rest={mean_rest:.4g}; expected '{direction}'",
    ))
    if not dir_ok:
        return ValidationResult(passed=False, checks=checks)

    abs_stat = {
        category: (
            abs(stat)
            if category == label
            else abs(_group_stat(
                y[(labels == category).to_numpy()],
                y[(labels != category).to_numpy()],
                is_binary,
            )[0])
        )
        for category in uniques
    }
    s_star = abs_stat[label]
    others = [v for k, v in abs_stat.items() if k != label]
    second = max(others) if others else 0.0
    dom_ok = (s_star >= max(abs_stat.values())) and (s_star >= second + margin)
    checks.append(CheckDetail(
        name="dominance_margin", passed=bool(dom_ok), metric=float(s_star - second),
        threshold=float(margin),
        detail=f"|stat| for '{label}' = {s_star:.4f}; second-best = {second:.4f} "
               f"(require gap >= {margin})",
    ))
    if not dom_ok:
        return ValidationResult(passed=False, checks=checks)

    finite = np.isfinite(y)
    sd = float(np.std(y[finite], ddof=1)) if int(finite.sum()) > 1 else 0.0
    if not is_binary and sd > 0:
        gap_ok = abs(mean_label - mean_rest) >= min_gap_factor * sd
        checks.append(CheckDetail(
            name="min_absolute_gap", passed=bool(gap_ok), metric=float(abs(mean_label - mean_rest)),
            threshold=float(min_gap_factor * sd),
            detail=f"|mean gap| = {abs(mean_label - mean_rest):.4g} "
                   f"(require >= {min_gap_factor} * sd = {min_gap_factor * sd:.4g})",
        ))
        if not gap_ok:
            return ValidationResult(passed=False, checks=checks)

    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Gold answer: the neutral bucket label whose membership is target-dependent."""
    return effects["injected_label"]


PHENOMENON = Phenomenon(
    name="dq_missing_label_target_dependent",
    inject=inject,
    validate=validate,
    compute_answers={
        "dq_missing_label_target_dependent_v0": compute_answer_v0,
    },
    summary_fields=("categorical",),
)
