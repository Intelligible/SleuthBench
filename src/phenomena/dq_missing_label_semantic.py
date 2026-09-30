"""Semantic missing-label phenomenon.

Adds a missing-like sentinel to one clean categorical feature. The
ground-truth answer is the planted label. Validation requires it to be the
only missing-like category and to appear in at least
``max(min_inject, ceil(0.05 * n))`` rows; that floor is fixed and does not
follow ``inject_fraction``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import clean_categorical_features
from shared.missing_labels import DEFAULT_MISSING_LABELS, is_missing_like

from ._base import (
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
    last_failed_detail,
    row_ids,
)


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Add one sentinel to an eligible categorical feature.

    Source labels must contain no missing-like value and meet the configured
    cardinality bounds. ``n_inject`` is the larger of ``min_inject`` and the
    configured row fraction; the candidate must pass semantic validation.
    """
    df = df.copy()
    categorical_cols = list(params.get("categorical_cols", []) or [])
    inject_fraction = float(params.get("inject_fraction", 0.05))
    min_inject = int(params.get("min_inject", 10))
    min_cat = int(params.get("min_categories", 2))
    max_cat = int(params.get("max_categories", 30))

    n = len(df)
    n_inject = max(min_inject, math.ceil(inject_fraction * n))
    n_inject = min(n_inject, n - 1)  # keep at least one original label
    if n_inject < min_inject:
        raise InjectionRejected(f"dataset too small to plant {min_inject} missing labels (n={n})")
    min_count = max(min_inject, math.ceil(0.05 * n))

    candidates = clean_categorical_features(
        df,
        categorical_cols,
        min_cat,
        max_cat,
        exclude_missing_like=True,
    )
    if not candidates:
        raise InjectionRejected("no clean categorical feature (2..30 labels, no missing-like values)")

    order = rng.permutation(len(candidates))
    last_error = "no attempts"
    for ci in order:
        feature = candidates[ci]
        label = str(DEFAULT_MISSING_LABELS[int(rng.integers(0, len(DEFAULT_MISSING_LABELS)))])
        # Feature has no missing-like labels, so `label` is guaranteed novel.
        positions = rng.choice(n, size=n_inject, replace=False)

        trial = df.copy()
        col = trial[feature].astype(object)
        col.iloc[positions] = label
        trial[feature] = col

        effects = {
            "feature": feature,
            "injected_label": label,
            "n_injected_rows": int(n_inject),
            "row_ids": row_ids(df, positions),
            "min_count": int(min_count),
            "missing_label_lexicon_match": True,
        }
        result = validate(trial, effects, params.get("outcome_col", ""))
        if result.passed:
            return trial, {
                "type": "dq_missing_label_semantic",
                "params": params,
                "effects": effects,
            }
        last_error = last_failed_detail(result)

    raise InjectionRejected(f"No candidate feature survived validation. Last error: {last_error}")


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    feature = effects["feature"]
    label = str(effects["injected_label"])
    min_count = int(effects.get("min_count", 10))
    checks: list[CheckDetail] = []

    if feature not in df.columns:
        checks.append(CheckDetail(
            name="feature_present", passed=False, metric=0.0, threshold=1.0,
            detail=f"Feature '{feature}' missing from the table",
        ))
        return ValidationResult(passed=False, checks=checks)

    values = df[feature]
    uniques = values.unique().tolist()

    present = label in {str(v) for v in uniques}
    checks.append(CheckDetail(
        name="label_present", passed=bool(present), metric=1.0 if present else 0.0, threshold=1.0,
        detail=f"Label '{label}' {'found' if present else 'not found'} in '{feature}'",
    ))
    if not present:
        return ValidationResult(passed=False, checks=checks)

    lex_ok = is_missing_like(label)
    checks.append(CheckDetail(
        name="lexicon_match", passed=bool(lex_ok), metric=1.0 if lex_ok else 0.0, threshold=1.0,
        detail=f"Label '{label}' {'is' if lex_ok else 'is not'} missing-like",
    ))
    if not lex_ok:
        return ValidationResult(passed=False, checks=checks)

    missing_like = sorted({str(v) for v in uniques if is_missing_like(v)})
    unique_ok = missing_like == [label]
    checks.append(CheckDetail(
        name="unique_missing_label", passed=bool(unique_ok),
        metric=float(len(missing_like)), threshold=1.0,
        detail=f"Missing-like categories in '{feature}' = {missing_like} (expected ['{label}'])",
    ))
    if not unique_ok:
        return ValidationResult(passed=False, checks=checks)

    count = int((values.astype(str) == label).sum())
    count_ok = count >= min_count
    checks.append(CheckDetail(
        name="min_count", passed=bool(count_ok), metric=float(count), threshold=float(min_count),
        detail=f"'{label}' appears {count} times (require >= {min_count})",
    ))
    if not count_ok:
        return ValidationResult(passed=False, checks=checks)

    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Gold answer: the planted missing-like category label."""
    return effects["injected_label"]


PHENOMENON = Phenomenon(
    name="dq_missing_label_semantic",
    inject=inject,
    validate=validate,
    compute_answers={
        "dq_missing_label_semantic_v0": compute_answer_v0,
    },
    summary_fields=("categorical",),
)
