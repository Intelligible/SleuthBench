"""Underpowered two-group comparison phenomenon.

Adds a rare group while leaving the target unchanged. The ground-truth answer
is ``"not enough evidence"``. Validation requires the rare group's Welch
confidence interval to contain zero and be wide relative to the target spread.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import coerce_numeric, welch_mean_difference_ci

from ._base import (
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
    last_failed_detail,
)


def _target_values(df: pd.DataFrame, target_col: str) -> np.ndarray:
    y = coerce_numeric(df[target_col]).to_numpy(dtype=float)
    return y


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Add a rare-group categorical column for an underpowered comparison.

    Finite-target rows are sampled into group ``B`` within the configured size
    caps. A split is accepted only if validation confirms rarity and a wide
    Welch interval containing zero.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    group_col = str(params.get("group_col", "subgroup"))
    label_a = str(params.get("label_a", "A"))
    label_b = str(params.get("label_b", "B"))
    conf = float(params.get("conf", 0.95))
    max_b_frac = float(params.get("max_b_frac", 0.05))
    min_width_ratio = float(params.get("min_width_ratio", 1.0))
    min_b = int(params.get("min_b", 3))
    max_b = int(params.get("max_b", 8))

    if group_col in df.columns:
        raise InjectionRejected(f"group column '{group_col}' already exists in the dataset")
    if label_a == label_b:
        raise ValueError("label_a and label_b must differ")

    n = len(df)
    y = _target_values(df, outcome_col)
    finite = np.isfinite(y)
    if int(finite.sum()) < 2:
        raise InjectionRejected("target has fewer than 2 finite values")
    sy = float(np.std(y[finite], ddof=1))
    if sy <= 0 or not np.isfinite(sy):
        raise InjectionRejected("target has zero / invalid spread")

    # n_B: small in absolute terms (max_b) and a small fraction (max_b_frac).
    n_b = min(max_b, max(min_b, int(round(max_b_frac * n))))
    n_b = min(n_b, int(math.floor(max_b_frac * n)) if max_b_frac * n >= min_b else n_b)
    if n_b < min_b or n_b < 2 or n - n_b < 2:
        raise InjectionRejected(
            f"dataset too small for a rare B group (n={n}, n_b={n_b}, min_b={min_b})"
        )
    if n_b / n > max_b_frac + 1e-9:
        raise InjectionRejected(
            f"cannot honour rarity: n_b/n={n_b / n:.3f} > max_b_frac={max_b_frac}"
        )

    # Only finite-target rows are eligible for group B so the CI is well-defined.
    eligible = np.flatnonzero(finite)
    if len(eligible) < n_b + 2:
        raise InjectionRejected("too few finite-target rows to form the comparison")

    last_error = "no attempts"
    for _ in range(40):
        b_pos = rng.choice(eligible, size=n_b, replace=False)
        b_mask = np.zeros(n, dtype=bool)
        b_mask[b_pos] = True

        groups = np.where(b_mask, label_b, label_a)
        trial = df.copy()
        trial[group_col] = groups

        a_vals = y[(~b_mask) & finite]
        b_vals = y[b_mask]
        delta, lo, hi, width = welch_mean_difference_ci(a_vals, b_vals, conf)

        effects = {
            "group_col": group_col,
            "label_a": label_a,
            "label_b": label_b,
            "conf": float(conf),
            "max_b_frac": float(max_b_frac),
            "min_width_ratio": float(min_width_ratio),
            "n_a": int((~b_mask).sum()),
            "n_b": int(n_b),
            "delta": float(delta),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_width": float(width),
            "target_sd": float(sy),
            "width_ratio": float(width / sy) if sy > 0 else 0.0,
        }
        if "row_id" in df.columns:
            effects["b_row_ids"] = sorted(int(r) for r in df["row_id"].to_numpy()[b_mask])

        result = validate(trial, effects, outcome_col)
        if result.passed:
            return trial, {
                "type": "dq_group_evidence_underpowered",
                "params": params,
                "effects": effects,
            }
        last_error = last_failed_detail(result)

    raise InjectionRejected(f"No random split was underpowered enough. Last error: {last_error}")


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    group_col = effects["group_col"]
    label_a = str(effects["label_a"])
    label_b = str(effects["label_b"])
    conf = float(effects["conf"])
    max_b_frac = float(effects["max_b_frac"])
    min_width_ratio = float(effects["min_width_ratio"])
    checks: list[CheckDetail] = []

    if group_col not in df.columns:
        checks.append(CheckDetail(
            name="group_column_present", passed=False, metric=0.0, threshold=1.0,
            detail=f"Group column '{group_col}' missing from the table",
        ))
        return ValidationResult(passed=False, checks=checks)

    g = df[group_col].astype(str)
    levels = set(g.unique())
    two_groups = levels == {label_a, label_b}
    checks.append(CheckDetail(
        name="exactly_two_groups", passed=bool(two_groups), metric=float(len(levels)), threshold=2.0,
        detail=f"Levels in '{group_col}' = {sorted(levels)}, expected {{{label_a}, {label_b}}}",
    ))
    if not two_groups:
        return ValidationResult(passed=False, checks=checks)

    y = _target_values(df, target_col)
    a_mask = (g == label_a).to_numpy() & np.isfinite(y)
    b_mask = (g == label_b).to_numpy() & np.isfinite(y)
    a_vals = y[a_mask]
    b_vals = y[b_mask]
    n = len(df)
    n_a, n_b = len(a_vals), len(b_vals)

    if n_b < 2 or n_a < 2:
        checks.append(CheckDetail(
            name="groups_computable", passed=False, metric=float(min(n_a, n_b)), threshold=2.0,
            detail=f"Need >= 2 finite-target rows per group (n_a={n_a}, n_b={n_b})",
        ))
        return ValidationResult(passed=False, checks=checks)

    frac_b = n_b / n
    rare_ok = frac_b <= max_b_frac + 1e-9
    checks.append(CheckDetail(
        name="group_b_rare", passed=bool(rare_ok), metric=float(frac_b), threshold=float(max_b_frac),
        detail=f"n_B/n = {frac_b:.4f} (n_B={n_b}), require <= {max_b_frac}",
    ))
    if not rare_ok:
        return ValidationResult(passed=False, checks=checks)

    delta, lo, hi, width = welch_mean_difference_ci(a_vals, b_vals, conf)
    straddle_ok = lo <= 0.0 <= hi
    checks.append(CheckDetail(
        name="ci_contains_zero", passed=bool(straddle_ok), metric=float(delta), threshold=0.0,
        detail=f"Delta = {delta:.4f}, {conf:.0%} CI = [{lo:.4f}, {hi:.4f}] "
               f"({'contains' if straddle_ok else 'excludes'} 0)",
    ))
    if not straddle_ok:
        return ValidationResult(passed=False, checks=checks)

    sy = float(np.std(y[np.isfinite(y)], ddof=1))
    if sy <= 0 or not np.isfinite(sy):
        checks.append(CheckDetail(
            name="ci_wide", passed=False, metric=0.0, threshold=min_width_ratio,
            detail="Target spread collapsed to zero",
        ))
        return ValidationResult(passed=False, checks=checks)
    ratio = width / sy
    wide_ok = ratio >= min_width_ratio
    checks.append(CheckDetail(
        name="ci_wide", passed=bool(wide_ok), metric=float(ratio), threshold=float(min_width_ratio),
        detail=f"CI width / sd(Y) = {ratio:.4f}, require >= {min_width_ratio}",
    ))
    if not wide_ok:
        return ValidationResult(passed=False, checks=checks)

    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Gold answer: the comparison is underpowered, so there is not enough evidence."""
    return "not enough evidence"


PHENOMENON = Phenomenon(
    name="dq_group_evidence_underpowered",
    inject=inject,
    validate=validate,
    compute_answers={
        "dq_group_evidence_underpowered_v0": compute_answer_v0,
    },
)
