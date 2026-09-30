"""Interaction-dominant feature pair phenomenon.

Adds a centred XOR-like effect between two numeric features without a net main
effect for either feature. Ground-truth answers identify the strongest pair or
its conditional direction. Injection rejects candidate pairs whose members
have ``|Pearson r| > 0.7`` with the outcome; validation checks XOR
detectability, dominance over every other eligible pair, and, for direction
questions, the reliability of the conditional direction.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from shared.metrics import (
    cohens_d,
    coerce_numeric,
    decimal_places,
    eligible_numeric_columns,
    id_like_cols,
    safe_std,
    xor_signal,
)

from ._base import CheckDetail, InjectionRejected, Phenomenon, ValidationResult


_DIRECTION_MIN_ROWS = 20
_DIRECTION_MIN_ABS_R = 0.2
_DIRECTION_MAX_P = 0.05


def _xor_binary(col_a: pd.Series, col_b: pd.Series) -> pd.Series:
    """Return the uncentred 0/1 XOR grouping used for effect comparisons."""
    a_vals = coerce_numeric(col_a)
    b_vals = coerce_numeric(col_b)
    bin_a = (a_vals > a_vals.median()).astype(int)
    bin_b = (b_vals > b_vals.median()).astype(int)
    return (bin_a + bin_b) % 2


def _conditional_direction_stats(
    df: pd.DataFrame,
    feat_a: str,
    feat_b: str,
    outcome_values: pd.Series,
) -> tuple[str, str, int, float, float]:
    """Return the exact sample and correlation used by the direction answer."""
    first, second = sorted([feat_a, feat_b])
    outcome = coerce_numeric(outcome_values)
    first_vals = coerce_numeric(df[first])
    second_vals = coerce_numeric(df[second])
    above_mask = (
        (first_vals > first_vals.median())
        & second_vals.notna()
        & outcome.notna()
    )
    n_rows = int(above_mask.sum())
    if n_rows < _DIRECTION_MIN_ROWS:
        return first, second, n_rows, float("nan"), float("nan")

    second_sample = second_vals[above_mask]
    outcome_sample = outcome[above_mask]
    if second_sample.nunique(dropna=True) < 2 or outcome_sample.nunique(dropna=True) < 2:
        return first, second, n_rows, float("nan"), float("nan")

    corr, p_value = pearsonr(second_sample, outcome_sample)
    return first, second, n_rows, float(corr), float(p_value)


def _direction_is_reliable(n_rows: int, corr: float, p_value: float) -> bool:
    return bool(
        n_rows >= _DIRECTION_MIN_ROWS
        and np.isfinite(corr)
        and np.isfinite(p_value)
        and abs(corr) >= _DIRECTION_MIN_ABS_R
        and p_value < _DIRECTION_MAX_P
    )


def _requires_direction(values: dict) -> bool:
    return str(values.get("require_direction", "false")).strip().lower() == "true"


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Plant an XOR-shaped interaction between two eligible numeric features.

    Candidate pairs are tried in seed-shuffled order. Acceptance requires an
    XOR-group Cohen's d of at least 0.5 and post-injection ``|Pearson r| <=
    0.7`` for each member; direction questions add their own reliability gate.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    effect_strength = float(params.get("effect_strength", "1.5"))
    require_direction = _requires_direction(params)
    id_no_cols = set(params.get("id_no_cols", []) or []) | id_like_cols(df)

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
    amplitude = effect_strength * outcome_std
    outcome_dp = decimal_places(outcome_vals)

    all_pairs = [
        (numeric_cols[i], numeric_cols[j])
        for i in range(len(numeric_cols))
        for j in range(i + 1, len(numeric_cols))
    ]
    order = rng.permutation(len(all_pairs))

    last_error = None

    for pair_idx in order:
        feat_a, feat_b = all_pairs[pair_idx]

        interaction = xor_signal(df[feat_a], df[feat_b])
        trial_outcome = (outcome_vals + amplitude * interaction).round(outcome_dp)

        xor_binary = _xor_binary(df[feat_a], df[feat_b])
        group1 = trial_outcome[xor_binary == 1]
        group0 = trial_outcome[xor_binary == 0]
        interaction_d = cohens_d(group1, group0)

        if interaction_d < 0.5:
            last_error = (
                f"XOR interaction too weak for ({feat_a}, {feat_b}): "
                f"Cohen's d = {interaction_d:.4f} < 0.5"
            )
            continue

        main_effect_ok = True
        for feat in [feat_a, feat_b]:
            corr = abs(float(df[feat].astype(float).corr(trial_outcome)))
            if corr > 0.7:
                last_error = (
                    f"Feature '{feat}' has too strong a main effect: "
                    f"|r| = {corr:.4f} > 0.7"
                )
                main_effect_ok = False
                break
        if not main_effect_ok:
            continue

        if require_direction:
            first, second, n_rows, corr, p_value = _conditional_direction_stats(
                df, feat_a, feat_b, trial_outcome,
            )
            if not _direction_is_reliable(n_rows, corr, p_value):
                last_error = (
                    f"Conditional direction for ({first}, {second}) is not reliable: "
                    f"r={corr:.4f}, p={p_value:.3g}, n={n_rows}; require "
                    f"n >= {_DIRECTION_MIN_ROWS}, |r| >= {_DIRECTION_MIN_ABS_R}, "
                    f"and p < {_DIRECTION_MAX_P}"
                )
                continue

        df[outcome_col] = trial_outcome

        effects = {
            "feat_a": str(feat_a),
            "feat_b": str(feat_b),
            "amplitude": float(amplitude),
            "require_direction": require_direction,
            "id_no_cols": sorted(id_no_cols),
        }

        return df, {
            "type": "fc_interaction_dominant",
            "params": params,
            "effects": effects,
        }

    raise InjectionRejected(
        f"No candidate pair survived validation. Last error: {last_error}"
    )


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    feat_a = effects["feat_a"]
    feat_b = effects["feat_b"]
    outcome_vals = coerce_numeric(df[target_col])
    id_no_cols = set(effects.get("id_no_cols", []) or []) | id_like_cols(df)
    checks = []

    injected_ids = sorted({feat_a, feat_b} & id_no_cols)
    if injected_ids:
        checks.append(CheckDetail(
            name="interaction_features_not_identifiers", passed=False,
            metric=0.0, threshold=1.0,
            detail=f"Injected features are identifier-like: {injected_ids}",
        ))
        return ValidationResult(passed=False, checks=checks)

    xor = _xor_binary(df[feat_a], df[feat_b])
    group1 = outcome_vals[xor == 1]
    group0 = outcome_vals[xor == 0]
    interaction_d = cohens_d(group1, group0)

    mean_xor1 = float(group1.mean())
    mean_xor0 = float(group0.mean())
    xor_detectable = mean_xor1 > mean_xor0
    checks.append(CheckDetail(
        name="xor_detectable", passed=xor_detectable,
        metric=mean_xor1, threshold=mean_xor0,
        detail=f"mean(XOR=1) = {mean_xor1:.4f}, mean(XOR=0) = {mean_xor0:.4f}",
    ))
    if not xor_detectable:
        return ValidationResult(passed=False, checks=checks)

    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {target_col},
        min_unique=5,
    )

    for ca, cb in combinations(numeric_cols, 2):
        if {ca, cb} == {feat_a, feat_b}:
            continue
        other_binary = _xor_binary(df[ca], df[cb])
        og1 = outcome_vals[other_binary == 1]
        og0 = outcome_vals[other_binary == 0]
        other_d = cohens_d(og1, og0)
        if other_d >= interaction_d:
            checks.append(CheckDetail(
                name="dominance_interaction", passed=False,
                metric=other_d, threshold=interaction_d,
                detail=f"Pair ({ca}, {cb}) has Cohen's d = {other_d:.4f} >= "
                       f"injected ({feat_a}, {feat_b}) d = {interaction_d:.4f}",
            ))
            return ValidationResult(passed=False, checks=checks)

    checks.append(CheckDetail(
        name="dominance_interaction", passed=True,
        metric=interaction_d, threshold=0.0,
        detail=f"Injected pair d = {interaction_d:.4f}",
    ))

    if not _requires_direction(effects):
        return ValidationResult(passed=True, checks=checks)

    # Validate the exact conditional sample used by the direction answer so a
    # PASS can never turn into answer=None later in the pipeline.
    first, second, n_rows, corr, p_value = _conditional_direction_stats(
        df, feat_a, feat_b, outcome_vals,
    )
    enough_rows = n_rows >= _DIRECTION_MIN_ROWS
    checks.append(CheckDetail(
        name="direction_min_rows", passed=enough_rows,
        metric=float(n_rows), threshold=float(_DIRECTION_MIN_ROWS),
        detail=f"Rows with '{first}' above median: {n_rows} "
               f"(require >= {_DIRECTION_MIN_ROWS})",
    ))
    if not enough_rows:
        return ValidationResult(passed=False, checks=checks)

    effect_ok = np.isfinite(corr) and abs(corr) >= _DIRECTION_MIN_ABS_R
    checks.append(CheckDetail(
        name="direction_effect_size", passed=bool(effect_ok),
        metric=abs(corr) if np.isfinite(corr) else 0.0,
        threshold=_DIRECTION_MIN_ABS_R,
        detail=f"Conditional correlation of '{second}' with outcome: "
               f"r={corr:.4f} (require |r| >= {_DIRECTION_MIN_ABS_R})",
    ))
    if not effect_ok:
        return ValidationResult(passed=False, checks=checks)

    significance_ok = np.isfinite(p_value) and p_value < _DIRECTION_MAX_P
    checks.append(CheckDetail(
        name="direction_significance", passed=bool(significance_ok),
        metric=p_value if np.isfinite(p_value) else 1.0,
        threshold=_DIRECTION_MAX_P,
        detail=f"Conditional direction p={p_value:.3g} "
               f"(require < {_DIRECTION_MAX_P})",
    ))
    if not significance_ok:
        return ValidationResult(passed=False, checks=checks)

    return ValidationResult(passed=True, checks=checks)


def compute_answer_dominant_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the feature pair with the strongest interaction effect.

    Answer format: "column_a, column_b" (sorted alphabetically).
    """
    pair = sorted([effects["feat_a"], effects["feat_b"]])
    return f"{pair[0]}, {pair[1]}"


def compute_answer_direction_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return POSITIVE or NEGATIVE for the interaction direction.

    Sorts the interacting pair alphabetically, then computes: when the
    first feature is above its median, does increasing the second
    feature increase (POSITIVE) or decrease (NEGATIVE) the outcome?
    """
    first, _second, n_rows, corr, p_value = _conditional_direction_stats(
        df,
        effects["feat_a"],
        effects["feat_b"],
        df[slot_assignments["OUTCOME_COL"]],
    )
    if n_rows < _DIRECTION_MIN_ROWS:
        raise ValueError(
            f"Too few rows with '{first}' above median: {n_rows} "
            f"< {_DIRECTION_MIN_ROWS}"
        )

    if not _direction_is_reliable(n_rows, corr, p_value):
        raise ValueError(
            f"Conditional direction is not reliable: r={corr:.4f}, "
            f"p={p_value:.3g}, n={n_rows}; require "
            f"|r| >= {_DIRECTION_MIN_ABS_R} and p < {_DIRECTION_MAX_P}"
        )

    return "POSITIVE" if corr > 0 else "NEGATIVE"


PHENOMENON = Phenomenon(
    name="fc_interaction_dominant",
    inject=inject,
    validate=validate,
    compute_answers={
        "fc_interaction_dominant_v0": compute_answer_dominant_v0,
        "fc_interaction_direction_v0": compute_answer_direction_v0,
    },
    summary_fields=("id_no",),
)
