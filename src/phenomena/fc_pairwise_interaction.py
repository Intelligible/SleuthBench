"""Pairwise binary-interaction phenomenon.

Plants an easy-to-state interaction between two numeric features, each
binarized at its median into conditions A and B. 
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import (
    coerce_numeric,
    decimal_places,
    eligible_numeric_columns,
    id_like_cols,
    safe_std,
)

from ._base import (
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
    last_failed_detail,
)


# pattern -> (beta_A, beta_B, gamma) for f = bA*A + bB*B + g*A*B
_PATTERNS: dict[str, tuple[int, int, int]] = {
    "positive_synergy": (2, 3, 6),
    "antagonistic_reversal": (2, 3, -10),
    "compensatory_reversal": (2, -3, 8),
}

# expected sign of the combined effect mu11 - mu00 (i.e. bA + bB + g).
_COMBINED_SIGN: dict[str, int] = {
    "positive_synergy": 1,
    "antagonistic_reversal": -1,
    "compensatory_reversal": 1,
}


def _binarize(series: pd.Series) -> pd.Series:
    v = coerce_numeric(series)
    return (v > v.median()).astype(float)


def _cell_means(
    y: pd.Series, A: pd.Series, B: pd.Series, min_cell: int
) -> tuple[float, float, float, float] | None:
    """Return (m00, m10, m01, m11) cell means, or None if any cell is too small."""
    means: dict[tuple[float, float], float] = {}
    for av in (0.0, 1.0):
        for bv in (0.0, 1.0):
            mask = (A == av) & (B == bv)
            if int(mask.sum()) < min_cell:
                return None
            means[(av, bv)] = float(y[mask].mean())
    return means[(0.0, 0.0)], means[(1.0, 0.0)], means[(0.0, 1.0)], means[(1.0, 1.0)]


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Plant a pairwise interaction of the requested ``pattern``.

    Eligible features are numeric with >= 5 unique values, excluding the
    outcome and any ``id_no`` column.  Candidate pairs are shuffled and tried
    in order; a pair is used only if both median splits are balanced (each side
    25-75%) and the injected table passes :func:`validate` for this pattern
    (correct conditional main-effect signs, correct interaction-contrast sign,
    correct combined-effect sign mu11-mu00, and a strictly dominant ``|contrast|``
    over every other pair).
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    pattern = str(params.get("pattern", "positive_synergy"))
    if pattern not in _PATTERNS:
        raise ValueError(f"Unknown pattern '{pattern}'; expected {sorted(_PATTERNS)}")
    effect_strength = float(params.get("effect_strength", 1.0))
    noise_sd = float(params.get("noise_sd", 0.3))
    id_no_cols = set(params.get("id_no_cols", []) or []) | id_like_cols(df)
    cA, cB, cAB = _PATTERNS[pattern]

    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {outcome_col},
        min_unique=5,
    )
    if len(numeric_cols) < 3:
        raise InjectionRejected(
            f"Need at least 3 eligible numeric features, found {len(numeric_cols)}"
        )

    y = coerce_numeric(df[outcome_col])
    y_std = safe_std(y)
    K = effect_strength * y_std
    y_dp = decimal_places(y)

    # Realism noise epsilon ~ N(0, noise_sd * sd(Y)); drawn once for stability.
    noise = pd.Series(rng.normal(0.0, noise_sd * y_std, len(df)), index=df.index)

    all_pairs = [
        (numeric_cols[i], numeric_cols[j])
        for i in range(len(numeric_cols))
        for j in range(i + 1, len(numeric_cols))
    ]
    order = rng.permutation(len(all_pairs))

    last_error = None
    for pair_idx in order:
        feat_a, feat_b = all_pairs[pair_idx]
        A = _binarize(df[feat_a])
        B = _binarize(df[feat_b])
        if not (0.25 <= float(A.mean()) <= 0.75 and 0.25 <= float(B.mean()) <= 0.75):
            last_error = f"Unbalanced median split for ({feat_a}, {feat_b})"
            continue

        f = cA * A + cB * B + cAB * (A * B)
        trial = (y + K * f + noise).round(y_dp)

        effects = {
            "feat_a": str(feat_a),
            "feat_b": str(feat_b),
            "pattern": pattern,
            "amplitude": float(K),
            "id_no_cols": sorted(id_no_cols),
        }

        trial_df = df.assign(**{outcome_col: trial})
        result = validate(trial_df, effects, outcome_col)
        if result.passed:
            out = df.copy()
            out[outcome_col] = trial
            return out, {
                "type": "fc_pairwise_interaction",
                "params": params,
                "effects": effects,
            }

        last_error = last_failed_detail(result)

    raise InjectionRejected(f"No candidate pair survived validation. Last error: {last_error}")


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    feat_a = effects["feat_a"]
    feat_b = effects["feat_b"]
    pattern = effects["pattern"]
    cA, cB, cAB = _PATTERNS[pattern]
    id_no_cols = set(effects.get("id_no_cols", []) or [])
    y = coerce_numeric(df[target_col])
    n = len(df)
    min_cell = max(3, int(0.05 * n))
    checks: list[CheckDetail] = []

    A = _binarize(df[feat_a])
    B = _binarize(df[feat_b])
    cells = _cell_means(y, A, B, min_cell)
    if cells is None:
        checks.append(CheckDetail(
            name="cell_support", passed=False, metric=0.0, threshold=float(min_cell),
            detail=f"At least one (A,B) cell has < {min_cell} rows for ({feat_a}, {feat_b})",
        ))
        return ValidationResult(passed=False, checks=checks)
    m00, m10, m01, m11 = cells
    eff_a = m10 - m00
    eff_b = m01 - m00
    contrast = m11 - m10 - m01 + m00

    a_ok = np.sign(eff_a) == np.sign(cA)
    checks.append(CheckDetail(
        name="main_effect_a", passed=bool(a_ok), metric=eff_a, threshold=0.0,
        detail=f"Conditional effect of '{feat_a}' (B=0) = {eff_a:.4f}, expected sign {np.sign(cA):+.0f}",
    ))
    if not a_ok:
        return ValidationResult(passed=False, checks=checks)

    b_ok = np.sign(eff_b) == np.sign(cB)
    checks.append(CheckDetail(
        name="main_effect_b", passed=bool(b_ok), metric=eff_b, threshold=0.0,
        detail=f"Conditional effect of '{feat_b}' (A=0) = {eff_b:.4f}, expected sign {np.sign(cB):+.0f}",
    ))
    if not b_ok:
        return ValidationResult(passed=False, checks=checks)

    ix_ok = np.sign(contrast) == np.sign(cAB)
    checks.append(CheckDetail(
        name="interaction_sign", passed=bool(ix_ok), metric=contrast, threshold=0.0,
        detail=f"Interaction contrast = {contrast:.4f}, expected sign {np.sign(cAB):+.0f}",
    ))
    if not ix_ok:
        return ValidationResult(passed=False, checks=checks)

    # Combined effect mu11 - mu00: the interaction is strong enough that having
    # both conditions flips the joint outcome to the pattern's expected side.
    combined = m11 - m00
    comb_ok = np.sign(combined) == _COMBINED_SIGN[pattern]
    checks.append(CheckDetail(
        name="combined_effect", passed=bool(comb_ok), metric=combined, threshold=0.0,
        detail=f"mu11 - mu00 = {combined:.4f}, expected sign {_COMBINED_SIGN[pattern]:+d}",
    ))
    if not comb_ok:
        return ValidationResult(passed=False, checks=checks)

    # Dominance: |contrast| strictly largest (>= 10% margin) over every other pair.
    inj = abs(contrast)
    eligible = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {target_col},
        min_unique=5,
    )
    bins = {c: _binarize(df[c]) for c in eligible}
    max_other = 0.0
    arg_other = None
    for ca, cb in combinations(eligible, 2):
        if {ca, cb} == {feat_a, feat_b}:
            continue
        other_cells = _cell_means(y, bins[ca], bins[cb], min_cell)
        if other_cells is None:
            continue
        o00, o10, o01, o11 = other_cells
        other = abs(o11 - o10 - o01 + o00)
        if other > max_other:
            max_other = other
            arg_other = (ca, cb)

    dom_ok = inj >= 1.1 * max_other
    checks.append(CheckDetail(
        name="interaction_dominance", passed=bool(dom_ok), metric=inj, threshold=1.1 * max_other,
        detail=f"Injected |contrast| = {inj:.4f}; strongest other "
               f"{arg_other} = {max_other:.4f}",
    ))
    return ValidationResult(passed=bool(dom_ok), checks=checks)


def _answer_pair(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the interacting feature pair as "column_a, column_b" (alphabetical)."""
    pair = sorted([effects["feat_a"], effects["feat_b"]])
    return f"{pair[0]}, {pair[1]}"


def _label(effects: dict) -> str:
    """Build a stable label from the pattern and interacting feature names."""
    return "_".join(
        str(effects[k]) for k in ("pattern", "feat_a", "feat_b") if effects.get(k)
    )


PHENOMENON = Phenomenon(
    name="fc_pairwise_interaction",
    inject=inject,
    validate=validate,
    compute_answers={
        "fc_pairwise_positive_synergy_v0": _answer_pair,
        "fc_pairwise_antagonistic_reversal_v0": _answer_pair,
        "fc_pairwise_compensatory_reversal_v0": _answer_pair,
    },
    summary_fields=("id_no",),
    label=_label,
)
