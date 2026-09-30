"""Non-monotone peak phenomenon.

Adds an inverted-U (quadratic) relationship between one randomly chosen
numeric feature and the outcome column. The injection is additive: the
original outcome values are shifted so that rows where the chosen feature is
near the injected centre get a boost and rows far from it get a penalty.
Other features' relationships are preserved.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from shared.metrics import (
    coerce_numeric,
    decimal_places,
    eligible_numeric_columns,
    id_like_cols,
    nonmonotone_score,
    safe_std,
)

from ._base import (
    AnswerUnavailable,
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
)


_PEAK_ATTEMPTS_PER_FEATURE = 5

# Observed-peak estimator settings.
SUPPORT_QUANTILES = (0.05, 0.95)
GRID_SIZE = 201
KERNEL_BANDWIDTH_FRACTION = 0.12
LOCAL_QUADRATIC_SPAN = 0.50
MIN_SAMPLES = 100
MIN_UNIQUE_X = 10


@dataclass(frozen=True)
class PeakGold:
    feature: str
    raw_estimate: float           # mean of the two smoothers' argmax
    kernel_peak: float
    local_quadratic_peak: float
    integer_feature: bool
    value: float                  # rounded gold value

    @property
    def answer(self) -> str:
        return format_answer(self.feature, self.value, self.integer_feature)


def is_integer_valued(series: pd.Series) -> bool:
    """True when every finite value is a whole number (dtype is irrelevant)."""
    values = coerce_numeric(series).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return len(values) > 0 and bool(np.all(values == np.round(values)))


def format_answer(feature: str, value: float, integer_feature: bool) -> str:
    """Serialize the gold as ``feature, value``.

    Integer features print an integer; other features use ``repr(float)``,
    the shortest round-trippable decimal, so no significant digit is lost.
    """
    if integer_feature:
        return f"{feature}, {int(round(value))}"
    return f"{feature}, {float(value)!r}"


def _finite_pairs(df: pd.DataFrame, feature: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    x_series = coerce_numeric(df[feature])
    y_series = coerce_numeric(df[target])
    finite = np.isfinite(x_series.to_numpy(dtype=float)) & np.isfinite(
        y_series.to_numpy(dtype=float)
    )
    return (
        x_series.to_numpy(dtype=float)[finite],
        y_series.to_numpy(dtype=float)[finite],
    )


def _local_linear_curve(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    support_low: float,
    support_high: float,
    bandwidth_fraction: float,
) -> np.ndarray:
    scale = support_high - support_low
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("feature support has zero width")
    x_norm = (x - support_low) / scale
    grid_norm = (grid - support_low) / scale
    dx = x_norm[None, :] - grid_norm[:, None]
    weights = np.exp(-0.5 * (dx / bandwidth_fraction) ** 2)
    s0 = weights.sum(axis=1)
    s1 = (weights * dx).sum(axis=1)
    s2 = (weights * dx * dx).sum(axis=1)
    t0 = (weights * y[None, :]).sum(axis=1)
    t1 = (weights * dx * y[None, :]).sum(axis=1)
    denominator = s0 * s2 - s1 * s1
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = (s2 * t0 - s1 * t1) / denominator
    fallback = np.divide(t0, s0, out=np.full_like(t0, np.nan), where=s0 > 0)
    curve = np.where(np.isfinite(curve), curve, fallback)
    return curve.astype(float)


def _local_quadratic_curve(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    support_low: float,
    support_high: float,
    span: float,
) -> np.ndarray:
    """Return a fixed-span degree-2 LOESS-style curve."""
    scale = support_high - support_low
    x_norm = (x - support_low) / scale
    grid_norm = (grid - support_low) / scale
    neighbour_count = min(len(x), max(7, int(np.ceil(span * len(x)))))
    dx = x_norm[None, :] - grid_norm[:, None]
    distance = np.abs(dx)
    bandwidth = np.partition(
        distance,
        neighbour_count - 1,
        axis=1,
    )[:, neighbour_count - 1]
    valid = bandwidth > 0
    scaled = np.divide(
        distance,
        bandwidth[:, None],
        out=np.full_like(distance, np.inf),
        where=valid[:, None],
    )
    weights = np.where(
        scaled < 1.0,
        (1.0 - scaled ** 3) ** 3,
        0.0,
    )
    dx2 = dx * dx
    weighted_dx = weights * dx
    weighted_dx2 = weights * dx2
    moments = [
        weights.sum(axis=1),
        weighted_dx.sum(axis=1),
        weighted_dx2.sum(axis=1),
        (weighted_dx2 * dx).sum(axis=1),
        (weighted_dx2 * dx2).sum(axis=1),
    ]
    matrices = np.empty((len(grid), 3, 3), dtype=float)
    matrices[:, 0, :] = np.column_stack(moments[:3])
    matrices[:, 1, :] = np.column_stack(moments[1:4])
    matrices[:, 2, :] = np.column_stack(moments[2:5])
    right_hand_side = np.column_stack([
        (weights * y[None, :]).sum(axis=1),
        (weighted_dx * y[None, :]).sum(axis=1),
        (weighted_dx2 * y[None, :]).sum(axis=1),
    ])
    curve = np.full(len(grid), np.nan, dtype=float)
    if bool(valid.any()):
        # Batched 3x3 pseudo-inverses are materially faster than hundreds of
        # large weighted least-squares calls and remain deterministic for
        # duplicate-heavy samples.
        inverse = np.linalg.pinv(
            matrices[valid],
            rcond=1e-12,
            hermitian=True,
        )
        coefficients = np.einsum(
            "gij,gj->gi",
            inverse,
            right_hand_side[valid],
        )
        curve[valid] = coefficients[:, 0]
    return curve


def estimate_observed_peak(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
) -> PeakGold:
    """Estimate the peak of the final table's feature -> target relationship."""
    x, y = _finite_pairs(df, feature, target_col)
    if len(x) < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} finite rows, found {len(x)}")
    if len(np.unique(x)) < MIN_UNIQUE_X:
        raise ValueError(
            f"need at least {MIN_UNIQUE_X} distinct feature values, "
            f"found {len(np.unique(x))}"
        )
    low, high = np.quantile(x, SUPPORT_QUANTILES)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("feature support has zero width")
    grid = np.linspace(low, high, GRID_SIZE)

    kernel = _local_linear_curve(x, y, grid, low, high, KERNEL_BANDWIDTH_FRACTION)
    local = _local_quadratic_curve(x, y, grid, low, high, LOCAL_QUADRATIC_SPAN)
    kernel_peak = float(grid[int(np.nanargmax(kernel))])
    local_peak = float(grid[int(np.nanargmax(local))])
    raw = float((kernel_peak + local_peak) / 2.0)

    integer_feature = is_integer_valued(df[feature])
    if integer_feature:
        value = float(np.round(raw))
    else:
        places = max(decimal_places(df[feature]), 2)
        value = float(round(raw, places))

    return PeakGold(
        feature=feature,
        raw_estimate=raw,
        kernel_peak=kernel_peak,
        local_quadratic_peak=local_peak,
        integer_feature=integer_feature,
        value=value,
    )


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Plant an inverted-U effect in one eligible numeric feature.

    Requires at least two eligible numeric features (excluding the outcome and
    identifier-like columns), each with at least ten unique values. Features
    are shuffled once and cycled through five rounds; every attempt draws a
    fresh centre percentile from 35%-65%. The quadratic weight becomes
    negative beyond half the observed range. A candidate must have a
    non-degenerate range, bin into at least three quantiles with an interior
    highest-mean bin, reach a non-monotone score of at least 0.5 times the
    outcome standard deviation, and then pass :func:`validate`.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    effect_strength = float(params.get("effect_strength", "2.0"))
    id_no_cols = set(params.get("id_no_cols", []) or []) | id_like_cols(df)

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

    feature_order = rng.permutation(len(numeric_cols))
    order = np.tile(feature_order, _PEAK_ATTEMPTS_PER_FEATURE)
    last_error = None

    for idx in order:
        chosen_feature = numeric_cols[idx]
        feature_vals = df[chosen_feature].astype(float)

        pct = rng.uniform(0.35, 0.65)
        centre_raw = float(np.percentile(feature_vals.dropna(), pct * 100))
        feature_dp = decimal_places(df[chosen_feature])
        centre = round(centre_raw, feature_dp)

        feat_min = float(feature_vals.min())
        feat_max = float(feature_vals.max())
        half_range = (feat_max - feat_min) / 2.0
        if half_range == 0:
            last_error = f"Feature '{chosen_feature}' has zero range"
            continue

        normalised_dist = (feature_vals - centre) / half_range
        quadratic = 1.0 - normalised_dist ** 2
        trial_outcome = (outcome_vals + amplitude * quadratic).round(outcome_dp)

        try:
            bins = pd.qcut(feature_vals, q=5, duplicates="drop")
        except ValueError:
            last_error = f"Cannot bin feature '{chosen_feature}' into quantiles"
            continue
        bin_means = trial_outcome.groupby(bins, observed=True).mean().sort_index()
        bin_labels = bin_means.index.tolist()

        if len(bin_labels) < 3:
            last_error = f"Feature '{chosen_feature}' produced fewer than 3 bins"
            continue

        peak_bin = bin_means.idxmax()
        if peak_bin == bin_labels[0] or peak_bin == bin_labels[-1]:
            last_error = (
                f"Peak bin is at an extreme for feature '{chosen_feature}'; "
                f"inverted-U shape not achieved"
            )
            continue

        injected_score = nonmonotone_score(feature_vals, trial_outcome)
        if injected_score < 0.5 * outcome_std:
            last_error = (
                f"Injected non-monotone amplitude too weak: "
                f"{injected_score:.4f} < {0.5 * outcome_std:.4f}"
            )
            continue

        effects = {
            "peak_feature": str(chosen_feature),
            "injected_centre": float(centre),
            "amplitude": float(amplitude),
            "half_range": float(half_range),
            "id_no_cols": sorted(id_no_cols),
        }

        trial_df = df.copy()
        trial_df[outcome_col] = trial_outcome
        validation = validate(trial_df, effects, outcome_col)
        if not validation.passed:
            failed_check = next(
                (check for check in validation.checks if not check.passed),
                None,
            )
            if failed_check is None:
                last_error = f"Candidate '{chosen_feature}' failed full validation"
            else:
                last_error = (
                    f"Candidate '{chosen_feature}' failed {failed_check.name}: "
                    f"{failed_check.detail}"
                )
            continue

        return trial_df, {
            "type": "fc_nonmonotone_peak",
            "params": params,
            "effects": effects,
        }

    raise InjectionRejected(
        f"No candidate survived validation. Last error: {last_error}"
    )


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    """Stage 2 checks on the injected centre.

    Checks, in order: the feature is not identifier-like; a single quintile
    bin has the highest mean outcome (``peak_bin_unique``); that bin is
    interior (``peak_not_at_edge``); the injected centre lies inside it
    (``injected_centre_within_observed_peak_bin``); and the feature's
    non-monotone score exceeds every other eligible feature
    (``dominance_nonmonotone_score``).
    """
    feature = effects["peak_feature"]
    id_no_cols = set(effects.get("id_no_cols", []) or []) | id_like_cols(df)
    checks = []

    if feature in id_no_cols:
        checks.append(CheckDetail(
            name="peak_feature_not_identifier", passed=False, metric=0.0, threshold=1.0,
            detail=f"Injected feature '{feature}' is identifier-like",
        ))
        return ValidationResult(passed=False, checks=checks)

    feature_vals = coerce_numeric(df[feature])
    outcome_vals = coerce_numeric(df[target_col])

    try:
        bins, bin_edges = pd.qcut(
            feature_vals, q=5, duplicates="drop", retbins=True,
        )
    except ValueError:
        checks.append(CheckDetail(
            name="peak_not_at_edge", passed=False, metric=0.0, threshold=1.0,
            detail=f"Cannot bin feature '{feature}' into quantiles",
        ))
        return ValidationResult(passed=False, checks=checks)

    bin_means = outcome_vals.groupby(bins, observed=True).mean().sort_index()
    bin_labels = list(bins.cat.categories)

    if len(bin_labels) < 3:
        checks.append(CheckDetail(
            name="peak_not_at_edge", passed=False, metric=float(len(bin_labels)), threshold=3.0,
            detail=f"Feature '{feature}' produced fewer than 3 bins",
        ))
        return ValidationResult(passed=False, checks=checks)

    finite_bin_means = bin_means[np.isfinite(bin_means.to_numpy(dtype=float))]
    if finite_bin_means.empty:
        checks.append(CheckDetail(
            name="peak_mean_available", passed=False, metric=0.0, threshold=1.0,
            detail=f"Feature '{feature}' has no quantile bin with a finite outcome mean",
        ))
        return ValidationResult(passed=False, checks=checks)

    max_mean = float(finite_bin_means.max())
    peak_bins = finite_bin_means[finite_bin_means == max_mean].index.tolist()
    unique_peak = len(peak_bins) == 1
    checks.append(CheckDetail(
        name="peak_bin_unique", passed=unique_peak,
        metric=1.0 if unique_peak else 0.0, threshold=1.0,
        detail=f"Found {len(peak_bins)} quantile bin(s) tied for highest mean outcome",
    ))
    if not unique_peak:
        return ValidationResult(passed=False, checks=checks)

    peak_bin = peak_bins[0]
    peak_at_edge = peak_bin == bin_labels[0] or peak_bin == bin_labels[-1]
    peak_idx = bin_labels.index(peak_bin)
    checks.append(CheckDetail(
        name="peak_not_at_edge", passed=not peak_at_edge,
        metric=float(peak_idx), threshold=1.0,
        detail=f"Peak in bin {peak_idx + 1} of {len(bin_labels)}",
    ))
    if peak_at_edge:
        return ValidationResult(passed=False, checks=checks)

    injected_centre = float(effects["injected_centre"])
    peak_bin_left = float(bin_edges[peak_idx])
    peak_bin_right = float(bin_edges[peak_idx + 1])
    scale = max(1.0, abs(peak_bin_left), abs(peak_bin_right), abs(injected_centre))
    tolerance = 8 * np.finfo(float).eps * scale
    centre_in_bin = bool(
        np.isfinite(injected_centre)
        and peak_bin_left - tolerance <= injected_centre <= peak_bin_right + tolerance
    )
    checks.append(CheckDetail(
        name="injected_centre_within_observed_peak_bin",
        passed=centre_in_bin,
        metric=1.0 if centre_in_bin else 0.0,
        threshold=1.0,
        detail=f"Injected centre {injected_centre:g}; highest-mean bin {peak_idx + 1} "
               f"spans [{peak_bin_left:g}, {peak_bin_right:g}]",
    ))
    if not centre_in_bin:
        return ValidationResult(passed=False, checks=checks)

    injected_score = nonmonotone_score(feature_vals, outcome_vals)
    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {target_col, feature},
        min_unique=10,
    )

    for other_col in numeric_cols:
        other_score = nonmonotone_score(df[other_col].astype(float), outcome_vals)
        if other_score >= injected_score:
            checks.append(CheckDetail(
                name="dominance_nonmonotone_score", passed=False,
                metric=other_score, threshold=injected_score,
                detail=f"Feature '{other_col}' has nonmonotone score "
                       f"({other_score:.4f}) >= injected '{feature}' ({injected_score:.4f})",
            ))
            return ValidationResult(passed=False, checks=checks)

    checks.append(CheckDetail(
        name="dominance_nonmonotone_score", passed=True,
        metric=injected_score, threshold=0.0,
        detail=f"Injected score {injected_score:.4f}",
    ))
    return ValidationResult(passed=True, checks=checks)


def verify_observed_peak(
    df: pd.DataFrame,
    effects: dict,
    target_col: str,
) -> tuple[ValidationResult, PeakGold | None]:
    """Stage 2's checks with the observed peak in place of the injected centre.

    Records ``peak_bin_unique``, ``peak_not_at_edge``,
    ``observed_peak_in_peak_bin`` and ``dominance_nonmonotone_score``;
    ``observed_peak_estimable`` is recorded only when the estimator cannot run;
    ``peak_feature_not_identifier`` and ``peak_mean_available`` only on those
    early failures.
    """
    feature = str(effects["peak_feature"])
    id_no_cols = set(effects.get("id_no_cols", []) or []) | id_like_cols(df)
    checks: list[CheckDetail] = []

    if feature in id_no_cols:
        checks.append(CheckDetail(
            name="peak_feature_not_identifier", passed=False, metric=0.0, threshold=1.0,
            detail=f"Injected feature '{feature}' is identifier-like",
        ))
        return ValidationResult(passed=False, checks=checks), None

    try:
        gold = estimate_observed_peak(df, feature, target_col)
    except ValueError as exc:
        checks.append(CheckDetail(
            name="observed_peak_estimable", passed=False, metric=0.0, threshold=1.0,
            detail=str(exc),
        ))
        return ValidationResult(passed=False, checks=checks), None

    feature_vals = coerce_numeric(df[feature])
    outcome_vals = coerce_numeric(df[target_col])

    try:
        bins, bin_edges = pd.qcut(
            feature_vals, q=5, duplicates="drop", retbins=True,
        )
    except ValueError:
        checks.append(CheckDetail(
            name="peak_not_at_edge", passed=False, metric=0.0, threshold=1.0,
            detail=f"Cannot bin feature '{feature}' into quantiles",
        ))
        return ValidationResult(passed=False, checks=checks), gold

    bin_means = outcome_vals.groupby(bins, observed=True).mean().sort_index()
    bin_labels = list(bins.cat.categories)
    if len(bin_labels) < 3:
        checks.append(CheckDetail(
            name="peak_not_at_edge", passed=False, metric=float(len(bin_labels)), threshold=3.0,
            detail=f"Feature '{feature}' produced fewer than 3 bins",
        ))
        return ValidationResult(passed=False, checks=checks), gold

    finite_bin_means = bin_means[np.isfinite(bin_means.to_numpy(dtype=float))]
    if finite_bin_means.empty:
        checks.append(CheckDetail(
            name="peak_mean_available", passed=False, metric=0.0, threshold=1.0,
            detail=f"Feature '{feature}' has no quantile bin with a finite outcome mean",
        ))
        return ValidationResult(passed=False, checks=checks), gold

    # 1. A single highest-mean quintile bin.
    max_mean = float(finite_bin_means.max())
    peak_bins = finite_bin_means[finite_bin_means == max_mean].index.tolist()
    unique_peak = len(peak_bins) == 1
    checks.append(CheckDetail(
        name="peak_bin_unique", passed=unique_peak,
        metric=1.0 if unique_peak else 0.0, threshold=1.0,
        detail=f"Found {len(peak_bins)} quantile bin(s) tied for highest mean outcome",
    ))
    if not unique_peak:
        return ValidationResult(passed=False, checks=checks), gold

    # 2. That bin is interior.
    peak_bin = peak_bins[0]
    peak_idx = bin_labels.index(peak_bin)
    peak_at_edge = peak_idx == 0 or peak_idx == len(bin_labels) - 1
    checks.append(CheckDetail(
        name="peak_not_at_edge", passed=not peak_at_edge,
        metric=float(peak_idx), threshold=1.0,
        detail=f"Peak in bin {peak_idx + 1} of {len(bin_labels)}",
    ))
    if peak_at_edge:
        return ValidationResult(passed=False, checks=checks), gold

    # 3. The observed peak lies inside that bin.
    left = float(bin_edges[peak_idx])
    right = float(bin_edges[peak_idx + 1])
    scale = max(1.0, abs(left), abs(right), abs(gold.value))
    tolerance = 8 * np.finfo(float).eps * scale
    in_bin = bool(
        np.isfinite(gold.value) and left - tolerance <= gold.value <= right + tolerance
    )
    checks.append(CheckDetail(
        name="observed_peak_in_peak_bin", passed=in_bin,
        metric=1.0 if in_bin else 0.0, threshold=1.0,
        detail=(
            f"Observed peak {gold.answer!r} (raw estimate {gold.raw_estimate:.4g}); "
            f"highest-mean bin {peak_idx + 1} spans [{left:g}, {right:g}]"
        ),
    ))
    if not in_bin:
        return ValidationResult(passed=False, checks=checks), gold

    # 4. No other numeric feature has an equal or higher nonmonotone score.
    injected_score = nonmonotone_score(feature_vals, outcome_vals)
    numeric_cols = eligible_numeric_columns(
        df,
        exclude=id_no_cols | {target_col, feature},
        min_unique=10,
    )
    for other_col in numeric_cols:
        other_score = nonmonotone_score(df[other_col].astype(float), outcome_vals)
        if other_score >= injected_score:
            checks.append(CheckDetail(
                name="dominance_nonmonotone_score", passed=False,
                metric=other_score, threshold=injected_score,
                detail=f"Feature '{other_col}' has nonmonotone score "
                       f"({other_score:.4f}) >= injected '{feature}' ({injected_score:.4f})",
            ))
            return ValidationResult(passed=False, checks=checks), gold
    checks.append(CheckDetail(
        name="dominance_nonmonotone_score", passed=True,
        metric=injected_score, threshold=0.0,
        detail=f"Injected score {injected_score:.4f}",
    ))
    return ValidationResult(passed=True, checks=checks), gold


def compute_answer_v0(
    df: pd.DataFrame,
    slot_assignments: dict,
    effects: dict,
) -> Any:
    """Return the observed-peak gold, or raise ``AnswerUnavailable``.

    Answer format: "feature_name, observed_peak". The verifier runs on the
    table passed in; the injected centre in ``effects`` is not used as the
    answer and ``effects`` is left unchanged.
    """
    target_col = str(
        slot_assignments.get("OUTCOME_COL")
        or effects.get("outcome_col", "")
    )
    if not target_col:
        raise ValueError(
            "fc_nonmonotone_peak_v0 needs OUTCOME_COL in slot_assignments"
        )
    result, gold = verify_observed_peak(df, effects, target_col)
    if not result.passed or gold is None:
        failed = [c for c in result.checks if not c.passed]
        detail = "; ".join(f"{c.name}: {c.detail}" for c in failed) or "unknown"
        raise AnswerUnavailable(f"observed-peak verification failed: {detail}")
    return gold.answer


PHENOMENON = Phenomenon(
    name="fc_nonmonotone_peak",
    inject=inject,
    validate=validate,
    compute_answers={
        "fc_nonmonotone_peak_v0": compute_answer_v0,
    },
    summary_fields=("id_no",),
)
