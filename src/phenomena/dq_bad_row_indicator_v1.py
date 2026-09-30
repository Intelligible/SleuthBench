"""Conditional-error row indicator phenomenon.

Rewrites a small set of outcomes to be anomalous relative to an out-of-fold
prediction while staying inside the clean target's observed range, then adds a
0/1 indicator for those rows plus same-prevalence placebo flags and renames
non-target, non-identifier columns. The ground-truth answer is the indicator
column. Validation requires it to be the unique best identifier of the largest
conditional residuals, with every 0/1 column excluded from the residual model.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from shared.metrics import coerce_numeric, id_like_cols

from ._base import (
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
    last_failed_detail,
)

_PHENOMENON_NAME = "dq_bad_row_indicator_v1"
_TEMPLATE_ID = "dq_bad_row_indicator_v1"


@dataclass(frozen=True)
class _DetectorConfig:
    """Configuration persisted in ``effects`` for deterministic validation."""

    n_splits: int = 5
    n_estimators: int = 96
    min_samples_leaf: int = 2
    max_categories: int = 32
    min_indicator_auc: float = 0.90
    min_auc_margin: float = 0.20
    min_binary_hybrid: float = 0.70
    min_binary_hybrid_margin: float = 0.10

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "_DetectorConfig":
        values = values or {}
        defaults = cls()
        return cls(
            n_splits=int(values.get("n_splits", defaults.n_splits)),
            n_estimators=int(values.get("n_estimators", defaults.n_estimators)),
            min_samples_leaf=int(
                values.get("min_samples_leaf", defaults.min_samples_leaf)
            ),
            max_categories=int(values.get("max_categories", defaults.max_categories)),
            min_indicator_auc=float(
                values.get("min_indicator_auc", defaults.min_indicator_auc)
            ),
            min_auc_margin=float(
                values.get("min_auc_margin", defaults.min_auc_margin)
            ),
            min_binary_hybrid=float(
                values.get("min_binary_hybrid", defaults.min_binary_hybrid)
            ),
            min_binary_hybrid_margin=float(
                values.get(
                    "min_binary_hybrid_margin",
                    defaults.min_binary_hybrid_margin,
                )
            ),
        )


@dataclass(frozen=True)
class _ResidualProfile:
    predictions: np.ndarray
    signed_residuals: np.ndarray
    residual_scores: np.ndarray
    raw_features: tuple[str, ...]
    target_is_binary: bool


@dataclass(frozen=True)
class _CandidateScores:
    indicator_auc: float
    runner_up_auc: float
    auc_margin: float
    overlap: int
    top_positions: tuple[int, ...]
    auc_by_column: dict[str, float]
    hybrid_score: float | None
    runner_up_hybrid: float | None
    hybrid_by_column: dict[str, float]


def _mad_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def _target_values(df: pd.DataFrame, target_col: str) -> np.ndarray:
    if target_col not in df.columns:
        raise InjectionRejected(f"target column {target_col!r} is missing")
    values = coerce_numeric(df[target_col]).to_numpy(dtype=float)
    if len(values) < 100 or not np.all(np.isfinite(values)):
        raise InjectionRejected("target must be finite and contain at least 100 rows")
    if len(np.unique(values)) < 2:
        raise InjectionRejected("target has fewer than two distinct values")
    return values


def _is_binary_target(values: np.ndarray) -> bool:
    return len(np.unique(values)) == 2


def _zero_one_columns(
    df: pd.DataFrame,
    target_col: str,
    id_no_cols: set[str],
) -> list[str]:
    """Return complete, numeric/coercible columns whose values are exactly 0/1."""
    excluded = set(id_no_cols) | {target_col, "row_id"}
    columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        values = coerce_numeric(df[column])
        if not bool(values.notna().all()):
            continue
        observed = set(values.astype(float).unique().tolist())
        if observed == {0.0, 1.0}:
            columns.append(str(column))
    return columns


def _model_matrix(
    df: pd.DataFrame,
    target_col: str,
    id_no_cols: set[str],
    max_categories: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build a target-blind model matrix, excluding every 0/1 candidate.

    Numeric columns are median-filled.  Low-cardinality non-numeric columns are
    one-hot encoded; high-cardinality strings are excluded because they are
    commonly identifiers or timestamps and do not generalise out of fold.
    """
    binary_candidates = set(_zero_one_columns(df, target_col, id_no_cols))
    excluded = set(id_no_cols) | id_like_cols(df) | binary_candidates
    excluded |= {target_col, "row_id"}

    frames: list[pd.DataFrame] = []
    raw_features: list[str] = []
    min_observed = max(20, len(df) // 2)

    for column in df.columns:
        name = str(column)
        if name in excluded:
            continue
        series = df[column]

        if is_numeric_dtype(series):
            numeric = coerce_numeric(series).replace([np.inf, -np.inf], np.nan)
            if int(numeric.notna().sum()) < min_observed:
                continue
            median = float(numeric.median())
            if not np.isfinite(median):
                continue
            filled = numeric.fillna(median).astype(float)
            if int(filled.nunique(dropna=False)) < 2:
                continue
            frames.append(pd.DataFrame({f"numeric::{name}": filled}, index=df.index))
            raw_features.append(name)
            continue

        categorical = series.astype("string").fillna("__missing__")
        cardinality = int(categorical.nunique(dropna=False))
        if not (2 <= cardinality <= max_categories):
            continue
        encoded = pd.get_dummies(
            categorical,
            prefix=f"categorical::{name}",
            dtype=float,
        )
        frames.append(encoded)
        raw_features.append(name)

    if len(raw_features) < 2 or not frames:
        raise InjectionRejected(
            "need at least two usable non-ID, non-binary predictor columns"
        )
    matrix = pd.concat(frames, axis=1)
    return matrix, tuple(raw_features)


def _oof_profile(
    df: pd.DataFrame,
    target_col: str,
    id_no_cols: set[str],
    config: _DetectorConfig,
    cv_seed: int,
) -> _ResidualProfile:
    y = _target_values(df, target_col)
    target_is_binary = _is_binary_target(y)
    matrix, raw_features = _model_matrix(
        df,
        target_col,
        id_no_cols,
        config.max_categories,
    )
    predictions = np.empty(len(df), dtype=float)
    model_target = y

    if target_is_binary:
        target_classes = np.unique(y)
        model_target = (y == target_classes[-1]).astype(int)
        class_counts = np.bincount(model_target, minlength=2)
        n_splits = min(config.n_splits, int(class_counts.min()))
        if n_splits < 3:
            raise InjectionRejected("too few examples per target class for cross-fitting")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=cv_seed,
        )
        folds = splitter.split(matrix, model_target)
    else:
        n_splits = min(config.n_splits, max(2, len(df) // 20))
        if n_splits < 3:
            raise InjectionRejected("too few rows for cross-fitting")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
        folds = splitter.split(matrix)

    for fold, (train, test) in enumerate(folds):
        random_state = int((cv_seed + 104729 * (fold + 1)) % (2**31 - 1))
        if target_is_binary:
            model = ExtraTreesClassifier(
                n_estimators=config.n_estimators,
                min_samples_leaf=config.min_samples_leaf,
                max_features=1.0,
                class_weight="balanced",
                n_jobs=1,
                random_state=random_state,
            )
            model.fit(matrix.iloc[train], model_target[train])
            classes = list(model.classes_)
            if 1 not in classes:
                raise InjectionRejected("a classifier fold has no positive target class")
            predictions[test] = model.predict_proba(matrix.iloc[test])[:, classes.index(1)]
        else:
            model = ExtraTreesRegressor(
                n_estimators=config.n_estimators,
                min_samples_leaf=config.min_samples_leaf,
                max_features=1.0,
                n_jobs=1,
                random_state=random_state,
            )
            model.fit(matrix.iloc[train], y[train])
            predictions[test] = model.predict(matrix.iloc[test])

    signed_residuals = model_target - predictions
    return _ResidualProfile(
        predictions=predictions,
        signed_residuals=signed_residuals,
        residual_scores=np.abs(signed_residuals),
        raw_features=raw_features,
        target_is_binary=target_is_binary,
    )


def _binary_hybrid_scores(
    df: pd.DataFrame,
    target_col: str,
    candidate_columns: list[str],
) -> dict[str, float]:
    """Combine directed target lift and point-biserial correlation.

    The extra gate is used only for binary targets.  It is less sensitive than
    a fitted probability model to the small number of positive AI4I examples,
    while still penalising tiny natural flags through the correlation term.
    """
    target = coerce_numeric(df[target_col]).to_numpy(dtype=float)
    classes = np.unique(target)
    if len(classes) != 2:
        return {}
    encoded_target = (target == classes[-1]).astype(float)
    scores: dict[str, float] = {}
    for column in candidate_columns:
        flag = coerce_numeric(df[column]).to_numpy(dtype=float)
        ones = flag == 1.0
        zeros = flag == 0.0
        if not bool(ones.any()) or not bool(zeros.any()):
            continue
        lift = float(encoded_target[ones].mean() - encoded_target[zeros].mean())
        correlation = float(np.corrcoef(flag, encoded_target)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
        scores[column] = 0.5 * max(0.0, lift) + 0.5 * max(0.0, correlation)
    return scores


def _candidate_scores(
    df: pd.DataFrame,
    target_col: str,
    indicator_col: str,
    bad_positions: np.ndarray,
    n_bad: int,
    id_no_cols: set[str],
    config: _DetectorConfig,
    cv_seed: int,
) -> tuple[_ResidualProfile, _CandidateScores]:
    profile = _oof_profile(df, target_col, id_no_cols, config, cv_seed)
    order = np.argsort(-profile.residual_scores, kind="mergesort")
    top_positions = tuple(sorted(int(value) for value in order[:n_bad]))
    problematic = np.zeros(len(df), dtype=int)
    problematic[np.asarray(top_positions, dtype=int)] = 1

    candidates = _zero_one_columns(df, target_col, id_no_cols)
    if indicator_col not in candidates:
        raise InjectionRejected("indicator is not a complete 0/1 candidate column")
    auc_by_column = {
        column: float(
            roc_auc_score(
                problematic,
                coerce_numeric(df[column]).to_numpy(dtype=float),
            )
        )
        for column in candidates
    }
    indicator_auc = auc_by_column[indicator_col]
    runner_up_auc = max(
        (score for column, score in auc_by_column.items() if column != indicator_col),
        default=0.5,
    )
    overlap = len(set(int(value) for value in bad_positions) & set(top_positions))

    hybrid_by_column: dict[str, float] = {}
    hybrid_score: float | None = None
    runner_up_hybrid: float | None = None
    if profile.target_is_binary:
        hybrid_by_column = _binary_hybrid_scores(df, target_col, candidates)
        hybrid_score = hybrid_by_column.get(indicator_col, 0.0)
        runner_up_hybrid = max(
            (
                score
                for column, score in hybrid_by_column.items()
                if column != indicator_col
            ),
            default=0.0,
        )

    return profile, _CandidateScores(
        indicator_auc=indicator_auc,
        runner_up_auc=runner_up_auc,
        auc_margin=indicator_auc - runner_up_auc,
        overlap=overlap,
        top_positions=top_positions,
        auc_by_column=auc_by_column,
        hybrid_score=hybrid_score,
        runner_up_hybrid=runner_up_hybrid,
        hybrid_by_column=hybrid_by_column,
    )


def _trial_is_strong(scores: _CandidateScores, config: _DetectorConfig) -> bool:
    if scores.indicator_auc < config.min_indicator_auc:
        return False
    if scores.auc_margin < config.min_auc_margin:
        return False
    if scores.hybrid_score is not None:
        runner = scores.runner_up_hybrid or 0.0
        if scores.hybrid_score < config.min_binary_hybrid:
            return False
        if scores.hybrid_score - runner < config.min_binary_hybrid_margin:
            return False
    return True


def _unique_internal_name(df: pd.DataFrame, stem: str) -> str:
    name = stem
    counter = 1
    while name in df.columns:
        name = f"{stem}_{counter}"
        counter += 1
    return name


def _insert_flag(
    df: pd.DataFrame,
    column: str,
    positions: np.ndarray,
    rng: np.random.Generator,
) -> None:
    values = np.zeros(len(df), dtype=int)
    values[np.asarray(positions, dtype=int)] = 1
    insert_at = int(rng.integers(0, len(df.columns) + 1))
    df.insert(insert_at, column, values)


def _add_indicator_and_placebos(
    df: pd.DataFrame,
    bad_positions: np.ndarray,
    n_placebos: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, str, list[str]]:
    trial = df.copy()
    indicator_col = _unique_internal_name(trial, "__v1_bad_flag")
    _insert_flag(trial, indicator_col, bad_positions, rng)

    placebo_columns: list[str] = []
    bad_set = frozenset(int(value) for value in bad_positions)
    used_sets = {bad_set}
    for number in range(n_placebos):
        placebo_col = _unique_internal_name(trial, f"__v1_placebo_{number}")
        for _ in range(32):
            positions = np.sort(
                rng.choice(len(df), size=len(bad_positions), replace=False)
            )
            position_set = frozenset(int(value) for value in positions)
            if position_set not in used_sets:
                used_sets.add(position_set)
                break
        else:
            raise InjectionRejected("could not generate a distinct placebo flag")
        _insert_flag(trial, placebo_col, positions, rng)
        placebo_columns.append(placebo_col)
    return trial, indicator_col, placebo_columns


def _unique_random_name(
    rng: np.random.Generator,
    taken: set[str],
    length: int = 5,
) -> str:
    letters = np.asarray(list(string.ascii_lowercase))
    while True:
        name = "".join(str(value) for value in rng.choice(letters, size=length))
        if name not in taken:
            taken.add(name)
            return name


def _rename_columns(
    df: pd.DataFrame,
    target_col: str,
    id_no_cols: set[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, str]]:
    preserve = set(id_no_cols) | {target_col, "row_id"}
    preserve &= set(str(column) for column in df.columns)
    taken = set(preserve)
    rename_map: dict[str, str] = {}
    for column in df.columns:
        name = str(column)
        if name in preserve:
            continue
        rename_map[name] = _unique_random_name(rng, taken)
    return df.rename(columns=rename_map), rename_map


def _continuous_proposals(
    target: np.ndarray,
    profile: _ResidualProfile,
    pool: np.ndarray,
    allowed_low: float,
    allowed_high: float,
    amplitude: float,
    integer_target: bool,
    target_dtype: Any,
) -> dict[int, float]:
    proposals: dict[int, float] = {}
    for position in pool:
        prediction = float(profile.predictions[position])
        room_low = prediction - allowed_low
        room_high = allowed_high - prediction
        direction = 1.0 if room_high >= room_low else -1.0
        if max(room_low, room_high) < amplitude:
            continue
        value = prediction + direction * amplitude
        if integer_target:
            value = float(np.rint(value))
        # Judge feasibility in the representation that inject() will persist.
        # In particular, a float64 value at a quantile boundary can round just
        # outside that boundary when the original target dtype is float32.
        value = float(
            pd.Series([value], dtype=float).astype(target_dtype).iloc[0]
        )
        if not np.isfinite(value):
            continue
        if not (allowed_low <= value <= allowed_high):
            continue
        if abs(value - prediction) < 0.98 * amplitude:
            continue
        if np.isclose(value, target[position]):
            continue
        proposals[int(position)] = value
    return proposals


def _renamed_effects(
    effects: dict[str, Any],
    rename_map: dict[str, str],
) -> dict[str, Any]:
    renamed = dict(effects)
    renamed["indicator_col"] = rename_map[effects["indicator_col"]]
    renamed["placebo_cols"] = [
        rename_map[column] for column in effects["placebo_cols"]
    ]
    renamed["model_features"] = [
        rename_map.get(column, column) for column in effects["model_features"]
    ]
    return renamed


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Inject a uniquely discoverable conditional-error flag.

    Regression targets are rewritten inside Q05--Q95 when feasible, with
    deterministic fallbacks that never exceed the clean observed range.  For
    a binary target, high-confidence OOF negatives are flipped to the original
    positive-class value.  A trial is returned only when the final, blind
    cross-fitted detector gives the planted indicator directed ROC-AUC >= 0.90
    and a >= 0.20 advantage over every other 0/1 candidate, including
    same-prevalence placebos.
    """
    target_col = str(params["outcome_col"])
    target = _target_values(df, target_col)
    n = len(df)
    bad_fraction = float(params.get("bad_fraction", 0.05))
    if not (0.0 < bad_fraction < 0.25):
        raise ValueError("bad_fraction must be between 0 and 0.25")
    n_bad = max(3, int(round(bad_fraction * n)))
    if 2 * n_bad >= n:
        raise InjectionRejected("too few clean rows for the requested bad fraction")

    n_placebos = int(params.get("n_placebos", 3))
    if not (1 <= n_placebos <= 10):
        raise ValueError("n_placebos must be between 1 and 10")
    max_attempts = int(params.get("max_attempts", 12))
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    config = _DetectorConfig.from_mapping(params)
    if config.n_splits < 3 or config.n_estimators < 16:
        raise ValueError("n_splits must be >= 3 and n_estimators must be >= 16")
    if not (0.5 <= config.min_indicator_auc <= 1.0):
        raise ValueError("min_indicator_auc must be in [0.5, 1.0]")
    if not (0.0 <= config.min_auc_margin <= 0.5):
        raise ValueError("min_auc_margin must be in [0.0, 0.5]")

    id_no_cols = set(str(value) for value in params.get("id_no_cols", []) or [])
    id_no_cols |= id_like_cols(df)
    cv_seed = int(rng.integers(0, 2**31 - 1))
    clean_profile = _oof_profile(df, target_col, id_no_cols, config, cv_seed)
    target_is_binary = clean_profile.target_is_binary

    pool_fraction = float(params.get("clean_pool_fraction", 0.5))
    if not (0.0 < pool_fraction <= 1.0):
        raise ValueError("clean_pool_fraction must be in (0, 1]")
    pool_size = max(n_bad, int(np.floor(pool_fraction * n)))
    clean_pool = np.argsort(
        clean_profile.residual_scores,
        kind="mergesort",
    )[:pool_size]

    original_min = float(np.min(target))
    original_max = float(np.max(target))
    allowed_low, allowed_high = (
        float(value) for value in np.quantile(target, [0.05, 0.95])
    )
    allowed_quantile_low: float | None = 0.05
    allowed_quantile_high: float | None = 0.95
    integer_target = bool(np.allclose(target, np.rint(target)))

    amplitude: float | None = None
    proposals: dict[int, float] = {}
    if target_is_binary:
        target_classes = np.unique(target)
        negative_class = float(target_classes[0])
        positive_class = float(target_classes[-1])
        clean_pool = clean_pool[target[clean_pool] == negative_class]
        if len(clean_pool) < n_bad:
            raise InjectionRejected(
                f"need {n_bad} high-confidence negative rows, found {len(clean_pool)}"
            )
        allowed_low, allowed_high = original_min, original_max
        allowed_quantile_low = None
        allowed_quantile_high = None
    else:
        residual_scale = _mad_std(clean_profile.signed_residuals)
        residual_q99 = float(np.quantile(clean_profile.residual_scores, 0.99))
        amplitude = max(4.0 * residual_scale, 1.25 * residual_q99)
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            raise InjectionRejected("clean residual scale collapsed")
        # Prefer the central support so planted values remain ordinary in
        # isolation.  Heavy-tailed but otherwise suitable targets (notably KC
        # house prices) may need a wider *observed* support band to accommodate
        # the q99-derived residual amplitude.  The first feasible band wins;
        # values are never allowed beyond the clean minimum/maximum.
        support_bands: list[tuple[float | None, float | None, float, float]] = []
        for quantile_low, quantile_high in ((0.05, 0.95), (0.02, 0.98), (0.01, 0.99)):
            band_low, band_high = (
                float(value)
                for value in np.quantile(target, [quantile_low, quantile_high])
            )
            support_bands.append(
                (quantile_low, quantile_high, band_low, band_high)
            )
        support_bands.append((None, None, original_min, original_max))

        proposals = {}
        best_infeasible: dict[int, float] = {}
        best_infeasible_band = (allowed_quantile_low, allowed_quantile_high, allowed_low, allowed_high)
        for quantile_low, quantile_high, band_low, band_high in support_bands:
            candidate_proposals = _continuous_proposals(
                target,
                clean_profile,
                clean_pool,
                band_low,
                band_high,
                amplitude,
                integer_target,
                df[target_col].dtype,
            )
            if len(candidate_proposals) > len(best_infeasible):
                best_infeasible = candidate_proposals
                best_infeasible_band = (
                    quantile_low,
                    quantile_high,
                    band_low,
                    band_high,
                )
            if len(candidate_proposals) < n_bad:
                continue
            proposals = candidate_proposals
            allowed_low, allowed_high = band_low, band_high
            allowed_quantile_low = quantile_low
            allowed_quantile_high = quantile_high
            break
        if not proposals:
            proposals = best_infeasible
            (
                allowed_quantile_low,
                allowed_quantile_high,
                allowed_low,
                allowed_high,
            ) = best_infeasible_band
        clean_pool = np.asarray(sorted(proposals), dtype=int)
        if len(clean_pool) < n_bad:
            raise InjectionRejected(
                f"observed target range [{original_min:.6g}, {original_max:.6g}] "
                f"cannot support {n_bad} required conditional corruptions at "
                f"residual amplitude {amplitude:.6g}; the best tested band "
                f"[{allowed_low:.6g}, {allowed_high:.6g}] supports only "
                f"{len(clean_pool)}"
            )

    last_error = "no candidate trials"
    for _ in range(max_attempts):
        bad_positions = np.sort(rng.choice(clean_pool, size=n_bad, replace=False))
        corrupted = df.copy()
        if target_is_binary:
            positive_positions = np.flatnonzero(target == positive_class)
            if positive_positions.size == 0:
                raise InjectionRejected("binary target has no positive-class value")
            positive_value = df[target_col].iloc[int(positive_positions[0])]
            replacement = corrupted[target_col].copy()
            replacement.iloc[bad_positions] = positive_value
            corrupted[target_col] = replacement.astype(df[target_col].dtype)
            corruption_mode = "high_confidence_negative_to_positive"
        else:
            new_target = target.copy()
            for position in bad_positions:
                new_target[position] = proposals[int(position)]
            replacement = pd.Series(new_target, index=corrupted.index).astype(
                df[target_col].dtype
            )
            corrupted[target_col] = replacement
            corruption_mode = "bounded_conditional_residual"

        trial, indicator_col, placebo_cols = _add_indicator_and_placebos(
            corrupted,
            bad_positions,
            n_placebos,
            rng,
        )
        try:
            final_profile, scores = _candidate_scores(
                trial,
                target_col,
                indicator_col,
                bad_positions,
                n_bad,
                id_no_cols,
                config,
                cv_seed,
            )
        except InjectionRejected as exc:
            last_error = str(exc)
            continue
        if not _trial_is_strong(scores, config):
            hybrid_detail = ""
            if scores.hybrid_score is not None:
                hybrid_detail = (
                    f", hybrid={scores.hybrid_score:.4f}, "
                    f"hybrid runner-up={scores.runner_up_hybrid:.4f}"
                )
            last_error = (
                f"indicator AUC={scores.indicator_auc:.4f}, "
                f"runner-up={scores.runner_up_auc:.4f}, "
                f"margin={scores.auc_margin:.4f}{hybrid_detail}"
            )
            continue

        effects: dict[str, Any] = {
            "contract_version": 2,
            "indicator_col": indicator_col,
            "placebo_cols": placebo_cols,
            "n_bad_rows": int(n_bad),
            "bad_fraction": float(bad_fraction),
            "bad_indices": [int(value) for value in bad_positions],
            "bad_row_ids": (
                sorted(
                    int(value)
                    for value in coerce_numeric(df["row_id"]).to_numpy()[bad_positions]
                )
                if "row_id" in df.columns
                else []
            ),
            "corruption_mode": corruption_mode,
            "target_is_binary": bool(target_is_binary),
            "target_support": {
                "original_min": original_min,
                "original_max": original_max,
                "allowed_low": allowed_low,
                "allowed_high": allowed_high,
                "quantile_low": allowed_quantile_low,
                "quantile_high": allowed_quantile_high,
            },
            "residual_amplitude": amplitude,
            "cv_seed": int(cv_seed),
            "model_features": list(final_profile.raw_features),
            "id_no_cols": sorted(id_no_cols),
            "detector_config": {
                "n_splits": config.n_splits,
                "n_estimators": config.n_estimators,
                "min_samples_leaf": config.min_samples_leaf,
                "max_categories": config.max_categories,
                "min_indicator_auc": config.min_indicator_auc,
                "min_auc_margin": config.min_auc_margin,
                "min_binary_hybrid": config.min_binary_hybrid,
                "min_binary_hybrid_margin": config.min_binary_hybrid_margin,
            },
            "injection_metrics": {
                "indicator_auc": scores.indicator_auc,
                "runner_up_auc": scores.runner_up_auc,
                "auc_margin": scores.auc_margin,
                "top_residual_overlap": scores.overlap,
                "binary_hybrid": scores.hybrid_score,
                "runner_up_binary_hybrid": scores.runner_up_hybrid,
            },
        }

        renamed_df, rename_map = _rename_columns(
            trial,
            target_col,
            id_no_cols,
            rng,
        )
        effects = _renamed_effects(effects, rename_map)
        result = validate(renamed_df, effects, target_col)
        if result.passed:
            return renamed_df, {
                "type": _PHENOMENON_NAME,
                "params": params,
                "effects": effects,
            }
        last_error = last_failed_detail(result)

    raise InjectionRejected(
        f"no bounded conditional-error flag survived validation; last error: {last_error}"
    )


def _failure(name: str, detail: str) -> ValidationResult:
    return ValidationResult(
        passed=False,
        checks=[
            CheckDetail(
                name=name,
                passed=False,
                metric=0.0,
                threshold=1.0,
                detail=detail,
            )
        ],
    )


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    """Validate bounded support and blind residual-based flag uniqueness."""
    try:
        indicator_col = str(effects["indicator_col"])
        placebo_cols = [str(value) for value in effects.get("placebo_cols", [])]
        n_bad = int(effects["n_bad_rows"])
        bad_positions = np.asarray(effects["bad_indices"], dtype=int)
        cv_seed = int(effects["cv_seed"])
        config = _DetectorConfig.from_mapping(effects.get("detector_config"))
        id_no_cols = set(str(value) for value in effects.get("id_no_cols", []) or [])
        support = effects["target_support"]
        original_min = float(support["original_min"])
        original_max = float(support["original_max"])
        allowed_low = float(support["allowed_low"])
        allowed_high = float(support["allowed_high"])
    except (KeyError, TypeError, ValueError) as exc:
        return _failure("effects_contract", f"invalid v1 effects: {exc}")

    checks: list[CheckDetail] = []
    present = indicator_col in df.columns
    checks.append(
        CheckDetail(
            name="indicator_exists",
            passed=present,
            metric=1.0 if present else 0.0,
            threshold=1.0,
            detail=f"Indicator column {indicator_col!r} {'found' if present else 'missing'}",
        )
    )
    if not present:
        return ValidationResult(passed=False, checks=checks)

    preserve = {target_col, "row_id"} | id_no_cols
    anonymised_candidates = [str(column) for column in df.columns if column not in preserve]
    short_random = sum(
        len(column) == 5 and column.isalpha() and column.islower()
        for column in anonymised_candidates
    )
    rename_ratio = short_random / max(len(anonymised_candidates), 1)
    rename_ok = rename_ratio > 0.5
    checks.append(
        CheckDetail(
            name="columns_renamed",
            passed=rename_ok,
            metric=float(rename_ratio),
            threshold=0.5,
            detail=(
                f"{short_random}/{len(anonymised_candidates)} non-preserved "
                "columns have random five-letter names"
            ),
        )
    )
    if not rename_ok:
        return ValidationResult(passed=False, checks=checks)

    if len(bad_positions) != n_bad or len(np.unique(bad_positions)) != n_bad:
        checks.append(
            CheckDetail(
                name="bad_indices",
                passed=False,
                metric=float(len(np.unique(bad_positions))),
                threshold=float(n_bad),
                detail="bad_indices do not contain the expected unique positions",
            )
        )
        return ValidationResult(passed=False, checks=checks)
    if np.any(bad_positions < 0) or np.any(bad_positions >= len(df)):
        return ValidationResult(
            passed=False,
            checks=checks
            + [
                CheckDetail(
                    name="bad_indices",
                    passed=False,
                    metric=0.0,
                    threshold=1.0,
                    detail="bad_indices contain out-of-range positions",
                )
            ],
        )

    candidate_columns = _zero_one_columns(df, target_col, id_no_cols)
    flag_shape_ok = indicator_col in candidate_columns
    if flag_shape_ok:
        flag_sum = int(coerce_numeric(df[indicator_col]).sum())
        flag_shape_ok = flag_sum == n_bad
    else:
        flag_sum = 0
    placebo_shape_ok = len(placebo_cols) >= 1 and all(
        column in candidate_columns
        and int(coerce_numeric(df[column]).sum()) == n_bad
        for column in placebo_cols
    )
    shape_ok = flag_shape_ok and placebo_shape_ok
    checks.append(
        CheckDetail(
            name="same_prevalence_flags",
            passed=shape_ok,
            metric=float(flag_sum),
            threshold=float(n_bad),
            detail=(
                f"Indicator has {flag_sum} ones; {len(placebo_cols)} placebo "
                f"flags must each have {n_bad} ones"
            ),
        )
    )
    if not shape_ok:
        return ValidationResult(passed=False, checks=checks)

    # Reconstruct the planted rows from the observable flag.  The recorded
    # positions are checked only as manifest integrity; they never define the
    # residual anomaly label used by the validator.
    flagged_positions = np.flatnonzero(
        coerce_numeric(df[indicator_col]).to_numpy(dtype=float) == 1.0
    )
    positions_match = bool(
        np.array_equal(np.sort(bad_positions), flagged_positions)
    )
    checks.append(
        CheckDetail(
            name="indicator_positions_match_effects",
            passed=positions_match,
            metric=float(np.intersect1d(bad_positions, flagged_positions).size),
            threshold=float(n_bad),
            detail=(
                "The observable indicator positions must match the recorded "
                "injection positions"
            ),
        )
    )
    if not positions_match:
        return ValidationResult(passed=False, checks=checks)
    bad_positions = flagged_positions

    try:
        target = _target_values(df, target_col)
    except InjectionRejected as exc:
        return ValidationResult(
            passed=False,
            checks=checks
            + [
                CheckDetail(
                    name="target_support",
                    passed=False,
                    metric=0.0,
                    threshold=1.0,
                    detail=str(exc),
                )
            ],
        )
    tolerance = max(1e-12, (original_max - original_min) * 1e-12)
    global_support_ok = bool(
        np.all((target >= original_min - tolerance) & (target <= original_max + tolerance))
    )
    planted = target[bad_positions]
    planted_support_ok = bool(
        np.all((planted >= allowed_low - tolerance) & (planted <= allowed_high + tolerance))
    )
    support_ok = global_support_ok and planted_support_ok
    checks.append(
        CheckDetail(
            name="bounded_target_support",
            passed=support_ok,
            metric=float(np.max(planted)),
            threshold=float(allowed_high),
            detail=(
                f"Injected targets span [{planted.min():.6g}, {planted.max():.6g}] "
                f"inside allowed [{allowed_low:.6g}, {allowed_high:.6g}]; "
                f"original support [{original_min:.6g}, {original_max:.6g}]"
            ),
        )
    )
    if not support_ok:
        return ValidationResult(passed=False, checks=checks)

    try:
        profile, scores = _candidate_scores(
            df,
            target_col,
            indicator_col,
            bad_positions,
            n_bad,
            id_no_cols,
            config,
            cv_seed,
        )
    except (InjectionRejected, TypeError, ValueError) as exc:
        return ValidationResult(
            passed=False,
            checks=checks
            + [
                CheckDetail(
                    name="blind_residual_detector",
                    passed=False,
                    metric=0.0,
                    threshold=1.0,
                    detail=str(exc),
                )
            ],
        )

    expected_features = tuple(str(value) for value in effects.get("model_features", []))
    feature_contract_ok = expected_features == profile.raw_features
    checks.append(
        CheckDetail(
            name="observable_model_features",
            passed=feature_contract_ok,
            metric=float(len(profile.raw_features)),
            threshold=float(len(expected_features)),
            detail=(
                f"Blind detector used {list(profile.raw_features)}; "
                f"expected {list(expected_features)}"
            ),
        )
    )
    if not feature_contract_ok:
        return ValidationResult(passed=False, checks=checks)

    auc_ok = scores.indicator_auc >= config.min_indicator_auc
    checks.append(
        CheckDetail(
            name="indicator_residual_auc",
            passed=auc_ok,
            metric=float(scores.indicator_auc),
            threshold=float(config.min_indicator_auc),
            detail=(
                f"Indicator directed AUC for top-{n_bad} OOF residuals = "
                f"{scores.indicator_auc:.4f}; overlap={scores.overlap}/{n_bad}"
            ),
        )
    )
    if not auc_ok:
        return ValidationResult(passed=False, checks=checks)

    margin_ok = scores.auc_margin >= config.min_auc_margin
    checks.append(
        CheckDetail(
            name="indicator_auc_margin",
            passed=margin_ok,
            metric=float(scores.auc_margin),
            threshold=float(config.min_auc_margin),
            detail=(
                f"Indicator AUC={scores.indicator_auc:.4f}, best competing "
                f"0/1 column={scores.runner_up_auc:.4f}"
            ),
        )
    )
    if not margin_ok:
        return ValidationResult(passed=False, checks=checks)

    if profile.target_is_binary:
        hybrid_score = float(scores.hybrid_score or 0.0)
        runner_up = float(scores.runner_up_hybrid or 0.0)
        hybrid_ok = (
            hybrid_score >= config.min_binary_hybrid
            and hybrid_score - runner_up >= config.min_binary_hybrid_margin
        )
        checks.append(
            CheckDetail(
                name="binary_target_hybrid_margin",
                passed=hybrid_ok,
                metric=hybrid_score - runner_up,
                threshold=float(config.min_binary_hybrid_margin),
                detail=(
                    f"Indicator lift/correlation hybrid={hybrid_score:.4f}, "
                    f"runner-up={runner_up:.4f}; absolute minimum="
                    f"{config.min_binary_hybrid:.4f}"
                ),
            )
        )
        if not hybrid_ok:
            return ValidationResult(passed=False, checks=checks)

    return ValidationResult(passed=True, checks=checks)


def compute_answer_v1(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the anonymised 0/1 column that identifies conditional errors."""
    del df, slot_assignments
    return effects["indicator_col"]


PHENOMENON = Phenomenon(
    name=_PHENOMENON_NAME,
    inject=inject,
    validate=validate,
    compute_answers={_TEMPLATE_ID: compute_answer_v1},
    summary_fields=("id_no",),
)
