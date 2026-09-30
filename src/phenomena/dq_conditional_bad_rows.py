"""Conditional bad-rows phenomenon.

Rewrites typical feature-space rows so their targets remain inside the global
5th-95th percentile band but have large residuals relative to similar rows.
The ground-truth answer is the injected ``row_id`` values. Validation can
optionally require dominance over naturally occurring conditional outliers.
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
)

from ._base import (
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
    last_failed_detail,
)


def _mad_std(values: np.ndarray) -> float:
    """Robust scale = 1.4826 * median(|x - median(x)|)."""
    med = float(np.median(values))
    return float(np.median(np.abs(values - med))) * 1.4826


def _standardize(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Z-score the feature matrix (NaN-filled to column mean) for distances."""
    X = df[features].apply(coerce_numeric).to_numpy(dtype=float, copy=True)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    return (X - X.mean(axis=0)) / std


def _knn_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int
) -> np.ndarray:
    """m_hat(x) ~ E[Y | features] via mean of k nearest training neighbours."""
    from sklearn.neighbors import NearestNeighbors

    k = max(1, min(k, len(X_train)))
    nn = NearestNeighbors(n_neighbors=k).fit(X_train)
    _, idx = nn.kneighbors(X_query)
    return y_train[idx].mean(axis=1)


def _smoother_features(df: pd.DataFrame, outcome_col: str, id_no_cols: set[str]) -> list[str]:
    return eligible_numeric_columns(
        df,
        exclude=id_no_cols | {outcome_col},
        min_unique=3,
    )


def _knn_k(n: int) -> int:
    return max(10, min(30, n // 20))


def inject(
    df: pd.DataFrame,
    params: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Plant conditional outliers in dense feature-space rows.

    Requires ``row_id`` and at least two eligible smoother features. Candidate
    rows are accepted only when validation confirms that they remain globally
    in-band while being conditionally anomalous.
    """
    df = df.copy()
    outcome_col = params["outcome_col"]
    bad_fraction = float(params.get("bad_fraction", 0.02))
    tau = float(params.get("tau", 6.0))
    central_frac = float(params.get("central_frac", 0.5))
    z_thresh = float(params.get("z_thresh", 4.0))
    gap_factor = float(params.get("gap_factor", 1.5))
    require_dominance = str(params.get("require_dominance", "false")).lower() in {"1", "true", "yes"}
    max_bad = int(params.get("max_bad", 10))
    id_no_cols = set(params.get("id_no_cols", []) or []) | id_like_cols(df)

    if "row_id" not in df.columns:
        raise InjectionRejected("dataset has no row_id column")

    n = len(df)
    m = min(max_bad, max(3, int(round(bad_fraction * n))))

    features = _smoother_features(df, outcome_col, id_no_cols)
    if len(features) < 2:
        raise InjectionRejected("Need at least 2 smoother features")
    k = _knn_k(n)

    y_series = coerce_numeric(df[outcome_col])
    y = y_series.to_numpy(dtype=float)
    y_dp = decimal_places(y_series)
    q05, q95 = (float(v) for v in np.quantile(y, [0.05, 0.95]))
    row_id = df["row_id"].to_numpy()

    Xs = _standardize(df, features)
    center = np.median(Xs, axis=0)
    dist_to_center = np.sqrt(((Xs - center) ** 2).sum(axis=1))
    core = np.argsort(dist_to_center)[: max(m, int(central_frac * n))]

    last_error = "no attempts"
    for _ in range(8):
        S = rng.choice(core, size=m, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[S] = False

        m_hat = _knn_predict(Xs[train_mask], y[train_mask], Xs, k)
        sigma_res = _mad_std(y[train_mask] - m_hat[train_mask])
        if sigma_res <= 1e-9 or not np.isfinite(sigma_res):
            last_error = "Zero residual scale"
            continue

        new_y = y.copy()
        for i in S:
            mh = float(m_hat[i])
            s = 1.0 if (q95 - mh) >= (mh - q05) else -1.0
            new_y[i] = min(max(mh + s * tau * sigma_res, q05), q95)

        trial = pd.Series(new_y, index=df.index).round(y_dp)
        effects = {
            "smoother_features": [str(c) for c in features],
            "k": int(k),
            "bad_row_ids": sorted(int(r) for r in row_id[S]),
            # Surfaced to both question variants via the {N_INJECT_SAMPLES} placeholder
            # (filled post-injection by phenomena_pipeline._fill_effect_placeholders).
            "n_inject_samples": int(m),
            "tau": float(tau),
            "z_thresh": float(z_thresh),
            "gap_factor": float(gap_factor),
            "require_dominance": bool(require_dominance),
            "sigma_res": float(sigma_res),
            "id_no_cols": sorted(id_no_cols),
        }
        trial_df = df.assign(**{outcome_col: trial})
        result = validate(trial_df, effects, outcome_col)
        if result.passed:
            out = df.copy()
            out[outcome_col] = trial
            return out, {
                "type": "dq_conditional_bad_rows",
                "params": params,
                "effects": effects,
            }
        last_error = last_failed_detail(result)

    raise InjectionRejected(f"No candidate survived validation. Last error: {last_error}")


def validate(df: pd.DataFrame, effects: dict, target_col: str) -> ValidationResult:
    features = [c for c in effects["smoother_features"] if c in df.columns]
    bad_ids = list(effects["bad_row_ids"])
    k = int(effects["k"])
    z_thresh = float(effects["z_thresh"])
    gap_factor = float(effects["gap_factor"])
    require_dominance = bool(effects.get("require_dominance", False))
    checks: list[CheckDetail] = []

    if "row_id" not in df.columns or len(features) < 2:
        checks.append(CheckDetail(
            name="columns_present", passed=False, metric=0.0, threshold=1.0,
            detail="Missing 'row_id' or too few smoother features",
        ))
        return ValidationResult(passed=False, checks=checks)

    y = coerce_numeric(df[target_col]).to_numpy(dtype=float)
    row_id = df["row_id"].to_numpy()
    S_mask = np.isin(row_id, bad_ids)
    if int(S_mask.sum()) != len(bad_ids):
        checks.append(CheckDetail(
            name="injected_rows_present", passed=False,
            metric=float(S_mask.sum()), threshold=float(len(bad_ids)),
            detail=f"Found {int(S_mask.sum())} of {len(bad_ids)} injected row_ids",
        ))
        return ValidationResult(passed=False, checks=checks)

    q05, q95 = (float(v) for v in np.quantile(y, [0.05, 0.95]))
    tol = (q95 - q05) * 0.02
    inj_y = y[S_mask]
    in_band = bool(np.all((inj_y >= q05 - tol) & (inj_y <= q95 + tol)))
    checks.append(CheckDetail(
        name="not_global_outliers", passed=in_band,
        metric=float(np.max(np.abs(inj_y - np.median(y)))), threshold=float(q95 - q05),
        detail=f"Injected Y in [{inj_y.min():.4g}, {inj_y.max():.4g}], band [{q05:.4g}, {q95:.4g}]",
    ))
    if not in_band:
        return ValidationResult(passed=False, checks=checks)

    Xs = _standardize(df, features)
    train_mask = ~S_mask
    m_hat = _knn_predict(Xs[train_mask], y[train_mask], Xs, k)
    sigma_res = _mad_std(y[train_mask] - m_hat[train_mask])
    if sigma_res <= 1e-9 or not np.isfinite(sigma_res):
        checks.append(CheckDetail(
            name="conditional_anomaly", passed=False, metric=0.0, threshold=z_thresh,
            detail="Residual scale collapsed to zero",
        ))
        return ValidationResult(passed=False, checks=checks)

    z = np.abs(y - m_hat) / sigma_res
    min_inj_z = float(z[S_mask].min())
    max_other_z = float(z[~S_mask].max())
    cond_ok = min_inj_z >= z_thresh
    checks.append(CheckDetail(
        name="conditional_anomaly", passed=cond_ok, metric=min_inj_z, threshold=z_thresh,
        detail=f"Weakest injected residual-z = {min_inj_z:.4f} (need >= {z_thresh}); "
               f"strongest natural z = {max_other_z:.4f}",
    ))
    if not cond_ok:
        return ValidationResult(passed=False, checks=checks)

    if require_dominance:
        dom_ok = min_inj_z >= gap_factor * max_other_z
        checks.append(CheckDetail(
            name="anomaly_dominance", passed=dom_ok, metric=min_inj_z,
            threshold=gap_factor * max_other_z,
            detail=f"Weakest injected z = {min_inj_z:.4f} vs {gap_factor}x natural "
                   f"max ({max_other_z:.4f})",
        ))
        if not dom_ok:
            return ValidationResult(passed=False, checks=checks)

    return ValidationResult(passed=True, checks=checks)


def compute_answer_v0(df: pd.DataFrame, slot_assignments: dict, effects: dict) -> Any:
    """Return the injected row_id values as a comma-separated list (ascending)."""
    return ", ".join(str(r) for r in sorted(effects["bad_row_ids"]))


PHENOMENON = Phenomenon(
    name="dq_conditional_bad_rows",
    inject=inject,
    validate=validate,
    compute_answers={
        "dq_conditional_bad_rows_v0": compute_answer_v0,
    },
    summary_fields=("id_no",),
)
