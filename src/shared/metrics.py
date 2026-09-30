"""Shared numeric and metric helpers used by injectors and validators.

These are pure, stateless computations on pandas Series and DataFrames.
They provide the standard numeric-coercion and identifier-detection policies
and measure statistical properties of feature→outcome relationships:
group comparisons, monotonicity, non-monotone peaks, heteroskedasticity,
binned importance, plateau shape, and interaction strength.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, ttest_ind

from shared.missing_labels import is_missing_like


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values, coercing invalid values to NaN."""
    return pd.to_numeric(series, errors="coerce")


def decimal_places(series: pd.Series) -> int:
    """Return the median number of decimal places in a numeric series."""
    dp_counts = []
    for v in series.dropna():
        s = str(float(v))
        if "." in s:
            dp_counts.append(len(s.split(".")[1]))
        else:
            dp_counts.append(0)
    return int(np.median(dp_counts)) if dp_counts else 2


def id_like_cols(df: pd.DataFrame) -> set[str]:
    """Return columns that should be treated as identifiers.

    This includes columns named like identifiers and integer-valued columns
    whose values are all distinct, such as a row index misclassified as a
    regression feature.
    """
    n = len(df)
    out: set[str] = set()
    for column in df.columns:
        if str(column).lower() in {"row_id", "id", "index", "uid", "uuid"}:
            out.add(str(column))
            continue
        values = coerce_numeric(df[column])
        if (
            n > 0
            and bool(values.notna().all())
            and values.nunique() == n
            and bool((values == values.round()).all())
        ):
            out.add(str(column))
    return out


def eligible_numeric_columns(
    df: pd.DataFrame,
    *,
    exclude: set[str] | tuple[str, ...] | list[str] = (),
    min_unique: int = 0,
) -> list[str]:
    """Return numeric columns that satisfy the shared feature policy."""
    excluded = set(exclude)
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded and df[column].nunique() >= min_unique
    ]


def safe_std(values: pd.Series, fallback: float = 1.0) -> float:
    """Return pandas' sample standard deviation, replacing zero/NaN values."""
    value = float(values.std())
    return fallback if value == 0 or np.isnan(value) else value


def clean_categorical_features(
    df: pd.DataFrame,
    categorical_cols: list[str],
    min_cat: int,
    max_cat: int,
    exclude_missing_like: bool = True,
) -> list[str]:
    """Return complete categorical columns whose cardinality is in range.

    Missing columns, columns containing NaN cells, and columns outside the
    inclusive ``[min_cat, max_cat]`` range are excluded. By default, columns
    containing a missing-like category label are excluded as well; callers
    that anonymise every category can retain them by setting
    ``exclude_missing_like=False``.
    """
    out: list[str] = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if bool(series.isna().any()):
            continue
        uniques = series.unique().tolist()
        if not (min_cat <= len(uniques) <= max_cat):
            continue
        if exclude_missing_like and any(is_missing_like(value) for value in uniques):
            continue
        out.append(str(col))
    return out


def _finite_values(values: np.ndarray) -> np.ndarray:
    """Return a one-dimensional float array containing only finite values."""
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def welch_t_test(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:
    """Return Welch's two-sample t statistic and two-sided p-value.

    The statistic's sign follows ``mean(sample_a) - mean(sample_b)``. Non-finite
    values are ignored. Comparisons with fewer than two finite observations per
    sample, or non-finite SciPy results, use the conservative ``(0.0, 1.0)``
    fallback for the affected values.
    """
    sample_a = _finite_values(sample_a)
    sample_b = _finite_values(sample_b)
    if len(sample_a) < 2 or len(sample_b) < 2:
        return 0.0, 1.0

    result = ttest_ind(sample_a, sample_b, equal_var=False)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(statistic):
        statistic = 0.0
    if not np.isfinite(p_value):
        p_value = 1.0
    return statistic, p_value


def two_proportion_z_test(
    sample_a: np.ndarray, sample_b: np.ndarray
) -> tuple[float, float]:
    """Return the pooled two-proportion z statistic and two-sided p-value.

    Samples contain binary 0/1 observations, and the statistic's sign follows
    ``mean(sample_a) - mean(sample_b)``. Non-finite values are ignored.
    """
    sample_a = _finite_values(sample_a)
    sample_b = _finite_values(sample_b)
    if len(sample_a) < 2 or len(sample_b) < 2:
        return 0.0, 1.0

    n_a, n_b = len(sample_a), len(sample_b)
    proportion_a = float(sample_a.mean())
    proportion_b = float(sample_b.mean())
    pooled = float(sample_a.sum() + sample_b.sum()) / (n_a + n_b)
    standard_error = math.sqrt(
        pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b)
    )
    if standard_error <= 0:
        return 0.0, 1.0

    statistic = (proportion_a - proportion_b) / standard_error
    p_value = 2.0 * (1.0 - float(norm.cdf(abs(statistic))))
    return float(statistic), float(p_value)


def welch_mean_difference_ci(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float]:
    """Return a Welch interval for ``mean(sample_b) - mean(sample_a)``.

    Returns ``(difference, ci_low, ci_high, width)`` and requires at least two
    finite observations per sample. SciPy's result is computed with the sample
    order reversed so its confidence interval has the documented direction.

    SciPy returns NaN interval bounds when both samples have zero standard
    error. That degenerate case is represented as a zero-width point interval,
    matching the mathematically determined mean difference.
    """
    sample_a = _finite_values(sample_a)
    sample_b = _finite_values(sample_b)
    if len(sample_a) < 2 or len(sample_b) < 2:
        raise ValueError("Welch confidence interval requires two values per sample")

    difference = float(np.mean(sample_b) - np.mean(sample_a))
    result = ttest_ind(sample_b, sample_a, equal_var=False)
    interval = result.confidence_interval(confidence_level=confidence_level)
    ci_low = float(interval.low)
    ci_high = float(interval.high)

    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        standard_error_squared = (
            float(np.var(sample_a, ddof=1)) / len(sample_a)
            + float(np.var(sample_b, ddof=1)) / len(sample_b)
        )
        if standard_error_squared <= 0:
            return difference, difference, difference, 0.0

    return difference, ci_low, ci_high, ci_high - ci_low


def _quantile_bins(
    feature_vals: pd.Series,
    n_bins: int,
) -> pd.Series | None:
    """Return quantile-bin labels, or ``None`` when qcut cannot form bins."""
    try:
        return pd.qcut(feature_vals, q=n_bins, duplicates="drop")
    except ValueError:
        return None


def _quantile_bin_means(
    feature_vals: pd.Series,
    outcome_vals: pd.Series,
    n_bins: int,
) -> pd.Series | None:
    """Return sorted outcome means for quantile bins of a feature."""
    bins = _quantile_bins(feature_vals, n_bins)
    if bins is None:
        return None
    return outcome_vals.groupby(bins, observed=True).mean().sort_index()


def nonmonotone_score(feature_vals: pd.Series, outcome_vals: pd.Series) -> float:
    """Measure how non-monotone the feature→outcome relationship is.

    Bins the feature into 5 quantile bins, computes mean outcome per
    bin, and returns: max(interior_bin_means) - min(edge_bin_means).
    A high score means the peak is in the interior — inverted-U shaped.
    """
    bin_means = _quantile_bin_means(feature_vals, outcome_vals, 5)
    if bin_means is None:
        return 0.0
    if len(bin_means) < 3:
        return 0.0
    interior_max = bin_means.iloc[1:-1].max()
    edge_min = min(bin_means.iloc[0], bin_means.iloc[-1])
    return float(interior_max - edge_min)


def spearman_bin_rho(
    feature_vals: pd.Series,
    outcome_vals: pd.Series,
    n_bins: int = 5,
) -> float:
    """Measure monotonicity via Spearman rho of binned means.

    Bins *feature_vals* into *n_bins* quantiles, computes the mean of
    *outcome_vals* within each bin, and returns the Spearman rank
    correlation between bin index and bin mean.  Returns 1.0 when
    binning fails (conservative: treat as monotone).
    """
    bin_means = _quantile_bin_means(feature_vals, outcome_vals, n_bins)
    if bin_means is None:
        return 1.0
    if len(bin_means) < 3:
        return 1.0
    rho, _ = spearmanr(range(len(bin_means)), bin_means.values)
    if np.isnan(rho):
        return 1.0
    return float(rho)


def has_direction_reversal(
    feature_vals: pd.Series,
    outcome_vals: pd.Series,
    n_bins: int = 5,
) -> bool:
    """Return True if bin-means change direction at least once."""
    bin_means = _quantile_bin_means(feature_vals, outcome_vals, n_bins)
    if bin_means is None:
        return False
    if len(bin_means) < 3:
        return False
    diffs = bin_means.diff().dropna().values
    signs = np.sign(diffs)
    for i in range(len(signs) - 1):
        if signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]:
            return True
    return False


def binned_importance(
    feature_vals: pd.Series, outcome_vals: pd.Series, n_bins: int = 5
) -> float:
    """Compute a non-linear predictive-power proxy for a feature.

    Bins the feature into quantiles and returns the variance of the
    bin-wise outcome means.  This captures non-linear relationships
    that Pearson correlation misses.
    """
    bin_means = _quantile_bin_means(feature_vals, outcome_vals, n_bins)
    if bin_means is None:
        return 0.0
    if len(bin_means) < 2:
        return 0.0
    return float(bin_means.var())


def heteroskedasticity_score(
    feature_vals: pd.Series, outcome_vals: pd.Series, n_bins: int = 5
) -> float:
    """Measure heteroskedasticity: ratio of max to min within-bin variance.

    A score of 1.0 means perfectly homoskedastic; higher means the
    outcome variance changes across bins of the feature.
    """
    bins = _quantile_bins(feature_vals, n_bins)
    if bins is None:
        return 1.0
    bin_vars = outcome_vals.groupby(bins, observed=True).var().dropna()
    if len(bin_vars) < 2:
        return 1.0
    min_var = float(bin_vars.min())
    if min_var < 1e-10:
        min_var = 1e-10
    return float(bin_vars.max() / min_var)


def plateau_score(
    feature_vals: pd.Series, outcome_vals: pd.Series, n_bins: int = 6
) -> float:
    """Measure how much a feature→outcome relationship looks like a plateau.

    A plateau has an *increasing* first half and a *flat* second half.
    Returns  first_half_trend / second_half_std.
    Returns 0.0 for non-plateau patterns (e.g. decreasing or noisy).
    """
    bin_means = _quantile_bin_means(feature_vals, outcome_vals, n_bins)
    if bin_means is None:
        return 0.0
    if len(bin_means) < 4:
        return 0.0

    mid = len(bin_means) // 2
    first_half = bin_means.iloc[:mid].values.astype(float)
    second_half = bin_means.iloc[mid:].values.astype(float)

    first_trend = first_half[-1] - first_half[0]
    if first_trend <= 0:
        return 0.0

    second_std = float(np.std(second_half))
    if second_std < 1e-10:
        return float("inf")

    return float(first_trend / second_std)


def cohens_d(group1: pd.Series, group0: pd.Series) -> float:
    """Compute Cohen's d between two groups."""
    n1, n0 = len(group1), len(group0)
    if n1 < 2 or n0 < 2:
        return 0.0
    m1, m0 = group1.mean(), group0.mean()
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.var() + (n0 - 1) * group0.var()) / (n1 + n0 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float(abs(m1 - m0) / pooled_std)


def xor_signal(col_a: pd.Series, col_b: pd.Series) -> pd.Series:
    """Compute a centred XOR interaction signal from two numeric columns."""
    med_a = col_a.astype(float).median()
    med_b = col_b.astype(float).median()
    bin_a = (col_a.astype(float) > med_a).astype(float)
    bin_b = (col_b.astype(float) > med_b).astype(float)
    interaction = (bin_a + bin_b) % 2  # XOR: 1 when exactly one is above median
    return interaction - interaction.mean()
