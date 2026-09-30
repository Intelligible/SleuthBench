"""Core types for the phenomenon registry.

A `Phenomenon` bundles the three callables that together define one phenomenon:
an injector that modifies a dataframe, a validator that checks the injection
produced a deterministic answer, and one or more answer computers keyed by
template_id (multiple templates can share an injector).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


class InjectionRejected(ValueError):
    """The requested phenomenon cannot be injected into this dataset."""


class AnswerUnavailable(ValueError):
    """An answer computer found no reliable gold for this QA pair.

    This is an expected outcome, not a failure: the answer pipeline clears the
    QA pair's answer (so evaluation skips it) and does not fail the stage.
    Other QA pairs on the same instance are unaffected.
    """


@dataclass
class CheckDetail:
    name: str
    passed: bool
    metric: float
    threshold: float
    detail: str


@dataclass
class ValidationResult:
    passed: bool
    checks: list[CheckDetail] = field(default_factory=list)


def last_failed_detail(
    result: ValidationResult,
    default: str = "validation failed",
) -> str:
    """Return the last failed validation detail, or a stable fallback."""
    return next(
        (check.detail for check in reversed(result.checks) if not check.passed),
        default,
    )


def row_ids(df: pd.DataFrame, positions: np.ndarray) -> list[int]:
    """Resolve positional row selections to stable row identifiers."""
    if "row_id" in df.columns:
        return sorted(int(row_id) for row_id in df["row_id"].to_numpy()[positions])
    return sorted(int(position) for position in positions)


InjectFn = Callable[[pd.DataFrame, dict, np.random.Generator], tuple[pd.DataFrame, dict]]
ValidateFn = Callable[[pd.DataFrame, dict, str], ValidationResult]
ComputeAnswerFn = Callable[[pd.DataFrame, dict, dict], Any]
# Optional: maps the injector's `effects` dict to a short, human-readable string
# used as the instance-directory suffix when one injector produces multiple
# instances (e.g. "positive_synergy_AveRooms_Longitude"). Returns "" / None to
# fall back to the deterministic params hash.
LabelFn = Callable[[dict], str]


@dataclass(frozen=True)
class Phenomenon:
    name: str
    inject: InjectFn
    validate: ValidateFn
    compute_answers: dict[str, ComputeAnswerFn]
    # Names of fields under summary["by_kind"] whose column lists should be
    # forwarded into params before inject() runs. Each name "X" is forwarded
    # as params["X_cols"]. Declarative way for a phenomenon to request
    # dataset-level metadata it needs — avoids the orchestrator having to
    # know which phenomena want which summary fields.
    summary_fields: tuple[str, ...] = ()
    # Optional readable-name builder for the instance directory (see LabelFn).
    label: LabelFn | None = None
