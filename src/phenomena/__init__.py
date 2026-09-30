"""Phenomenon registry.

Each phenomenon bundles an injector, a validator, and one or more
answer computers (keyed by template_id) into a single `Phenomenon`
object.  Pipelines look phenomena up two ways:

- `PHENOMENA[injector_type]` — used by the phenomena and validate
  pipelines, which dispatch on the injector name from the manifest.
- `TEMPLATE_TO_PHENOMENON[template_id]` — used by the answer pipeline,
  which dispatches on a template_id to pick the matching answer computer.

Adding a new phenomenon: create `src/phenomena/{name}.py` with an
`inject`, `validate`, and one or more `compute_answer_*` functions,
expose a module-level `PHENOMENON`, then import the module and append it
to `_MODULES` below. The two public registries are derived from that list.
"""
from __future__ import annotations

from . import (
    dq_bad_row_indicator,
    dq_bad_row_indicator_v1,
    dq_categorical_target_outlier,
    dq_conditional_bad_rows,
    dq_group_evidence_underpowered,
    dq_missing_label_semantic,
    dq_missing_label_target_dependent,
    dq_unreliable_feature,
    fc_interaction_dominant,
    fc_monotone_classify,
    fc_noise_feature,
    fc_nonmonotone_peak,
    fc_pairwise_interaction,
    fc_threshold_value,
)
from ._base import (
    AnswerUnavailable,
    CheckDetail,
    InjectionRejected,
    Phenomenon,
    ValidationResult,
)

_MODULES = [
    dq_bad_row_indicator,
    dq_bad_row_indicator_v1,
    dq_categorical_target_outlier,
    dq_conditional_bad_rows,
    dq_group_evidence_underpowered,
    dq_missing_label_semantic,
    dq_missing_label_target_dependent,
    dq_unreliable_feature,
    fc_interaction_dominant,
    fc_monotone_classify,
    fc_noise_feature,
    fc_nonmonotone_peak,
    fc_pairwise_interaction,
    fc_threshold_value,
]

PHENOMENA: dict[str, Phenomenon] = {m.PHENOMENON.name: m.PHENOMENON for m in _MODULES}

TEMPLATE_TO_PHENOMENON: dict[str, Phenomenon] = {}
for _phen in PHENOMENA.values():
    for _tid in _phen.compute_answers:
        if _tid in TEMPLATE_TO_PHENOMENON:
            raise RuntimeError(
                f"template_id '{_tid}' is claimed by both "
                f"'{TEMPLATE_TO_PHENOMENON[_tid].name}' and '{_phen.name}'"
            )
        TEMPLATE_TO_PHENOMENON[_tid] = _phen

__all__ = [
    "AnswerUnavailable",
    "CheckDetail",
    "InjectionRejected",
    "Phenomenon",
    "ValidationResult",
    "PHENOMENA",
    "TEMPLATE_TO_PHENOMENON",
]
