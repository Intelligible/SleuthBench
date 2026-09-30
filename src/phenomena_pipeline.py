"""Inject phenomena into datasets to produce benchmark instances.

For each dataset summary, calls get_template_matches to match templates,
resolves their param_mapping placeholders from slot assignments, then
dispatches through phenomena.PHENOMENA to inject each phenomenon into a
fresh copy of the data.

With a run ID, writes one directory per injected instance to
data/instances/runs/{run_id}/{dataset}/seed_{N}/{instance_name}/.
``instance_name`` is normally the injector name; when one injector has multiple
resolved parameter sets, it gains a readable phenomenon label or deterministic
parameter hash, with a hash added for collision disambiguation. Names that
would leave too little room for generated artifacts on legacy Windows paths
are truncated with a stable hash suffix. Each directory holds table.csv and a
manifest with metadata, effects, and QA pairs whose answers remain null until
answer_pipeline runs. Without a run ID, the standalone CLI retains the legacy
unscoped layout.

Deduplication: same injector + same resolved params = one instance
directory even if multiple templates reference it.  RNG is deterministic:
MD5(f"{seed}:{injector_type}:{sorted(params.items())}") % 2**31.

Usage:
    uv run python src/phenomena_pipeline.py --seed 42
    uv run python src/phenomena_pipeline.py --summary data/standardized/summaries/bike_sharing_100.json --seed 42
    uv run python src/phenomena_pipeline.py --run-id smoke --template fc_nonmonotone_peak_v0
    uv run python src/phenomena_pipeline.py --template fc_nonmonotone_peak_v0 fc_interaction_dominant_v0
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from find_applicable_templates import get_template_matches, TemplateMatch
from phenomena import InjectionRejected, PHENOMENA
from io_utils import (
    load_csv,
    load_json,
    reset_generated_instance_dir,
    save_csv,
    save_json,
)
from shared.cli import configure_cli_streams
from shared.path_utils import (
    GENERATED_ARTIFACT_NAME_RESERVE_UNITS,
    WINDOWS_LEGACY_MAX_PATH_UNITS,
    ensure_portable_child_namespace,
    portable_path_key,
    safe_path_component,
    truncate_to_utf16_units,
    windows_utf16_units,
)
from shared.table import print_table


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(r"^\{(\w+)\}$")
_TRUNCATED_INSTANCE_HASH_CHARS = 12


class UnexpectedInjectionErrors(RuntimeError):
    """One or more injectors crashed instead of rejecting their inputs."""

    def __init__(self, failures: List[tuple[Path, str, Exception]]) -> None:
        self.failures = list(failures)
        summary_path, injector_type, exc = self.failures[0]
        first = (
            f"{summary_path.name}/{injector_type}: "
            f"{type(exc).__name__}: {exc}"
        )
        super().__init__(
            f"Unexpected injection failure(s) for {len(self.failures)} "
            f"injector attempt(s); first: {first}"
        )


def _validate_injection_result(
    result: object,
    expected_injector_type: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate the injector contract before any generated files are changed."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "injector must return a (DataFrame, phenomenon dict) tuple"
        )
    df_injected, phenomenon = result
    if not isinstance(df_injected, pd.DataFrame):
        raise TypeError("injector result[0] must be a pandas DataFrame")
    if not isinstance(phenomenon, dict):
        raise TypeError("injector result[1] must be a dict")

    missing = {"type", "params", "effects"} - phenomenon.keys()
    if missing:
        raise KeyError(
            f"injector phenomenon dict missing keys: {sorted(missing)}"
        )
    if phenomenon["type"] != expected_injector_type:
        raise ValueError(
            f"injector {expected_injector_type!r} returned phenomenon type "
            f"{phenomenon['type']!r}"
        )
    if not isinstance(phenomenon["params"], dict):
        raise TypeError("injector phenomenon 'params' must be a dict")
    if not isinstance(phenomenon["effects"], dict):
        raise TypeError("injector phenomenon 'effects' must be a dict")
    return df_injected, phenomenon


def resolve_injector_params(
    param_mapping: Dict[str, str],
    slot_assignments: Dict[str, str],
) -> Dict[str, str]:
    """Substitute {SLOT_NAME} references with concrete slot values."""
    resolved: Dict[str, str] = {}
    for param_name, value in param_mapping.items():
        m = _SLOT_RE.match(value)
        if m:
            slot_name = m.group(1)
            resolved[param_name] = slot_assignments.get(slot_name, value)
        else:
            resolved[param_name] = value
    return resolved


# ---------------------------------------------------------------------------
# Phenomena collection (with deduplication)
# ---------------------------------------------------------------------------

def _freeze(params: Dict[str, Any]) -> tuple:
    return tuple(sorted(params.items()))


def collect_required_phenomena(
    matches: List[TemplateMatch],
) -> List[Dict[str, Any]]:
    """Gather unique injector specs from compatible template matches."""
    seen: set[tuple] = set()
    specs: List[Dict[str, Any]] = []

    for m in matches:
        if not m.is_compatible:
            continue
        for phenom in m.template.get("phenomena", []):
            injector = phenom["injector"]
            resolved = resolve_injector_params(
                phenom.get("param_mapping", {}),
                m.slot_assignments,
            )
            key = (injector, _freeze(resolved))
            if key not in seen:
                seen.add(key)
                specs.append({"type": injector, "params": resolved})

    return specs


# ---------------------------------------------------------------------------
# QA generation
# ---------------------------------------------------------------------------

def _fill_effect_placeholders(question: str, effects: Dict[str, Any] | None) -> str:
    """Substitute ``{EFFECT_KEY}`` placeholders left over after slot rendering.

    Slot placeholders are resolved at template-match time, but some questions
    need to name a value the injector only decides at inject time (e.g. which
    feature it planted the step into). Those appear as ``{UPPERCASED_EFFECT_KEY}``
    and are filled here from the injector's ``effects`` dict (scalar values only).
    Unknown placeholders are left intact.
    """
    if not effects:
        return question

    mapping = {
        str(k).upper(): v
        for k, v in effects.items()
        if isinstance(v, (str, int, float)) and not isinstance(v, bool)
    }

    class _Missing(dict):
        def __missing__(self, key: str) -> str:
            return f"{{{key}}}"

    return question.format_map(_Missing(mapping))


def generate_qa_pairs(
    matches: List[TemplateMatch],
    question_type: str = "ds",
    effects: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Build a list of QA dicts from compatible template matches.

    Answers are not computed here — they are filled in by answer_pipeline.py
    after validation passes. ``question_type`` is recorded so downstream
    stages know which question variant the rendered ``question`` came from.
    ``effects`` (from the just-run injection) fills any ``{EFFECT_KEY}``
    placeholders the slot pass left in the question.
    """
    qa_pairs: List[Dict[str, Any]] = []
    for m in matches:
        if not m.is_compatible:
            continue

        qa_pairs.append({
            "template_id": m.template["template_id"],
            "category": m.template.get("category", ""),
            "answer_format": m.template.get("answer_format", ""),
            "question_type": question_type,
            "question": _fill_effect_placeholders(m.rendered_question, effects),
            "slot_assignments": m.slot_assignments,
            "answer": None,
        })

    return qa_pairs


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def _get_templates_for_phenomenon(
    matches: List[TemplateMatch],
    injector_type: str,
    params: Dict[str, Any],
) -> List[TemplateMatch]:
    """Return templates that use the given injector type and resolved params."""
    result = []
    frozen_params = _freeze(params)
    for m in matches:
        if not m.is_compatible:
            continue
        for phenom in m.template.get("phenomena", []):
            if phenom["injector"] != injector_type:
                continue
            resolved = resolve_injector_params(
                phenom.get("param_mapping", {}),
                m.slot_assignments,
            )
            if _freeze(resolved) == frozen_params:
                result.append(m)
                break
    return result


def _params_hash(params: Dict[str, Any]) -> str:
    frozen = repr(_freeze(params)).encode()
    return hashlib.md5(frozen).hexdigest()[:8]


def _fit_instance_name_to_path(
    instance_name: str,
    instance_parent: str | Path,
    identity: object,
    *,
    max_path_units: int | None = None,
) -> str:
    """Bound a generated instance name while retaining a stable identity.

    Reserve enough room below the instance directory for current and future
    generated artifact filenames.  Ordinary names stay unchanged; only paths
    that would exceed legacy Win32 MAX_PATH are shortened.
    """
    instance_name = safe_path_component(instance_name, "instance_name")
    absolute_parent = Path(os.path.abspath(instance_parent))
    if max_path_units is None:
        if os.name != "nt" or str(absolute_parent).startswith("\\\\?\\"):
            return instance_name
        max_path_units = WINDOWS_LEGACY_MAX_PATH_UNITS

    max_name_units = (
        max_path_units
        - windows_utf16_units(absolute_parent)
        - 2  # separators before the instance and artifact filename
        - GENERATED_ARTIFACT_NAME_RESERVE_UNITS
    )
    if windows_utf16_units(instance_name) <= max_name_units:
        return instance_name

    digest = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[
        :_TRUNCATED_INSTANCE_HASH_CHARS
    ]
    separator = "__"
    suffix = f"{separator}{digest}"
    prefix_units = max_name_units - windows_utf16_units(suffix)
    if prefix_units < 1:
        raise OSError(
            "generated instance parent path is too long to allocate a "
            f"portable instance name below {absolute_parent}"
        )
    prefix = truncate_to_utf16_units(instance_name, prefix_units).rstrip("._-")
    if not prefix:
        raise OSError(
            "generated instance name has no usable prefix within the legacy "
            f"Windows path budget below {absolute_parent}"
        )
    return safe_path_component(
        f"{prefix}{suffix}",
        "instance_name",
    )


def _slugify(text: str, max_len: int = 60) -> str:
    """Filesystem-safe, readable slug: keep [A-Za-z0-9_-], collapse the rest to '_'."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")
    return s[:max_len].rstrip("_") if len(s) > max_len else s


def _fmt_params(params: dict, max_len: int = 50) -> str:
    """Format params dict as compact key=val string, truncated if needed."""
    parts = [f"{k}={v}" for k, v in params.items()]
    s = ", ".join(parts)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def build_instance(
    summary_path: str | Path,
    seed: int,
    templates_dir: str | Path = "templates",
    template_filter: List[str] | None = None,
    question_type: str = "ds",
    output_root: str | Path | None = None,
    run_id: str | None = None,
    generation_metadata: Dict[str, Any] | None = None,
) -> List[Path]:
    """Full pipeline: load -> match -> inject one phenomenon per instance -> QA -> save.

    Creates one output directory per unique injector + resolved-parameter set,
    each with its own table.csv containing only that single injection.
    ``question_type`` chooses which question variant ("ds" or "business") is
    rendered into the QA pairs.
    ``output_root`` overrides the default data/instances root. When supplied,
    ``run_id`` scopes output under ``runs/{run_id}`` and is recorded in each
    manifest for downstream traceability. If an exact target instance already
    exists, that generated directory is recreated before table and manifest
    writes so no derived state survives from an older table.
    """
    summary_path = Path(summary_path)
    templates_dir = Path(templates_dir)
    summary = load_json(summary_path)

    dataset_filename = summary["dataset"]
    dataset_dir = summary_path.parent.parent  # up from summaries/ to standardized/
    base_dataset = dataset_dir / dataset_filename
    if output_root is None:
        output_root = dataset_dir.parent / "instances"
    output_root = Path(output_root)
    if run_id is not None:
        run_id = safe_path_component(run_id, "run_id")
        runs_root = output_root / "runs"
        ensure_portable_child_namespace(runs_root, run_id, "run_id")
        output_root = runs_root / run_id
    target = summary.get("target", "")
    dataset_name = safe_path_component(
        Path(dataset_filename).stem,
        "dataset_name",
    )

    df_original = load_csv(base_dataset)

    matches = get_template_matches(
        summary_path, base_dataset, templates_dir, seed=seed,
        question_type=question_type,
    )

    if template_filter:
        filter_set = set(template_filter)
        matches = [m for m in matches if m.template["template_id"] in filter_set]

    compatible = [m for m in matches if m.is_compatible]

    phenomena_specs = collect_required_phenomena(matches)

    output_dirs: List[Path] = []
    table_rows: List[List[str]] = []
    unexpected_failures: List[tuple[Path, str, Exception]] = []
    by_kind = summary.get("by_kind", {})

    injector_counts: Dict[str, int] = {}
    for spec in phenomena_specs:
        injector_counts[spec["type"]] = injector_counts.get(spec["type"], 0) + 1

    # Track filesystem collision keys rather than Python string equality so
    # case/Unicode-equivalent labels cannot overwrite each other on Windows or
    # default macOS filesystems.
    used_names: set[str] = set()

    for spec in phenomena_specs:
        injector_type = spec["type"]
        resolved_params = spec["params"]
        params = resolved_params

        phenomenon = PHENOMENA.get(injector_type)
        if phenomenon is None:
            exc = LookupError(
                f"No phenomenon registered for injector {injector_type!r}"
            )
            unexpected_failures.append((summary_path, injector_type, exc))
            table_rows.append(
                [
                    injector_type,
                    "-",
                    _fmt_params(params),
                    "ERROR (unregistered injector)",
                ]
            )
            continue

        # Forward any summary fields the phenomenon declared it wants.
        for field in phenomenon.summary_fields:
            val = by_kind.get(field, [])
            if val:
                params = {**params, f"{field}_cols": val}

        df = df_original.copy()

        # Create a fresh RNG deterministically from seed + injector identity
        rng_key = f"{seed}:{injector_type}:{sorted(params.items())}"
        rng_seed = int(hashlib.md5(rng_key.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(rng_seed)

        try:
            df_injected, phenom_dict = _validate_injection_result(
                phenomenon.inject(df, params, rng),
                injector_type,
            )
        except InjectionRejected as exc:
            table_rows.append(
                [injector_type, "-", _fmt_params(params), f"SKIP ({exc})"]
            )
            continue
        except Exception as exc:
            table_rows.append(
                [
                    injector_type,
                    "-",
                    _fmt_params(params),
                    f"ERROR ({type(exc).__name__}: {exc})",
                ]
            )
            unexpected_failures.append((summary_path, injector_type, exc))
            continue

        relevant_matches = _get_templates_for_phenomenon(
            matches,
            injector_type,
            resolved_params,
        )

        # Generate QA pairs only for relevant templates (answers computed later).
        # Pass the injection's effects so {EFFECT_KEY} placeholders (e.g. a
        # feature the injector chose) get filled into the question text.
        qa_pairs = generate_qa_pairs(
            relevant_matches,
            question_type,
            phenom_dict["effects"],
        )

        # Save outputs to phenomenon-specific directory. When one injector
        # yields multiple instances, suffix the dir to keep them distinct:
        # prefer the phenomenon's readable label (built from effects), else the
        # deterministic params hash. Guard against label collisions by appending
        # the hash when a readable name is already taken this seed.
        instance_name = injector_type
        if injector_counts.get(injector_type, 0) > 1:
            suffix = ""
            if phenomenon.label is not None:
                try:
                    suffix = _slugify(
                        phenomenon.label(phenom_dict["effects"]) or ""
                    )
                except Exception:
                    suffix = ""
            if not suffix:
                suffix = _params_hash(resolved_params)
            instance_name = f"{injector_type}_{suffix}"
        instance_parent = output_root / dataset_name / f"seed_{seed}"
        instance_identity = (injector_type, _freeze(resolved_params))
        instance_name = _fit_instance_name_to_path(
            instance_name,
            instance_parent,
            instance_identity,
        )
        instance_key = portable_path_key(instance_name)
        if instance_key in used_names:
            instance_name = _fit_instance_name_to_path(
                f"{instance_name}_{_params_hash(resolved_params)}",
                instance_parent,
                (instance_identity, "collision"),
            )
            instance_key = portable_path_key(instance_name)
        if instance_key in used_names:
            raise ValueError(
                f"instance name collision after portable normalization: "
                f"{instance_name!r}"
            )
        used_names.add(instance_key)
        out_dir = instance_parent / instance_name
        # The directory is generated state owned by this instance. Recreate it
        # so generated files from an older table cannot survive beside the new
        # table and manifest.
        reset_generated_instance_dir(out_dir, output_root)

        save_csv(df_injected, out_dir / "table.csv")

        manifest = {
            "run_id": run_id,
            "dataset_name": dataset_name,
            "seed": seed,
            "base_dataset": str(base_dataset),
            "target": target,
            "phenomenon": {
                "injector_type": phenom_dict["type"],
                "params": phenom_dict["params"],
                "effects": phenom_dict["effects"],
            },
            "templates_applied": [
                m.template["template_id"]
                for m in relevant_matches
            ],
            "qa_pairs": qa_pairs,
        }
        if generation_metadata is not None:
            manifest["generation"] = dict(generation_metadata)
        save_json(manifest, out_dir / "manifest.json")

        template_ids = ", ".join(m.template["template_id"] for m in relevant_matches)
        table_rows.append([injector_type, template_ids, _fmt_params(params), "OK"])
        output_dirs.append(out_dir)

    for m in matches:
        if m.is_compatible:
            continue
        injectors = m.template.get("phenomena", [])
        injector_name = injectors[0]["injector"] if injectors else "-"
        reason = "; ".join(m.reasons)
        if len(reason) > 50:
            reason = reason[:47] + "..."
        table_rows.append([injector_name, m.template["template_id"], reason, "NO MATCH"])

    print(f"\n{dataset_name}  ({len(compatible)}/{len(matches)} templates, seed {seed})")
    print_table(
        ["Injector", "Template", "Params / Reason", "Status"],
        table_rows,
        max_last_col=50,
    )
    if unexpected_failures:
        raise UnexpectedInjectionErrors(unexpected_failures) from unexpected_failures[0][2]
    return output_dirs


def build_instances(
    summary_paths: List[str | Path],
    seed: int,
    templates_dir: str | Path = "templates",
    template_filter: List[str] | None = None,
    question_type: str = "ds",
    output_root: str | Path | None = None,
    run_id: str | None = None,
    generation_metadata: Dict[str, Any] | None = None,
) -> List[Path]:
    """Build multiple summaries while aggregating unexpected injector errors."""
    summary_paths = list(summary_paths)
    # Fail before writing anything if two summaries map to the same portable
    # dataset directory. Otherwise the later build could recreate and overwrite
    # instances produced by the earlier one on Windows/default macOS filesystems.
    seen_datasets: dict[str, str] = {}
    for summary_path in summary_paths:
        summary = load_json(summary_path)
        dataset_name = safe_path_component(
            Path(summary["dataset"]).stem,
            "dataset_name",
        )
        dataset_key = portable_path_key(dataset_name)
        if dataset_key in seen_datasets:
            raise ValueError(
                f"dataset names {seen_datasets[dataset_key]!r} and "
                f"{dataset_name!r} collide on a portable filesystem"
            )
        seen_datasets[dataset_key] = dataset_name

    output_dirs: List[Path] = []
    failures: List[tuple[Path, str, Exception]] = []
    for summary_path in summary_paths:
        try:
            output_dirs.extend(
                build_instance(
                    summary_path,
                    seed,
                    templates_dir,
                    template_filter=template_filter,
                    question_type=question_type,
                    output_root=output_root,
                    run_id=run_id,
                    generation_metadata=generation_metadata,
                )
            )
        except UnexpectedInjectionErrors as exc:
            failures.extend(exc.failures)

    if failures:
        raise UnexpectedInjectionErrors(failures) from failures[0][2]
    return output_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    configure_cli_streams()
    warnings.filterwarnings(
        "ignore",
        message="Setting an item of incompatible dtype is deprecated.*",
        category=FutureWarning,
    )
    parser = argparse.ArgumentParser(description="Phenomena injection pipeline")
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to dataset summary JSON. If omitted, runs on all summaries in data/standardized/summaries/.",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates",
        help="Path to templates directory (default: templates)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier used to isolate output under data/instances/runs/",
    )
    parser.add_argument(
        "--template",
        nargs="+",
        default=None,
        help=(
            "Only run specific template IDs "
            "(e.g. --template fc_nonmonotone_peak_v0 "
            "fc_interaction_dominant_v0)"
        ),
    )
    parser.add_argument(
        "--question-type",
        choices=["ds", "business"],
        default="ds",
        help="Which question variant to render into QA pairs (default: ds = data science question)",
    )
    args = parser.parse_args()

    if args.summary:
        summary_files = [Path(args.summary)]
    else:
        summaries_dir = Path("data/standardized/summaries")
        summary_files = sorted(summaries_dir.glob("*.json"))
        if not summary_files:
            print(f"No summary files found in {summaries_dir}")
            return

    build_instances(
        summary_files,
        args.seed,
        args.templates_dir,
        template_filter=args.template,
        question_type=args.question_type,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
