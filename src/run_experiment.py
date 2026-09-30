"""Single-config experiment runner.

Reads a YAML config and drives the full pipeline end-to-end:
phenomena → validate → answer → eval → grade.

Usage:
    uv run python src/run_experiment.py experiments/example.yaml
    uv run python src/run_experiment.py experiments/example.yaml --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

import phenomena_pipeline
import validate_pipeline
import answer_pipeline
import eval_pipeline
import grade_pipeline
from io_utils import remove_generated_instance_dir
from shared.cli import configure_cli_streams
from shared.files import sha256_file
from shared.path_utils import (
    ensure_portable_child_namespace,
    normalize_path_text,
    portable_path_key,
    safe_path_component,
)


VALID_TOOLS = frozenset(eval_pipeline.SUPPORTED_TOOLS)
VALID_QUESTION_TYPES = {"ds", "business"}
GENERATION_FINGERPRINT_VERSION = 1


@dataclass
class ExperimentConfig:
    name: str
    run_id: str
    seed: int
    summary: Path | None
    summaries_dir: Path | None
    templates_dir: Path
    templates: list[str] | None
    question_type: str
    dataset_filter: str | None
    injector_filter: list[str] | None
    models: list[str]
    columns_only: bool
    table_in_prompt: bool
    grader_model: str
    skip_phenomena: bool
    skip_validate: bool
    skip_answer: bool
    skip_eval: bool
    skip_grade: bool
    force: bool
    instances_dir: Path
    results_dir: Path
    runs: list[dict] | None = None


def _normalize_models(value: object, field_name: str) -> list[str]:
    """Validate one model list and reject duplicate paid work."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of model names")

    models: list[str] = []
    seen: set[str] = set()
    for index, model in enumerate(value):
        if not isinstance(model, str) or not model:
            raise ValueError(
                f"{field_name}[{index}] must be a non-empty model name"
            )
        if model in seen:
            raise ValueError(
                f"{field_name} contains duplicate model name {model!r}"
            )
        seen.add(model)
        models.append(model)
    return models


def load_config(path: Path, force_override: bool) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if "seed" not in raw:
        raise ValueError("config is missing required field: seed")
    if bool(raw.get("summary")) == bool(raw.get("summaries_dir")):
        raise ValueError("config must set exactly one of 'summary' or 'summaries_dir'")

    summary = Path(raw["summary"]).resolve() if raw.get("summary") else None
    summaries_dir = Path(raw["summaries_dir"]).resolve() if raw.get("summaries_dir") else None

    # Auto-derive dataset_filter from the summary stem when not set so the
    # validate/eval filters and skip_phenomena reuse are scoped to this summary's
    # dataset. Stages only ever consume the run-scoped manifest batch, so this
    # filter is not what isolates instances from other runs.
    dataset_filter = raw.get("dataset_filter")
    if dataset_filter is None and summary is not None:
        dataset_filter = summary.stem
    if dataset_filter is not None:
        dataset_filter = normalize_path_text(dataset_filter)

    name = safe_path_component(
        raw.get("name") or (summary.stem if summary else "experiment"),
        "name",
    )
    run_id = safe_path_component(raw.get("run_id") or name, "run_id")
    instances_dir = Path(raw.get("instances_dir", "data/instances"))
    results_dir = Path(raw.get("results_dir", "data/results"))
    ensure_portable_child_namespace(
        instances_dir / "runs",
        run_id,
        "run_id",
    )
    ensure_portable_child_namespace(
        results_dir / "runs",
        run_id,
        "run_id",
    )

    question_type = raw.get("question_type", "ds")
    if question_type not in VALID_QUESTION_TYPES:
        raise ValueError(
            f"invalid question_type {question_type!r}; expected one of {sorted(VALID_QUESTION_TYPES)}"
        )

    skip_phenomena = bool(raw.get("skip_phenomena", False))
    skip_validate = bool(raw.get("skip_validate", False))
    skip_answer = bool(raw.get("skip_answer", False))
    skip_eval = bool(raw.get("skip_eval", False))
    skip_grade = bool(raw.get("skip_grade", False))

    # Fresh Stage 1 instances deliberately contain neither validation nor
    # answers.  Skipping either prerequisite while continuing to eval cannot
    # reuse older state because Stage 1 cleanly recreates its output dirs; it
    # would therefore produce no evaluable QA pairs.  When Stage 1 itself is
    # skipped, existing completed manifests may legitimately supply either
    # prerequisite, so leave that path to the runtime manifest gates.
    if not skip_phenomena and not skip_eval:
        skipped_prerequisites = [
            stage
            for stage, skipped in (
                ("skip_validate", skip_validate),
                ("skip_answer", skip_answer),
            )
            if skipped
        ]
        if skipped_prerequisites:
            raise ValueError(
                "cannot run eval after generating fresh instances while "
                "required stages are skipped: "
                f"{', '.join(skipped_prerequisites)}; enable the validation "
                "and answer stages, set skip_eval: true, or set "
                "skip_phenomena: true to reuse completed manifests"
            )

    models = _normalize_models(raw.get("models"), "models")
    runs = raw.get("runs")
    if runs is not None:
        if not isinstance(runs, list) or not runs:
            raise ValueError("'runs' must be a non-empty list")
        seen: dict[str, str] = {}
        for i, r in enumerate(runs):
            if not isinstance(r, dict) or not r.get("name"):
                raise ValueError(f"runs[{i}] must be a mapping with a 'name'")
            run_name = safe_path_component(r["name"], f"runs[{i}].name")
            r["name"] = run_name
            run_name_key = portable_path_key(run_name)
            if run_name_key in seen:
                raise ValueError(
                    f"run names {seen[run_name_key]!r} and {run_name!r} "
                    "collide on case-insensitive filesystems or after "
                    "Unicode normalization"
                )
            seen[run_name_key] = run_name
            run_tools = r.get("tools") or []
            unknown = set(run_tools) - VALID_TOOLS
            if unknown:
                raise ValueError(f"unknown tools in runs[{i}] ('{r['name']}'): {sorted(unknown)}")
            if "models" in r:
                r["models"] = _normalize_models(
                    r["models"],
                    f"runs[{i}].models ({r['name']!r})",
                )

    return ExperimentConfig(
        name=name,
        run_id=run_id,
        seed=int(raw["seed"]),
        summary=summary,
        summaries_dir=summaries_dir,
        templates_dir=Path(raw.get("templates_dir", "templates")),
        templates=raw.get("templates"),
        question_type=question_type,
        dataset_filter=dataset_filter,
        injector_filter=raw.get("injector_filter"),
        models=models,
        columns_only=bool(raw.get("columns_only", False)),
        table_in_prompt=bool(raw.get("table_in_prompt", False)),
        grader_model=raw.get("grader_model", "gpt-4o-mini"),
        skip_phenomena=skip_phenomena,
        skip_validate=skip_validate,
        skip_answer=skip_answer,
        skip_eval=skip_eval,
        skip_grade=skip_grade,
        force=force_override or bool(raw.get("force", False)),
        instances_dir=instances_dir,
        results_dir=results_dir,
        runs=runs,
    )


def banner(label: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {label}\n{line}")


def _selected_summaries(cfg: ExperimentConfig) -> list[Path]:
    """Resolve the exact summary batch selected by an experiment config."""
    if cfg.summary:
        summaries = [cfg.summary]
    else:
        assert cfg.summaries_dir is not None
        summaries = sorted(cfg.summaries_dir.glob("*.json"))
    if not summaries:
        location = cfg.summary or cfg.summaries_dir
        raise RuntimeError(f"No summaries found in {location}")
    return summaries


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def build_generation_metadata(
    cfg: ExperimentConfig,
    summaries: list[Path] | None = None,
) -> dict:
    """Build a stable identity for all Stage 1 inputs that affect instances.

    The identity deliberately hashes content instead of absolute paths so the
    same experiment can be moved to another checkout without becoming a
    different batch.  It covers the summaries, their source tables, selected
    template definitions, seed, and rendered question variant.
    """
    summaries = list(summaries or _selected_summaries(cfg))
    selected_ids = set(cfg.templates) if cfg.templates else None
    identity = {
        "version": GENERATION_FINGERPRINT_VERSION,
        "seed": cfg.seed,
        "question_type": cfg.question_type,
        "template_selection": (
            sorted(selected_ids) if selected_ids is not None else "ALL"
        ),
        "summaries": [],
        "templates": [],
    }

    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset_path = summary_path.parent.parent / summary["dataset"]
        identity["summaries"].append(
            {
                "dataset_name": safe_path_component(
                    Path(summary["dataset"]).stem,
                    "dataset_name",
                ),
                "summary": summary,
                "table_sha256": sha256_file(dataset_path),
            }
        )
    identity["summaries"].sort(
        key=lambda item: portable_path_key(item["dataset_name"])
    )
    dataset_names = [item["dataset_name"] for item in identity["summaries"]]
    dataset_keys = [portable_path_key(name) for name in dataset_names]
    if len(dataset_keys) != len(set(dataset_keys)):
        raise ValueError(
            "selected summaries contain dataset names that collide on "
            "case-insensitive or Unicode-normalizing filesystems"
        )

    templates_dir = Path(cfg.templates_dir)
    for template_path in sorted(templates_dir.rglob("*.json")):
        if template_path.name == "template_schema.json":
            continue
        template = json.loads(template_path.read_text(encoding="utf-8"))
        template_id = template["template_id"]
        if selected_ids is not None and template_id not in selected_ids:
            continue
        identity["templates"].append(
            {"template_id": template_id, "template": template}
        )
    identity["templates"].sort(key=lambda item: item["template_id"])
    found_ids = [item["template_id"] for item in identity["templates"]]
    if len(found_ids) != len(set(found_ids)):
        raise ValueError("selected templates contain duplicate template IDs")
    if not found_ids:
        raise ValueError(f"no selected templates found under {templates_dir}")
    if selected_ids is not None:
        missing = selected_ids - set(found_ids)
        if missing:
            raise ValueError(
                f"selected template_id(s) not found: {sorted(missing)}"
            )

    fingerprint = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return {
        "version": GENERATION_FINGERPRINT_VERSION,
        "fingerprint": fingerprint,
        "question_type": cfg.question_type,
        "template_selection": identity["template_selection"],
        "datasets": dataset_names,
    }


def _cleanup_orphan_run_instances(
    cfg: ExperimentConfig,
    generation_metadata: dict,
    output_dirs: list[Path],
) -> int:
    """Delete stale instance directories only after Stage 1 fully succeeds."""
    run_root = (
        cfg.instances_dir
        / "runs"
        / safe_path_component(cfg.run_id, "run_id")
    )
    resolved_run_root = run_root.resolve()
    seed_part = f"seed_{cfg.seed}"
    dataset_names = set(generation_metadata["datasets"])

    expected: set[Path] = set()
    for output_dir in output_dirs:
        resolved_output = Path(output_dir).resolve()
        relative = resolved_output.relative_to(resolved_run_root)
        if (
            len(relative.parts) != 3
            or relative.parts[0] not in dataset_names
            or relative.parts[1] != seed_part
        ):
            raise ValueError(
                f"unexpected Stage 1 output scope: {output_dir}"
            )
        expected.add(resolved_output)

    # The exact three-level glob can also find datasets removed from the new
    # config. Validate every candidate before deleting any of them.
    orphan_dirs: list[Path] = []
    for manifest_path in sorted(
        run_root.glob(f"*/{seed_part}/*/manifest.json")
    ):
        instance_dir = manifest_path.parent
        relative = instance_dir.resolve().relative_to(resolved_run_root)
        if len(relative.parts) != 3 or relative.parts[1] != seed_part:
            raise ValueError(f"unexpected instance scope: {instance_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("run_id") == cfg.run_id
            and manifest.get("seed") == cfg.seed
            and manifest.get("dataset_name") == relative.parts[0]
            and instance_dir.resolve() not in expected
        ):
            orphan_dirs.append(instance_dir)

    for orphan_dir in orphan_dirs:
        remove_generated_instance_dir(orphan_dir, run_root)
    if orphan_dirs:
        print(f"  Removed {len(orphan_dirs)} stale instance(s)")
    return len(orphan_dirs)


def discover_run_manifests(cfg: ExperimentConfig) -> list[Path]:
    """Find an existing instance batch without falling back to a full-tree scan."""
    run_scope = ensure_portable_child_namespace(
        cfg.instances_dir / "runs",
        cfg.run_id,
        "run_id",
    )
    run_root = cfg.instances_dir / "runs" / run_scope
    expected_generation = build_generation_metadata(cfg)
    allowed_template_ids = set(cfg.templates or [])
    manifests: list[Path] = []
    mismatched: list[Path] = []
    for path in sorted(run_root.glob("**/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != cfg.run_id or manifest.get("seed") != cfg.seed:
            continue
        if (
            manifest.get("generation", {}).get("fingerprint")
            != expected_generation["fingerprint"]
        ):
            mismatched.append(path)
            continue
        dataset_name = normalize_path_text(
            manifest.get("dataset_name", "")
        )
        if cfg.dataset_filter and not dataset_name.endswith(cfg.dataset_filter):
            continue
        injector_type = manifest.get("phenomenon", {}).get("injector_type")
        if cfg.injector_filter and injector_type not in cfg.injector_filter:
            continue
        if allowed_template_ids and not any(
            qa.get("template_id") in allowed_template_ids
            for qa in (manifest.get("qa_pairs") or [])
        ):
            continue
        manifests.append(path)

    if mismatched:
        raise RuntimeError(
            f"Existing manifests for run_id={cfg.run_id!r}, seed={cfg.seed} "
            f"were generated from different Stage 1 inputs (or predate "
            f"generation fingerprints). Regenerate Stage 1; first mismatch: "
            f"{mismatched[0]}"
        )
    if not manifests:
        raise RuntimeError(
            f"No existing manifests found for run_id={cfg.run_id!r}, seed={cfg.seed} "
            f"under {run_root}. Regenerate Stage 1 or use the matching run_id."
        )
    return manifests


def stage_phenomena(cfg: ExperimentConfig) -> list[Path]:
    banner("Stage 1/5 — Phenomena injection")
    summaries = _selected_summaries(cfg)
    generation_metadata = build_generation_metadata(cfg, summaries)
    output_dirs = phenomena_pipeline.build_instances(
        summaries,
        cfg.seed,
        cfg.templates_dir,
        template_filter=cfg.templates,
        question_type=cfg.question_type,
        output_root=cfg.instances_dir,
        run_id=cfg.run_id,
        generation_metadata=generation_metadata,
    )
    # Reconcile orphans only after the complete multi-summary build.  A failed
    # build may already have refreshed successful instances, but it never
    # guesses which untouched old instances have become obsolete.
    _cleanup_orphan_run_instances(
        cfg,
        generation_metadata,
        output_dirs,
    )
    return [out_dir / "manifest.json" for out_dir in output_dirs]


def stage_validate(cfg: ExperimentConfig, manifest_paths: list[Path] | None = None) -> None:
    banner("Stage 2/5 — Validate")
    validate_pipeline.run(
        cfg.instances_dir,
        cfg.dataset_filter,
        cfg.injector_filter,
        cfg.force,
        manifest_paths=manifest_paths,
    )


def stage_answer(cfg: ExperimentConfig, manifest_paths: list[Path] | None = None) -> None:
    banner("Stage 3/5 — Compute answers")
    answer_pipeline.run(cfg.instances_dir, manifest_paths=manifest_paths)


def stage_eval(
    cfg: ExperimentConfig,
    tools: list[str],
    manifest_paths: list[Path] | None = None,
) -> Path:
    banner("Stage 4/5 — Evaluate models")
    if not cfg.models:
        raise RuntimeError("config has no 'models' — cannot run eval stage (set skip_eval: true to skip)")

    run_scope = ensure_portable_child_namespace(
        cfg.results_dir / "runs",
        cfg.run_id,
        "run_id",
    )
    run_results_dir = cfg.results_dir / "runs" / run_scope
    output_path = run_results_dir / f"eval_results_{cfg.name}.json"
    results = eval_pipeline.run_eval(
        cfg.models,
        cfg.instances_dir,
        output_path,
        dataset=cfg.dataset_filter,
        injector=cfg.injector_filter,
        template_ids=cfg.templates,
        enabled_tools=set(tools),
        columns_only=cfg.columns_only,
        table_in_prompt=cfg.table_in_prompt,
        debug=False,
        manifest_paths=manifest_paths,
        run_id=cfg.run_id,
    )
    eval_pipeline.print_summary(results)
    return output_path


def stage_grade(cfg: ExperimentConfig, eval_output: Path) -> Path:
    banner("Stage 5/5 — Grade")
    graded_path = eval_output.parent / "graded" / f"{eval_output.stem}_graded.json"
    return grade_pipeline.run(eval_output, graded_path, cfg.grader_model)


def main() -> None:
    configure_cli_streams()
    parser = argparse.ArgumentParser(description="Run a full SleuthBench experiment from a single YAML config")
    parser.add_argument("config", type=Path, help="Path to YAML experiment config")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run validation instead of its idempotent skip "
            "(overrides config.force)"
        ),
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config file not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config, args.force)

    print(f"Experiment: {cfg.name}")
    print(f"  run_id={cfg.run_id}")
    print(f"  seed={cfg.seed} dataset_filter={cfg.dataset_filter} templates={cfg.templates or 'ALL'}")
    print(f"  question_type={cfg.question_type}")
    print(f"  default models={cfg.models}")
    if cfg.runs:
        print(f"  runs ({len(cfg.runs)}):")
        for r in cfg.runs:
            print(f"    - {r['name']}: tools={r.get('tools') or []}"
                  + (f" models={r['models']}" if 'models' in r else ""))
    print(f"  skip_phenomena={cfg.skip_phenomena} skip_validate={cfg.skip_validate} skip_answer={cfg.skip_answer}")
    print(f"  skip_eval={cfg.skip_eval} skip_grade={cfg.skip_grade} force={cfg.force}")

    t_start = time.time()

    # If Stage 1 is skipped, discover the matching run-scoped batch so whichever
    # later stages remain enabled can reuse those manifests.
    manifest_paths: list[Path] | None = None
    if not cfg.skip_phenomena:
        manifest_paths = stage_phenomena(cfg)
    else:
        banner("Stage 1/5 — Phenomena injection [SKIPPED — reusing existing instances]")
        manifest_paths = discover_run_manifests(cfg)
        print(f"  Reusing {len(manifest_paths)} manifests for run_id={cfg.run_id}")

    if not cfg.skip_validate:
        stage_validate(cfg, manifest_paths)
    else:
        banner("Stage 2/5 — Validate [SKIPPED]")

    if not cfg.skip_answer:
        stage_answer(cfg, manifest_paths)
    else:
        banner("Stage 3/5 — Compute answers [SKIPPED]")

    base_models = list(cfg.models)
    runs = cfg.runs or [{"name": cfg.name}]
    completed: list[tuple[str, Path | None, Path | None]] = []
    for i, run in enumerate(runs, 1):
        cfg.name = run["name"]
        cfg.models = list(run.get("models", base_models))
        tools = run.get("tools") or []
        suffix = f"  [run {i}/{len(runs)}: {cfg.name}]" if cfg.runs else ""

        eval_output: Path | None = None
        if not cfg.skip_eval:
            eval_output = stage_eval(cfg, tools, manifest_paths)
        else:
            banner(f"Stage 4/5 — Evaluate models [SKIPPED]{suffix}")

        graded_output: Path | None = None
        if not cfg.skip_grade and eval_output is not None:
            graded_output = stage_grade(cfg, eval_output)
        elif cfg.skip_grade:
            banner(f"Stage 5/5 — Grade [SKIPPED]{suffix}")
        else:
            banner(f"Stage 5/5 — Grade [SKIPPED — eval did not run]{suffix}")

        completed.append((cfg.name, eval_output, graded_output))

    elapsed = time.time() - t_start
    banner(f"Experiment complete in {elapsed:.1f}s")
    for name, eval_output, graded_output in completed:
        print(f"  run '{name}':")
        if eval_output:
            print(f"    eval results: {eval_output}")
        if graded_output:
            print(f"    graded:       {graded_output}")


if __name__ == "__main__":
    main()
