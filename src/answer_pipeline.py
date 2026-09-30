"""Compute ground-truth answers for validated benchmark instances.

Walks data/instances/**/manifest.json, dispatches each QA pair's
template_id through phenomena.TEMPLATE_TO_PHENOMENON to the
matching answer computer, and writes the result into the manifest's
qa_pairs[].answer field in place.

Skips any instance that lacks a validation record or whose validation failed —
answers are only computed for injections proven deterministic. A skipped
instance also has stale answers from any earlier validation PASS cleared.

Usage:
    uv run python src/answer_pipeline.py
    uv run python src/answer_pipeline.py --instances-dir data/instances/runs/example
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from phenomena import TEMPLATE_TO_PHENOMENON
from phenomena._base import AnswerUnavailable
from io_utils import load_csv, load_json, save_json
from shared.cli import configure_cli_streams
from shared.manifests import resolve_manifest_paths
from shared.table import print_table


class AnswerComputationErrors(RuntimeError):
    """One or more registered answer computers failed."""

    def __init__(
        self,
        failures: list[tuple[Path, str, Exception]],
        rows: List[List[str]] | None = None,
    ) -> None:
        self.failures = list(failures)
        self.rows = list(rows or [])
        manifest_path, template_id, exc = self.failures[0]
        first = (
            f"{manifest_path}/{template_id}: "
            f"{type(exc).__name__}: {exc}"
        )
        super().__init__(
            f"Answer computation failed for {len(self.failures)} QA pair(s); "
            f"first: {first}"
        )


def _compute_answer_safe(
    template_id: str,
    df,
    slot_assignments: Dict[str, str],
    effects: Dict[str, Any],
) -> tuple[Any, str, Exception | None]:
    """Try to compute an answer; return (answer, status, error).

    status is one of "OK", "SKIP", "NO_GOLD", or "ERROR". "NO_GOLD" means the
    answer computer raised ``AnswerUnavailable``: an expected outcome that
    clears the answer without failing the stage.
    """
    phenomenon = TEMPLATE_TO_PHENOMENON.get(template_id)
    if phenomenon is None:
        return None, "SKIP", None
    try:
        compute_fn = phenomenon.compute_answers[template_id]
        answer = compute_fn(df, slot_assignments, effects)
        return answer, "OK", None
    except NotImplementedError:
        return None, "SKIP", None
    except AnswerUnavailable as exc:
        print(f"  NO GOLD for {template_id}: {exc}")
        return None, "NO_GOLD", None
    except Exception as exc:
        print(
            f"  ERROR computing {template_id}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None, "ERROR", exc


def process_instance(manifest_path: Path) -> List[List[str]]:
    """Process a single instance directory. Returns table rows."""
    manifest = load_json(manifest_path)
    instance_dir = manifest_path.parent

    dataset_name = manifest["dataset_name"]
    injector_type = manifest["phenomenon"]["injector_type"]

    # Skip invalid instances and clear any answers left by an earlier PASS.
    validation = manifest.get("validation")
    if validation is None or not validation.get("passed", False):
        updated = False
        for qa in manifest.get("qa_pairs", []):
            is_authored_manual_answer = (
                injector_type == "manual"
                and qa.get("category") == "manual"
            )
            if not is_authored_manual_answer and qa.get("answer") is not None:
                qa["answer"] = None
                updated = True
        if updated:
            save_json(manifest, manifest_path)
        reason = "not validated" if validation is None else "validation failed"
        return [[dataset_name, injector_type, "-", "-", f"SKIP ({reason})"]]

    df = load_csv(instance_dir / "table.csv")

    effects = manifest["phenomenon"]["effects"]

    rows: List[List[str]] = []
    failures: list[tuple[Path, str, Exception]] = []
    updated = False

    for qa in manifest["qa_pairs"]:
        template_id = qa["template_id"]
        is_authored_manual_answer = (
            injector_type == "manual"
            and qa.get("category") == "manual"
        )
        answer, status, error = _compute_answer_safe(
            template_id, df, qa["slot_assignments"], effects,
        )
        if status == "OK":
            qa["answer"] = answer
            updated = True
        elif status in ("SKIP", "NO_GOLD") and not is_authored_manual_answer:
            # A template registration or implementation can disappear after a
            # prior successful run.  Clear its now-unverifiable ground truth
            # instead of silently evaluating against the stale value.  Manual
            # instances are explicitly authored and intentionally have no
            # registry entry, so their supplied answers remain untouched.
            # NO_GOLD (AnswerUnavailable) likewise clears a stale answer so
            # evaluation skips the QA pair, without failing the stage.
            if qa.get("answer") is not None:
                qa["answer"] = None
                updated = True
        elif status == "ERROR":
            qa["answer"] = None
            updated = True
            assert error is not None
            failures.append((manifest_path, template_id, error))
        rows.append([dataset_name, injector_type, template_id, repr(answer), status])

    if updated:
        save_json(manifest, manifest_path)

    if failures:
        raise AnswerComputationErrors(failures, rows) from failures[0][2]
    return rows


def run(
    instances_dir: Path,
    manifest_paths: list[Path] | None = None,
) -> None:
    """Discover all instances and compute answers."""
    manifests = resolve_manifest_paths(instances_dir, manifest_paths)
    if not manifests:
        print(f"No manifest.json files found under {instances_dir}")
        sys.exit(1)

    print(f"Found {len(manifests)} instance(s) under {instances_dir}\n")

    all_rows: List[List[str]] = []
    failures: list[tuple[Path, str, Exception]] = []
    for mp in manifests:
        try:
            all_rows.extend(process_instance(mp))
        except AnswerComputationErrors as exc:
            all_rows.extend(exc.rows)
            failures.extend(exc.failures)

    print_table(
        ["Dataset", "Injector", "Template", "Answer", "Status"],
        all_rows,
    )

    counts = {}
    for row in all_rows:
        status = row[-1]
        counts[status] = counts.get(status, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"\nTotal: {len(all_rows)} QA pairs ({summary})")
    if failures:
        raise AnswerComputationErrors(failures, all_rows) from failures[0][2]


def main() -> None:
    configure_cli_streams()
    parser = argparse.ArgumentParser(description="Compute answers for generated instances")
    parser.add_argument(
        "--instances-dir",
        type=str,
        default="data/instances",
        help="Root directory containing instance folders (default: data/instances)",
    )
    args = parser.parse_args()
    run(Path(args.instances_dir))


if __name__ == "__main__":
    main()
