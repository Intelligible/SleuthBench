"""Verify each injection is the deterministic, unambiguous answer.

Walks data/instances/**/manifest.json and dispatches through
phenomena.PHENOMENA to the matching validator, which runs dominance
checks and statistical tests (does the injected feature actually
dominate?  is the shape detectable?  are other features plausible
distractors?).

Writes the ValidationResult — passed + per-check detail — into the
manifest's validation field.  Downstream pipelines skip instances that
haven't been validated or whose validation failed.

Usage:
    uv run python src/validate_pipeline.py                                # all unvalidated instances
    uv run python src/validate_pipeline.py --dataset bike_sharing_100     # filter by dataset
    uv run python src/validate_pipeline.py --injector fc_nonmonotone_peak # filter by injector
    uv run python src/validate_pipeline.py --force                        # re-validate already-validated
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from phenomena import CheckDetail, PHENOMENA, ValidationResult
from io_utils import load_csv, load_json, save_json
from shared.cli import configure_cli_streams
from shared.manifests import resolve_manifest_paths
from shared.path_utils import normalize_path_text
from shared.table import print_table


def _is_authored_manual_manifest(manifest: dict) -> bool:
    qa_pairs = manifest.get("qa_pairs") or []
    return (
        manifest.get("phenomenon", {}).get("injector_type") == "manual"
        and bool(qa_pairs)
        and all(qa.get("category") == "manual" for qa in qa_pairs)
    )


def validate_instance(manifest_path: Path) -> List[str]:
    """Validate a single instance. Returns a table row."""
    manifest = load_json(manifest_path)
    instance_dir = manifest_path.parent

    dataset_name = manifest["dataset_name"]
    injector_type = manifest["phenomenon"]["injector_type"]
    effects = manifest["phenomenon"]["effects"]
    target_col = manifest["target"]

    phenomenon = PHENOMENA.get(injector_type)
    validator = getattr(phenomenon, "validate", None)
    if not callable(validator):
        if _is_authored_manual_manifest(manifest):
            return [
                dataset_name,
                injector_type,
                "SKIP",
                "authored manual validation",
            ]
        # A registry entry can disappear (or temporarily lose its validator)
        # after this instance was validated.  Never leave the old PASS in
        # place: downstream stages use that field as their eligibility gate.
        if manifest.get("validation") is not None:
            manifest["validation"] = None
            save_json(manifest, manifest_path)
        return [dataset_name, injector_type, "SKIP", "no validator registered"]

    # Revalidation is a replacement, not an in-place update. Invalidate the
    # previous result before loading data or invoking the registered validator
    # so any crash cannot leave an old PASS eligible for downstream stages.
    if manifest.get("validation") is not None:
        manifest["validation"] = None
        save_json(manifest, manifest_path)

    df = load_csv(instance_dir / "table.csv")

    try:
        result = validator(df, effects, target_col)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        # Known serializable crash shapes include malformed effects, missing
        # columns, and bad data types. They are ERRORs (not ordinary validation
        # FAILs); preserve the detail in the manifest before the batch exits
        # nonzero.
        error_msg = f"{type(exc).__name__}: {exc}"
        manifest["validation"] = asdict(ValidationResult(
            passed=False,
            checks=[CheckDetail(
                name="validator_crashed", passed=False,
                metric=0.0, threshold=0.0,
                detail=error_msg,
            )],
        ))
        save_json(manifest, manifest_path)
        return [dataset_name, injector_type, "ERROR", error_msg[:60]]

    manifest["validation"] = asdict(result)
    save_json(manifest, manifest_path)

    status = "PASS" if result.passed else "FAIL"
    detail = ""
    if not result.passed and result.checks:
        failed = [c for c in result.checks if not c.passed]
        if failed:
            detail = failed[-1].detail[:60]

    return [dataset_name, injector_type, status, detail]


def run(
    instances_dir: Path,
    dataset_filter: str | None,
    injector_filter: list[str] | None,
    force: bool,
    manifest_paths: list[Path] | None = None,
) -> None:
    """Discover all instances and run validators."""
    if dataset_filter is not None:
        dataset_filter = normalize_path_text(dataset_filter)
    manifests = resolve_manifest_paths(instances_dir, manifest_paths)
    if not manifests:
        print(f"No manifest.json files found under {instances_dir}")
        sys.exit(1)

    print(f"Found {len(manifests)} instance(s) under {instances_dir}\n")

    all_rows: List[List[str]] = []
    skipped = 0

    for mp in manifests:
        manifest = load_json(mp)
        dataset_name = normalize_path_text(
            manifest.get("dataset_name", "")
        )
        injector_type = manifest.get("phenomenon", {}).get("injector_type", "")

        if dataset_filter and dataset_filter not in dataset_name:
            skipped += 1
            continue
        if injector_filter and injector_type not in injector_filter:
            skipped += 1
            continue
        phenomenon = PHENOMENA.get(injector_type)
        validator = getattr(phenomenon, "validate", None)
        # Registry removal must invalidate an old result even on the normal
        # idempotent path; otherwise the stale PASS would bypass
        # validate_instance() forever unless the caller happened to use
        # --force.
        if not callable(validator) and not _is_authored_manual_manifest(manifest):
            all_rows.append(validate_instance(mp))
            continue
        if not force and manifest.get("validation") is not None:
            skipped += 1
            continue

        all_rows.append(validate_instance(mp))

    print_table(
        ["Dataset", "Injector", "Status", "Detail"],
        all_rows,
    )

    passed = sum(1 for r in all_rows if r[2] == "PASS")
    failed = sum(1 for r in all_rows if r[2] == "FAIL")
    errors = sum(1 for r in all_rows if r[2] == "ERROR")
    print(f"\nValidated {len(all_rows)} instances: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")

    # Validator crashes indicate a code bug or malformed manifest, not a
    # legitimate "this injection wasn't dominant" outcome — exit nonzero
    # so CI/shell pipelines can detect it. FAIL is expected and does not
    # trigger a nonzero exit.
    if errors > 0:
        sys.exit(1)


def main() -> None:
    configure_cli_streams()
    parser = argparse.ArgumentParser(description="Validate injected instances are deterministic")
    parser.add_argument(
        "--instances-dir",
        type=str,
        default="data/instances",
        help="Root directory containing instance folders (default: data/instances)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only validate instances matching this dataset name substring",
    )
    parser.add_argument(
        "--injector",
        nargs="+",
        default=None,
        help="Only validate instances for these injector types",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-validate instances that already have a validation result",
    )
    args = parser.parse_args()
    run(Path(args.instances_dir), args.dataset, args.injector, args.force)


if __name__ == "__main__":
    main()
