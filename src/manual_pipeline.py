"""Build benchmark instances from user-provided questions and tables.

Bypasses Stages 1-3 of the standard pipeline (phenomenon injection,
validation, answer computation) by synthesizing manifest.json files
that look like already-completed instances. Stages 4-5 (eval and grade)
consume those manifest fields without requiring a registered phenomenon or
template.

Usage:
    uv run python src/manual_pipeline.py manual/my_questions.yaml
    uv run python src/manual_pipeline.py manual/my_questions.yaml --force
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from io_utils import reset_generated_instance_dir, save_json
from phenomena import TEMPLATE_TO_PHENOMENON
from shared.cli import configure_cli_streams
from shared.path_utils import safe_path_component


TEMPLATE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "templates" / "template_schema.json"
)


def _shell_quote(value: str) -> str:
    """Quote a dynamic command argument for the platform's documented shell."""
    if os.name == "nt":
        # README documents PowerShell on Windows. Single-quoted PowerShell
        # strings are literal; an embedded quote is represented by two quotes.
        return "'" + value.replace("'", "''") + "'"
    return shlex.quote(value)


def _load_valid_answer_formats() -> frozenset[str]:
    """Load the answer_format enum from the canonical template schema."""
    schema = json.loads(TEMPLATE_SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        answer_formats = schema["properties"]["answer_format"]["enum"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"template schema has no answer_format enum: {TEMPLATE_SCHEMA_PATH}"
        ) from exc

    if not isinstance(answer_formats, list) or not all(
        isinstance(value, str) for value in answer_formats
    ):
        raise ValueError(
            f"template schema answer_format enum must be a list of strings: "
            f"{TEMPLATE_SCHEMA_PATH}"
        )
    return frozenset(answer_formats)


VALID_ANSWER_FORMATS = _load_valid_answer_formats()

REQUIRED_TOP_LEVEL = ("dataset_name", "table", "target", "questions")
REQUIRED_QUESTION_FIELDS = ("id", "question", "answer", "answer_format")


def load_config(path: Path) -> dict:
    """Parse and validate a manual-questions YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    for field in REQUIRED_TOP_LEVEL:
        if field not in raw:
            raise ValueError(f"config missing required field: {field}")

    if not isinstance(raw["questions"], list) or not raw["questions"]:
        raise ValueError("'questions' must be a non-empty list")

    table_path = Path(raw["table"])
    if not table_path.is_absolute():
        table_path = (path.parent / table_path).resolve()
    raw["table"] = table_path

    seen_ids: set[str] = set()
    for i, q in enumerate(raw["questions"]):
        if not isinstance(q, dict):
            raise ValueError(f"questions[{i}] must be a mapping")
        for field in REQUIRED_QUESTION_FIELDS:
            if field not in q:
                raise ValueError(f"questions[{i}] missing required field: {field}")
        if q["id"] in seen_ids:
            raise ValueError(f"duplicate question id: {q['id']}")
        seen_ids.add(q["id"])
        if q["answer_format"] not in VALID_ANSWER_FORMATS:
            raise ValueError(
                f"questions[{i}] ('{q['id']}') has invalid answer_format "
                f"'{q['answer_format']}'. Valid: {sorted(VALID_ANSWER_FORMATS)}"
            )
        if q["id"] in TEMPLATE_TO_PHENOMENON:
            # answer_pipeline.py would otherwise dispatch to the registered
            # compute_answer for this template_id and overwrite the manual
            # answer. Better to fail loudly at synthesis.
            raise ValueError(
                f"questions[{i}] id '{q['id']}' collides with a registered "
                f"template_id; pick a unique name"
            )

    return raw


def build_instance(config: dict, instances_dir: Path, force: bool) -> Path:
    """Synthesize a clean instance directory and return its path.

    With ``force=True``, the exact generated instance directory is recreated so
    artifacts derived from an older table cannot survive the rebuild.
    """
    dataset_name = safe_path_component(
        config["dataset_name"],
        "dataset_name",
    )
    table_path: Path = config["table"]
    target = config["target"]

    if not table_path.exists():
        raise FileNotFoundError(f"table not found: {table_path}")

    df = pd.read_csv(table_path)
    if target not in df.columns:
        raise ValueError(
            f"target column '{target}' not in CSV. "
            f"Available: {sorted(df.columns.tolist())}"
        )

    instance_dir = instances_dir / dataset_name / "seed_0" / "manual"
    if instance_dir.exists() and not force:
        raise FileExistsError(
            f"instance dir already exists: {instance_dir}. "
            f"Use --force to overwrite."
        )

    # --force means a clean rebuild, not an in-place table/manifest overwrite,
    # so no generated state from an older table survives beside the new files.
    reset_generated_instance_dir(instance_dir, instances_dir)
    shutil.copy(table_path, instance_dir / "table.csv")

    qa_pairs = [
        {
            "template_id": q["id"],
            "category": "manual",
            "answer_format": q["answer_format"],
            "question": q["question"],
            "slot_assignments": {},
            "answer": q["answer"],
        }
        for q in config["questions"]
    ]

    manifest = {
        "dataset_name": dataset_name,
        "seed": 0,
        "base_dataset": str(table_path),
        "target": target,
        "phenomenon": {
            "injector_type": "manual",
            "params": {},
            "effects": {},
        },
        "templates_applied": [q["id"] for q in config["questions"]],
        "qa_pairs": qa_pairs,
        "validation": {
            "passed": True,
            "checks": [
                {
                    "name": "manual_instance",
                    "passed": True,
                    "metric": 0.0,
                    "threshold": 0.0,
                    "detail": "Authored by user; analytical validation skipped.",
                }
            ],
        },
    }

    save_json(manifest, instance_dir / "manifest.json")
    return instance_dir


def main() -> None:
    configure_cli_streams()
    parser = argparse.ArgumentParser(
        description="Build benchmark instances from user-provided questions"
    )
    parser.add_argument("config", type=Path, help="Path to manual questions YAML")
    parser.add_argument(
        "--force", action="store_true",
        help="Cleanly rebuild an existing manual instance directory",
    )
    parser.add_argument(
        "--instances-dir", type=Path, default=Path("data/instances"),
        help="Root directory containing instance folders (default: data/instances)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(args.config)
        instance_dir = build_instance(config, args.instances_dir, args.force)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    n_questions = len(config["questions"])
    dataset_name = instance_dir.parts[-3]

    print(f"Built manual instance: {instance_dir}")
    print(f"  dataset_name: {dataset_name}")
    print(f"  target:       {config['target']}")
    print(f"  questions:    {n_questions}")
    print()
    print("Note: running validate_pipeline.py / answer_pipeline.py against this")
    print("instance is a no-op (validation is pre-stamped, answers are pre-filled).")
    print("answer_pipeline will print SKIP rows; the manifest is not modified.")
    print()
    shell_name = "PowerShell" if os.name == "nt" else "POSIX shell"
    print(f"Next steps ({shell_name}):")
    eval_output = (
        f"data/results/eval_results_{dataset_name}_gpt-4o-mini.json"
    )
    dataset_arg = _shell_quote(dataset_name)
    output_arg = _shell_quote(eval_output)
    print("  uv run python src/eval_pipeline.py --models gpt-4o-mini "
          f"--dataset={dataset_arg} --injector manual --tools run_python "
          f"--output={output_arg}")
    print("  uv run python src/grade_pipeline.py "
          f"--input={output_arg}")


if __name__ == "__main__":
    main()
