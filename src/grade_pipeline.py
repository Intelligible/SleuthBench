"""Grade eval results with deterministic checks plus an LLM judge.

Reads an eval-results JSON produced by eval_pipeline.py. Errors, empty answers,
and supported numeric formats are graded deterministically; remaining answers
are compared with ground truth by an OpenAI judge. The output adds grade
(CORRECT / PARTIAL / INCORRECT) and reasoning fields to each entry.

``--model`` selects the OpenAI judge model; it does not filter result entries.

Usage:
    uv run python src/grade_pipeline.py --input data/results/eval_results_cluster.json
    uv run python src/grade_pipeline.py --input data/results/eval_results_cluster.json --output data/results/graded.json --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from io_utils import (
    JsonResultLifecycle,
    validate_atomic_output_target,
)
from shared.cli import configure_cli_streams


# Relative tolerance for standalone numeric answers and the numeric component
# of ``column_value`` answers.
NUMERIC_RELATIVE_TOLERANCE = 0.05
NUMERIC_ANSWER_FORMATS = {"value"}
COLUMN_VALUE_ANSWER_FORMATS = {"column_value"}

_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_AMBIGUOUS_TAIL_RE = re.compile(
    r"\b(?:not|no|wrong|incorrect|instead|rather|but|however|although|though|or)\b",
    re.IGNORECASE,
)
_ADDITIONAL_PAIRED_VALUE_RE = re.compile(
    r"(?:[,=:]|\b(?:is|at)\b)\s*" + _NUM,
    re.IGNORECASE,
)


def _to_float(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _extract_strict_number(text: str) -> float | None:
    """Parse a standalone numeric answer; free-form text defers to the judge."""
    if not isinstance(text, str):
        return None
    clean = (
        text.replace("*", "")
        .replace("`", "")
        .replace(",", "")
        .strip()
    )
    match = re.fullmatch(_NUM + r"\s*[.!]?", clean)
    return _to_float(match.group(1)) if match else None


def _relative_tolerance_check(
    expected: float,
    actual: float,
    tolerance: float,
) -> tuple[float, bool]:
    """Return the relative/zero-safe error and whether it meets tolerance."""
    error = (
        abs(actual)
        if expected == 0
        else abs(actual - expected) / abs(expected)
    )
    return error, error <= tolerance


def grade_numeric_value(
    expected, model_answer: str, tol: float = NUMERIC_RELATIVE_TOLERANCE
) -> tuple[str, str] | None:
    """Deterministically grade a numeric "value" answer with relative tolerance.

    Returns (grade, reasoning), or None to defer to the LLM judge when the
    expected value isn't numeric or the model's number can't be extracted.
    """
    exp = _to_float(expected)
    if exp is None:
        return None
    got = _extract_strict_number(model_answer)
    if got is None:
        return None  # non-canonical or ambiguous — let the LLM judge handle it
    rel, ok = _relative_tolerance_check(exp, got, tol)
    grade = "CORRECT" if ok else "INCORRECT"
    return grade, (
        f"Numeric value check (no LLM): model={got}, expected={exp}, "
        f"relative error={rel:.4f}, tolerance={tol:.0%} -> {grade}."
    )


def _parse_expected_column_value(expected) -> tuple[str, float] | None:
    """Parse the canonical ``column, value`` expected-answer representation."""
    if not isinstance(expected, str):
        return None
    clean = expected.replace("*", "").replace("`", "").strip()
    match = re.fullmatch(r"(.+?)\s*,\s*" + _NUM + r"\s*", clean)
    if match is None:
        return None
    value = _to_float(match.group(2))
    if value is None:
        return None
    return match.group(1).strip(), value


def _extract_value_for_column(text: str, column: str) -> float | None:
    """Parse a leading column/value answer unless its tail is ambiguous."""
    if not isinstance(text, str):
        return None
    clean = text.replace("*", "").replace("`", "").strip()
    flexible_column = r"\s+".join(re.escape(part) for part in column.split())
    connector = (
        r"(?:"
        r"\s*[,=:]\s*"
        r"|\s+(?:is|at|peaks?\s+at|threshold\s+(?:is|at))\s+"
        r")"
    )
    match = re.match(
        rf"{flexible_column}(?!\w){connector}" + _NUM + r"(?=$|[\s,;.!?])",
        clean,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None

    # A canonical pair may be followed by a neutral explanation, but explicit
    # negation, contrast, alternatives, or another paired value make the
    # assertion semantically ambiguous. Defer those cases to the LLM judge.
    tail = clean[match.end():]
    if (
        _AMBIGUOUS_TAIL_RE.search(tail)
        or _ADDITIONAL_PAIRED_VALUE_RE.search(tail)
    ):
        return None
    return _to_float(match.group(1))


def grade_column_value(
    expected, model_answer: str, tol: float = NUMERIC_RELATIVE_TOLERANCE
) -> tuple[str, str] | None:
    """Grade ``column, value`` deterministically with exact column + tolerance."""
    parsed = _parse_expected_column_value(expected)
    if parsed is None:
        return None
    expected_column, exp = parsed
    got = _extract_value_for_column(model_answer, expected_column)
    if got is None:
        return None

    rel, ok = _relative_tolerance_check(exp, got, tol)
    grade = "CORRECT" if ok else "PARTIAL"
    return grade, (
        f"Column+value check (no LLM): column={expected_column!r} matched, "
        f"model value={got}, expected value={exp}, relative error={rel:.4f}, "
        f"tolerance={tol:.0%} -> {grade}."
    )


GRADE_SYSTEM_PROMPT = """\
You are grading an LLM's answer to a data analysis question about a CSV dataset.

You will be given:
- The question asked
- The expected correct answer
- The model's answer

Grade the response as exactly one of:
- CORRECT: The answer matches the expected answer (allow minor formatting differences, \
extra explanation is fine as long as the core answer is right). For a single numeric \
value, treat it as CORRECT when it is within 5% relative error of the expected value.
- PARTIAL: The answer is partially correct (e.g. some items in a list are right, \
or the direction is right but the specific value is wrong)
- INCORRECT: The answer is wrong, irrelevant, or doesn't address the question

Return JSON with "grade" and "reasoning" fields.
"""


def grade_entry(client, model: str, entry: dict) -> tuple[str, str]:
    """Return (grade, reasoning) for a single result entry."""
    model_answer = entry.get("model_answer", "")

    # Auto-grade errors and empty answers without calling the API
    if isinstance(model_answer, str) and model_answer.startswith("ERROR:"):
        return "INCORRECT", "Request errored or timed out — no answer produced."
    if not model_answer or (isinstance(model_answer, str) and not model_answer.strip()):
        return "INCORRECT", "Model produced no answer."

    expected = entry["expected_answer"]

    # Deterministically grade canonical numbers; ambiguous text uses the judge.
    if entry.get("answer_format") in NUMERIC_ANSWER_FORMATS:
        numeric = grade_numeric_value(expected, model_answer)
        if numeric is not None:
            return numeric

    # Deterministically grade canonical column/value pairs; ambiguity uses the judge.
    if entry.get("answer_format") in COLUMN_VALUE_ANSWER_FORMATS:
        column_value = grade_column_value(expected, model_answer)
        if column_value is not None:
            return column_value

    user_content = (
        f"Question: {entry['question']}\n\n"
        f"Expected answer: {json.dumps(expected)}\n\n"
        f"Model answer: {model_answer}"
    )

    resp = client.chat.completions.create(
        model=model,
        timeout=60,
        messages=[
            {"role": "system", "content": GRADE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "grade",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "grade": {
                            "type": "string",
                            "enum": ["CORRECT", "PARTIAL", "INCORRECT"],
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": ["grade", "reasoning"],
                    "additionalProperties": False,
                },
            },
        },
    )
    result = json.loads(resp.choices[0].message.content)
    return result["grade"], result["reasoning"]


def run(
    input_path: Path,
    output_path: Path | None = None,
    model: str = "gpt-4o-mini",
) -> Path:
    """Grade eval results. Returns the path written."""
    if output_path is None:
        output_path = Path("data/results/graded") / f"{input_path.stem}_graded{input_path.suffix}"
    validate_atomic_output_target(output_path)

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key)

    entries = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"Grading {len(entries)} entries with {model}...")

    graded = []
    output_lifecycle = JsonResultLifecycle(
        output_path,
        kind="grade",
        records_key="graded",
    )
    output_lifecycle.checkpoint(graded)
    for i, entry in enumerate(entries):
        grade, reasoning = grade_entry(client, model, entry)
        graded_entry = {**entry, "grade": grade, "reasoning": reasoning}
        graded.append(graded_entry)
        print(f"  [{i + 1}/{len(entries)}] {entry['model']} / {entry['dataset']} / {entry['injector']} → {grade}")
        output_lifecycle.checkpoint(graded)

    # Publish only after the entire input has been graded. On failure, the
    # previous complete output remains in place and the explicitly marked
    # partial checkpoint keeps the recoverable progress.
    output_lifecycle.publish(graded)
    print(f"\nWrote {len(graded)} graded entries to {output_path}")

    from collections import Counter
    by_model: dict[str, Counter] = {}
    for e in graded:
        m = e["model"]
        if m not in by_model:
            by_model[m] = Counter()
        by_model[m][e["grade"]] += 1

    print("\nSummary:")
    for m, counts in sorted(by_model.items()):
        total = sum(counts.values())
        print(f"  {m}: {counts['CORRECT']}C / {counts['PARTIAL']}P / {counts['INCORRECT']}I  (n={total})")

    return output_path


def main() -> None:
    configure_cli_streams()
    parser = argparse.ArgumentParser(description="Grade eval results with an LLM")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to eval results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to write graded results "
            "(default: data/results/graded/<input-stem>_graded.<suffix>)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for grading (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    run(
        Path(args.input),
        Path(args.output) if args.output else None,
        args.model,
    )


if __name__ == "__main__":
    main()
