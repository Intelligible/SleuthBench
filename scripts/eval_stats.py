"""Print timing and token stats from an eval results JSON file.

Usage:
    uv run python scripts/eval_stats.py data/results/eval_results_example.json
    uv run python scripts/eval_stats.py data/results/eval_results_example.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.cli import configure_cli_streams  # noqa: E402


def load_results(path: str, model_filter: str | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if model_filter:
        data = [e for e in data if e["model"] == model_filter]
    return data


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    values = sorted(values)
    n = len(values)
    return {
        "count": n,
        "mean": sum(values) / n,
        "median": values[n // 2],
        "min": values[0],
        "max": values[-1],
        "p25": values[int(n * 0.25)],
        "p75": values[int(n * 0.75)],
        "p95": values[int(n * 0.95)],
        "total": sum(values),
    }


def fmt(val, unit=""):
    if isinstance(val, int):
        return f"{val:,}{unit}"
    return f"{val:,.2f}{unit}"


def print_stats(label: str, s: dict):
    if s["count"] == 0:
        print(f"  {label}: no data")
        return
    print(f"  {label} (n={s['count']}):")
    print(f"    mean={fmt(s['mean'])}  median={fmt(s['median'])}  min={fmt(s['min'])}  max={fmt(s['max'])}")
    print(f"    p25={fmt(s['p25'])}  p75={fmt(s['p75'])}  p95={fmt(s['p95'])}  total={fmt(s['total'])}")


def analyze(entries: list[dict], label: str = "ALL"):
    latencies = []
    input_tokens = []
    output_tokens = []
    tool_counts = []
    errors = 0

    for e in entries:
        m = e.get("metrics", {})
        answer = str(e.get("model_answer", ""))
        if answer.startswith("ERROR"):
            errors += 1
            continue
        if m.get("total_latency_s") is not None:
            latencies.append(m["total_latency_s"])
        if m.get("input_tokens") is not None:
            input_tokens.append(m["input_tokens"])
        if m.get("output_tokens") is not None:
            output_tokens.append(m["output_tokens"])
        tool_counts.append(len(e.get("tool_use", [])))

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {len(entries)} entries, {errors} errors (excluded from stats)")
    print(f"{'=' * 60}")
    print_stats("Latency (s)", stats(latencies))
    print_stats("Input tokens", stats(input_tokens))
    print_stats("Output tokens", stats(output_tokens))
    print_stats("Tool calls", stats(tool_counts))
    if input_tokens and output_tokens:
        total_in = sum(input_tokens)
        total_out = sum(output_tokens)
        print(f"\n  Token totals: {total_in:,} in + {total_out:,} out = {total_in + total_out:,}")


def main():
    configure_cli_streams()
    parser = argparse.ArgumentParser(description="Eval results timing/token stats")
    parser.add_argument("input", help="Path to eval results JSON")
    parser.add_argument("--model", help="Filter to a single model")
    args = parser.parse_args()

    data = load_results(args.input, args.model)
    if not data:
        print("No entries found.")
        sys.exit(1)

    analyze(data, f"Overall — {Path(args.input).name}")

    by_model = defaultdict(list)
    for e in data:
        by_model[e["model"]].append(e)
    if len(by_model) > 1:
        for model in sorted(by_model):
            analyze(by_model[model], f"Model: {model}")

    by_injector = defaultdict(list)
    for e in data:
        by_injector[e["injector"]].append(e)
    if len(by_injector) > 1:
        print(f"\n{'=' * 60}")
        print(f"  Per-injector latency summary (mean seconds)")
        print(f"{'=' * 60}")
        rows = []
        for inj in sorted(by_injector):
            lats = [e["metrics"]["total_latency_s"] for e in by_injector[inj]
                    if not str(e.get("model_answer", "")).startswith("ERROR")
                    and e["metrics"].get("total_latency_s") is not None]
            if lats:
                rows.append((inj, len(lats), sum(lats) / len(lats)))
        for name, n, mean in sorted(rows, key=lambda r: -r[2]):
            print(f"  {name:<30s}  n={n:<4d}  mean={mean:.2f}s")


if __name__ == "__main__":
    main()
