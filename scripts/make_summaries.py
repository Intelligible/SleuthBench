import argparse
import csv
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.cli import configure_cli_streams  # noqa: E402


_MISSING_TOKENS = {
    "na", "n/a", "nan", "null", "none", "nil", "missing", "?", "-", "--", "undefined"
}

_BOOL_TRUE = {"true", "t", "yes", "y", "on", "1"}
_BOOL_FALSE = {"false", "f", "no", "n", "off", "0"}
_BOOL_TOKENS = _BOOL_TRUE | _BOOL_FALSE
_ID_LIKE_NAMES = {"row_id", "id", "index", "uid", "uuid"}

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

_DATETIME_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
]

_CURRENCY_RE = re.compile(r"[\$\€\£\¥]")
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(\D|$))")


def _clean(s: str) -> str:
    return s.strip().lower()


def _is_missing(s: str) -> bool:
    return _clean(s) in _MISSING_TOKENS


def _normalize_numeric(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s = "-" + s[1:-1].strip()
    s = _CURRENCY_RE.sub("", s)
    s = _THOUSANDS_COMMA_RE.sub("", s)
    if s.endswith("%"):
        s = s[:-1].strip()
    return s.strip()


def _parse_datetime(s: str) -> Optional[datetime]:
    if len(s) > 32:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _try_parse_int(v: str) -> Optional[int]:
    if _is_missing(v):
        return None
    norm = _normalize_numeric(v)
    if _INT_RE.match(norm):
        try:
            return int(norm)
        except ValueError:
            return None
    return None


def _sample_column(values: List[Optional[str]], col_name: str, limit: int = 100) -> List[str]:
    non_empty = [(i, v) for i, v in enumerate(values) if v is not None and v.strip()]
    if not non_empty:
        return []
    rng = random.Random(col_name)
    rng.shuffle(non_empty)
    return [v for _, v in non_empty[:limit]]


def _read_columns(path: Path) -> Dict[str, List[Optional[str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        cols: Dict[str, List[Optional[str]]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                val = row.get(name)
                cols[name].append(None if val is None else val.strip())
    return cols


def _infer_dtype(sampled: List[str]) -> str:
    filtered = [v for v in sampled if not _is_missing(v)]
    if not filtered:
        return "empty"

    lowered = {_clean(v) for v in filtered}
    is_bool_col = lowered.issubset(_BOOL_TOKENS) and len(lowered) <= 6

    counts = {"bool": 0, "int": 0, "float": 0, "date": 0, "datetime": 0, "string": 0}
    for raw in filtered:
        st = raw.strip()
        if not st:
            continue
        token = _clean(st)
        if is_bool_col and token in _BOOL_TOKENS:
            counts["bool"] += 1
            continue
        dt = _parse_datetime(st)
        if dt is not None:
            has_time = any(ch in st for ch in [":", "T", "Z"]) or (" " in st and ":" in st)
            counts["datetime" if has_time else "date"] += 1
            continue
        norm = _normalize_numeric(st)
        if _INT_RE.match(norm):
            counts["int"] += 1
            continue
        if _FLOAT_RE.match(norm):
            counts["float"] += 1
            continue
        counts["string"] += 1

    total = sum(counts.values())
    if total == 0:
        return "empty"

    if counts["datetime"] + counts["date"] > 0 and (counts["datetime"] + counts["date"]) / total >= 0.9:
        return "datetime"

    dom = max(counts, key=counts.__getitem__)
    dom_ratio = counts[dom] / total

    if dom_ratio >= 0.9:
        if dom in ("int", "float"):
            numeric = counts["int"] + counts["float"]
            if numeric / total >= 0.9:
                return "float" if counts["float"] > 0 else "int"
        return dom

    numeric = counts["int"] + counts["float"]
    if numeric / total >= 0.9 and counts["bool"] + counts["date"] + counts["datetime"] + counts["string"] == 0:
        return "float" if counts["float"] > 0 else "int"

    return "mixed"


def _is_likely_regression(values: List[str]) -> bool:
    nums: List[float] = []
    for v in values:
        try:
            num = float(v)
        except (TypeError, ValueError):
            return False
        if math.isnan(num) or math.isinf(num):
            return False
        nums.append(num)
    if len(nums) < 2:
        return False

    unique = len(set(nums))
    total = len(nums)
    if unique / total > 0.5:
        return True
    if any(v != math.floor(v) for v in nums) and unique > 10:
        return True
    if unique > 20:
        return True
    return False


def _is_sequential_id(values: List[Optional[str]], check_n: int = 100) -> bool:
    ints: List[int] = []
    for v in values:
        if v is None or not v.strip():
            continue
        parsed = _try_parse_int(v)
        if parsed is None:
            return False
        ints.append(parsed)
        if len(ints) >= check_n:
            break
    if len(ints) < min(20, len(values)):
        return False
    return any(
        all(v == i + base for i, v in enumerate(ints))
        for base in (0, 1)
    )


def _classify_kind(col_name: str, values: List[Optional[str]], sampled: List[str], dtype: str) -> Tuple[str, int]:
    # unique_count is capped at 100 (sample size) to save work,
    # since no template constraint requires unique_count larger than 100
    unique_count = len(set(sampled))

    if col_name.lower() in _ID_LIKE_NAMES or _is_sequential_id(values):
        return "id_no", unique_count

    if unique_count == 2:
        return "binary", unique_count

    if dtype in ("int", "bool"):
        ints = [_try_parse_int(v) for v in sampled]
        ints = [x for x in ints if x is not None]
        if ints:
            rng = max(ints) - min(ints)
            if rng <= 31 and unique_count <= 64:
                return "categorical", unique_count

    if unique_count > 2 and _is_likely_regression(sampled):
        return "regression", unique_count

    return "categorical", unique_count


def main() -> None:
    configure_cli_streams()
    parser = argparse.ArgumentParser(description="Generate column summaries for standardized CSV datasets.")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific CSV files to process (default: all CSVs in standardized directory)",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Rewrite summaries even if they already exist.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    standardized_dir = repo_root / "data" / "standardized"
    summaries_dir = standardized_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        csv_paths = [Path(f) for f in args.files]
    else:
        csv_paths = sorted(standardized_dir.glob("*.csv"))

    for csv_path in csv_paths:
        out_path = summaries_dir / f"{csv_path.stem}.json"
        if out_path.exists() and not args.rewrite:
            continue
        cols = _read_columns(csv_path)
        if not cols:
            continue

        columns_out: Dict[str, dict] = {}
        by_kind: Dict[str, List[str]] = {"binary": [], "regression": [], "categorical": [], "id_no": []}

        for col_name, values in cols.items():
            sampled = _sample_column(values, col_name)
            dtype = _infer_dtype(sampled)
            kind, unique_count = _classify_kind(col_name, values, sampled, dtype)
            columns_out[col_name] = {"kind": kind, "dtype": dtype, "unique_count": unique_count}
            by_kind[kind].append(col_name)

        for k in by_kind:
            by_kind[k] = sorted(by_kind[k])

        last_col = list(cols.keys())[-1]
        payload = {
            "dataset": csv_path.name,
            "target": last_col,
            "columns": columns_out,
            "by_kind": by_kind,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
