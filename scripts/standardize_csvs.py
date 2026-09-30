import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.cli import configure_cli_streams  # noqa: E402


def _parse_int(value: str) -> Optional[int]:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)
        except ValueError:
            return None
        return int(f) if f.is_integer() else None


def _find_row_id_column(rows: List[Dict[str, str]], columns: List[str]) -> Optional[str]:
    for col in columns:
        for base in (0, 1):
            ok = True
            for row_idx, row in enumerate(rows, start=1):
                val = _parse_int(row.get(col, ""))
                if val is None or val != row_idx - base:
                    ok = False
                    break
            if ok:
                return col
    return None


def _read_rows(path: Path, limit: int = 10_000) -> tuple[List[Dict[str, str]], List[str]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return [], []
        # Drop empty-string fieldnames (from leading/trailing commas)
        fieldnames = [f for f in reader.fieldnames if f.strip()]
        rows = []
        for _, row in zip(range(limit), reader):
            cleaned = {k: v for k, v in row.items() if k.strip()}
            rows.append(cleaned)
        return rows, fieldnames


def _write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_cli_streams()
    parser = argparse.ArgumentParser(description="Standardize base CSVs into multiple row-count variants")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        metavar="N",
        help="Row counts to generate (default: 100 500 1000)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific CSV files to process (default: all CSVs in base directory)",
    )
    args = parser.parse_args()
    sizes = sorted(args.sizes)

    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "data" / "base"
    standardized_dir = repo_root / "data" / "standardized"
    standardized_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        csv_paths = [Path(f) for f in args.files]
    else:
        csv_paths = sorted(base_dir.glob("*.csv"))

    for csv_path in csv_paths:
        print(f"{csv_path.name}")
        rows, fieldnames = _read_rows(csv_path, limit=max(sizes))
        if not fieldnames:
            continue

        id_col = _find_row_id_column(rows, fieldnames)
        add_row_id = id_col is None

        if add_row_id:
            out_fields = ["row_id"] + fieldnames
        elif id_col != "row_id":
            out_fields = ["row_id" if name == id_col else name for name in fieldnames]
        else:
            out_fields = fieldnames

        all_out_rows = []
        for row_idx, row in enumerate(rows, start=1):
            row = dict(row)
            if id_col and id_col != "row_id":
                row["row_id"] = row.pop(id_col, "")
            if add_row_id:
                row["row_id"] = str(row_idx)
            all_out_rows.append(row)

        for size in sizes:
            out_rows = all_out_rows[:size]
            if len(out_rows) < size:
                print(f"  Warning: {csv_path.name} has only {len(out_rows)} rows, skipping {size}-row version")
                continue
            out_name = f"{csv_path.stem}_{size}{csv_path.suffix}"
            _write_rows(standardized_dir / out_name, out_rows, out_fields)
            print(f"  Wrote {out_name}")


if __name__ == "__main__":
    main()
