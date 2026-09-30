"""Split an eval results JSON file into one file per template_id.

Usage:
    uv run python scripts/split_results_by_template.py data/results/eval_results_example.json outdir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from io_utils import save_json_atomic  # noqa: E402
from shared.cli import configure_cli_streams  # noqa: E402


_MAX_SLUG_LENGTH = 80
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _portable_slug(template_id: str) -> str:
    """Return a bounded, portable display portion for an output filename."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", template_id).strip("_-")
    slug = (slug or "template")[:_MAX_SLUG_LENGTH].rstrip("_-")
    slug = slug or "template"
    if slug.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        slug = f"template_{slug}"
    return slug


def _template_digest(template_id: str) -> str:
    """Hash the exact ID so distinct IDs do not share a sanitized filename."""
    return hashlib.sha256(
        template_id.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _allocate_output_filenames(template_ids) -> dict[str, str]:
    """Allocate deterministic filenames, including an explicit collision fallback."""
    filenames: dict[str, str] = {}
    used: set[str] = set()
    for template_id in sorted(template_ids):
        slug = _portable_slug(template_id)
        base = f"{slug}__{_template_digest(template_id)}"
        candidate = f"{base}.json"
        collision_index = 2
        while candidate.casefold() in used:
            candidate = f"{base}__{collision_index}.json"
            collision_index += 1
        used.add(candidate.casefold())
        filenames[template_id] = candidate
    return filenames


def _contained_output_path(outdir: Path, filename: str) -> Path:
    """Resolve one direct child of outdir and reject traversal or symlinks."""
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise ValueError(f"unsafe output filename: {filename!r}")

    resolved_outdir = outdir.resolve()
    candidate = resolved_outdir / filename
    if candidate.is_symlink():
        raise ValueError(f"refusing to replace symlink output: {candidate}")
    try:
        relative = candidate.resolve().relative_to(resolved_outdir)
    except ValueError as exc:
        raise ValueError(
            f"output path escapes output directory: {candidate}"
        ) from exc
    if len(relative.parts) != 1:
        raise ValueError(f"output path is not a direct child: {candidate}")
    return candidate


def split_results(input_path: Path, outdir: Path) -> dict[str, Path]:
    """Split input results and return the path allocated to each template ID."""
    with Path(input_path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input results must be a JSON array")

    by_template: dict[str, list[dict]] = defaultdict(list)
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"result entry {index} must be a JSON object")
        template_id = entry.get("template_id")
        if not isinstance(template_id, str):
            raise ValueError(
                f"result entry {index} has a non-string template_id"
            )
        by_template[template_id].append(entry)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    filenames = _allocate_output_filenames(by_template)
    output_paths: dict[str, Path] = {}

    for template_id, entries in sorted(by_template.items()):
        out_path = _contained_output_path(outdir, filenames[template_id])
        save_json_atomic(entries, out_path)
        output_paths[template_id] = out_path
        print(
            f"  {template_id!r}: {len(entries)} entries -> {out_path}"
        )

    print(f"\nSplit {len(data)} entries into {len(by_template)} files.")
    return output_paths


def main():
    configure_cli_streams()
    parser = argparse.ArgumentParser(description="Split eval results by template_id")
    parser.add_argument("input", help="Path to eval results JSON")
    parser.add_argument("outdir", help="Output directory")
    args = parser.parse_args()

    split_results(Path(args.input), Path(args.outdir))


if __name__ == "__main__":
    main()
