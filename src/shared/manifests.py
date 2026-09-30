"""Shared manifest discovery for pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def resolve_manifest_paths(
    instances_dir: Path,
    manifest_paths: Iterable[Path] | None = None,
) -> list[Path]:
    """Return a stable manifest list from an explicit batch or tree scan.

    An explicitly supplied empty iterable intentionally remains empty instead
    of falling back to a recursive scan. Runner-scoped batches rely on that
    distinction to avoid processing unrelated instances.
    """
    if manifest_paths is not None:
        return sorted(Path(path) for path in manifest_paths)
    return sorted(Path(instances_dir).glob("**/manifest.json"))
