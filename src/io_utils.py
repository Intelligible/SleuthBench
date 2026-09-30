"""Shared I/O helpers for CSV and JSON.

Thin wrappers over pandas.read_csv / pd.to_csv and the stdlib json module
that every pipeline uses so serialization behavior stays consistent.
save_json installs a numpy-aware default encoder so numpy.bool_,
numpy.integer, numpy.floating, and ndarray values produced by
pandas/numpy operations serialize cleanly — without this, validators
that return bool-typed check results would blow up json.dumps on write.
The guarded instance-directory helpers reset or remove generated artifacts
while refusing targets outside or too close to the configured instances root.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from shared.path_utils import (
    PORTABLE_MAX_COMPONENT_BYTES,
    WINDOWS_LEGACY_MAX_PATH_UNITS,
    WINDOWS_MAX_COMPONENT_UNITS,
    truncate_to_utf16_units,
    windows_utf16_units,
)


# Win32's legacy MAX_PATH includes the terminating NUL, leaving 259 usable
# UTF-16 code units. Python installations can still hit this limit when the OS
# or process is not long-path enabled. Keep atomic sidecar names within both
# that full-path boundary and Windows' 255-unit filename-component boundary.


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _validate_generated_instance_path(
    instance_dir: str | Path,
    instances_root: str | Path,
    operation: str,
) -> Path:
    """Validate that a generated-instance target is narrow and contained."""
    instance_dir = Path(instance_dir)
    instances_root = Path(instances_root)
    lexical_root = instances_root.absolute()
    lexical_instance = instance_dir.absolute()
    resolved_root = instances_root.resolve()
    resolved_instance = instance_dir.resolve()

    relative: Path | None = None
    for candidate_root in (lexical_root, resolved_root):
        try:
            relative = lexical_instance.relative_to(candidate_root)
            break
        except ValueError:
            continue
    if relative is None:
        raise ValueError(
            f"instance directory must stay under instances root: {instance_dir}"
        )

    expected_resolved = resolved_root.joinpath(*relative.parts)
    if resolved_instance != expected_resolved:
        raise ValueError(
            f"refusing to {operation} an instance path redirected by an "
            f"intermediate symlink or junction: {instance_dir}"
        )
    if len(relative.parts) < 3:
        raise ValueError(
            f"refusing to {operation} a broad instance path; expected "
            "dataset/seed/instance below the instances root"
        )
    return instance_dir


def _remove_generated_instance_tree(
    instance_dir: str | Path,
    instances_root: str | Path,
    operation: str,
) -> Path:
    """Validate and remove an existing generated instance tree."""
    instance_dir = _validate_generated_instance_path(
        instance_dir,
        instances_root,
        operation,
    )
    if instance_dir.is_symlink() or (
        instance_dir.exists() and not instance_dir.is_dir()
    ):
        raise ValueError(
            f"refusing to {operation} a non-directory instance path: "
            f"{instance_dir}"
        )
    if instance_dir.exists():
        shutil.rmtree(instance_dir)
    return instance_dir


def reset_generated_instance_dir(
    instance_dir: str | Path,
    instances_root: str | Path,
) -> Path:
    """Recreate one safely contained generated instance directory."""
    instance_dir = _remove_generated_instance_tree(
        instance_dir,
        instances_root,
        "reset",
    )
    instance_dir.mkdir(parents=True, exist_ok=False)
    return instance_dir


def remove_generated_instance_dir(
    instance_dir: str | Path,
    instances_root: str | Path,
) -> Path:
    """Remove one generated instance without recreating an empty directory."""
    return _remove_generated_instance_tree(
        instance_dir,
        instances_root,
        "remove",
    )


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def partial_output_path(output_path: str | Path) -> Path:
    """Return a non-final checkpoint path that ``*.json`` scans ignore."""
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.name}.partial")


def _replaceable_target_mode(path: Path) -> int | None:
    """Return an existing file's mode, or reject non-replaceable targets."""
    try:
        target_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISREG(target_stat.st_mode):
        mode = stat.S_IMODE(target_stat.st_mode)
        if os.name == "nt" and not mode & stat.S_IWRITE:
            raise PermissionError(
                f"atomic JSON output target is read-only: {path}"
            )
        return mode
    if stat.S_ISLNK(target_stat.st_mode):
        # os.replace() replaces the link itself rather than following it.
        return None
    raise IsADirectoryError(
        f"atomic JSON output target is not a replaceable file: {path}"
    )


def validate_atomic_output_target(path: str | Path) -> None:
    """Fail before expensive work when a formal output cannot be replaced."""
    _replaceable_target_mode(Path(path))


def _filename_component_units(text: str, use_windows_units: bool) -> int:
    if use_windows_units:
        return windows_utf16_units(text)
    return len(os.fsencode(text))


def _truncate_to_filename_units(
    text: str,
    max_units: int,
    use_windows_units: bool,
) -> str:
    if use_windows_units:
        return truncate_to_utf16_units(text, max_units)
    if max_units < 0:
        raise ValueError("max_units must be non-negative")

    low = 0
    high = len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        if len(os.fsencode(text[:midpoint])) <= max_units:
            low = midpoint
        else:
            high = midpoint - 1
    return text[:low]


def _atomic_temp_candidate(
    path: Path,
    token: str,
    *,
    max_path_units: int | None = None,
) -> Path:
    """Return a same-directory atomic sidecar with a bounded path length.

    The target filename is useful in a temporary filename for diagnostics, but
    it is the only expendable part. Preserve the random token and ``.tmp``
    suffix, truncating the copied target-name prefix when a filesystem path or
    filename-component limit would otherwise be exceeded.
    """
    suffix = f".{token}.tmp"
    prefix = f".{path.name}"
    absolute_parent = os.path.abspath(path.parent)
    is_extended_windows_path = (
        os.name == "nt" and str(absolute_parent).startswith("\\\\?\\")
    )
    use_windows_units = os.name == "nt" or max_path_units is not None

    suffix_units = _filename_component_units(suffix, use_windows_units)
    max_component_units = (
        WINDOWS_MAX_COMPONENT_UNITS
        if use_windows_units
        else PORTABLE_MAX_COMPONENT_BYTES
    )
    max_prefix_units = max_component_units - suffix_units
    if (
        max_path_units is None
        and os.name == "nt"
        and not is_extended_windows_path
    ):
        max_path_units = WINDOWS_LEGACY_MAX_PATH_UNITS
    if max_path_units is not None:
        fixed_candidate = Path(os.path.abspath(path.parent / suffix))
        max_prefix_units = min(
            max_prefix_units,
            max_path_units - windows_utf16_units(fixed_candidate),
        )

    if max_prefix_units < 0:
        raise OSError(
            "atomic JSON parent path is too long to allocate a temporary "
            f"filename beside {path}"
        )
    prefix = _truncate_to_filename_units(
        prefix,
        max_prefix_units,
        use_windows_units,
    )
    return path.parent / f"{prefix}{suffix}"


def save_json_atomic(obj: Any, path: str | Path) -> None:
    """Atomically replace one JSON file via a temporary file beside it."""
    path = Path(path)
    existing_mode = _replaceable_target_mode(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = -1
    temp_path: Path | None = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        candidate = _atomic_temp_candidate(
            path,
            secrets.token_hex(8),
        )
        try:
            fd = os.open(candidate, flags, 0o666)
        except FileExistsError:
            continue
        temp_path = candidate
        break
    if temp_path is None:
        raise FileExistsError(
            f"could not allocate an atomic temporary file beside {path}"
        )

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1
            json.dump(
                obj,
                handle,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
            handle.write("\n")
        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)
        os.replace(temp_path, path)
    finally:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


class JsonResultLifecycle:
    """Manage one batch result's partial checkpoint and final publication.

    ``checkpoint()`` writes an explicitly marked ``*.partial`` payload while
    leaving any previous complete output untouched. ``publish()`` atomically
    replaces the complete output first, then removes the recoverable partial
    checkpoint.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        kind: str,
        records_key: str,
    ) -> None:
        self.output_path = Path(output_path)
        self.partial_path = partial_output_path(self.output_path)
        self.kind = kind
        self.records_key = records_key

    def checkpoint(self, records: list[dict[str, Any]]) -> None:
        save_json_atomic(
            {
                "status": "partial",
                "kind": self.kind,
                "completed": len(records),
                self.records_key: records,
            },
            self.partial_path,
        )

    def publish(self, records: list[dict[str, Any]]) -> None:
        save_json_atomic(records, self.output_path)
        try:
            self.partial_path.unlink(missing_ok=True)
        except OSError as exc:
            print(
                "WARNING: could not remove partial checkpoint "
                f"{self.partial_path}: {exc}"
            )


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
