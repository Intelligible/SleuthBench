"""Helpers for values used as filesystem path components."""

from __future__ import annotations

import unicodedata
from pathlib import Path


_INVALID_PATH_CHARS = set('<>:"/\\|?*')
WINDOWS_LEGACY_MAX_PATH_UNITS = 259
WINDOWS_MAX_COMPONENT_UNITS = 255
PORTABLE_MAX_COMPONENT_BYTES = 255
GENERATED_ARTIFACT_NAME_RESERVE_UNITS = 64
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def windows_utf16_units(value: object) -> int:
    """Return the number of UTF-16 code units Win32 uses for a path."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in str(value))


def truncate_to_utf16_units(text: str, max_units: int) -> str:
    """Return the longest prefix of ``text`` within a UTF-16 unit budget."""
    if max_units < 0:
        raise ValueError("max_units must be non-negative")

    used = 0
    for index, char in enumerate(text):
        char_units = 2 if ord(char) > 0xFFFF else 1
        if used + char_units > max_units:
            return text[:index]
        used += char_units
    return text


def normalize_path_text(value: object) -> str:
    """Normalize user/path text to the canonical form used for comparisons."""
    return unicodedata.normalize("NFC", str(value))


def safe_path_component(value: object, field_name: str) -> str:
    """Return ``value`` as one portable path component or raise ValueError."""
    text = normalize_path_text(value)
    invalid = (
        not text
        or text in {".", ".."}
        or len(text) > 128
        or windows_utf16_units(text) > WINDOWS_MAX_COMPONENT_UNITS
        or (
            len(text.encode("utf-8", errors="surrogatepass"))
            > PORTABLE_MAX_COMPONENT_BYTES
        )
        or text != text.strip()
        or text.endswith(".")
        or any(
            ord(char) < 32
            or 0xD800 <= ord(char) <= 0xDFFF
            or char in _INVALID_PATH_CHARS
            for char in text
        )
        or text.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
    )
    if invalid:
        raise ValueError(
            f"{field_name} must be a non-empty, portable filesystem name "
            "without path separators or reserved characters"
        )
    return text


def portable_path_key(value: object) -> str:
    """Return the collision key used by common case/Unicode-folding filesystems."""
    return normalize_path_text(value).casefold()


def ensure_portable_child_namespace(
    parent: str | Path,
    requested: object,
    field_name: str,
) -> str:
    """Reject an existing child whose portable key aliases another spelling."""
    parent = Path(parent)
    requested_name = safe_path_component(requested, field_name)
    if not parent.exists():
        return requested_name
    if not parent.is_dir():
        raise NotADirectoryError(
            f"{field_name} namespace parent is not a directory: {parent}"
        )

    requested_key = portable_path_key(requested_name)
    for child in parent.iterdir():
        if portable_path_key(child.name) != requested_key:
            continue
        # macOS may return a decomposed spelling for the exact same directory.
        # Allow that only when both spellings resolve to the same filesystem
        # object; on normalization-sensitive filesystems, fail instead of
        # creating a second namespace that would collapse elsewhere.
        if normalize_path_text(child.name) == requested_name:
            requested_path = parent / requested_name
            try:
                if child.samefile(requested_path):
                    return requested_name
            except OSError:
                pass
        raise ValueError(
            f"{field_name} {requested_name!r} collides with existing "
            f"namespace {child.name!r} under {parent}"
        )
    return requested_name
