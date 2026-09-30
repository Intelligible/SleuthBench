"""Missing-like category label lexicon.

``is_missing_like`` recognises a broad fixed set of normalised tokens, while
``DEFAULT_MISSING_LABELS`` contains only display forms safe to inject. ``"N/A"``
is recognised but excluded from the injectable set because pandas' default CSV
parser would coerce it to NaN.

The semantic injector and default categorical-feature filter use this lexicon;
the target-dependent bucket injector opts out before anonymising all labels.
"""
from __future__ import annotations

import re

# Strings pandas.read_csv coerces to NaN by default.
_PANDAS_DEFAULT_NA_STRINGS: frozenset[str] = frozenset({
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null",
})

# Injectable display forms; the first entry is the default.
DEFAULT_MISSING_LABELS: tuple[str, ...] = (
    "Unknown",
    "Not reported",
    "Not specified",
    "Missing",
    "Undisclosed",
)

# Canonical (normalised) tokens recognised as missing-like.
_MISSING_TOKENS: frozenset[str] = frozenset({
    "unknown",
    "notreported",
    "notrecorded",
    "notspecified",
    "unspecified",
    "notavailable",
    "unavailable",
    "notapplicable",
    "nodata",
    "na",
    "nan",
    "missing",
    "none",
    "null",
    "undisclosed",
    "refused",
})


def _normalize(label: object) -> str:
    """Case-fold and drop non-alphanumerics: "N/A" -> "na", "Not reported" -> "notreported"."""
    return re.sub(r"[^a-z0-9]", "", str(label).strip().lower())


def is_missing_like(label: object) -> bool:
    """True when ``label`` reads as a missing / unknown / not-reported marker.

    An empty / pure-punctuation label ("", "?", "-") normalises to "" and is
    also treated as missing-like — as is a NaN cell, since ``str(nan)`` -> "nan".
    """
    norm = _normalize(label)
    if norm == "":
        return True
    return norm in _MISSING_TOKENS


# Enforce the round-trip invariant at import time: no injectable label may be a
# string pandas would read back as NaN, and every injectable label must be
# recognised by is_missing_like.
assert not (set(DEFAULT_MISSING_LABELS) & _PANDAS_DEFAULT_NA_STRINGS), (
    "DEFAULT_MISSING_LABELS contains a pandas NA string; it would not survive the CSV round-trip"
)
assert all(is_missing_like(_l) for _l in DEFAULT_MISSING_LABELS), (
    "every DEFAULT_MISSING_LABELS entry must be recognised by is_missing_like"
)
