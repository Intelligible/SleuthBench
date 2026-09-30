"""Helpers shared by command-line entry points."""
from __future__ import annotations

import sys
from typing import TextIO


def _reconfigure_stream(stream: TextIO | None) -> None:
    """Make a standard text stream safe for arbitrary Unicode output."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError, TypeError, ValueError):
        # Redirected/test streams may expose no usable reconfiguration API.
        # They are already responsible for accepting the strings written to
        # them, so leaving them unchanged is safer than replacing the object.
        return


def configure_cli_streams() -> None:
    """Prevent locale-limited stdout/stderr from crashing a CLI run."""
    _reconfigure_stream(sys.stdout)
    _reconfigure_stream(sys.stderr)
