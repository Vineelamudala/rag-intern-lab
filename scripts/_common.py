"""Shared helpers for the command-line scripts (PROVIDED)."""

from __future__ import annotations

import sys

EXIT_ERROR = 1
EXIT_TODO = 2


def configure_console() -> None:
    """Never crash on characters the Windows console cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def report_todo(exc: NotImplementedError) -> int:
    """Explain which milestone blocks this command and return the exit code."""
    print(f"NOT IMPLEMENTED YET: {exc}", file=sys.stderr)
    print(
        "This command needs a milestone you have not finished. Implement it, run its "
        "tests (python -m pytest -m <milestone>), then try again.",
        file=sys.stderr,
    )
    return EXIT_TODO


def report_error(exc: Exception) -> int:
    print(f"ERROR: {exc}", file=sys.stderr)
    return EXIT_ERROR
