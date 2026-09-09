"""Portable executable-Python fixtures for isolated runtime contract tests."""

from __future__ import annotations

from pathlib import Path
import sys


def write_current_python_launcher(path: Path) -> Path:
    """Write a regular launcher while keeping the active Python at its install root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import os",
                "import sys",
                f"os.execv({sys.executable!r}, [__file__, *sys.argv[1:]])",
                "",
            )
        ),
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path
