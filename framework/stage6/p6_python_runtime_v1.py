"""Validate the explicit Python runtime used by public P6 adapters."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess


MINIMUM_ADAPTER_PYTHON = (3, 10)
_VERSION_PROBE = (
    "import sys; "
    "print(f'{sys.version_info.major}.{sys.version_info.minor}')"
)


class P6PythonRuntimeError(ValueError):
    """Stable path-free adapter runtime failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


def validate_adapter_python(raw: object) -> Path:
    """Require one canonical single-link executable Python 3.10 or newer."""
    path = raw if isinstance(raw, Path) else Path(raw) if isinstance(raw, str) else None
    if (
        path is None
        or not path.is_absolute()
        or str(path) != str(raw)
        or _contains_symlink_component(path)
    ):
        raise P6PythonRuntimeError()
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as error:
        raise P6PythonRuntimeError() from error
    if (
        resolved != path
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or not os.access(path, os.X_OK)
    ):
        raise P6PythonRuntimeError()
    _require_supported_version(path)
    return path


def _require_supported_version(path: Path) -> None:
    try:
        completed = subprocess.run(
            [str(path), "-c", _VERSION_PROBE],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = _parse_version(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise P6PythonRuntimeError() from error
    if completed.returncode != 0 or version < MINIMUM_ADAPTER_PYTHON:
        raise P6PythonRuntimeError()


def _parse_version(raw: str) -> tuple[int, int]:
    parts = raw.rstrip("\n").split(".")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise ValueError
    return int(parts[0]), int(parts[1])


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )
