"""Regression coverage for direct pytest invocation in a clean checkout."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_direct_pytest_collects_release_tool_and_test_helper_imports() -> None:
    """The CI command imports both checkout-local namespace packages."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != "PYTHONPATH" and not name.startswith("COV_CORE_")
    }
    pytest_executable = shutil.which("pytest")

    assert pytest_executable is not None

    result = subprocess.run(
        [
            pytest_executable,
            "--collect-only",
            "-q",
            "tests/release/test_p6_history_execution_adapters.py",
            "tests/release/test_run_p6_h800_search.py",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
