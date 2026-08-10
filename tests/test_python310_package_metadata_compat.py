"""Python 3.10 compatibility coverage for package metadata tests."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_TEST = PROJECT_ROOT / "tests" / "test_package_metadata.py"


def _tomli_is_available() -> bool:
    try:
        importlib.import_module("tomli")
    except ModuleNotFoundError as exc:
        if exc.name == "tomli":
            return False
        raise
    return True


def _run_metadata_tests_with_tomllib_shadow(
    tmp_path: Path, missing_module: str
) -> subprocess.CompletedProcess[str]:
    shadow_directory = tmp_path / "stdlib-shadow"
    shadow_directory.mkdir()
    missing_message = f"No module named '{missing_module}'"
    (shadow_directory / "tomllib.py").write_text(
        f"raise ModuleNotFoundError({missing_message!r}, name={missing_module!r})\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(shadow_directory), existing_python_path) if path
    )
    runner = "\n".join(
        (
            "import runpy",
            f"module = runpy.run_path({str(METADATA_TEST)!r})",
            "module['test_dev_extra_contains_all_dependencies_needed_by_stage1_tests']()",
            "module['test_repro_extra_remains_independent_of_optional_scan_runtime']()",
        )
    )

    return subprocess.run(
        [sys.executable, "-c", runner],
        cwd=PROJECT_ROOT,
        capture_output=True,
        env=environment,
        text=True,
        check=False,
    )


def test_package_metadata_tests_fall_back_when_tomllib_is_unavailable(tmp_path: Path) -> None:
    """Catches removing the Python 3.10 ``tomli`` fallback from the test module."""
    if not _tomli_is_available():
        pytest.skip("tomli is unavailable; the Python 3.10 fallback probe cannot run")

    result = _run_metadata_tests_with_tomllib_shadow(tmp_path, "tomllib")

    assert result.returncode == 0, result.stdout + result.stderr


def test_fallback_probe_skips_when_tomli_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches running the Python 3.10 fallback probe without its ``tomli`` dependency."""
    monkeypatch.setattr(sys.modules[__name__], "_tomli_is_available", lambda: False, raising=False)

    with pytest.raises(pytest.skip.Exception, match="tomli"):
        test_package_metadata_tests_fall_back_when_tomllib_is_unavailable(tmp_path)


def test_package_metadata_tests_preserve_tomllib_dependency_import_errors(tmp_path: Path) -> None:
    """Catches swallowing a broken ``tomllib`` dependency as a Python 3.10 fallback."""
    result = _run_metadata_tests_with_tomllib_shadow(tmp_path, "missing_dependency")

    assert result.returncode != 0
    assert "No module named 'missing_dependency'" in result.stderr
