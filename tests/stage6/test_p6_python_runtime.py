from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from framework.stage6 import p6_python_runtime_v1 as runtime


def _copied_python(tmp_path: Path) -> Path:
    python = tmp_path / "adapter-env/bin/python3.10"
    python.parent.mkdir(parents=True)
    shutil.copyfile("/usr/bin/python3.10", python)
    python.chmod(0o700)
    return python


def _poison_pythonpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    poison = tmp_path / "poison"
    poison.mkdir()
    poison.joinpath("sitecustomize.py").write_text(
        "raise SystemExit('poisoned Python startup')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(poison))
    monkeypatch.delenv("PYTHONHOME", raising=False)


def _poison_pythonhome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "missing-python-home"))


@pytest.mark.parametrize("poison", (_poison_pythonpath, _poison_pythonhome))
def test_validate_adapter_python_ignores_ambient_python_startup_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison: Any,
) -> None:
    python = _copied_python(tmp_path)
    poison(tmp_path, monkeypatch)

    assert runtime.validate_adapter_python(python) == python


def test_validate_adapter_python_uses_isolated_direct_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _copied_python(tmp_path)
    observed: dict[str, Any] = {}

    def record_probe(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, "3.10\n", "")

    monkeypatch.setattr(runtime.subprocess, "run", record_probe)

    assert runtime.validate_adapter_python(python) == python
    assert observed["argv"][:4] == [str(python), "-I", "-S", "-c"]
    assert len(observed["argv"]) == 5
    assert observed["env"] == {}
    assert {
        key: observed[key]
        for key in ("shell", "check", "capture_output", "text", "timeout")
    } == {
        "shell": False,
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 10,
    }


def test_validate_adapter_python_still_rejects_python39(tmp_path: Path) -> None:
    python = tmp_path / "python3.9"
    python.write_text("#!/bin/sh\nprintf '3.9\\n'\n", encoding="utf-8")
    python.chmod(0o700)

    with pytest.raises(runtime.P6PythonRuntimeError):
        runtime.validate_adapter_python(python)
