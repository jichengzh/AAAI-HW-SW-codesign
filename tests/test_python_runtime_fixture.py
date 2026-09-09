from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from framework.stage6.p6_python_runtime_v1 import validate_adapter_python
from tests.python_runtime_fixture import write_current_python_launcher


def test_current_python_launcher_is_a_valid_isolated_runtime(tmp_path: Path) -> None:
    launcher = write_current_python_launcher(tmp_path / "runtime/bin/python")

    assert launcher.is_file()
    assert not launcher.is_symlink()
    assert launcher.lstat().st_nlink == 1
    assert os.access(launcher, os.X_OK)
    assert validate_adapter_python(launcher) == launcher

    completed = subprocess.run(
        [str(launcher), "-I", "-S", "-c", "import sys; print(sys.executable)"],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )
    assert Path(completed.stdout.strip()) == launcher
    assert launcher != Path(sys.executable).absolute()
