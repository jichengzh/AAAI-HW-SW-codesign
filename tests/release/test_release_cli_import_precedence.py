from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

REAL_CHAIN_CLIS = (
    ("build_p6_history_registry.py", 0),
    ("run_p6_h800_search.py", 0),
    ("measure_p6_history_batch.py", 0),
    ("build_p6_stage1_manifest.py", 0),
)
OTHER_REPOSITORY_IMPORTING_CLIS = (
    ("derive_p6_history_recipe.py", 2),
    ("normalize_p6_history_root.py", 2),
    ("preflight_p6_materializer_training_bridge.py", 2),
    ("provision_p6_full_chain_local_config.py", 2),
    ("provision_p6_history_local_config.py", 0),
    ("render_p6_source_wrapper.py", 2),
    ("validate_environment_contract.py", 2),
    ("verify_p6_materializer_training_run.py", 2),
)


def _poisoned_pythonpath(tmp_path: Path) -> tuple[dict[str, str], Path]:
    ambient_root = tmp_path / "ambient"
    ambient_framework = ambient_root / "framework"
    ambient_framework.mkdir(parents=True)
    marker = tmp_path / "ambient-framework-imported"
    (ambient_framework / "__init__.py").write_text(
        """from pathlib import Path
import os

Path(os.environ["P6_AMBIENT_IMPORT_MARKER"]).write_text("imported", encoding="utf-8")
raise RuntimeError("ambient framework imported")
""",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "P6_AMBIENT_IMPORT_MARKER": str(marker),
        "PYTHONPATH": os.pathsep.join((str(ambient_root), str(REPOSITORY_ROOT))),
    }
    return env, marker


@pytest.mark.parametrize(("script_name", "expected_returncode"), REAL_CHAIN_CLIS)
def test_real_chain_cli_prioritizes_its_repository_over_ambient_framework(
    tmp_path: Path,
    script_name: str,
    expected_returncode: int,
) -> None:
    env, marker = _poisoned_pythonpath(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tools/release" / script_name), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_returncode, completed.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("script_name", "expected_returncode"), OTHER_REPOSITORY_IMPORTING_CLIS
)
def test_release_cli_prioritizes_its_repository_over_ambient_framework(
    tmp_path: Path,
    script_name: str,
    expected_returncode: int,
) -> None:
    env, marker = _poisoned_pythonpath(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "tools/release" / script_name), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_returncode, completed.stderr
    assert not marker.exists()
