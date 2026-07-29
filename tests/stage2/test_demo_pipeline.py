"""End-to-end contract for the explicit, deterministic Stage2 demo CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPOSITORY_ROOT / "scripts" / "prepare_stage2_demo_data.py"
OPTIMIZE = REPOSITORY_ROOT / "scripts" / "stage2_optimize_model.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )


def test_demo_requires_an_explicit_safe_output_root(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _run(str(PREPARE))

    unsafe = subprocess.run(
        [sys.executable, str(PREPARE), "--output-root", str(REPOSITORY_ROOT)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert unsafe.returncode != 0
    assert "repository root" in unsafe.stderr

    non_demo = tmp_path / "not-demo"
    non_demo.mkdir()
    (non_demo / "user-file.txt").write_text("preserve me", encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, str(PREPARE), "--output-root", str(non_demo)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "existing non-demo directory" in rejected.stderr
    assert (non_demo / "user-file.txt").read_text(encoding="utf-8") == "preserve me"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    escaped = subprocess.run(
        [sys.executable, str(PREPARE), "--output-root", str(symlink_parent / "demo")],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert escaped.returncode != 0
    assert "symlink" in escaped.stderr


def test_demo_pipeline_is_schema_valid_and_byte_stable_across_two_runs(tmp_path: Path) -> None:
    demo_root = tmp_path / "demo"
    out_json = tmp_path / "out" / "stage2.json"
    command = [
        str(OPTIMIZE),
        "--manifest",
        str(demo_root / "framework/partitions/pyramid_lidar_partition.yaml"),
        "--classification",
        str(demo_root / "results/model_classifier.json"),
        "--out-json",
        str(out_json),
    ]

    _run(str(PREPARE), "--output-root", str(demo_root))
    _run(*command)
    first_bytes = out_json.read_bytes()

    _run(str(PREPARE), "--output-root", str(demo_root))
    _run(*command)
    second_bytes = out_json.read_bytes()
    output = json.loads(second_bytes)

    assert output["schema"] == "stage2_output_v1"
    assert output["search_space_summary"]["n_software_candidates"] == 1
    assert output["optimization_status"]["allowed"] is True
    assert output["optimization_status"]["mode"] == "joint"
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(second_bytes).hexdigest()
