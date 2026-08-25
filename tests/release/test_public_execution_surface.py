"""Regression checks for public scripts that can be run from a fresh clone."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUNNER = REPOSITORY_ROOT / "scripts" / "phase2" / "stage1_s2_anchor_runner.py"
DEMO_PREPARE = REPOSITORY_ROOT / "scripts" / "prepare_stage2_demo_data.py"


def test_anchor_runner_requires_explicit_runtime_and_output_inputs_without_echoing_values(
    tmp_path: Path,
) -> None:
    """A missing H800 selection fails before importing or running TVM."""
    private_marker = "private-output-location-marker"
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)

    result = subprocess.run(
        [
            sys.executable,
            str(ANCHOR_RUNNER),
            "--out-json",
            str(tmp_path / private_marker / "report.json"),
            "--work-root",
            str(tmp_path / private_marker / "work"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        env=environment,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "CUDA_VISIBLE_DEVICES" in result.stderr
    assert private_marker not in result.stderr
    assert private_marker not in result.stdout
    assert "/home/" not in result.stderr


def test_anchor_runner_rejects_a_missing_cli_output_without_echoing_other_inputs(
    tmp_path: Path,
) -> None:
    private_marker = "private-output-location-marker"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(
        [
            sys.executable,
            str(ANCHOR_RUNNER),
            "--out-json",
            str(tmp_path / private_marker / "report.json"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        env=environment,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--work-root" in result.stderr
    assert private_marker not in result.stderr
    assert private_marker not in result.stdout
    assert "/home/" not in result.stderr


def test_public_runner_source_contains_no_personal_absolute_path() -> None:
    source = ANCHOR_RUNNER.read_text(encoding="utf-8")

    assert "/home/" not in source
    assert "/exdata/" not in source
    assert 'required runtime setting: CUDA_VISIBLE_DEVICES' in source
    assert 'parser.add_argument("--out-json", required=True)' in source
    assert 'parser.add_argument("--work-root", required=True)' in source


def test_tracked_tree_contains_no_personal_absolute_paths() -> None:
    """Ignored local evidence must never become a tracked privacy disclosure."""
    result = subprocess.run(
        ["git", "grep", "-Il", "-e", "/home/jichengzhi", "-e", "/exdata/"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    disclosed = {
        path
        for path in result.stdout.splitlines()
        if path != Path(__file__).relative_to(REPOSITORY_ROOT).as_posix()
    }

    assert result.returncode in {0, 1}
    assert disclosed == set()


def test_demo_preparation_docstring_promises_explicit_output_root_only() -> None:
    source = DEMO_PREPARE.read_text(encoding="utf-8")

    assert "only beneath an explicit --output-root" in source
    assert "writes a minimal local demo under results/" not in source


def test_gitignore_excludes_generated_experiment_outputs_without_hiding_public_inputs() -> None:
    generated_paths = (
        "results/generated.json",
        "models/model.bin",
        "checkpoints/model.ckpt",
        "framework/partitions/demo.yaml",
        "outside/model.engine",
        "outside/model.onnx",
        "outside/model.pt",
        "outside/model.safetensors",
        "outside/model.trt",
    )
    public_inputs = (
        "data/demo/candidate_pool.jsonl",
        "artifacts/verified/manifest.json",
        "framework/stage2/contracts.py",
    )

    for path in generated_paths:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", path],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"expected generated path to be ignored: {path}"

    for path in public_inputs:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", path],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1, f"public input/source must remain trackable: {path}"
