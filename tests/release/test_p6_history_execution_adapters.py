from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5.single_target_search_v2 import (
    SearchTask,
    build_task_candidate_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_CLI = REPOSITORY_ROOT / "tools/release/build_p6_history_registry.py"


def _plan() -> dict[str, Any]:
    candidates = [
        {
            "width": [17, 31, 63],
            "q_mode": q_mode,
            "source_point_ids": [
                f"stage1-{q_mode}-17",
                f"stage2-{q_mode}-31",
                f"stage3-{q_mode}-63",
            ],
        }
        for q_mode in ("fp16", "int8")
    ]
    return {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "source_schema": "stage2_search_space_v1",
        "target_model": "pyramid",
        "hardware_target": "h800",
        "execution_backend": "tvm_auto",
        "candidate_source_mode": "framework_stage2_search_space",
        "structure_count": 1,
        "candidate_count": 2,
        "candidates": candidates,
    }


def _binding(tmp_path: Path) -> dict[str, Any]:
    private_root = tmp_path / "synthetic-history"
    private_root.mkdir(exist_ok=True)
    evidence_sha = hashlib.sha256(b"synthetic-history-evidence").hexdigest()
    outputs = {
        q_mode: {
            f"{kind}_path_template": (
                f"materialized/{{group_id}}/{{q_mode}}/{kind}.json"
            )
            for kind in ("training", "checkpoint", "onnx", "calibration")
        }
        for q_mode in ("fp16", "int8")
    }
    return {
        "schema_version": "p6_history_binding_v1",
        "target": {
            "model": "pyramid",
            "hardware": "h800",
            "backend": "tvm_auto",
        },
        "private_root": str(private_root),
        "source_contract_template": {
            "schema_version": "stage5_source_contract_v1",
            "group_id": "pyramid|16x32x64",
            "model": "pyramid",
            "width": [16, 32, 64],
            "artifact_id": "synthetic-template",
            "source_status": "ready",
            "source_evidence_sha256": evidence_sha,
            "materialization_scope": "synthetic_fixture",
            "dynamic_materialization_recipe": {
                "schema_version": "p6_history_dynamic_materialization_recipe_v1",
                "stage_width_fields": [
                    "stage1_width",
                    "stage2_width",
                    "stage3_width",
                ],
                "group_id_template": (
                    "pyramid|{stage1_width}x{stage2_width}x{stage3_width}"
                ),
                "artifact_id_template": (
                    "pyramid-{stage1_width}-{stage2_width}-{stage3_width}"
                ),
                "output_path_templates_by_q_mode": outputs,
            },
        },
        "status": "validated",
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_registry_cli(
    binding_path: Path,
    plan_path: Path,
    registry_path: Path,
    local_output_root: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REGISTRY_CLI),
            "--binding",
            str(binding_path),
            "--pyramid-candidate-plan",
            str(plan_path),
            "--source-registry-json",
            str(registry_path),
            "--local-output-root",
            str(local_output_root),
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )


def _identity_map(rows: list[Mapping[str, Any]]) -> set[tuple[tuple[int, ...], str]]:
    return {(tuple(row["width"]), str(row["q_mode"])) for row in rows}


def test_registry_cli_writes_stage5_compatible_exact_dynamic_manifest(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    binding_path = _write_json(local_output_root / "binding.json", _binding(tmp_path))
    plan = _plan()
    plan_path = _write_json(local_output_root / "plan.json", plan)
    registry_path = local_output_root / "registry.json"

    result = _run_registry_cli(
        binding_path, plan_path, registry_path, local_output_root
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "source_registry_written\n"
    assert result.stderr == ""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    profile = build_capability_profile(
        capability_profile_id="h800-tvm_auto",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"tvm_auto").hexdigest(),
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )
    task = SearchTask(
        task_id="P6-H800-PYRAMID",
        target_model="pyramid",
        hardware_id="h800",
        capability_profile=profile,
    )
    manifest = build_task_candidate_manifest(
        registry, task=task, measured_row_ids=set()
    )
    assert _identity_map(manifest["rows"]) == _identity_map(plan["candidates"])
    assert manifest["eligible_row_count"] == plan["candidate_count"]


def test_registry_cli_rejects_extra_command_surface_with_category_only_stderr(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    binding_path = _write_json(local_output_root / "binding.json", _binding(tmp_path))
    plan_path = _write_json(local_output_root / "plan.json", _plan())
    registry_path = local_output_root / "registry.json"

    result = _run_registry_cli(
        binding_path,
        plan_path,
        registry_path,
        local_output_root,
        "--command",
        "private-shell-content",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not registry_path.exists()


def test_registry_cli_private_binding_failure_is_category_only_and_writes_nothing(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    private_marker = "PRIVATE-BINDING-CONTENT-MUST-NOT-LEAK"
    binding_path = local_output_root / f"{private_marker}.json"
    binding_path.write_text(f"{{not-json:{private_marker}}}", encoding="utf-8")
    plan_path = _write_json(local_output_root / "plan.json", _plan())
    registry_path = local_output_root / "registry.json"

    result = _run_registry_cli(
        binding_path, plan_path, registry_path, local_output_root
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "source_registry_invalid\n"
    assert private_marker not in result.stderr
    assert not registry_path.exists()


def test_registry_cli_rejects_output_path_outside_local_root_without_leak(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    binding_path = _write_json(local_output_root / "binding.json", _binding(tmp_path))
    plan_path = _write_json(local_output_root / "plan.json", _plan())
    registry_path = tmp_path / "outside-private-output.json"

    result = _run_registry_cli(
        binding_path, plan_path, registry_path, local_output_root
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "source_registry_invalid\n"
    assert str(tmp_path) not in result.stderr
    assert not registry_path.exists()
