from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest

from framework.stage1.structural_axis_digest import canonical_digest
from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5.single_target_search_v2 import (
    SearchTask,
    build_task_candidate_manifest,
)
from framework.stage6.hardware_execution_profile_v1 import (
    load_hardware_execution_profile,
)
from framework.stage6.pyramid_search_space_adapter_v1 import (
    build_pyramid_candidate_plan,
)
from tools.release import measure_p6_history_batch as measurement_cli
from tests.release.p6_post_source_adapter_chain_fixture import (
    assert_adapter_leaf_chain,
    build_adapter_measurement_request,
    install_adapter_chain,
    write_source_materializer,
)
from tests.stage6.pyramid_formal_space_support import (
    scanner_owned_pyramid_stage2_space,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_CLI = REPOSITORY_ROOT / "tools/release/build_p6_history_registry.py"
MEASUREMENT_CLI = REPOSITORY_ROOT / "tools/release/measure_p6_history_batch.py"
SYNTHETIC_GPU_INDICES = (101, 103, 107)
RTX_GPU_INDICES = (*SYNTHETIC_GPU_INDICES, 109)


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


def _scanner_owned_rtx_plan() -> dict[str, Any]:
    search_space = scanner_owned_pyramid_stage2_space()
    hardware_target = dict(search_space["hardware_target"])
    hardware_target["name"] = "NVIDIA RTX 4090"
    search_space["hardware_target"] = hardware_target
    provenance = dict(search_space["formal_q_mode_provenance"])
    provenance["hardware_target"] = dict(hardware_target)
    unsigned = {
        key: value for key, value in provenance.items() if key != "digest"
    }
    provenance["digest"] = canonical_digest(unsigned)
    search_space["formal_q_mode_provenance"] = provenance
    search_space["hardware_candidates"][0]["hardware"] = "NVIDIA RTX 4090"
    return build_pyramid_candidate_plan(
        search_space,
        profile=load_hardware_execution_profile("rtx4090"),
    )


def _write_executable(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!{sys.executable}
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

stage_log = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"]) / "executed-stages.log"
with stage_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(dict(
        stage=Path(__file__).name,
        argv=sys.argv[1:],
        cwd=str(Path.cwd()),
        round_output_root=os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"],
        task_state=os.environ["P6_HISTORY_TASK_STATE"],
    ), sort_keys=True) + "\\n")
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return str(path)


def _write_finalizer(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!{sys.executable}
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

request_path, state_path, result_path, receipt_path, barrier_path, round_root = map(Path, sys.argv[1:])
stage_log = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"]) / "executed-stages.log"
with stage_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(dict(
        stage=Path(__file__).name,
        argv=sys.argv[1:],
        cwd=str(Path.cwd()),
        round_output_root=os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"],
        task_state=os.environ["P6_HISTORY_TASK_STATE"],
    ), sort_keys=True) + "\\n")
request = json.loads(request_path.read_text(encoding="utf-8"))
row_hashes = request["row_sha256"]
evidence = {{row["row_id"]: row["source_evidence_sha256"] for row in request["rows"]}}
state_rows = [
    {{
        "row_id": row["row_id"],
        "row_sha256": row_hashes[row["row_id"]],
        "source_evidence_sha256": row["source_evidence_sha256"],
        "terminal_status": "measured_success_gold",
    }}
    for row in request["rows"]
]
result_rows = [
    {{
        **row,
        "latency_ms": 2.0,
        "energy_j": 0.5,
        "ap30": 0.91,
        "ap50": 0.82,
        "ap70": 0.73,
    }}
    for row in state_rows
]
validation = {{
    "measurement_request_sha256": request["measurement_request_sha256"],
    "row_sha256": row_hashes,
    "source_evidence_sha256": evidence,
}}
outputs = (
    (state_path, {{"stage": "finalization", "rows": state_rows}}),
    (result_path, {{
        "measurement_request_sha256": request["measurement_request_sha256"],
        "rows": result_rows,
    }}),
    (receipt_path, validation),
    (barrier_path, validation),
)
for output_path, payload in outputs:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return str(path)


def _execution_binding_fields(
    private_root: Path,
    post_source_wrappers: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    chain_root = private_root / "documented-stage5-chain"
    component_names = {
        "controller": "stage5_task_round_controller_v3.sh",
        "source_materializer": "stage5_materialize_round_sources_v1.sh",
        "performance_plan": "stage5_build_performance_plan_v2.py",
        "finalizer": "stage5_finalize_feedback_v2.py",
    }
    components = {
        role: str(post_source_wrappers["finalization"])
        if role == "finalizer" and post_source_wrappers is not None
        else str(post_source_wrappers["performance"])
        if role == "performance_plan" and post_source_wrappers is not None
        else write_source_materializer(chain_root / name)
        if role == "source_materializer"
        else _write_finalizer(chain_root / name)
        if role == "finalizer"
        else _write_executable(chain_root / name)
        for role, name in component_names.items()
    }
    private_bin = private_root / "private-runner" / "bin"
    quantize = (
        str(post_source_wrappers["quantization"])
        if post_source_wrappers is not None
        else _write_executable(private_bin / "quantize-private")
    )
    measure_ap = (
        str(post_source_wrappers["ap"])
        if post_source_wrappers is not None
        else _write_executable(private_bin / "measure-ap-private")
    )
    activate = _write_executable(private_bin / "activate-private")
    terminal_statuses = [
        "measured_success_gold",
        "feasibility_failure",
        "numerical_feasibility_failure",
    ]
    row_fields = {
        "rows_key": "rows",
        "row_id_key": "row_id",
        "row_hash_key": "row_sha256",
        "source_evidence_key": "source_evidence_sha256",
        "status_key": "terminal_status",
    }
    validation_location = {
        "format": "json",
        "request_sha256_key": "measurement_request_sha256",
        "row_hashes_key": "row_sha256",
        "source_evidence_key": "source_evidence_sha256",
    }
    interface = {
        "schema_version": "p6_history_runner_interface_v1",
        "controller": {"argv": [components["controller"]]},
        "execution_chain": [
            {
                "stage": "source_materialization",
                "argv": [
                    components["source_materializer"],
                    "{measurement_request}",
                    "{round_output_root}",
                ],
                "required_placeholders": [
                    "{measurement_request}",
                    "{round_output_root}",
                ],
            },
            {
                "stage": "quantization",
                "argv": [quantize, "{task_state}", "{round_output_root}"],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "performance",
                "argv": [
                    components["performance_plan"],
                    "{task_state}",
                    "{round_output_root}",
                ],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "ap",
                "argv": [measure_ap, "{task_state}", "{round_output_root}"],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "finalization",
                "argv": [
                    components["finalizer"],
                    "{measurement_request}",
                    "{task_state}",
                    "{actual_feedback}",
                    "{actual_receipt}",
                    "{finalization_barrier}",
                    "{round_output_root}",
                ],
                "required_placeholders": [
                    "{measurement_request}",
                    "{task_state}",
                    "{actual_feedback}",
                    "{actual_receipt}",
                    "{finalization_barrier}",
                    "{round_output_root}",
                ],
            },
        ],
        "environment": {
            "values": {
                "CUDA_VISIBLE_DEVICES": {
                    "kind": "literal",
                    "value": ",".join(str(index) for index in SYNTHETIC_GPU_INDICES),
                },
                "P6_HISTORY_RUN_MODE": {"kind": "literal", "value": "bound"},
                "P6_HISTORY_PRIVATE_ROOT": {
                    "kind": "private_path",
                    "value": str(private_root),
                },
                "P6_HISTORY_TASK_STATE": {
                    "kind": "placeholder",
                    "value": "{task_state}",
                },
                "P6_HISTORY_ROUND_OUTPUT_ROOT": {
                    "kind": "placeholder",
                    "value": "{round_output_root}",
                },
            },
            "activation_argv": [activate, "private-bound"],
        },
        "output_layout": {
            "round_root_template": "private-runs/{round_id}",
            "task_state": {
                "path_template": "private-runs/{round_id}/state/task-state.json",
                "format": "json",
                **row_fields,
                "stage_key": "stage",
                "row_count": 4,
                "allowed_terminal_statuses": terminal_statuses,
                "stage_order": [
                    "source_materialization",
                    "quantization",
                    "performance",
                    "ap",
                    "finalization",
                ],
            },
        },
        "actual_feedback": {
            "result": {
                "path_template": "private-runs/{round_id}/actual-feedback.json",
                "format": "json",
                **row_fields,
                "row_count": 4,
                "allowed_terminal_statuses": terminal_statuses,
                "metric_keys": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
            },
            "receipt": {
                "path_template": "private-runs/{round_id}/receipt.json",
                **validation_location,
            },
            "finalization_barrier": {
                "path_template": "private-runs/{round_id}/barrier.json",
                **validation_location,
            },
        },
    }
    return {"component_paths": components, "execution_interface": interface}


def _synthetic_history_binding(
    tmp_path: Path, *, adapter_chain: bool = False
) -> dict[str, Any]:
    private_root = tmp_path / "synthetic-history"
    private_root.mkdir(exist_ok=True)
    wrappers = install_adapter_chain(private_root) if adapter_chain else None
    evidence_sha = hashlib.sha256(b"synthetic-history-evidence").hexdigest()
    outputs = {
        q_mode: {
            f"{kind}_path_template": (f"materialized/{{group_id}}/{{q_mode}}/{kind}.json")
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
        **_execution_binding_fields(private_root, wrappers),
        "gpu_policy": {
            "indices": list(SYNTHETIC_GPU_INDICES),
            "uuid_by_index": {
                str(index): f"GPU-fixture-{index}" for index in SYNTHETIC_GPU_INDICES
            },
            "model": "h800",
            "maximum_occupancy": 0.05,
        },
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
                "group_id_template": ("pyramid|{stage1_width}x{stage2_width}x{stage3_width}"),
                "artifact_id_template": ("pyramid-{stage1_width}-{stage2_width}-{stage3_width}"),
                "output_path_templates_by_q_mode": outputs,
            },
        },
        "status": "validated",
    }


def _synthetic_rtx_history_binding(tmp_path: Path) -> dict[str, Any]:
    binding = _synthetic_history_binding(tmp_path)
    binding["target"]["hardware"] = "rtx4090"
    binding["execution_interface"]["environment"]["values"][
        "CUDA_VISIBLE_DEVICES"
    ]["value"] = ",".join(str(index) for index in RTX_GPU_INDICES)
    binding["gpu_policy"] = {
        "indices": list(RTX_GPU_INDICES),
        "uuid_by_index": {
            str(index): f"GPU-fixture-{index}" for index in RTX_GPU_INDICES
        },
        "hardware_profile": "rtx4090",
    }
    return binding


def test_registry_cli_rejects_tampered_execution_interface_without_leak(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    binding = _synthetic_history_binding(tmp_path)
    private_marker = "PRIVATE-INTERFACE-MUST-NOT-LEAK"
    binding["execution_interface"]["execution_chain"][1]["argv"][0] = str(tmp_path / private_marker)
    binding_path = _write_json(local_output_root / "binding.json", binding)
    plan_path = _write_json(local_output_root / "plan.json", _plan())
    registry_path = local_output_root / "registry.json"

    result = _run_registry_cli(binding_path, plan_path, registry_path, local_output_root)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "source_registry_invalid\n"
    assert private_marker not in result.stderr
    assert not registry_path.exists()


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
        env={**_subprocess_base_env(), "PYTHONPATH": str(REPOSITORY_ROOT)},
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
    binding_path = _write_json(
        local_output_root / "binding.json", _synthetic_history_binding(tmp_path)
    )
    plan = _plan()
    plan_path = _write_json(local_output_root / "plan.json", plan)
    registry_path = local_output_root / "registry.json"

    result = _run_registry_cli(binding_path, plan_path, registry_path, local_output_root)

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
    manifest = build_task_candidate_manifest(registry, task=task, measured_row_ids=set())
    assert _identity_map(manifest["rows"]) == _identity_map(plan["candidates"])
    assert manifest["eligible_row_count"] == plan["candidate_count"]


def test_registry_cli_materializes_scanner_owned_rtx_plan_through_real_chain(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    binding_path = _write_json(
        local_output_root / "binding.json", _synthetic_rtx_history_binding(tmp_path)
    )
    plan = _scanner_owned_rtx_plan()
    plan_path = _write_json(local_output_root / "plan.json", plan)
    registry_path = local_output_root / "registry.json"

    result = _run_registry_cli(
        binding_path, plan_path, registry_path, local_output_root
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "source_registry_written\n"
    assert result.stderr == ""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert _identity_map(
        [
            {
                "width": group["width"],
                "q_mode": q_mode,
            }
            for group in registry["groups"]
            for q_mode in group["available_q_modes"]
        ]
    ) == _identity_map(plan["candidates"])


@pytest.mark.parametrize(
    ("plan_factory", "binding_factory"),
    [
        (_scanner_owned_rtx_plan, _synthetic_history_binding),
        (_plan, _synthetic_rtx_history_binding),
    ],
    ids=["rtx-plan-h800-binding", "h800-plan-rtx-binding"],
)
def test_registry_cli_rejects_plan_and_binding_profile_mismatch_without_leak(
    tmp_path: Path,
    plan_factory: Any,
    binding_factory: Any,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    binding_path = _write_json(
        local_output_root / "binding.json", binding_factory(tmp_path)
    )
    plan_path = _write_json(local_output_root / "plan.json", plan_factory())
    registry_path = local_output_root / "registry.json"

    result = _run_registry_cli(
        binding_path, plan_path, registry_path, local_output_root
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "source_registry_invalid\n"
    assert "101" not in result.stderr
    assert "GPU-fixture" not in result.stderr
    assert "rtx4090" not in result.stderr
    assert not registry_path.exists()


@pytest.mark.parametrize(
    "hardware_target",
    ["unknown", "h800"],
    ids=["unknown", "mixed-h800-outer-rtx-provenance"],
)
def test_registry_cli_rejects_unknown_or_mixed_rtx_plan_target(
    tmp_path: Path, hardware_target: str
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    binding_path = _write_json(
        local_output_root / "binding.json", _synthetic_history_binding(tmp_path)
    )
    plan = _scanner_owned_rtx_plan()
    plan["hardware_target"] = hardware_target
    plan_path = _write_json(local_output_root / "plan.json", plan)
    registry_path = local_output_root / "registry.json"

    result = _run_registry_cli(
        binding_path, plan_path, registry_path, local_output_root
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "source_registry_invalid\n"
    assert not registry_path.exists()


def test_registry_cli_rejects_extra_command_surface_with_category_only_stderr(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    binding_path = _write_json(
        local_output_root / "binding.json", _synthetic_history_binding(tmp_path)
    )
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

    result = _run_registry_cli(binding_path, plan_path, registry_path, local_output_root)

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
    binding_path = _write_json(
        local_output_root / "binding.json", _synthetic_history_binding(tmp_path)
    )
    plan_path = _write_json(local_output_root / "plan.json", _plan())
    registry_path = tmp_path / "outside-private-output.json"

    result = _run_registry_cli(binding_path, plan_path, registry_path, local_output_root)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "source_registry_invalid\n"
    assert str(tmp_path) not in result.stderr
    assert not registry_path.exists()


def test_registry_cli_rejects_unignored_repository_destination_without_leak(
    tmp_path: Path,
) -> None:
    """Catches the CLI exposing a source contract through a trackable repository path."""
    registry_path = REPOSITORY_ROOT / "p6-private-registry-must-not-exist.json"
    assert not registry_path.exists()
    binding_path = _write_json(
        tmp_path / "binding.json", _synthetic_history_binding(tmp_path)
    )
    plan_path = _write_json(tmp_path / "plan.json", _plan())

    try:
        result = _run_registry_cli(
            binding_path, plan_path, registry_path, REPOSITORY_ROOT
        )
    finally:
        registry_path.unlink(missing_ok=True)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "source_registry_invalid\n"
    assert str(tmp_path) not in result.stderr
    assert not registry_path.exists()


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _measurement_request() -> dict[str, Any]:
    task_sha = hashlib.sha256(b"measurement-task").hexdigest()
    rows: list[dict[str, Any]] = []
    for index in range(4):
        width = [16 + index, 32 + index, 64 + index]
        group_id = f"pyramid|{width[0]}x{width[1]}x{width[2]}"
        evidence = hashlib.sha256(f"measurement-evidence-{index}".encode()).hexdigest()
        contract = {
            "schema_version": "stage5_source_contract_v1",
            "group_id": group_id,
            "model": "pyramid",
            "width": width,
            "artifact_id": f"measurement-artifact-{index}",
            "source_status": "ready",
            "source_evidence_sha256": evidence,
            "materialization_scope": "synthetic_fixture",
        }
        q_mode = "fp16" if index % 2 == 0 else "int8"
        row_id = f"{group_id}|q={q_mode}|profile=h800-tvm-auto"
        rows.append(
            {
                "schema_version": "stage5_candidate_row_v2",
                "task_id": "P6-H800-PYRAMID",
                "task_sha256": task_sha,
                "row_id": row_id,
                "manifest_job_id": row_id,
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "width_schema": ["w0", "w1", "w2"],
                "structure_widths": {
                    "w0": width[0],
                    "w1": width[1],
                    "w2": width[2],
                },
                "genome": [*width, q_mode],
                "strategy_id": f"q={q_mode}",
                "q_mode": q_mode,
                "hardware_id": "h800",
                "capability_profile_id": "h800-tvm-auto",
                "capability_digest": hashlib.sha256(b"profile").hexdigest(),
                "dispatch_key": "tvm_auto",
                "source_status": "ready",
                "materialization_kind": "local_pyramid_tvm",
                "source_evidence_kind": "local_synthetic",
                "source_contract": contract,
                "source_contract_sha256": _canonical_sha(contract),
                "source_evidence_sha256": evidence,
                "graph_features": {"group_id": group_id, "model": "pyramid", "width": width},
            }
        )
    body = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "P6-H800-PYRAMID",
        "task_sha256": task_sha,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _canonical_sha(row) for row in rows},
        "rows": rows,
    }
    return {**body, "measurement_request_sha256": _canonical_sha(body)}


def _write_synthetic_nvidia_smi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = " ".join(
        f"'{index}, GPU-fixture-{index}, NVIDIA H800 80GB HBM3, 0, 100'"
        for index in SYNTHETIC_GPU_INDICES
    )
    path.write_text(
        f"#!/bin/sh\nprintf '%s\\n' {records}\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def test_gpu_query_argv_preserves_supplied_private_policy_order() -> None:
    """Catches a measurement query that sorts the authoritative GPU policy."""
    assert measurement_cli._gpu_query_argv((23, 19, 17)) == (
        "nvidia-smi",
        "--id=23,19,17",
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )


def test_gpu_query_argv_accepts_two_gpu_policy_in_supplied_order() -> None:
    policy_indices = SYNTHETIC_GPU_INDICES[:2]

    assert measurement_cli._gpu_query_argv(policy_indices) == (
        "nvidia-smi",
        "--id=" + ",".join(str(index) for index in policy_indices),
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )


def test_gpu_probe_returns_records_in_supplied_private_policy_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a measurement probe that returns nvidia-smi discovery order."""
    observed_argv: tuple[str, ...] | None = None
    observed_shell: object = None

    def fake_run(argv: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal observed_argv, observed_shell
        observed_argv = argv
        observed_shell = kwargs.get("shell")
        return subprocess.CompletedProcess(
            argv,
            0,
            "17, GPU-synthetic-17, NVIDIA H800 80GB HBM3, 0, 100\n"
            "19, GPU-synthetic-19, NVIDIA H800 80GB HBM3, 0, 100\n"
            "23, GPU-synthetic-23, NVIDIA H800 80GB HBM3, 0, 100\n",
            "",
        )

    monkeypatch.setattr(measurement_cli.subprocess, "run", fake_run)

    records = measurement_cli.NvidiaSmiGpuProbe().snapshot((23, 19, 17))

    assert observed_argv == (
        "nvidia-smi",
        "--id=23,19,17",
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )
    assert observed_shell is False
    assert [record.index for record in records] == [23, 19, 17]


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param([23, 19, 17], id="non-tuple"),
        pytest.param((), id="empty"),
        pytest.param((23, 19, 23), id="duplicate"),
        pytest.param((True, 19, 17), id="bool"),
        pytest.param((23, "19", 17), id="non-int"),
        pytest.param((23, 19, -1), id="negative"),
    ],
)
def test_gpu_query_argv_rejects_malformed_policy(
    indices: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches malformed GPU policies reaching the production subprocess."""
    monkeypatch.setattr(
        measurement_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match="canonical GPU indices required"):
        measurement_cli.NvidiaSmiGpuProbe().snapshot(indices)


def _run_measurement_cli(
    binding_path: Path,
    request_path: Path,
    feedback_path: Path,
    round_output_root: Path,
    fake_bin: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(MEASUREMENT_CLI),
            "--binding",
            str(binding_path),
            "--measurement-request",
            str(request_path),
            "--feedback-json",
            str(feedback_path),
            "--round-output-root",
            str(round_output_root),
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        env={
            **_subprocess_base_env(),
            "PYTHONPATH": str(REPOSITORY_ROOT),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        },
        text=True,
        capture_output=True,
        check=False,
    )


def _subprocess_base_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE_") and key != "COVERAGE_PROCESS_START"
    }


def test_measurement_cli_executes_synthetic_chain_and_atomically_writes_feedback(
    tmp_path: Path,
) -> None:
    """Catches a CLI that validates but never persists complete four-row feedback."""
    binding = _synthetic_history_binding(tmp_path, adapter_chain=True)
    private_root = Path(binding["private_root"])
    round_output_root = private_root / "controller-round"
    round_output_root.mkdir()
    binding_path = _write_json(round_output_root / "binding.json", binding)
    request = build_adapter_measurement_request(private_root)
    request_path = _write_json(round_output_root / "request.json", request)
    feedback_path = round_output_root / "feedback.json"
    fake_bin = tmp_path / "fake-bin"
    _write_synthetic_nvidia_smi(fake_bin / "nvidia-smi")

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        round_output_root,
        fake_bin,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "measurement_feedback_written\n"
    assert result.stderr == ""
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert feedback["measurement_request_sha256"] == request["measurement_request_sha256"]
    assert len(feedback["rows"]) == 4
    assert_adapter_leaf_chain(private_root / "private-runs/0", request, binding)


def test_measurement_cli_keeps_history_artifacts_separate_from_external_feedback_root(
    tmp_path: Path,
) -> None:
    """Catches coupling controller feedback storage to the historical artifact root."""
    binding = _synthetic_history_binding(tmp_path, adapter_chain=True)
    history_root = Path(binding["private_root"])
    controller_round_root = tmp_path / "external-controller-round"
    controller_round_root.mkdir()
    binding_path = _write_json(controller_round_root / "binding.json", binding)
    request = build_adapter_measurement_request(controller_round_root.parent)
    request_path = _write_json(controller_round_root / "request.json", request)
    feedback_path = controller_round_root / "feedback.json"
    fake_bin = tmp_path / "fake-bin"
    _write_synthetic_nvidia_smi(fake_bin / "nvidia-smi")

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        controller_round_root,
        fake_bin,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "measurement_feedback_written\n"
    assert result.stderr == ""
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert feedback["measurement_request_sha256"] == request["measurement_request_sha256"]
    assert len(feedback["rows"]) == 4
    history_round = history_root / "private-runs" / "0"
    assert (history_round / "measurement-request.json").is_file()
    assert (history_round / "state" / "task-state.json").is_file()
    assert (history_round / "actual-feedback.json").is_file()
    assert (history_round / "receipt.json").is_file()
    assert (history_round / "barrier.json").is_file()
    stage_records = [
        json.loads(line)
        for line in (history_round / "executed-stages.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected_groups = sorted({row["group_id"] for row in request["rows"]})
    assert [record["stage"] for record in stage_records] == [
        "activate-private",
        *(["stage5_materialize_round_sources_v1.sh"] * len(expected_groups)),
    ]
    source_records = [
        record
        for record in stage_records
        if record["stage"] == "stage5_materialize_round_sources_v1.sh"
    ]
    policy_indices = binding["gpu_policy"]["indices"]
    assert [record["argv"] for record in source_records] == [
        [
            "--request",
            str(history_round / "measurement-request.json"),
            "--model",
            "pyramid",
            "--group-id",
            group_id,
            "--gpu",
            str(policy_indices[index % len(policy_indices)]),
        ]
        for index, group_id in enumerate(expected_groups)
    ]
    assert all(Path(record["cwd"]) == history_round for record in stage_records)
    assert all(record["round_output_root"] == str(history_round) for record in stage_records)
    assert all(
        record["task_state"] == str(history_round / "state" / "task-state.json")
        for record in stage_records
    )
    assert all(
        not Path(token).is_absolute() or Path(token).is_relative_to(history_round)
        for record in stage_records
        for token in record["argv"]
    )
    assert_adapter_leaf_chain(history_round, request, binding)
    assert not (controller_round_root / "private-runs").exists()
    assert set(path.name for path in controller_round_root.iterdir()) == {
        "binding.json",
        "feedback.json",
        "request.json",
    }


def test_measurement_cli_unknown_option_is_category_only_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """Catches accidental command/path override surface and argparse leakage."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    binding_path = _write_json(private_root / "binding.json", {"private": "marker"})
    request_path = _write_json(private_root / "request.json", _measurement_request())
    feedback_path = private_root / "feedback.json"
    fake_bin = tmp_path / "fake-bin"

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        private_root,
        fake_bin,
        "--command",
        "PRIVATE-COMMAND-MARKER",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert "PRIVATE" not in result.stderr
    assert not feedback_path.exists()


def test_measurement_cli_rejects_symlink_feedback_without_private_leak(
    tmp_path: Path,
) -> None:
    """Catches replacement through a pre-existing symlink destination."""
    binding = _synthetic_history_binding(tmp_path)
    private_root = Path(binding["private_root"])
    round_output_root = private_root / "controller-round"
    round_output_root.mkdir()
    binding_path = _write_json(round_output_root / "binding.json", binding)
    request_path = _write_json(round_output_root / "request.json", _measurement_request())
    outside = tmp_path / "PRIVATE-OUTSIDE.json"
    feedback_path = round_output_root / "feedback.json"
    feedback_path.symlink_to(outside)
    fake_bin = tmp_path / "fake-bin"

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        round_output_root,
        fake_bin,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "unsafe_destination\n"
    assert "PRIVATE" not in result.stderr
    assert not outside.exists()


def test_measurement_cli_rejects_preexisting_feedback_leaf(tmp_path: Path) -> None:
    """Catches an adapter overwriting stale or attacker-prepared feedback."""
    binding = _synthetic_history_binding(tmp_path)
    private_root = Path(binding["private_root"])
    round_output_root = private_root / "controller-round"
    round_output_root.mkdir()
    binding_path = _write_json(round_output_root / "binding.json", binding)
    request_path = _write_json(round_output_root / "request.json", _measurement_request())
    feedback_path = _write_json(round_output_root / "feedback.json", {"stale": True})

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        round_output_root,
        tmp_path / "fake-bin",
    )

    assert result.returncode == 1
    assert result.stderr == "unsafe_destination\n"
    assert json.loads(feedback_path.read_text(encoding="utf-8")) == {"stale": True}


def test_measurement_cli_rejects_repository_nonignored_destination(
    tmp_path: Path,
) -> None:
    """Catches private feedback targeting a tracked/public repository root."""
    feedback_path = REPOSITORY_ROOT / "p6-private-feedback-must-not-exist.json"
    assert not feedback_path.exists()
    private = tmp_path / "private"
    private.mkdir()
    binding_path = _write_json(private / "binding.json", {"private": "marker"})
    request_path = _write_json(private / "request.json", _measurement_request())

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        REPOSITORY_ROOT,
        tmp_path / "fake-bin",
    )

    assert result.returncode == 1
    assert result.stderr == "unsafe_destination\n"
    assert not feedback_path.exists()


def test_measurement_cli_private_binding_error_is_category_only(tmp_path: Path) -> None:
    """Catches malformed binding paths or contents escaping via stderr."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    marker = "PRIVATE-BINDING-MARKER"
    binding_path = private_root / f"{marker}.json"
    binding_path.write_text(f"{{not-json:{marker}}}", encoding="utf-8")
    request_path = _write_json(private_root / "request.json", _measurement_request())
    feedback_path = private_root / "feedback.json"

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        private_root,
        tmp_path / "fake-bin",
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_execution_invalid\n"
    assert marker not in result.stderr
    assert not feedback_path.exists()


def test_measurement_cli_rejects_symlinked_feedback_parent_within_private_root(
    tmp_path: Path,
) -> None:
    """Catches writing through an in-boundary symlinked destination parent."""
    binding = _synthetic_history_binding(tmp_path)
    private_root = Path(binding["private_root"])
    round_output_root = private_root / "controller-round"
    round_output_root.mkdir()
    binding_path = _write_json(round_output_root / "binding.json", binding)
    request_path = _write_json(round_output_root / "request.json", _measurement_request())
    actual_parent = round_output_root / "actual-private-parent"
    actual_parent.mkdir()
    link_parent = round_output_root / "link-parent"
    link_parent.symlink_to(actual_parent, target_is_directory=True)
    feedback_path = link_parent / "feedback.json"
    fake_bin = tmp_path / "fake-bin"
    _write_synthetic_nvidia_smi(fake_bin / "nvidia-smi")

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        round_output_root,
        fake_bin,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "unsafe_destination\n"
    assert not (actual_parent / "feedback.json").exists()


def test_prepare_destination_checks_git_ignore_on_exact_feedback_path(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches checking only the round root when a leaf may be explicitly unignored."""
    repository = tmp_path
    round_root = repository / "ignored-round"
    round_root.mkdir()
    feedback_path = round_root / "explicitly-unignored-feedback.json"
    checked: list[Path] = []

    monkeypatch.setattr(measurement_cli, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        measurement_cli,
        "_git_ignored",
        lambda _repository, candidate: checked.append(candidate) or True,
    )

    assert measurement_cli._prepare_destination(round_root, feedback_path) == feedback_path
    assert checked == [feedback_path]
