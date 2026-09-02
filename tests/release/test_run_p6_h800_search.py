from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage6.p6_history_normalization_v1 import normalize_history_inputs
from framework.stage6.p6_history_recipe_profiles_v1 import (
    PROFILE_V1,
    RECIPE_V2,
    get_recipe_profile,
)
from tests.release.test_p6_history_execution_adapters import (
    RTX_GPU_INDICES,
    _synthetic_history_binding,
    _synthetic_rtx_history_binding,
    _write_synthetic_nvidia_smi,
)
from tests.release.scanner_owned_stage1_fixture import (
    scanner_owned_pyramid_stage1_manifest,
)
from tests.stage6.test_p6_history_normalization import _history_root
from tests.stage6.test_p6_post_source_adapter_profile import v5_private_source_map

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide POSIX limits.
    resource = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/release/run_p6_h800_search.py"


def _public_contract(hardware_profile: str = "h800") -> dict[str, Any]:
    contract = {
        "schema_version": "p6_h800_coptv2x_search_contract_v2",
        "search_id": f"p6-pyramid-{hardware_profile}-tvm",
        "target": hardware_profile,
        "target_model": "pyramid",
        "execution_backend": "tvm_auto",
        "seed": 73,
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
        "configuration_label": f"p6-pyramid-{hardware_profile}-tvm",
        "candidate_space_label": "coptv2x-pyramid-width-grid-v1",
        "metric_names": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }
    if hardware_profile != "h800":
        contract.update(
            {
                "schema_version": "p6_coptv2x_search_contract_v3",
                "hardware_profile": hardware_profile,
            }
        )
    return contract


def _profile(hardware_profile: str = "h800") -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id=f"{hardware_profile}-tvm-auto",
        hardware_target=hardware_profile,
        compiler_fingerprint="a" * 64,
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )


def _non_target_profile(hardware_profile: str = "h800") -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id=f"{hardware_profile}-trt-engine",
        hardware_target=hardware_profile,
        compiler_fingerprint="b" * 64,
        dispatch_key="trt_engine",
        features={"int8_propagation": 1.0, "qdq_fold": 1.0},
    )


def _graph(group_id: str, width: list[int]) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "conv_count": 27,
        "conv_macs": float(width[0] * width[1] * width[2]),
        "group_conv_count": 3,
    }


def _gold176(
    hardware_profile: str = "h800",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for index in range(176):
        width = [16 + (index % 7) * 8, 32 + (index % 8) * 8, 64 + (index % 9) * 8]
        group_id = f"gold-{index:03d}"
        q_mode = "int8" if index % 2 else "fp16"
        non_target = index >= 88
        dispatch_key = "trt_engine" if non_target else "tvm_auto"
        profile_id = (
            f"{hardware_profile}-trt-engine"
            if non_target
            else f"{hardware_profile}-tvm-auto"
        )
        graphs.append(_graph(group_id, width))
        rows.append(
            {
                "manifest_job_id": f"{group_id}|q={q_mode}|profile={profile_id}",
                "row_id": f"{group_id}|q={q_mode}|profile={profile_id}",
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "dispatch_key": dispatch_key,
                "capability_profile_id": profile_id,
                "q_mode": q_mode,
                "latency_ms": 2.0 + index * 0.01,
                "energy_j": 0.5 + index * 0.005,
                "ap30": 0.90,
                "ap50": 0.80,
                "ap70": 0.70 - index * 0.0001,
                "terminal_status": "measured_success_gold",
                "training_source": "initial_coldstart",
            }
        )
    return rows, graphs


def _closure() -> dict[str, Any]:
    return {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": 176},
        "frozen_holdout": {"groups": []},
    }


def _source_group(group_id: str, width: list[int]) -> dict[str, Any]:
    evidence_sha = hashlib.sha256(f"source:{group_id}".encode()).hexdigest()
    source_contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": group_id,
        "model": "pyramid",
        "width": width,
        "artifact_id": f"fixture-{group_id}",
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "synthetic_fixture",
    }
    contract_sha = hashlib.sha256(
        json.dumps(
            source_contract,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": width,
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "source_contract": source_contract,
        "source_contract_sha256": contract_sha,
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
        "graph_features": _graph(group_id, width),
    }


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    return path


def _write_source_template(path: Path) -> Path:
    groups = [
        _source_group(
            f"pyramid|{first}x{second}x{third}",
            [first, second, third],
        )
        for first in range(16, 72, 8)
        for second in range(32, 88, 8)
        for third in range(64, 120, 8)
    ]
    assert len(groups) == 343
    path.write_text(
        json.dumps(
            {
                "schema_version": "stage5_candidate_source_registry_v1",
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fake_source_registry_adapter(path: Path) -> Path:
    path.write_text(
        """from __future__ import annotations
from pathlib import Path
import shutil
import sys

output_root, template_path, output_path = map(Path, sys.argv[1:])
del output_root
shutil.copyfile(template_path, output_path)
""",
        encoding="utf-8",
    )
    return path


def _real_stage1_partition_manifest(
    hardware_profile: str = "h800",
) -> dict[str, Any]:
    return scanner_owned_pyramid_stage1_manifest(
        {
            "schema": "stage1_partition_manifest_v1",
            "stage": "stage1_partition",
            "model": "pyramid_lidar",
            "scan_status": "ok",
            "hw_capability": {"name": hardware_profile},
            "view_b1_search_groups": [
                {
                    "search_group_id": f"pyramid_group.{suffix}",
                    "bucket": "pyramid_backbone",
                    "widths": [224],
                    "round_to": 32,
                    "int8_buildable_align": 32,
                    "max_rate": 0.875,
                    "grouped_conv": True,
                    "criterion_pool": ["L1"],
                    "member_b1_groups": [f"pyramid_group.{suffix}"],
                }
                for suffix in ("s0", "s1", "s2")
            ],
            "view_b2_quant_units": [
                {
                    "unit": "pyramid_backbone",
                    "quantizable": True,
                    "legal_bits": ["FP16", "INT8"],
                    "member_groups": [
                        "pyramid_group.s0",
                        "pyramid_group.s1",
                        "pyramid_group.s2",
                    ],
                }
            ],
            "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
        }
    )


def _write_fake_stage1_adapter(
    path: Path, hardware_profile: str = "h800"
) -> Path:
    manifest = json.dumps(
        _real_stage1_partition_manifest(hardware_profile), sort_keys=True
    )
    path.write_text(
        f"""from __future__ import annotations
from pathlib import Path
import sys

manifest_path, output_root, call_log = map(Path, sys.argv[1:])
del output_root
manifest_path.write_text({manifest!r}, encoding="utf-8")
with call_log.open("a", encoding="utf-8") as handle:
    handle.write("stage1\\n")
""",
        encoding="utf-8",
    )
    return path


def _write_noop_source_registry_adapter(path: Path) -> Path:
    path.write_text(
        """from __future__ import annotations
import sys

del sys.argv
""",
        encoding="utf-8",
    )
    return path


def _write_fake_measurement_adapter(path: Path) -> Path:
    path.write_text(
        """from __future__ import annotations
import json
from pathlib import Path
import sys

request_path, feedback_path, round_root, call_log, mode = sys.argv[1:]
del round_root
request = json.loads(Path(request_path).read_text(encoding="utf-8"))
with Path(call_log).open("a", encoding="utf-8") as handle:
    handle.write("measure\\n")
if mode == "fail":
    sys.stderr.write("PRIVATE_ADAPTER_STDERR\\n")
    raise SystemExit(9)
feedback = {
    "schema_version": "p6_h800_coptv2x_feedback_v2",
    "measurement_request_sha256": request["measurement_request_sha256"],
    "rows": [
        {
            "row_id": row["row_id"],
            "terminal_status": "measured_success_gold",
            "latency_ms": 2.0,
            "energy_j": 0.5,
            "ap30": 0.9,
            "ap50": 0.8,
            "ap70": 0.7,
        }
        for row in request["rows"]
    ],
}
Path(feedback_path).write_text(json.dumps(feedback), encoding="utf-8")
""",
        encoding="utf-8",
    )
    return path


def _cli_fixture(
    tmp_path: Path,
    *,
    mode: str = "success",
    hardware_profile: str = "h800",
) -> dict[str, Path]:
    contract_path = _write_yaml(
        tmp_path / "contract.yaml", _public_contract(hardware_profile)
    )
    gold_rows, gold_graphs = _gold176(hardware_profile)
    input_payloads = {
        "gold176_rows": gold_rows,
        "gold176_graph_features": gold_graphs,
        "capability_profiles": [
            _profile(hardware_profile),
            _non_target_profile(hardware_profile),
        ],
        "closure": _closure(),
    }
    input_paths: dict[str, str] = {}
    for name, payload in input_payloads.items():
        input_path = tmp_path / f"{name}.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        input_paths[name] = str(input_path)

    asset_paths: dict[str, str] = {}
    for label in ("training-data", "model-init", "toolchain"):
        asset_path = tmp_path / label
        asset_path.mkdir()
        asset_paths[label] = str(asset_path)

    source_template = _write_source_template(tmp_path / "source_template.json")
    source_adapter = _write_fake_source_registry_adapter(tmp_path / "fake_registry.py")
    measurement_adapter = _write_fake_measurement_adapter(tmp_path / "fake_measurement.py")
    call_log = tmp_path / "PRIVATE_CALL_LOG.txt"
    output_root = tmp_path / "PRIVATE_LOCAL_OUTPUT"
    local_payload = {
            "schema_version": "p6_h800_coptv2x_local_v2",
            "target": hardware_profile,
            "asset_paths": asset_paths,
            "local_input_paths": input_paths,
            "source_registry_step": {
                "name": "build_source_registry",
                "argv": [
                    sys.executable,
                    str(source_adapter),
                    "{local_output_root}",
                    str(source_template),
                    "{source_registry_json}",
                ],
            },
            "measurement_step": {
                "name": "measure_batch",
                "argv": [
                    sys.executable,
                    str(measurement_adapter),
                    "{measurement_request}",
                    "{feedback_json}",
                    "{round_output_root}",
                    str(call_log),
                    mode,
                ],
            },
            "local_output_root": str(output_root),
        }
    if hardware_profile != "h800":
        local_payload.update(
            {
                "schema_version": "p6_coptv2x_local_v3",
                "hardware_profile": hardware_profile,
            }
        )
    local_path = _write_yaml(
        tmp_path / "local.yaml",
        local_payload,
    )
    return {
        "contract": contract_path,
        "local": local_path,
        "call_log": call_log,
        "output_root": output_root,
    }


def _history_cli_fixture(
    tmp_path: Path,
    *,
    materializable: bool = True,
    hardware_profile: str = "h800",
) -> dict[str, Any]:
    """Build a private generated-binding integration fixture without real discovery."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    paths: dict[str, Any] = _cli_fixture(
        private_root, hardware_profile=hardware_profile
    )
    stage1 = private_root / "stage1.yaml"
    stage1_call_log = private_root / "stage1-call.log"
    stage1_adapter = _write_fake_stage1_adapter(
        private_root / "fake-stage1.py", hardware_profile
    )
    binding_path = private_root / "p6-history-binding.json"
    if hardware_profile == "rtx4090":
        post_source_workspace = tmp_path / "post-source-workspace"
        post_source_workspace.mkdir()
        source_map, runner_template = v5_private_source_map(post_source_workspace)
        source_map["hardware_profile"] = hardware_profile
        runtime_site = post_source_workspace / "approved-runtime/site-packages"
        runtime_site.mkdir(parents=True)
        nvlibs = post_source_workspace / "approved-runtime/nvlibs.path"
        nvlibs.write_text("/runtime/lib\n", encoding="utf-8")
        support = next(
            item
            for item in source_map["execution_code_closure"]["roots"]
            if item["closure_id"] == "tvm-support"
        )
        source_runner = yaml.safe_load(runner_template.read_text(encoding="utf-8"))
        source_runner["execution_interface"]["environment"]["values"].update(
            {
                "P6_TVM_PYTHON": {
                    "kind": "external_executable",
                    "value": "/usr/bin/python3.10",
                },
                "P6_TVM_SITE": {
                    "kind": "external_directory",
                    "value": str(runtime_site),
                },
                "P6_TVM_NVLIBS_FILE": {
                    "kind": "external_file",
                    "value": str(nvlibs),
                },
                "P6_TVM_SUPPORT_ROOT": {
                    "kind": "private_path",
                    "value": support["source_root"],
                },
                "P6_TVM_SUPPORT_ROOT_SHA256": {
                    "kind": "literal",
                    "value": support["sha256"],
                },
            }
        )
        runner_template.write_text(
            yaml.safe_dump(source_runner, sort_keys=False), encoding="utf-8"
        )
        normalized = normalize_history_inputs(
            source_map,
            _history_root(source_map),
            private_root / "synthetic-history",
            runner_template_path=runner_template,
        )
        binding = _synthetic_rtx_history_binding(private_root)
        normalized_runner = yaml.safe_load(
            normalized["runner_template"].read_text(encoding="utf-8")
        )
        formal_values = normalized_runner["execution_interface"]["environment"][
            "values"
        ]
        binding["execution_interface"]["environment"]["values"].update(
            {
                key: formal_values[key]
                for key in (
                    "P6_TVM_PYTHON",
                    "P6_TVM_SITE",
                    "P6_TVM_NVLIBS_FILE",
                    "P6_TVM_SUPPORT_ROOT",
                    "P6_TVM_SUPPORT_ROOT_SHA256",
                )
            }
        )
    else:
        binding = _synthetic_history_binding(private_root)
    profile = get_recipe_profile(PROFILE_V1)
    assert profile is not None
    operator_root = tmp_path / "operator-assets"
    base_checkpoint = operator_root / "base" / "model.ckpt"
    base_checkpoint.parent.mkdir(parents=True)
    base_checkpoint.write_text("synthetic base checkpoint\n", encoding="utf-8")
    dataset_root = operator_root / "dataset"
    dataset_root.mkdir(parents=True)
    pyramid_config = operator_root / "configs" / "pyramid.py"
    pyramid_config.parent.mkdir(parents=True)
    synthetic_caps = (128, 128, 128)
    pyramid_config.write_text(
        "model:\n  args:\n    fusion_backbone:\n"
        f"      num_filters: {list(synthetic_caps)!r}\n",
        encoding="utf-8",
    )
    source_contract = binding["source_contract_template"]
    source_contract.update(
        {
            "dynamic_materialization_recipe": {
                "schema_version": RECIPE_V2,
                "stage_width_fields": list(profile.stage_width_fields),
                "group_id_template": profile.group_id_template,
                "artifact_id_template": profile.artifact_id_template,
                "shared_source_path_templates": dict(
                    profile.shared_source_path_templates
                ),
            },
            "external_training_binding": {
                "schema_version": "p6_external_training_binding_v1",
                "training_required": True,
                "training_source_kind": "selected_candidate_finetune",
                "base_checkpoint_path": str(base_checkpoint),
                "base_checkpoint_sha256": hashlib.sha256(
                    base_checkpoint.read_bytes()
                ).hexdigest(),
                "dataset_root": str(dataset_root),
                "pyramid_config_path": str(pyramid_config),
                "pyramid_config_sha256": hashlib.sha256(
                    pyramid_config.read_bytes()
                ).hexdigest(),
                "training_parameters": {
                    "training_mode": "finetune_selected_width",
                    "epochs": 3,
                    "target_epoch": 31,
                    "seed": 20260821,
                    "optimizer": "adamw",
                    "learning_rate": 0.0001,
                    "batch_size": 1,
                    "dataset_split": "trainval_coptv2x",
                    "checkpoint_selection": "best_ap70",
                    "freeze_policy": "pyramid_backbone_partial",
                    "groups": 3,
                    "width_per_group": 5,
                    "base_stage_widths": list(synthetic_caps),
                },
            },
        }
    )
    if not materializable:
        source_contract["dynamic_materialization_recipe"][
            "shared_source_path_templates"
        ].pop("calibration_npz")
    source_materializer = _write_fake_source_materializer(
        Path(binding["private_root"])
        / "fixture-bin"
        / "stage5_materialize_round_sources_v1.sh"
    )
    binding["component_paths"]["source_materializer"] = source_materializer
    binding["execution_interface"]["execution_chain"][0]["argv"][0] = source_materializer
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local.update({"candidate_source_mode": "framework_stage2_search_space",
                  "stage2_search_space_path": str(stage1),
                  "stage1_scan_step": {"name": "build_stage1_partition", "argv": [
                      sys.executable, str(stage1_adapter), "{stage1_partition_manifest}",
                      "{local_output_root}", str(stage1_call_log)]},
                  "source_registry_step": {"name": "build_source_registry", "argv": [
                      sys.executable, str(REPOSITORY_ROOT / "tools/release/build_p6_history_registry.py"),
                      "--binding", str(binding_path), "--pyramid-candidate-plan",
                      "{pyramid_candidate_plan}", "--source-registry-json", "{source_registry_json}",
                      "--local-output-root", "{local_output_root}"]},
                  "measurement_step": {"name": "measure_batch", "argv": [
                      sys.executable, str(REPOSITORY_ROOT / "tools/release/measure_p6_history_batch.py"),
                      "--binding", str(binding_path), "--measurement-request",
                      "{measurement_request}", "--feedback-json", "{feedback_json}",
                      "--round-output-root", "{round_output_root}"]}})
    _write_yaml(paths["local"], local)
    fake_bin = private_root / "fake-bin"
    fake_bin.mkdir()
    gpu_probe = fake_bin / "nvidia-smi"
    if hardware_profile == "rtx4090":
        _write_rtx_nvidia_smi(gpu_probe)
    else:
        _write_synthetic_nvidia_smi(gpu_probe)
    paths.update({"private_root": private_root,
                  "binding": binding_path,
                  "stage1": stage1,
                  "stage1_call_log": stage1_call_log,
                  "env": {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}})
    return paths


def _write_rtx_nvidia_smi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = " ".join(
        f"'{index}, GPU-fixture-{index}, NVIDIA GeForce RTX 4090, 0, 100'"
        for index in RTX_GPU_INDICES
    )
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' {records}\n", encoding="utf-8")
    path.chmod(0o700)


def _write_fake_source_materializer(path: Path) -> str:
    """Materialize the complete recipe-v2 bundle for one first-use group only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!{sys.executable}
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

request_path = Path(sys.argv[sys.argv.index("--request") + 1])
group_id = sys.argv[sys.argv.index("--group-id") + 1]
request = json.loads(request_path.read_text(encoding="utf-8"))
row = next(row for row in request["rows"] if row["group_id"] == group_id)
contract = row["source_contract"]
paths = contract.get("shared_source_paths", contract)
directory_keys = {{"checkpoint_dir", "calibration_root", "trt_calibration_dir"}}
for key in (
    "checkpoint_path", "checkpoint_dir", "config_path", "training_done_marker",
    "onnx_path", "onnx_report_path", "calibration_root", "calibration_npz",
    "calibration_summary", "trt_calibration_dir", "source_done_marker",
):
    target = Path(paths[key])
    if key in directory_keys:
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture:{{key}}:{{group_id}}\\n", encoding="utf-8")
stage_log = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"]) / "executed-stages.log"
with stage_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(dict(
        stage="stage5_materialize_round_sources_v1.sh",
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

def _run_cli(
    paths: Mapping[str, Path],
    *extra: str,
    env: Mapping[str, str] | None = None,
    preexec_fn: object | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--local-config",
            str(paths["local"]),
            "--code-revision",
            "test-revision",
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
        preexec_fn=preexec_fn,
    )


def test_cli_runs_v2_loop_without_public_summary_and_keeps_outputs_local(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = _run_cli(paths)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "completed\n"
    assert "public-summary" not in result.stdout + result.stderr
    state = json.loads((paths["output_root"] / "state.json").read_text(encoding="utf-8"))
    serialized_state = json.dumps(state, sort_keys=True)
    assert state["status"] == "completed"
    assert state["completed_rounds"] == 4
    assert state["measured_candidate_count"] == 16
    assert str(tmp_path) not in serialized_state
    assert "candidate-" not in serialized_state
    static_registry = json.loads(
        (paths["output_root"] / "source_registry.json").read_text(encoding="utf-8")
    )
    static_group_count = len(static_registry["groups"])
    assert (static_group_count, static_group_count * 2) == (343, 686)
    assert paths["call_log"].read_text(encoding="utf-8").splitlines() == [
        "measure",
        "measure",
        "measure",
        "measure",
    ]


def test_cli_runs_v3_rtx_hardware_profile_through_existing_executable(
    tmp_path: Path,
) -> None:
    paths = _history_cli_fixture(tmp_path, hardware_profile="rtx4090")

    result = _run_cli(paths, env=paths["env"])

    assert result.returncode == 0, result.stderr
    assert result.stdout == "completed\n"
    requests = [
        json.loads(
            (paths["output_root"] / f"round-{round_index:02d}" / "measurement_request.json")
            .read_text(encoding="utf-8")
        )
        for round_index in range(4)
    ]
    assert {row["hardware_id"] for request in requests for row in request["rows"]} == {
        "rtx4090"
    }
    assert len({row["row_id"] for request in requests for row in request["rows"]}) == 16


def test_cli_rejects_hardware_profile_mismatch_before_adapter_launch(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path, hardware_profile="rtx4090")
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local["hardware_profile"] = "h800"
    _write_yaml(paths["local"], local)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()


def test_legacy_h800_hardware_profile_cli_state_remains_unchanged(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)

    result = _run_cli(paths)

    state = json.loads(
        (paths["output_root"] / "state.json").read_text(encoding="utf-8")
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "completed\n"
    assert result.stderr == ""
    assert "hardware_profile" not in state


def test_cli_runs_full_framework_lifecycle_through_provisioned_history_binding(
    tmp_path: Path,
) -> None:
    """A non-static Stage2 plan must traverse the complete fake history chain."""
    paths = _history_cli_fixture(tmp_path)
    public_documents = {
        path: path.read_bytes()
        for path in (
            REPOSITORY_ROOT / "docs/release-manifests/P6_H800_SEARCH_EXECUTION.md",
            REPOSITORY_ROOT / "docs/AAAI27_RELEASE_AUDIT.md",
        )
    }

    result = _run_cli(paths, env=paths["env"])

    assert result.returncode == 0, result.stderr
    assert result.stdout == "completed\n"
    assert result.stderr == ""
    assert paths["stage1_call_log"].read_text(encoding="utf-8").splitlines() == [
        "stage1"
    ]
    assert yaml.safe_load(paths["stage1"].read_text(encoding="utf-8")) == (
        _real_stage1_partition_manifest()
    )
    binding = json.loads(paths["binding"].read_text(encoding="utf-8"))
    binding_private_root = Path(binding["private_root"])
    plan = json.loads((paths["output_root"] / "pyramid_candidate_plan.json").read_text())
    registry = json.loads((paths["output_root"] / "source_registry.json").read_text())
    active_candidate_count = plan["candidate_count"]
    assert active_candidate_count >= 16
    assert active_candidate_count not in {343, 686}
    plan_identities = {
        (tuple(row["width"]), row["q_mode"]): tuple(row["source_point_ids"])
        for row in plan["candidates"]
    }
    registry_identities = {
        (tuple(group["width"]), q_mode): tuple(
            group["source_point_ids_by_q_mode"][q_mode]
        )
        for group in registry["groups"]
        for q_mode in group["available_q_modes"]
    }
    assert registry_identities == plan_identities

    requests = [
        json.loads(
            (paths["output_root"] / f"round-{round_index:02d}" / "measurement_request.json")
            .read_text(encoding="utf-8")
        )
        for round_index in range(4)
    ]
    assert [request["round_index"] for request in requests] == [0, 1, 2, 3]
    assert [len(request["rows"]) for request in requests] == [4, 4, 4, 4]
    selected = [row["row_id"] for request in requests for row in request["rows"]]
    assert len(selected) == len(set(selected)) == 16
    materialized_groups: set[str] = set()
    for request in requests:
        round_index = request["round_index"]
        history_root = (
            binding_private_root
            / "private-runs"
            / str(round_index)
        )
        assert json.loads(
            (history_root / "measurement-request.json").read_text(encoding="utf-8")
        ) == request
        stage_records = [
            json.loads(line)
            for line in (history_root / "executed-stages.log")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        expected_groups = sorted({row["group_id"] for row in request["rows"]})
        first_use_groups = [
            group_id for group_id in expected_groups if group_id not in materialized_groups
        ]
        assert [record["stage"] for record in stage_records] == [
                "activate-private",
                *(["stage5_materialize_round_sources_v1.sh"] * len(first_use_groups)),
            "quantize-private",
            "stage5_build_performance_plan_v2.py",
            "measure-ap-private",
            "stage5_finalize_feedback_v2.py",
        ]
        source_records = [
            record
            for record in stage_records
            if record["stage"] == "stage5_materialize_round_sources_v1.sh"
        ]
        expected_request_path = str(history_root / "measurement-request.json")
        policy_indices = binding["gpu_policy"]["indices"]
        assert [record["argv"] for record in source_records] == [
            [
                "--request",
                expected_request_path,
                "--model",
                "pyramid",
                "--group-id",
                group_id,
                "--gpu",
                str(policy_indices[index % len(policy_indices)]),
            ]
            for index, group_id in enumerate(first_use_groups)
        ]
        materialized_groups.update(expected_groups)
    assert not any(
        artifact.name == "private-runs"
        for artifact in paths["output_root"].rglob("*")
    )

    state = json.loads((paths["output_root"] / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["completed_rounds"] == 4
    assert state["measured_candidate_count"] == 16
    public_surface = result.stdout + result.stderr + json.dumps(state, sort_keys=True)
    assert str(tmp_path) not in public_surface
    assert "synthetic-history" not in public_surface
    assert "gpu_policy" not in public_surface
    assert all(
        uuid not in public_surface
        for uuid in binding["gpu_policy"]["uuid_by_index"].values()
    )
    assert all(path.read_bytes() == content for path, content in public_documents.items())
    assert all(
        artifact.resolve().is_relative_to(paths["private_root"].resolve())
        for artifact in paths["private_root"].rglob("*")
    )


def test_cli_rejects_unmaterializable_history_candidate_before_measurement(
    tmp_path: Path,
) -> None:
    """One incomplete source contract must reject the complete dynamic plan."""
    paths = _history_cli_fixture(tmp_path, materializable=False)

    result = _run_cli(paths, env=paths["env"])

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "execution_failed\n"
    plan = json.loads((paths["output_root"] / "pyramid_candidate_plan.json").read_text())
    assert plan["candidate_count"] >= 16
    assert plan["candidate_count"] not in {343, 686}
    assert not (paths["output_root"] / "source_registry.json").exists()
    assert not any(paths["output_root"].glob("round-*"))
    assert not any(paths["private_root"].rglob("executed-stages.log"))


def test_cli_rejects_legacy_public_summary_argument(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = _run_cli(paths, "--public-summary", str(tmp_path / "summary.json"))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not paths["call_log"].exists()


def test_cli_requires_explicit_local_config_without_starting_an_adapter(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--code-revision",
            "test-revision",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not paths["call_log"].exists()


def test_cli_fails_closed_for_an_invalid_local_contract(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local["measurement_step"]["argv"] = ["bash", "-c", "private"]
    _write_yaml(paths["local"], local)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()


def test_cli_rejects_framework_candidate_plan_token_in_static_source_mode(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local["source_registry_step"]["argv"].append("{pyramid_candidate_plan}")
    _write_yaml(paths["local"], local)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()


def test_cli_reports_an_unsafe_local_output_boundary_as_a_contract_error(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)
    protected_directory = tmp_path / "protected-output"
    protected_directory.mkdir()
    paths["output_root"].symlink_to(protected_directory, target_is_directory=True)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()


@pytest.mark.parametrize("document", ["contract", "local"])
def test_cli_reports_invalid_utf8_as_a_stable_contract_error(
    tmp_path: Path,
    document: str,
) -> None:
    paths = _cli_fixture(tmp_path)
    paths[document].write_bytes(b"\xff\xfe")

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert "Traceback" not in result.stderr
    assert not paths["call_log"].exists()


def test_cli_reports_a_local_measurement_failure_without_adapter_details(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path, mode="fail")

    result = _run_cli(paths)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "execution_failed\n"
    assert paths["call_log"].read_text(encoding="utf-8").splitlines() == ["measure"]
    state_text = (paths["output_root"] / "state.json").read_text(encoding="utf-8")
    state = json.loads(state_text)
    assert state == {
        "code_revision": "test-revision",
        "completed_rounds": 0,
        "failure_code": "command_failed",
        "measured_candidate_count": 0,
        "schema_version": "p6_h800_coptv2x_local_state_v2",
        "status": "failed",
    }
    for sentinel in ("PRIVATE_ADAPTER_STDERR", str(tmp_path), "candidate-", "checkpoint"):
        assert sentinel.lower() not in (result.stdout + result.stderr + state_text).lower()


@pytest.mark.skipif(
    resource is None or not hasattr(resource, "RLIMIT_FSIZE"),
    reason="requires a POSIX process file-size limit",
)
def test_cli_normalizes_a_local_record_write_error(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)
    paths["output_root"].mkdir()
    _write_source_template(paths["output_root"] / "source_registry.json")
    noop_source = _write_noop_source_registry_adapter(tmp_path / "noop_source.py")
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local["source_registry_step"]["argv"] = [
        sys.executable,
        str(noop_source),
        "{local_output_root}",
        "{source_registry_json}",
    ]
    _write_yaml(paths["local"], local)
    assert resource is not None
    _, hard_limit = resource.getrlimit(resource.RLIMIT_FSIZE)

    def prevent_controller_record_writes() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, hard_limit))

    result = _run_cli(
        paths,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("COV_CORE_", "COVERAGE_PROCESS_"))
            },
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONWARNINGS": "ignore",
        },
        preexec_fn=prevent_controller_record_writes,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "execution_failed\n"
    request_path = paths["output_root"] / "round-00" / "measurement_request.json"
    assert request_path.is_file()
    assert request_path.stat().st_size == 0
    assert not paths["call_log"].exists()
    for sentinel in (str(tmp_path), str(REPOSITORY_ROOT), "argv", "Traceback"):
        assert sentinel.lower() not in (result.stdout + result.stderr).lower()


def test_cli_rejects_an_empty_code_revision_without_starting_an_adapter(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--local-config",
            str(paths["local"]),
            "--code-revision",
            "",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not paths["call_log"].exists()
