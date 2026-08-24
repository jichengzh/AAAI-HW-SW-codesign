from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6.coptv2x_h800_search_v2 import (
    LocalExecutionStep,
    LocalP6CoptV2XConfig,
    P6CoptV2XContractError,
    P6CoptV2XExecutionError,
    PublicP6CoptV2XContract,
    load_local_config,
    load_public_contract,
    run_p6_coptv2x_search,
)
from framework.stage6.p6_full_chain_bootstrap_v1 import materialize_full_chain_binding
from framework.stage6.p6_history_binding_v1 import COMPONENT_MARKERS, GpuRecord
from framework.stage6.p6_history_normalization_v1 import normalize_history_inputs
from framework.stage6.p6_history_recipe_profiles_v1 import (
    RECIPE_V2,
    SHARED_SOURCE_PATH_KEYS,
    get_recipe_profile,
)
from framework.stage6.p6_history_registry_v1 import materialize_history_registry
from framework.stage6.p6_runner_template_validator_v1 import (
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_history_source_materialization_v1 import (
    P6HistorySourceMaterializationError,
    build_source_invocations,
    project_source_materialization_request,
    run_source_invocations,
)
from tests.p6_source_wrapper_support import write_test_project_python
from tests.stage6.pyramid_formal_space_support import (
    scanner_owned_pyramid_stage1_manifest,
    scanner_owned_pyramid_stage2_space,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _OfflineGpuProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        self.calls.append(indices)
        return tuple(
            GpuRecord(
                index=index,
                uuid=f"GPU-synthetic-{index}",
                model_name="NVIDIA H800 80GB HBM3",
                occupancy=0.0,
            )
            for index in indices
        )


def _write_yaml(path: Path, content: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def _public_contract(**overrides: Any) -> dict[str, Any]:
    contract = {
        "schema_version": "p6_h800_coptv2x_search_contract_v2",
        "search_id": "p6-pyramid-h800-tvm",
        "target": "h800",
        "target_model": "pyramid",
        "execution_backend": "tvm_auto",
        "seed": 73,
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
        "configuration_label": "p6-pyramid-h800-tvm",
        "candidate_space_label": "coptv2x-pyramid-width-grid-v1",
        "metric_names": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }
    return {**contract, **overrides}


def _local_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    output_root = tmp_path / "private-output"
    payload = {
        "schema_version": "p6_h800_coptv2x_local_v2",
        "target": "h800",
        "asset_paths": {
            "training-data": str(tmp_path / "training-data"),
            "model-init": str(tmp_path / "model-init"),
            "toolchain": str(tmp_path / "toolchain"),
        },
        "local_input_paths": {
            "gold176_rows": str(tmp_path / "gold176_rows.json"),
            "gold176_graph_features": str(tmp_path / "gold176_graph_features.json"),
            "capability_profiles": str(tmp_path / "capability_profiles.json"),
            "closure": str(tmp_path / "closure.json"),
        },
        "candidate_source_mode": "coptv2x_static_registry",
        "stage2_search_space_path": None,
        "source_registry_step": {
            "name": "build_source_registry",
            "argv": [
                "python",
                "local_build_registry.py",
                "{local_output_root}",
                "{source_registry_json}",
            ],
        },
        "measurement_step": {
            "name": "measure_batch",
            "argv": [
                "python",
                "local_measure.py",
                "{measurement_request}",
                "{feedback_json}",
                "{round_output_root}",
            ],
        },
        "local_output_root": str(output_root),
    }
    return {**payload, **overrides}


def _profile() -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id="h800-tvm-auto",
        hardware_target="h800",
        compiler_fingerprint="a" * 64,
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )


def _non_target_profile() -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id="h800-trt-engine",
        hardware_target="h800",
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
    *, include_non_target_backend: bool = False, include_graph_provenance: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for index in range(176):
        width = [16 + (index % 7) * 8, 32 + (index % 8) * 8, 64 + (index % 9) * 8]
        group_id = f"gold-{index:03d}"
        q_mode = "int8" if index % 2 else "fp16"
        non_target = include_non_target_backend and index >= 88
        dispatch_key = "trt_engine" if non_target else "tvm_auto"
        profile_id = "h800-trt-engine" if non_target else "h800-tvm-auto"
        graph = _graph(group_id, width)
        if include_graph_provenance:
            graph["source_annotation"] = "legacy_metadata"
        graphs.append(graph)
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


def test_initial_coldstart_keeps_true_failures_as_evidence_outside_value_fit() -> None:
    """The reviewed Gold176 ledger has 174 value rows and two real failures."""
    rows, graphs = _gold176(include_non_target_backend=True)
    for index, status in ((174, "feasibility_failure"), (175, "numerical_feasibility_failure")):
        rows[index] = {
            key: value
            for key, value in rows[index].items()
            if key not in {"latency_ms", "energy_j", "ap30", "ap50", "ap70"}
        }
        rows[index]["terminal_status"] = status

    bundle = execution.fit_initial_coldstart_bundle(
        rows, graphs, [_profile(), _non_target_profile()], seed=73
    )

    assert bundle.manifest["input_row_count"] == 176
    assert bundle.manifest["value_training_row_count"] == 174


def test_p6_graph_projection_keeps_only_structural_features() -> None:
    """Numerical labels or provenance must not become cold-start model features."""
    projected = execution._normalize_gold_graph_features(
        [
            {
                "group_id": "gold-000",
                "model": "pyramid",
                "width": [16, 32, 64],
                "input_dims": [1, 4, 192, 352],
                "conv_count": 27,
                "conv_macs": 32768,
                "latency_ms": 1.0,
                "ap70": 0.7,
                "source_rank": 3,
                "legacy_run_index": 4,
                "source_annotation": "legacy_metadata",
            }
        ]
    )

    assert projected == [
        {
            "group_id": "gold-000",
            "model": "pyramid",
            "width": [16, 32, 64],
            "input_dims": [1, 4, 192, 352],
            "conv_count": 27.0,
            "conv_macs": 32768.0,
        }
    ]


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


def _dynamic_recipe() -> dict[str, Any]:
    outputs = {
        q_mode: {
            f"{kind}_path_template": (f"materialized/{{group_id}}/{{q_mode}}/{kind}.json")
            for kind in ("training", "checkpoint", "onnx", "calibration")
        }
        for q_mode in ("fp16", "int8")
    }
    return {
        "schema_version": "p6_history_dynamic_materialization_recipe_v1",
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
        "group_id_template": "pyramid|{stage1_width}x{stage2_width}x{stage3_width}",
        "artifact_id_template": "pyramid-{stage1_width}-{stage2_width}-{stage3_width}",
        "output_path_templates_by_q_mode": outputs,
    }


def _dynamic_source_group() -> dict[str, Any]:
    evidence_sha = hashlib.sha256(b"p6-normalized-full-chain-source").hexdigest()
    contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": "pyramid|16x32x64",
        "model": "pyramid",
        "width": [16, 32, 64],
        "artifact_id": "pyramid-dynamic-template",
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "synthetic_fixture",
        "dynamic_materialization_recipe": _dynamic_recipe(),
    }
    return {
        "group_id": contract["group_id"],
        "model": contract["model"],
        "width": contract["width"],
        "source_status": contract["source_status"],
        "source_evidence_sha256": evidence_sha,
        "source_contract": contract,
        "source_contract_sha256": hashlib.sha256(
            json.dumps(
                contract,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "synthetic_fixture",
    }


def _write_executable(path: Path, body: str = "pass\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\nfrom __future__ import annotations\n{body}",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def _write_stage1_scan_adapter(path: Path) -> Path:
    manifest = json.dumps(_real_stage1_partition_manifest(), sort_keys=True)
    return _write_executable(
        path,
        (
            "from pathlib import Path\n"
            "import sys\n"
            "manifest_path, output_root = map(Path, sys.argv[1:])\n"
            "del output_root\n"
            f"manifest_path.write_text({manifest!r}, encoding='utf-8')\n"
        ),
    )


def _write_normalized_history_source_map(tmp_path: Path) -> dict[str, Any]:
    history_root = tmp_path / "history-source"
    history_root.mkdir()
    subprocess.run(["git", "init", "-q", str(history_root)], check=True)
    inputs = history_root / "inputs"
    inputs.mkdir()
    gold_rows, gold_graphs = _gold176(include_non_target_backend=True)
    for name, payload in {
        "gold176_rows": gold_rows,
        "gold176_graph_features": gold_graphs,
        "capability_profiles": [_profile(), _non_target_profile()],
        "closure": _closure(),
    }.items():
        (inputs / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    (history_root / "training-data").mkdir()
    (history_root / "model-init").mkdir()
    toolchain = history_root / "toolchain"
    _write_stage1_scan_adapter(toolchain / "scan-private")
    for marker, _version in COMPONENT_MARKERS.values():
        _write_executable(toolchain / marker)
    for name in ("activate-private", "quantize-private", "measure-ap-private"):
        _write_executable(toolchain / name)
    source_group = _dynamic_source_group()
    source_contract = source_group["source_contract"]
    source_contract.update(_complete_training_contract(history_root))
    source_group["source_contract_sha256"] = hashlib.sha256(
        json.dumps(
            source_contract,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "p6_history_normalization_source_v1",
        "history_root": str(history_root),
        "asset_paths": {
            "training-data": str(history_root / "training-data"),
            "model-init": str(history_root / "model-init"),
            "toolchain": str(toolchain),
        },
        "input_sources": {
            name: str(inputs / f"{name}.json")
            for name in (
                "gold176_rows",
                "gold176_graph_features",
                "capability_profiles",
                "closure",
            )
        },
        "source_contract": source_group,
        "dynamic_materialization_recipe": _dynamic_recipe(),
    }


def _complete_training_contract(
    history_root: Path,
    *,
    base_stage_widths: tuple[int, int, int] = (128, 128, 128),
) -> dict[str, Any]:
    operator_root = history_root.parent / f"{history_root.name}-operator-assets"
    base_checkpoint = operator_root / "base" / "model.ckpt"
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.write_text("synthetic base\n", encoding="utf-8")
    dataset_root = operator_root / "dataset"
    dataset_root.mkdir(exist_ok=True)
    pyramid_config = operator_root / "configs" / "pyramid.py"
    pyramid_config.parent.mkdir(exist_ok=True)
    pyramid_config.write_text(
        "model:\n  args:\n    fusion_backbone:\n"
        f"      num_filters: {list(base_stage_widths)!r}\n",
        encoding="utf-8",
    )
    return {
        "external_training_binding": {
            "schema_version": "p6_external_training_binding_v1",
            "training_required": True,
            "training_source_kind": "selected_candidate_finetune",
            "base_checkpoint_path": str(base_checkpoint),
            "base_checkpoint_sha256": hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(),
            "dataset_root": str(dataset_root),
            "pyramid_config_path": str(pyramid_config),
            "pyramid_config_sha256": hashlib.sha256(pyramid_config.read_bytes()).hexdigest(),
            "training_parameters": {
                "training_mode": "finetune_selected_width",
                "epochs": 3,
                "target_epoch": 9,
                "seed": 20260821,
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "batch_size": 1,
                "dataset_split": "trainval_coptv2x",
                "checkpoint_selection": "best_ap70",
                "freeze_policy": "pyramid_backbone_partial",
                "groups": 3,
                "width_per_group": 5,
                "base_stage_widths": list(base_stage_widths),
            },
        },
    }


def _migrate_normalized_training_registry(registry_path: Path, normalized_root: Path) -> Path:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    group = registry["groups"][0]
    contract = group["source_contract"]
    profile = get_recipe_profile("p6_stage5_pyramid_h800_tvm_profile_v1")
    assert profile is not None
    contract["dynamic_materialization_recipe"] = {
        "schema_version": RECIPE_V2,
        "stage_width_fields": list(profile.stage_width_fields),
        "group_id_template": profile.group_id_template,
        "artifact_id_template": profile.artifact_id_template,
        "shared_source_path_templates": dict(profile.shared_source_path_templates),
    }
    contract.update(_complete_training_contract(normalized_root))
    group["source_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    external_path = normalized_root / "external-training-binding.yaml"
    _write_yaml(external_path, contract["external_training_binding"])
    return external_path


def _write_normalized_source_wrapper_profile(path: Path, normalized_root: Path) -> Path:
    marker = "stage5_materialize_round_sources_v1.sh"
    implementation_name = "stage5_materialize_round_sources_v1.original.sh"
    implementation = normalized_root / "source-implementation" / implementation_name
    _write_executable(implementation)
    (normalized_root / "toolchain" / marker).unlink()
    project_python = write_test_project_python(path.parent)
    return _write_yaml(
        path,
        {
            "schema_version": "p6_private_source_wrapper_profile_v2",
            "wrapper_kind": "repo_cwd_exec_v1",
            "destination_relative_path": f"toolchain/{marker}",
            "implementation_relative_path": (f"source-implementation/{implementation_name}"),
            "implementation_cwd_relative_path": "source-implementation",
            "project_python": str(project_python),
        },
    )


def _write_runner_template(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "p6_history_runner_template_v1",
                "stage1_scan": {
                    "name": "build_stage1_partition",
                    "argv": [
                        "toolchain/scan-private",
                        "{stage1_partition_manifest}",
                        "{local_output_root}",
                    ],
                },
                "execution_interface": {
                    "schema_version": "p6_history_runner_interface_v1",
                    "controller": {"argv": ["toolchain/stage5_task_round_controller_v3.sh"]},
                    "execution_chain": [
                        {
                            "stage": "source_materialization",
                            "argv": [
                                "toolchain/stage5_materialize_round_sources_v1.sh",
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
                            "argv": [
                                "toolchain/quantize-private",
                                "{task_state}",
                                "{round_output_root}",
                            ],
                            "required_placeholders": [
                                "{task_state}",
                                "{round_output_root}",
                            ],
                        },
                        {
                            "stage": "performance",
                            "argv": [
                                "toolchain/stage5_build_performance_plan_v2.py",
                                "{task_state}",
                                "{round_output_root}",
                            ],
                            "required_placeholders": [
                                "{task_state}",
                                "{round_output_root}",
                            ],
                        },
                        {
                            "stage": "ap",
                            "argv": [
                                "toolchain/measure-ap-private",
                                "{task_state}",
                                "{round_output_root}",
                            ],
                            "required_placeholders": [
                                "{task_state}",
                                "{round_output_root}",
                            ],
                        },
                        {
                            "stage": "finalization",
                            "argv": [
                                "toolchain/stage5_finalize_feedback_v2.py",
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
                                "value": "17,19,23",
                            },
                            "P6_HISTORY_RUN_MODE": {
                                "kind": "literal",
                                "value": "bound",
                            },
                            "P6_HISTORY_PRIVATE_ROOT": {
                                "kind": "private_path",
                                "value": ".",
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
                        "activation_argv": [
                            "toolchain/activate-private",
                            "private-bound",
                        ],
                    },
                    "output_layout": {
                        "round_root_template": "private-runs/{round_id}",
                        "task_state": {
                            "path_template": ("private-runs/{round_id}/state/task-state.json"),
                            "format": "json",
                            "rows_key": "rows",
                            "row_id_key": "row_id",
                            "row_hash_key": "row_sha256",
                            "source_evidence_key": "source_evidence_sha256",
                            "status_key": "terminal_status",
                            "stage_key": "stage",
                            "row_count": 4,
                            "allowed_terminal_statuses": [
                                "measured_success_gold",
                                "feasibility_failure",
                                "numerical_feasibility_failure",
                            ],
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
                            "path_template": ("private-runs/{round_id}/actual-feedback.json"),
                            "format": "json",
                            "rows_key": "rows",
                            "row_id_key": "row_id",
                            "row_hash_key": "row_sha256",
                            "source_evidence_key": "source_evidence_sha256",
                            "status_key": "terminal_status",
                            "row_count": 4,
                            "allowed_terminal_statuses": [
                                "measured_success_gold",
                                "feasibility_failure",
                                "numerical_feasibility_failure",
                            ],
                            "metric_keys": [
                                "latency_ms",
                                "energy_j",
                                "ap30",
                                "ap50",
                                "ap70",
                            ],
                        },
                        "receipt": {
                            "path_template": "private-runs/{round_id}/receipt.json",
                            "format": "json",
                            "request_sha256_key": "measurement_request_sha256",
                            "row_hashes_key": "row_sha256",
                            "source_evidence_key": "source_evidence_sha256",
                        },
                        "finalization_barrier": {
                            "path_template": "private-runs/{round_id}/barrier.json",
                            "format": "json",
                            "request_sha256_key": "measurement_request_sha256",
                            "row_hashes_key": "row_sha256",
                            "source_evidence_key": "source_evidence_sha256",
                        },
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_source_registry(path: Path, *, count: int) -> None:
    widths = [
        [16 + (index // 49) * 8, 32 + (index // 7 % 7) * 8, 64 + (index % 7) * 8]
        for index in range(count)
    ]
    groups = [_source_group(f"pyramid|{'x'.join(map(str, width))}", width) for width in widths]
    path.write_text(
        json.dumps({"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}),
        encoding="utf-8",
    )


def _source_registry_from_plan(
    plan: Mapping[str, Any],
    *,
    materializable_widths: frozenset[tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    candidates_by_width: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
    for candidate in plan["candidates"]:
        candidates_by_width.setdefault(tuple(candidate["width"]), []).append(candidate)
    groups = []
    for width_identity, candidates in sorted(candidates_by_width.items()):
        width = list(width_identity)
        group = _source_group(f"pyramid|{'x'.join(map(str, width))}", width)
        group["available_q_modes"] = sorted(str(candidate["q_mode"]) for candidate in candidates)
        group["source_point_ids_by_q_mode"] = {
            str(candidate["q_mode"]): list(candidate["source_point_ids"])
            for candidate in candidates
        }
        if materializable_widths is not None and width_identity not in materializable_widths:
            group["source_status"] = "unavailable"
            group["source_contract"]["source_status"] = "unavailable"
            group["source_contract_sha256"] = hashlib.sha256(
                json.dumps(
                    group["source_contract"],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        groups.append(group)
    return {"schema_version": "stage5_candidate_source_registry_v2", "groups": groups}


def _write_source_registry_from_plan(
    path: Path,
    plan: Mapping[str, Any],
    *,
    materializable_widths: frozenset[tuple[int, ...]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            _source_registry_from_plan(
                plan, materializable_widths=materializable_widths
            )
        ),
        encoding="utf-8",
    )


def _write_recipe_v2_training_registry_from_plan(
    path: Path,
    plan: Mapping[str, Any],
    local_output_root: Path | None = None,
    *,
    materializable_widths: frozenset[tuple[int, ...]] | None = None,
) -> None:
    _write_source_registry_from_plan(
        path, plan, materializable_widths=materializable_widths
    )
    registry = json.loads(path.read_text(encoding="utf-8"))
    operator = path.parent.parent / "operator-assets"
    dataset = operator / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    checkpoint = operator / "base" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic base")
    config = operator / "configs" / "pyramid.py"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )
    for group in registry["groups"]:
        width = group["width"]
        group_slug = "-".join(map(str, width))
        contract = group["source_contract"]
        contract.update(
            {
                "artifact_id": f"pyramid-{group_slug}",
                "stage_widths": {
                    "stage1_width": width[0],
                    "stage2_width": width[1],
                    "stage3_width": width[2],
                },
                "external_training_binding": {
                    "schema_version": "p6_external_training_binding_v1",
                    "training_required": True,
                    "training_source_kind": "selected_candidate_finetune",
                    "base_checkpoint_path": str(checkpoint),
                    "base_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "dataset_root": str(dataset),
                    "pyramid_config_path": str(config),
                    "pyramid_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "training_parameters": {
                        "training_mode": "finetune_selected_width",
                        "epochs": 2,
                        "target_epoch": 9,
                        "seed": 20260821,
                        "optimizer": "adamw",
                        "learning_rate": 0.0001,
                        "batch_size": 1,
                        "dataset_split": "trainval_coptv2x",
                        "checkpoint_selection": "best_ap70",
                        "freeze_policy": "pyramid_backbone_partial",
                        "groups": 3,
                        "width_per_group": 5,
                        "base_stage_widths": [3, 5, 7],
                    },
                },
                "shared_source_paths": {
                    key: str((local_output_root or path.parent) / "materialized" / group_slug / key)
                    for key in SHARED_SOURCE_PATH_KEYS
                },
            }
        )
        group["source_contract_sha256"] = hashlib.sha256(
            json.dumps(
                contract,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    path.write_text(json.dumps(registry), encoding="utf-8")


def _synthetic_framework_plan(
    structure_count: int = 63,
    q_modes: tuple[str, ...] = ("fp16", "int8"),
) -> dict[str, Any]:
    widths = tuple((index, 1, 1) for index in range(1, structure_count + 1))
    candidates = [
        {
            "width": list(width),
            "q_mode": q_mode,
            "source_point_ids": [
                f"synthetic-s{stage}-{q_mode}-w{value}"
                for stage, value in enumerate(width, start=1)
            ],
        }
        for width in widths
        for q_mode in q_modes
    ]
    candidates.sort(key=lambda row: (tuple(row["width"]), row["q_mode"]))
    return {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "source_schema": "stage2_search_space_v1",
        "target_model": "pyramid",
        "hardware_target": "h800",
        "execution_backend": "tvm_auto",
        "candidate_source_mode": "framework_stage2_search_space",
        "structure_count": structure_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _complete_framework_stage2_search_space() -> dict[str, Any]:
    return scanner_owned_pyramid_stage2_space()


def _write_framework_stage1_partition_manifest(tmp_path: Path) -> Path:
    return _write_real_stage1_manifest(tmp_path / "framework-stage1.yaml")


def _real_stage1_partition_manifest() -> dict[str, Any]:
    """Return the minimum real Stage1 artifact accepted by the P6 scan gate."""
    return scanner_owned_pyramid_stage1_manifest()


def _write_real_stage1_manifest(path: Path) -> Path:
    return _write_yaml(path, _real_stage1_partition_manifest())


def _write_framework_registry(path: Path) -> None:
    plan = json.loads((path.parent / "pyramid_candidate_plan.json").read_text(encoding="utf-8"))
    _write_recipe_v2_training_registry_from_plan(path, plan, path.parent)


def _framework_stage1_scan_step() -> dict[str, Any]:
    return {
        "name": "build_stage1_partition",
        "argv": [
            "fake-stage1",
            "{stage1_partition_manifest}",
            "{local_output_root}",
        ],
    }


def _framework_local_payload(tmp_path: Path) -> dict[str, Any]:
    return _local_config(
        tmp_path,
        candidate_source_mode="framework_stage2_search_space",
        stage2_search_space_path=str(tmp_path / "stage1-partition.yaml"),
        stage1_scan_step=_framework_stage1_scan_step(),
        source_registry_step={
            "name": "build_source_registry",
            "argv": [
                "fake-registry",
                "{local_output_root}",
                "{source_registry_json}",
                "{pyramid_candidate_plan}",
            ],
        },
        measurement_step={
            "name": "measure_batch",
            "argv": [
                "fake-measure",
                "{measurement_request}",
                "{feedback_json}",
                "{round_output_root}",
            ],
        },
    )


def _framework_local_config_with_stage1_step(tmp_path: Path) -> LocalP6CoptV2XConfig:
    gold_rows, gold_graphs = _gold176(include_non_target_backend=True)
    for name, payload in {
        "gold176_rows": gold_rows,
        "gold176_graph_features": gold_graphs,
        "capability_profiles": [_profile(), _non_target_profile()],
        "closure": _closure(),
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    return load_local_config(
        _write_yaml(tmp_path / "local.yaml", _framework_local_payload(tmp_path)),
        contract,
    )


def _write_feedback_from_request(request_path: Path, feedback_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "p6_h800_coptv2x_feedback_v2",
        "measurement_request_sha256": request["measurement_request_sha256"],
        "rows": [
            {
                "row_id": row["row_id"],
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for row in request["rows"]
        ],
    }
    feedback_path.write_text(json.dumps(payload), encoding="utf-8")


class _FullChainCalls:
    """Record the public command boundary while producing only contract artifacts."""

    def __init__(self, *, static_registry: bool = False) -> None:
        self.static_registry = static_registry
        self.names: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.measurement_count = 0
        self.plan_identities: set[tuple[tuple[int, ...], str, tuple[str, ...]]] = set()
        self.registry_identities: set[tuple[tuple[int, ...], str, tuple[str, ...]]] = set()

    def runner(self, argv: tuple[str, ...], cwd: Path) -> int:
        command = argv[0]
        self.names = [*self.names, command]
        if command == "fake-stage1":
            _write_real_stage1_manifest(Path(argv[1]))
        elif command == "fake-registry":
            plan_path = Path(argv[3])
            registry_path = Path(argv[2])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.plan_identities = {
                (
                    tuple(candidate["width"]),
                    str(candidate["q_mode"]),
                    tuple(candidate["source_point_ids"]),
                )
                for candidate in plan["candidates"]
            }
            if self.static_registry:
                _write_source_registry(registry_path, count=343)
            else:
                _write_recipe_v2_training_registry_from_plan(
                    registry_path, plan, registry_path.parent
                )
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                self.registry_identities = {
                    (
                        tuple(group["width"]),
                        str(q_mode),
                        tuple(group["source_point_ids_by_q_mode"][q_mode]),
                    )
                    for group in registry["groups"]
                    for q_mode in group["available_q_modes"]
                }
        elif command == "fake-measure":
            request_path = Path(argv[1])
            feedback_path = Path(argv[2])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.requests = [*self.requests, request]
            self.measurement_count += 1
            _write_feedback_from_request(request_path, feedback_path)
        else:
            raise AssertionError(f"unexpected argv: {argv!r} from {cwd}")
        return 0


def _full_chain_local_config_and_fake_runner(
    tmp_path: Path,
) -> tuple[LocalP6CoptV2XConfig, _FullChainCalls]:
    return _framework_local_config_with_stage1_step(tmp_path), _FullChainCalls()


def _full_chain_local_config_and_static_registry_runner(
    tmp_path: Path,
) -> tuple[LocalP6CoptV2XConfig, _FullChainCalls]:
    return (
        _framework_local_config_with_stage1_step(tmp_path),
        _FullChainCalls(static_registry=True),
    )


def _minimal_request() -> dict[str, Any]:
    return {
        "measurement_request_sha256": "request-identity",
        "rows": [{"row_id": f"row-{index}"} for index in range(4)],
    }


def _minimal_feedback() -> dict[str, Any]:
    return {
        "schema_version": "p6_h800_coptv2x_feedback_v2",
        "measurement_request_sha256": "request-identity",
        "rows": [
            {
                "row_id": f"row-{index}",
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for index in range(4)
        ],
    }


def _minimal_task() -> execution.SearchTask:
    return execution.SearchTask(
        task_id="minimal-task",
        target_model="pyramid",
        hardware_id="h800",
        capability_profile=_profile(),
    )


def _predicted_manifest_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    predicted = []
    for index, row in enumerate(manifest["rows"]):
        values = {
            "latency_ms": 1.0 + index,
            "energy_j": 0.2 + index / 10,
            "ap70": 0.8 - index / 100,
        }
        predicted.append(
            {
                **copy.deepcopy(row),
                "predictions": values,
                "prediction_intervals": {
                    key: {
                        "lower": value - 0.1,
                        "median": value,
                        "upper": value + 0.1,
                    }
                    for key, value in values.items()
                },
            }
        )
    return predicted


def test_framework_controller_uses_only_materializable_registry_subset(
    tmp_path: Path,
) -> None:
    plan = _synthetic_framework_plan()
    ready_widths = frozenset((index, 1, 1) for index in range(1, 10))
    registry_path = tmp_path / "private-registry/source_registry.json"
    registry_path.parent.mkdir()
    _write_recipe_v2_training_registry_from_plan(
        registry_path,
        plan,
        tmp_path / "private-output",
        materializable_widths=ready_widths,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    task = _minimal_task()

    execution._validate_framework_registry_plan(registry, plan)
    manifest = execution.build_task_candidate_manifest(
        registry, task=task, measured_row_ids=set()
    )
    execution._validate_p6_source_space(registry, task, framework_plan=plan)

    assert len(registry["groups"]) == 63
    assert manifest["eligible_row_count"] == 18
    assert len(manifest["excluded"]) == 54
    assert {
        tuple(row["width"]) for row in manifest["rows"]
    } == ready_widths
    selection = execution.select_task_batch(
        _predicted_manifest_rows(manifest),
        measured_rows=[],
        measured_graph_features=[],
        task=task,
    )
    request = execution.build_measurement_request(
        task=task,
        selected_rows=selection["selected_rows"],
        round_index=0,
    )
    assert all(tuple(row["width"]) in ready_widths for row in request["rows"])
    projected = project_source_materialization_request(request)
    projected_path = tmp_path / "projected-request.json"
    projected_path.write_text(json.dumps(projected.request), encoding="utf-8")
    source_materializer = _write_executable(tmp_path / "source-materializer")
    invocations = build_source_invocations(
        projected_path,
        projected.ordered_group_ids,
        source_materializer=source_materializer,
        validated_gpu_policy={
            "indices": [101, 103, 107],
            "uuid_by_index": {
                "101": "GPU-synthetic-101",
                "103": "GPU-synthetic-103",
                "107": "GPU-synthetic-107",
            },
            "model": "h800",
            "maximum_occupancy": 0.05,
        },
    )

    class RecordingRunner:
        def __init__(self) -> None:
            self.group_ids: list[str] = []

        def run(
            self,
            argv: Sequence[str],
            *,
            cwd: Path,
            env: Mapping[str, str],
            shell: bool,
        ) -> subprocess.CompletedProcess[str]:
            del cwd, env
            assert shell is False
            self.group_ids.append(str(argv[6]))
            return subprocess.CompletedProcess(argv, 0)

    runner = RecordingRunner()
    run_source_invocations(
        invocations,
        runner=runner,
        cwd=tmp_path,
        env={"P6_OFFLINE_GATE": "synthetic"},
    )
    assert runner.group_ids
    assert all(
        tuple(map(int, group_id.removeprefix("pyramid|").split("x"))) in ready_widths
        for group_id in runner.group_ids
    )


def test_framework_controller_rejects_ready_subset_below_sample_budget(
    tmp_path: Path,
) -> None:
    plan = _synthetic_framework_plan()
    ready_widths = frozenset((index, 1, 1) for index in range(1, 8))
    registry_path = tmp_path / "private-registry/source_registry.json"
    registry_path.parent.mkdir()
    _write_source_registry_from_plan(
        registry_path,
        plan,
        materializable_widths=ready_widths,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    with pytest.raises(
        P6CoptV2XContractError, match="framework source space is below the sample budget"
    ):
        execution._validate_p6_source_space(
            registry, _minimal_task(), framework_plan=plan
        )


@pytest.mark.parametrize("invalid_status", ("diagnostic", "blocked"))
def test_framework_controller_rejects_unknown_nonmaterializable_status(
    tmp_path: Path,
    invalid_status: str,
) -> None:
    del tmp_path
    plan = _synthetic_framework_plan()
    ready_widths = frozenset((index, 1, 1) for index in range(1, 10))
    registry = _source_registry_from_plan(
        plan, materializable_widths=ready_widths
    )
    mutated = next(
        group for group in registry["groups"] if group["source_status"] == "unavailable"
    )
    mutated["source_status"] = invalid_status
    mutated["source_contract"]["source_status"] = invalid_status
    mutated["source_contract_sha256"] = hashlib.sha256(
        json.dumps(
            mutated["source_contract"],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(
        P6CoptV2XContractError, match="framework registry source status is invalid"
    ):
        execution._validate_p6_source_space(
            registry, _minimal_task(), framework_plan=plan
        )


def _loaded_local_config(
    tmp_path: Path,
    *,
    contract: PublicP6CoptV2XContract | None = None,
    include_stage2_search_space: bool = False,
    include_non_target_backend: bool = True,
    include_graph_provenance: bool = False,
) -> LocalP6CoptV2XConfig:
    gold_rows, gold_graphs = _gold176(
        include_non_target_backend=include_non_target_backend,
        include_graph_provenance=include_graph_provenance,
    )
    payloads = {
        "gold176_rows": gold_rows,
        "gold176_graph_features": gold_graphs,
        "capability_profiles": [
            _profile(),
            *([_non_target_profile()] if include_non_target_backend else []),
        ],
        "closure": _closure(),
    }
    for name, payload in payloads.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()
    loaded_contract = contract or load_public_contract(
        _write_yaml(tmp_path / "contract.yaml", _public_contract())
    )
    if include_stage2_search_space:
        local_config = _local_config(
            tmp_path,
            candidate_source_mode="framework_stage2_search_space",
            stage2_search_space_path=str(tmp_path / "framework-stage1.yaml"),
            stage1_scan_step=_framework_stage1_scan_step(),
            source_registry_step={
                "name": "build_source_registry",
                "argv": [
                    "python",
                    "local_build_registry.py",
                    "{local_output_root}",
                    "{source_registry_json}",
                    "{pyramid_candidate_plan}",
                ],
            },
        )
    else:
        local_config = _local_config(tmp_path)
    return load_local_config(_write_yaml(tmp_path / "local.yaml", local_config), loaded_contract)


def test_local_contract_accepts_only_known_candidate_source_modes(tmp_path: Path) -> None:
    """Unknown source modes cannot quietly restore a static candidate grid."""
    contract = load_public_contract(_write_yaml(tmp_path / "public.yaml", _public_contract()))
    static_contract = load_local_config(
        _write_yaml(
            tmp_path / "static.yaml",
            _local_config(tmp_path, candidate_source_mode="coptv2x_static_registry"),
        ),
        contract,
    )
    framework_contract = load_local_config(
        _write_yaml(
            tmp_path / "framework.yaml",
            _local_config(
                tmp_path,
                candidate_source_mode="framework_stage2_search_space",
                stage2_search_space_path=str(tmp_path / "space.yaml"),
                stage1_scan_step=_framework_stage1_scan_step(),
                source_registry_step={
                    "name": "build_source_registry",
                    "argv": [
                        "python",
                        "local_build_registry.py",
                        "{local_output_root}",
                        "{source_registry_json}",
                        "{pyramid_candidate_plan}",
                    ],
                },
            ),
        ),
        contract,
    )

    assert static_contract.candidate_source_mode == "coptv2x_static_registry"
    assert framework_contract.candidate_source_mode == "framework_stage2_search_space"

    with pytest.raises(P6CoptV2XContractError, match="candidate_source_mode"):
        load_local_config(
            _write_yaml(
                tmp_path / "bad.yaml",
                _local_config(tmp_path, candidate_source_mode="static_fallback"),
            ),
            contract,
        )


def test_local_contract_defaults_legacy_local_configuration_to_static_mode(
    tmp_path: Path,
) -> None:
    """Existing private P6.1 configurations remain static unless they opt into framework mode."""
    contract = load_public_contract(_write_yaml(tmp_path / "public.yaml", _public_contract()))
    legacy_config = _local_config(tmp_path)
    del legacy_config["candidate_source_mode"]
    del legacy_config["stage2_search_space_path"]

    loaded = load_local_config(_write_yaml(tmp_path / "legacy.yaml", legacy_config), contract)

    assert loaded.candidate_source_mode == "coptv2x_static_registry"
    assert loaded.stage2_search_space_path is None
    assert loaded.stage1_scan_step is None


def test_framework_mode_requires_a_stage1_scan_step(tmp_path: Path) -> None:
    """Dynamic P6 must not build candidates from a pre-existing stale manifest."""
    payload = _framework_local_payload(tmp_path)
    payload.pop("stage1_scan_step")

    with pytest.raises(P6CoptV2XContractError, match="stage1_scan_step"):
        load_local_config(
            _write_yaml(tmp_path / "local.yaml", payload),
            load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract())),
        )


def test_framework_run_executes_scan_before_registry_or_measurement(tmp_path: Path) -> None:
    """Stage1 writes the only manifest used to construct dynamic P6 candidates."""
    local = _framework_local_config_with_stage1_step(tmp_path)
    events: list[str] = []

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        events.append(argv[0])
        if argv[0] == "fake-stage1":
            _write_real_stage1_manifest(Path(argv[1]))
        elif argv[0] == "fake-registry":
            _write_framework_registry(cwd / "source_registry.json")
        else:
            _write_feedback_from_request(Path(argv[1]), Path(argv[2]))
        return 0

    state = run_p6_coptv2x_search(
        load_public_contract(_write_yaml(tmp_path / "public.yaml", _public_contract())),
        local,
        "test-revision",
        runner,
    )

    assert state.status == "completed"
    assert events[0] == "fake-stage1"
    assert events.count("fake-measure") == 4


def test_full_framework_lifecycle_uses_actual_scan_artifact_and_four_feedback_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public controller must execute the complete dynamic offline lifecycle."""
    local, calls = _full_chain_local_config_and_fake_runner(tmp_path)
    contract = load_public_contract(tmp_path / "contract.yaml")
    initial_fit_input_counts: list[int] = []
    online_fit_input_counts: list[int] = []
    real_initial_fit = execution.fit_initial_coldstart_bundle
    real_online_fit = execution.fit_online_bundle

    def recording_initial_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        initial_fit_input_counts.append(len(rows))
        return real_initial_fit(rows, *args, **kwargs)

    def recording_online_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        online_fit_input_counts.append(len(rows))
        return real_online_fit(rows, *args, **kwargs)

    monkeypatch.setattr(execution, "fit_initial_coldstart_bundle", recording_initial_fit)
    monkeypatch.setattr(execution, "fit_online_bundle", recording_online_fit)

    state = run_p6_coptv2x_search(contract, local, "test-revision", calls.runner)

    assert state.status == "completed"
    assert calls.names == ["fake-stage1", "fake-registry", *(["fake-measure"] * 4)]
    assert state.completed_rounds == 4
    assert state.measured_candidate_count == 16
    assert initial_fit_input_counts == [176]
    assert online_fit_input_counts == [180, 184, 188]
    assert [request["round_index"] for request in calls.requests] == [0, 1, 2, 3]
    assert all(len(request["rows"]) == 4 for request in calls.requests)
    selected = [row["row_id"] for request in calls.requests for row in request["rows"]]
    assert len(selected) == len(set(selected)) == 16
    assert calls.registry_identities == calls.plan_identities
    plan = json.loads(
        (local.local_output_root / "pyramid_candidate_plan.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (local.local_output_root / "source_registry.json").read_text(encoding="utf-8")
    )
    assert plan["schema_version"] == "p6_pyramid_candidate_plan_v2"
    assert plan["candidate_source_mode"] == "framework_stage2_search_space"
    assert plan["candidate_count"] >= 16
    assert plan["candidate_count"] not in {343, 686}
    assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
    public_state = state.local_state_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in public_state
    assert all(row_id not in public_state for row_id in selected)


def test_normalized_private_root_reaches_dynamic_stage2_and_registry_without_measurement(
    tmp_path: Path,
) -> None:
    """The normalizer/provision path must finish dynamic preflight before measurement."""
    source_map = _write_normalized_history_source_map(tmp_path)
    normalized_root = tmp_path / "normalized-private-root"
    normalized = normalize_history_inputs(
        source_map, Path(str(source_map["history_root"])), normalized_root
    )
    external_training = _migrate_normalized_training_registry(
        normalized["registry"], normalized_root
    )
    runner_template = _write_runner_template(normalized_root / "runner-template.yaml")
    source_wrapper_profile = _write_normalized_source_wrapper_profile(
        tmp_path / "source-wrapper-profile.yaml", normalized_root
    )
    provisioned_root = normalized_root / "provisioned"
    provisioned_root.mkdir()
    binding_path = provisioned_root / "binding.json"
    local_config_path = provisioned_root / "local.yaml"

    probe = _OfflineGpuProbe()
    materialize_full_chain_binding(
        normalized["legacy"],
        runner_template,
        provisioned_root,
        binding_path,
        local_config_path,
        probe,
        source_wrapper_profile=source_wrapper_profile,
        external_training_binding=external_training,
    )
    local = load_local_config(
        local_config_path,
        load_public_contract(_write_yaml(tmp_path / "public.yaml", _public_contract())),
    )
    command_roles: list[str] = []
    measurement_boundaries = 0
    measurement_adapter_calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_adapter_calls, measurement_boundaries
        if len(argv) > 1 and Path(argv[1]).name == "measure_p6_history_batch.py":
            command_roles.append("measurement-boundary")
            measurement_boundaries += 1
            return 1
        if len(argv) > 1 and Path(argv[1]).name == "build_p6_history_registry.py":
            command_roles.append("registry")
        else:
            command_roles.append("stage1")
        completed = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if Path(argv[0]).name == "measure_p6_history_batch.py":
            measurement_adapter_calls += 1
        return completed.returncode

    state = run_p6_coptv2x_search(
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract())),
        local,
        "test-revision",
        runner,
    )

    assert state.status == "failed"
    assert state.failure_code == "command_failed"
    assert state.measured_candidate_count == 0
    assert command_roles == ["stage1", "registry", "measurement-boundary"]
    assert measurement_boundaries == 1
    assert measurement_adapter_calls == 0
    assert probe.calls == [(17, 19, 23), (17, 19, 23)]
    manifest = yaml.safe_load(local.stage2_search_space_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "stage1_partition_manifest_v1"
    plan = json.loads(
        (local.local_output_root / "pyramid_candidate_plan.json").read_text(encoding="utf-8")
    )
    assert plan["schema_version"] == "p6_pyramid_candidate_plan_v2"
    assert plan["candidate_source_mode"] == "framework_stage2_search_space"
    assert plan["candidate_count"] == len(plan["candidates"])
    assert plan["candidate_count"] >= 16
    registry = json.loads(
        (local.local_output_root / "source_registry.json").read_text(encoding="utf-8")
    )
    assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
    task = execution.SearchTask(
        task_id="pre-measurement-proof",
        target_model="pyramid",
        hardware_id="h800",
        capability_profile=_profile(),
    )
    candidate_manifest = execution.build_task_candidate_manifest(
        registry, task=task, measured_row_ids=set()
    )
    assert candidate_manifest["eligible_row_count"] >= 16


def test_zero_gpu_stage2_to_source_invocation_black_box(tmp_path: Path) -> None:
    """Gate dynamic public planning through shared-source direct argv without processes."""
    plan = execution.build_pyramid_candidate_plan(
        scanner_owned_pyramid_stage2_space(
            {
                "stage1": [128],
                "stage2": [128],
                "stage3": [64, 128],
            }
        )
    )

    source_map = _write_normalized_history_source_map(tmp_path)
    history_root = Path(str(source_map["history_root"]))
    validated_template = validate_pre_provision_runner_template(
        _write_runner_template(history_root / "runner-template.yaml"),
        history_root,
    )
    profile = get_recipe_profile("p6_stage5_pyramid_h800_tvm_profile_v1")
    assert profile is not None
    template = copy.deepcopy(source_map["source_contract"]["source_contract"])
    template["dynamic_materialization_recipe"] = {
        "schema_version": RECIPE_V2,
        "stage_width_fields": list(profile.stage_width_fields),
        "group_id_template": profile.group_id_template,
        "artifact_id_template": profile.artifact_id_template,
        "shared_source_path_templates": dict(profile.shared_source_path_templates),
    }
    template.update(_complete_training_contract(history_root))
    synthetic_gpu_indices = (17, 19, 23)
    execution_interface = copy.deepcopy(validated_template.execution_interface)
    execution_interface["environment"]["values"]["P6_HISTORY_PRIVATE_ROOT"]["value"] = str(
        history_root
    )
    binding = {
        "schema_version": "p6_history_binding_v1",
        "target": {"model": "pyramid", "hardware": "h800", "backend": "tvm_auto"},
        "private_root": str(history_root),
        "component_paths": {
            role: str(path) for role, path in validated_template.component_paths.items()
        },
        "execution_interface": execution_interface,
        "source_contract_template": template,
        "gpu_policy": {
            "indices": list(synthetic_gpu_indices),
            "uuid_by_index": {
                str(index): f"GPU-offline-{index}" for index in synthetic_gpu_indices
            },
            "model": "h800",
            "maximum_occupancy": 0.05,
        },
        "status": "validated",
    }
    registry_root = tmp_path / "private-registry"
    registry_root.mkdir()
    registry = materialize_history_registry(plan, binding, registry_root)

    plan_identity = {
        (tuple(candidate["width"]), candidate["q_mode"]): tuple(candidate["source_point_ids"])
        for candidate in plan["candidates"]
    }
    registry_identity = {
        (tuple(group["width"]), q_mode): tuple(group["source_point_ids_by_q_mode"][q_mode])
        for group in registry["groups"]
        for q_mode in group["available_q_modes"]
    }
    task = _minimal_task()
    manifest = execution.build_task_candidate_manifest(registry, task=task, measured_row_ids=set())
    predicted_rows = []
    for index, row in enumerate(manifest["rows"]):
        predictions = {
            "latency_ms": 1.0 + index,
            "energy_j": 0.2 + index / 10,
            "ap70": 0.8 - index / 100,
        }
        predicted_rows.append(
            {
                **copy.deepcopy(row),
                "predictions": predictions,
                "prediction_intervals": {
                    key: {
                        "lower": value - 0.1,
                        "median": value,
                        "upper": value + 0.1,
                    }
                    for key, value in predictions.items()
                },
            }
        )
    selection = execution.select_task_batch(
        predicted_rows,
        measured_rows=[],
        measured_graph_features=[],
        task=task,
    )
    request = execution.build_measurement_request(
        task=task,
        selected_rows=selection["selected_rows"],
        round_index=0,
    )
    projected = project_source_materialization_request(request)
    projected_path = tmp_path / "projected-request.json"
    projected_path.write_text(json.dumps(projected.request), encoding="utf-8")
    invocations = build_source_invocations(
        projected_path,
        projected.ordered_group_ids,
        source_materializer=validated_template.component_paths["source_materializer"],
        validated_gpu_policy=binding["gpu_policy"],
    )

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []
            self.subprocess_calls = 0

        def run(
            self,
            argv: Sequence[str],
            *,
            cwd: Path,
            env: Mapping[str, str],
            shell: bool,
        ) -> subprocess.CompletedProcess[str]:
            assert cwd == tmp_path
            assert env == {"P6_OFFLINE_GATE": "synthetic"}
            assert shell is False
            self.calls.append(tuple(argv))
            return subprocess.CompletedProcess(argv, 0)

    runner = RecordingRunner()
    run_source_invocations(
        invocations,
        runner=runner,
        cwd=tmp_path,
        env={"P6_OFFLINE_GATE": "synthetic"},
    )

    assert plan_identity == registry_identity
    assert plan["candidate_count"] == len(plan["candidates"])
    assert manifest["eligible_row_count"] == plan["candidate_count"]
    assert selection["selected_row_count"] == plan["candidate_count"] == 4
    rows_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in projected.request["rows"]:
        rows_by_group.setdefault(row["group_id"], []).append(row)
    mixed_rows = rows_by_group["pyramid|128x128x64"]
    assert {row["q_mode"] for row in mixed_rows} == {"fp16", "int8"}
    assert {key: mixed_rows[0]["source_contract"][key] for key in SHARED_SOURCE_PATH_KEYS} == {
        key: mixed_rows[1]["source_contract"][key] for key in SHARED_SOURCE_PATH_KEYS
    }
    all_paths = [
        row_group[0]["source_contract"][key]
        for row_group in rows_by_group.values()
        for key in SHARED_SOURCE_PATH_KEYS
    ]
    assert (
        len(all_paths) == len(set(all_paths)) == (len(rows_by_group) * len(SHARED_SOURCE_PATH_KEYS))
    )
    assert len(runner.calls) == len(rows_by_group)
    assert [call[1::2] for call in runner.calls] == [
        ("--request", "--model", "--group-id", "--gpu")
    ] * len(rows_by_group)
    assert [call[6] for call in runner.calls] == sorted(rows_by_group)
    assert [call[8] for call in runner.calls] == [
        str(index) for index in synthetic_gpu_indices[: len(rows_by_group)]
    ]
    for row in projected.request["rows"]:
        assert (
            row["source_contract_sha256"]
            == hashlib.sha256(
                json.dumps(
                    row["source_contract"],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        assert (
            projected.request["row_sha256"][row["row_id"]]
            == hashlib.sha256(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
    request_body = {
        key: value
        for key, value in projected.request.items()
        if key != "measurement_request_sha256"
    }
    assert (
        projected.request["measurement_request_sha256"]
        == hashlib.sha256(
            json.dumps(
                request_body,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert runner.subprocess_calls == 0
    called_executables = {Path(call[0]).name for call in runner.calls}
    assert "stage5_task_round_controller_v3.sh" not in called_executables
    assert called_executables == {"stage5_materialize_round_sources_v1.sh"}
    assert not called_executables.intersection(
        {
            "quantize-private",
            "stage5_build_performance_plan_v2.py",
            "measure-ap-private",
            "stage5_finalize_feedback_v2.py",
        }
    )


def test_framework_lifecycle_never_uses_static_registry_after_stage1(
    tmp_path: Path,
) -> None:
    """A fresh Stage1 artifact cannot be rebound to the legacy static registry."""
    local, calls = _full_chain_local_config_and_static_registry_runner(tmp_path)
    contract = load_public_contract(tmp_path / "contract.yaml")

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "test-revision", calls.runner)

    assert captured.value.failure_code == "source_registry_invalid"
    assert calls.measurement_count == 0
    assert calls.names == ["fake-stage1", "fake-registry"]


def test_framework_scan_rejects_a_stale_manifest_when_the_command_noops(
    tmp_path: Path,
) -> None:
    """A current Stage1 run cannot silently consume the prior run's manifest."""
    local = _framework_local_config_with_stage1_step(tmp_path)
    _write_real_stage1_manifest(local.stage2_search_space_path)
    events: list[str] = []

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        events.append(argv[0])
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(
            load_public_contract(_write_yaml(tmp_path / "public.yaml", _public_contract())),
            local,
            "test-revision",
            runner,
        )

    assert captured.value.failure_code == "unsafe_output"
    assert events == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            lambda tmp_path: _local_config(
                tmp_path,
                candidate_source_mode="framework_stage2_search_space",
                stage2_search_space_path="relative-stage2.yaml",
                source_registry_step={
                    "name": "build_source_registry",
                    "argv": [
                        "python",
                        "local_build_registry.py",
                        "{local_output_root}",
                        "{source_registry_json}",
                        "{pyramid_candidate_plan}",
                    ],
                },
            ),
            "stage2_search_space_path.*absolute path",
        ),
        (
            lambda tmp_path: _local_config(
                tmp_path, stage2_search_space_path=str(tmp_path / "stage2.yaml")
            ),
            "stage2_search_space_path.*framework mode",
        ),
        (
            lambda tmp_path: _local_config(
                tmp_path,
                source_registry_step={
                    "name": "build_source_registry",
                    "argv": [
                        "python",
                        "local_build_registry.py",
                        "{local_output_root}",
                        "{source_registry_json}",
                        "{pyramid_candidate_plan}",
                    ],
                },
            ),
            "template tokens",
        ),
    ],
)
def test_local_contract_rejects_stage2_paths_and_plan_tokens_outside_framework_mode(
    tmp_path: Path,
    payload: Callable[[Path], dict[str, Any]],
    message: str,
) -> None:
    """Framework-only inputs cannot change the static P6.1 boundary."""
    contract = load_public_contract(_write_yaml(tmp_path / "public.yaml", _public_contract()))

    with pytest.raises(P6CoptV2XContractError, match=message):
        load_local_config(_write_yaml(tmp_path / "local.yaml", payload(tmp_path)), contract)


def test_run_p6_framework_mode_builds_dynamic_candidate_plan_and_runs_four_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Framework candidates must originate from the Stage1-to-Stage2 path."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(
        tmp_path,
        contract=contract,
        include_stage2_search_space=True,
    )
    monkeypatch.setattr(
        execution,
        "load_stage2_search_space",
        lambda path: _complete_framework_stage2_search_space(),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []
    requests: list[dict[str, Any]] = []
    initial_fit_input_counts: list[int] = []
    online_fit_input_counts: list[int] = []
    observed_plan: dict[str, Any] = {}
    observed_manifest_candidate_identities: set[tuple[tuple[int, ...], str]] = set()
    real_initial_fit = execution.fit_initial_coldstart_bundle
    real_online_fit = execution.fit_online_bundle

    def recording_initial_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        initial_fit_input_counts.append(len(rows))
        return real_initial_fit(rows, *args, **kwargs)

    def recording_online_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        online_fit_input_counts.append(len(rows))
        return real_online_fit(rows, *args, **kwargs)

    monkeypatch.setattr(execution, "fit_initial_coldstart_bundle", recording_initial_fit)
    monkeypatch.setattr(execution, "fit_online_bundle", recording_online_fit)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal observed_plan, observed_manifest_candidate_identities
        calls.append((argv[0], argv))
        if argv[0] == "fake-stage1":
            _write_real_stage1_manifest(Path(argv[1]))
            return 0
        del cwd
        if argv[1] == "local_build_registry.py":
            candidate_plan_path = Path(argv[4])
            plan = json.loads(candidate_plan_path.read_text(encoding="utf-8"))
            observed_plan = plan
            assert candidate_plan_path.name == "pyramid_candidate_plan.json"
            assert plan["schema_version"] == "p6_pyramid_candidate_plan_v2"
            assert plan["candidate_source_mode"] == "framework_stage2_search_space"
            _write_recipe_v2_training_registry_from_plan(Path(argv[3]), plan, Path(argv[3]).parent)
            registry = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
            manifest = execution.build_task_candidate_manifest(
                registry, task=_minimal_task(), measured_row_ids=set()
            )
            observed_manifest_candidate_identities = {
                (tuple(row["width"]), row["q_mode"]) for row in manifest["rows"]
            }
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        requests.append(request)
        _write_feedback_from_request(Path(argv[2]), Path(argv[3]))
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    observed_plan_candidate_identities = {
        (tuple(candidate["width"]), candidate["q_mode"])
        for candidate in observed_plan["candidates"]
    }

    assert state.status == "completed"
    assert state.completed_rounds == 4
    assert state.measured_candidate_count == 16
    active_candidate_count = observed_plan["candidate_count"]
    assert active_candidate_count >= 16
    assert active_candidate_count not in {343, 686}
    assert observed_manifest_candidate_identities == observed_plan_candidate_identities
    assert [request["round_index"] for request in requests] == [0, 1, 2, 3]
    assert [len(request["rows"]) for request in requests] == [4, 4, 4, 4]
    assert all(
        request["schema_version"] == "stage5_measurement_request_v2"
        and request["required_metrics"] == ["latency_ms", "energy_j", "ap30", "ap50", "ap70"]
        and request["atomic_feedback"] is True
        for request in requests
    )
    selected = [row["row_id"] for request in requests for row in request["rows"]]
    assert len(selected) == len(set(selected)) == 16
    assert initial_fit_input_counts == [176]
    assert online_fit_input_counts == [180, 184, 188]
    assert calls[0][0] == "fake-stage1"


@pytest.mark.parametrize("mutation", ["changed", "omitted", "added", "duplicate"])
def test_run_p6_framework_mode_rejects_registry_plan_identity_mismatches_before_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """An adapter cannot change, omit, add, or duplicate a framework candidate."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(
        tmp_path,
        contract=contract,
        include_stage2_search_space=True,
    )
    monkeypatch.setattr(
        execution,
        "load_stage2_search_space",
        lambda path: _complete_framework_stage2_search_space(),
    )
    measurement_calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_calls
        del cwd
        if argv[0] == "fake-stage1":
            _write_real_stage1_manifest(Path(argv[1]))
            return 0
        if argv[1] != "local_build_registry.py":
            measurement_calls += 1
            raise AssertionError("measurement must not start after a registry-plan mismatch")
        plan = json.loads(Path(argv[4]).read_text(encoding="utf-8"))
        registry_path = Path(argv[3])
        _write_source_registry_from_plan(registry_path, plan)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if mutation == "changed":
            q_mode = registry["groups"][0]["available_q_modes"][0]
            registry["groups"][0]["source_point_ids_by_q_mode"][q_mode][0] = "changed"
        elif mutation == "omitted":
            q_mode = registry["groups"][0]["available_q_modes"].pop()
            del registry["groups"][0]["source_point_ids_by_q_mode"][q_mode]
        elif mutation == "added":
            group = copy.deepcopy(registry["groups"][0])
            group["group_id"] = "pyramid|1x1x1"
            group["width"] = [1, 1, 1]
            group["source_contract"]["group_id"] = group["group_id"]
            group["source_contract"]["width"] = group["width"]
            registry["groups"].append(group)
        else:
            registry["groups"][-1] = dict(registry["groups"][0])
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == "source_registry_invalid"
    assert measurement_calls == 0


def test_run_p6_framework_mode_rejects_v1_registry_before_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Framework provenance cannot be erased behind the static v1 registry schema."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(
        tmp_path,
        contract=contract,
        include_stage2_search_space=True,
    )
    monkeypatch.setattr(
        execution,
        "load_stage2_search_space",
        lambda path: _complete_framework_stage2_search_space(),
    )
    measurement_calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_calls
        del cwd
        if argv[0] == "fake-stage1":
            _write_real_stage1_manifest(Path(argv[1]))
            return 0
        if argv[1] != "local_build_registry.py":
            measurement_calls += 1
            raise AssertionError("measurement must not start for a framework v1 registry")
        plan = json.loads(Path(argv[4]).read_text(encoding="utf-8"))
        widths = sorted({tuple(candidate["width"]) for candidate in plan["candidates"]})
        groups = [
            _source_group(f"pyramid|{'x'.join(map(str, width))}", list(width)) for width in widths
        ]
        Path(argv[3]).write_text(
            json.dumps(
                {
                    "schema_version": "stage5_candidate_source_registry_v1",
                    "groups": groups,
                }
            ),
            encoding="utf-8",
        )
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == "source_registry_invalid"
    assert measurement_calls == 0


def test_run_p6_framework_source_space_rejects_fewer_than_16_eligible_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dynamic source must still provide the frozen public sample budget."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(
        tmp_path,
        contract=contract,
        include_stage2_search_space=True,
    )

    def undersized_search_space(path: Path) -> dict[str, Any]:
        del path
        return scanner_owned_pyramid_stage2_space(
            {
                "stage1": [128],
                "stage2": [128],
                "stage3": [128],
            },
            q_modes=["fp16"],
        )

    monkeypatch.setattr(execution, "load_stage2_search_space", undersized_search_space)
    measurement_calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_calls
        del cwd
        if argv[0] == "fake-stage1":
            _write_real_stage1_manifest(Path(argv[1]))
            return 0
        if argv[1] != "local_build_registry.py":
            measurement_calls += 1
            raise AssertionError("measurement must not start below the sample budget")
        plan = json.loads(Path(argv[4]).read_text(encoding="utf-8"))
        assert plan["candidate_count"] == 1
        _write_source_registry_from_plan(Path(argv[3]), plan)
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == "source_registry_invalid"
    assert measurement_calls == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda space: space["axis_schema"]["free_axes"][0].update({"legal_widths": []}),
        lambda space: space["formal_q_modes"].append("fp32"),
        lambda space: space["axis_schema"]["free_axes"][0].update(
            {"base_width": 1}
        ),
    ],
    ids=["missing-axis-widths", "unsupported-fp32", "over-base-axis"],
)
def test_run_p6_framework_mode_rejects_invalid_stage2_points_before_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: Callable[[dict[str, Any]], Any]
) -> None:
    """Framework validation fails closed before registry materialization or measurement."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(
        tmp_path,
        contract=contract,
        include_stage2_search_space=True,
    )
    command_calls = 0

    def invalid_loader(path: Path) -> dict[str, Any]:
        search_space = _complete_framework_stage2_search_space()
        mutation(search_space)
        return search_space

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal command_calls
        del cwd
        command_calls += 1
        if argv[0] == "fake-stage1":
            _write_real_stage1_manifest(Path(argv[1]))
        return 0

    monkeypatch.setattr(execution, "load_stage2_search_space", invalid_loader)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == "source_registry_invalid"
    assert command_calls == 1


def test_run_p6_static_p61_rejects_343_groups_below_686_genomes(
    tmp_path: Path,
) -> None:
    """The fixed genome gate remains independent of the source-group count gate."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    measurement_calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_calls
        del cwd
        if argv[1] != "local_build_registry.py":
            measurement_calls += 1
            raise AssertionError("measurement must not start below the 686-genome threshold")
        registry_path = Path(argv[3])
        _write_source_registry(registry_path, count=343)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["groups"][0]["source_status"] = "unavailable"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == "source_registry_invalid"
    assert measurement_calls == 0


def test_run_p6_rejects_incomplete_coldstart_profile_context(tmp_path: Path) -> None:
    """Gold176 fitting must receive every capability profile represented in its ledger."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    profiles_path = local.local_input_paths["capability_profiles"]
    profiles_path.write_text(json.dumps([_profile()]), encoding="utf-8")

    with pytest.raises(P6CoptV2XContractError, match="coldstart capability"):
        run_p6_coptv2x_search(
            contract,
            local,
            "abc123",
            lambda argv, cwd: (_ for _ in ()).throw(
                AssertionError(f"source step must not run: {argv} {cwd}")
            ),
        )


def test_run_p6_builds_registry_refits_gold176_and_runs_four_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(
        tmp_path,
        include_non_target_backend=True,
        include_graph_provenance=True,
    )
    requests: list[dict[str, Any]] = []
    initial_fit_input_counts: list[int] = []
    online_fit_input_counts: list[int] = []
    real_initial_fit = execution.fit_initial_coldstart_bundle
    real_online_fit = execution.fit_online_bundle

    def recording_initial_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        initial_fit_input_counts.append(len(rows))
        return real_initial_fit(rows, *args, **kwargs)

    def recording_online_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        online_fit_input_counts.append(len(rows))
        return real_online_fit(rows, *args, **kwargs)

    monkeypatch.setattr(execution, "fit_initial_coldstart_bundle", recording_initial_fit)
    monkeypatch.setattr(execution, "fit_online_bundle", recording_online_fit)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            source_registry_path = Path(argv[3])
            _write_source_registry(source_registry_path, count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        requests.append(request)
        feedback = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": [
                {
                    "row_id": row["row_id"],
                    "terminal_status": "measured_success_gold",
                    "latency_ms": 3.0 + len(requests),
                    "energy_j": 0.7 + len(requests) / 10,
                    "ap30": 0.91,
                    "ap50": 0.82,
                    "ap70": 0.73,
                }
                for row in request["rows"]
            ],
        }
        Path(argv[3]).write_text(json.dumps(feedback), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert state.completed_rounds == 4
    assert state.measured_candidate_count == 16
    assert len(requests) == 4
    assert [request["round_index"] for request in requests] == [0, 1, 2, 3]
    assert all(
        request["required_metrics"] == ["latency_ms", "energy_j", "ap30", "ap50", "ap70"]
        for request in requests
    )
    selected = [row["row_id"] for request in requests for row in request["rows"]]
    assert len(selected) == len(set(selected)) == 16
    assert {row["q_mode"] for request in requests for row in request["rows"]} <= {
        "fp16",
        "int8",
    }
    assert {row["dispatch_key"] for request in requests for row in request["rows"]} == {"tvm_auto"}
    assert initial_fit_input_counts == [176]
    assert online_fit_input_counts == [180, 184, 188]
    assert state.local_state_path == local.local_output_root / "state.json"
    stored_state = json.loads(state.local_state_path.read_text(encoding="utf-8"))
    assert stored_state == {
        "schema_version": "p6_h800_coptv2x_local_state_v2",
        "status": "completed",
        "code_revision": "abc123",
        "completed_rounds": 4,
        "measured_candidate_count": 16,
        "failure_code": None,
    }
    serialized_state = json.dumps(stored_state, sort_keys=True)
    assert str(tmp_path) not in serialized_state
    assert not any(row_id in serialized_state for row_id in selected)


@pytest.mark.parametrize(
    "terminal_status", ["feasibility_failure", "numerical_feasibility_failure"]
)
def test_run_p6_accepts_true_candidate_failures_as_budget_consuming_feedback(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    """Catches treating valid candidate failures as an invalid or free batch."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        feedback = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": [
                {
                    "row_id": row["row_id"],
                    "terminal_status": terminal_status,
                    "failure_reason": "synthetic_feasibility",
                }
                for row in request["rows"]
            ],
        }
        Path(argv[3]).write_text(json.dumps(feedback), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert state.measured_candidate_count == 16


def test_run_p6_excludes_failure_only_graph_features_from_online_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches feasibility-only graph fields changing the online model schema."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    real_select = execution.select_task_batch
    real_online_fit = execution.fit_online_bundle
    online_bundles: list[Any] = []
    online_fit_rows: list[list[dict[str, Any]]] = []
    selection_call_count = 0

    def select_failure_candidate(
        predicted_rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal selection_call_count
        selection = real_select(predicted_rows, *args, **kwargs)
        failure_candidate = next(
            (row for row in predicted_rows if "failure_only_feature" in row["graph_features"]),
            None,
        )
        selected_by_id: dict[str, Mapping[str, Any]] = {}
        preferred = [failure_candidate] if selection_call_count == 0 and failure_candidate else []
        for row in [*preferred, *selection["selected_rows"], *predicted_rows]:
            if "failure_only_feature" in row["graph_features"] and row not in preferred:
                continue
            selected_by_id.setdefault(str(row["row_id"]), row)
        selected = list(selected_by_id.values())[:4]
        selection_call_count += 1
        return {
            **selection,
            "selected_row_count": len(selected),
            "selected_row_ids": [row["row_id"] for row in selected],
            "selected_rows": selected,
        }

    def record_online_fit(rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        online_fit_rows.append([dict(row) for row in rows])
        bundle = real_online_fit(rows, *args, **kwargs)
        online_bundles.append(bundle)
        return bundle

    monkeypatch.setattr(execution, "select_task_batch", select_failure_candidate)
    monkeypatch.setattr(execution, "fit_online_bundle", record_online_fit)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            registry_path = Path(argv[3])
            _write_source_registry(registry_path, count=343)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["groups"][0]["graph_features"] = {
                **registry["groups"][0]["graph_features"],
                "failure_only_feature": 1.0,
            }
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        rows = []
        for row in request["rows"]:
            if "failure_only_feature" in row["graph_features"]:
                rows.append(
                    {
                        "row_id": row["row_id"],
                        "terminal_status": "feasibility_failure",
                        "failure_reason": "synthetic_feasibility",
                    }
                )
            else:
                rows.append(
                    {
                        "row_id": row["row_id"],
                        "terminal_status": "measured_success_gold",
                        "latency_ms": 3.0,
                        "energy_j": 0.8,
                        "ap30": 0.91,
                        "ap50": 0.82,
                        "ap70": 0.73,
                    }
                )
        Path(argv[3]).write_text(
            json.dumps(
                {
                    "schema_version": "p6_h800_coptv2x_feedback_v2",
                    "measurement_request_sha256": request["measurement_request_sha256"],
                    "rows": rows,
                }
            ),
            encoding="utf-8",
        )
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert state.measured_candidate_count == 16
    assert [bundle.manifest["input_row_count"] for bundle in online_bundles] == [179, 183, 187]
    assert [bundle.manifest["value_training_row_count"] for bundle in online_bundles] == [
        179,
        183,
        187,
    ]
    assert all(
        "graph:failure_only_feature" not in bundle.feature_names for bundle in online_bundles
    )
    assert all(
        "failure_only_feature" not in row.get("graph_features", {})
        for rows in online_fit_rows
        for row in rows
    )


def test_release_feedback_rows_detaches_nested_request_identity_context() -> None:
    """Catches later request mutation changing released training rows."""
    request = _minimal_request()
    request["rows"][0].update(
        {
            "graph_features": {"group_id": "row-0", "stable_feature": 1.0},
            "source_contract": {"group_id": "row-0", "artifact_id": "fixture-row-0"},
        }
    )

    released = execution._release_feedback_rows(_minimal_feedback(), request, task=_minimal_task())
    request["rows"][0]["graph_features"]["stable_feature"] = 9.0
    request["rows"][0]["source_contract"]["artifact_id"] = "mutated"

    assert released[0]["graph_features"] == {"group_id": "row-0", "stable_feature": 1.0}
    assert released[0]["source_contract"] == {
        "group_id": "row-0",
        "artifact_id": "fixture-row-0",
    }


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_row", "feedback_candidate_mismatch"),
        ("extra_row", "feedback_candidate_mismatch"),
        ("wrong_request_sha", "feedback_request_mismatch"),
        ("wrong_row_id", "feedback_candidate_mismatch"),
        ("missing_metric", "feedback_metrics_invalid"),
        ("nan_metric", "feedback_metrics_invalid"),
        ("public_runner_failure", "feedback_terminal_status_invalid"),
    ],
)
def test_run_p6_quarantines_invalid_feedback_batches(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    """Catches partial release or budget advancement from malformed feedback."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        rows = [
            {
                "row_id": row["row_id"],
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for row in request["rows"]
        ]
        if mutation == "missing_row":
            rows.pop()
        elif mutation == "extra_row":
            rows.append(dict(rows[0]))
        elif mutation == "wrong_row_id":
            rows[0]["row_id"] = "wrong"
        elif mutation == "missing_metric":
            del rows[0]["ap50"]
        elif mutation == "nan_metric":
            rows[0]["latency_ms"] = float("nan")
        elif mutation == "public_runner_failure":
            rows[0] = {"row_id": rows[0]["row_id"], "terminal_status": "public_runner_failure"}
        payload = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": (
                "wrong"
                if mutation == "wrong_request_sha"
                else request["measurement_request_sha256"]
            ),
            "rows": rows,
        }
        Path(argv[3]).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "failed"
    assert state.failure_code == expected_code
    assert state.completed_rounds == 0
    assert state.measured_candidate_count == 0
    failure = json.loads((local.local_output_root / "round-00" / "failure.json").read_text())
    assert failure == {
        "schema_version": "p6_h800_coptv2x_failure_v2",
        "failure_code": expected_code,
        "completed_rounds": 0,
    }


def test_run_p6_static_p61_rejects_non_343_v1_registry(tmp_path: Path) -> None:
    """P6.1 retains its fixed 343-group registry gate."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] != "local_build_registry.py":
            raise AssertionError("measurement must not start for an incomplete space")
        _write_source_registry(Path(argv[3]), count=342)
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == "source_registry_invalid"


@pytest.mark.parametrize(
    ("registry_payload", "expected_code"),
    [(None, "source_registry_missing"), ([], "source_registry_invalid")],
)
def test_run_p6_reports_stable_source_registry_failures(
    tmp_path: Path, registry_payload: object | None, expected_code: str
) -> None:
    """Catches source output errors escaping with unstable local exceptions."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] != "local_build_registry.py":
            raise AssertionError("measurement must not start after source failure")
        if registry_payload is not None:
            Path(argv[3]).write_text(json.dumps(registry_payload), encoding="utf-8")
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == expected_code


def test_run_p6_reports_invalid_local_input_with_a_stable_code(tmp_path: Path) -> None:
    """Catches unreadable controller inputs being mislabeled as contract failures."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    local.local_input_paths["gold176_rows"].unlink()

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(
            contract,
            local,
            "abc123",
            lambda argv, cwd: (_ for _ in ()).throw(AssertionError((argv, cwd))),
        )

    assert captured.value.failure_code == "local_input_invalid"


def test_run_p6_never_selects_a_gold176_source_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    overlap_group = "pyramid|16x32x64"
    gold_rows_path = local.local_input_paths["gold176_rows"]
    gold_graphs_path = local.local_input_paths["gold176_graph_features"]
    gold_rows = json.loads(gold_rows_path.read_text(encoding="utf-8"))
    gold_graphs = json.loads(gold_graphs_path.read_text(encoding="utf-8"))
    gold_rows[0] = {**gold_rows[0], "group_id": overlap_group, "width": [16, 32, 64]}
    gold_graphs[0] = {**gold_graphs[0], "group_id": overlap_group, "width": [16, 32, 64]}
    gold_rows_path.write_text(json.dumps(gold_rows), encoding="utf-8")
    gold_graphs_path.write_text(json.dumps(gold_graphs), encoding="utf-8")
    selected_groups: list[str] = []
    real_select = execution.select_task_batch

    def select_without_gold(predicted_rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any):
        assert all(str(row["group_id"]) != overlap_group for row in predicted_rows)
        return real_select(predicted_rows, *args, **kwargs)

    monkeypatch.setattr(execution, "select_task_batch", select_without_gold)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        request_path = Path(argv[2])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        selected_groups.extend(str(row["group_id"]) for row in request["rows"])
        _write_feedback_from_request(request_path, Path(argv[3]))
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert overlap_group not in selected_groups


@pytest.mark.parametrize(
    "mutation",
    [
        lambda closure: closure.update(stage4_closed=False),
        lambda closure: closure.update(stage5_search_ready=False),
        lambda closure: closure["training_source_rows"].update(initial_coldstart=175),
        lambda closure: closure["canonical_value_heads"].update(ap70="legacy-head"),
        lambda closure: closure.update(uncertainty_policy="legacy-policy"),
        lambda closure: closure.update(selected_acquisition_policy="legacy-policy"),
    ],
)
def test_run_p6_rejects_unclosed_or_drifted_stage4_contract(tmp_path: Path, mutation: Any) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    closure_path = local.local_input_paths["closure"]
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    mutation(closure)
    closure_path.write_text(json.dumps(closure), encoding="utf-8")

    with pytest.raises(P6CoptV2XContractError, match="closure"):
        run_p6_coptv2x_search(
            contract,
            local,
            "abc123",
            lambda argv, cwd: (_ for _ in ()).throw(AssertionError((argv, cwd))),
        )


def test_run_p6_does_not_reuse_feedback_from_an_earlier_run(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)

    def first_runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        else:
            _write_feedback_from_request(Path(argv[2]), Path(argv[3]))
        return 0

    assert run_p6_coptv2x_search(contract, local, "abc123", first_runner).status == "completed"

    def stale_runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", stale_runner)

    assert state.status == "failed"
    assert state.failure_code == "feedback_missing"
    assert state.completed_rounds == 0
    assert state.measured_candidate_count == 0


@pytest.mark.parametrize(
    "metric,value",
    [
        ("latency_ms", 0.0),
        ("energy_j", -0.1),
        ("ap30", -0.01),
        ("ap50", 1.01),
        ("ap70", float("nan")),
    ],
)
def test_release_feedback_rejects_nonphysical_metrics(metric: str, value: float) -> None:
    feedback = _minimal_feedback()
    feedback["rows"][0][metric] = value

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        execution._release_feedback_rows(feedback, _minimal_request(), task=_minimal_task())

    assert captured.value.failure_code == "feedback_metrics_invalid"


def test_run_step_redacts_runner_exceptions() -> None:
    step = LocalExecutionStep("measure_batch", ("python", "/private/adapter.py"))

    def raising_runner(argv: tuple[str, ...], cwd: Path) -> int:
        raise RuntimeError(f"secret failure: {argv!r} from {cwd}")

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        execution._run_step(
            step,
            {},
            cwd=Path("/private/output"),
            runner=raising_runner,
        )

    assert str(captured.value) == "local command failed"
    assert "/private" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "/private" not in repr(captured.value.__context__)


def test_run_p6_rejects_symlinked_round_output_without_touching_victim(
    tmp_path: Path,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    local.local_output_root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    victim_feedback = victim / "feedback.json"
    victim_feedback.write_text("retain", encoding="utf-8")
    (local.local_output_root / "round-00").symlink_to(victim, target_is_directory=True)
    measurement_started = False

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_started
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        else:
            measurement_started = True
        return 0

    with pytest.raises(P6CoptV2XExecutionError, match="output"):
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert measurement_started is False
    assert victim_feedback.read_text(encoding="utf-8") == "retain"
    assert not (victim / "measurement_request.json").exists()


def test_run_p6_projection_failure_is_atomic_before_measurement_request_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches writing or measuring the original request after projection fails."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    measurement_started = False

    def fail_projection(request: Mapping[str, Any]) -> None:
        del request
        raise P6HistorySourceMaterializationError()

    monkeypatch.setattr(
        execution,
        "project_source_materialization_request",
        fail_projection,
        raising=False,
    )

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_started
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        else:
            measurement_started = True
            _write_feedback_from_request(Path(argv[2]), Path(argv[3]))
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "failed"
    assert state.failure_code == "history_execution_invalid"
    assert measurement_started is False
    assert not (local.local_output_root / "round-00" / "measurement_request.json").exists()


def test_round_request_write_uses_projected_training_hash_only(tmp_path: Path) -> None:
    """Catches persisting or measuring a provisional unprojected request hash."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _framework_local_config_with_stage1_step(tmp_path)
    captured_requests: list[dict[str, Any]] = []

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[0] == "fake-stage1":
            _write_real_stage1_manifest(Path(argv[1]))
        elif argv[0] == "fake-registry":
            plan = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
            _write_recipe_v2_training_registry_from_plan(Path(argv[2]), plan)
        else:
            request_path = Path(argv[1])
            captured_requests.append(json.loads(request_path.read_text(encoding="utf-8")))
            _write_feedback_from_request(request_path, Path(argv[2]))
        return 0

    state = run_p6_coptv2x_search(contract, local, "rev-projected-training", runner)

    assert state.completed_rounds == 4
    assert len(captured_requests) == 4
    for request in captured_requests:
        first_row = request["rows"][0]
        assert (
            first_row["source_contract"]["external_training_binding"]["training_required"] is True
        )
        assert "shared_source_paths" not in first_row["source_contract"]
        assert (
            request["row_sha256"][first_row["row_id"]]
            == hashlib.sha256(
                json.dumps(
                    first_row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        body = {key: value for key, value in request.items() if key != "measurement_request_sha256"}
        assert (
            request["measurement_request_sha256"]
            == hashlib.sha256(
                json.dumps(
                    body,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )


def test_run_p6_rejects_symlinked_source_registry_before_adapter(
    tmp_path: Path,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    local.local_output_root.mkdir()
    victim = tmp_path / "victim-registry.json"
    victim.write_text("retain", encoding="utf-8")
    (local.local_output_root / "source_registry.json").symlink_to(victim)
    adapter_started = False

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal adapter_started
        del cwd
        adapter_started = True
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        return 1

    with pytest.raises(P6CoptV2XExecutionError, match="output|command"):
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert adapter_started is False
    assert victim.read_text(encoding="utf-8") == "retain"


def test_run_p6_rejects_symlinked_state_without_touching_victim(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    local.local_output_root.mkdir()
    victim = tmp_path / "victim-state.json"
    victim.write_text("retain", encoding="utf-8")
    (local.local_output_root / "state.json").symlink_to(victim)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        else:
            _write_feedback_from_request(Path(argv[2]), Path(argv[3]))
        return 0

    with pytest.raises(P6CoptV2XExecutionError, match="output"):
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert victim.read_text(encoding="utf-8") == "retain"


def test_run_p6_revalidates_feedback_leaf_after_adapter(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    captured_request: dict[str, Any] | None = None

    def capture_runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal captured_request
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        captured_request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        return 1

    failed = run_p6_coptv2x_search(contract, local, "abc123", capture_runner)
    assert failed.status == "failed"
    assert failed.failure_code == "command_failed"
    assert captured_request is not None
    victim = tmp_path / "external-feedback.json"
    victim_payload = {
        "schema_version": "p6_h800_coptv2x_feedback_v2",
        "measurement_request_sha256": captured_request["measurement_request_sha256"],
        "rows": [
            {
                "row_id": row["row_id"],
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for row in captured_request["rows"]
        ],
    }
    victim_text = json.dumps(victim_payload, sort_keys=True)
    victim.write_text(victim_text, encoding="utf-8")
    measurement_calls = 0

    def symlink_runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_calls
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        measurement_calls += 1
        if measurement_calls == 1:
            Path(argv[3]).symlink_to(victim)
            return 0
        return 1

    with pytest.raises(P6CoptV2XExecutionError, match="output") as captured:
        run_p6_coptv2x_search(contract, local, "abc123", symlink_runner)

    assert captured.value.failure_code == "unsafe_output"
    assert measurement_calls == 1
    assert victim.read_text(encoding="utf-8") == victim_text


def test_release_feedback_rejects_duplicate_or_extra_rows() -> None:
    feedback = _minimal_feedback()
    feedback["rows"].append(dict(feedback["rows"][0]))

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        execution._release_feedback_rows(feedback, _minimal_request(), task=_minimal_task())

    assert captured.value.failure_code == "feedback_candidate_mismatch"


def test_load_public_contract_requires_fixed_pyramid_h800_tvm_budget(tmp_path: Path) -> None:
    invalid_contracts = [
        _public_contract(target="orin"),
        _public_contract(target_model="codriving"),
        _public_contract(execution_backend="unsupported_backend"),
        _public_contract(sample_budget=15),
        _public_contract(batch_size=2),
        _public_contract(round_count=5),
    ]

    for index, payload in enumerate(invalid_contracts):
        with pytest.raises(P6CoptV2XContractError):
            load_public_contract(_write_yaml(tmp_path / f"contract-{index}.yaml", payload))


@pytest.mark.parametrize(
    "forbidden_key",
    ["path", "argv", "host", "candidate_id", "raw_logs", "checkpoint", "sha256", "hash"],
)
def test_load_public_contract_rejects_private_execution_keys(
    tmp_path: Path, forbidden_key: str
) -> None:
    payload = _public_contract()
    payload[forbidden_key] = "private-value"

    with pytest.raises(P6CoptV2XContractError, match="forbidden"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_id", "/private/run"),
        ("configuration_label", "h800-host-01"),
        ("configuration_label", "hostname-01"),
        ("configuration_label", "raw-logs-2026"),
        ("configuration_label", "candidate-42"),
        ("configuration_label", "candidate-id-42"),
        ("target_model", "python train.py"),
    ],
)
def test_load_public_contract_rejects_private_or_command_style_public_values(
    tmp_path: Path, field: str, value: str
) -> None:
    with pytest.raises(P6CoptV2XContractError, match="restricted"):
        load_public_contract(
            _write_yaml(tmp_path / "contract.yaml", _public_contract(**{field: value}))
        )


@pytest.mark.parametrize(
    "asset",
    [
        {"label": "checkpoints", "version": "v1", "license_status": "cleared"},
        {"label": "training-data", "version": "sha256:abc123", "license_status": "cleared"},
        {"label": "training-data", "version": "hash:abc123", "license_status": "cleared"},
        {"label": "training-data", "version": "a" * 64, "license_status": "cleared"},
    ],
)
def test_load_public_contract_rejects_restricted_asset_values(
    tmp_path: Path, asset: dict[str, str]
) -> None:
    with pytest.raises(P6CoptV2XContractError, match="restricted"):
        load_public_contract(
            _write_yaml(tmp_path / "contract.yaml", _public_contract(assets=[asset]))
        )


def test_load_public_contract_accepts_the_public_example() -> None:
    contract = load_public_contract(
        REPOSITORY_ROOT / "configs/execution/p6_h800_search.example.yaml"
    )

    assert isinstance(contract, PublicP6CoptV2XContract)
    assert contract.target == "h800"
    assert contract.target_model == "pyramid"
    assert contract.execution_backend == "tvm_auto"
    assert (contract.round_count, contract.batch_size, contract.sample_budget) == (4, 4, 16)
    assert contract.metric_names == ("latency_ms", "energy_j", "ap30", "ap50", "ap70")


@pytest.mark.parametrize("field", ["legacy_round_limit", "legacy_report", "legacy_output"])
def test_load_public_contract_rejects_unknown_round_and_output_keys(
    tmp_path: Path, field: str
) -> None:
    payload = _public_contract()
    payload[field] = "legacy"

    with pytest.raises(P6CoptV2XContractError, match="unknown|forbidden"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", payload))


def test_load_local_config_requires_source_and_measurement_steps(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()

    loaded = load_local_config(
        _write_yaml(tmp_path / "local.yaml", _local_config(tmp_path)), contract
    )

    assert isinstance(loaded, LocalP6CoptV2XConfig)
    assert loaded.source_registry_step.name == "build_source_registry"
    assert loaded.measurement_step.name == "measure_batch"
    assert loaded.local_output_root == tmp_path / "private-output"


def test_load_local_config_rejects_relative_asset_and_output_paths(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    relative_root_payload = _local_config(tmp_path, local_output_root="relative")
    relative_asset_payload = _local_config(tmp_path)
    relative_asset_payload["asset_paths"]["training-data"] = "relative"

    with pytest.raises(P6CoptV2XContractError, match="absolute path"):
        load_local_config(
            _write_yaml(tmp_path / "relative-root.yaml", relative_root_payload), contract
        )
    with pytest.raises(P6CoptV2XContractError, match="absolute path"):
        load_local_config(
            _write_yaml(tmp_path / "relative-asset.yaml", relative_asset_payload), contract
        )


@pytest.mark.parametrize(
    "key",
    ["candidate_registry", "measurements", "result_path_template", "steps", "result_step"],
)
def test_load_local_config_rejects_legacy_stage5_input_and_result_keys(
    tmp_path: Path, key: str
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    payload = _local_config(tmp_path)
    payload[key] = "legacy"

    with pytest.raises(P6CoptV2XContractError, match="unknown|forbidden"):
        load_local_config(_write_yaml(tmp_path / "local.yaml", payload), contract)


def test_load_local_config_rejects_shell_or_unknown_template_tokens(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    shell_payload = _local_config(
        tmp_path,
        measurement_step={
            "name": "measure_batch",
            "argv": ["bash", "-c", "private", "{measurement_request}"],
        },
    )
    token_payload = _local_config(
        tmp_path,
        measurement_step={
            "name": "measure_batch",
            "argv": ["python", "local_measure.py", "{candidate_id}"],
        },
    )

    with pytest.raises(P6CoptV2XContractError, match="shell executable"):
        load_local_config(_write_yaml(tmp_path / "shell.yaml", shell_payload), contract)
    with pytest.raises(P6CoptV2XContractError, match="template"):
        load_local_config(_write_yaml(tmp_path / "token.yaml", token_payload), contract)
