from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import pytest

from framework.stage5.production_search_v1 import validate_source_contract
from framework.stage6.p6_history_registry_v1 import (
    P6HistoryRegistryError,
    materialize_history_registry,
)
from framework.stage6.p6_history_training_contract_v1 import (
    REQUIRED_TRAINING_PARAMETER_KEYS,
)
import framework.stage6.p6_history_registry_v1 as registry_module


SYNTHETIC_GPU_INDICES = (101, 103, 107)


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan(q_modes: tuple[str, ...], *, duplicate: bool = False) -> dict[str, Any]:
    widths = ([17, 31, 63], [23, 47, 95])
    candidates = [
        {
            "width": list(width),
            "q_mode": q_mode,
            "source_point_ids": [
                f"stage{stage}-{q_mode}-w{value}" for stage, value in enumerate(width, start=1)
            ],
        }
        for width in widths
        for q_mode in q_modes
    ]
    if duplicate:
        candidates.append(copy.deepcopy(candidates[0]))
    candidates.sort(
        key=lambda row: (
            tuple(row["width"]),
            row["q_mode"],
            tuple(row["source_point_ids"]),
        )
    )
    return {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "source_schema": "stage2_search_space_v1",
        "target_model": "pyramid",
        "hardware_target": "h800",
        "execution_backend": "tvm_auto",
        "candidate_source_mode": "framework_stage2_search_space",
        "structure_count": len({tuple(row["width"]) for row in candidates}),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _recipe() -> dict[str, Any]:
    output_templates = {
        q_mode: {
            name: (
                f"materialized/{{group_id}}/{{q_mode}}/{name.removesuffix('_path_template')}.json"
            )
            for name in (
                "training_path_template",
                "checkpoint_path_template",
                "onnx_path_template",
                "calibration_path_template",
            )
        }
        for q_mode in ("fp16", "int8")
    }
    return {
        "schema_version": "p6_history_dynamic_materialization_recipe_v1",
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
        "group_id_template": ("pyramid|{stage1_width}x{stage2_width}x{stage3_width}"),
        "artifact_id_template": ("pyramid-{stage1_width}-{stage2_width}-{stage3_width}"),
        "output_path_templates_by_q_mode": output_templates,
    }


def _recipe_v2() -> dict[str, Any]:
    return {
        "schema_version": "p6_history_dynamic_materialization_recipe_v2",
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
        "group_id_template": ("pyramid|{stage1_width}x{stage2_width}x{stage3_width}"),
        "artifact_id_template": ("pyramid-{stage1_width}-{stage2_width}-{stage3_width}"),
        "shared_source_path_templates": {
            "checkpoint_path": "materialized/{artifact_id}/checkpoint/model.ckpt",
            "checkpoint_dir": "materialized/{artifact_id}/checkpoint",
            "config_path": "materialized/{artifact_id}/config/source-config.json",
            "training_done_marker": "materialized/{artifact_id}/markers/training.done",
            "onnx_path": "materialized/{artifact_id}/onnx/model.onnx",
            "onnx_report_path": "materialized/{artifact_id}/onnx/report.json",
            "calibration_root": "materialized/{artifact_id}/calibration",
            "calibration_npz": "materialized/{artifact_id}/calibration/cache.npz",
            "calibration_summary": "materialized/{artifact_id}/calibration/summary.json",
            "trt_calibration_dir": "materialized/{artifact_id}/trt-calibration",
            "source_done_marker": "materialized/{artifact_id}/markers/source.done",
        },
    }


def _write_executable(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic private executable\n", encoding="utf-8")
    path.chmod(0o700)
    return str(path)


def _execution_binding_fields(private_root: Path) -> dict[str, Any]:
    chain_root = private_root / "documented-stage5-chain"
    component_names = {
        "controller": "stage5_task_round_controller_v3.sh",
        "source_materializer": "stage5_materialize_round_sources_v1.sh",
        "performance_plan": "stage5_build_performance_plan_v2.py",
        "finalizer": "stage5_finalize_feedback_v2.py",
    }
    components = {
        role: _write_executable(chain_root / name) for role, name in component_names.items()
    }
    private_bin = private_root / "private-runner" / "bin"
    quantize = _write_executable(private_bin / "quantize-private")
    measure_ap = _write_executable(private_bin / "measure-ap-private")
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


def _binding(tmp_path: Path) -> dict[str, Any]:
    private_root = tmp_path / "synthetic-history"
    private_root.mkdir(exist_ok=True)
    evidence_sha = hashlib.sha256(b"synthetic-history-evidence").hexdigest()
    return {
        "schema_version": "p6_history_binding_v1",
        "target": {
            "model": "pyramid",
            "hardware": "h800",
            "backend": "tvm_auto",
        },
        "private_root": str(private_root),
        **_execution_binding_fields(private_root),
        "source_contract_template": {
            "schema_version": "stage5_source_contract_v1",
            "group_id": "pyramid|16x32x64",
            "model": "pyramid",
            "width": [16, 32, 64],
            "artifact_id": "synthetic-legacy-template",
            "source_status": "ready",
            "source_evidence_sha256": evidence_sha,
            "materialization_scope": "synthetic_fixture",
            "dynamic_materialization_recipe": _recipe(),
        },
        "status": "validated",
    }


def _private_root_with_training_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic-history"
    root.mkdir(parents=True, exist_ok=True)
    operator = tmp_path / "operator-assets"
    (operator / "checkpoints").mkdir(parents=True, exist_ok=True)
    (operator / "datasets" / "coptv2x").mkdir(parents=True, exist_ok=True)
    (operator / "configs").mkdir(parents=True, exist_ok=True)
    (operator / "checkpoints" / "base.ckpt").write_text("base\n", encoding="utf-8")
    (operator / "configs" / "pyramid.py").write_text(
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )
    return root


def _binding_with_recipe_v2_training_template(private_root: Path) -> dict[str, Any]:
    binding = _binding(private_root.parent)
    template = binding["source_contract_template"]
    operator = private_root.parent / "operator-assets"
    template.update(
        {
            "external_training_binding": {
                "schema_version": "p6_external_training_binding_v1",
                "training_required": True,
                "training_source_kind": "selected_candidate_finetune",
                "base_checkpoint_path": str(operator / "checkpoints" / "base.ckpt"),
                "base_checkpoint_sha256": hashlib.sha256(b"base\n").hexdigest(),
                "dataset_root": str(operator / "datasets" / "coptv2x"),
                "pyramid_config_path": str(operator / "configs" / "pyramid.py"),
                "pyramid_config_sha256": hashlib.sha256(
                    (operator / "configs" / "pyramid.py").read_bytes()
                ).hexdigest(),
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
            "dynamic_materialization_recipe": _recipe_v2(),
        }
    )
    return binding


@pytest.mark.parametrize("tampering", ["missing", "environment_field"])
def test_registry_rejects_unverified_execution_interface_before_write(
    tmp_path: Path, tampering: str
) -> None:
    binding = _binding(tmp_path)
    if tampering == "missing":
        binding.pop("execution_interface")
    else:
        binding["execution_interface"]["environment"]["values"].pop("CUDA_VISIBLE_DEVICES")
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    registry_path = local_output_root / "source_registry.json"

    with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
        materialize_history_registry(_plan(("fp16",)), binding, local_output_root)

    assert not registry_path.exists()


def _plan_identity_map(
    plan: Mapping[str, Any],
) -> dict[tuple[tuple[int, ...], str], tuple[str, ...]]:
    return {
        (tuple(row["width"]), row["q_mode"]): tuple(row["source_point_ids"])
        for row in plan["candidates"]
    }


def _registry_identity_map(
    registry: Mapping[str, Any],
) -> dict[tuple[tuple[int, ...], str], tuple[str, ...]]:
    return {
        (tuple(group["width"]), q_mode): tuple(group["source_point_ids_by_q_mode"][q_mode])
        for group in registry["groups"]
        for q_mode in group["available_q_modes"]
    }


@pytest.mark.parametrize("q_modes", [("fp16",), ("int8",), ("fp16", "int8")])
def test_registry_materializes_every_dynamic_identity_without_pruning(
    tmp_path: Path, q_modes: tuple[str, ...]
) -> None:
    plan = _plan(q_modes)
    binding = _binding(tmp_path)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    original_plan = copy.deepcopy(plan)
    original_binding = copy.deepcopy(binding)

    registry = materialize_history_registry(plan, binding, local_output_root)

    assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
    assert _registry_identity_map(registry) == _plan_identity_map(plan)
    assert (
        sum(len(group["available_q_modes"]) for group in registry["groups"])
        == plan["candidate_count"]
    )
    assert all("source_contract" in group for group in registry["groups"])
    assert all(validate_source_contract(group) for group in registry["groups"])
    assert json.loads((local_output_root / "source_registry.json").read_text()) == registry
    assert plan == original_plan
    assert binding == original_binding


def test_registry_binds_all_authoritative_output_paths_beneath_local_root(
    tmp_path: Path,
) -> None:
    plan = _plan(("fp16", "int8"))
    binding = _binding(tmp_path)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()

    registry = materialize_history_registry(plan, binding, local_output_root)

    rendered_paths: set[str] = set()
    for group in registry["groups"]:
        contract = group["source_contract"]
        assert "dynamic_materialization_recipe" not in contract
        assert contract["stage_widths"] == {
            "stage1_width": group["width"][0],
            "stage2_width": group["width"][1],
            "stage3_width": group["width"][2],
        }
        outputs_by_q_mode = contract["materialization_outputs_by_q_mode"]
        assert set(outputs_by_q_mode) == set(group["available_q_modes"])
        for outputs in outputs_by_q_mode.values():
            assert set(outputs) == {
                "training_path",
                "checkpoint_path",
                "onnx_path",
                "calibration_path",
            }
            for value in outputs.values():
                resolved = Path(value)
                assert resolved.is_relative_to(local_output_root.resolve())
                assert value not in rendered_paths
                rendered_paths.add(value)


@pytest.mark.parametrize("q_modes", [("fp16",), ("int8",), ("fp16", "int8")])
def test_registry_v2_materializes_one_shared_bundle_per_group(
    tmp_path: Path, q_modes: tuple[str, ...]
) -> None:
    """Catches q-mode-specific source bundles or dropped private static fields."""
    private_root = _private_root_with_training_inputs(tmp_path)
    binding = _binding_with_recipe_v2_training_template(private_root)
    binding["source_contract_template"].update(
        {
            **{
                key: str(private_root / "legacy-flat" / key)
                for key in (
                    "training_path",
                    "checkpoint_path",
                    "onnx_path",
                    "calibration_path",
                )
            },
            "materialization_outputs_by_q_mode": {
                "fp16": {
                    key: str(private_root / "legacy-per-q" / key)
                    for key in (
                        "training_path",
                        "checkpoint_path",
                        "onnx_path",
                        "calibration_path",
                    )
                }
            },
            "shared_source_paths": {
                key: str(private_root / "untrusted-self-report" / key)
                for key in _recipe_v2()["shared_source_path_templates"]
            },
        }
    )
    original_binding = copy.deepcopy(binding)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()

    registry = materialize_history_registry(_plan(q_modes), binding, local_output_root)

    observed_paths: set[str] = set()
    for group in registry["groups"]:
        contract = group["source_contract"]
        shared_paths = contract["shared_source_paths"]
        assert set(shared_paths) == set(_recipe_v2()["shared_source_path_templates"])
        assert all("q_mode" not in value for value in shared_paths.values())
        assert all(
            Path(value).is_relative_to(local_output_root.resolve())
            for value in shared_paths.values()
        )
        assert not observed_paths.intersection(shared_paths.values())
        observed_paths.update(shared_paths.values())
        external = contract["external_training_binding"]
        assert external["base_checkpoint_path"].endswith("/checkpoints/base.ckpt")
        assert external["dataset_root"].endswith("/datasets/coptv2x")
        assert external["training_parameters"]["epochs"] == 2
        assert "untrusted-self-report" not in json.dumps(shared_paths)
        assert "materialization_outputs_by_q_mode" not in contract
        assert not {
            "training_path",
            "checkpoint_path",
            "onnx_path",
            "calibration_path",
        }.intersection(contract)
        assert "dynamic_materialization_recipe" not in contract
        assert group["available_q_modes"] == sorted(q_modes)

    assert binding == original_binding


def test_recipe_v2_registry_preserves_training_fields_and_rehashes_contract(
    tmp_path: Path,
) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    binding = _binding_with_recipe_v2_training_template(private_root)
    local_output_root = tmp_path / "ignored-output"
    local_output_root.mkdir()
    registry = materialize_history_registry(_plan(("fp16", "int8")), binding, local_output_root)

    first = registry["groups"][0]["source_contract"]
    external = first["external_training_binding"]
    assert external["training_required"] is True
    assert external["training_source_kind"] == "selected_candidate_finetune"
    assert set(external["training_parameters"]) == set(REQUIRED_TRAINING_PARAMETER_KEYS)
    operator = private_root.parent / "operator-assets"
    assert external["base_checkpoint_path"].startswith(str(operator))
    assert external["dataset_root"].startswith(str(operator))
    assert external["pyramid_config_path"].startswith(str(operator))
    assert registry["groups"][0]["source_contract_sha256"] == _canonical_sha(first)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_training_required",
        "false_training_required",
        "checkpoint_already_exists",
        "missing_pyramid_config_path",
        "static_path_overlaps_code_root",
        "shared_output_outside_root",
        "shared_output_collision",
    ],
)
def test_recipe_v2_registry_rejects_unsafe_training_contract_before_write(
    tmp_path: Path, mutation: str
) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    binding = _binding_with_recipe_v2_training_template(private_root)
    template = binding["source_contract_template"]
    external = template["external_training_binding"]
    if mutation == "missing_training_required":
        external.pop("training_required")
    elif mutation == "false_training_required":
        external["training_required"] = False
    elif mutation == "checkpoint_already_exists":
        external["training_source_kind"] = "checkpoint_already_exists"
    elif mutation == "missing_pyramid_config_path":
        external.pop("pyramid_config_path")
    elif mutation == "static_path_overlaps_code_root":
        overlapping = private_root / "outside.ckpt"
        overlapping.write_text("outside", encoding="utf-8")
        external["base_checkpoint_path"] = str(overlapping)
    elif mutation == "shared_output_outside_root":
        template["dynamic_materialization_recipe"]["shared_source_path_templates"][
            "checkpoint_path"
        ] = "../outside.ckpt"
    else:
        template["dynamic_materialization_recipe"]["shared_source_path_templates"][
            "checkpoint_dir"
        ] = template["dynamic_materialization_recipe"]["shared_source_path_templates"][
            "checkpoint_path"
        ]
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()

    with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
        materialize_history_registry(_plan(("fp16",)), binding, local_output_root)

    assert not (local_output_root / "source_registry.json").exists()


def test_registry_output_contains_no_search_result_or_terminal_leakage(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    registry = materialize_history_registry(
        _plan(("fp16", "int8")), _binding(tmp_path), local_output_root
    )

    forbidden = {"metrics", "objectives", "status", "cache", "result", "terminal"}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            assert not (set(value) & forbidden)
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(registry)


def test_registry_rejects_unignored_repository_destination_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches writing a private source contract into a trackable repository path."""
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    local_output_root = repository / "private-output"
    local_output_root.mkdir()
    registry_path = local_output_root / "source_registry.json"
    monkeypatch.setattr(registry_module, "REPOSITORY_ROOT", repository, raising=False)

    try:
        with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
            materialize_history_registry(
                _plan(("fp16",)), _binding(tmp_path), local_output_root, registry_path
            )
    finally:
        registry_path.unlink(missing_ok=True)

    assert not registry_path.exists()


def test_registry_accepts_ignored_repository_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches rejecting a private registry path that Git explicitly ignores."""
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text("private-output/\n", encoding="utf-8")
    local_output_root = repository / "private-output"
    local_output_root.mkdir()
    registry_path = local_output_root / "source_registry.json"
    monkeypatch.setattr(registry_module, "REPOSITORY_ROOT", repository, raising=False)

    registry = materialize_history_registry(
        _plan(("fp16",)), _binding(tmp_path), local_output_root, registry_path
    )

    assert json.loads(registry_path.read_text(encoding="utf-8")) == registry


def _drop_recipe(binding: dict[str, Any]) -> None:
    binding["source_contract_template"].pop("dynamic_materialization_recipe")


def _drop_one_calibration_mapping(binding: dict[str, Any]) -> None:
    recipe = binding["source_contract_template"]["dynamic_materialization_recipe"]
    recipe["output_path_templates_by_q_mode"]["int8"].pop("calibration_path_template")


def _invalidate_contract(binding: dict[str, Any]) -> None:
    binding["source_contract_template"]["materialization_scope"] = ""


def _escape_output_root(binding: dict[str, Any]) -> None:
    recipe = binding["source_contract_template"]["dynamic_materialization_recipe"]
    recipe["output_path_templates_by_q_mode"]["fp16"]["training_path_template"] = (
        "../escaped-training.json"
    )


def _collide_output_paths(binding: dict[str, Any]) -> None:
    recipe = binding["source_contract_template"]["dynamic_materialization_recipe"]
    recipe["output_path_templates_by_q_mode"]["fp16"] = {
        "training_path_template": "materialized/collision.json",
        "checkpoint_path_template": "materialized/collision.json",
        "onnx_path_template": "materialized/collision.json",
        "calibration_path_template": "materialized/collision.json",
    }


def _add_forbidden_result_context(binding: dict[str, Any]) -> None:
    binding["source_contract_template"]["metrics"] = {"latency_ms": 1.0}


def _add_legacy_path_escape(binding: dict[str, Any]) -> None:
    binding["source_contract_template"]["checkpoint_path"] = str(
        Path(binding["private_root"]).parent / "outside-checkpoint.pt"
    )


def _add_template_sha_mismatch(binding: dict[str, Any]) -> None:
    binding["source_contract_template_sha256"] = "0" * 64


def _invalidate_private_root(binding: dict[str, Any]) -> None:
    binding["private_root"] = None


@pytest.mark.parametrize(
    "mutate_binding",
    [
        _drop_recipe,
        _drop_one_calibration_mapping,
        _invalidate_contract,
        _escape_output_root,
        _collide_output_paths,
        _add_forbidden_result_context,
        _add_legacy_path_escape,
        _add_template_sha_mismatch,
        _invalidate_private_root,
    ],
)
def test_registry_fails_closed_before_write_for_incomplete_or_unsafe_recipe(
    tmp_path: Path, mutate_binding: Callable[[dict[str, Any]], None]
) -> None:
    plan = _plan(("fp16", "int8"))
    binding = _binding(tmp_path)
    mutate_binding(binding)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    registry_path = local_output_root / "source_registry.json"

    with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
        materialize_history_registry(plan, binding, local_output_root)

    assert not registry_path.exists()
    assert plan["candidate_count"] == 4


def test_registry_rejects_duplicate_plan_identity_before_write(tmp_path: Path) -> None:
    plan = _plan(("fp16",), duplicate=True)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()

    with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
        materialize_history_registry(plan, _binding(tmp_path), local_output_root)

    assert not (local_output_root / "source_registry.json").exists()
    assert len(plan["candidates"]) == 3


def test_registry_rejects_source_evidence_sha_inconsistency_before_write(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)
    binding["source_contract_template"]["source_evidence_sha256"] = "0" * 63
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()

    with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
        materialize_history_registry(_plan(("fp16",)), binding, local_output_root)

    assert not (local_output_root / "source_registry.json").exists()
