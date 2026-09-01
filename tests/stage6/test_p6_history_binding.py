from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import pytest

from framework.stage6.hardware_execution_profile_v1 import (
    load_hardware_execution_profile,
)
from framework.stage6.p6_history_binding_v1 import (
    EXPECTED_HISTORY_ENV_KEYS,
    GpuRecord,
    P6HistoryBindingError,
    discover_history_binding,
    public_binding_projection,
    validate_binding_recipe_consistency,
    validate_history_execution_binding,
    write_private_binding_pair,
)
from framework.stage6.p6_history_recipe_profiles_v1 import RECIPE_V2, SHARED_SOURCE_PATH_KEYS
from framework.stage6.p6_history_training_contract_v1 import P6HistoryTrainingContractError


MARKERS = {
    "controller": "stage5_task_round_controller_v3.sh",
    "source_materializer": "stage5_materialize_round_sources_v1.sh",
    "performance_plan": "stage5_build_performance_plan_v2.py",
    "finalizer": "stage5_finalize_feedback_v2.py",
}
LOCAL_INPUT_NAMES = (
    "gold176_rows",
    "gold176_graph_features",
    "capability_profiles",
    "closure",
)


def test_binding_exports_canonical_history_environment_key_order() -> None:
    """Catches wrapper/provisioning consumers defining a second env contract."""
    assert EXPECTED_HISTORY_ENV_KEYS == (
        "CUDA_VISIBLE_DEVICES",
        "P6_HISTORY_RUN_MODE",
        "P6_HISTORY_PRIVATE_ROOT",
        "P6_HISTORY_TASK_STATE",
        "P6_HISTORY_ROUND_OUTPUT_ROOT",
    )


class SequenceProbe:
    def __init__(self, *snapshots: Iterable[GpuRecord]) -> None:
        self._snapshots = tuple(tuple(snapshot) for snapshot in snapshots)
        self._calls = 0
        self.calls: list[tuple[int, ...]] = []

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        self.calls.append(indices)
        position = min(self._calls, len(self._snapshots) - 1)
        self._calls += 1
        return self._snapshots[position]


def _gpu_records(
    *,
    indices: tuple[int, ...] = (17, 19, 23),
    model_name: str = "NVIDIA H800 80GB HBM3",
    occupancy: float = 0.0,
) -> tuple[GpuRecord, ...]:
    return tuple(
        GpuRecord(
            index=index,
            uuid=f"GPU-fixture-{index}",
            model_name=model_name,
            occupancy=occupancy,
        )
        for index in indices
    )


def _probe(
    first: Iterable[GpuRecord] | None = None,
    second: Iterable[GpuRecord] | None = None,
) -> SequenceProbe:
    initial = tuple(first or _gpu_records())
    return SequenceProbe(initial, tuple(second or initial))


def _canonical_json_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_group() -> dict[str, Any]:
    evidence_sha = hashlib.sha256(b"synthetic-pyramid-source").hexdigest()
    contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": "pyramid|16x32x64",
        "model": "pyramid",
        "width": [16, 32, 64],
        "artifact_id": "synthetic-pyramid-template",
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "synthetic_fixture",
    }
    return {
        "group_id": contract["group_id"],
        "model": contract["model"],
        "width": contract["width"],
        "source_status": contract["source_status"],
        "source_evidence_sha256": evidence_sha,
        "source_contract": contract,
        "source_contract_sha256": _canonical_json_sha(contract),
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _history_runner_manifest(root: Path) -> dict[str, Any]:
    private_runner = root / "private-runner"
    commands = private_runner / "bin"
    commands.mkdir(parents=True, exist_ok=True)
    for name in ("quantize-private", "measure-ap-private", "activate-private"):
        command = commands / name
        command.write_text("synthetic private executable\n", encoding="utf-8")
        command.chmod(0o700)
    chain_root = root / "documented-stage5-chain"
    return {
        "schema_version": "p6_history_runner_interface_v1",
        "controller": {"argv": [str(chain_root / MARKERS["controller"])]},
        "execution_chain": [
            {
                "stage": "source_materialization",
                "argv": [
                    str(chain_root / MARKERS["source_materializer"]),
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
                    str(commands / "quantize-private"),
                    "{task_state}",
                    "{round_output_root}",
                ],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "performance",
                "argv": [
                    str(chain_root / MARKERS["performance_plan"]),
                    "{task_state}",
                    "{round_output_root}",
                ],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "ap",
                "argv": [
                    str(commands / "measure-ap-private"),
                    "{task_state}",
                    "{round_output_root}",
                ],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "finalization",
                "argv": [
                    str(chain_root / MARKERS["finalizer"]),
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
                "CUDA_VISIBLE_DEVICES": {"kind": "literal", "value": "17,19,23"},
                "P6_HISTORY_RUN_MODE": {"kind": "literal", "value": "bound"},
                "P6_HISTORY_PRIVATE_ROOT": {
                    "kind": "private_path",
                    "value": str(root),
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
            "activation_argv": [str(commands / "activate-private"), "private-bound"],
        },
        "output_layout": {
            "round_root_template": "private-runs/{round_id}",
            "task_state": {
                "path_template": "private-runs/{round_id}/state/task-state.json",
                "format": "json",
                "rows_key": "rows",
                "row_id_key": "row_id",
                "row_hash_key": "row_sha256",
                "source_evidence_key": "source_evidence_sha256",
                "stage_key": "stage",
                "status_key": "terminal_status",
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
                "path_template": "private-runs/{round_id}/actual-feedback.json",
                "format": "json",
                "rows_key": "rows",
                "row_count": 4,
                "row_id_key": "row_id",
                "row_hash_key": "row_sha256",
                "source_evidence_key": "source_evidence_sha256",
                "status_key": "terminal_status",
                "allowed_terminal_statuses": [
                    "measured_success_gold",
                    "feasibility_failure",
                    "numerical_feasibility_failure",
                ],
                "metric_keys": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
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
    }


def _history_root(tmp_path: Path) -> Path:
    root = tmp_path / "history"
    marker_root = root / "documented-stage5-chain"
    marker_root.mkdir(parents=True)
    for marker in MARKERS.values():
        command = marker_root / marker
        command.write_text("synthetic marker\n", encoding="utf-8")
        command.chmod(0o700)
    _write_json(
        root / "registry" / "candidate_source_registry.json",
        {
            "schema_version": "stage5_candidate_source_registry_v1",
            "groups": [_source_group()],
        },
    )
    for name in LOCAL_INPUT_NAMES:
        _write_json(root / "inputs" / f"{name}.json", {"fixture": name})
    _write_json(
        root / "private-runner" / "p6-history-runner-interface.json",
        _history_runner_manifest(root),
    )
    return root


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, Iterable):
        for child in value:
            yield from _walk_strings(child)


def _expect_category(category: str):
    return pytest.raises(
        P6HistoryBindingError,
        match=rf"^{category}:",
    )


def test_binding_recipe_consistency_uses_canonical_json_equality() -> None:
    expected = {
        "schema_version": "p6_history_dynamic_materialization_recipe_v2",
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
    }
    binding = {
        "source_contract_template": {
            "dynamic_materialization_recipe": {
                "stage_width_fields": [
                    "stage1_width",
                    "stage2_width",
                    "stage3_width",
                ],
                "schema_version": "p6_history_dynamic_materialization_recipe_v2",
            }
        }
    }

    validate_binding_recipe_consistency(binding, expected)
    validate_binding_recipe_consistency({}, None)


def test_binding_recipe_consistency_rejects_canonical_drift() -> None:
    expected = {
        "schema_version": "p6_history_dynamic_materialization_recipe_v2",
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
    }
    binding = copy.deepcopy(
        {
            "source_contract_template": {
                "dynamic_materialization_recipe": expected,
            }
        }
    )
    binding["source_contract_template"]["dynamic_materialization_recipe"][
        "stage_width_fields"
    ].reverse()

    with _expect_category("history_recipe_derivation_invalid"):
        validate_binding_recipe_consistency(binding, expected)


def test_discovers_documented_history_and_returns_no_leak_projection(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)

    binding = discover_history_binding(history_root, _probe())
    projected = public_binding_projection(binding)

    assert binding["schema_version"] == "p6_history_binding_v1"
    assert binding["target"] == {
        "model": "pyramid",
        "hardware": "h800",
        "backend": "tvm_auto",
    }
    assert binding["gpu_policy"]["indices"] == [17, 19, 23]
    assert set(binding["gpu_policy"]["uuid_by_index"]) == {"17", "19", "23"}
    assert binding["source_contract_template"]["schema_version"] == ("stage5_source_contract_v1")
    assert set(binding["local_input_paths"]) == set(LOCAL_INPUT_NAMES)
    interface = binding["execution_interface"]
    assert tuple(step["stage"] for step in interface["execution_chain"]) == (
        "source_materialization",
        "quantization",
        "performance",
        "ap",
        "finalization",
    )
    with pytest.raises(TypeError):
        interface["environment"] = {}  # type: ignore[index]
    assert validate_history_execution_binding(binding) == interface
    assert interface["actual_feedback"]["result"]["row_count"] == 4
    assert interface["actual_feedback"]["result"]["metric_keys"] == (
        "latency_ms",
        "energy_j",
        "ap30",
        "ap50",
        "ap70",
    )
    assert "private_root" not in projected
    assert not {
        "base_checkpoint_path",
        "dataset_root",
        "pyramid_config_path",
        "training_parameters",
        "gpu_policy",
        "execution_interface",
        "source_contract_template",
        "component_paths",
    }.intersection(projected)
    projected_strings = tuple(_walk_strings(projected))
    assert not any(str(history_root) in value for value in projected_strings)
    assert not any("GPU-fixture" in value for value in projected_strings)
    assert not any("private-bound" in value for value in projected_strings)
    assert projected == {
        "schema_version": "p6_history_binding_public_v1",
        "binding_schema_version": "p6_history_binding_v1",
        "hardware_profile": "h800",
        "target": {
            "model": "pyramid",
            "hardware": "h800",
            "backend": "tvm_auto",
        },
        "component_versions": {
            "controller": "v3",
            "source_materializer": "v1",
            "performance_plan": "v2",
            "finalizer": "v2",
        },
        "status": "validated",
    }


def test_binding_discovers_only_ready_recipe_v2_template_and_validates_training(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)
    operator_root = tmp_path / "operator-assets"
    (operator_root / "checkpoints").mkdir(parents=True)
    (operator_root / "datasets" / "coptv2x").mkdir(parents=True)
    (operator_root / "configs").mkdir()
    (operator_root / "checkpoints" / "base.ckpt").write_text("base\n", encoding="utf-8")
    (operator_root / "configs" / "pyramid.py").write_text(
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )
    registry_path = history_root / "registry" / "candidate_source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    ready = registry["groups"][0]
    template = ready["source_contract"]
    template.update(
        {
            "external_training_binding": {
                "schema_version": "p6_external_training_binding_v1",
                "training_required": True,
                "training_source_kind": "selected_candidate_finetune",
                "base_checkpoint_path": str(operator_root / "checkpoints" / "base.ckpt"),
                "base_checkpoint_sha256": hashlib.sha256(b"base\n").hexdigest(),
                "dataset_root": str(operator_root / "datasets" / "coptv2x"),
                "pyramid_config_path": str(operator_root / "configs" / "pyramid.py"),
                "pyramid_config_sha256": hashlib.sha256(
                    (operator_root / "configs" / "pyramid.py").read_bytes()
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
            "dynamic_materialization_recipe": {
                "schema_version": RECIPE_V2,
                "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
                "group_id_template": "pyramid|{stage1_width}x{stage2_width}x{stage3_width}",
                "artifact_id_template": "pyramid-{stage1_width}-{stage2_width}-{stage3_width}",
                "shared_source_path_templates": {
                    key: f"materialized/{{artifact_id}}/{key}" for key in SHARED_SOURCE_PATH_KEYS
                },
            },
        }
    )
    ready["source_contract_sha256"] = _canonical_json_sha(template)
    materializable = copy.deepcopy(ready)
    materializable["source_status"] = "materializable"
    materializable["source_contract"]["source_status"] = "materializable"
    materializable["source_contract"]["external_training_binding"]["training_required"] = False
    materializable["source_contract_sha256"] = _canonical_json_sha(
        materializable["source_contract"]
    )
    registry["groups"] = [materializable, ready]
    _write_json(registry_path, registry)

    binding = discover_history_binding(history_root, _probe())

    external = binding["source_contract_template"]["external_training_binding"]
    assert external["training_required"] is True
    assert external["training_source_kind"] == ("selected_candidate_finetune")


def test_binding_rejects_invalid_recipe_v2_training_template(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    registry_path = history_root / "registry" / "candidate_source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    template = registry["groups"][0]["source_contract"]
    template["dynamic_materialization_recipe"] = {"schema_version": RECIPE_V2}
    registry["groups"][0]["source_contract_sha256"] = _canonical_json_sha(template)
    _write_json(registry_path, registry)

    with pytest.raises(P6HistoryTrainingContractError, match=r"^history_execution_invalid:"):
        discover_history_binding(history_root, _probe())


def test_binding_derives_probe_indices_from_private_cuda_policy(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)
    probe = _probe(_gpu_records(indices=(17, 19, 23)))

    binding = discover_history_binding(history_root, probe)

    assert probe.calls == [(17, 19, 23), (17, 19, 23)]
    assert binding["gpu_policy"]["indices"] == [17, 19, 23]


def test_binding_preserves_private_cuda_policy_existing_order(
    tmp_path: Path,
) -> None:
    """Catches sorting or rejection of the runner-template GPU order."""
    policy_indices = (23, 19, 17)
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"] = ",".join(
        str(index) for index in policy_indices
    )
    _write_json(manifest_path, manifest)
    probe = _probe(_gpu_records(indices=policy_indices))

    binding = discover_history_binding(history_root, probe)

    assert probe.calls == [policy_indices, policy_indices]
    assert binding["gpu_policy"]["indices"] == list(policy_indices)


def test_binding_accepts_two_gpu_policy_and_double_probes_exact_order(
    tmp_path: Path,
) -> None:
    """Freezes two-card admission without sorting or weakening the UUID policy."""
    policy_indices = tuple(record.index for record in reversed(_gpu_records()[:2]))
    policy_csv = ",".join(str(index) for index in policy_indices)
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = manifest["environment"]
    values = environment["values"]
    cuda_policy = values["CUDA_VISIBLE_DEVICES"]
    manifest = {
        **manifest,
        "environment": {
            **environment,
            "values": {
                **values,
                "CUDA_VISIBLE_DEVICES": {**cuda_policy, "value": policy_csv},
            },
        },
    }
    _write_json(manifest_path, manifest)
    records = _gpu_records(indices=policy_indices)
    probe = _probe(records)

    binding = discover_history_binding(history_root, probe)

    assert probe.calls == [policy_indices, policy_indices]
    assert binding["gpu_policy"] == {
        "indices": list(policy_indices),
        "uuid_by_index": {
            str(index): f"GPU-fixture-{index}" for index in policy_indices
        },
        "hardware_profile": "h800",
    }


def test_binding_hardware_profile_rtx_admits_exact_four_ordered_cards(
    tmp_path: Path,
) -> None:
    """Catches RTX discovery falling back to H800 or reordering its policy."""
    policy_indices = (29, 17, 31, 23)
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"] = (
        "29,17,31,23"
    )
    _write_json(manifest_path, manifest)
    records = _gpu_records(
        indices=policy_indices,
        model_name="NVIDIA GeForce RTX 4090",
        occupancy=0.05,
    )
    probe = _probe(records)

    binding = discover_history_binding(
        history_root,
        probe,
        load_hardware_execution_profile("rtx4090"),
    )

    assert probe.calls == [policy_indices, policy_indices]
    assert binding["target"] == {
        "model": "pyramid",
        "hardware": "rtx4090",
        "backend": "tvm_auto",
    }
    assert binding["gpu_policy"] == {
        "indices": [29, 17, 31, 23],
        "uuid_by_index": {
            "29": "GPU-fixture-29",
            "17": "GPU-fixture-17",
            "31": "GPU-fixture-31",
            "23": "GPU-fixture-23",
        },
        "hardware_profile": "rtx4090",
    }
    assert public_binding_projection(binding)["hardware_profile"] == "rtx4090"


@pytest.mark.parametrize("policy_indices", [(17, 19, 23), (11, 13, 17, 19, 23)])
def test_binding_hardware_profile_rtx_rejects_wrong_cardinality_before_probe(
    tmp_path: Path,
    policy_indices: tuple[int, ...],
) -> None:
    """Catches an RTX binding probing a policy other than exactly four cards."""
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"] = ",".join(
        str(index) for index in policy_indices
    )
    _write_json(manifest_path, manifest)
    records = _gpu_records(
        indices=policy_indices,
        model_name="NVIDIA RTX 4090",
    )
    probe = _probe(records)

    with _expect_category("gpu_admission") as captured:
        discover_history_binding(
            history_root,
            probe,
            load_hardware_execution_profile("rtx4090"),
        )

    assert probe.calls == []
    assert not any(
        token in str(captured.value)
        for token in ("17", "19", "23", "NVIDIA", "RTX 4090")
    )


@pytest.mark.parametrize(
    ("model_name", "occupancy"),
    [
        ("NVIDIA H800 80GB HBM3", 0.0),
        ("NVIDIA RTX 4090", 0.050001),
    ],
)
def test_binding_hardware_profile_rtx_rejects_model_or_occupancy(
    tmp_path: Path,
    model_name: str,
    occupancy: float,
) -> None:
    """Catches RTX admission bypassing registry model or occupancy limits."""
    policy_indices = (11, 13, 17, 19)
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"] = (
        "11,13,17,19"
    )
    _write_json(manifest_path, manifest)
    records = _gpu_records(
        indices=policy_indices,
        model_name=model_name,
        occupancy=occupancy,
    )
    probe = _probe(records)

    with _expect_category("gpu_admission") as captured:
        discover_history_binding(
            history_root,
            probe,
            load_hardware_execution_profile("rtx4090"),
        )

    assert probe.calls == [policy_indices, policy_indices]
    assert model_name not in str(captured.value)


def test_binding_hardware_profile_rtx_rejects_second_snapshot_uuid_drift(
    tmp_path: Path,
) -> None:
    """Catches profile parameterization weakening ordered two-snapshot identity."""
    policy_indices = (11, 13, 17, 19)
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"] = (
        "11,13,17,19"
    )
    _write_json(manifest_path, manifest)
    first = _gpu_records(indices=policy_indices, model_name="NVIDIA RTX 4090")
    second = list(first)
    second[1] = GpuRecord(13, "GPU-drifted", "NVIDIA RTX 4090", 0.0)
    probe = _probe(first, second)

    with _expect_category("gpu_drift") as captured:
        discover_history_binding(
            history_root,
            probe,
            load_hardware_execution_profile("rtx4090"),
        )

    assert probe.calls == [policy_indices, policy_indices]
    assert "GPU-drifted" not in str(captured.value)


def _noncanonical_private_cuda_policies() -> tuple[str, ...]:
    first, second = (record.index for record in _gpu_records()[:2])
    return (
        "",
        f"{first},{first}",
        f"{first},{second},x",
        f"-1,{second}",
        f"0{first},{second}",
        f"{first}, {second}",
    )


@pytest.mark.parametrize("policy", _noncanonical_private_cuda_policies())
def test_binding_rejects_noncanonical_private_cuda_policy(
    tmp_path: Path,
    policy: str,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"] = policy
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize(
    ("stage_index", "placeholder"),
    [
        (0, "{measurement_request}"),
        (1, "{task_state}"),
        (2, "{round_output_root}"),
        (3, "{task_state}"),
        (4, "{actual_receipt}"),
    ],
)
def test_rejects_stage_without_its_required_explicit_binding(
    tmp_path: Path,
    stage_index: int,
    placeholder: str,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_chain"][stage_index]["argv"].remove(placeholder)
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


def test_rejects_required_binding_descriptor_that_does_not_match_argv(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_chain"][4]["required_placeholders"].remove("{finalization_barrier}")
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize(
    "mutation",
    [
        "task_row_count",
        "task_field",
        "result_location",
        "result_metric",
        "result_status",
        "receipt_validation",
        "barrier_validation",
    ],
)
def test_rejects_incomplete_or_unsupported_four_row_actual_feedback_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_state = manifest["output_layout"]["task_state"]
    result = manifest["actual_feedback"]["result"]
    if mutation == "task_row_count":
        task_state["row_count"] = 3
    elif mutation == "task_field":
        task_state.pop("source_evidence_key")
    elif mutation == "result_location":
        result.pop("path_template")
    elif mutation == "result_metric":
        result["metric_keys"].pop()
    elif mutation == "result_status":
        result["allowed_terminal_statuses"] = ["private_unknown_status"]
    elif mutation == "receipt_validation":
        manifest["actual_feedback"]["receipt"].pop("row_hashes_key")
    else:
        manifest["actual_feedback"]["finalization_barrier"]["source_evidence_key"] = "not a field"
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize("environment_key", ["PATH", "PYTHONPATH", "LD_PRELOAD"])
def test_rejects_ambient_or_loader_environment_keys(
    tmp_path: Path,
    environment_key: str,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["values"][environment_key] = {
        "kind": "literal",
        "value": "private-value",
    }
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


def test_rejects_private_environment_path_outside_history_root(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["values"]["P6_HISTORY_PRIVATE_ROOT"]["value"] = str(tmp_path)
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


def test_rejects_non_executable_private_command_endpoint(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    command = history_root / "private-runner" / "bin" / "quantize-private"
    command.chmod(0o600)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_missing_or_ambiguous_private_history_runner_manifest(
    tmp_path: Path,
    count: int,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest_path.unlink()
    for index in range(count):
        _write_json(
            history_root / f"manifest-{index}" / "interface.json",
            _history_runner_manifest(history_root),
        )

    with _expect_category("execution_interface") as raised:
        discover_history_binding(history_root, _probe())

    assert str(history_root) not in str(raised.value)
    assert "private-bound" not in str(raised.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "out_of_order",
        "controller_mismatch",
        "source_materializer_mismatch",
        "performance_mismatch",
        "finalizer_mismatch",
    ],
)
def test_rejects_invalid_or_mismatched_history_execution_chain(
    tmp_path: Path,
    mutation: str,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        manifest["execution_chain"].pop(2)
    elif mutation == "duplicate":
        manifest["execution_chain"][2]["stage"] = "quantization"
    elif mutation == "out_of_order":
        manifest["execution_chain"][1], manifest["execution_chain"][2] = (
            manifest["execution_chain"][2],
            manifest["execution_chain"][1],
        )
    elif mutation == "controller_mismatch":
        manifest["controller"]["argv"] = [
            str(history_root / "private-runner" / "bin" / "quantize-private")
        ]
    else:
        stage_index = {
            "source_materializer_mismatch": 0,
            "performance_mismatch": 2,
            "finalizer_mismatch": 4,
        }[mutation]
        manifest["execution_chain"][stage_index]["argv"][0] = str(
            history_root / "private-runner" / "bin" / "quantize-private"
        )
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize(
    "mutation",
    ["shell_command", "unknown_token", "path_escape", "unknown_environment"],
)
def test_rejects_unsafe_private_execution_interface_values(
    tmp_path: Path,
    mutation: str,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "shell_command":
        manifest["execution_chain"][1]["argv"][0] = "bash -c private-bound"
    elif mutation == "unknown_token":
        manifest["execution_chain"][1]["argv"].append("{ambient_path}")
    elif mutation == "path_escape":
        manifest["execution_chain"][1]["argv"][0] = str(tmp_path / "outside")
    else:
        manifest["environment"]["ambient"] = "private-bound"
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface") as raised:
        discover_history_binding(history_root, _probe())

    assert "private-bound" not in str(raised.value)


def test_rejects_duplicate_private_environment_key_without_leaking_value(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_text = manifest_text.replace(
        '"P6_HISTORY_RUN_MODE": {',
        '"CUDA_VISIBLE_DEVICES": {"kind": "literal", '
        '"value": "private-duplicate"}, "P6_HISTORY_RUN_MODE": {',
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")

    with _expect_category("execution_interface") as raised:
        discover_history_binding(history_root, _probe())

    assert "private-duplicate" not in str(raised.value)


@pytest.mark.parametrize(
    "mutation",
    ["task_state_omission", "output_escape", "receipt_omission", "barrier_escape"],
)
def test_rejects_incomplete_or_escaping_private_feedback_layout(
    tmp_path: Path,
    mutation: str,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "task_state_omission":
        manifest["output_layout"].pop("task_state")
    elif mutation == "output_escape":
        manifest["output_layout"]["round_root_template"] = "../outside/{round_id}"
    elif mutation == "receipt_omission":
        manifest["actual_feedback"].pop("receipt")
    else:
        manifest["actual_feedback"]["finalization_barrier"]["path_template"] = "../barrier.json"
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize(
    ("location", "template"),
    [
        ("round_root", "private-runs/fixed"),
        ("task_state", "private-runs/fixed/task-state.json"),
        ("result", "private-runs/fixed/actual-feedback.json"),
        ("receipt", "private-runs/fixed/receipt.json"),
        ("barrier", "private-runs/fixed/barrier.json"),
        ("round_root", "private-runs/{round_id}/{round_id}"),
        ("task_state", "private-runs/{unknown_round}/task-state.json"),
        ("result", "../outside/{round_id}/actual-feedback.json"),
        ("receipt", "private-runs/{round_id}}/receipt.json"),
        ("barrier", "private-runs/{{round_id}/barrier.json"),
    ],
)
def test_rejects_per_round_layout_template_without_exactly_one_round_id(
    tmp_path: Path,
    location: str,
    template: str,
) -> None:
    history_root = _history_root(tmp_path)
    manifest_path = history_root / "private-runner" / "p6-history-runner-interface.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if location == "round_root":
        manifest["output_layout"]["round_root_template"] = template
    elif location == "task_state":
        manifest["output_layout"]["task_state"]["path_template"] = template
    elif location == "result":
        manifest["actual_feedback"]["result"]["path_template"] = template
    elif location == "receipt":
        manifest["actual_feedback"]["receipt"]["path_template"] = template
    else:
        manifest["actual_feedback"]["finalization_barrier"]["path_template"] = template
    _write_json(manifest_path, manifest)

    with _expect_category("execution_interface"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_rejects_zero_or_multiple_documented_component_markers(
    tmp_path: Path,
    mode: str,
) -> None:
    history_root = _history_root(tmp_path)
    marker = MARKERS["controller"]
    (history_root / "documented-stage5-chain" / marker).unlink()
    if mode == "duplicate":
        for directory in (history_root / "first", history_root / "second"):
            directory.mkdir()
            (directory / marker).write_text("duplicate\n", encoding="utf-8")

    with _expect_category("component_discovery"):
        discover_history_binding(history_root, _probe())


def test_rejects_missing_or_ambiguous_required_local_input(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    missing_path = history_root / "inputs" / "closure.json"
    missing_path.unlink()

    with _expect_category("local_inputs"):
        discover_history_binding(history_root, _probe())

    _write_json(missing_path, {"fixture": "closure"})
    _write_json(history_root / "duplicate" / "closure.json", {"fixture": "closure"})
    with _expect_category("local_inputs"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize(
    ("mutation", "category"),
    [
        ("registry_version", "source_registry"),
        ("contract_version", "source_registry"),
        ("target_model", "source_registry"),
        ("backend", "source_registry"),
        ("incomplete", "source_registry"),
    ],
)
def test_rejects_incompatible_or_incomplete_source_registry_template(
    tmp_path: Path,
    mutation: str,
    category: str,
) -> None:
    history_root = _history_root(tmp_path)
    registry_path = history_root / "registry" / "candidate_source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    group = registry["groups"][0]
    contract = group["source_contract"]
    if mutation == "registry_version":
        registry["schema_version"] = "stage5_candidate_source_registry_v2"
    elif mutation == "contract_version":
        contract["schema_version"] = "stage5_source_contract_v2"
        group["source_contract_sha256"] = _canonical_json_sha(contract)
    elif mutation == "target_model":
        group["model"] = contract["model"] = "resnet"
        group["source_contract_sha256"] = _canonical_json_sha(contract)
    elif mutation == "backend":
        group["materialization_kind"] = "local_pyramid_trt"
    else:
        contract.pop("artifact_id")
        group["source_contract_sha256"] = _canonical_json_sha(contract)
    _write_json(registry_path, registry)

    with _expect_category(category):
        discover_history_binding(history_root, _probe())


def test_rejects_discovered_path_resolving_outside_history_root(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    marker = MARKERS["finalizer"]
    marker_path = history_root / "documented-stage5-chain" / marker
    marker_path.unlink()
    outside = tmp_path / marker
    outside.write_text("outside\n", encoding="utf-8")
    marker_path.symlink_to(outside)

    with _expect_category("path_escape"):
        discover_history_binding(history_root, _probe())


@pytest.mark.parametrize(
    "records",
    [
        _gpu_records(indices=(17, 19)),
        _gpu_records(indices=(13, 17, 19, 23)),
        _gpu_records(model_name="NVIDIA A100-SXM4-80GB"),
        (
            GpuRecord(17, "duplicate", "NVIDIA H800 80GB HBM3", 0.0),
            GpuRecord(19, "duplicate", "NVIDIA H800 80GB HBM3", 0.0),
            GpuRecord(23, "unique", "NVIDIA H800 80GB HBM3", 0.0),
        ),
        (
            GpuRecord(17, "GPU-fixture-17", "NVIDIA H800 80GB HBM3", 0.0),
            GpuRecord(19, "", "NVIDIA H800 80GB HBM3", 0.0),
            GpuRecord(23, "GPU-fixture-23", "NVIDIA H800 80GB HBM3", 0.0),
        ),
        _gpu_records(occupancy=0.25),
    ],
)
def test_gpu_admission_fails_closed(records: tuple[GpuRecord, ...], tmp_path: Path) -> None:
    with _expect_category("gpu_admission"):
        discover_history_binding(_history_root(tmp_path), _probe(records))


@pytest.mark.parametrize(
    ("uuid", "occupancy"),
    [
        (None, 0.0),
        (12345, 0.0),
        ("GPU-fixture-17", "0"),
        ("GPU-fixture-17", "idle"),
    ],
)
def test_malformed_gpu_record_fields_fail_with_stable_admission_category(
    tmp_path: Path,
    uuid: Any,
    occupancy: Any,
) -> None:
    records = list(_gpu_records())
    records[0] = GpuRecord(17, uuid, "NVIDIA H800 80GB HBM3", occupancy)

    with _expect_category("gpu_admission"):
        discover_history_binding(_history_root(tmp_path), _probe(records))


def test_deceptive_h800_substring_model_fails_admission(tmp_path: Path) -> None:
    records = _gpu_records(model_name="NOT-H800-COMPATIBLE")

    with _expect_category("gpu_admission"):
        discover_history_binding(_history_root(tmp_path), _probe(records))


def test_gpu_second_snapshot_uuid_drift_fails_closed(tmp_path: Path) -> None:
    drifted = list(_gpu_records())
    drifted[2] = GpuRecord(23, "GPU-drifted", "NVIDIA H800 80GB HBM3", 0.0)

    with _expect_category("gpu_drift"):
        discover_history_binding(
            _history_root(tmp_path),
            _probe(_gpu_records(), drifted),
        )


def _private_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "schema_version": "p6_history_binding_v1",
            "target": {
                "model": "pyramid",
                "hardware": "h800",
                "backend": "tvm_auto",
            },
        },
        {"caller_owned_local_config": True},
    )


def test_production_git_check_ignore_rejects_nonignored_repository_destination(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-q", str(repo_root)], check=True)
    output_root = repo_root / "private"
    output_root.mkdir()
    binding, config = _private_payloads()

    with _expect_category("unsafe_destination"):
        write_private_binding_pair(
            binding,
            config,
            output_root / "binding.json",
            output_root / "config.json",
            repo_root,
        )

    assert not tuple(output_root.iterdir())


def test_rejects_existing_symlink_destination(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    output_root = repo_root / "private"
    output_root.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n", encoding="utf-8")
    binding_path = output_root / "binding.json"
    binding_path.symlink_to(outside)
    binding, config = _private_payloads()

    with _expect_category("unsafe_destination"):
        write_private_binding_pair(
            binding,
            config,
            binding_path,
            output_root / "config.json",
            repo_root,
            ignore_predicate=lambda path: True,
        )

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_pair_prevalidation_preserves_both_existing_targets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    output_root = repo_root / "private"
    output_root.mkdir(parents=True)
    binding_path = output_root / "binding.json"
    config_path = output_root / "config.json"
    binding_path.write_text("old binding\n", encoding="utf-8")
    config_path.write_text("old config\n", encoding="utf-8")
    binding, config = _private_payloads()

    with _expect_category("unsafe_destination"):
        write_private_binding_pair(
            binding,
            config,
            binding_path,
            config_path,
            repo_root,
            ignore_predicate=lambda path: path.name == "binding.json",
        )

    assert binding_path.read_text(encoding="utf-8") == "old binding\n"
    assert config_path.read_text(encoding="utf-8") == "old config\n"
    assert not tuple(output_root.glob("*.tmp"))


def test_success_atomically_replaces_both_private_json_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    output_root = repo_root / "private"
    output_root.mkdir(parents=True)
    binding_path = output_root / "binding.json"
    config_path = output_root / "config.json"
    binding_path.write_text("old binding\n", encoding="utf-8")
    config_path.write_text("old config\n", encoding="utf-8")
    binding, config = _private_payloads()

    write_private_binding_pair(
        binding,
        config,
        binding_path,
        config_path,
        repo_root,
        ignore_predicate=lambda path: True,
    )

    assert json.loads(binding_path.read_text(encoding="utf-8")) == binding
    assert json.loads(config_path.read_text(encoding="utf-8")) == config
    assert not tuple(output_root.glob("*.tmp"))


@pytest.mark.parametrize(
    "initial_contents",
    [
        (b"old binding\xff\n", b"old config\x00\n"),
        (None, None),
    ],
    ids=("preexisting_destinations", "initially_absent_destinations"),
)
def test_second_replace_failure_restores_the_original_private_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_contents: tuple[bytes | None, bytes | None],
) -> None:
    repo_root = tmp_path / "repo"
    output_root = repo_root / "private"
    output_root.mkdir(parents=True)
    binding_path = output_root / "binding.json"
    config_path = output_root / "config.json"
    old_binding, old_config = initial_contents
    if old_binding is not None:
        binding_path.write_bytes(old_binding)
    if old_config is not None:
        config_path.write_bytes(old_config)
    binding, config = _private_payloads()
    real_replace = os.replace
    replace_count = 0

    def fail_second_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("synthetic second replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "framework.stage6.p6_history_binding_v1.os.replace",
        fail_second_replace,
    )

    with _expect_category("persistence"):
        write_private_binding_pair(
            binding,
            config,
            binding_path,
            config_path,
            repo_root,
            ignore_predicate=lambda path: True,
        )

    assert _read_bytes_or_none(binding_path) == old_binding
    assert _read_bytes_or_none(config_path) == old_config
    assert not tuple(output_root.glob("*.tmp"))


def test_second_replace_failure_does_not_overwrite_a_competing_binding_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_root = repo_root / "private"
    output_root.mkdir(parents=True)
    binding_path = output_root / "binding.json"
    config_path = output_root / "config.json"
    binding_path.write_bytes(b"old binding\n")
    config_path.write_bytes(b"old config\n")
    binding, config = _private_payloads()
    real_replace = os.replace
    replace_count = 0
    competitor_bytes = b"competing binding update\n"

    def fail_second_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            binding_path.write_bytes(competitor_bytes)
            raise OSError("synthetic second replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "framework.stage6.p6_history_binding_v1.os.replace",
        fail_second_replace,
    )

    with _expect_category("persistence"):
        write_private_binding_pair(
            binding,
            config,
            binding_path,
            config_path,
            repo_root,
            ignore_predicate=lambda path: True,
        )

    assert binding_path.read_bytes() == competitor_bytes
    assert config_path.read_bytes() == b"old config\n"
    assert not tuple(output_root.glob("*.tmp"))


def _read_bytes_or_none(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def test_discovery_builders_do_not_mutate_registry_or_gpu_records(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    registry_path = history_root / "registry" / "candidate_source_registry.json"
    registry_before = copy.deepcopy(json.loads(registry_path.read_text(encoding="utf-8")))
    records = _gpu_records()

    discover_history_binding(history_root, _probe(records))

    assert json.loads(registry_path.read_text(encoding="utf-8")) == registry_before
    assert records == _gpu_records()
