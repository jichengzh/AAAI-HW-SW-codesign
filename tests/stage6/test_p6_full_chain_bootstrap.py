from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import pytest
import yaml

import framework.stage6.p6_full_chain_bootstrap_v1 as bootstrap
from framework.stage6.p6_full_chain_bootstrap_outputs_v1 import (
    resolve_private_outputs,
)
import framework.stage6.p6_runner_template_validator_v1 as runner_template_validator
from framework.stage6.coptv2x_h800_search_v2 import (
    PublicP6CoptV2XContract,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_full_chain_bootstrap_v1 import (
    FullChainBootstrapError,
    materialize_full_chain_binding,
)
from framework.stage6.p6_history_binding_v1 import GpuRecord
from framework.stage6.p6_source_wrapper_profile_v1 import (
    render_self_contained_source_wrapper,
)
from tests.p6_source_wrapper_support import write_test_project_python
from tests.stage6.test_p6_history_normalization import _recipe_v2


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
STAGES = (
    "source_materialization",
    "quantization",
    "performance",
    "ap",
    "finalization",
)


class SequenceProbe:
    def __init__(self, snapshots: tuple[tuple[GpuRecord, ...], ...]) -> None:
        self._snapshots = snapshots
        self._index = 0
        self.calls: list[tuple[int, ...]] = []

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        assert indices == (17, 19, 23)
        self.calls.append(indices)
        snapshot = self._snapshots[min(self._index, len(self._snapshots) - 1)]
        self._index += 1
        return snapshot


def _gpu_probe() -> SequenceProbe:
    records = tuple(
        GpuRecord(
            index=index,
            uuid=f"GPU-synthetic-{index}",
            model_name="NVIDIA H800 80GB HBM3",
            occupancy=0.0,
        )
        for index in (17, 19, 23)
    )
    return SequenceProbe((records, records))


def _write_yaml(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic executable\n", encoding="utf-8")
    path.chmod(0o700)
    return path


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


def _public_contract(tmp_path: Path) -> PublicP6CoptV2XContract:
    payload = {
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
            {
                "label": "training-data",
                "version": "v1",
                "license_status": "cleared",
            },
            {
                "label": "model-init",
                "version": "v2",
                "license_status": "cleared",
            },
            {
                "label": "toolchain",
                "version": "v3",
                "license_status": "cleared",
            },
        ],
    }
    return load_public_contract(_write_yaml(tmp_path / "public.yaml", payload))


def _runner_template() -> dict[str, Any]:
    return {
        "schema_version": "p6_history_runner_template_v1",
        "stage1_scan": {
            "name": "build_stage1_partition",
            "argv": [
                "private-runner/bin/scan-private",
                "{stage1_partition_manifest}",
                "{local_output_root}",
            ],
        },
        "execution_interface": {
            "schema_version": "p6_history_runner_interface_v1",
            "controller": {"argv": [f"documented-stage5-chain/{MARKERS['controller']}"]},
            "execution_chain": [
                {
                    "stage": "source_materialization",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['source_materializer']}",
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
                        "private-runner/bin/quantize-private",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                    "required_placeholders": ["{task_state}", "{round_output_root}"],
                },
                {
                    "stage": "performance",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['performance_plan']}",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                    "required_placeholders": ["{task_state}", "{round_output_root}"],
                },
                {
                    "stage": "ap",
                    "argv": [
                        "private-runner/bin/measure-ap-private",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                    "required_placeholders": ["{task_state}", "{round_output_root}"],
                },
                {
                    "stage": "finalization",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['finalizer']}",
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
                    "private-runner/bin/activate-private",
                    "private-bound",
                ],
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
                    "stage_order": list(STAGES),
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
    }


def _initialize_source_root(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for marker in MARKERS.values():
        _write_executable(root / "documented-stage5-chain" / marker)
    for executable in (
        "scan-private",
        "quantize-private",
        "measure-ap-private",
        "activate-private",
    ):
        _write_executable(root / "private-runner" / "bin" / executable)
    registry = _write_json(
        root / "registry" / "candidate_source_registry.json",
        {
            "schema_version": "stage5_candidate_source_registry_v1",
            "groups": [_source_group()],
        },
    )
    inputs = {
        name: _write_json(root / "inputs" / f"{name}.json", {"fixture": name})
        for name in LOCAL_INPUT_NAMES
    }
    return {**inputs, "registry": registry}


def _legacy_locator(root: Path, inputs: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": "p6_h800_coptv2x_local_v2",
        "target": "h800",
        "asset_paths": {
            "training-data": str(root / "inputs"),
            "model-init": str(inputs["registry"]),
            "toolchain": str(root / "documented-stage5-chain" / MARKERS["performance_plan"]),
        },
        "local_input_paths": {name: str(inputs[name]) for name in LOCAL_INPUT_NAMES},
        "source_registry_step": {
            "name": "build_source_registry",
            "argv": [
                "legacy-registry",
                "{local_output_root}",
                "{source_registry_json}",
            ],
        },
        "measurement_step": {
            "name": "measure_batch",
            "argv": [
                "legacy-measure",
                "{measurement_request}",
                "{feedback_json}",
                "{round_output_root}",
            ],
        },
        "local_output_root": str(root / "legacy-output"),
    }


def _write_valid_private_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "source"
    inputs = _initialize_source_root(root)
    legacy_config = _write_yaml(
        tmp_path / "private-inputs" / "legacy.local.yaml",
        _legacy_locator(root, inputs),
    )
    template = _write_yaml(
        tmp_path / "private-inputs" / "runner-template.yaml",
        _runner_template(),
    )
    output_root = tmp_path / "private-output"
    output_root.mkdir()
    return legacy_config, template, output_root


def _attach_expected_recipe(
    tmp_path: Path,
    legacy_config: Path,
    *,
    actual_recipe: Mapping[str, Any],
    expected_recipe: Mapping[str, Any],
) -> Path:
    root = tmp_path / "source"
    operator = tmp_path / "operator-assets"
    dataset = operator / "datasets" / "coptv2x"
    stable = operator / "stable"
    dataset.mkdir(parents=True, exist_ok=True)
    stable.mkdir(parents=True, exist_ok=True)
    checkpoint = stable / "base.ckpt"
    config = stable / "pyramid.py"
    checkpoint.write_text("base\n", encoding="utf-8")
    config.write_text(
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )
    external = {
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
    }
    _write_yaml(tmp_path / "private-inputs" / "external-training-binding.yaml", external)
    registry_path = root / "registry" / "candidate_source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    contract = registry["groups"][0]["source_contract"]
    contract.update(
        {
            "external_training_binding": copy.deepcopy(external),
            "dynamic_materialization_recipe": copy.deepcopy(dict(actual_recipe)),
        }
    )
    registry["groups"][0]["source_contract_sha256"] = _canonical_json_sha(contract)
    _write_json(registry_path, registry)
    expected_path = _write_json(root / "derivation" / "recipe.json", expected_recipe)
    legacy = yaml.safe_load(legacy_config.read_text(encoding="utf-8"))
    legacy["history_recipe_derivation_path"] = str(expected_path)
    _write_yaml(legacy_config, legacy)
    implementation = (
        root
        / "private-relocated-history-repo"
        / "bin"
        / "stage5_materialize_round_sources_v1.original.sh"
    )
    _write_executable(implementation)
    marker = root / "documented-stage5-chain" / MARKERS["source_materializer"]
    marker.unlink()
    project_python = write_test_project_python(tmp_path)
    profile_payload = {
        "schema_version": "p6_private_source_wrapper_profile_v2",
        "wrapper_kind": "repo_cwd_exec_v1",
        "destination_relative_path": (f"documented-stage5-chain/{MARKERS['source_materializer']}"),
        "implementation_relative_path": (
            "private-relocated-history-repo/bin/stage5_materialize_round_sources_v1.original.sh"
        ),
        "implementation_cwd_relative_path": "private-relocated-history-repo",
        "project_python": str(project_python),
    }
    profile_path = _write_yaml(
        tmp_path / "private-inputs" / "source-wrapper-profile.yaml",
        profile_payload,
    )
    render_self_contained_source_wrapper(profile_payload, history_root=root)
    return profile_path


def _write_invalid_private_inputs(tmp_path: Path, mutation: str) -> tuple[Path, Path, Path]:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    legacy_payload = yaml.safe_load(legacy_config.read_text(encoding="utf-8"))
    template_payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if mutation == "two_common_roots":
        second_root = tmp_path / "second-source"
        second_inputs = _initialize_source_root(second_root)
        legacy_payload["local_input_paths"]["closure"] = str(second_inputs["closure"])
        _write_yaml(legacy_config, legacy_payload)
    elif mutation == "missing_stage":
        template_payload["execution_interface"]["execution_chain"].pop(2)
        _write_yaml(template, template_payload)
    elif mutation == "escape":
        _write_executable(tmp_path / "escape-bin")
        template_payload["execution_interface"]["execution_chain"][1]["argv"][0] = "../escape-bin"
        _write_yaml(template, template_payload)
    elif mutation == "bad_template":
        template_payload["schema_version"] = "p6_history_runner_template_v0"
        _write_yaml(template, template_payload)
    elif mutation == "stage1_shell_token":
        template_payload["stage1_scan"]["argv"].insert(1, "unsafe;token")
        _write_yaml(template, template_payload)
    else:
        raise AssertionError(f"unsupported mutation: {mutation}")
    return legacy_config, template, output_root


def _template_payload(template: Path) -> dict[str, Any]:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _add_archived_duplicate_markers(root: Path) -> None:
    for marker in MARKERS.values():
        _write_executable(root / "archived-stage5-copy" / marker)


def test_materialize_full_chain_binding_renders_dynamic_config(
    tmp_path: Path,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)

    binding = materialize_full_chain_binding(
        legacy_config,
        template,
        output_root,
        output_root / "binding.json",
        output_root / "local.yaml",
        _gpu_probe(),
    )

    local = load_local_config(output_root / "local.yaml", _public_contract(tmp_path))
    config = yaml.safe_load((output_root / "local.yaml").read_text(encoding="utf-8"))
    assert binding["schema_version"] == "p6_history_binding_v1"
    assert (
        tuple(step["stage"] for step in binding["execution_interface"]["execution_chain"]) == STAGES
    )
    assert local.candidate_source_mode == "framework_stage2_search_space"
    assert local.stage1_scan_step is not None
    assert local.stage1_scan_step.name == "build_stage1_partition"
    assert config["stage2_search_space_path"] == str(output_root / "stage1_partition_manifest.json")
    assert set(config["stage1_scan_step"]["argv"]) >= {
        "{stage1_partition_manifest}",
        "{local_output_root}",
    }
    assert set(config["source_registry_step"]["argv"]) >= {
        "{local_output_root}",
        "{source_registry_json}",
        "{pyramid_candidate_plan}",
    }
    assert set(config["measurement_step"]["argv"]) >= {
        "{measurement_request}",
        "{feedback_json}",
        "{round_output_root}",
    }


def test_materialize_preserves_virtual_environment_python_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated child commands must keep the active virtual environment."""
    python_target = _write_executable(tmp_path / "runtime/python3.10")
    python_alias = tmp_path / "venv/bin/python"
    python_alias.parent.mkdir(parents=True)
    python_alias.symlink_to(python_target)
    assert python_alias != python_alias.resolve()
    monkeypatch.setattr(bootstrap.sys, "executable", str(python_alias))
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)

    materialize_full_chain_binding(
        legacy_config,
        template,
        output_root,
        output_root / "binding.json",
        output_root / "local.yaml",
        _gpu_probe(),
    )

    config = yaml.safe_load((output_root / "local.yaml").read_text(encoding="utf-8"))
    assert config["source_registry_step"]["argv"][0] == str(python_alias)
    assert config["measurement_step"]["argv"][0] == str(python_alias)


def test_materialize_selects_runtime_gpu_count_and_late_binds_without_mutating_template(
    tmp_path: Path,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    template_payload = _template_payload(template)
    template_payload["execution_interface"]["environment"]["values"][
        "CUDA_VISIBLE_DEVICES"
    ] = {"kind": "runtime", "value": "GPU_POOL"}
    _write_yaml(template, template_payload)
    original_template_bytes = template.read_bytes()
    class RuntimeCountProbe:
        def __init__(self) -> None:
            self.all_calls = 0
            self.calls: list[tuple[int, ...]] = []

        def snapshot_all(self) -> tuple[GpuRecord, ...]:
            self.all_calls += 1
            records = (
                GpuRecord(19, "GPU-runtime-19", "NVIDIA H800 80GB HBM3", 0.0),
                GpuRecord(7, "GPU-busy-7", "NVIDIA H800 80GB HBM3", 0.5),
                GpuRecord(3, "GPU-wrong-3", "NVIDIA GeForce RTX 4090", 0.0),
            )
            return records if self.all_calls == 1 else tuple(reversed(records))

        def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
            self.calls.append(indices)
            return tuple(
                GpuRecord(
                    index=index,
                    uuid=f"GPU-runtime-{index}",
                    model_name="NVIDIA H800 80GB HBM3",
                    occupancy=0.0,
                )
                for index in indices
            )

    probe = RuntimeCountProbe()
    binding = materialize_full_chain_binding(
        legacy_config,
        template,
        output_root,
        output_root / "binding.json",
        output_root / "local.yaml",
        probe,
        runtime_gpu_count=1,
    )

    assert probe.all_calls == 2
    assert probe.calls == [(19,), (19,)]
    assert binding["gpu_policy"] == {
        "indices": [19],
        "uuid_by_index": {"19": "GPU-runtime-19"},
        "hardware_profile": "h800",
    }
    assert (
        binding["execution_interface"]["environment"]["values"]
        ["CUDA_VISIBLE_DEVICES"]
        == {"kind": "literal", "value": "19"}
    )
    assert template.read_bytes() == original_template_bytes


def test_materialize_rejects_runtime_gpu_count_when_too_few_stable_idle_devices(
    tmp_path: Path,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)

    class InsufficientProbe:
        def __init__(self) -> None:
            self.all_calls = 0

        def snapshot_all(self) -> tuple[GpuRecord, ...]:
            self.all_calls += 1
            uuid = "GPU-drift-first" if self.all_calls == 1 else "GPU-drift-second"
            return (
                GpuRecord(7, "GPU-stable-7", "NVIDIA H800 80GB HBM3", 0.0),
                GpuRecord(9, uuid, "NVIDIA H800 80GB HBM3", 0.0),
            )

        def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
            pytest.fail(f"selected snapshot must not run: {indices}")

    probe = InsufficientProbe()
    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            probe,
            runtime_gpu_count=2,
        )

    assert captured.value.category == "gpu_admission"
    assert probe.all_calls == 2
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_accepts_matching_normalized_recipe_before_pair_write(
    tmp_path: Path,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    recipe = _recipe_v2()
    profile = _attach_expected_recipe(
        tmp_path,
        legacy_config,
        actual_recipe=recipe,
        expected_recipe=recipe,
    )

    binding = materialize_full_chain_binding(
        legacy_config,
        template,
        output_root,
        output_root / "binding.json",
        output_root / "local.yaml",
        _gpu_probe(),
        source_wrapper_profile=profile,
        external_training_binding=(tmp_path / "private-inputs" / "external-training-binding.yaml"),
    )

    assert binding["source_contract_template"]["dynamic_materialization_recipe"] == recipe
    assert (output_root / "binding.json").exists()
    assert (output_root / "local.yaml").exists()


def test_materialize_rejects_runner_path_argv_tail_before_probe_or_pair_write(
    tmp_path: Path,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    recipe = _recipe_v2()
    profile = _attach_expected_recipe(
        tmp_path,
        legacy_config,
        actual_recipe=recipe,
        expected_recipe=recipe,
    )
    payload = _template_payload(template)
    payload["execution_interface"]["controller"]["argv"].append(
        "/tmp/undeclared-source-tree/config.py"
    )
    _write_yaml(template, payload)
    probe = _gpu_probe()

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            probe,
            source_wrapper_profile=profile,
            external_training_binding=(
                tmp_path / "private-inputs" / "external-training-binding.yaml"
            ),
        )

    assert captured.value.category == "execution_interface_unavailable"
    assert probe.calls == []
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_rejects_recipe_drift_before_pair_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    actual = _recipe_v2()
    expected = copy.deepcopy(actual)
    expected["artifact_id_template"] = "pyramid-drift-{stage1_width}-{stage2_width}-{stage3_width}"
    profile = _attach_expected_recipe(
        tmp_path,
        legacy_config,
        actual_recipe=actual,
        expected_recipe=expected,
    )
    monkeypatch.setattr(
        bootstrap,
        "write_private_binding_pair",
        lambda *_args, **_kwargs: pytest.fail("pair writer must not be called"),
    )

    with pytest.raises(
        FullChainBootstrapError,
        match=r"^history_recipe_derivation_invalid:",
    ):
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            _gpu_probe(),
            source_wrapper_profile=profile,
            external_training_binding=(
                tmp_path / "private-inputs" / "external-training-binding.yaml"
            ),
        )

    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_recipe_v2_requires_wrapper_profile_before_probe_or_pair_write(
    tmp_path: Path,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    recipe = _recipe_v2()
    _attach_expected_recipe(
        tmp_path,
        legacy_config,
        actual_recipe=recipe,
        expected_recipe=recipe,
    )
    probe = _gpu_probe()

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            probe,
        )

    assert captured.value.category == "history_execution_invalid"
    assert probe.calls == []
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_rejects_differing_wrapper_marker_without_probe_or_pair_write(
    tmp_path: Path,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    recipe = _recipe_v2()
    profile = _attach_expected_recipe(
        tmp_path,
        legacy_config,
        actual_recipe=recipe,
        expected_recipe=recipe,
    )
    marker = tmp_path / "source" / "documented-stage5-chain" / MARKERS["source_materializer"]
    marker.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
    marker.chmod(0o700)
    probe = _gpu_probe()

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            probe,
            source_wrapper_profile=profile,
            external_training_binding=(
                tmp_path / "private-inputs" / "external-training-binding.yaml"
            ),
        )

    assert captured.value.category == "history_execution_invalid"
    assert probe.calls == []
    assert marker.read_text(encoding="utf-8") == "#!/bin/sh\nexit 91\n"
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_rejects_out_of_root_recipe_before_read_or_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    outside_recipe = _write_json(tmp_path / "outside" / "recipe.json", _recipe_v2())
    legacy = yaml.safe_load(legacy_config.read_text(encoding="utf-8"))
    legacy["history_recipe_derivation_path"] = str(outside_recipe)
    _write_yaml(legacy_config, legacy)
    probe = _gpu_probe()
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == outside_recipe:
            pytest.fail("out-of-root recipe must not be read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    with pytest.raises(
        FullChainBootstrapError,
        match=r"^history_recipe_derivation_invalid:",
    ):
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            probe,
        )

    assert probe.calls == []
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_uses_template_component_paths_when_history_has_archived_duplicates(
    tmp_path: Path,
) -> None:
    """Catches falling back to recursive marker discovery instead of the template."""
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    root = tmp_path / "source"
    _add_archived_duplicate_markers(root)

    binding = materialize_full_chain_binding(
        legacy_config,
        template,
        output_root,
        output_root / "binding.json",
        output_root / "local.yaml",
        _gpu_probe(),
    )

    assert binding["status"] == "validated"
    assert binding["component_paths"] == {
        role: str(root / "documented-stage5-chain" / marker) for role, marker in MARKERS.items()
    }


def test_materialize_rejects_template_component_outside_history_root_without_pair(
    tmp_path: Path,
) -> None:
    """Catches accepting a template-selected Stage5 component outside the Git root."""
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    payload = _template_payload(template)
    outside = _write_executable(tmp_path / "outside" / MARKERS["controller"])
    payload["execution_interface"]["controller"]["argv"] = [str(outside)]
    _write_yaml(template, payload)

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            _gpu_probe(),
        )

    assert captured.value.category == "history_root_ambiguous"
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_rejects_template_component_role_mismatch_without_pair(
    tmp_path: Path,
) -> None:
    """Catches binding one Stage5 role to a different rendered component argv."""
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    payload = _template_payload(template)
    payload["execution_interface"]["execution_chain"][2]["argv"][0] = (
        f"documented-stage5-chain/{MARKERS['finalizer']}"
    )
    _write_yaml(template, payload)

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            _gpu_probe(),
        )

    assert captured.value.category == "history_root_ambiguous"
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_rejects_absolute_symlink_leaf_stage1_template_path_without_pair(
    tmp_path: Path,
) -> None:
    """Catches accepting a non-component template executable through a symlink leaf."""
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    root = tmp_path / "source"
    payload = _template_payload(template)
    link = root / "private-runner" / "bin" / "scan-private-link"
    link.symlink_to(root / "private-runner" / "bin" / "scan-private")
    payload["stage1_scan"]["argv"][0] = str(link)
    _write_yaml(template, payload)

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            _gpu_probe(),
        )

    assert captured.value.category == "execution_interface_unavailable"
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_materialize_rejects_relative_symlink_parent_quantization_path_without_pair(
    tmp_path: Path,
) -> None:
    """Catches accepting a non-component template executable through a symlink parent."""
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    root = tmp_path / "source"
    payload = _template_payload(template)
    (root / "linked-runner").symlink_to(root / "private-runner", target_is_directory=True)
    payload["execution_interface"]["execution_chain"][1]["argv"][0] = (
        "linked-runner/bin/quantize-private"
    )
    _write_yaml(template, payload)

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            _gpu_probe(),
        )

    assert captured.value.category == "execution_interface_unavailable"
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_bootstrap_rejects_nonignored_repository_runner_template_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches parsing or probing from a trackable in-repository runner template."""
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    repository = tmp_path / "template-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    repository_template = _write_yaml(
        repository / "local" / "runner-template.yaml",
        yaml.safe_load(template.read_text(encoding="utf-8")),
    )
    probe = _gpu_probe()
    monkeypatch.setattr(runner_template_validator, "REPOSITORY_ROOT", repository)

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            repository_template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            probe,
        )

    assert captured.value.category == "execution_interface_unavailable"
    assert probe.calls == []
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_bootstrap_rejects_symlinked_repository_template_parent_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches treating a repository symlink escape as an external template."""
    legacy_config, external_template, output_root = _write_valid_private_inputs(tmp_path)
    repository = tmp_path / "template-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "linked-private-inputs").symlink_to(
        external_template.parent, target_is_directory=True
    )
    template = repository / "linked-private-inputs" / external_template.name
    probe = _gpu_probe()
    monkeypatch.setattr(runner_template_validator, "REPOSITORY_ROOT", repository)

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            probe,
        )

    assert captured.value.category == "execution_interface_unavailable"
    assert probe.calls == []
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


@pytest.mark.parametrize("template_location", ["ignored_repository", "repository_external"])
def test_bootstrap_accepts_private_runner_template_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    template_location: str,
) -> None:
    """Locks the accepted ignored-repository and repository-external boundaries."""
    legacy_config, external_template, output_root = _write_valid_private_inputs(tmp_path)
    repository = tmp_path / "template-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text("private-runner/\n", encoding="utf-8")
    if template_location == "ignored_repository":
        template = _write_yaml(
            repository / "private-runner" / "runner-template.yaml",
            yaml.safe_load(external_template.read_text(encoding="utf-8")),
        )
    else:
        template = external_template
    probe = _gpu_probe()
    monkeypatch.setattr(runner_template_validator, "REPOSITORY_ROOT", repository)

    binding = materialize_full_chain_binding(
        legacy_config,
        template,
        output_root,
        output_root / "binding.json",
        output_root / "local.yaml",
        probe,
    )

    assert binding["status"] == "validated"
    assert probe.calls == [(17, 19, 23), (17, 19, 23)]


@pytest.mark.parametrize(
    "mutation",
    [
        "two_common_roots",
        "missing_stage",
        "escape",
        "bad_template",
        "stage1_shell_token",
    ],
)
def test_bootstrap_rejects_untrusted_or_ambiguous_private_inputs(
    tmp_path: Path, mutation: str
) -> None:
    legacy_config, template, output_root = _write_invalid_private_inputs(tmp_path, mutation)

    with pytest.raises(FullChainBootstrapError):
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            _gpu_probe(),
        )

    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_bootstrap_rejects_binding_path_reserved_for_fresh_stage1_manifest(
    tmp_path: Path,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)

    with pytest.raises(FullChainBootstrapError):
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "stage1_partition_manifest.json",
            output_root / "local.yaml",
            _gpu_probe(),
        )

    assert not (output_root / "stage1_partition_manifest.json").exists()
    assert not (output_root / "local.yaml").exists()


@pytest.mark.parametrize("symlink_kind", ["leaf", "parent"])
def test_bootstrap_rejects_symlinked_legacy_locator_paths_before_writing(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    legacy_config, template, output_root = _write_valid_private_inputs(tmp_path)
    legacy_payload = yaml.safe_load(legacy_config.read_text(encoding="utf-8"))
    closure = Path(legacy_payload["local_input_paths"]["closure"])
    if symlink_kind == "leaf":
        link = closure.parent / "linked" / closure.name
        link.parent.mkdir()
        link.symlink_to(closure)
    else:
        linked_parent = closure.parent.parent / "linked-inputs"
        linked_parent.symlink_to(closure.parent, target_is_directory=True)
        link = linked_parent / closure.name
    legacy_payload["local_input_paths"]["closure"] = str(link)
    _write_yaml(legacy_config, legacy_payload)

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            _gpu_probe(),
        )

    assert captured.value.category == "legacy_locator_invalid"
    assert not (output_root / "binding.json").exists()
    assert not (output_root / "local.yaml").exists()


def test_nonignored_repository_output_is_rejected_before_validation_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_config, template, _ = _write_valid_private_inputs(tmp_path)
    repository = tmp_path / "destination-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    output_root = repository / "nonignored-output"
    output_root.mkdir()
    temporary_calls: list[Path] = []
    real_mkstemp = bootstrap.tempfile.mkstemp

    def recording_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        temporary_calls.append(Path(kwargs["dir"]))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(bootstrap.tempfile, "mkstemp", recording_mkstemp)

    with pytest.raises(FullChainBootstrapError) as captured:
        materialize_full_chain_binding(
            legacy_config,
            template,
            output_root,
            output_root / "binding.json",
            output_root / "local.yaml",
            _gpu_probe(),
        )

    assert captured.value.category == "unsafe_destination"
    assert temporary_calls == []
    assert tuple(output_root.iterdir()) == ()


@pytest.mark.parametrize(
    ("mutation", "detail"),
    (
        ("missing_root", "private output root is unavailable"),
        ("file_root", "private output root is invalid"),
        ("symlink_leaf", "private output path is unsafe"),
        ("missing_parent", "private output path is unavailable"),
        ("escape", "private output escapes its root"),
        ("same_leaf", "private output paths must differ"),
    ),
)
def test_private_output_resolution_preserves_failure_categories(
    tmp_path: Path,
    mutation: str,
    detail: str,
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    binding = root / "binding.json"
    config = root / "local.yaml"
    if mutation == "missing_root":
        root = tmp_path / "missing"
        binding, config = root / "binding.json", root / "local.yaml"
    elif mutation == "file_root":
        root = tmp_path / "root-file"
        root.write_text("not-a-directory", encoding="utf-8")
        binding, config = root / "binding.json", root / "local.yaml"
    elif mutation == "symlink_leaf":
        target = root / "target.json"
        target.write_text("{}", encoding="utf-8")
        binding.symlink_to(target)
    elif mutation == "missing_parent":
        config = root / "missing" / "local.yaml"
    elif mutation == "escape":
        binding = tmp_path / "outside.json"
    elif mutation == "same_leaf":
        config = binding

    with pytest.raises(FullChainBootstrapError) as captured:
        resolve_private_outputs(
            root,
            binding,
            config,
            error_factory=FullChainBootstrapError,
        )

    assert captured.value.category == "unsafe_destination"
    assert captured.value.detail == detail
