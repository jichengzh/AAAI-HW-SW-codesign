"""Acceptance coverage for the zero-process P6 preflight boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

import pytest
import yaml

from framework.stage6.coptv2x_h800_search_v2 import P6CoptV2XExecutionError
from framework.stage6 import p6_full_chain_bootstrap_v1 as bootstrap
from framework.stage6.p6_full_chain_bootstrap_v1 import materialize_full_chain_binding
from framework.stage6.p6_history_binding_v1 import FORMAL_TVM_ENV_KEYS, GpuRecord
from framework.stage6.p6_history_measurement_v1 import (
    plan_validated_history_round_paths,
)
from framework.stage6.p6_history_normalization_v1 import normalize_history_inputs
from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
from framework.stage6.p6_tvm_runtime_authority_v1 import (
    canonical_tvm_support_tree_sha256,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (
    GROUP_RECEIPT_RELATIVE_ROOT,
    RUN_CONTEXT_RELATIVE_PATH,
    RUN_METADATA_RELATIVE_ROOT,
)
from tools.release import preflight_p6_materializer_training_bridge as preflight_module
from tools.release.preflight_p6_materializer_training_bridge import (
    P6MaterializerPreflightReport,
    main,
    preflight_materializer_training_bridge,
)
from tests.stage6.test_coptv2x_h800_search import (
    _OfflineGpuProbe,
    _public_contract,
    _write_yaml,
)
from tests.stage6.test_p6_history_normalization import (
    _as_v2_procedural,
    valid_private_source_map,
)
from tests.stage6.test_p6_post_source_adapter_profile import v5_private_source_map


RTX_GPU_INDICES = (31, 29, 23, 19)


class _RtxGpuProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        self.calls.append(indices)
        return tuple(
            GpuRecord(
                index=index,
                uuid=f"GPU-rtx-preflight-{index}",
                model_name="NVIDIA GeForce RTX 4090",
                occupancy=0.0,
            )
            for index in indices
        )


def _build_preflight_inputs(root: Path) -> dict[str, Path]:
    source_map, source_runner = _as_v2_procedural(valid_private_source_map(root), root)
    private_root = root / "normalized-private-root"
    normalized = normalize_history_inputs(
        source_map,
        Path(str(source_map["history_root"])),
        private_root,
        runner_template_path=source_runner,
    )
    runner_template = normalized["runner_template"]
    wrapper_profile = normalized["source_wrapper_profile"]
    external_training = normalized["external_training_binding"]
    local_output_root = root / "provisioned"
    local_output_root.mkdir()
    binding = local_output_root / "binding.json"
    local_config = local_output_root / "local.yaml"
    materialize_full_chain_binding(
        normalized["legacy"],
        runner_template,
        local_output_root,
        binding,
        local_config,
        _OfflineGpuProbe(),
        source_wrapper_profile=wrapper_profile,
        external_training_binding=external_training,
    )
    return {
        "public_contract_path": _write_yaml(root / "public.yaml", _public_contract()),
        "local_config_path": local_config,
        "private_binding_path": binding,
        "runner_template_path": runner_template,
        "source_wrapper_profile_path": wrapper_profile,
        "external_training_binding_path": external_training,
    }


def _build_v5_preflight_inputs(root: Path) -> dict[str, Path]:
    source_map, source_runner = v5_private_source_map(root)
    private_root = root / "normalized-private-root"
    normalized = normalize_history_inputs(
        source_map,
        Path(str(source_map["history_root"])),
        private_root,
        runner_template_path=source_runner,
    )
    local_output_root = root / "provisioned"
    local_output_root.mkdir()
    binding = local_output_root / "binding.json"
    local_config = local_output_root / "local.yaml"
    materialize_full_chain_binding(
        normalized["legacy"],
        normalized["runner_template"],
        local_output_root,
        binding,
        local_config,
        _OfflineGpuProbe(),
        source_wrapper_profile=normalized["source_wrapper_profile"],
        external_training_binding=normalized["external_training_binding"],
        post_source_adapter_profile=normalized["post_source_adapter_profile"],
    )
    return {
        "public_contract_path": _write_yaml(root / "public.yaml", _public_contract()),
        "local_config_path": local_config,
        "private_binding_path": binding,
        "runner_template_path": normalized["runner_template"],
        "source_wrapper_profile_path": normalized["source_wrapper_profile"],
        "external_training_binding_path": normalized["external_training_binding"],
        "post_source_adapter_profile_path": normalized["post_source_adapter_profile"],
    }


def _build_rtx_preflight_inputs(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_formal_runtime: bool = True,
) -> tuple[dict[str, Path], _RtxGpuProbe]:
    source_map, source_runner = v5_private_source_map(root)
    source_map["hardware_profile"] = "rtx4090"
    if with_formal_runtime:
        runtime_root = root / "approved-runtime"
        runtime_python = _write_executable(runtime_root / "bin" / "python")
        runtime_site = runtime_root / "site"
        runtime_site.mkdir(parents=True)
        runtime_nvlibs = runtime_root / "nvlibs.json"
        runtime_nvlibs.write_text("{}\n", encoding="utf-8")
        support = next(
            item
            for item in source_map["execution_code_closure"]["roots"]
            if item["closure_id"] == "tvm-support"
        )
        source_runner_payload = yaml.safe_load(source_runner.read_text(encoding="utf-8"))
        source_runner_payload["execution_interface"]["environment"]["values"].update(
            {
                "P6_TVM_PYTHON": {
                    "kind": "external_executable",
                    "value": str(runtime_python),
                },
                "P6_TVM_SITE": {
                    "kind": "external_directory",
                    "value": str(runtime_site),
                },
                "P6_TVM_NVLIBS_FILE": {
                    "kind": "external_file",
                    "value": str(runtime_nvlibs),
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
        _write_yaml(source_runner, source_runner_payload)
    private_root = root / "rtx-normalized-private-root"
    normalized = normalize_history_inputs(
        source_map,
        Path(str(source_map["history_root"])),
        private_root,
        runner_template_path=source_runner,
    )
    legacy = yaml.safe_load(normalized["legacy"].read_text(encoding="utf-8"))
    legacy.update(
        {
            "schema_version": "p6_coptv2x_local_v3",
            "target": "rtx4090",
            "hardware_profile": "rtx4090",
        }
    )
    _write_yaml(normalized["legacy"], legacy)
    runner = yaml.safe_load(normalized["runner_template"].read_text(encoding="utf-8"))
    runner["execution_interface"]["environment"]["values"][
        "CUDA_VISIBLE_DEVICES"
    ]["value"] = ",".join(str(index) for index in RTX_GPU_INDICES)
    _write_yaml(normalized["runner_template"], runner)
    public_contract = _write_yaml(
        root / "p6_rtx4090_search.example.yaml",
        _public_contract(
            schema_version="p6_coptv2x_search_contract_v3",
            search_id="p6-pyramid-rtx4090-tvm",
            target="rtx4090",
            hardware_profile="rtx4090",
            configuration_label="p6-pyramid-rtx4090-tvm",
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "PUBLIC_CONTRACT_PATH",
        root / "p6_{hardware_profile}_search.example.yaml",
    )
    local_output_root = root / "rtx-provisioned"
    local_output_root.mkdir()
    binding = local_output_root / "binding.json"
    local_config = local_output_root / "local.yaml"
    probe = _RtxGpuProbe()
    materialize_full_chain_binding(
        normalized["legacy"],
        normalized["runner_template"],
        local_output_root,
        binding,
        local_config,
        probe,
        source_wrapper_profile=normalized["source_wrapper_profile"],
        external_training_binding=normalized["external_training_binding"],
        post_source_adapter_profile=normalized["post_source_adapter_profile"],
    )
    return (
        {
            "public_contract_path": public_contract,
            "local_config_path": local_config,
            "private_binding_path": binding,
            "runner_template_path": normalized["runner_template"],
            "source_wrapper_profile_path": normalized["source_wrapper_profile"],
            "external_training_binding_path": normalized["external_training_binding"],
            "post_source_adapter_profile_path": normalized[
                "post_source_adapter_profile"
            ],
        },
        probe,
    )


def _write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o700)
    return path


def _post_source_private_root(inputs: dict[str, Path]) -> Path:
    return Path(_read_json(inputs["private_binding_path"])["private_root"])


def _alternate_post_source_entrypoint(root: Path, stage: str) -> str:
    relative = {
        "quantization": "private-runner/bin/alternate-quantize-private",
        "performance": "archived-stage5-copy/stage5_build_performance_plan_v2.py",
        "ap": "private-runner/bin/alternate-measure-ap-private",
        "finalization": "archived-stage5-copy/stage5_finalize_feedback_v2.py",
    }[stage]
    _write_executable(root / relative)
    return relative


def _replace_runner_post_source_entrypoint(
    inputs: dict[str, Path],
    *,
    stage: str,
) -> None:
    runner_path = inputs["runner_template_path"]
    payload = yaml.safe_load(runner_path.read_text(encoding="utf-8"))
    for entry in payload["execution_interface"]["execution_chain"]:
        if entry["stage"] == stage:
            entry["argv"][0] = _alternate_post_source_entrypoint(
                _post_source_private_root(inputs), stage
            )
            break
    _write_yaml(runner_path, payload)


@pytest.fixture(scope="module")
def _preflight_template(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    workspace = tmp_path_factory.mktemp("p6-preflight")
    live = workspace / "live"
    live.mkdir()
    inputs = _build_preflight_inputs(live)
    snapshot = workspace / "snapshot"
    shutil.copytree(live, snapshot, symlinks=True)
    yield {"live": live, "snapshot": snapshot, "inputs": inputs}


@pytest.fixture
def preflight_inputs(_preflight_template: dict[str, Any]) -> dict[str, Path]:
    live = _preflight_template["live"]
    shutil.rmtree(live)
    shutil.copytree(_preflight_template["snapshot"], live, symlinks=True)
    return dict(_preflight_template["inputs"])


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _local_output_root(inputs: dict[str, Path]) -> Path:
    payload = yaml.safe_load(inputs["local_config_path"].read_text(encoding="utf-8"))
    return Path(payload["local_output_root"])


def _apply_preflight_mutation(inputs: dict[str, Path], mutation: str) -> None:
    local_path = inputs["local_config_path"]
    binding_path = inputs["private_binding_path"]
    local = yaml.safe_load(local_path.read_text(encoding="utf-8"))
    binding = _read_json(binding_path)
    output_root = Path(local["local_output_root"])
    private_root = Path(binding["private_root"])

    if mutation == "static_mode_rejected":
        local["candidate_source_mode"] = "coptv2x_static_registry"
        local["stage2_search_space_path"] = None
        local.pop("stage1_scan_step")
        argv = local["source_registry_step"]["argv"]
        plan_flag = argv.index("--pyramid-candidate-plan")
        del argv[plan_flag : plan_flag + 2]
        _write_yaml(local_path, local)
        return
    if mutation == "old_binding_missing_training_required":
        binding["source_contract_template"]["external_training_binding"].pop("training_required")
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
        return
    if mutation == "old_binding_false_training_required":
        binding["source_contract_template"]["external_training_binding"]["training_required"] = (
            False
        )
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
        return
    if mutation == "missing_static_training_field":
        binding["source_contract_template"]["external_training_binding"].pop("base_checkpoint_path")
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
        return
    if mutation == "wrapper_not_self_contained":
        profile = yaml.safe_load(inputs["source_wrapper_profile_path"].read_text(encoding="utf-8"))
        profile["wrapper_kind"] = "PRIVATE-PREFLIGHT-PATH"
        _write_yaml(inputs["source_wrapper_profile_path"], profile)
        return
    if mutation == "runner_path_argv_tail":
        runner = yaml.safe_load(inputs["runner_template_path"].read_text(encoding="utf-8"))
        runner["execution_interface"]["controller"]["argv"].append(
            "/tmp/undeclared-source-tree/config.py"
        )
        _write_yaml(inputs["runner_template_path"], runner)
        return
    if mutation == "output_root_symlink":
        external = output_root.parent / "external-preflight-inputs"
        external.mkdir()
        moved_binding = external / "binding-PRIVATE-PREFLIGHT-PATH.json"
        moved_local = external / "local-GPU-private-preflight.yaml"
        shutil.copy2(binding_path, moved_binding)
        shutil.copy2(local_path, moved_local)
        shutil.rmtree(output_root)
        symlink_target = output_root.parent / "symlink-target"
        symlink_target.mkdir()
        output_root.symlink_to(symlink_target, target_is_directory=True)
        inputs["private_binding_path"] = moved_binding
        inputs["local_config_path"] = moved_local
        return

    paths = {
        "metadata_namespace_preexists": output_root / RUN_METADATA_RELATIVE_ROOT,
        "run_context_preexists": output_root / RUN_CONTEXT_RELATIVE_PATH,
        "receipt_namespace_preexists": output_root / GROUP_RECEIPT_RELATIVE_ROOT,
        "one_group_receipt_preexists": (
            output_root / GROUP_RECEIPT_RELATIVE_ROOT / f"{'0' * 64}.json"
        ),
        "materialized_root_preexists": output_root / "materialized",
        "source_registry_preexists": output_root / "source_registry.json",
        "candidate_plan_preexists": output_root / "pyramid_candidate_plan.json",
        "stage1_manifest_preexists": Path(local["stage2_search_space_path"]),
        "state_preexists": output_root / "state.json",
        "public_round_root_preexists": output_root / "round-00",
    }
    if mutation in SHARED_SOURCE_PATH_KEYS:
        recipe = binding["source_contract_template"]["dynamic_materialization_recipe"]
        width_values = dict(
            zip(
                recipe["stage_width_fields"],
                binding["source_contract_template"]["width"],
                strict=True,
            )
        )
        artifact_id = recipe["artifact_id_template"].format(**width_values)
        relative_path = recipe["shared_source_path_templates"][mutation].format(
            artifact_id=artifact_id
        )
        path = output_root / relative_path
    elif mutation in {
        "private_round_root_preexists",
        "round_request_preexists",
        "task_state_preexists",
        "actual_feedback_preexists",
        "actual_completion_receipt_preexists",
        "barrier_preexists",
    }:
        planned = plan_validated_history_round_paths(
            binding["execution_interface"], private_root, 0
        )
        key = {
            "private_round_root_preexists": "round_root",
            "round_request_preexists": "measurement_request",
            "task_state_preexists": "task_state",
            "actual_feedback_preexists": "actual_feedback",
            "actual_completion_receipt_preexists": "actual_receipt",
            "barrier_preexists": "finalization_barrier",
        }[mutation]
        path = planned[key]
    else:
        path = paths[mutation]

    directory_mutations = {
        "metadata_namespace_preexists",
        "receipt_namespace_preexists",
        "materialized_root_preexists",
        "public_round_root_preexists",
        "private_round_root_preexists",
        "checkpoint_dir",
        "calibration_root",
        "trt_calibration_dir",
    }
    if mutation in directory_mutations:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PRIVATE-PREFLIGHT-PATH GPU-private-preflight\n", encoding="utf-8")


def test_preflight_report_has_only_public_release_fields() -> None:
    assert tuple(P6MaterializerPreflightReport.__dataclass_fields__) == (
        "schema_version",
        "status",
        "validated_round_count",
        "wrapper_marker",
        "training_required",
        "historical_process_launch_count",
        "gpu_probe_count",
    )


def test_preflight_accepts_regenerated_training_binding_without_process_or_gpu(
    preflight_inputs: dict[str, Path],
) -> None:
    report = preflight_materializer_training_bridge(**preflight_inputs)

    assert report.status == "accepted"
    assert report.validated_round_count == 4
    assert report.wrapper_marker == "stage5_materialize_round_sources_v1.sh"
    assert report.training_required is True
    assert report.historical_process_launch_count == 0
    assert report.gpu_probe_count == 0


def test_rtx_hardware_profile_preflight_accepts_four_planned_rounds_without_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, provision_probe = _build_rtx_preflight_inputs(tmp_path, monkeypatch)

    report = preflight_materializer_training_bridge(**inputs)

    assert provision_probe.calls == [RTX_GPU_INDICES, RTX_GPU_INDICES]
    assert report == P6MaterializerPreflightReport(
        schema_version="p6_materializer_training_bridge_preflight_v1",
        status="accepted",
        validated_round_count=4,
        wrapper_marker="stage5_materialize_round_sources_v1.sh",
        training_required=True,
        historical_process_launch_count=0,
        gpu_probe_count=0,
    )


def test_rtx_preflight_rejects_missing_formal_tvm_runtime_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = _build_rtx_preflight_inputs(
        tmp_path,
        monkeypatch,
        with_formal_runtime=False,
    )

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)


@pytest.mark.parametrize("missing_key", FORMAL_TVM_ENV_KEYS)
def test_rtx_preflight_rejects_each_missing_formal_runtime_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_key: str,
) -> None:
    inputs, _ = _build_rtx_preflight_inputs(tmp_path, monkeypatch)
    binding = _read_json(inputs["private_binding_path"])
    binding["execution_interface"]["environment"]["values"].pop(missing_key)
    inputs["private_binding_path"].write_text(json.dumps(binding), encoding="utf-8")

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)


@pytest.mark.parametrize("mutation", ("symlink", "non_executable", "valid_drift"))
def test_rtx_preflight_rejects_invalid_or_drifted_formal_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    inputs, _ = _build_rtx_preflight_inputs(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate-runtime" / "python"
    alternate.parent.mkdir()
    alternate.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    if mutation == "non_executable":
        alternate.chmod(0o600)
    else:
        alternate.chmod(0o700)
    selected = alternate
    if mutation == "symlink":
        selected = alternate.with_name("python-link")
        selected.symlink_to(alternate)
    binding = _read_json(inputs["private_binding_path"])
    binding["execution_interface"]["environment"]["values"]["P6_TVM_PYTHON"][
        "value"
    ] = str(selected)
    inputs["private_binding_path"].write_text(json.dumps(binding), encoding="utf-8")

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)


def test_rtx_preflight_rejects_alternate_self_consistent_support_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = _build_rtx_preflight_inputs(tmp_path, monkeypatch)
    private_root = _post_source_private_root(inputs)
    alternate_root = private_root / "alternate-support"
    alternate_root.mkdir()
    (alternate_root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    alternate_digest = canonical_tvm_support_tree_sha256(alternate_root)
    binding = _read_json(inputs["private_binding_path"])
    runner = yaml.safe_load(inputs["runner_template_path"].read_text(encoding="utf-8"))
    for payload in (binding["execution_interface"], runner["execution_interface"]):
        values = payload["environment"]["values"]
        values["P6_TVM_SUPPORT_ROOT"]["value"] = str(alternate_root)
        values["P6_TVM_SUPPORT_ROOT_SHA256"]["value"] = alternate_digest
    inputs["private_binding_path"].write_text(json.dumps(binding), encoding="utf-8")
    _write_yaml(inputs["runner_template_path"], runner)

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)


def test_rtx_hardware_profile_binding_mismatch_stops_before_profile_or_runner_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = _build_rtx_preflight_inputs(tmp_path, monkeypatch)
    binding = _read_json(inputs["private_binding_path"])
    binding["target"]["hardware"] = "h800"
    binding["gpu_policy"]["hardware_profile"] = "h800"
    inputs["private_binding_path"].write_text(json.dumps(binding), encoding="utf-8")
    post_source_calls = 0
    runner_calls = 0
    real_post_source_validation = preflight_module._validate_post_source_profile

    def counted_post_source_validation(*args: Any, **kwargs: Any) -> None:
        nonlocal post_source_calls
        post_source_calls += 1
        real_post_source_validation(*args, **kwargs)

    def counted_runner_validation(*_args: Any, **_kwargs: Any) -> None:
        nonlocal runner_calls
        runner_calls += 1

    monkeypatch.setattr(
        preflight_module,
        "_validate_post_source_profile",
        counted_post_source_validation,
    )
    monkeypatch.setattr(
        preflight_module,
        "validate_pre_provision_runner_template",
        counted_runner_validation,
    )

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)

    assert post_source_calls == 0
    assert runner_calls == 0


def test_preflight_requires_profile_argument_for_v5_recipe_v2(
    tmp_path: Path,
) -> None:
    inputs = _build_v5_preflight_inputs(tmp_path)
    post_source_profile = inputs.pop("post_source_adapter_profile_path")

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)

    report = preflight_materializer_training_bridge(
        **inputs,
        post_source_adapter_profile_path=post_source_profile,
    )
    assert report.status == "accepted"


def test_preflight_rejects_v1_profile_before_runner_role_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _build_v5_preflight_inputs(tmp_path)
    payload = yaml.safe_load(
        inputs["post_source_adapter_profile_path"].read_text(encoding="utf-8")
    )
    payload["schema_version"] = "p6_post_source_adapter_profile_v1"
    payload.pop("adapter_python")
    payload.pop("adapter_dependency_root_relative_path")
    v1_profile = _post_source_private_root(inputs) / "legacy-v1-profile.yaml"
    v1_profile.write_text(yaml.safe_dump(payload), encoding="utf-8")
    runner_validation_calls = 0
    wrapper_validation_calls = 0

    def record_runner_validation(*_args: Any, **_kwargs: Any) -> None:
        nonlocal runner_validation_calls
        runner_validation_calls += 1

    def record_wrapper_validation(*_args: Any, **_kwargs: Any) -> dict[str, Path]:
        nonlocal wrapper_validation_calls
        wrapper_validation_calls += 1
        return {}

    monkeypatch.setattr(
        preflight_module,
        "validate_post_source_wrapper_runner_binding",
        record_runner_validation,
    )
    monkeypatch.setattr(
        preflight_module,
        "validate_post_source_adapter_wrappers",
        record_wrapper_validation,
    )

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(
            **{
                **inputs,
                "post_source_adapter_profile_path": v1_profile,
            }
        )

    assert runner_validation_calls == 0
    assert wrapper_validation_calls == 0


def test_preflight_rejects_v2_profile_before_wrapper_or_runner_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _build_v5_preflight_inputs(tmp_path)
    payload = yaml.safe_load(
        inputs["post_source_adapter_profile_path"].read_text(encoding="utf-8")
    )
    payload["schema_version"] = "p6_post_source_adapter_profile_v2"
    payload.pop("adapter_dependency_root_relative_path")
    v2_profile = _post_source_private_root(inputs) / "legacy-v2-profile.yaml"
    v2_profile.write_text(yaml.safe_dump(payload), encoding="utf-8")
    inputs["post_source_adapter_profile_path"] = v2_profile
    wrapper_validation_calls = 0
    runner_validation_calls = 0

    def record_wrapper_validation(*_args: Any, **_kwargs: Any) -> dict[str, Path]:
        nonlocal wrapper_validation_calls
        wrapper_validation_calls += 1
        return {}

    def record_runner_validation(*_args: Any, **_kwargs: Any) -> None:
        nonlocal runner_validation_calls
        runner_validation_calls += 1

    monkeypatch.setattr(
        preflight_module,
        "validate_post_source_adapter_wrappers",
        record_wrapper_validation,
    )
    monkeypatch.setattr(
        preflight_module,
        "validate_post_source_wrapper_runner_binding",
        record_runner_validation,
    )

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)

    assert wrapper_validation_calls == 0
    assert runner_validation_calls == 0


@pytest.mark.parametrize(
    "stage",
    ["quantization", "performance", "ap", "finalization"],
)
def test_preflight_rejects_v5_runner_not_bound_to_generated_post_source_wrappers(
    tmp_path: Path,
    stage: str,
) -> None:
    inputs = _build_v5_preflight_inputs(tmp_path)
    _replace_runner_post_source_entrypoint(inputs, stage=stage)

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)


def test_preflight_rejects_tampered_generated_post_source_wrapper_bytes(
    tmp_path: Path,
) -> None:
    inputs = _build_v5_preflight_inputs(tmp_path)
    wrapper = _post_source_private_root(inputs) / "private-runner/bin/quantize-private"
    _write_executable(wrapper, "#!/usr/bin/python3\nraise SystemExit(0)\n")

    with pytest.raises(P6CoptV2XExecutionError):
        preflight_materializer_training_bridge(**inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        "static_mode_rejected",
        "old_binding_missing_training_required",
        "old_binding_false_training_required",
        "missing_static_training_field",
        "wrapper_not_self_contained",
        "runner_path_argv_tail",
        "metadata_namespace_preexists",
        "run_context_preexists",
        "receipt_namespace_preexists",
        "one_group_receipt_preexists",
        "materialized_root_preexists",
        *SHARED_SOURCE_PATH_KEYS,
        "source_registry_preexists",
        "candidate_plan_preexists",
        "stage1_manifest_preexists",
        "state_preexists",
        "public_round_root_preexists",
        "private_round_root_preexists",
        "round_request_preexists",
        "task_state_preexists",
        "actual_feedback_preexists",
        "actual_completion_receipt_preexists",
        "barrier_preexists",
        "output_root_symlink",
    ],
)
def test_preflight_rejects_stale_or_incomplete_state_without_process_or_gpu(
    preflight_inputs: dict[str, Path], mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_preflight_mutation(preflight_inputs, mutation)
    process_calls: list[tuple[str, ...]] = []
    real_run = subprocess.run

    def record_validation_process(
        argv: list[str], *args: Any, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        process_calls.append(tuple(argv))
        assert argv[0] == "git"
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_validation_process)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        preflight_materializer_training_bridge(**preflight_inputs)

    assert captured.value.failure_code in {
        "history_execution_invalid",
        "unsafe_destination",
        "source_registry_invalid",
    }
    assert all(call[0] == "git" for call in process_calls)
    assert "PRIVATE-PREFLIGHT-PATH" not in str(captured.value)
    assert "GPU-private-preflight" not in str(captured.value)


def test_preflight_source_has_no_historical_or_gpu_execution_call() -> None:
    tree = ast.parse(Path(preflight_module.__file__).read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_names.isdisjoint(
        {"_run_step", "run_history_measurement_batch", "run_source_invocations"}
    )
    assert called_attributes.isdisjoint({"snapshot", "run", "Popen"})


def test_preflight_cli_emits_only_allowlisted_public_fields(
    preflight_inputs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "--contract",
            str(preflight_inputs["public_contract_path"]),
            "--local-config",
            str(preflight_inputs["local_config_path"]),
            "--binding",
            str(preflight_inputs["private_binding_path"]),
            "--runner-template",
            str(preflight_inputs["runner_template_path"]),
            "--source-wrapper-profile",
            str(preflight_inputs["source_wrapper_profile_path"]),
            "--external-training-binding",
            str(preflight_inputs["external_training_binding_path"]),
        ]
    )
    output = capsys.readouterr()

    assert result == 0
    assert output.err == ""
    assert set(json.loads(output.out)) == set(P6MaterializerPreflightReport.__dataclass_fields__)


def test_preflight_cli_accepts_v5_post_source_adapter_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = _build_v5_preflight_inputs(tmp_path)
    result = main(
        [
            "--contract",
            str(inputs["public_contract_path"]),
            "--local-config",
            str(inputs["local_config_path"]),
            "--binding",
            str(inputs["private_binding_path"]),
            "--runner-template",
            str(inputs["runner_template_path"]),
            "--source-wrapper-profile",
            str(inputs["source_wrapper_profile_path"]),
            "--external-training-binding",
            str(inputs["external_training_binding_path"]),
            "--post-source-adapter-profile",
            str(inputs["post_source_adapter_profile_path"]),
        ]
    )
    output = capsys.readouterr()

    assert result == 0
    assert output.err == ""
    assert set(json.loads(output.out)) == set(P6MaterializerPreflightReport.__dataclass_fields__)


def test_preflight_cli_redacts_private_path_and_gpu_tokens(
    preflight_inputs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    preflight_inputs["private_binding_path"] = (
        _local_output_root(preflight_inputs) / "PRIVATE-PREFLIGHT-PATH-GPU-private-preflight.json"
    )
    result = main(
        [
            "--contract",
            str(preflight_inputs["public_contract_path"]),
            "--local-config",
            str(preflight_inputs["local_config_path"]),
            "--binding",
            str(preflight_inputs["private_binding_path"]),
            "--runner-template",
            str(preflight_inputs["runner_template_path"]),
            "--source-wrapper-profile",
            str(preflight_inputs["source_wrapper_profile_path"]),
            "--external-training-binding",
            str(preflight_inputs["external_training_binding_path"]),
        ]
    )
    output = capsys.readouterr()

    assert result == 1
    assert output.out == ""
    assert output.err == "preflight_failed\n"
    assert "PRIVATE-PREFLIGHT-PATH" not in output.out + output.err
    assert "GPU-private-preflight" not in output.out + output.err
