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
from framework.stage6.p6_full_chain_bootstrap_v1 import materialize_full_chain_binding
from framework.stage6.p6_history_measurement_v1 import (
    plan_validated_history_round_paths,
)
from framework.stage6.p6_history_normalization_v1 import normalize_history_inputs
from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
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
    _migrate_normalized_training_registry,
    _public_contract,
    _write_normalized_history_source_map,
    _write_normalized_source_wrapper_profile,
    _write_runner_template,
    _write_yaml,
)


def _build_preflight_inputs(root: Path) -> dict[str, Path]:
    source_map = _write_normalized_history_source_map(root)
    private_root = root / "normalized-private-root"
    normalized = normalize_history_inputs(
        source_map, Path(str(source_map["history_root"])), private_root
    )
    _migrate_normalized_training_registry(normalized["registry"], private_root)
    runner_template = _write_runner_template(private_root / "runner-template.yaml")
    wrapper_profile = _write_normalized_source_wrapper_profile(
        root / "source-wrapper-profile.yaml", private_root
    )
    local_output_root = private_root / "provisioned"
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
    )
    return {
        "public_contract_path": _write_yaml(root / "public.yaml", _public_contract()),
        "local_config_path": local_config,
        "private_binding_path": binding,
        "runner_template_path": runner_template,
        "source_wrapper_profile_path": wrapper_profile,
    }


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
        binding["source_contract_template"].pop("training_required")
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
        return
    if mutation == "old_binding_false_training_required":
        binding["source_contract_template"]["training_required"] = False
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
        return
    if mutation == "missing_static_training_field":
        binding["source_contract_template"].pop("base_checkpoint_path")
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
        return
    if mutation == "wrapper_not_self_contained":
        profile = yaml.safe_load(
            inputs["source_wrapper_profile_path"].read_text(encoding="utf-8")
        )
        profile["wrapper_kind"] = "PRIVATE-PREFLIGHT-PATH"
        _write_yaml(inputs["source_wrapper_profile_path"], profile)
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
        recipe = binding["source_contract_template"][
            "dynamic_materialization_recipe"
        ]
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


@pytest.mark.parametrize(
    "mutation",
    [
        "static_mode_rejected",
        "old_binding_missing_training_required",
        "old_binding_false_training_required",
        "missing_static_training_field",
        "wrapper_not_self_contained",
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
        _local_output_root(preflight_inputs)
        / "PRIVATE-PREFLIGHT-PATH-GPU-private-preflight.json"
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
        ]
    )
    output = capsys.readouterr()

    assert result == 1
    assert output.out == ""
    assert output.err == "preflight_failed\n"
    assert "PRIVATE-PREFLIGHT-PATH" not in output.out + output.err
    assert "GPU-private-preflight" not in output.out + output.err
