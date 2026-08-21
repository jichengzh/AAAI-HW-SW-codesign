from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from framework.stage6.coptv2x_h800_search_v2 import (
    load_local_config,
    load_public_contract,
)
from tools.release import provision_p6_history_local_config as provision_cli
from tests.stage6.test_p6_history_normalization import _recipe_v2


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROVISIONER = REPOSITORY_ROOT / "tools/release/provision_p6_full_chain_local_config.py"
PUBLIC_CONTRACT = REPOSITORY_ROOT / "configs/execution/p6_h800_search.example.yaml"
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
SYNTHETIC_GPU_INDICES = (23, 19, 17)


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


def _template_payload(template: Path) -> dict[str, Any]:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _add_archived_duplicate_markers(root: Path) -> None:
    for marker in MARKERS.values():
        _write_executable(root / "archived-stage5-copy" / marker)


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
    encoded = json.dumps(
        contract, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "group_id": contract["group_id"],
        "model": contract["model"],
        "width": contract["width"],
        "source_status": contract["source_status"],
        "source_evidence_sha256": evidence_sha,
        "source_contract": contract,
        "source_contract_sha256": hashlib.sha256(encoded).hexdigest(),
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
    }


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
            "controller": {
                "argv": [f"documented-stage5-chain/{MARKERS['controller']}"]
            },
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
                    "CUDA_VISIBLE_DEVICES": {
                        "kind": "literal",
                        "value": ",".join(str(index) for index in SYNTHETIC_GPU_INDICES),
                    },
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


def _valid_args(tmp_path: Path, *, bad_template: bool = False) -> tuple[str, ...]:
    root = tmp_path / "private-history"
    root.mkdir()
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
    legacy = _write_yaml(
        tmp_path / "private-inputs" / "legacy.local.yaml",
        {
            "schema_version": "p6_h800_coptv2x_local_v2",
            "target": "h800",
            "asset_paths": {
                "training-data": str(root / "inputs"),
                "model-init": str(registry),
                "toolchain": str(
                    root / "documented-stage5-chain" / MARKERS["performance_plan"]
                ),
            },
            "local_input_paths": {name: str(inputs[name]) for name in LOCAL_INPUT_NAMES},
            "source_registry_step": {
                "name": "build_source_registry",
                "argv": ["legacy-registry", "{local_output_root}", "{source_registry_json}"],
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
        },
    )
    template_payload = _runner_template()
    if bad_template:
        template_payload["schema_version"] = "invalid-private-template"
    template = _write_yaml(
        tmp_path / "private-inputs" / "runner-template.yaml", template_payload
    )
    output_root = tmp_path / "private-output"
    output_root.mkdir()
    return (
        "--legacy-local-config",
        str(legacy),
        "--runner-template",
        str(template),
        "--local-output-root",
        str(output_root),
        "--binding-output",
        str(output_root / "binding.json"),
        "--config-output",
        str(output_root / "p6.local.yaml"),
    )


def _private_pair_paths(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "private-output"
    return root / "binding.json", root / "p6.local.yaml"


def _attach_expected_recipe(
    tmp_path: Path,
    args: tuple[str, ...],
    *,
    actual_recipe: dict[str, Any],
    expected_recipe: dict[str, Any],
) -> Path:
    root = tmp_path / "private-history"
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "datasets" / "coptv2x").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(exist_ok=True)
    (root / "checkpoints" / "base.ckpt").write_text("base\n", encoding="utf-8")
    (root / "configs" / "pyramid.py").write_text("config\n", encoding="utf-8")
    registry_path = root / "registry" / "candidate_source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    contract = registry["groups"][0]["source_contract"]
    contract.update(
        {
            "training_required": True,
            "training_source_kind": "selected_candidate_finetune",
            "base_checkpoint_path": str(root / "checkpoints" / "base.ckpt"),
            "dataset_root": str(root / "datasets" / "coptv2x"),
            "pyramid_config_path": str(root / "configs" / "pyramid.py"),
            "training_parameters": {
                "training_mode": "finetune_selected_width",
                "epochs": 2,
                "seed": 20260821,
                "optimizer": "adamw",
                "learning_rate": 0.0001,
                "batch_size": 1,
                "dataset_split": "trainval_coptv2x",
                "checkpoint_selection": "best_ap70",
                "freeze_policy": "pyramid_backbone_partial",
            },
            "dynamic_materialization_recipe": copy.deepcopy(actual_recipe),
        }
    )
    encoded = json.dumps(
        contract, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    registry["groups"][0]["source_contract_sha256"] = hashlib.sha256(
        encoded
    ).hexdigest()
    _write_json(registry_path, registry)
    expected_path = _write_json(
        root / "derivation" / "recipe.json", expected_recipe
    )
    legacy_path = Path(args[1])
    legacy = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))
    legacy["history_recipe_derivation_path"] = str(expected_path)
    _write_yaml(legacy_path, legacy)
    implementation = (
        root
        / "private-relocated-history-repo"
        / "bin"
        / "stage5_materialize_round_sources_v1.original.sh"
    )
    _write_executable(implementation)
    (root / "documented-stage5-chain" / MARKERS["source_materializer"]).unlink()
    return _write_yaml(
        tmp_path / "private-inputs" / "source-wrapper-profile.yaml",
        {
            "schema_version": "p6_private_source_wrapper_profile_v1",
            "wrapper_kind": "repo_cwd_exec_v1",
            "destination_relative_path": (
                f"documented-stage5-chain/{MARKERS['source_materializer']}"
            ),
            "implementation_relative_path": (
                "private-relocated-history-repo/bin/"
                "stage5_materialize_round_sources_v1.original.sh"
            ),
            "implementation_cwd_relative_path": "private-relocated-history-repo",
        },
    )


def _fake_nvidia_smi(tmp_path: Path, rows: tuple[str, ...] | None = None) -> Path:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    binary = fake_bin / "nvidia-smi"
    records = rows or tuple(
        f"{index}, GPU-private-{index}, NVIDIA H800 80GB HBM3, 0, 100"
        for index in SYNTHETIC_GPU_INDICES
    )
    binary.write_text(
        f"#!{sys.executable}\n"
        f"for row in {records!r}:\n"
        "    print(row)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return fake_bin


def test_gpu_query_argv_preserves_supplied_private_policy_order() -> None:
    """Catches a provision query that sorts the authoritative GPU policy."""
    assert provision_cli._gpu_query_argv((23, 19, 17)) == (
        "nvidia-smi",
        "--id=23,19,17",
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )


def test_gpu_probe_returns_records_in_supplied_private_policy_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a provision probe that returns nvidia-smi discovery order."""
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

    monkeypatch.setattr(provision_cli.subprocess, "run", fake_run)

    records = provision_cli.NvidiaSmiGpuProbe().snapshot((23, 19, 17))

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
        pytest.param((23, 19), id="too-few"),
        pytest.param((23, 19, 17, 11), id="too-many"),
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
        provision_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match="canonical GPU indices required"):
        provision_cli.NvidiaSmiGpuProbe().snapshot(indices)


def _run_cli(tmp_path: Path, *args: str, rows: tuple[str, ...] | None = None) -> subprocess.CompletedProcess[str]:
    fake_bin = _fake_nvidia_smi(tmp_path, rows)
    return subprocess.run(
        [sys.executable, str(PROVISIONER), *args],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"},
        text=True,
        capture_output=True,
        check=False,
    )


def _load_local_config_without_echoing_private_values(tmp_path: Path) -> Any:
    _, config_path = _private_pair_paths(tmp_path)
    return load_local_config(config_path, load_public_contract(PUBLIC_CONTRACT))


def test_cli_writes_only_ignored_private_pair(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, *_valid_args(tmp_path))

    assert result.returncode == 0
    assert result.stdout == "p6_full_chain_config_written\n"
    assert result.stderr == ""
    binding = json.loads(_private_pair_paths(tmp_path)[0].read_text(encoding="utf-8"))
    assert binding["gpu_policy"]["indices"] == list(SYNTHETIC_GPU_INDICES)
    assert (
        binding["execution_interface"]["environment"]["values"]
        ["CUDA_VISIBLE_DEVICES"]["value"]
        == "23,19,17"
    )
    assert _load_local_config_without_echoing_private_values(tmp_path).stage1_scan_step is not None


def test_cli_enforces_matching_normalized_recipe_before_writing_pair(
    tmp_path: Path,
) -> None:
    args = _valid_args(tmp_path)
    recipe = _recipe_v2()
    profile = _attach_expected_recipe(
        tmp_path,
        args,
        actual_recipe=recipe,
        expected_recipe=recipe,
    )

    result = _run_cli(
        tmp_path, *args, "--source-wrapper-profile", str(profile)
    )

    assert result.returncode == 0
    assert result.stdout == "p6_full_chain_config_written\n"
    assert result.stderr == ""


def test_cli_rejects_normalized_recipe_drift_without_pair(tmp_path: Path) -> None:
    args = _valid_args(tmp_path)
    actual = _recipe_v2()
    expected = copy.deepcopy(actual)
    expected["group_id_template"] = (
        "pyramid-drift|{stage1_width}x{stage2_width}x{stage3_width}"
    )
    profile = _attach_expected_recipe(
        tmp_path,
        args,
        actual_recipe=actual,
        expected_recipe=expected,
    )

    result = _run_cli(
        tmp_path, *args, "--source-wrapper-profile", str(profile)
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()


def test_cli_requires_source_wrapper_profile_for_recipe_v2_without_pair(
    tmp_path: Path,
) -> None:
    args = _valid_args(tmp_path)
    recipe = _recipe_v2()
    _attach_expected_recipe(
        tmp_path,
        args,
        actual_recipe=recipe,
        expected_recipe=recipe,
    )

    result = _run_cli(tmp_path, *args)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_execution_invalid\n"
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()


def test_cli_does_not_overwrite_differing_source_marker_or_write_pair(
    tmp_path: Path,
) -> None:
    args = _valid_args(tmp_path)
    recipe = _recipe_v2()
    profile = _attach_expected_recipe(
        tmp_path,
        args,
        actual_recipe=recipe,
        expected_recipe=recipe,
    )
    marker = (
        tmp_path
        / "private-history"
        / "documented-stage5-chain"
        / MARKERS["source_materializer"]
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
    marker.chmod(0o700)

    result = _run_cli(
        tmp_path, *args, "--source-wrapper-profile", str(profile)
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_execution_invalid\n"
    assert marker.read_text(encoding="utf-8") == "#!/bin/sh\nexit 91\n"
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()


def test_cli_uses_template_component_paths_when_history_has_archived_duplicates(
    tmp_path: Path,
) -> None:
    args = _valid_args(tmp_path)
    _add_archived_duplicate_markers(tmp_path / "private-history")

    result = _run_cli(tmp_path, *args)

    assert result.returncode == 0
    assert result.stdout == "p6_full_chain_config_written\n"
    assert result.stderr == ""


def test_cli_rejects_template_component_outside_history_root_without_pair(
    tmp_path: Path,
) -> None:
    args = _valid_args(tmp_path)
    template = Path(args[3])
    payload = _template_payload(template)
    payload["execution_interface"]["controller"]["argv"] = [
        str(_write_executable(tmp_path / "outside" / MARKERS["controller"]))
    ]
    _write_yaml(template, payload)

    result = _run_cli(tmp_path, *args)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_root_ambiguous\n"
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()


def test_cli_rejects_template_component_role_mismatch_without_pair(
    tmp_path: Path,
) -> None:
    args = _valid_args(tmp_path)
    template = Path(args[3])
    payload = _template_payload(template)
    payload["execution_interface"]["execution_chain"][2]["argv"][0] = (
        f"documented-stage5-chain/{MARKERS['finalizer']}"
    )
    _write_yaml(template, payload)

    result = _run_cli(tmp_path, *args)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_root_ambiguous\n"
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()


def test_cli_redacts_private_template_failure_and_writes_nothing(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, *_valid_args(tmp_path, bad_template=True))

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "execution_interface_unavailable\n"
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()


def test_cli_rejects_relative_paths_without_echoing_them(tmp_path: Path) -> None:
    result = _run_cli(
        tmp_path,
        "--legacy-local-config",
        "relative-private.yaml",
        "--runner-template",
        "/private/template.yaml",
        "--local-output-root",
        "/private/output",
        "--binding-output",
        "/private/output/binding.json",
        "--config-output",
        "/private/output/config.yaml",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert "relative-private.yaml" not in result.stderr


def test_cli_redacts_help_request_as_argument_error(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--help")

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"


def test_cli_rejects_invalid_gpu_probe_without_pair(tmp_path: Path) -> None:
    result = _run_cli(
        tmp_path,
        *_valid_args(tmp_path),
        rows=tuple(
            f"{index}, GPU-synthetic-{index}, NVIDIA H800 80GB HBM3, 0, 100"
            for index in (*SYNTHETIC_GPU_INDICES, SYNTHETIC_GPU_INDICES[-1])
        ),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "gpu_admission\n"
    assert "GPU-private" not in result.stderr
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()


def test_cli_rejects_nonunique_history_root_without_pair(tmp_path: Path) -> None:
    args = _valid_args(tmp_path)
    legacy_path = Path(args[1])
    payload = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))
    second_root = tmp_path / "second-private-history"
    second_root.mkdir()
    subprocess.run(["git", "init", "-q", str(second_root)], check=True)
    closure = _write_json(second_root / "inputs" / "closure.json", {"fixture": "two"})
    payload["local_input_paths"]["closure"] = str(closure)
    _write_yaml(legacy_path, payload)

    result = _run_cli(tmp_path, *args)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_root_ambiguous\n"
    assert not _private_pair_paths(tmp_path)[0].exists()
    assert not _private_pair_paths(tmp_path)[1].exists()


def test_cli_rejects_nonignored_repository_pair_before_writing(tmp_path: Path) -> None:
    args = list(_valid_args(tmp_path))
    repository_output = REPOSITORY_ROOT / "tests"
    binding_path = repository_output / f".p6-full-chain-{tmp_path.name}.json"
    config_path = repository_output / f".p6-full-chain-{tmp_path.name}.yaml"
    args[5] = str(repository_output)
    args[7] = str(binding_path)
    args[9] = str(config_path)

    result = _run_cli(tmp_path, *args)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "unsafe_destination\n"
    assert not binding_path.exists()
    assert not config_path.exists()
