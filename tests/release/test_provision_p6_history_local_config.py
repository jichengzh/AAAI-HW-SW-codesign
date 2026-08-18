from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import yaml

from framework.stage6.coptv2x_h800_search_v2 import (
    load_local_config,
    load_public_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROVISIONER = REPOSITORY_ROOT / "tools/release/provision_p6_history_local_config.py"
PUBLIC_CONTRACT = REPOSITORY_ROOT / "configs/execution/p6_h800_search.example.yaml"
MARKERS = (
    "stage5_task_round_controller_v3.sh",
    "stage5_materialize_round_sources_v1.sh",
    "stage5_build_performance_plan_v2.py",
    "stage5_finalize_feedback_v2.py",
)
LOCAL_INPUT_NAMES = (
    "gold176_rows",
    "gold176_graph_features",
    "capability_profiles",
    "closure",
)
LOCAL_CONFIG_KEYS = {
    "schema_version",
    "target",
    "asset_paths",
    "local_input_paths",
    "candidate_source_mode",
    "stage2_search_space_path",
    "source_registry_step",
    "measurement_step",
    "local_output_root",
}


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


def _stage1_manifest() -> dict[str, Any]:
    return {
        "schema": "stage1_partition_manifest_demo_v1",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {"name": "h800_tvm_demo"},
        "view_b1_search_groups": [
            {
                "search_group_id": "pyramid_group",
                "bucket": "pyramid_backbone",
                "widths": [16, 32, 48, 64],
                "round_to": 16,
                "int8_buildable_align": 64,
                "max_rate": 0.75,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": ["pyramid_group"],
            }
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": ["pyramid_group"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _history_root(tmp_path: Path) -> Path:
    root = tmp_path / "history"
    marker_root = root / "documented-stage5-chain"
    marker_root.mkdir(parents=True)
    for marker in MARKERS:
        (marker_root / marker).write_text("synthetic marker\n", encoding="utf-8")
    _write_json(
        root / "registry" / "candidate_source_registry.json",
        {
            "schema_version": "stage5_candidate_source_registry_v1",
            "groups": [_source_group()],
        },
    )
    for name in LOCAL_INPUT_NAMES:
        _write_json(root / "inputs" / f"{name}.json", {"fixture": name})
    _write_json(root / "stage2" / "pyramid_partition.json", _stage1_manifest())
    return root


def _fake_nvidia_smi(tmp_path: Path, *, drift: bool = False) -> Path:
    binary = tmp_path / "fake-bin" / "nvidia-smi"
    binary.parent.mkdir()
    drift_source = """
counter_path = Path(__file__).with_suffix('.calls')
calls = int(counter_path.read_text()) if counter_path.exists() else 0
counter_path.write_text(str(calls + 1))
drift = calls > 0
""" if drift else "drift = False\n"
    binary.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        f"{drift_source}"
        "for index in (5, 6, 7):\n"
        "    suffix = '-drifted' if drift and index == 7 else ''\n"
        "    print(f'{index}, GPU-fixture-{index}{suffix}, NVIDIA H800 80GB HBM3, 0, 100')\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary.parent


def _tracked_snapshot() -> dict[str, bytes]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    names = tuple(filter(None, completed.stdout.decode().split("\0")))
    return {name: (REPOSITORY_ROOT / name).read_bytes() for name in names}


def _run_cli(
    history_root: Path,
    local_output_root: Path,
    binding_output: Path,
    config_output: Path,
    fake_bin: Path,
    *extra: str,
    provisioner: Path = PROVISIONER,
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    return subprocess.run(
        [
            sys.executable,
            str(provisioner),
            "--history-root",
            str(history_root),
            "--local-output-root",
            str(local_output_root),
            "--binding-output",
            str(binding_output),
            "--config-output",
            str(config_output),
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_provisions_loader_compatible_framework_config_without_tracked_leak(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "history-binding.json"
    config_path = private_root / "p6.local.yaml"
    tracked_before = _tracked_snapshot()

    result = _run_cli(
        history_root,
        private_root,
        binding_path,
        config_path,
        _fake_nvidia_smi(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "provisioned history-binding.json p6.local.yaml\n"
    assert result.stderr == ""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = load_public_contract(PUBLIC_CONTRACT)
    loaded = load_local_config(config_path, contract)
    assert set(payload) == LOCAL_CONFIG_KEYS
    assert payload["schema_version"] == "p6_h800_coptv2x_local_v2"
    assert payload["target"] == "h800"
    assert set(payload["asset_paths"]) == {
        "training-data",
        "model-init",
        "toolchain",
    }
    assert payload["asset_paths"] == {
        "training-data": str(history_root / "inputs"),
        "model-init": str(
            history_root / "registry" / "candidate_source_registry.json"
        ),
        "toolchain": str(
            history_root
            / "documented-stage5-chain"
            / "stage5_build_performance_plan_v2.py"
        ),
    }
    assert set(payload["local_input_paths"]) == set(LOCAL_INPUT_NAMES)
    assert payload["candidate_source_mode"] == "framework_stage2_search_space"
    assert Path(payload["stage2_search_space_path"]).name == "pyramid_partition.json"
    assert loaded.source_registry_step.name == "build_source_registry"
    assert loaded.measurement_step.name == "measure_batch"
    source_argv = payload["source_registry_step"]["argv"]
    measurement_argv = payload["measurement_step"]["argv"]
    assert source_argv[0] == str(Path(sys.executable).resolve())
    assert source_argv[1] == str(
        REPOSITORY_ROOT / "tools/release/build_p6_history_registry.py"
    )
    assert set(source_argv) >= {
        "{local_output_root}",
        "{source_registry_json}",
        "{pyramid_candidate_plan}",
    }
    assert measurement_argv[0] == str(Path(sys.executable).resolve())
    assert measurement_argv[1] == str(
        REPOSITORY_ROOT / "tools/release/measure_p6_history_batch.py"
    )
    assert set(measurement_argv) >= {
        "{measurement_request}",
        "{feedback_json}",
        "{round_output_root}",
    }
    assert all("{" not in token and "}" not in token for token in source_argv if token not in {
        "{local_output_root}", "{source_registry_json}", "{pyramid_candidate_plan}"
    })
    assert all("{" not in token and "}" not in token for token in measurement_argv if token not in {
        "{measurement_request}", "{feedback_json}", "{round_output_root}"
    })
    generated_text = config_path.read_text(encoding="utf-8")
    assert "stage5_task_round_controller_v3.sh" not in generated_text
    assert "stage5_materialize_round_sources_v1.sh" not in generated_text
    assert _tracked_snapshot() == tracked_before


def test_cli_rejects_ambiguous_history_without_writing(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    duplicate = history_root / "duplicate" / MARKERS[0]
    duplicate.parent.mkdir()
    duplicate.write_text("duplicate\n", encoding="utf-8")
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "binding.json"
    config_path = private_root / "config.yaml"

    result = _run_cli(
        history_root,
        private_root,
        binding_path,
        config_path,
        _fake_nvidia_smi(tmp_path),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "component_discovery\n"
    assert not binding_path.exists()
    assert not config_path.exists()


def test_cli_requires_one_valid_stage2_json_without_writing(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    _write_json(history_root / "stage2" / "duplicate.json", _stage1_manifest())
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "binding.json"
    config_path = private_root / "config.yaml"

    result = _run_cli(
        history_root,
        private_root,
        binding_path,
        config_path,
        _fake_nvidia_smi(tmp_path),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "stage2_search_space_unavailable\n"
    assert not binding_path.exists()
    assert not config_path.exists()


def test_cli_rejects_gpu_drift_without_private_details_or_output(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "binding.json"
    config_path = private_root / "config.yaml"

    result = _run_cli(
        history_root,
        private_root,
        binding_path,
        config_path,
        _fake_nvidia_smi(tmp_path, drift=True),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "gpu_drift\n"
    assert "GPU-fixture" not in result.stderr
    assert str(history_root) not in result.stderr
    assert not binding_path.exists()
    assert not config_path.exists()


def test_cli_rejects_repository_internal_unignored_output_root(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "binding.json"
    config_path = private_root / "config.yaml"

    result = _run_cli(
        history_root,
        REPOSITORY_ROOT / "tests",
        binding_path,
        config_path,
        _fake_nvidia_smi(tmp_path),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "unsafe_destination\n"
    assert not binding_path.exists()
    assert not config_path.exists()


def test_cli_rejects_nonprivate_config_and_preserves_external_binding(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "binding.json"
    config_path = (
        REPOSITORY_ROOT / "tests" / f".p6-provision-nonprivate-{tmp_path.name}.yaml"
    )
    assert not config_path.exists()

    result = _run_cli(
        history_root,
        private_root,
        binding_path,
        config_path,
        _fake_nvidia_smi(tmp_path),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "unsafe_destination\n"
    assert not binding_path.exists()
    assert not config_path.exists()


def test_cli_rejects_nonprivate_binding_and_preserves_external_config(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = (
        REPOSITORY_ROOT / "tests" / f".p6-provision-nonprivate-{tmp_path.name}.json"
    )
    config_path = private_root / "config.yaml"
    assert not binding_path.exists()

    result = _run_cli(
        history_root,
        private_root,
        binding_path,
        config_path,
        _fake_nvidia_smi(tmp_path),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "unsafe_destination\n"
    assert not binding_path.exists()
    assert not config_path.exists()


def test_cli_rejects_relative_paths_and_gpu_override_arguments(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    fake_bin = _fake_nvidia_smi(tmp_path)

    relative = _run_cli(
        history_root,
        private_root,
        private_root / "binding.json",
        Path("relative-config.yaml"),
        fake_bin,
    )
    override = _run_cli(
        history_root,
        private_root,
        private_root / "binding.json",
        private_root / "config.yaml",
        fake_bin,
        "--gpu-indices",
        "0,1,2",
    )

    assert relative.returncode == 2
    assert relative.stdout == ""
    assert relative.stderr == "argument_error\n"
    assert override.returncode == 2
    assert override.stdout == ""
    assert override.stderr == "argument_error\n"
    assert not (private_root / "binding.json").exists()
    assert not (private_root / "config.yaml").exists()


def test_cli_rejects_invalid_tracked_public_contract_in_isolated_copy(
    tmp_path: Path,
) -> None:
    copied_repository = tmp_path / "isolated-repository"
    copied_provisioner = copied_repository / "tools/release" / PROVISIONER.name
    copied_provisioner.parent.mkdir(parents=True)
    shutil.copy2(PROVISIONER, copied_provisioner)
    shutil.copytree(REPOSITORY_ROOT / "framework", copied_repository / "framework")
    copied_contract = copied_repository / "configs/execution" / PUBLIC_CONTRACT.name
    copied_contract.parent.mkdir(parents=True)
    copied_contract.write_text("schema_version: invalid\n", encoding="utf-8")
    history_root = _history_root(tmp_path / "fixture")
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "binding.json"
    config_path = private_root / "config.yaml"

    result = _run_cli(
        history_root,
        private_root,
        binding_path,
        config_path,
        _fake_nvidia_smi(tmp_path),
        provisioner=copied_provisioner,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert not binding_path.exists()
    assert not config_path.exists()
