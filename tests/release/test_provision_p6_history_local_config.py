from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

from tools.release import provision_p6_history_local_config as legacy_provisioner


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
SYNTHETIC_GPU_INDICES = (107, 103, 101)
RTX_GPU_INDICES = (109, 107, 103, 101)


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


def _rtx_public_contract() -> dict[str, Any]:
    return {
        "schema_version": "p6_coptv2x_search_contract_v3",
        "search_id": "p6-pyramid-rtx4090-tvm",
        "target": "rtx4090",
        "hardware_profile": "rtx4090",
        "target_model": "pyramid",
        "execution_backend": "tvm_auto",
        "seed": 73,
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
        "configuration_label": "p6-pyramid-rtx4090-tvm",
        "candidate_space_label": "coptv2x-pyramid-width-grid-v1",
        "metric_names": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _history_runner_manifest(root: Path) -> dict[str, Any]:
    commands = root / "private-runner" / "bin"
    commands.mkdir(parents=True, exist_ok=True)
    for name in ("quantize-private", "measure-ap-private", "activate-private"):
        command = commands / name
        command.write_text("synthetic private executable\n", encoding="utf-8")
        command.chmod(0o700)
    chain_root = root / "documented-stage5-chain"
    return {
        "schema_version": "p6_history_runner_interface_v1",
        "controller": {"argv": [str(chain_root / MARKERS[0])]},
        "execution_chain": [
            {
                "stage": "source_materialization",
                "argv": [
                    str(chain_root / MARKERS[1]),
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
                    str(chain_root / MARKERS[2]),
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
                    str(chain_root / MARKERS[3]),
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


def _history_root(
    tmp_path: Path,
    *,
    include_execution_interface: bool = True,
) -> Path:
    root = tmp_path / "history"
    marker_root = root / "documented-stage5-chain"
    marker_root.mkdir(parents=True)
    for marker in MARKERS:
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
    _write_json(root / "stage2" / "pyramid_partition.json", _stage1_manifest())
    if include_execution_interface:
        _write_json(
            root / "private-runner" / "p6-history-runner-interface.json",
            _history_runner_manifest(root),
        )
    return root


def _fake_nvidia_smi(
    tmp_path: Path,
    *,
    drift: bool = False,
    indices: tuple[int, ...] = SYNTHETIC_GPU_INDICES,
) -> Path:
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
        f"for index in {indices!r}:\n"
        f"    suffix = '-drifted' if drift and index == {SYNTHETIC_GPU_INDICES[-1]} else ''\n"
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


def test_cli_rejects_obsolete_framework_provisioning_without_output_or_tracked_leak(
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

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "stage1_scan_unavailable\n"
    assert not binding_path.exists()
    assert not config_path.exists()
    assert _tracked_snapshot() == tracked_before


def test_legacy_cli_rejects_rtx_before_any_gpu_snapshot(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    history_root = _history_root(tmp_path)
    interface_path = history_root / "private-runner/p6-history-runner-interface.json"
    interface = json.loads(interface_path.read_text(encoding="utf-8"))
    interface["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"] = ",".join(
        str(index) for index in RTX_GPU_INDICES
    )
    _write_json(interface_path, interface)
    public_contract = _write_json(tmp_path / "rtx-public.yaml", _rtx_public_contract())
    monkeypatch.setattr(legacy_provisioner, "PUBLIC_CONTRACT_PATH", public_contract)
    probe_calls: list[tuple[int, ...]] = []

    def fake_snapshot(
        _self: Any,
        indices: tuple[int, ...],
    ) -> tuple[legacy_provisioner.GpuRecord, ...]:
        probe_calls.append(indices)
        return tuple(
            legacy_provisioner.GpuRecord(
                index=index,
                uuid=f"GPU-rtx-fixture-{index}",
                model_name="NVIDIA GeForce RTX 4090",
                occupancy=0.0,
            )
            for index in indices
        )

    monkeypatch.setattr(
        legacy_provisioner.NvidiaSmiGpuProbe,
        "snapshot",
        fake_snapshot,
    )
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "history-binding.json"
    config_path = private_root / "p6.local.yaml"

    exit_code = legacy_provisioner.main(
        [
            "--history-root",
            str(history_root),
            "--local-output-root",
            str(private_root),
            "--binding-output",
            str(binding_path),
            "--config-output",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "legacy_h800_only\n"
    assert probe_calls == []
    assert not binding_path.exists()
    assert not config_path.exists()


def test_legacy_cli_help_names_h800_diagnostic_and_only_rtx_provision_boundary() -> None:
    result = subprocess.run(
        [sys.executable, str(PROVISIONER), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "legacy H800-only" in result.stdout
    assert "non-materializing diagnostic" in result.stdout
    assert "provision_p6_full_chain_local_config.py" in result.stdout
    assert "RTX" in result.stdout


def test_obsolete_framework_route_does_not_reach_loader_without_stage1_step(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    history_root = _history_root(tmp_path)
    private_root = tmp_path / "private-output"
    private_root.mkdir()
    binding_path = private_root / "history-binding.json"
    config_path = private_root / "p6.local.yaml"

    def unreachable_loader(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("obsolete route reached load_local_config")

    monkeypatch.setattr(legacy_provisioner, "load_local_config", unreachable_loader)
    monkeypatch.setattr(
        legacy_provisioner.NvidiaSmiGpuProbe,
        "snapshot",
        lambda self, indices: tuple(
            legacy_provisioner.GpuRecord(
                index, f"fixture-{index}", "NVIDIA H800", 0.0
            )
            for index in SYNTHETIC_GPU_INDICES
        ),
    )

    exit_code = legacy_provisioner.main(
        [
            "--history-root",
            str(history_root),
            "--local-output-root",
            str(private_root),
            "--binding-output",
            str(binding_path),
            "--config-output",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "stage1_scan_unavailable\n"
    assert not binding_path.exists()
    assert not config_path.exists()


def test_cli_rejects_history_without_execution_interface_and_writes_nothing(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path, include_execution_interface=False)
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
    assert result.stderr == "execution_interface\n"
    assert not binding_path.exists()
    assert not config_path.exists()


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


def test_cli_rejects_obsolete_route_before_legacy_stage2_discovery(
    tmp_path: Path,
) -> None:
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
    assert result.stderr == "stage1_scan_unavailable\n"
    assert not binding_path.exists()
    assert not config_path.exists()


def test_cli_rejects_obsolete_route_before_legacy_stage2_loader_failure(
    tmp_path: Path,
) -> None:
    history_root = _history_root(tmp_path)
    malformed_manifest = _stage1_manifest()
    malformed_manifest["view_b2_quant_units"] = [None]
    _write_json(
        history_root / "stage2" / "pyramid_partition.json",
        malformed_manifest,
    )
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
    assert result.stderr == "stage1_scan_unavailable\n"
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


def test_cli_rejects_duplicate_gpu_indices_without_writing(tmp_path: Path) -> None:
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
        _fake_nvidia_smi(
            tmp_path,
            indices=(*SYNTHETIC_GPU_INDICES, SYNTHETIC_GPU_INDICES[-1]),
        ),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "gpu_admission\n"
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
