from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import pytest

from framework.stage6.p6_history_binding_v1 import (
    GpuRecord,
    P6HistoryBindingError,
    discover_history_binding,
    public_binding_projection,
    write_private_binding_pair,
)


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


class SequenceProbe:
    def __init__(self, *snapshots: Iterable[GpuRecord]) -> None:
        self._snapshots = tuple(tuple(snapshot) for snapshot in snapshots)
        self._calls = 0

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        del indices
        position = min(self._calls, len(self._snapshots) - 1)
        self._calls += 1
        return self._snapshots[position]


def _gpu_records(
    *,
    indices: tuple[int, ...] = (5, 6, 7),
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
        (commands / name).write_text("synthetic private executable\n", encoding="utf-8")
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
            },
            {
                "stage": "quantization",
                "argv": [
                    str(commands / "quantize-private"),
                    "{measurement_request}",
                    "{round_output_root}",
                ],
            },
            {
                "stage": "performance",
                "argv": [
                    str(chain_root / MARKERS["performance_plan"]),
                    "{measurement_request}",
                    "{round_output_root}",
                ],
            },
            {
                "stage": "ap",
                "argv": [
                    str(commands / "measure-ap-private"),
                    "{measurement_request}",
                    "{round_output_root}",
                ],
            },
            {
                "stage": "finalization",
                "argv": [
                    str(chain_root / MARKERS["finalizer"]),
                    "{measurement_request}",
                    "{round_output_root}",
                ],
            },
        ],
        "environment": {
            "values": {
                "CUDA_VISIBLE_DEVICES": "5,6,7",
                "P6_HISTORY_RUN_MODE": "private-bound",
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
                "stage_key": "stage",
                "status_key": "status",
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
            "receipt": {
                "path_template": "private-runs/{round_id}/receipt.json",
                "format": "json",
            },
            "finalization_barrier": {
                "path_template": "private-runs/{round_id}/barrier.json",
                "format": "json",
            },
            "validation_fields": {
                "request_sha256": "measurement_request_sha256",
                "row_sha256": "row_sha256",
                "source_evidence_sha256": "source_evidence_sha256",
            },
        },
    }


def _history_root(tmp_path: Path) -> Path:
    root = tmp_path / "history"
    marker_root = root / "documented-stage5-chain"
    marker_root.mkdir(parents=True)
    for marker in MARKERS.values():
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
    assert binding["gpu_policy"]["indices"] == [5, 6, 7]
    assert set(binding["gpu_policy"]["uuid_by_index"]) == {"5", "6", "7"}
    assert binding["source_contract_template"]["schema_version"] == (
        "stage5_source_contract_v1"
    )
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
    assert "private_root" not in projected
    projected_strings = tuple(_walk_strings(projected))
    assert not any(str(history_root) in value for value in projected_strings)
    assert not any("GPU-fixture" in value for value in projected_strings)
    assert not any("private-bound" in value for value in projected_strings)
    assert projected == {
        "schema_version": "p6_history_binding_public_v1",
        "binding_schema_version": "p6_history_binding_v1",
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
        '"CUDA_VISIBLE_DEVICES": "5,6,7",',
        '"CUDA_VISIBLE_DEVICES": "5,6,7", '
        '"CUDA_VISIBLE_DEVICES": "private-duplicate",',
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
        manifest["actual_feedback"]["finalization_barrier"]["path_template"] = (
            "../barrier.json"
        )
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
        _gpu_records(indices=(5, 6)),
        _gpu_records(indices=(4, 5, 6, 7)),
        _gpu_records(model_name="NVIDIA A100-SXM4-80GB"),
        (
            GpuRecord(5, "duplicate", "NVIDIA H800 80GB HBM3", 0.0),
            GpuRecord(6, "duplicate", "NVIDIA H800 80GB HBM3", 0.0),
            GpuRecord(7, "unique", "NVIDIA H800 80GB HBM3", 0.0),
        ),
        (
            GpuRecord(5, "GPU-5", "NVIDIA H800 80GB HBM3", 0.0),
            GpuRecord(6, "", "NVIDIA H800 80GB HBM3", 0.0),
            GpuRecord(7, "GPU-7", "NVIDIA H800 80GB HBM3", 0.0),
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
        ("GPU-5", "0"),
        ("GPU-5", "idle"),
    ],
)
def test_malformed_gpu_record_fields_fail_with_stable_admission_category(
    tmp_path: Path,
    uuid: Any,
    occupancy: Any,
) -> None:
    records = list(_gpu_records())
    records[0] = GpuRecord(5, uuid, "NVIDIA H800 80GB HBM3", occupancy)

    with _expect_category("gpu_admission"):
        discover_history_binding(_history_root(tmp_path), _probe(records))


def test_deceptive_h800_substring_model_fails_admission(tmp_path: Path) -> None:
    records = _gpu_records(model_name="NOT-H800-COMPATIBLE")

    with _expect_category("gpu_admission"):
        discover_history_binding(_history_root(tmp_path), _probe(records))


def test_gpu_second_snapshot_uuid_drift_fails_closed(tmp_path: Path) -> None:
    drifted = list(_gpu_records())
    drifted[2] = GpuRecord(7, "GPU-drifted", "NVIDIA H800 80GB HBM3", 0.0)

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


def test_independent_replace_failure_exposes_documented_pair_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_root = repo_root / "private"
    output_root.mkdir(parents=True)
    binding_path = output_root / "binding.json"
    config_path = output_root / "config.json"
    binding_path.write_text(json.dumps({"old": "binding"}), encoding="utf-8")
    config_path.write_text(json.dumps({"old": "config"}), encoding="utf-8")
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

    assert json.loads(binding_path.read_text(encoding="utf-8")) == binding
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"old": "config"}
    assert not tuple(output_root.glob("*.tmp"))


def test_discovery_builders_do_not_mutate_registry_or_gpu_records(tmp_path: Path) -> None:
    history_root = _history_root(tmp_path)
    registry_path = history_root / "registry" / "candidate_source_registry.json"
    registry_before = copy.deepcopy(json.loads(registry_path.read_text(encoding="utf-8")))
    records = _gpu_records()

    discover_history_binding(history_root, _probe(records))

    assert json.loads(registry_path.read_text(encoding="utf-8")) == registry_before
    assert records == _gpu_records()
