from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
from framework.stage6.p6_history_source_materialization_v1 import (
    P6HistorySourceMaterializationError,
    build_source_invocations,
    project_source_materialization_request,
    run_source_invocations,
    validate_projected_training_marker_pairs,
)


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shared_paths(group_slug: str) -> dict[str, str]:
    return {
        key: f"/private/synthetic/materialized/{group_slug}/{key}"
        for key in SHARED_SOURCE_PATH_KEYS
    }


def _source_contract(width: list[int]) -> dict[str, Any]:
    group_id = f"pyramid|{width[0]}x{width[1]}x{width[2]}"
    evidence = _sha({"evidence": group_id})
    return {
        "schema_version": "stage5_source_contract_v1",
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "artifact_id": f"pyramid-{width[0]}-{width[1]}-{width[2]}",
        "source_status": "ready",
        "source_evidence_sha256": evidence,
        "materialization_scope": "synthetic_fixture",
        "stage_widths": {
            "stage1_width": width[0],
            "stage2_width": width[1],
            "stage3_width": width[2],
        },
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
        "base_checkpoint_path": "/private/synthetic/base/model.ckpt",
        "dataset_root": "/private/synthetic/dataset",
        "pyramid_config_path": "/private/synthetic/configs/pyramid.py",
        "training_parameters": {
            "training_mode": "finetune_selected_width",
            "epochs": 7,
            "seed": 20260821,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "batch_size": 1,
            "dataset_split": "trainval_coptv2x",
            "checkpoint_selection": "best_ap70",
            "freeze_policy": "pyramid_backbone_partial",
        },
        "shared_source_paths": _shared_paths("-".join(map(str, width))),
    }


def _row(width: list[int], q_mode: str, index: int, task_sha: str) -> dict[str, Any]:
    contract = _source_contract(width)
    group_id = contract["group_id"]
    evidence = contract["source_evidence_sha256"]
    row_id = f"{group_id}|q={q_mode}|candidate={index}"
    return {
        "schema_version": "stage5_candidate_row_v2",
        "task_id": "P6-H800-PYRAMID",
        "task_sha256": task_sha,
        "row_id": row_id,
        "manifest_job_id": row_id,
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "width_schema": ["w0", "w1", "w2"],
        "structure_widths": {"w0": width[0], "w1": width[1], "w2": width[2]},
        "genome": [*width, q_mode],
        "strategy_id": f"q={q_mode}",
        "q_mode": q_mode,
        "hardware_id": "h800",
        "capability_profile_id": "h800-tvm-auto",
        "capability_digest": _sha({"profile": "h800-tvm-auto"}),
        "dispatch_key": "tvm_auto",
        "source_status": "ready",
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
        "source_contract": contract,
        "source_contract_sha256": _sha(contract),
        "source_evidence_sha256": evidence,
        "graph_features": {"group_id": group_id, "model": "pyramid", "width": width},
    }


def _request() -> dict[str, Any]:
    task_sha = _sha({"task": "P6-H800-PYRAMID"})
    rows = [
        _row([17, 31, 63], "fp16", 0, task_sha),
        _row([17, 31, 63], "int8", 1, task_sha),
        _row([23, 47, 95], "fp16", 2, task_sha),
        _row([29, 53, 101], "int8", 3, task_sha),
    ]
    body = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "P6-H800-PYRAMID",
        "task_sha256": task_sha,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**body, "measurement_request_sha256": _sha(body)}


def _rehash_request(request: dict[str, Any]) -> None:
    request["row_sha256"] = {
        row["row_id"]: _sha(row) for row in request["rows"]
    }
    body = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request["measurement_request_sha256"] = _sha(body)


def _rehash_row_contract(request: dict[str, Any], row_index: int) -> None:
    row = request["rows"][row_index]
    row["source_contract_sha256"] = _sha(row["source_contract"])
    _rehash_request(request)


def _gpu_policy() -> dict[str, Any]:
    indices = (101, 103, 107)
    return {
        "indices": list(indices),
        "uuid_by_index": {
            str(index): f"GPU-synthetic-{index}" for index in indices
        },
        "model": "h800",
        "maximum_occupancy": 0.05,
    }


def _executable(path: Path) -> Path:
    path.write_text("synthetic executable\n", encoding="utf-8")
    path.chmod(0o700)
    return path


@dataclass(frozen=True)
class _Result:
    returncode: int = 0


class _RecordingRunner:
    def __init__(self, *, returncode: int = 0, raises: bool = False) -> None:
        self.returncode = returncode
        self.raises = raises
        self.calls: list[
            tuple[tuple[str, ...], Path, Mapping[str, str], bool]
        ] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        self.calls.append((tuple(argv), cwd, dict(env), shell))
        if self.raises:
            raise RuntimeError("PRIVATE path and runner diagnostics")
        return _Result(self.returncode)


def test_projection_flattens_shared_paths_and_recomputes_only_existing_hashes() -> None:
    """Catches mutation, q-mode path splitting, static-field loss, and stale hashes."""
    request = _request()
    original = copy.deepcopy(request)

    projected = project_source_materialization_request(request)

    assert request == original
    assert projected.request is not request
    assert projected.request["rows"][0] is not request["rows"][0]
    assert projected.ordered_group_ids == (
        "pyramid|17x31x63",
        "pyramid|23x47x95",
        "pyramid|29x53x101",
    )
    first, second = projected.request["rows"][:2]
    assert first["q_mode"] == "fp16"
    assert second["q_mode"] == "int8"
    assert first["strategy_id"] != second["strategy_id"]
    assert first["genome"] != second["genome"]
    for row in projected.request["rows"]:
        contract = row["source_contract"]
        assert "shared_source_paths" not in contract
        assert contract["base_checkpoint_path"] == "/private/synthetic/base/model.ckpt"
        assert contract["training_required"] is True
        assert contract["training_source_kind"] == "selected_candidate_finetune"
        assert contract["pyramid_config_path"] == (
            "/private/synthetic/configs/pyramid.py"
        )
        assert set(contract["training_parameters"]) == {
            "training_mode",
            "epochs",
            "seed",
            "optimizer",
            "learning_rate",
            "batch_size",
            "dataset_split",
            "checkpoint_selection",
            "freeze_policy",
        }
        assert row["source_contract_sha256"] == _sha(contract)
        assert projected.request["row_sha256"][row["row_id"]] == _sha(row)
    assert {
        key: first["source_contract"][key] for key in SHARED_SOURCE_PATH_KEYS
    } == {
        key: second["source_contract"][key] for key in SHARED_SOURCE_PATH_KEYS
    }
    request_body = {
        key: value
        for key, value in projected.request.items()
        if key != "measurement_request_sha256"
    }
    assert projected.request["measurement_request_sha256"] == _sha(request_body)
    assert set(projected.request) == set(original)
    assert set(projected.request["rows"][0]) == set(original["rows"][0])
    projected.request["rows"][0]["source_contract"]["training_parameters"][
        "epochs"
    ] = 99
    assert request["rows"][0]["source_contract"]["training_parameters"][
        "epochs"
    ] == 7


def test_projection_preserves_complete_training_contract_and_rehashes() -> None:
    """Catches dropping validated static inputs or retaining legacy aliases."""
    request = _request()
    for row in request["rows"]:
        row["source_contract"]["training_path"] = "/private/untrusted/training"
        row["source_contract_sha256"] = _sha(row["source_contract"])
    _rehash_request(request)
    original = copy.deepcopy(request)

    projected = project_source_materialization_request(request)

    assert request == original
    assert request["measurement_request_sha256"] != projected.request[
        "measurement_request_sha256"
    ]
    for row in projected.request["rows"]:
        contract = row["source_contract"]
        assert contract["training_required"] is True
        assert contract["training_source_kind"] == "selected_candidate_finetune"
        assert contract["base_checkpoint_path"] == (
            "/private/synthetic/base/model.ckpt"
        )
        assert contract["dataset_root"] == "/private/synthetic/dataset"
        assert contract["pyramid_config_path"] == (
            "/private/synthetic/configs/pyramid.py"
        )
        assert "shared_source_paths" not in contract
        assert "training_path" not in contract
        assert row["source_contract_sha256"] == _sha(contract)
        assert projected.request["row_sha256"][row["row_id"]] == _sha(row)


@pytest.mark.parametrize(
    "mutation",
    [
        "same_group_training_parameters_drift",
        "same_group_base_checkpoint_drift",
        "same_group_dataset_drift",
        "same_group_pyramid_config_drift",
        "same_group_training_required_false",
        "missing_training_done_marker",
        "missing_source_done_marker",
        "shared_path_collision_across_groups",
        "mixed_legacy_and_recipe_v2_rows",
        "stale_source_contract_hash",
    ],
)
def test_projection_rejects_untrusted_training_or_shared_path_drift(
    mutation: str,
) -> None:
    """Catches accepting self-consistent but untrusted recipe-v2 row contracts."""
    request = _request()
    first = request["rows"][0]["source_contract"]
    second = request["rows"][1]["source_contract"]
    third = request["rows"][2]["source_contract"]
    rehash_index = 1
    if mutation == "same_group_training_parameters_drift":
        second["training_parameters"]["epochs"] = 8
    elif mutation == "same_group_base_checkpoint_drift":
        second["base_checkpoint_path"] += ".drift"
    elif mutation == "same_group_dataset_drift":
        second["dataset_root"] += "-drift"
    elif mutation == "same_group_pyramid_config_drift":
        second["pyramid_config_path"] += ".drift"
    elif mutation == "same_group_training_required_false":
        second["training_required"] = False
    elif mutation == "missing_training_done_marker":
        first["shared_source_paths"].pop("training_done_marker")
        rehash_index = 0
    elif mutation == "missing_source_done_marker":
        first["shared_source_paths"].pop("source_done_marker")
        rehash_index = 0
    elif mutation == "shared_path_collision_across_groups":
        third["shared_source_paths"]["source_done_marker"] = first[
            "shared_source_paths"
        ]["training_done_marker"]
        rehash_index = 2
    elif mutation == "mixed_legacy_and_recipe_v2_rows":
        first.pop("shared_source_paths")
        rehash_index = 0
    else:
        first["training_parameters"]["epochs"] = 8
        _rehash_request(request)
        rehash_index = -1
    if rehash_index >= 0:
        _rehash_row_contract(request, rehash_index)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "training_required_false",
        "missing_training_source_kind",
        "checkpoint_reuse_source_kind",
        "missing_base_checkpoint_path",
        "missing_dataset_root",
        "missing_pyramid_config_path",
        "missing_training_parameter",
        "relative_static_training_path",
    ],
)
def test_projection_rejects_uniformly_invalid_training_contract(
    mutation: str,
) -> None:
    """Catches invalid training fields agreeing across every q-mode and group."""
    request = _request()
    for row in request["rows"]:
        contract = row["source_contract"]
        if mutation == "training_required_false":
            contract["training_required"] = False
        elif mutation == "missing_training_source_kind":
            contract.pop("training_source_kind")
        elif mutation == "checkpoint_reuse_source_kind":
            contract["training_source_kind"] = "checkpoint_already_exists"
        elif mutation == "missing_base_checkpoint_path":
            contract.pop("base_checkpoint_path")
        elif mutation == "missing_dataset_root":
            contract.pop("dataset_root")
        elif mutation == "missing_pyramid_config_path":
            contract.pop("pyramid_config_path")
        elif mutation == "missing_training_parameter":
            contract["training_parameters"].pop("freeze_policy")
        else:
            contract["dataset_root"] = "relative/dataset"
        row["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"


def test_projected_training_marker_pairs_are_unique_and_canonically_ordered() -> None:
    """Catches row-order leakage or duplicate q-mode marker pairs in runtime gates."""
    projected = project_source_materialization_request(_request())

    pairs = validate_projected_training_marker_pairs(projected.request)

    assert pairs == (
        (
            Path(
                "/private/synthetic/materialized/17-31-63/training_done_marker"
            ),
            Path("/private/synthetic/materialized/17-31-63/source_done_marker"),
        ),
        (
            Path(
                "/private/synthetic/materialized/23-47-95/training_done_marker"
            ),
            Path("/private/synthetic/materialized/23-47-95/source_done_marker"),
        ),
        (
            Path(
                "/private/synthetic/materialized/29-53-101/training_done_marker"
            ),
            Path("/private/synthetic/materialized/29-53-101/source_done_marker"),
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_training_marker",
        "relative_source_marker",
        "malformed_training_marker",
        "self_alias",
        "same_group_pair_drift",
        "cross_group_marker_collision",
    ],
)
def test_projected_training_marker_pairs_reject_malformed_or_colliding_paths(
    mutation: str,
) -> None:
    """Catches unsafe marker admission before any source/downstream process."""
    request = dict(project_source_materialization_request(_request()).request)
    first = request["rows"][0]["source_contract"]
    second = request["rows"][1]["source_contract"]
    third = request["rows"][2]["source_contract"]
    row_index = 0
    if mutation == "missing_training_marker":
        first.pop("training_done_marker")
    elif mutation == "relative_source_marker":
        first["source_done_marker"] = "relative/source.done"
    elif mutation == "malformed_training_marker":
        first["training_done_marker"] += "\nprivate"
    elif mutation == "self_alias":
        first["source_done_marker"] = first["training_done_marker"]
    elif mutation == "same_group_pair_drift":
        second["source_done_marker"] += ".drift"
        row_index = 1
    else:
        third["training_done_marker"] = first["source_done_marker"]
        row_index = 2
    _rehash_row_contract(request, row_index)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        validate_projected_training_marker_pairs(request)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"


@pytest.mark.parametrize(
    "mutation",
    ["training_required_false", "cross_group_flat_output_collision"],
)
def test_projection_rejects_invalid_already_projected_request(mutation: str) -> None:
    """Catches the idempotent projection path bypassing recipe-v2 validation."""
    request = dict(project_source_materialization_request(_request()).request)
    if mutation == "training_required_false":
        for row in request["rows"]:
            row["source_contract"]["training_required"] = False
            row["source_contract_sha256"] = _sha(row["source_contract"])
    else:
        first = request["rows"][0]["source_contract"]
        third = request["rows"][2]["source_contract"]
        third["checkpoint_path"] = first["checkpoint_path"]
        request["rows"][2]["source_contract_sha256"] = _sha(third)
    _rehash_request(request)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"


def test_projection_sanitizes_and_rehashes_self_consistent_already_flat_request() -> None:
    """Catches flat recipe-v2 requests retaining a second source truth."""
    request = dict(project_source_materialization_request(_request()).request)
    legacy_values = {
        "training_path": "/private/untrusted/training",
        "calibration_path": "/private/untrusted/calibration",
        "dynamic_materialization_recipe": {"schema_version": "untrusted-recipe"},
        "materialization_outputs_by_q_mode": {
            "fp16": {"checkpoint_path": "/private/untrusted/checkpoint"}
        },
    }
    for row in request["rows"]:
        row["source_contract"].update(copy.deepcopy(legacy_values))
        row["source_contract_sha256"] = _sha(row["source_contract"])
    _rehash_request(request)
    original = copy.deepcopy(request)

    projected = project_source_materialization_request(request)

    expected_contract_keys = {
        "schema_version",
        "group_id",
        "model",
        "width",
        "artifact_id",
        "source_status",
        "source_evidence_sha256",
        "materialization_scope",
        "stage_widths",
        "training_required",
        "training_source_kind",
        "base_checkpoint_path",
        "dataset_root",
        "pyramid_config_path",
        "training_parameters",
        "checkpoint_path",
        "checkpoint_dir",
        "config_path",
        "training_done_marker",
        "onnx_path",
        "onnx_report_path",
        "calibration_root",
        "calibration_npz",
        "calibration_summary",
        "trt_calibration_dir",
        "source_done_marker",
    }
    assert request == original
    assert projected.request["measurement_request_sha256"] != request[
        "measurement_request_sha256"
    ]
    for row in projected.request["rows"]:
        assert set(row["source_contract"]) == expected_contract_keys
        assert row["source_contract_sha256"] == _sha(row["source_contract"])
        assert projected.request["row_sha256"][row["row_id"]] == _sha(row)
    projected_body = {
        key: value
        for key, value in projected.request.items()
        if key != "measurement_request_sha256"
    }
    assert projected.request["measurement_request_sha256"] == _sha(projected_body)
    assert project_source_materialization_request(projected.request).request == (
        projected.request
    )


def test_projection_rejects_lexical_alias_before_cross_group_path_ownership() -> None:
    """Catches raw-string ownership accepting two spellings of one output path."""
    request = _request()
    alias = (
        "/private/synthetic/materialized/17-31-63/../23-47-95/checkpoint_path"
    )
    for row_index in (0, 1):
        contract = request["rows"][row_index]["source_contract"]
        contract["shared_source_paths"]["checkpoint_path"] = alias
        request["rows"][row_index]["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
    assert alias not in str(captured.value)


@pytest.mark.parametrize(
    "path_key",
    [*SHARED_SOURCE_PATH_KEYS, "base_checkpoint_path", "dataset_root", "pyramid_config_path"],
)
def test_projection_rejects_parent_components_in_every_recipe_v2_path(
    path_key: str,
) -> None:
    """Catches any static/shared path bypassing lexical component validation."""
    request = _request()
    contract = request["rows"][2]["source_contract"]
    malformed = f"/private/synthetic/materialized/23-47-95/../escape/{path_key}"
    if path_key in SHARED_SOURCE_PATH_KEYS:
        contract["shared_source_paths"][path_key] = malformed
    else:
        contract[path_key] = malformed
    request["rows"][2]["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
    assert malformed not in str(captured.value)


def test_source_invocations_dedupe_canonical_group_order_and_round_robin_policy(
    tmp_path: Path,
) -> None:
    """Catches first-seen ordering, ambient GPUs, duplicate calls, or positional argv."""
    materializer = _executable(tmp_path / "source-materializer")
    projected_request = tmp_path / "projected-request.json"
    groups = (
        "pyramid|29x53x101",
        "pyramid|17x31x63",
        "pyramid|23x47x95",
        "pyramid|17x31x63",
    )

    invocations = build_source_invocations(
        projected_request,
        groups,
        source_materializer=materializer,
        validated_gpu_policy=_gpu_policy(),
    )

    assert invocations == (
        (
            str(materializer),
            "--request",
            str(projected_request),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|17x31x63",
            "--gpu",
            "101",
        ),
        (
            str(materializer),
            "--request",
            str(projected_request),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|23x47x95",
            "--gpu",
            "103",
        ),
        (
            str(materializer),
            "--request",
            str(projected_request),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|29x53x101",
            "--gpu",
            "107",
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_gpu",
        "nonpolicy_uuid",
        "duplicate_uuid",
        "malformed_index",
        "wrong_model",
        "wrong_occupancy_gate",
    ],
)
def test_source_invocations_reject_malformed_or_nonpolicy_gpu_policy(
    tmp_path: Path, mutation: str
) -> None:
    """Catches invocation planning from anything except the validated binding policy."""
    policy = _gpu_policy()
    if mutation == "duplicate_gpu":
        policy["indices"] = [101, 101, 107]
    elif mutation == "nonpolicy_uuid":
        policy["uuid_by_index"] = {
            "101": "GPU-synthetic-101",
            "103": "GPU-synthetic-103",
            "109": "GPU-synthetic-109",
        }
    elif mutation == "duplicate_uuid":
        policy["uuid_by_index"]["103"] = " GPU-synthetic-101 "
    elif mutation == "malformed_index":
        policy["indices"] = [101, True, 107]
    elif mutation == "wrong_model":
        policy["model"] = "h100"
    else:
        policy["maximum_occupancy"] = 0.5

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        build_source_invocations(
            tmp_path / "request.json",
            ("pyramid|17x31x63",),
            source_materializer=_executable(tmp_path / "source-materializer"),
            validated_gpu_policy=policy,
        )

    assert captured.value.category == "history_execution_invalid"
    assert str(tmp_path) not in str(captured.value)


def test_source_invocations_preserve_validated_gpu_policy_existing_order(
    tmp_path: Path,
) -> None:
    """Catches sorting or guessing GPU order inside the source adapter."""
    policy = _gpu_policy()
    policy["indices"] = [107, 103, 101]
    policy["uuid_by_index"] = {
        str(index): f"GPU-synthetic-{index}" for index in policy["indices"]
    }

    invocations = build_source_invocations(
        tmp_path / "request.json",
        (
            "pyramid|29x53x101",
            "pyramid|17x31x63",
            "pyramid|23x47x95",
        ),
        source_materializer=_executable(tmp_path / "source-materializer"),
        validated_gpu_policy=policy,
    )

    assert [argv[6] for argv in invocations] == [
        "pyramid|17x31x63",
        "pyramid|23x47x95",
        "pyramid|29x53x101",
    ]
    assert [argv[8] for argv in invocations] == ["107", "103", "101"]


def test_source_runner_validates_complete_plan_before_direct_argv_execution(
    tmp_path: Path,
) -> None:
    """Catches shell execution, mutable env reuse, or a duplicate partial invocation."""
    materializer = _executable(tmp_path / "source-materializer")
    request_path = tmp_path / "request.json"
    invocations = build_source_invocations(
        request_path,
        ("pyramid|17x31x63", "pyramid|23x47x95"),
        source_materializer=materializer,
        validated_gpu_policy=_gpu_policy(),
    )
    duplicate_plan = (*invocations, invocations[0])
    runner = _RecordingRunner()

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        run_source_invocations(
            duplicate_plan,
            runner=runner,
            cwd=tmp_path,
            env={"SYNTHETIC_MODE": "offline"},
        )

    assert captured.value.category == "history_execution_invalid"
    assert runner.calls == []

    environment = {"SYNTHETIC_MODE": "offline"}
    run_source_invocations(
        invocations,
        runner=runner,
        cwd=tmp_path,
        env=environment,
    )
    environment["SYNTHETIC_MODE"] = "mutated"
    assert [call[0] for call in runner.calls] == list(invocations)
    assert all(call[1] == tmp_path for call in runner.calls)
    assert all(call[2] == {"SYNTHETIC_MODE": "offline"} for call in runner.calls)
    assert all(call[3] is False for call in runner.calls)


@pytest.mark.parametrize("mode", ["nonzero", "exception"])
def test_source_runner_failure_is_stable_and_redacted(
    tmp_path: Path, mode: str
) -> None:
    """Catches source failures leaking private runner or path details."""
    invocations = build_source_invocations(
        tmp_path / "request.json",
        ("pyramid|17x31x63",),
        source_materializer=_executable(tmp_path / "source-materializer"),
        validated_gpu_policy=_gpu_policy(),
    )
    runner = _RecordingRunner(returncode=17, raises=mode == "exception")

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        run_source_invocations(
            invocations,
            runner=runner,
            cwd=tmp_path,
            env={"PRIVATE_PATH": str(tmp_path)},
        )

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
    assert str(tmp_path) not in str(captured.value)


@pytest.mark.parametrize("mutation", ["same_group_mismatch", "cross_group_collision"])
def test_projection_rejects_shared_path_mismatch_or_collision(mutation: str) -> None:
    request = _request()
    if mutation == "same_group_mismatch":
        request["rows"][1]["source_contract"]["shared_source_paths"][
            "checkpoint_path"
        ] += "-drift"
        _rehash_row_contract(request, 1)
    else:
        first_path = request["rows"][0]["source_contract"]["shared_source_paths"][
            "checkpoint_path"
        ]
        request["rows"][2]["source_contract"]["shared_source_paths"][
            "checkpoint_path"
        ] = first_path
        _rehash_row_contract(request, 2)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"


def test_projection_rejects_same_group_static_contract_drift() -> None:
    """Catches one deduplicated group carrying conflicting private training inputs."""
    request = _request()
    request["rows"][1]["source_contract"]["training_parameters"]["epochs"] = 8
    _rehash_row_contract(request, 1)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"


def test_projection_strips_all_legacy_dynamic_output_aliases() -> None:
    """Catches a v2 request retaining flat or per-q recipe-v1 source truths."""
    request = _request()
    original_shared_paths = [
        copy.deepcopy(row["source_contract"]["shared_source_paths"])
        for row in request["rows"]
    ]
    legacy_keys = (
        "training_path",
        "checkpoint_path",
        "onnx_path",
        "calibration_path",
    )
    for row in request["rows"]:
        contract = row["source_contract"]
        contract.update(
            {key: f"/private/untrusted/legacy-flat/{key}" for key in legacy_keys}
        )
        contract["materialization_outputs_by_q_mode"] = {
            "fp16": {key: f"/private/untrusted/per-q/{key}" for key in legacy_keys}
        }
        row["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)
    original = copy.deepcopy(request)

    projected = project_source_materialization_request(request)

    for row, shared_paths in zip(
        projected.request["rows"], original_shared_paths, strict=True
    ):
        contract = row["source_contract"]
        assert "materialization_outputs_by_q_mode" not in contract
        assert "training_path" not in contract
        assert "calibration_path" not in contract
        assert contract["checkpoint_path"] == shared_paths["checkpoint_path"]
        assert contract["onnx_path"] == shared_paths["onnx_path"]
    assert request == original


@pytest.mark.parametrize("mutation", ["identity", "missing_path", "extra_path"])
def test_projection_rejects_registry_identity_or_shared_key_drift(mutation: str) -> None:
    request = _request()
    contract: dict[str, Any] = request["rows"][0]["source_contract"]
    if mutation == "identity":
        contract["artifact_id"] = "pyramid-identity-drift"
    elif mutation == "missing_path":
        contract["shared_source_paths"].pop("onnx_report_path")
    else:
        contract["shared_source_paths"]["untrusted_extra_path"] = (
            "/private/synthetic/untrusted"
        )
    _rehash_row_contract(request, 0)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"


def test_projection_preserves_legacy_request_without_shared_bundle() -> None:
    """Catches breaking the recipe-v1/static registry measurement path."""
    request = _request()
    for row in request["rows"]:
        row["source_contract"].pop("shared_source_paths")
        row["source_contract_sha256"] = _sha(row["source_contract"])
    _rehash_request(request)
    original = copy.deepcopy(request)

    projected = project_source_materialization_request(request)

    assert projected.request == original
    assert projected.request is not request
    assert request == original


def test_projection_rejects_mixed_legacy_and_shared_requests() -> None:
    request = _request()
    request["rows"][0]["source_contract"].pop("shared_source_paths")
    _rehash_row_contract(request, 0)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
