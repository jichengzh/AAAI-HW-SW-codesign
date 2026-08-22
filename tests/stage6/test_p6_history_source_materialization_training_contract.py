from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
from framework.stage6.p6_history_source_materialization_v1 import (
    P6HistorySourceMaterializationError,
    project_source_materialization_request,
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
        "external_training_binding": {
            "schema_version": "p6_external_training_binding_v1",
            "training_required": True,
            "training_source_kind": "selected_candidate_finetune",
            "base_checkpoint_path": "/etc/hosts",
            "base_checkpoint_sha256": hashlib.sha256(Path("/etc/hosts").read_bytes()).hexdigest(),
            "dataset_root": "/tmp",
            "pyramid_config_path": "/etc/passwd",
            "pyramid_config_sha256": hashlib.sha256(Path("/etc/passwd").read_bytes()).hexdigest(),
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


def _true_legacy_request() -> dict[str, Any]:
    request = _request()
    recipe_v2_keys = {
        "shared_source_paths",
        "stage_widths",
        "external_training_binding",
    }
    for row in request["rows"]:
        contract = row["source_contract"]
        for key in recipe_v2_keys:
            contract.pop(key, None)
        row["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)
    return request


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
        training = contract["external_training_binding"]
        assert training["base_checkpoint_path"] == "/etc/hosts"
        assert training["training_required"] is True
        assert training["training_source_kind"] == "selected_candidate_finetune"
        assert training["pyramid_config_path"] == "/etc/passwd"
        assert set(training["training_parameters"]) == {
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
    projected.request["rows"][0]["source_contract"]["external_training_binding"]["training_parameters"][
        "epochs"
    ] = 99
    assert request["rows"][0]["source_contract"]["external_training_binding"]["training_parameters"][
        "epochs"
    ] == 7


def test_projection_preserves_complete_training_contract_and_rehashes() -> None:
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
        training = contract["external_training_binding"]
        assert training["training_required"] is True
        assert training["training_source_kind"] == "selected_candidate_finetune"
        assert training["base_checkpoint_path"] == "/etc/hosts"
        assert training["dataset_root"] == "/tmp"
        assert training["pyramid_config_path"] == "/etc/passwd"
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
    second_training = second["external_training_binding"]
    rehash_index = 1
    if mutation == "same_group_training_parameters_drift":
        second_training["training_parameters"]["epochs"] = 8
    elif mutation == "same_group_base_checkpoint_drift":
        second_training["base_checkpoint_path"] += ".drift"
    elif mutation == "same_group_dataset_drift":
        second_training["dataset_root"] += "-drift"
    elif mutation == "same_group_pyramid_config_drift":
        second_training["pyramid_config_path"] += ".drift"
    elif mutation == "same_group_training_required_false":
        second_training["training_required"] = False
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
        first["external_training_binding"]["training_parameters"]["epochs"] = 8
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
        training = contract["external_training_binding"]
        if mutation == "training_required_false":
            training["training_required"] = False
        elif mutation == "missing_training_source_kind":
            training.pop("training_source_kind")
        elif mutation == "checkpoint_reuse_source_kind":
            training["training_source_kind"] = "checkpoint_already_exists"
        elif mutation == "missing_base_checkpoint_path":
            training.pop("base_checkpoint_path")
        elif mutation == "missing_dataset_root":
            training.pop("dataset_root")
        elif mutation == "missing_pyramid_config_path":
            training.pop("pyramid_config_path")
        elif mutation == "missing_training_parameter":
            training["training_parameters"].pop("freeze_policy")
        else:
            training["dataset_root"] = "relative/dataset"
        row["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"


def test_projected_training_marker_pairs_are_unique_and_canonically_ordered() -> None:
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
            row["source_contract"]["external_training_binding"]["training_required"] = False
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
        "external_training_binding",
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
        contract["external_training_binding"][path_key] = malformed
    request["rows"][2]["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
    assert malformed not in str(captured.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "recipe_v2_schema",
        "per_q_output_map",
        "partial_flat_shared_path",
        "partial_shared_bundle",
        "training_required_signature",
        "static_path_signature",
    ],
)
def test_projection_rejects_incomplete_request_with_any_recipe_v2_signal(
    mutation: str,
) -> None:
    """Catches malformed recipe-v2 identity falling through legacy unchanged."""
    request = _true_legacy_request()
    for row in request["rows"]:
        contract = row["source_contract"]
        if mutation == "recipe_v2_schema":
            contract["dynamic_materialization_recipe"] = {
                "schema_version": "p6_history_dynamic_materialization_recipe_v2"
            }
        elif mutation == "per_q_output_map":
            contract["materialization_outputs_by_q_mode"] = {
                "fp16": {"checkpoint_path": "/private/untrusted/checkpoint"}
            }
        elif mutation == "partial_flat_shared_path":
            contract["training_done_marker"] = "/private/incomplete/training.done"
        elif mutation == "partial_shared_bundle":
            contract["shared_source_paths"] = {
                "source_done_marker": "/private/incomplete/source.done"
            }
        elif mutation == "training_required_signature":
            contract["training_required"] = True
        else:
            contract["base_checkpoint_path"] = "/private/incomplete/base.ckpt"
        row["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)
    original = copy.deepcopy(request)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert request == original
    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"


@pytest.mark.parametrize(
    "path_key,malformed",
    [
        (
            "base_checkpoint_path",
            "/private/synthetic/base/./model.ckpt",
        ),
        (
            "base_checkpoint_path",
            "/private//synthetic/base/model.ckpt",
        ),
        (
            "base_checkpoint_path",
            "/private/synthetic/base/model.ckpt/",
        ),
        (
            "dataset_root",
            "/private/synthetic/./dataset",
        ),
        (
            "dataset_root",
            "/private/synthetic//dataset",
        ),
        (
            "dataset_root",
            "/private/synthetic/dataset/",
        ),
        (
            "pyramid_config_path",
            "/private/synthetic/configs/./pyramid.py",
        ),
        (
            "pyramid_config_path",
            "/private/synthetic//configs/pyramid.py",
        ),
        (
            "pyramid_config_path",
            "/private/synthetic/configs/pyramid.py/",
        ),
    ],
)
def test_projection_rejects_noncanonical_static_training_path_spelling(
    path_key: str,
    malformed: str,
) -> None:
    """Catches ambiguous absolute static paths admitted without filesystem reads."""
    request = _request()
    contract = request["rows"][2]["source_contract"]
    contract["external_training_binding"][path_key] = malformed
    request["rows"][2]["source_contract_sha256"] = _sha(contract)
    _rehash_request(request)

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        project_source_materialization_request(request)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
    assert malformed not in str(captured.value)


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
    request["rows"][1]["source_contract"]["external_training_binding"]["training_parameters"]["epochs"] = 8
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
    request = _true_legacy_request()
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
