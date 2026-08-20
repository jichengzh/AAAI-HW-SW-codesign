"""Pure projection of shared P6 source bundles into historical flat requests."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from framework.stage5.genome_contract_v1 import (
    canonical_group_id,
    validate_structure_identity,
)
from framework.stage5.production_search_v1 import validate_source_contract
from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS


REQUEST_SCHEMA_VERSION = "stage5_measurement_request_v2"
REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "task_sha256",
        "round_index",
        "batch_size",
        "sample_budget",
        "required_metrics",
        "atomic_feedback",
        "real_h800_measurement_required",
        "row_sha256",
        "rows",
        "measurement_request_sha256",
    }
)
ROW_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "task_sha256",
        "row_id",
        "manifest_job_id",
        "group_id",
        "model",
        "width",
        "width_schema",
        "structure_widths",
        "genome",
        "strategy_id",
        "q_mode",
        "hardware_id",
        "capability_profile_id",
        "capability_digest",
        "dispatch_key",
        "source_status",
        "materialization_kind",
        "source_evidence_kind",
        "source_contract",
        "source_contract_sha256",
        "source_evidence_sha256",
        "graph_features",
    }
)
METRIC_KEYS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
ALLOWED_Q_MODES = frozenset({"fp16", "int8"})
STAGE_WIDTH_FIELDS = ("stage1_width", "stage2_width", "stage3_width")
LEGACY_DYNAMIC_OUTPUT_KEYS = (
    "training_path",
    "checkpoint_path",
    "onnx_path",
    "calibration_path",
)


class P6HistorySourceMaterializationError(ValueError):
    """Stable failure category for projection and source invocation contracts."""

    def __init__(self) -> None:
        self.category = "history_execution_invalid"
        super().__init__(self.category)


@dataclass(frozen=True)
class ProjectedSourceRequest:
    """A detached canonical request and its first-seen group order."""

    request: Mapping[str, Any]
    ordered_group_ids: tuple[str, ...]


def project_source_materialization_request(
    request: Mapping[str, Any],
) -> ProjectedSourceRequest:
    """Flatten recipe-v2 shared bundles without mutating the Stage5 request."""
    try:
        projected = _validated_request_copy(request)
        rows = projected["rows"]
        ordered_group_ids = tuple(dict.fromkeys(row["group_id"] for row in rows))
        shared_presence = [
            "shared_source_paths" in row["source_contract"] for row in rows
        ]
        if not any(shared_presence):
            return ProjectedSourceRequest(projected, ordered_group_ids)
        if not all(shared_presence):
            _invalid()

        paths_by_group: dict[str, tuple[str, ...]] = {}
        paths_owned_by_group: dict[str, str] = {}
        contract_by_group: dict[str, dict[str, Any]] = {}
        for row in rows:
            group_id = row["group_id"]
            contract = row["source_contract"]
            _validate_shared_identity(row, contract)
            shared_paths = _validated_shared_paths(contract["shared_source_paths"])
            path_identity = tuple(shared_paths[key] for key in SHARED_SOURCE_PATH_KEYS)
            previous = paths_by_group.setdefault(group_id, path_identity)
            if previous != path_identity:
                _invalid()
            for path in path_identity:
                owner = paths_owned_by_group.setdefault(path, group_id)
                if owner != group_id:
                    _invalid()

            flat_contract = copy.deepcopy(contract)
            flat_contract.pop("shared_source_paths")
            flat_contract.pop("dynamic_materialization_recipe", None)
            flat_contract.pop("materialization_outputs_by_q_mode", None)
            for key in LEGACY_DYNAMIC_OUTPUT_KEYS:
                flat_contract.pop(key, None)
            flat_contract.update(shared_paths)
            canonical_contract = contract_by_group.setdefault(group_id, flat_contract)
            if canonical_contract != flat_contract:
                _invalid()
            row["source_contract"] = flat_contract
            row["source_contract_sha256"] = _canonical_sha(flat_contract)
            validate_source_contract(row)

        projected["row_sha256"] = {
            row["row_id"]: _canonical_sha(row) for row in rows
        }
        request_body = {
            key: value
            for key, value in projected.items()
            if key != "measurement_request_sha256"
        }
        projected["measurement_request_sha256"] = _canonical_sha(request_body)
        return ProjectedSourceRequest(projected, ordered_group_ids)
    except P6HistorySourceMaterializationError:
        raise
    except (KeyError, OverflowError, TypeError, ValueError):
        raise P6HistorySourceMaterializationError() from None


def _validated_request_copy(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping) or set(request) != REQUEST_KEYS:
        _invalid()
    projected = copy.deepcopy(dict(request))
    rows = projected.get("rows")
    row_hashes = projected.get("row_sha256")
    task_id = projected.get("task_id")
    task_sha = projected.get("task_sha256")
    batch_size = projected.get("batch_size")
    round_index = projected.get("round_index")
    sample_budget = projected.get("sample_budget")
    if (
        projected.get("schema_version") != REQUEST_SCHEMA_VERSION
        or not isinstance(task_id, str)
        or not task_id
        or not _is_sha(task_sha)
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or isinstance(sample_budget, bool)
        or not isinstance(sample_budget, int)
        or sample_budget < batch_size
        or isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
        or projected.get("required_metrics") != list(METRIC_KEYS)
        or projected.get("atomic_feedback") is not True
        or projected.get("real_h800_measurement_required") is not True
        or not isinstance(rows, list)
        or len(rows) != batch_size
        or not isinstance(row_hashes, Mapping)
    ):
        _invalid()
    row_ids = [
        _validate_row(row, str(task_id), str(task_sha))
        for row in rows
    ]
    if len(set(row_ids)) != len(row_ids) or set(row_hashes) != set(row_ids):
        _invalid()
    if any(
        row_hashes[row_id] != _canonical_sha(row)
        for row, row_id in zip(rows, row_ids, strict=True)
    ):
        _invalid()
    request_body = {
        key: value
        for key, value in projected.items()
        if key != "measurement_request_sha256"
    }
    if projected.get("measurement_request_sha256") != _canonical_sha(request_body):
        _invalid()
    return projected


def _validate_row(row: object, task_id: str, task_sha: str) -> str:
    if not isinstance(row, Mapping) or set(row) != ROW_KEYS:
        _invalid()
    row_id = row.get("row_id")
    q_mode = row.get("q_mode")
    if (
        row.get("schema_version") != "stage5_candidate_row_v2"
        or not isinstance(row_id, str)
        or not row_id
        or row.get("manifest_job_id") != row_id
        or row.get("task_id") != task_id
        or row.get("task_sha256") != task_sha
        or row.get("model") != "pyramid"
        or row.get("hardware_id") != "h800"
        or row.get("dispatch_key") != "tvm_auto"
        or row.get("source_status") not in {"ready", "materializable"}
        or q_mode not in ALLOWED_Q_MODES
    ):
        _invalid()
    identity = validate_structure_identity(row)
    validate_source_contract(row)
    width = list(identity.width)
    if (
        row.get("strategy_id") != f"q={q_mode}"
        or row.get("genome") != [*width, q_mode]
    ):
        _invalid()
    return row_id


def _validate_shared_identity(
    row: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    width = row["width"]
    width_values = dict(zip(STAGE_WIDTH_FIELDS, width, strict=True))
    if (
        row["group_id"] != canonical_group_id("pyramid", width)
        or contract.get("artifact_id")
        != f"pyramid-{width[0]}-{width[1]}-{width[2]}"
        or contract.get("stage_widths") != width_values
    ):
        _invalid()


def _validated_shared_paths(raw_paths: object) -> dict[str, str]:
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != set(
        SHARED_SOURCE_PATH_KEYS
    ):
        _invalid()
    paths = {key: raw_paths[key] for key in SHARED_SOURCE_PATH_KEYS}
    if any(
        not isinstance(value, str)
        or not value
        or not Path(value).is_absolute()
        or any(character in value for character in ("\x00", "\r", "\n"))
        for value in paths.values()
    ) or len(set(paths.values())) != len(SHARED_SOURCE_PATH_KEYS):
        _invalid()
    return paths


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _invalid() -> None:
    raise P6HistorySourceMaterializationError()
