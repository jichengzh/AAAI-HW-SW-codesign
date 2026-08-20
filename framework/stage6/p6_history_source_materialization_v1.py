"""Project shared P6 source bundles and invoke the private materializer safely."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from framework.stage5.genome_contract_v1 import (
    canonical_group_id,
    validate_structure_identity,
)
from framework.stage5.production_search_v1 import validate_source_contract
from framework.stage6.p6_history_binding_v1 import MAX_GPU_OCCUPANCY
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
    """A detached canonical request and its canonical group order."""

    request: Mapping[str, Any]
    ordered_group_ids: tuple[str, ...]


class RunnerResult(Protocol):
    """Minimal completed-process surface consumed by source invocation."""

    returncode: int


class Runner(Protocol):
    """Injected direct-argv process boundary."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> RunnerResult: ...


def project_source_materialization_request(
    request: Mapping[str, Any],
) -> ProjectedSourceRequest:
    """Flatten recipe-v2 shared bundles without mutating the Stage5 request."""
    try:
        projected = _validated_request_copy(request)
        rows = projected["rows"]
        ordered_group_ids = tuple(sorted({row["group_id"] for row in rows}))
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


def build_source_invocations(
    projected_request_path: Path,
    ordered_group_ids: Sequence[str],
    *,
    source_materializer: Path,
    validated_gpu_policy: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Build one canonical direct argv per distinct source group."""
    try:
        request_path = _absolute_path(projected_request_path)
        materializer = _validated_materializer(source_materializer)
        groups = tuple(
            sorted({_validated_group_id(group_id) for group_id in ordered_group_ids})
        )
        if not groups:
            _invalid()
        gpu_indices = _validated_gpu_indices(validated_gpu_policy)
        return tuple(
            (
                str(materializer),
                "--request",
                str(request_path),
                "--model",
                "pyramid",
                "--group-id",
                group_id,
                "--gpu",
                str(gpu_indices[index % len(gpu_indices)]),
            )
            for index, group_id in enumerate(groups)
        )
    except P6HistorySourceMaterializationError:
        raise
    except (KeyError, OSError, OverflowError, TypeError, ValueError):
        raise P6HistorySourceMaterializationError() from None


def run_source_invocations(
    invocations: Sequence[Sequence[str]],
    *,
    runner: Runner,
    cwd: Path,
    env: Mapping[str, str],
) -> None:
    """Validate a complete plan, then execute it using direct argv only."""
    try:
        canonical_invocations = _validated_invocations(invocations)
        canonical_cwd = _validated_cwd(cwd)
        canonical_env = _validated_environment(env)
        for argv in canonical_invocations:
            completed = runner.run(
                argv,
                cwd=canonical_cwd,
                env=dict(canonical_env),
                shell=False,
            )
            returncode = completed.returncode
            if (
                isinstance(returncode, bool)
                or not isinstance(returncode, int)
                or returncode != 0
            ):
                _invalid()
    except P6HistorySourceMaterializationError:
        raise
    except Exception:
        raise P6HistorySourceMaterializationError() from None


def _absolute_path(raw_path: Path) -> Path:
    path = Path(raw_path)
    serialized = str(path)
    if (
        not path.is_absolute()
        or not serialized
        or any(character in serialized for character in ("\x00", "\r", "\n"))
    ):
        _invalid()
    return path


def _validated_materializer(raw_path: Path) -> Path:
    return _absolute_path(raw_path)


def _validated_group_id(raw_group_id: object) -> str:
    if not isinstance(raw_group_id, str) or not raw_group_id.startswith("pyramid|"):
        _invalid()
    raw_width = raw_group_id.removeprefix("pyramid|").split("x")
    if len(raw_width) != len(STAGE_WIDTH_FIELDS):
        _invalid()
    width = tuple(int(value) for value in raw_width)
    if canonical_group_id("pyramid", width) != raw_group_id:
        _invalid()
    return raw_group_id


def _validated_gpu_indices(policy: Mapping[str, Any]) -> tuple[int, int, int]:
    if not isinstance(policy, Mapping) or set(policy) != {
        "indices",
        "uuid_by_index",
        "model",
        "maximum_occupancy",
    }:
        _invalid()
    raw_indices = policy.get("indices")
    if (
        not isinstance(raw_indices, Sequence)
        or isinstance(raw_indices, (str, bytes))
        or len(raw_indices) != 3
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in raw_indices
        )
    ):
        _invalid()
    indices = tuple(raw_indices)
    if (
        len(set(indices)) != len(indices)
        or any(index < 0 for index in indices)
        or policy.get("model") != "h800"
        or policy.get("maximum_occupancy") != MAX_GPU_OCCUPANCY
    ):
        _invalid()
    uuid_by_index = policy.get("uuid_by_index")
    if (
        not isinstance(uuid_by_index, Mapping)
        or set(uuid_by_index) != {str(index) for index in indices}
    ):
        _invalid()
    raw_uuids = [uuid_by_index[str(index)] for index in indices]
    if any(not isinstance(uuid, str) for uuid in raw_uuids):
        _invalid()
    uuids = tuple(uuid.strip() for uuid in raw_uuids)
    if any(not uuid for uuid in uuids) or len(set(uuids)) != len(uuids):
        _invalid()
    return (indices[0], indices[1], indices[2])


def _validated_invocations(
    invocations: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(invocations, Sequence) or isinstance(invocations, (str, bytes)):
        _invalid()
    canonical: list[tuple[str, ...]] = []
    groups: set[str] = set()
    ordered_groups: list[str] = []
    executable: str | None = None
    request_path: str | None = None
    for raw_argv in invocations:
        if (
            not isinstance(raw_argv, Sequence)
            or isinstance(raw_argv, (str, bytes))
            or len(raw_argv) != 9
            or any(not isinstance(token, str) or not token for token in raw_argv)
        ):
            _invalid()
        argv = tuple(raw_argv)
        if (
            argv[1::2] != ("--request", "--model", "--group-id", "--gpu")
            or argv[4] != "pyramid"
            or str(_absolute_path(Path(argv[0]))) != argv[0]
            or str(_absolute_path(Path(argv[2]))) != argv[2]
            or _validated_group_id(argv[6]) != argv[6]
            or not argv[8].isdigit()
            or str(int(argv[8])) != argv[8]
        ):
            _invalid()
        if argv[6] in groups:
            _invalid()
        groups.add(argv[6])
        ordered_groups.append(argv[6])
        executable = executable or argv[0]
        request_path = request_path or argv[2]
        if argv[0] != executable or argv[2] != request_path:
            _invalid()
        canonical.append(argv)
    if not canonical:
        _invalid()
    if ordered_groups != sorted(ordered_groups):
        _invalid()
    return tuple(canonical)


def _validated_cwd(cwd: Path) -> Path:
    path = _absolute_path(cwd)
    if path.is_symlink() or not path.is_dir():
        _invalid()
    return path.resolve(strict=True)


def _validated_environment(env: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(env, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or any(character in key for character in ("\x00", "=", "\r", "\n"))
        or any(character in value for character in ("\x00", "\r", "\n"))
        for key, value in env.items()
    ):
        _invalid()
    return dict(env)


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
