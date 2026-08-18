"""Fail-closed execution bridge from one P6 batch to private history feedback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Protocol

from framework.stage5.genome_contract_v1 import validate_structure_identity
from framework.stage5.production_search_v1 import validate_source_contract
from framework.stage6.p6_history_binding_v1 import (
    ALLOWED_NORMALIZED_H800_MODELS,
    EXPECTED_GPU_INDICES,
    MAX_GPU_OCCUPANCY,
    GpuProbe,
    GpuRecord,
    P6HistoryBindingError,
    validate_history_execution_binding,
)


REQUEST_SCHEMA_VERSION = "stage5_measurement_request_v2"
FEEDBACK_SCHEMA_VERSION = "p6_h800_coptv2x_feedback_v2"
METRIC_KEYS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
SUCCESS_STATUS = "measured_success_gold"
FAILURE_STATUSES = frozenset({"feasibility_failure", "numerical_feasibility_failure"})
ALLOWED_STATUSES = frozenset({SUCCESS_STATUS, *FAILURE_STATUSES})
ALLOWED_Q_MODES = frozenset({"fp16", "int8"})
PUBLIC_FAILURE_REASON = re.compile(r"^[a-z0-9_-]+$")
MAX_PRIVATE_JSON_BYTES = 16 * 1024 * 1024

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


class P6HistoryMeasurementError(ValueError):
    """Stable public category for private history measurement failures."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class RunnerResult(Protocol):
    """Minimal completed-process surface consumed by the adapter."""

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


def run_history_measurement_batch(
    request: Mapping[str, Any],
    binding: Mapping[str, Any],
    round_output_root: str | Path,
    runner: Runner,
    gpu_probe: GpuProbe,
) -> dict[str, Any]:
    """Execute one validated four-row history batch and return safe P6 feedback."""
    interface = _validated_interface(binding)
    verified_request = _validate_request(request)
    private_root, gpu_policy = _validate_binding_runtime(binding)
    paths = _resolve_round_paths(
        interface, private_root, Path(round_output_root), verified_request["round_index"]
    )
    _validate_gpu(gpu_probe, gpu_policy)
    _initialize_private_round(verified_request, interface, paths)
    environment = _render_environment(interface, paths)
    substitutions = _substitutions(paths)
    _execute(interface["environment"]["activation_argv"], substitutions, runner, paths, environment)
    for stage in interface["execution_chain"]:
        _execute(stage["argv"], substitutions, runner, paths, environment)
    _validate_gpu(gpu_probe, gpu_policy)
    return _translate_feedback(verified_request, interface, paths)


def _validated_interface(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return validate_history_execution_binding(binding)
    except (P6HistoryBindingError, OSError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        if not isinstance(request, Mapping) or set(request) != REQUEST_KEYS:
            _request_invalid()
        if (
            request.get("schema_version") != REQUEST_SCHEMA_VERSION
            or request.get("batch_size") != 4
            or request.get("sample_budget") != 16
            or request.get("required_metrics") != list(METRIC_KEYS)
            or request.get("atomic_feedback") is not True
            or request.get("real_h800_measurement_required") is not True
        ):
            _request_invalid()
        round_index = request.get("round_index")
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            _request_invalid()
        task_id = request.get("task_id")
        task_sha = request.get("task_sha256")
        if not isinstance(task_id, str) or not task_id or not _is_sha(task_sha):
            _request_invalid()
        rows = request.get("rows")
        row_hashes = request.get("row_sha256")
        if (
            not isinstance(rows, list)
            or len(rows) != 4
            or not isinstance(row_hashes, Mapping)
            or not all(isinstance(row, Mapping) for row in rows)
        ):
            _request_invalid()
        row_ids = [_validate_request_row(row, task_id, task_sha) for row in rows]
        if len(set(row_ids)) != 4 or set(row_hashes) != set(row_ids):
            _request_invalid()
        for row, row_id in zip(rows, row_ids, strict=True):
            if row_hashes.get(row_id) != _canonical_sha(row):
                _request_invalid()
        body = {key: value for key, value in request.items() if key != "measurement_request_sha256"}
        request_sha = request.get("measurement_request_sha256")
        if not _is_sha(request_sha) or request_sha != _canonical_sha(body):
            _request_invalid()
        return _json_detached(request)
    except P6HistoryMeasurementError:
        raise
    except (KeyError, OverflowError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_request_invalid") from None


def _validate_request_row(row: Mapping[str, Any], task_id: str, task_sha: str) -> str:
    if set(row) != ROW_KEYS:
        _request_invalid()
    row_id = row.get("row_id")
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
        or not isinstance(row.get("capability_profile_id"), str)
        or not row["capability_profile_id"]
        or not _is_sha(row.get("capability_digest"))
        or not _is_sha(row.get("source_evidence_sha256"))
    ):
        _request_invalid()
    identity = validate_structure_identity(row)
    validate_source_contract(row)
    width = list(identity.width)
    q_mode = row.get("q_mode")
    if (
        q_mode not in ALLOWED_Q_MODES
        or row.get("strategy_id") != f"q={q_mode}"
        or row.get("genome") != [*width, q_mode]
    ):
        _request_invalid()
    graph = row.get("graph_features")
    if (
        not isinstance(graph, Mapping)
        or graph.get("group_id") != row.get("group_id")
        or graph.get("model") != "pyramid"
        or graph.get("width") != width
    ):
        _request_invalid()
    return row_id


def _validate_binding_runtime(
    binding: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    try:
        private_root = Path(binding["private_root"])
        if not private_root.is_absolute() or private_root.is_symlink():
            raise ValueError
        private_root = private_root.resolve(strict=True)
        if not private_root.is_dir():
            raise ValueError
        raw_policy = binding.get("gpu_policy")
        if not isinstance(raw_policy, Mapping) or set(raw_policy) != {
            "indices",
            "uuid_by_index",
            "model",
            "maximum_occupancy",
        }:
            raise ValueError
        uuid_by_index = raw_policy.get("uuid_by_index")
        if (
            raw_policy.get("indices") != list(EXPECTED_GPU_INDICES)
            or raw_policy.get("model") != "h800"
            or raw_policy.get("maximum_occupancy") != MAX_GPU_OCCUPANCY
            or not isinstance(uuid_by_index, Mapping)
            or set(uuid_by_index) != {str(index) for index in EXPECTED_GPU_INDICES}
        ):
            raise ValueError
        uuids = [uuid_by_index[str(index)] for index in EXPECTED_GPU_INDICES]
        if any(not isinstance(uuid, str) or not uuid.strip() for uuid in uuids) or len(
            set(uuids)
        ) != len(uuids):
            raise ValueError
        policy = {
            "uuid_by_index": {
                str(index): str(uuid_by_index[str(index)]).strip() for index in EXPECTED_GPU_INDICES
            },
            "maximum_occupancy": MAX_GPU_OCCUPANCY,
        }
        return private_root, policy
    except (KeyError, OSError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _validate_gpu(gpu_probe: GpuProbe, policy: Mapping[str, Any]) -> None:
    try:
        snapshot = gpu_probe.snapshot(EXPECTED_GPU_INDICES)
        if not isinstance(snapshot, tuple) or len(snapshot) != len(EXPECTED_GPU_INDICES):
            raise ValueError
        if any(not isinstance(record, GpuRecord) for record in snapshot):
            raise ValueError
        by_index = {record.index: record for record in snapshot}
        if set(by_index) != set(EXPECTED_GPU_INDICES) or len(by_index) != len(snapshot):
            raise ValueError
        for index in EXPECTED_GPU_INDICES:
            record = by_index[index]
            if (
                record.uuid != policy["uuid_by_index"][str(index)]
                or _normalized_model(record.model_name) not in ALLOWED_NORMALIZED_H800_MODELS
                or isinstance(record.occupancy, bool)
                or not isinstance(record.occupancy, (int, float))
                or not math.isfinite(record.occupancy)
                or record.occupancy < 0.0
                or record.occupancy > policy["maximum_occupancy"]
            ):
                raise ValueError
    except Exception:
        raise P6HistoryMeasurementError("history_gpu_admission_failed") from None


def _resolve_round_paths(
    interface: Mapping[str, Any],
    private_root: Path,
    supplied_root: Path,
    round_index: int,
) -> dict[str, Path]:
    try:
        if not supplied_root.is_absolute() or supplied_root.is_symlink():
            raise ValueError
        supplied = supplied_root.resolve(strict=True)
        if not supplied.is_dir() or not _beneath(supplied, private_root):
            raise ValueError
        round_root = _resolve_template(
            supplied,
            interface["output_layout"]["round_root_template"],
            round_index,
            private_root,
        )
        task_state = _resolve_template(
            supplied,
            interface["output_layout"]["task_state"]["path_template"],
            round_index,
            private_root,
        )
        actual_feedback = interface["actual_feedback"]
        result = _resolve_template(
            supplied,
            actual_feedback["result"]["path_template"],
            round_index,
            private_root,
        )
        receipt = _resolve_template(
            supplied,
            actual_feedback["receipt"]["path_template"],
            round_index,
            private_root,
        )
        barrier = _resolve_template(
            supplied,
            actual_feedback["finalization_barrier"]["path_template"],
            round_index,
            private_root,
        )
        request_path = (round_root / "measurement-request.json").resolve(strict=False)
        paths = {
            "supplied_root": supplied,
            "round_root": round_root,
            "measurement_request": request_path,
            "task_state": task_state,
            "actual_feedback": result,
            "actual_receipt": receipt,
            "finalization_barrier": barrier,
        }
        if len(set(paths.values())) != len(paths) or any(
            not _beneath(path, supplied) or not _beneath(path, private_root)
            for key, path in paths.items()
            if key != "supplied_root"
        ):
            raise ValueError
        return paths
    except (KeyError, OSError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _resolve_template(
    supplied_root: Path, template: object, round_index: int, private_root: Path
) -> Path:
    if not isinstance(template, str):
        raise ValueError
    path = (supplied_root / template.replace("{round_id}", str(round_index))).resolve(strict=False)
    if not _beneath(path, supplied_root) or not _beneath(path, private_root):
        raise ValueError
    return path


def _initialize_private_round(
    request: Mapping[str, Any], interface: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    try:
        round_root = paths["round_root"]
        _mkdir_private(round_root, paths["supplied_root"])
        for key in (
            "measurement_request",
            "task_state",
            "actual_feedback",
            "actual_receipt",
            "finalization_barrier",
        ):
            path = paths[key]
            if path.is_symlink() or path.exists():
                raise OSError
            _mkdir_private(path.parent, paths["supplied_root"])
        _atomic_write_json(paths["measurement_request"], request)
        task_schema = interface["output_layout"]["task_state"]
        state_rows = [
            {
                task_schema["row_id_key"]: row["row_id"],
                task_schema["row_hash_key"]: request["row_sha256"][row["row_id"]],
                task_schema["source_evidence_key"]: row["source_evidence_sha256"],
                task_schema["status_key"]: "pending",
            }
            for row in request["rows"]
        ]
        _atomic_write_json(
            paths["task_state"],
            {
                task_schema["stage_key"]: "initialized",
                task_schema["rows_key"]: state_rows,
            },
        )
    except (KeyError, OSError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _render_environment(interface: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, str]:
    substitutions = _substitutions(paths)
    rendered: dict[str, str] = {}
    try:
        for key, specification in interface["environment"]["values"].items():
            kind = specification["kind"]
            value = specification["value"]
            if kind in {"literal", "private_path"}:
                rendered[key] = value
            elif kind == "placeholder" and value in substitutions:
                rendered[key] = substitutions[value]
            else:
                raise ValueError
        return rendered
    except (KeyError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _substitutions(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        "{measurement_request}": str(paths["measurement_request"]),
        "{round_output_root}": str(paths["round_root"]),
        "{task_state}": str(paths["task_state"]),
        "{actual_feedback}": str(paths["actual_feedback"]),
        "{actual_receipt}": str(paths["actual_receipt"]),
        "{finalization_barrier}": str(paths["finalization_barrier"]),
    }


def _execute(
    raw_argv: Sequence[str],
    substitutions: Mapping[str, str],
    runner: Runner,
    paths: Mapping[str, Path],
    environment: Mapping[str, str],
) -> None:
    try:
        argv = tuple(substitutions.get(token, token) for token in raw_argv)
        if not argv or any(token.startswith("{") or token.endswith("}") for token in argv):
            raise ValueError
        completed = runner.run(
            argv,
            cwd=paths["round_root"],
            env=dict(environment),
            shell=False,
        )
        returncode = completed.returncode
        if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0:
            raise RuntimeError
    except Exception:
        raise P6HistoryMeasurementError("history_execution_failed") from None


def _translate_feedback(
    request: Mapping[str, Any],
    interface: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    try:
        state = _read_private_json(paths["task_state"])
        result = _read_private_json(paths["actual_feedback"])
        receipt = _read_private_json(paths["actual_receipt"])
        barrier = _read_private_json(paths["finalization_barrier"])
        task_schema = interface["output_layout"]["task_state"]
        result_schema = interface["actual_feedback"]["result"]
        expected = _expected_identity(request)
        state_statuses = _validate_task_state(state, task_schema, expected)
        feedback_rows = _validate_result(result, result_schema, request, expected, state_statuses)
        _validate_completion_mapping(
            receipt, interface["actual_feedback"]["receipt"], request, expected
        )
        _validate_completion_mapping(
            barrier,
            interface["actual_feedback"]["finalization_barrier"],
            request,
            expected,
        )
        return {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": feedback_rows,
        }
    except P6HistoryMeasurementError:
        raise
    except (KeyError, OSError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _expected_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        row["row_id"]: {
            "row_sha256": request["row_sha256"][row["row_id"]],
            "source_evidence_sha256": row["source_evidence_sha256"],
        }
        for row in request["rows"]
    }


def _validate_task_state(
    state: Mapping[str, Any], schema: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, str]:
    stage_key = schema["stage_key"]
    rows_key = schema["rows_key"]
    if set(state) != {stage_key, rows_key} or state.get(stage_key) != schema["stage_order"][-1]:
        raise ValueError
    rows = _exact_rows(state.get(rows_key), expected, schema)
    for row_id, row in rows.items():
        if row[schema["status_key"]] not in ALLOWED_STATUSES:
            raise ValueError
        _validate_identity_row(row, row_id, schema, expected)
    return {row_id: str(row[schema["status_key"]]) for row_id, row in rows.items()}


def _validate_result(
    result: Mapping[str, Any],
    schema: Mapping[str, Any],
    request: Mapping[str, Any],
    expected: Mapping[str, Any],
    state_statuses: Mapping[str, str],
) -> list[dict[str, Any]]:
    if (
        set(result) != {"measurement_request_sha256", schema["rows_key"]}
        or result.get("measurement_request_sha256") != request["measurement_request_sha256"]
    ):
        raise ValueError
    rows = _exact_rows(result.get(schema["rows_key"]), expected, schema, variable=True)
    translated: list[dict[str, Any]] = []
    for request_row in request["rows"]:
        row_id = request_row["row_id"]
        row = rows[row_id]
        _validate_identity_row(row, row_id, schema, expected)
        status = row[schema["status_key"]]
        if status != state_statuses.get(row_id):
            raise ValueError
        base = {"row_id": row_id, "terminal_status": status}
        if status == SUCCESS_STATUS:
            expected_keys = {
                schema["row_id_key"],
                schema["row_hash_key"],
                schema["source_evidence_key"],
                schema["status_key"],
                *schema["metric_keys"],
            }
            if set(row) != expected_keys:
                raise ValueError
            metrics = {key: row[key] for key in schema["metric_keys"]}
            if (
                any(not _finite(value) for value in metrics.values())
                or float(metrics["latency_ms"]) <= 0.0
                or float(metrics["energy_j"]) <= 0.0
                or any(not 0.0 <= float(metrics[key]) <= 1.0 for key in ("ap30", "ap50", "ap70"))
            ):
                raise ValueError
            translated.append(
                {**base, **{key: float(metrics[key]) for key in schema["metric_keys"]}}
            )
        elif status in FAILURE_STATUSES:
            expected_keys = {
                schema["row_id_key"],
                schema["row_hash_key"],
                schema["source_evidence_key"],
                schema["status_key"],
                "failure_reason",
            }
            reason = row.get("failure_reason")
            if set(row) != expected_keys or not isinstance(reason, str) or not reason:
                raise ValueError
            translated.append(
                {
                    **base,
                    "failure_reason": (
                        reason if PUBLIC_FAILURE_REASON.fullmatch(reason) else "unspecified"
                    ),
                }
            )
        else:
            raise ValueError
    return translated


def _exact_rows(
    raw_rows: object,
    expected: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    variable: bool = False,
) -> dict[str, Mapping[str, Any]]:
    if (
        not isinstance(raw_rows, list)
        or len(raw_rows) != 4
        or not all(isinstance(row, Mapping) for row in raw_rows)
    ):
        raise ValueError
    row_ids = [row.get(schema["row_id_key"]) for row in raw_rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise ValueError
    if len(set(row_ids)) != 4 or set(row_ids) != set(expected):
        raise ValueError
    rows = dict(zip(row_ids, raw_rows, strict=True))
    if not variable:
        required = {
            schema["row_id_key"],
            schema["row_hash_key"],
            schema["source_evidence_key"],
            schema["status_key"],
        }
        if any(set(row) != required for row in rows.values()):
            raise ValueError
    return rows


def _validate_identity_row(
    row: Mapping[str, Any],
    row_id: str,
    schema: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    identity = expected[row_id]
    if (
        row.get(schema["row_hash_key"]) != identity["row_sha256"]
        or row.get(schema["source_evidence_key"]) != identity["source_evidence_sha256"]
    ):
        raise ValueError


def _validate_completion_mapping(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    request: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    request_key = schema["request_sha256_key"]
    row_hashes_key = schema["row_hashes_key"]
    evidence_key = schema["source_evidence_key"]
    if set(payload) != {request_key, row_hashes_key, evidence_key}:
        raise ValueError
    if payload.get(request_key) != request["measurement_request_sha256"]:
        raise ValueError
    row_hashes = payload.get(row_hashes_key)
    evidence = payload.get(evidence_key)
    if not isinstance(row_hashes, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError
    if set(row_hashes) != set(expected) or set(evidence) != set(expected):
        raise ValueError
    if row_hashes != {row_id: values["row_sha256"] for row_id, values in expected.items()}:
        raise ValueError
    if evidence != {
        row_id: values["source_evidence_sha256"] for row_id, values in expected.items()
    }:
        raise ValueError


def _read_private_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise OSError
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > MAX_PRIVATE_JSON_BYTES:
        raise OSError
    payload = json.loads(
        resolved.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(payload, Mapping):
        raise ValueError
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError
        payload[key] = value
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _mkdir_private(path: Path, boundary: Path) -> None:
    if not _beneath(path, boundary):
        raise OSError
    relative = path.relative_to(boundary)
    current = boundary
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise OSError
        current.mkdir(exist_ok=True)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_detached(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True))


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _normalized_model(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.upper() if character.isalnum())


def _beneath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _request_invalid() -> None:
    raise P6HistoryMeasurementError("history_request_invalid")
