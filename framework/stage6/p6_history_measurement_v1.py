"""Fail-closed execution bridge from one P6 batch to private history feedback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Protocol

from framework.stage5.genome_contract_v1 import validate_structure_identity
from framework.stage5.production_search_v1 import validate_source_contract
from framework.stage6.p6_history_binding_v1 import (
    ALLOWED_NORMALIZED_H800_MODELS,
    EXPECTED_HISTORY_ENV_KEYS,
    MAX_GPU_OCCUPANCY,
    GpuProbe,
    GpuRecord,
    P6HistoryBindingError,
    validate_history_execution_binding,
)
from framework.stage6.p6_history_feedback_validation_v1 import (
    P6HistoryFeedbackValidationError,
    translate_history_feedback,
)
from framework.stage6.p6_history_source_materialization_v1 import (
    P6HistorySourceMaterializationError,
    build_source_invocations,
    project_source_materialization_request,
    run_source_invocations,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (
    P6FreshRunContext,
    P6SourceReuseEvidenceError,
    classify_selected_group_sources,
    first_use_group_ids,
    load_fresh_run_context,
    require_selected_groups_ready_current_run,
    requires_current_run_source_evidence,
    validate_and_publish_group_receipt,
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
    try:
        projected = project_source_materialization_request(verified_request)
    except P6HistorySourceMaterializationError:
        raise P6HistoryMeasurementError("history_execution_invalid") from None
    canonical_request = projected.request
    private_root, gpu_policy = _validate_binding_runtime(binding)
    _validate_interface_gpu_policy(interface, gpu_policy)
    paths = _resolve_round_paths(
        interface, private_root, Path(round_output_root), canonical_request["round_index"]
    )
    run_context, source_group_ids = _plan_current_run_sources(
        canonical_request,
        projected.ordered_group_ids,
        interface=interface,
        private_root=private_root,
        local_output_root=paths["local_output_root"],
    )
    source_invocations = (
        build_source_invocations(
            paths["measurement_request"],
            source_group_ids,
            source_materializer=Path(interface["execution_chain"][0]["argv"][0]),
            validated_gpu_policy=gpu_policy,
        )
        if source_group_ids
        else ()
    )
    _validate_gpu(gpu_probe, gpu_policy)
    _initialize_private_round(canonical_request, interface, paths)
    environment = _render_environment(interface, paths)
    substitutions = _substitutions(paths)
    _execute(interface["environment"]["activation_argv"], substitutions, runner, paths, environment)
    try:
        if run_context is None:
            run_source_invocations(
                source_invocations,
                runner=runner,
                cwd=paths["round_root"],
                env=environment,
            )
        else:
            _run_first_use_sources(
                canonical_request,
                source_group_ids,
                source_invocations,
                run_context=run_context,
                interface=interface,
                private_root=private_root,
                paths=paths,
                runner=runner,
                environment=environment,
            )
            require_selected_groups_ready_current_run(
                canonical_request,
                run_context=run_context,
                local_output_root=paths["local_output_root"],
                interface=interface,
                private_root=private_root,
            )
    except (P6HistorySourceMaterializationError, P6SourceReuseEvidenceError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None
    for stage in interface["execution_chain"][1:]:
        _execute(stage["argv"], substitutions, runner, paths, environment)
    _validate_gpu(gpu_probe, gpu_policy)
    return _translate_feedback(canonical_request, interface, paths, private_root)


def _plan_current_run_sources(request: Mapping[str, Any],
    ordered_group_ids: Sequence[str], *, interface: Mapping[str, Any],
    private_root: Path, local_output_root: Path,
) -> tuple[P6FreshRunContext | None, tuple[str, ...]]:
    try:
        if not requires_current_run_source_evidence(request):
            return None, tuple(ordered_group_ids)
        context = load_fresh_run_context(
            local_output_root=local_output_root,
            expected_task_id=request["task_id"],
            expected_task_sha256=request["task_sha256"],
        )
        decisions = classify_selected_group_sources(
            request,
            run_context=context,
            local_output_root=local_output_root,
            interface=interface,
            private_root=private_root,
        )
        return context, first_use_group_ids(decisions)
    except P6SourceReuseEvidenceError:
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _run_first_use_sources(request: Mapping[str, Any], group_ids: Sequence[str],
    invocations: Sequence[Sequence[str]], *, run_context: P6FreshRunContext,
    interface: Mapping[str, Any], private_root: Path, paths: Mapping[str, Path],
    runner: Runner, environment: Mapping[str, str]) -> None:
    if len(group_ids) != len(invocations):
        raise P6SourceReuseEvidenceError("history_execution_mismatch")
    for group_id, invocation in zip(group_ids, invocations, strict=True):
        decisions = classify_selected_group_sources(
            request,
            run_context=run_context,
            local_output_root=paths["local_output_root"],
            interface=interface,
            private_root=private_root,
        )
        decision_by_group = {item.group_id: item for item in decisions}
        if decision_by_group.get(group_id) is None or (
            decision_by_group[group_id].state != "UNSEEN"
        ):
            raise P6SourceReuseEvidenceError("history_execution_mismatch")
        run_source_invocations(
            (invocation,),
            runner=runner,
            cwd=paths["round_root"],
            env=environment,
        )
        validate_and_publish_group_receipt(
            request,
            group_id=group_id,
            run_context=run_context,
            local_output_root=paths["local_output_root"],
            interface=interface,
            private_root=private_root,
        )

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
        raw_indices = raw_policy.get("indices")
        uuid_by_index = raw_policy.get("uuid_by_index")
        if (
            not isinstance(raw_indices, list)
            or len(raw_indices) != 3
            or any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indices)
            or len(set(raw_indices)) != 3
            or any(index < 0 for index in raw_indices)
            or raw_policy.get("model") != "h800"
            or raw_policy.get("maximum_occupancy") != MAX_GPU_OCCUPANCY
            or not isinstance(uuid_by_index, Mapping)
            or set(uuid_by_index) != {str(index) for index in raw_indices}
        ):
            raise ValueError
        indices: tuple[int, int, int] = (
            raw_indices[0], raw_indices[1], raw_indices[2]
        )
        raw_uuids = [uuid_by_index[str(index)] for index in indices]
        if any(not isinstance(uuid, str) for uuid in raw_uuids):
            raise ValueError
        uuids = [uuid.strip() for uuid in raw_uuids]
        if any(not uuid for uuid in uuids) or len(set(uuids)) != len(uuids):
            raise ValueError
        policy = {
            "indices": indices,
            "uuid_by_index": {
                str(index): uuid for index, uuid in zip(indices, uuids, strict=True)
            },
            "model": "h800",
            "maximum_occupancy": MAX_GPU_OCCUPANCY,
        }
        return private_root, policy
    except (KeyError, OSError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None

def _validate_gpu(gpu_probe: GpuProbe, policy: Mapping[str, Any]) -> None:
    try:
        indices = policy["indices"]
        snapshot = gpu_probe.snapshot(indices)
        if not isinstance(snapshot, tuple) or len(snapshot) != len(indices):
            raise ValueError
        if any(not isinstance(record, GpuRecord) for record in snapshot):
            raise ValueError
        by_index = {record.index: record for record in snapshot}
        if set(by_index) != set(indices) or len(by_index) != len(snapshot):
            raise ValueError
        ordered = [by_index[index] for index in indices]
        if any(not isinstance(record.uuid, str) for record in ordered):
            raise ValueError
        observed_uuids = [record.uuid.strip() for record in ordered]
        if any(not uuid for uuid in observed_uuids) or len(set(observed_uuids)) != len(
            observed_uuids
        ):
            raise ValueError
        for index, observed_uuid in zip(indices, observed_uuids, strict=True):
            record = by_index[index]
            if (
                observed_uuid != policy["uuid_by_index"][str(index)]
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


def _validate_interface_gpu_policy(
    interface: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    try:
        raw_value = interface["environment"]["values"]["CUDA_VISIBLE_DEVICES"]["value"]
        interface_indices = tuple(int(part) for part in raw_value.split(","))
        if interface_indices != policy["indices"]:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _resolve_round_paths(
    interface: Mapping[str, Any],
    private_root: Path,
    supplied_root: Path,
    round_index: int,
) -> dict[str, Path]:
    try:
        validated_supplied_root = _validate_controller_round_root(supplied_root)
        local_output_root = _validate_controller_round_root(
            validated_supplied_root.parent
        )
        round_root = _resolve_template(
            private_root,
            interface["output_layout"]["round_root_template"],
            round_index,
            private_root,
        )
        task_state = _resolve_template(
            private_root,
            interface["output_layout"]["task_state"]["path_template"],
            round_index,
            private_root,
        )
        actual_feedback = interface["actual_feedback"]
        result = _resolve_template(
            private_root,
            actual_feedback["result"]["path_template"],
            round_index,
            private_root,
        )
        receipt = _resolve_template(
            private_root,
            actual_feedback["receipt"]["path_template"],
            round_index,
            private_root,
        )
        barrier = _resolve_template(
            private_root,
            actual_feedback["finalization_barrier"]["path_template"],
            round_index,
            private_root,
        )
        request_path = (round_root / "measurement-request.json").resolve(strict=False)
        paths = {
            "history_root": private_root,
            "local_output_root": local_output_root,
            "round_root": round_root,
            "measurement_request": request_path,
            "task_state": task_state,
            "actual_feedback": result,
            "actual_receipt": receipt,
            "finalization_barrier": barrier,
        }
        artifact_paths = {
            key: path
            for key, path in paths.items()
            if key not in {"history_root", "local_output_root"}
        }
        if len(set(artifact_paths.values())) != len(artifact_paths) or any(
            not _beneath(path, private_root)
            for path in artifact_paths.values()
        ):
            raise ValueError
        return paths
    except (KeyError, OSError, TypeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None


def _validate_controller_round_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError
    return path.resolve(strict=True)

def _resolve_template(
    supplied_root: Path, template: object, round_index: int, private_root: Path
) -> Path:
    if not isinstance(template, str):
        raise ValueError
    lexical_path = supplied_root / template.replace("{round_id}", str(round_index))
    _reject_symlink_components(lexical_path, supplied_root, allow_missing=True)
    path = lexical_path.resolve(strict=False)
    if not _beneath(path, supplied_root) or not _beneath(path, private_root):
        raise ValueError
    return path

def _initialize_private_round(
    request: Mapping[str, Any], interface: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    try:
        round_root = paths["round_root"]
        _mkdir_private(round_root, paths["history_root"])
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
            _mkdir_private(path.parent, paths["history_root"])
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
        if set(rendered) != set(EXPECTED_HISTORY_ENV_KEYS):
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
    private_root: Path,
) -> dict[str, Any]:
    try:
        return translate_history_feedback(
            request,
            interface,
            paths,
            read_json=lambda path: _read_private_json(
                path, paths["history_root"], private_root
            ),
        )
    except P6HistoryFeedbackValidationError:
        raise P6HistoryMeasurementError("history_execution_invalid") from None

def _read_private_json(path: Path, supplied_root: Path, private_root: Path) -> Mapping[str, Any]:
    _reject_symlink_components(path, supplied_root)
    resolved = path.resolve(strict=True)
    if (
        not _beneath(resolved, supplied_root)
        or not _beneath(resolved, private_root)
        or not resolved.is_file()
        or resolved.stat().st_size > MAX_PRIVATE_JSON_BYTES
    ):
        raise OSError
    payload = json.loads(
        resolved.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(payload, Mapping):
        raise ValueError
    return payload

def _reject_symlink_components(
    path: Path, boundary: Path, *, allow_missing: bool = False
) -> None:
    try:
        relative = path.relative_to(boundary)
    except ValueError:
        raise OSError from None
    current = boundary
    boundary_mode = current.lstat().st_mode
    if stat.S_ISLNK(boundary_mode) or not stat.S_ISDIR(boundary_mode):
        raise OSError
    components = relative.parts
    for index, component in enumerate(components):
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if allow_missing:
                return
            raise
        if stat.S_ISLNK(mode):
            raise OSError
        if index < len(components) - 1 and not stat.S_ISDIR(mode):
            raise OSError
        if not allow_missing and index == len(components) - 1 and not stat.S_ISREG(mode):
            raise OSError

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
