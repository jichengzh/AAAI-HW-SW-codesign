"""P6 post-source performance plan and execution adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
)
from framework.stage6.p6_round_adapter_runtime_v1 import (
    LeafRunner,
    P6RoundAdapterRuntimeError,
    RoundContext,
    advance_task_state,
    load_round_context,
)


TERMINAL_NATIVE_STATUSES = frozenset({"success", "confirmed_failure"})
NATIVE_MANIFEST_SCHEMA = "stage5_performance_manifest_v2"
NATIVE_MANIFEST_ROW_SCHEMA = "stage5_performance_manifest_row_v1"
NATIVE_JOB_SCHEMA = "stage5_performance_job_v1"
NATIVE_STATE_SCHEMA = "stage3_execute_performance_plan_v3_state"
OWNED_NATIVE_FILES = (
    "performance_manifest.json",
    "performance_jobs.jsonl",
    "performance_state.jsonl",
)
OWNED_NATIVE_DIRECTORIES = ("attempts", "artifacts")


class P6PerformanceRoundAdapterError(ValueError):
    """Stable path-free performance adapter failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


@dataclass(frozen=True)
class _QuantBinding:
    path: Path
    sha256: str


def run_performance_round(
    profile: ValidatedPostSourceAdapterProfile,
    task_state: Path,
    round_root: Path,
    runner: LeafRunner,
) -> None:
    """Build and execute one performance plan, then advance after validation."""
    try:
        context = load_round_context(
            profile,
            task_state,
            round_root,
            prior_stage="quantization",
        )
        plan_leaf = _single_leaf(context.profile, "performance_plan")
        execute_leaf = _single_leaf(context.profile, "performance_execute")
        performance_root = context.round_root / "performance"
        _require_fresh_native_outputs(performance_root)
        quant_bindings = _validate_int8_quant_contracts(context)
        _run_planner(context, plan_leaf, performance_root, runner)
        jobs = _validate_native_plan(context, performance_root, quant_bindings)
        _run_executor(context, execute_leaf, performance_root, runner)
        _validate_native_state(
            performance_root / "performance_state.jsonl",
            jobs,
            performance_root,
        )
        advance_task_state(context, "performance")
    except (P6PerformanceRoundAdapterError, P6RoundAdapterRuntimeError):
        raise P6PerformanceRoundAdapterError() from None
    except Exception:
        raise P6PerformanceRoundAdapterError() from None


def _run_planner(
    context: RoundContext,
    leaf: PostSourceLeaf,
    performance_root: Path,
    runner: LeafRunner,
) -> None:
    argv = (
        str(context.profile.project_python),
        str(leaf.implementation),
        "--request-json",
        str(context.round_root / "measurement-request.json"),
        "--remote-artifact-root",
        str(performance_root / "artifacts"),
        "--output-dir",
        str(performance_root),
        "--quant-contract-root",
        str(context.round_root / "quant_contracts"),
        "--gpus",
        _gpu_csv(context),
    )
    _require_zero(runner.run(argv, cwd=leaf.implementation_cwd, env=_leaf_env(context), shell=False))


def _run_executor(
    context: RoundContext,
    leaf: PostSourceLeaf,
    performance_root: Path,
    runner: LeafRunner,
) -> None:
    argv = (
        str(context.profile.project_python),
        str(leaf.implementation),
        "--jobs-jsonl",
        str(performance_root / "performance_jobs.jsonl"),
        "--state-jsonl",
        str(performance_root / "performance_state.jsonl"),
        "--gpus",
        _gpu_csv(context),
        "--max-workers",
        str(len(context.gpu_indices)),
    )
    _require_zero(runner.run(argv, cwd=leaf.implementation_cwd, env=_leaf_env(context), shell=False))


def _validate_native_plan(
    context: RoundContext,
    performance_root: Path,
    quant_bindings: Mapping[str, _QuantBinding],
) -> tuple[Mapping[str, Any], ...]:
    manifest = _read_mapping(performance_root / "performance_manifest.json")
    jobs = _read_jsonl_mappings(performance_root / "performance_jobs.jsonl")
    request_rows = tuple(context.request["rows"])
    manifest_rows = _sequence(manifest.get("jobs"))
    if len(jobs) != 4:
        raise P6PerformanceRoundAdapterError()
    _validate_manifest_header(context, manifest, request_rows)
    for index, (request_row, manifest_row, job) in enumerate(
        zip(request_rows, manifest_rows, jobs, strict=True)
    ):
        binding = quant_bindings.get(str(request_row["manifest_job_id"]))
        _validate_manifest_row(request_row, manifest_row, binding)
        _validate_performance_job(
            context,
            index,
            request_row,
            manifest_row,
            job,
            binding,
            performance_root,
        )
    return jobs


def _validate_int8_quant_contracts(
    context: RoundContext,
) -> Mapping[str, _QuantBinding]:
    bindings: dict[str, _QuantBinding] = {}
    for row in context.request["rows"]:
        if row["q_mode"] != "int8":
            continue
        path = _quant_contract_path(context.round_root, row["width"])
        payload = _read_mapping(path)
        _validate_quant_payload(payload, row["source_contract"])
        bindings[str(row["manifest_job_id"])] = _QuantBinding(path, _sha256_file(path))
    return bindings


def _quant_contract_path(round_root: Path, width: object) -> Path:
    values = _valid_width(width)
    return round_root / "quant_contracts" / "x".join(map(str, values)) / "tensor_quant_params.json"


def _valid_width(width: object) -> tuple[int, int, int]:
    if (
        not isinstance(width, list)
        or len(width) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in width)
    ):
        raise P6PerformanceRoundAdapterError()
    return (width[0], width[1], width[2])


def _validate_quant_payload(payload: Mapping[str, Any], source: object) -> None:
    if not isinstance(source, Mapping):
        raise P6PerformanceRoundAdapterError()
    if payload.get("schema") != "stage3_tvm_int8_quant_contract_v3":
        raise P6PerformanceRoundAdapterError()
    params = payload.get("params")
    if not isinstance(params, Mapping) or not params:
        raise P6PerformanceRoundAdapterError()
    fields = (
        ("onnx_path", "onnx_path", "onnx_sha256"),
        ("calibration_npz", "calibration_npz", "calibration_npz_sha256"),
        ("calibration_summary", "calibration_summary", "calibration_summary_sha256"),
    )
    for source_key, path_key, sha_key in fields:
        source_path = _plain_input_file(source.get(source_key))
        if _resolved_payload_path(payload.get(path_key)) != source_path.resolve(strict=False):
            raise P6PerformanceRoundAdapterError()
        if payload.get(sha_key) != _sha256_file(source_path):
            raise P6PerformanceRoundAdapterError()


def _plain_input_file(value: object) -> Path:
    if not isinstance(value, str) or not value or any(char in value for char in "\x00\r\n"):
        raise P6PerformanceRoundAdapterError()
    path = Path(value)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise OSError
    except OSError:
        raise P6PerformanceRoundAdapterError() from None
    return path


def _resolved_payload_path(value: object) -> Path:
    if not isinstance(value, str) or not value or any(char in value for char in "\x00\r\n"):
        raise P6PerformanceRoundAdapterError()
    try:
        return Path(value).resolve(strict=False)
    except OSError:
        raise P6PerformanceRoundAdapterError() from None


def _require_fresh_native_outputs(performance_root: Path) -> None:
    files = tuple(performance_root / name for name in OWNED_NATIVE_FILES)
    directories = tuple(performance_root / name for name in OWNED_NATIVE_DIRECTORIES)
    if any(path.is_symlink() or path.exists() for path in files):
        raise P6PerformanceRoundAdapterError()
    if any(_directory_has_native_entries(path) for path in directories):
        raise P6PerformanceRoundAdapterError()


def _directory_has_native_entries(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        if not path.exists():
            return False
        if not path.is_dir():
            return True
        return next(path.iterdir(), None) is not None
    except OSError:
        raise P6PerformanceRoundAdapterError()


def _validate_manifest_header(
    context: RoundContext,
    manifest: Mapping[str, Any],
    request_rows: tuple[Mapping[str, Any], ...],
) -> None:
    expected = {
        "schema_version": NATIVE_MANIFEST_SCHEMA,
        "source_request_schema": context.request["schema_version"],
        "source_request_sha256": context.request["measurement_request_sha256"],
        "task_id": context.request["task_id"],
        "task_sha256": context.request["task_sha256"],
        "source_pool": "stage5_online_feedback",
        "genome_count": 4,
        "row_count": 4,
        "group_count": len({str(row["group_id"]) for row in request_rows}),
        "group_ids": sorted({str(row["group_id"]) for row in request_rows}),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise P6PerformanceRoundAdapterError()


def _validate_manifest_row(
    request_row: object,
    native_row: object,
    quant_binding: _QuantBinding | None,
) -> None:
    if not isinstance(request_row, Mapping) or not isinstance(native_row, Mapping):
        raise P6PerformanceRoundAdapterError()
    excluded = frozenset({"schema_version", "source_contract", "source_evidence_sha256"})
    if any(native_row.get(key) != value for key, value in request_row.items() if key not in excluded):
        raise P6PerformanceRoundAdapterError()
    expected = {
        "schema_version": NATIVE_MANIFEST_ROW_SCHEMA,
        "job_id": request_row["manifest_job_id"],
        "manifest_job_id": request_row["manifest_job_id"],
        "split": "online_feedback",
        "source_pool": "stage5_online_feedback",
        "required_metrics": ["latency", "energy", "ap"],
        "source_status": "ready",
        "source_plan_sha256": request_row["source_evidence_sha256"],
        "terminal_status": "pending",
    }
    if any(native_row.get(key) != value for key, value in expected.items()):
        raise P6PerformanceRoundAdapterError()
    _validate_source_contract(
        request_row.get("source_contract"),
        native_row.get("source_contract"),
        quant_binding,
    )


def _validate_source_contract(
    request_contract: object,
    native_contract: object,
    quant_binding: _QuantBinding | None,
) -> None:
    if not isinstance(request_contract, Mapping) or not isinstance(native_contract, Mapping):
        raise P6PerformanceRoundAdapterError()
    if any(native_contract.get(key) != value for key, value in request_contract.items()):
        raise P6PerformanceRoundAdapterError()
    quant_keys = ("tensor_quant_params_json", "tensor_quant_params_sha256")
    if quant_binding is None:
        if any(key in native_contract for key in quant_keys):
            raise P6PerformanceRoundAdapterError()
        return
    expected = {
        "tensor_quant_params_json": str(quant_binding.path),
        "tensor_quant_params_sha256": quant_binding.sha256,
    }
    if any(native_contract.get(key) != value for key, value in expected.items()):
        raise P6PerformanceRoundAdapterError()


def _validate_performance_job(
    context: RoundContext,
    index: int,
    request_row: Mapping[str, Any],
    manifest_row: object,
    job: Mapping[str, Any],
    quant_binding: _QuantBinding | None,
    performance_root: Path,
) -> None:
    if not isinstance(manifest_row, Mapping):
        raise P6PerformanceRoundAdapterError()
    runner_key = _native_runner_key(request_row)
    source_contract = manifest_row.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise P6PerformanceRoundAdapterError()
    expected = {
        "schema_version": NATIVE_JOB_SCHEMA,
        "job_id": f"{request_row['group_id']}|{runner_key}",
        "manifest_job_id": request_row["manifest_job_id"],
        "group_id": request_row["group_id"],
        "model": request_row["model"],
        "width_key": "x".join(str(value) for value in request_row["width"]),
        "q_mode": request_row["q_mode"],
        "runner_key": runner_key,
        "dispatch_key": request_row["dispatch_key"],
        "split": "online_feedback",
        "source_contract": source_contract,
        "onnx_path": source_contract.get("onnx_path"),
        "calibration_root": source_contract.get("calibration_root"),
        "assigned_gpu": int(context.gpu_indices[index % len(context.gpu_indices)]),
        "gpu_pool": _gpu_csv(context),
        "max_attempts": 2,
        "terminal_status": "pending",
    }
    if any(job.get(key) != value for key, value in expected.items()):
        raise P6PerformanceRoundAdapterError()
    _validate_source_contract(request_row.get("source_contract"), source_contract, quant_binding)
    _validate_job_outputs(job, performance_root)
    _validate_job_quant_command(job, quant_binding)


def _validate_job_outputs(job: Mapping[str, Any], performance_root: Path) -> None:
    artifact_root = performance_root / "artifacts"
    if _resolved_payload_path(job.get("remote_artifact_root")) != artifact_root.resolve(
        strict=False
    ):
        raise P6PerformanceRoundAdapterError()
    expected_result = _resolved_payload_path(job.get("expected_result_json"))
    if not _is_relative_to(expected_result, artifact_root.resolve(strict=False)):
        raise P6PerformanceRoundAdapterError()


def _validate_job_quant_command(
    job: Mapping[str, Any],
    quant_binding: _QuantBinding | None,
) -> None:
    command = job.get("command")
    if not isinstance(command, list) or not command or any(
        not isinstance(value, str) or not value or "\x00" in value for value in command
    ):
        raise P6PerformanceRoundAdapterError()
    flag = "--tensor-quant-params-json"
    if quant_binding is None:
        if flag in command:
            raise P6PerformanceRoundAdapterError()
        return
    if command.count(flag) != 1:
        raise P6PerformanceRoundAdapterError()
    index = command.index(flag)
    if index + 1 >= len(command):
        raise P6PerformanceRoundAdapterError()
    if _resolved_payload_path(command[index + 1]) != quant_binding.path.resolve(strict=False):
        raise P6PerformanceRoundAdapterError()


def _native_runner_key(row: Mapping[str, Any]) -> str:
    mapping = {
        ("tvm_auto", "fp16"): "tvm_fp16",
        ("tvm_auto", "int8"): "tvm_int8",
        ("trt_engine", "fp16"): "trt_fp16",
        ("trt_engine", "int8"): "trt_int8",
    }
    try:
        return mapping[(str(row["dispatch_key"]), str(row["q_mode"]))]
    except (KeyError, TypeError):
        raise P6PerformanceRoundAdapterError() from None


def _validate_native_state(
    state_path: Path,
    jobs: tuple[Mapping[str, Any], ...],
    performance_root: Path,
) -> None:
    rows = _read_jsonl_mappings(state_path)
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    grouped: dict[str, list[Mapping[str, Any]]] = {job_id: [] for job_id in jobs_by_id}
    for row in rows:
        job_id = str(row.get("job_id") or "")
        if job_id not in jobs_by_id:
            raise P6PerformanceRoundAdapterError()
        _validate_native_state_row(row, jobs_by_id[job_id], performance_root)
        grouped[job_id].append(row)
    if any(not job_rows for job_rows in grouped.values()):
        raise P6PerformanceRoundAdapterError()
    for job_id, job_rows in grouped.items():
        _validate_terminal_rows(job_rows, jobs_by_id[job_id])


def _validate_native_state_row(
    row: Mapping[str, Any],
    job: Mapping[str, Any],
    performance_root: Path,
) -> None:
    required = {
        "schema_version",
        "job_id",
        "attempt",
        "status",
        "returncode",
        "start_time_unix",
        "end_time_unix",
        "elapsed_s",
        "stdout_path",
        "stderr_path",
        "result_json",
        "result_sha256",
        "failure_reasons",
    }
    if set(row) != required or row.get("schema_version") != NATIVE_STATE_SCHEMA:
        raise P6PerformanceRoundAdapterError()
    attempt = row.get("attempt")
    returncode = row.get("returncode")
    status = row.get("status")
    max_attempts = job.get("max_attempts")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or attempt not in range(1, max_attempts + 1)
        or isinstance(returncode, bool)
        or not isinstance(returncode, int)
        or status not in {"failed", *TERMINAL_NATIVE_STATUSES}
    ):
        raise P6PerformanceRoundAdapterError()
    _validate_state_timing(row)
    reasons = row.get("failure_reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise P6PerformanceRoundAdapterError()
    if status == "success":
        if returncode != 0 or reasons:
            raise P6PerformanceRoundAdapterError()
        _validate_state_result(row, job, performance_root, require_metrics=True)
    else:
        if not reasons:
            raise P6PerformanceRoundAdapterError()
        _validate_optional_state_result(row, job, performance_root)


def _validate_state_timing(row: Mapping[str, Any]) -> None:
    started = _finite_number(row.get("start_time_unix"))
    ended = _finite_number(row.get("end_time_unix"))
    elapsed = _finite_number(row.get("elapsed_s"))
    if started is None or ended is None or elapsed is None or ended < started or elapsed < 0:
        raise P6PerformanceRoundAdapterError()
    for key in ("stdout_path", "stderr_path"):
        value = row.get(key)
        if not isinstance(value, str) or not value or any(char in value for char in "\x00\r\n"):
            raise P6PerformanceRoundAdapterError()


def _validate_optional_state_result(
    row: Mapping[str, Any],
    job: Mapping[str, Any],
    performance_root: Path,
) -> None:
    result_path = row.get("result_json")
    result_sha = row.get("result_sha256")
    if result_path is None and result_sha is None:
        return
    if result_path is None or result_sha is None:
        raise P6PerformanceRoundAdapterError()
    _validate_state_result(row, job, performance_root, require_metrics=False)


def _validate_state_result(
    row: Mapping[str, Any],
    job: Mapping[str, Any],
    performance_root: Path,
    *,
    require_metrics: bool,
) -> None:
    path = _plain_input_file(row.get("result_json"))
    if path.resolve(strict=False) not in _native_result_candidates(job, performance_root, row):
        raise P6PerformanceRoundAdapterError()
    if row.get("result_sha256") != _sha256_file(path):
        raise P6PerformanceRoundAdapterError()
    if require_metrics:
        _validate_success_result_payload(_read_mapping(path))


def _validate_terminal_rows(
    rows: list[Mapping[str, Any]],
    job: Mapping[str, Any],
) -> None:
    terminals = [row for row in rows if row.get("status") in TERMINAL_NATIVE_STATUSES]
    if len(terminals) != 1:
        raise P6PerformanceRoundAdapterError()
    terminal = terminals[0]
    if terminal["status"] == "confirmed_failure" and terminal["attempt"] != job["max_attempts"]:
        raise P6PerformanceRoundAdapterError()
    if any(
        row["status"] == "failed" and row["attempt"] > terminal["attempt"]
        for row in rows
    ):
        raise P6PerformanceRoundAdapterError()


def _single_leaf(
    profile: ValidatedPostSourceAdapterProfile,
    name: str,
) -> PostSourceLeaf:
    matches = tuple(leaf for leaf in profile.leaves if leaf.name == name)
    if len(matches) != 1:
        raise P6PerformanceRoundAdapterError()
    return matches[0]


def _leaf_env(context: RoundContext) -> dict[str, str]:
    inherited = _validated_incoming_env(context)
    return {
        "CUDA_VISIBLE_DEVICES": _gpu_csv(context),
        **inherited,
        "PATH": os.pathsep.join((str(context.profile.project_python.parent), "/usr/bin", "/bin")),
        "PYTHONPATH": os.pathsep.join((str(context.profile.private_root), str(Path.cwd()))),
    }


def _validated_incoming_env(context: RoundContext) -> dict[str, str]:
    keys = (
        "P6_HISTORY_RUN_MODE",
        "P6_HISTORY_PRIVATE_ROOT",
        "P6_HISTORY_TASK_STATE",
        "P6_HISTORY_ROUND_OUTPUT_ROOT",
    )
    inherited = {key: os.environ[key] for key in keys if key in os.environ}
    if set(inherited) != set(keys):
        raise P6PerformanceRoundAdapterError()
    try:
        private_root = Path(inherited["P6_HISTORY_PRIVATE_ROOT"]).resolve(strict=False)
        task_state = Path(inherited["P6_HISTORY_TASK_STATE"]).resolve(strict=False)
        round_root = Path(inherited["P6_HISTORY_ROUND_OUTPUT_ROOT"]).resolve(strict=False)
    except OSError:
        raise P6PerformanceRoundAdapterError()
    if (
        private_root != context.profile.private_root.resolve(strict=False)
        or task_state != context.task_state_path
        or round_root != context.round_root
    ):
        raise P6PerformanceRoundAdapterError()
    return inherited


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise OSError
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise P6PerformanceRoundAdapterError() from None
    if not isinstance(payload, Mapping):
        raise P6PerformanceRoundAdapterError()
    return payload


def _native_result_candidates(
    job: Mapping[str, Any],
    performance_root: Path,
    state_row: Mapping[str, Any],
) -> set[Path]:
    attempt = int(state_row["attempt"])
    slug = str(job["job_id"]).replace("|", "__").replace("/", "_")
    work_dir = performance_root / "attempts" / slug / f"attempt_{attempt:02d}"
    candidates = _declared_result_candidates(job, work_dir)
    state_path = _resolved_payload_path(state_row.get("result_json"))
    known_names = {
        "route_b_fp16_auto_result.json",
        "route_b_int8_auto_decomp_result.json",
        "trt_profile_result.json",
    }
    if state_path.name in known_names and _is_relative_to(
        state_path, work_dir.resolve(strict=False)
    ):
        candidates.add(state_path)
    return {path.resolve(strict=False) for path in candidates}


def _declared_result_candidates(job: Mapping[str, Any], work_dir: Path) -> set[Path]:
    expected = Path(str(job["expected_result_json"]))
    candidates = {expected, work_dir / expected.name}
    command = job["command"]
    if "--out" in command:
        output = _flag_path(command, "--out")
        candidates.update((output, work_dir / output.name))
    if "--out-dir" in command:
        output_dir = _flag_path(command, "--out-dir")
        names = ("route_b_fp16_auto_result.json", "route_b_int8_auto_decomp_result.json")
        candidates.update(output_dir / name for name in names)
        candidates.update(work_dir / output_dir.name / name for name in names)
        if "--label" in command:
            label = command[command.index("--label") + 1]
            candidates.update(output_dir / label / name for name in names)
    fallback = (
        "route_b_fp16_auto_result.json"
        if job["runner_key"] == "tvm_fp16"
        else "route_b_int8_auto_decomp_result.json"
        if job["runner_key"] == "tvm_int8"
        else "trt_profile_result.json"
    )
    candidates.add(work_dir / fallback)
    return candidates


def _flag_path(command: list[str], flag: str) -> Path:
    index = command.index(flag)
    if index + 1 >= len(command):
        raise P6PerformanceRoundAdapterError()
    return Path(command[index + 1])


def _validate_success_result_payload(payload: Mapping[str, Any]) -> None:
    if _extract_latency(payload) is None or _extract_energy(payload) is None:
        raise P6PerformanceRoundAdapterError()
    success = payload.get("status") == "success" or payload.get("build_success") is True
    correctness = payload.get("numerical_finite") is True or payload.get(
        "correctness_all_exact"
    ) is True
    correctness = correctness or _nonempty_list(payload.get("correctness_vs_default_fp16"))
    correctness = correctness or _nonempty_list(payload.get("correctness_vs_native_direct"))
    if not success or not correctness:
        raise P6PerformanceRoundAdapterError()


def _extract_latency(payload: Mapping[str, Any]) -> float | None:
    nested = payload.get("latency")
    candidates = (
        payload.get("lat_p50_ms"),
        payload.get("latency_ms"),
        nested.get("latency_ms_p50") if isinstance(nested, Mapping) else None,
        nested.get("lat_p50_ms") if isinstance(nested, Mapping) else None,
    )
    return next((value for item in candidates if (value := _finite_number(item)) is not None), None)


def _extract_energy(payload: Mapping[str, Any]) -> float | None:
    nested = payload.get("energy")
    nested_keys = ("energy_j", "joules", "joules_per_inference", "joule_per_inference", "energy_J")
    candidates = [payload.get("energy_j")]
    if isinstance(nested, Mapping):
        candidates.extend(nested.get(key) for key in nested_keys)
    return next((value for item in candidates if (value := _finite_number(item)) is not None), None)


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonempty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise P6PerformanceRoundAdapterError() from None
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_jsonl_mappings(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise OSError
        rows = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise P6PerformanceRoundAdapterError() from None
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise P6PerformanceRoundAdapterError()
    return rows


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise P6PerformanceRoundAdapterError()
    return tuple(value)


def _gpu_csv(context: RoundContext) -> str:
    return ",".join(context.gpu_indices)


def _require_zero(result: object) -> None:
    returncode = getattr(result, "returncode", None)
    if isinstance(returncode, bool) or returncode != 0:
        raise P6PerformanceRoundAdapterError()
