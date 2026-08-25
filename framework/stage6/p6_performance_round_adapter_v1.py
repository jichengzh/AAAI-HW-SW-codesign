"""P6 post-source performance plan and execution adapter."""

from __future__ import annotations

from collections.abc import Mapping
import json
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
        _run_planner(context, plan_leaf, performance_root, runner)
        jobs = _validate_native_plan(context, performance_root)
        _run_executor(context, execute_leaf, performance_root, runner)
        _validate_native_state(performance_root / "performance_state.jsonl", jobs)
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
        _validate_manifest_row(request_row, manifest_row)
        _validate_performance_job(context, index, request_row, manifest_row, job)
    return jobs


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


def _validate_manifest_row(request_row: object, native_row: object) -> None:
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
    _validate_source_contract(request_row.get("source_contract"), native_row.get("source_contract"))


def _validate_source_contract(request_contract: object, native_contract: object) -> None:
    if not isinstance(request_contract, Mapping) or not isinstance(native_contract, Mapping):
        raise P6PerformanceRoundAdapterError()
    if any(native_contract.get(key) != value for key, value in request_contract.items()):
        raise P6PerformanceRoundAdapterError()


def _validate_performance_job(
    context: RoundContext,
    index: int,
    request_row: Mapping[str, Any],
    manifest_row: object,
    job: Mapping[str, Any],
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


def _validate_native_state(state_path: Path, jobs: tuple[Mapping[str, Any], ...]) -> None:
    rows = _read_jsonl_mappings(state_path)
    latest: dict[str, str] = {}
    for row in rows:
        job_id = row.get("job_id")
        status = row.get("status")
        if not isinstance(job_id, str) or not isinstance(status, str):
            raise P6PerformanceRoundAdapterError()
        latest[job_id] = status
    expected_job_ids = tuple(str(job["job_id"]) for job in jobs)
    if set(latest) != set(expected_job_ids):
        raise P6PerformanceRoundAdapterError()
    if any(latest[job_id] not in TERMINAL_NATIVE_STATUSES for job_id in expected_job_ids):
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
