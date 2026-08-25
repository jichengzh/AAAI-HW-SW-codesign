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
    expected = tuple(str(row["manifest_job_id"]) for row in context.request["rows"])
    manifest_jobs = manifest.get("jobs", manifest.get("rows"))
    if tuple(_identity(row) for row in _sequence(manifest_jobs)) != expected:
        raise P6PerformanceRoundAdapterError()
    if tuple(_identity(row) for row in jobs) != expected:
        raise P6PerformanceRoundAdapterError()
    return jobs


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


def _identity(row: object) -> str:
    if not isinstance(row, Mapping):
        raise P6PerformanceRoundAdapterError()
    value = row.get("manifest_job_id", row.get("row_id"))
    if not isinstance(value, str) or not value:
        raise P6PerformanceRoundAdapterError()
    return value


def _gpu_csv(context: RoundContext) -> str:
    return ",".join(context.gpu_indices)


def _require_zero(result: object) -> None:
    returncode = getattr(result, "returncode", None)
    if isinstance(returncode, bool) or returncode != 0:
        raise P6PerformanceRoundAdapterError()
