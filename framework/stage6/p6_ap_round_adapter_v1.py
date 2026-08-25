"""P6 post-source AP plan and execution adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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


READY_TERMINAL = "ready"
NUMERICAL_FAILURE = "numerical_feasibility_failure"
FULL_NUMERICAL_SKIP = "skipped_numerical_feasibility"


class P6APRoundAdapterError(ValueError):
    """Stable path-free AP adapter failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


def run_ap_round(
    profile: ValidatedPostSourceAdapterProfile,
    task_state: Path,
    round_root: Path,
    runner: LeafRunner,
) -> None:
    """Build and execute one AP plan, then advance after native completeness."""
    try:
        context = load_round_context(profile, task_state, round_root, prior_stage="performance")
        plan_leaf = _single_leaf(context.profile, "ap_plan")
        execute_leaf = _single_leaf(context.profile, "ap_execute")
        ap_root = context.round_root / "ap"
        _run_planner(context, plan_leaf, ap_root, runner)
        rows = _validate_native_plan(context, ap_root)
        shards = _write_plan_shards(context, ap_root, rows)
        _run_stage(context, execute_leaf, "sanity", shards, runner)
        _run_stage(context, execute_leaf, "full", shards, runner)
        state_path = _merge_shard_state(ap_root, len(shards))
        _validate_native_state(state_path, rows)
        advance_task_state(context, "ap")
    except (P6APRoundAdapterError, P6RoundAdapterRuntimeError):
        raise P6APRoundAdapterError() from None
    except Exception:
        raise P6APRoundAdapterError() from None


def _run_planner(
    context: RoundContext,
    leaf: PostSourceLeaf,
    ap_root: Path,
    runner: LeafRunner,
) -> None:
    argv = (
        str(context.profile.project_python),
        str(leaf.implementation),
        "--manifest-json",
        str(context.round_root / "performance/performance_manifest.json"),
        "--performance-jobs-jsonl",
        str(context.round_root / "performance/performance_jobs.jsonl"),
        "--performance-state-jsonl",
        str(context.round_root / "performance/performance_state.jsonl"),
        "--output-root",
        str(context.round_root / "ap_execution"),
        "--output-json",
        str(ap_root / "ap_plan.json"),
        "--output-jsonl",
        str(ap_root / "ap_plan.jsonl"),
    )
    _require_zero(runner.run(argv, cwd=leaf.implementation_cwd, env=_leaf_env(context, None), shell=False))


def _run_stage(
    context: RoundContext,
    leaf: PostSourceLeaf,
    stage: str,
    shards: Sequence[Path],
    runner: LeafRunner,
) -> None:
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = [
            executor.submit(_run_shard, context, leaf, stage, shard_index, shard, runner)
            for shard_index, shard in enumerate(shards)
        ]
        for future in futures:
            _require_zero(future.result())


def _run_shard(
    context: RoundContext,
    leaf: PostSourceLeaf,
    stage: str,
    shard_index: int,
    shard: Path,
    runner: LeafRunner,
) -> object:
    gpu = context.gpu_indices[shard_index]
    argv = (
        str(context.profile.project_python),
        str(leaf.implementation),
        "--ap-plan-jsonl",
        str(shard),
        "--stage",
        stage,
        "--state-jsonl",
        str(shard.parent / f"ap_state_shard_{shard_index}.jsonl"),
        "--gpu",
        gpu,
        "--univ2x-python",
        str(context.profile.project_python),
        "--artifact-root",
        str(context.round_root / f"ap_execution/{stage}/shard_{shard_index}"),
    )
    return runner.run(argv, cwd=leaf.implementation_cwd, env=_leaf_env(context, gpu), shell=False)


def _write_plan_shards(
    context: RoundContext,
    ap_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, ...]:
    ap_root.mkdir(parents=True, exist_ok=True)
    shards = tuple(ap_root / f"ap_plan_shard_{index}.jsonl" for index in range(len(context.gpu_indices)))
    grouped: list[list[Mapping[str, Any]]] = [[] for _ in shards]
    for row_index, row in enumerate(rows):
        grouped[row_index % len(shards)].append(row)
    for path, shard_rows in zip(shards, grouped, strict=True):
        _write_jsonl(path, shard_rows)
    return shards


def _validate_native_plan(
    context: RoundContext,
    ap_root: Path,
) -> tuple[Mapping[str, Any], ...]:
    payload = _read_mapping(ap_root / "ap_plan.json")
    rows = _read_jsonl_mappings(ap_root / "ap_plan.jsonl")
    if (
        payload.get("schema_version") != "stage5_ap_plan_v2"
        or payload.get("row_count") != len(rows)
        or tuple(payload.get("jobs") or ()) != rows
    ):
        raise P6APRoundAdapterError()
    expected = tuple(_request_manifest_id(row) for row in context.request["rows"])
    if tuple(_manifest_job_id(row) for row in rows) != expected:
        raise P6APRoundAdapterError()
    return rows


def _validate_native_state(state_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    terminals = _latest_terminal_rows(_read_jsonl_mappings(state_path, allow_empty=True))
    for row in rows:
        if row.get("ap_terminal") != READY_TERMINAL:
            continue
        job_id = _manifest_job_id(row)
        sanity = terminals.get((job_id, "sanity"))
        full = terminals.get((job_id, "full"))
        if _is_numerical_sanity_terminal(sanity):
            if not _is_full_numerical_skip(full):
                raise P6APRoundAdapterError()
        elif not _is_success_terminal(sanity) or not _is_success_terminal(full):
            raise P6APRoundAdapterError()


def _latest_terminal_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("record_type") != "job_terminal":
            continue
        job_id = row.get("job_id")
        stage = row.get("stage")
        if isinstance(job_id, str) and isinstance(stage, str):
            latest[(job_id, stage)] = row
    return latest


def _is_success_terminal(row: Mapping[str, Any] | None) -> bool:
    return row is not None and row.get("status") == "success"


def _is_numerical_sanity_terminal(row: Mapping[str, Any] | None) -> bool:
    return (
        row is not None
        and row.get("stage") == "sanity"
        and row.get("status") == "failed"
        and row.get("failure_reason") == NUMERICAL_FAILURE
    )


def _is_full_numerical_skip(row: Mapping[str, Any] | None) -> bool:
    return (
        row is not None
        and row.get("stage") == "full"
        and row.get("status") == FULL_NUMERICAL_SKIP
        and row.get("failure_reason") == NUMERICAL_FAILURE
    )


def _merge_shard_state(ap_root: Path, shard_count: int) -> Path:
    state_path = ap_root / "ap_state.jsonl"
    temporary = ap_root / f".{state_path.name}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for index in range(shard_count):
            shard = ap_root / f"ap_state_shard_{index}.jsonl"
            if shard.is_file():
                handle.write(shard.read_text(encoding="utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, state_path)
    _fsync_directory(ap_root)
    return state_path


def _single_leaf(
    profile: ValidatedPostSourceAdapterProfile,
    name: str,
) -> PostSourceLeaf:
    matches = tuple(leaf for leaf in profile.leaves if leaf.name == name)
    if len(matches) != 1:
        raise P6APRoundAdapterError()
    return matches[0]


def _leaf_env(context: RoundContext, gpu: str | None) -> dict[str, str]:
    inherited = _validated_incoming_env(context)
    return {
        "CUDA_VISIBLE_DEVICES": gpu if gpu is not None else ",".join(context.gpu_indices),
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
        raise P6APRoundAdapterError()
    _validate_incoming_paths(context, inherited)
    return inherited


def _validate_incoming_paths(context: RoundContext, inherited: Mapping[str, str]) -> None:
    try:
        private_root = Path(inherited["P6_HISTORY_PRIVATE_ROOT"]).resolve(strict=False)
        task_state = Path(inherited["P6_HISTORY_TASK_STATE"]).resolve(strict=False)
        round_root = Path(inherited["P6_HISTORY_ROUND_OUTPUT_ROOT"]).resolve(strict=False)
    except OSError:
        raise P6APRoundAdapterError() from None
    if (
        private_root != context.profile.private_root.resolve(strict=False)
        or task_state != context.task_state_path
        or round_root != context.round_root
    ):
        raise P6APRoundAdapterError()


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise OSError
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise P6APRoundAdapterError() from None
    if not isinstance(payload, Mapping):
        raise P6APRoundAdapterError()
    return payload


def _read_jsonl_mappings(path: Path, *, allow_empty: bool = False) -> tuple[Mapping[str, Any], ...]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        if not allow_empty and path.stat().st_size <= 0:
            raise OSError
        rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise P6APRoundAdapterError() from None
    if any(not isinstance(row, Mapping) for row in rows):
        raise P6APRoundAdapterError()
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _request_manifest_id(row: object) -> str:
    if not isinstance(row, Mapping):
        raise P6APRoundAdapterError()
    value = row.get("manifest_job_id") or row.get("row_id")
    if not isinstance(value, str) or not value:
        raise P6APRoundAdapterError()
    return value


def _manifest_job_id(row: Mapping[str, Any]) -> str:
    value = row.get("manifest_job_id")
    if not isinstance(value, str) or not value:
        raise P6APRoundAdapterError()
    return value


def _require_zero(result: object) -> None:
    returncode = getattr(result, "returncode", None)
    if isinstance(returncode, bool) or returncode != 0:
        raise P6APRoundAdapterError()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
