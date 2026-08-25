"""P6 post-source AP plan and execution adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
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
ALLOWED_NONREADY_TERMINALS = frozenset(
    {
        "feasibility_failure",
        "blocked_performance_not_success",
        "blocked_result_json_missing",
        "blocked_compiled_artifact_missing",
        "blocked_runner_missing",
    }
)
NONREADY_TERMINALS_REQUIRING_REASON = frozenset(
    {"blocked_compiled_artifact_missing", "blocked_runner_missing"}
)


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
        planner_returncode = _run_planner(context, plan_leaf, ap_root, runner)
        rows = _validate_native_plan(context, ap_root, planner_returncode)
        shards = _write_plan_shards(context, ap_root, rows)
        _run_stage(context, execute_leaf, "sanity", shards, runner)
        _run_stage(context, execute_leaf, "full", shards, runner)
        state_path = _merge_shard_state(ap_root, len(shards))
        _validate_native_state(context, state_path, rows)
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
) -> int:
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
    result = runner.run(argv, cwd=leaf.implementation_cwd, env=_leaf_env(context, None), shell=False)
    returncode = getattr(result, "returncode", None)
    if isinstance(returncode, bool) or returncode not in {0, 1}:
        raise P6APRoundAdapterError()
    return int(returncode)


def _run_stage(
    context: RoundContext,
    leaf: PostSourceLeaf,
    stage: str,
    shards: Sequence[Path],
    runner: LeafRunner,
) -> None:
    active = tuple(
        (index, shard)
        for index, shard in enumerate(shards)
        if _shard_has_ready_rows(shard)
    )
    if not active:
        return
    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        futures = [
            executor.submit(_run_shard, context, leaf, stage, shard_index, shard, runner)
            for shard_index, shard in active
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
    planner_returncode: int,
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
    _validate_plan_returncode(rows, planner_returncode)
    for row in rows:
        _validate_plan_terminal(row)
    return rows


def _validate_plan_returncode(rows: Sequence[Mapping[str, Any]], returncode: int) -> None:
    ready_count = sum(row.get("ap_terminal") == READY_TERMINAL for row in rows)
    if returncode == 0 and ready_count != len(rows):
        raise P6APRoundAdapterError()
    if returncode == 1 and ready_count == len(rows):
        raise P6APRoundAdapterError()


def _validate_plan_terminal(row: Mapping[str, Any]) -> None:
    terminal = row.get("ap_terminal")
    if terminal == READY_TERMINAL:
        return
    if terminal not in ALLOWED_NONREADY_TERMINALS:
        raise P6APRoundAdapterError()
    reason = row.get("block_reason")
    if terminal in NONREADY_TERMINALS_REQUIRING_REASON and not _nonempty_string(reason):
        raise P6APRoundAdapterError()


def _validate_native_state(
    context: RoundContext,
    state_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    terminals = _latest_terminal_rows(_read_jsonl_mappings(state_path, allow_empty=True))
    for row in rows:
        if row.get("ap_terminal") != READY_TERMINAL:
            continue
        job_id = _manifest_job_id(row)
        sanity = terminals.get((job_id, "sanity"))
        full = terminals.get((job_id, "full"))
        if _is_numerical_sanity_terminal(row, sanity):
            _require_terminal_report(context, sanity)
            if not _is_full_numerical_skip(row, full):
                raise P6APRoundAdapterError()
            _require_same_report_evidence(sanity, full)
        elif _is_success_terminal(row, sanity, "sanity") and _is_success_terminal(row, full, "full"):
            _require_terminal_report(context, sanity)
            _require_terminal_report(context, full)
        else:
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


def _is_success_terminal(
    plan: Mapping[str, Any],
    row: Mapping[str, Any] | None,
    stage: str,
) -> bool:
    return (
        row is not None
        and row.get("stage") == stage
        and row.get("status") == "success"
        and row.get("plan_fingerprint") == _plan_fingerprint(plan, stage)
    )


def _is_numerical_sanity_terminal(
    plan: Mapping[str, Any],
    row: Mapping[str, Any] | None,
) -> bool:
    return (
        row is not None
        and row.get("stage") == "sanity"
        and row.get("status") == "failed"
        and row.get("failure_reason") == NUMERICAL_FAILURE
        and row.get("plan_fingerprint") == _plan_fingerprint(plan, "sanity")
    )


def _is_full_numerical_skip(
    plan: Mapping[str, Any],
    row: Mapping[str, Any] | None,
) -> bool:
    return (
        row is not None
        and row.get("stage") == "full"
        and row.get("status") == FULL_NUMERICAL_SKIP
        and row.get("failure_reason") == NUMERICAL_FAILURE
        and row.get("plan_fingerprint") == _plan_fingerprint(plan, "full")
    )


def _require_same_report_evidence(
    sanity: Mapping[str, Any] | None,
    full: Mapping[str, Any] | None,
) -> None:
    if (
        sanity is None
        or full is None
        or full.get("report_path") != sanity.get("report_path")
        or full.get("report_sha256") != sanity.get("report_sha256")
    ):
        raise P6APRoundAdapterError()


def _require_terminal_report(context: RoundContext, row: Mapping[str, Any] | None) -> None:
    if row is None:
        raise P6APRoundAdapterError()
    report = _canonical_report_path(row.get("report_path"), context.round_root / "ap_execution")
    expected = row.get("report_sha256")
    if not isinstance(expected, str) or not expected or _sha256_file(report) != expected:
        raise P6APRoundAdapterError()


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


def _shard_has_ready_rows(path: Path) -> bool:
    return any(row.get("ap_terminal") == READY_TERMINAL for row in _read_jsonl_mappings(path, allow_empty=True))


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


def _canonical_report_path(raw: object, root: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        raise P6APRoundAdapterError()
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink():
        raise P6APRoundAdapterError()
    try:
        resolved = path.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError:
        raise P6APRoundAdapterError() from None
    if path != resolved or not resolved.is_file() or not _beneath(resolved, resolved_root):
        raise P6APRoundAdapterError()
    return resolved


def _plan_fingerprint(job: Mapping[str, Any], stage: str) -> str:
    raw_command = job.get(f"{stage}_command")
    command = list(map(str, raw_command)) if isinstance(raw_command, list) else None
    binding = {
        "runner_key": job.get("runner_key"),
        "compiled_artifact_digest": job.get("compiled_artifact_digest"),
        "compiled_artifact_path": job.get("compiled_artifact_path") or job.get("compiled_artifact"),
        "stage": stage,
        "stage_command": command,
        "report_path": _report_path_from_command(command or ()),
    }
    encoded = json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _report_path_from_command(command: Sequence[str]) -> str | None:
    for index, token in enumerate(command):
        for option in ("--report-json", "--out-json", "--export-report-json"):
            if token == option and index + 1 < len(command):
                return command[index + 1]
            if token.startswith(option + "="):
                return token.split("=", 1)[1]
    return None


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


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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
