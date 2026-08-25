"""Finalize one P6 round through the historical Stage5 leaves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
)
from framework.stage6.p6_round_adapter_runtime_v1 import (
    LeafRunner,
    P6RoundAdapterRuntimeError,
    RoundContext,
    load_round_context,
)


SUCCESS_STATUS = "measured_success_gold"
FAILURE_STATUSES = frozenset({"feasibility_failure", "numerical_feasibility_failure"})
METRIC_KEYS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
PUBLIC_FAILURE_REASON = re.compile(r"^[a-z0-9_-]+$")
Writer = Callable[[Path, Mapping[str, Any]], None]


class P6FinalizationRoundAdapterError(ValueError):
    """Stable path-free finalization adapter failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


def run_finalization_round(
    profile: ValidatedPostSourceAdapterProfile,
    measurement_request: Path,
    task_state: Path,
    actual_feedback: Path,
    actual_receipt: Path,
    finalization_barrier: Path,
    round_root: Path,
    runner: LeafRunner,
    *,
    writer: Writer | None = None,
) -> None:
    """Run native finalization/promotion and publish the P6 commit set."""
    context: RoundContext | None = None
    outputs: tuple[Path, Path, Path] = ()
    try:
        publication_writer = writer or _atomic_write_json
        context = load_round_context(profile, task_state, round_root, prior_stage="ap")
        request_path = _bound_request_path(measurement_request, context)
        outputs = _bound_outputs(
            context, actual_feedback, actual_receipt, finalization_barrier
        )
        _require_fresh_completion(context, outputs)
        _require_native_inputs(context)
        finalize = _single_leaf(profile, "feedback_finalize")
        promote = _single_leaf(profile, "feedback_promote")
        _run_finalize(context, request_path, finalize, runner)
        finalized = _validate_finalized(context)
        _run_promote(context, request_path, promote, runner)
        promoted = _validate_promoted(context, finalized)
        result, state, completion = _project_completion(context, promoted)
        _publish(context, outputs, result, state, completion, publication_writer)
    except (P6FinalizationRoundAdapterError, P6RoundAdapterRuntimeError):
        _rollback_publication(context, outputs)
        raise P6FinalizationRoundAdapterError() from None
    except Exception:
        _rollback_publication(context, outputs)
        raise P6FinalizationRoundAdapterError() from None


def _run_finalize(
    context: RoundContext,
    request_path: Path,
    leaf: PostSourceLeaf,
    runner: LeafRunner,
) -> None:
    root = context.round_root
    argv = (
        str(context.profile.project_python), str(leaf.implementation),
        "--manifest-json", str(root / "performance/performance_manifest.json"),
        "--measurement-request-json", str(request_path),
        "--ap-plan-jsonl", str(root / "ap/ap_plan.jsonl"),
        "--performance-state-jsonl", str(root / "performance/performance_state.jsonl"),
        "--ap-state-jsonl", str(root / "ap/ap_state.jsonl"),
        "--output-dir", str(root / "final"),
    )
    _require_zero(runner.run(argv, cwd=leaf.implementation_cwd, env=_leaf_env(context), shell=False))


def _run_promote(
    context: RoundContext,
    request_path: Path,
    leaf: PostSourceLeaf,
    runner: LeafRunner,
) -> None:
    root = context.round_root
    argv = (
        str(context.profile.project_python), str(leaf.implementation),
        "--measurement-request-json", str(request_path),
        "--feedback-json", str(root / "final/stage5_feedback_v2_final.json"),
        "--output-dir", str(root / "actual_feedback"),
    )
    _require_zero(runner.run(argv, cwd=leaf.implementation_cwd, env=_leaf_env(context), shell=False))


def _validate_finalized(context: RoundContext) -> tuple[Mapping[str, Any], ...]:
    root = context.round_root / "final"
    rows = _read_rows(root / "stage5_feedback_v2_final.json")
    audit = _read_mapping(root / "stage5_feedback_v2_audit.json")
    atomic = _read_mapping(root / "atomic_batch_audit.json")
    summary = audit.get("summary")
    if (
        audit.get("schema_version") != "stage5_feedback_batch_v2"
        or not isinstance(summary, Mapping)
        or summary.get("total") != 4
        or summary.get("pending") != 0
        or atomic.get("schema_version") != "stage5_atomic_batch_audit_v2"
        or atomic.get("feedback_released") is not True
        or atomic.get("batch_quarantined") is not False
        or atomic.get("released_feedback_rows") != list(rows)
    ):
        raise P6FinalizationRoundAdapterError()
    _validate_native_rows(context, rows)
    return rows


def _validate_promoted(
    context: RoundContext,
    finalized: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    root = context.round_root / "actual_feedback"
    rows = _read_rows(root / "stage5_feedback_v3_actual.json")
    audit = _read_mapping(root / "actual_feedback_batch_audit_v3.json")
    if (
        audit.get("schema_version") != "stage5_actual_feedback_batch_audit_v3"
        or audit.get("promoted_row_count") != 4
        or audit.get("silent_surrogate_fallback_count") != 0
        or tuple(_row_id(row) for row in rows) != tuple(_row_id(row) for row in finalized)
    ):
        raise P6FinalizationRoundAdapterError()
    _validate_native_rows(context, rows)
    _validate_promotion_provenance(rows, finalized, audit.get("rows"))
    return rows


def _validate_promotion_provenance(
    rows: Sequence[Mapping[str, Any]],
    finalized: Sequence[Mapping[str, Any]],
    audit_rows: object,
) -> None:
    if not isinstance(audit_rows, list) or len(audit_rows) != 4:
        raise P6FinalizationRoundAdapterError()
    for row, historical, audit in zip(rows, finalized, audit_rows, strict=True):
        without_sha = {key: value for key, value in row.items() if key != "actual_feedback_row_sha256"}
        if (
            not isinstance(audit, Mapping)
            or row.get("feedback_feature_contract") != "actual_feedback_v3"
            or row.get("graph_feature_promotion_schema") != "stage5_actual_feedback_promotion_v3"
            or row.get("historical_feedback_row_sha256") != _canonical_sha(historical)
            or row.get("actual_feedback_row_sha256") != _canonical_sha(without_sha)
            or audit.get("manifest_job_id") != row.get("manifest_job_id")
            or audit.get("candidate_graph_features_sha256") != row.get("candidate_graph_features_sha256")
            or audit.get("materialized_graph_features_sha256") != row.get("materialized_graph_features_sha256")
            or audit.get("actual_feedback_row_sha256") != row.get("actual_feedback_row_sha256")
        ):
            raise P6FinalizationRoundAdapterError()


def _validate_native_rows(
    context: RoundContext,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = tuple(str(row["row_id"]) for row in context.request["rows"])
    if tuple(_row_id(row) for row in rows) != expected:
        raise P6FinalizationRoundAdapterError()
    for request_row, row in zip(context.request["rows"], rows, strict=True):
        row_id = request_row["row_id"]
        if (
            row.get("manifest_job_id") != row_id
            or row.get("measurement_request_row_sha256") != context.request["row_sha256"][row_id]
            or row.get("source_evidence_sha256") != request_row["source_evidence_sha256"]
            or row.get("terminal_status") not in {SUCCESS_STATUS, *FAILURE_STATUSES}
        ):
            raise P6FinalizationRoundAdapterError()


def _project_completion(
    context: RoundContext,
    promoted: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    projected = [
        _project_row(request_row, row, context.request["row_sha256"][request_row["row_id"]])
        for request_row, row in zip(context.request["rows"], promoted, strict=True)
    ]
    completion = {
        "measurement_request_sha256": context.request["measurement_request_sha256"],
        "row_sha256": dict(context.request["row_sha256"]),
        "source_evidence_sha256": {
            row["row_id"]: row["source_evidence_sha256"] for row in context.request["rows"]
        },
    }
    result = {"measurement_request_sha256": context.request["measurement_request_sha256"], "rows": projected}
    state = {"stage": "finalization", "rows": [{key: row[key] for key in (
        "row_id", "row_sha256", "source_evidence_sha256", "terminal_status"
    )} for row in projected]}
    return result, state, completion


def _project_row(
    request_row: Mapping[str, Any],
    native: Mapping[str, Any],
    row_sha: str,
) -> dict[str, Any]:
    status = native["terminal_status"]
    base = {
        "row_id": request_row["row_id"],
        "row_sha256": row_sha,
        "source_evidence_sha256": request_row["source_evidence_sha256"],
        "terminal_status": status,
    }
    if status == SUCCESS_STATUS:
        metrics = {key: native.get(key) for key in METRIC_KEYS}
        if not _valid_metrics(metrics):
            raise P6FinalizationRoundAdapterError()
        return {**base, **{key: float(value) for key, value in metrics.items()}}
    reason = native.get("failure_reason")
    if not isinstance(reason, str) or not reason:
        raise P6FinalizationRoundAdapterError()
    return {**base, "failure_reason": reason if PUBLIC_FAILURE_REASON.fullmatch(reason) else "unspecified"}


def _publish(
    context: RoundContext,
    outputs: tuple[Path, Path, Path],
    result: Mapping[str, Any],
    state: Mapping[str, Any],
    completion: Mapping[str, Any],
    writer: Writer,
) -> None:
    feedback, receipt, barrier = outputs
    if _read_mapping(context.task_state_path) != context.task_state:
        raise P6FinalizationRoundAdapterError()
    writer(feedback, result)
    writer(receipt, completion)
    writer(context.task_state_path, state)
    writer(barrier, completion)


def _rollback_publication(
    context: RoundContext | None,
    outputs: tuple[Path, Path, Path],
) -> None:
    for path in outputs:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    if context is None:
        return
    try:
        current = _read_mapping(context.task_state_path)
        if current.get("stage") == "finalization":
            _atomic_write_json(context.task_state_path, context.task_state)
    except Exception:
        pass


def _bound_request_path(path: Path, context: RoundContext) -> Path:
    resolved = Path(path).resolve(strict=True)
    expected = context.round_root / "measurement-request.json"
    if resolved != expected or _read_mapping(resolved) != context.request:
        raise P6FinalizationRoundAdapterError()
    return resolved


def _bound_outputs(
    context: RoundContext,
    feedback: Path,
    receipt: Path,
    barrier: Path,
) -> tuple[Path, Path, Path]:
    expected = (
        context.round_root / "actual-feedback.json",
        context.round_root / "receipt.json",
        context.round_root / "barrier.json",
    )
    resolved = tuple(Path(path).resolve(strict=False) for path in (feedback, receipt, barrier))
    if resolved != expected:
        raise P6FinalizationRoundAdapterError()
    return expected


def _require_fresh_completion(
    context: RoundContext,
    outputs: Sequence[Path],
) -> None:
    native_roots = (context.round_root / "final", context.round_root / "actual_feedback")
    if any(path.exists() or path.is_symlink() for path in (*outputs, *native_roots)):
        raise P6FinalizationRoundAdapterError()


def _require_native_inputs(context: RoundContext) -> None:
    paths = (
        context.round_root / "performance/performance_manifest.json",
        context.round_root / "performance/performance_state.jsonl",
        context.round_root / "ap/ap_plan.jsonl",
        context.round_root / "ap/ap_state.jsonl",
    )
    if any(path.is_symlink() or not path.is_file() or path.stat().st_size <= 0 for path in paths):
        raise P6FinalizationRoundAdapterError()


def _single_leaf(profile: ValidatedPostSourceAdapterProfile, name: str) -> PostSourceLeaf:
    matches = tuple(leaf for leaf in profile.leaves if leaf.name == name)
    if len(matches) != 1:
        raise P6FinalizationRoundAdapterError()
    return matches[0]


def _leaf_env(context: RoundContext) -> dict[str, str]:
    keys = (
        "P6_HISTORY_RUN_MODE", "P6_HISTORY_PRIVATE_ROOT",
        "P6_HISTORY_TASK_STATE", "P6_HISTORY_ROUND_OUTPUT_ROOT",
    )
    inherited = {key: os.environ[key] for key in keys if key in os.environ}
    expected = {
        "P6_HISTORY_PRIVATE_ROOT": context.profile.private_root,
        "P6_HISTORY_TASK_STATE": context.task_state_path,
        "P6_HISTORY_ROUND_OUTPUT_ROOT": context.round_root,
    }
    if set(inherited) != set(keys) or any(
        Path(inherited[key]).resolve(strict=False) != value for key, value in expected.items()
    ):
        raise P6FinalizationRoundAdapterError()
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(context.gpu_indices), **inherited,
        "PATH": os.pathsep.join((str(context.profile.project_python.parent), "/usr/bin", "/bin")),
        "PYTHONPATH": os.pathsep.join((str(context.profile.private_root), str(Path.cwd()))),
    }


def _read_mapping(path: Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise P6FinalizationRoundAdapterError()
    return payload


def _read_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    payload = _read_json(path)
    if not isinstance(payload, list) or len(payload) != 4 or not all(isinstance(row, Mapping) for row in payload):
        raise P6FinalizationRoundAdapterError()
    return tuple(payload)


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise P6FinalizationRoundAdapterError()
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("row_id")
    if not isinstance(value, str) or not value:
        raise P6FinalizationRoundAdapterError()
    return value


def _valid_metrics(metrics: Mapping[str, Any]) -> bool:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in metrics.values()):
        return False
    return (
        float(metrics["latency_ms"]) > 0
        and float(metrics["energy_j"]) > 0
        and all(0 <= float(metrics[key]) <= 1 for key in ("ap30", "ap50", "ap70"))
    )


def _require_zero(result: object) -> None:
    returncode = getattr(result, "returncode", None)
    if isinstance(returncode, bool) or returncode != 0:
        raise P6FinalizationRoundAdapterError()
