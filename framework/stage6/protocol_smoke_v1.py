"""Small deterministic protocol probes for Stage6 serial and blind-search arms."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


BLIND_FIELDS = ("parameter_count", "flops", "ap_surrogate")
BACKEND_LABEL_FIELDS = {
    "latency_ms",
    "energy_j",
    "latency_prediction",
    "energy_prediction",
    "capability_profile",
    "dispatch_key",
    "backend",
}
PUBLIC_SELECTION_FIELDS = (
    "candidate_id",
    "width",
    "q_mode",
    "parameter_count",
    "flops",
    "ap_surrogate",
    "hardware_blind_acquisition_score",
)
ALLOWED_Q_MODES = frozenset({"fp16", "int8", "fp32"})


def _invalid_candidate() -> ValueError:
    return ValueError("invalid hardware-blind candidate")


def _finite_static_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _public_candidate_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "/" in value
        or "\\" in value
    ):
        raise _invalid_candidate()
    return value


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy only hardware-independent, public candidate fields."""
    candidate_id = _public_candidate_id(row.get("candidate_id"))
    public: dict[str, Any] = {"candidate_id": candidate_id}
    if "width" in row:
        width = row["width"]
        if (
            not isinstance(width, (list, tuple))
            or len(width) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in width
            )
        ):
            raise _invalid_candidate()
        public["width"] = list(width)
    if "q_mode" in row:
        q_mode = row["q_mode"]
        if not isinstance(q_mode, str) or q_mode not in ALLOWED_Q_MODES:
            raise _invalid_candidate()
        public["q_mode"] = q_mode
    for field in BLIND_FIELDS:
        number = _finite_static_number(row.get(field))
        if number is None:
            raise _invalid_candidate()
        public[field] = number
    return public


def _public_locked_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = _public_candidate_id(row.get("candidate_id"))
    screen_rank = row.get("screen_rank")
    if isinstance(screen_rank, bool) or not isinstance(screen_rank, int) or screen_rank < 0:
        raise _invalid_candidate()
    return {"candidate_id": candidate_id, "screen_rank": screen_rank}


def _normalize(values: Sequence[float], value: float) -> float:
    low, high = min(values), max(values)
    return 0.0 if high == low else (value - low) / (high - low)


def select_hardware_blind_batch(
    candidates: Iterable[Mapping[str, Any]], *, batch_size: int
) -> list[dict[str, Any]]:
    """Select candidates using static features only, never backend labels."""
    rows = []
    for row in candidates:
        if not isinstance(row, Mapping):
            raise _invalid_candidate()
        rows.append(_public_candidate(row))
    if len(rows) < batch_size or batch_size <= 0:
        raise ValueError("candidate count must cover a positive batch")
    columns = {field: [row[field] for row in rows] for field in BLIND_FIELDS}
    scored = []
    for row in rows:
        score = (
            _normalize(columns["ap_surrogate"], row["ap_surrogate"])
            - 0.5 * _normalize(columns["parameter_count"], row["parameter_count"])
            - 0.5 * _normalize(columns["flops"], row["flops"])
        )
        scored.append({**row, "hardware_blind_acquisition_score": score})
    scored.sort(key=lambda row: (-row["hardware_blind_acquisition_score"], str(row["candidate_id"])))
    return [
        {field: row[field] for field in PUBLIC_SELECTION_FIELDS if field in row}
        for row in scored[:batch_size]
    ]


def lock_compress_then_tune(screened: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze the four tuning candidates before any backend work begins."""
    rows = []
    for row in screened:
        if not isinstance(row, Mapping):
            raise _invalid_candidate()
        rows.append(_public_locked_candidate(row))
    if len(rows) != 12:
        raise ValueError("compress-then-tune requires exactly 12 screened candidates")
    ranked = sorted(rows, key=lambda row: (row["screen_rank"], row["candidate_id"]))
    locked = ranked[:4]
    return {
        "schema_version": "stage6_compress_then_tune_lock_smoke_v1",
        "screen_count": len(rows),
        "locked_count": len(locked),
        "locked_candidate_ids": [row["candidate_id"] for row in locked],
        "lock_precedes_tuning": True,
        "tuning_started": False,
        "runtime_budget_mutation": False,
    }


def record_reverse_transfer_attempts(
    attempts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Preserve infeasible reverse transfers instead of silently falling back."""
    rows = []
    for attempt_index, source in enumerate(attempts):
        if not isinstance(source, Mapping):
            raise _invalid_candidate()
        applicable = source.get("applicable") is True
        fallback_used = source.get("fallback_used") is True
        retuned = source.get("compressed_shape_retuned") is True
        rows.append(
            {
                "attempt_index": attempt_index,
                "terminal_status": "transferred_success" if applicable else "feasibility_failure",
                "reason_code": "transfer_applicable" if applicable else "transfer_inapplicable",
                "fallback_used": fallback_used,
                "compressed_shape_retuned": retuned,
            }
        )
    failures = sum(row["terminal_status"] == "feasibility_failure" for row in rows)
    fallback_count = sum(row["fallback_used"] for row in rows)
    retune_count = sum(row["compressed_shape_retuned"] for row in rows)
    return {
        "schema_version": "stage6_tune_then_compress_accounting_smoke_v1",
        "attempt_count": len(rows),
        "success_count": len(rows) - failures,
        "failure_count": failures,
        "fallback_count": fallback_count,
        "retune_count": retune_count,
        "attempts": rows,
    }
