"""Evidence-conservative Stage6 paper-table selection.

The frozen helpers below are migrated from the Stage6 release candidate.  The
public adapter validates and copies external records before invoking them.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from typing import Any, Mapping, Sequence


DEFAULT_ARMS = (
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "tune_then_compress",
    "joint_shcosearch",
)
LATENCY_CLOSE_FRACTION = 0.01
PUBLIC_DELTA_AP70 = 0.10
FROZEN_SOURCE_SHA256 = "a929e6bf58637f5ee74a7232f2a923474b9ed64c8623d0f53a61bd5d967e7b58"


class PublicInputError(ValueError):
    """Validation error whose text is safe to show from the reproducibility CLI."""


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value!r}")
    return number


def _trusted(point: Mapping[str, Any]) -> bool:
    return (
        point.get("terminal_status") == "measured_success_gold"
        and point.get("independent_validation_passed") is True
        and point.get("evidence_sha_verified") is True
    )


def _select(points: Sequence[Mapping[str, Any]], floor: float) -> Mapping[str, Any] | None:
    eligible = [point for point in points if _trusted(point) and _finite(point["AP70"]) >= floor]
    if not eligible:
        return None
    minimum = min(_finite(point["latency_ms"]) for point in eligible)
    close = [
        point
        for point in eligible
        if _finite(point["latency_ms"]) <= minimum * (1.0 + LATENCY_CLOSE_FRACTION)
    ]
    return min(close, key=lambda point: (_finite(point["energy_j"]), _finite(point["latency_ms"])))


def _fastest_trusted(points: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [point for point in points if _trusted(point)]
    if not eligible:
        return None
    minimum = min(_finite(point["latency_ms"]) for point in eligible)
    close = [
        point
        for point in eligible
        if _finite(point["latency_ms"]) <= minimum * (1.0 + LATENCY_CLOSE_FRACTION)
    ]
    return min(close, key=lambda point: (_finite(point["energy_j"]), _finite(point["latency_ms"])))


def build_backend_tables(
    *,
    backend: str,
    baseline: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    deltas: Sequence[float],
    required_arms: Sequence[str] = DEFAULT_ARMS,
) -> dict[str, Any]:
    baseline_ap = _finite(baseline["AP70"])
    baseline_latency = _finite(baseline["latency_ms"])
    missing = [arm for arm in required_arms if arm not in arms]
    nonterminal = []
    for arm in required_arms:
        if arm not in arms:
            continue
        summary = arms[arm]
        status = summary.get("status")
        if status == "complete":
            ready = summary.get("independent_validation_complete") is True
        elif status == "complete_failure":
            ready = summary.get("failure_evidence_sha_verified") is True
        else:
            ready = False
        if not ready:
            nonterminal.append(arm)
    tables: dict[str, list[dict[str, Any]]] = {}
    for delta in deltas:
        if not 0 < float(delta) < 1:
            raise ValueError("AP delta must be between zero and one")
        floor = baseline_ap - float(delta)
        rows = [
            {
                "method": "original_default",
                "backend": baseline.get("backend") or "pytorch_eager",
                "delta_ap_max": float(delta),
                "ap70_floor": floor,
                "selection_status": "selected",
                "outcome": "selected",
                "failure_reason": None,
                "ap_constraint_violated": False,
                "config": baseline.get("config") or [64, 128, 256, "fp32"],
                "AP70": baseline_ap,
                "latency_ms": baseline_latency,
                "energy_j": _finite(baseline["energy_j"]),
                "speedup": 1.0,
                "HV": None,
                "pareto_count": 1,
                "outer_genomes": 1,
                "tuning_trials": 0,
                "schedule_policy": baseline.get("schedule_policy"),
                "gpu_hours": baseline.get("gpu_hours"),
                "wallclock_s": baseline.get("wallclock_s"),
                "failure_count": 0,
                "failure_rate": 0.0,
                "evidence_origin": baseline.get("evidence_origin")
                or "stage6_new_original_default_measurement",
            }
        ]
        for arm in required_arms:
            summary = arms.get(arm) or {}
            selected = _select(summary.get("points") or [], floor)
            representative = selected or _fastest_trusted(summary.get("points") or [])
            outer = int(summary.get("outer_genomes") or 0)
            failures = int(summary.get("failure_count") or 0)
            failure_reason = summary.get("failure_reason")
            if selected:
                selection_status = "selected"
                outcome = "selected"
            elif summary.get("status") == "complete_failure":
                selection_status = "feasibility_failure"
                outcome = f"feasibility_failure:{failure_reason or 'unspecified'}"
            else:
                selection_status = "no_feasible_point"
                outcome = "no_point_satisfies_ap_floor"
            row = {
                "method": arm,
                "backend": backend,
                "delta_ap_max": float(delta),
                "ap70_floor": floor,
                "selection_status": selection_status,
                "outcome": outcome,
                "failure_reason": failure_reason,
                "ap_constraint_violated": (
                    selected is None and representative is not None
                    if summary.get("status") != "complete_failure"
                    else None
                ),
                "config": representative.get("config") if representative else None,
                "AP70": _finite(representative["AP70"]) if representative else None,
                "latency_ms": _finite(representative["latency_ms"]) if representative else None,
                "energy_j": _finite(representative["energy_j"]) if representative else None,
                "speedup": (
                    baseline_latency / _finite(representative["latency_ms"])
                    if representative
                    else None
                ),
                "HV": summary.get("HV"),
                "pareto_count": summary.get("pareto_count"),
                "outer_genomes": outer,
                "tuning_trials": summary.get("tuning_trials"),
                "gpu_hours": summary.get("gpu_hours"),
                "wallclock_s": summary.get("wallclock_s"),
                "failure_count": failures,
                "failure_rate": failures / outer if outer else None,
                "evidence_origin": (
                    representative.get("evidence_origin")
                    if representative
                    else summary.get("evidence_origin")
                ),
            }
            rows.append(row)
        ranked = sorted(
            (row for row in rows if row["selection_status"] == "selected"),
            key=lambda row: (row["latency_ms"], row["energy_j"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row["latency_rank"] = rank
        tables[f"delta_{float(delta):.2f}"] = rows
    return {
        "schema_version": "stage6_paper_main_tables_v1",
        "backend": backend,
        "baseline_ap70": baseline_ap,
        "baseline_latency_ms": baseline_latency,
        "latency_close_fraction": LATENCY_CLOSE_FRACTION,
        "deltas": [float(value) for value in deltas],
        "tables": tables,
        "missing_arms": missing,
        "nonterminal_arms": nonterminal,
        "paper_ready": not missing and not nonterminal,
    }


def _frozen_selection_contract_sha256() -> str:
    payload = "\n".join(
        (
            repr(DEFAULT_ARMS),
            repr(LATENCY_CLOSE_FRACTION),
            inspect.getsource(_finite),
            inspect.getsource(_trusted),
            inspect.getsource(_select),
            inspect.getsource(_fastest_trusted),
            inspect.getsource(build_backend_tables),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


FROZEN_SELECTION_CONTRACT_SHA256 = "2141f006b2facae3b8761210f891f0c8346ebced043dd744958ad08be7465539"
_FROZEN_SOURCE_CONTRACTS = {FROZEN_SOURCE_SHA256: FROZEN_SELECTION_CONTRACT_SHA256}


def verify_frozen_source(source_sha256: str, *, contract_sha256: str | None = None) -> None:
    """Fail closed unless the public source identity binds the frozen selection contract."""
    expected = _FROZEN_SOURCE_CONTRACTS.get(source_sha256)
    actual = _frozen_selection_contract_sha256() if contract_sha256 is None else contract_sha256
    if expected is None or actual != expected:
        raise PublicInputError("frozen source verification failed")


_MEASUREMENT_FIELDS = frozenset(
    {
        "evidence_id",
        "model",
        "backend",
        "method",
        "AP70",
        "latency_ms",
        "energy_j",
        "terminal_status",
        "independent_validation_passed",
        "evidence_sha_verified",
    }
)
_CELL_FIELDS = frozenset({"model", "backend", "method", "status"})
_PUBLIC_REPRESENTATIVE_FIELDS = ("evidence_id", "AP70", "latency_ms", "energy_j")


def _public_metric(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicInputError("measurement metrics must be finite numbers") from exc
    if not math.isfinite(number):
        raise PublicInputError("measurement metrics must be finite numbers")
    if (
        (field == "AP70" and not 0.0 <= number <= 100.0)
        or (field == "latency_ms" and number <= 0.0)
        or (field == "energy_j" and number < 0.0)
    ):
        raise PublicInputError("measurement metrics must be within public physical ranges")
    return number


def _public_string(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicInputError("measurement and cell identities must be non-empty strings")
    return value


def _validated_measurements(measurements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(measurements, Sequence) or isinstance(measurements, (str, bytes)):
        raise PublicInputError("measurements must be an array of objects")
    copied: list[dict[str, Any]] = []
    evidence_ids: set[str] = set()
    for row in measurements:
        if not isinstance(row, Mapping):
            raise PublicInputError("measurements must contain objects")
        if _MEASUREMENT_FIELDS - set(row):
            raise PublicInputError("measurements have missing required fields")
        identity = _public_string(row["evidence_id"])
        if identity in evidence_ids:
            raise PublicInputError("duplicate evidence_id in measurements")
        evidence_ids.add(identity)
        copied_row: dict[str, Any] = {}
        for field in ("model", "backend", "method", "terminal_status"):
            copied_row[field] = _public_string(row[field])
        copied_row["evidence_id"] = identity
        for field in ("AP70", "latency_ms", "energy_j"):
            copied_row[field] = _public_metric(row[field], field=field)
        if not isinstance(row["independent_validation_passed"], bool):
            raise PublicInputError("independent_validation_passed must be boolean")
        if not isinstance(row["evidence_sha_verified"], bool):
            raise PublicInputError("evidence_sha_verified must be boolean")
        copied_row["independent_validation_passed"] = row["independent_validation_passed"]
        copied_row["evidence_sha_verified"] = row["evidence_sha_verified"]
        copied.append(copied_row)
    return copied


def _validated_cells(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)) or not cells:
        raise PublicInputError("cells must be a non-empty array of objects")
    copied: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping) or _CELL_FIELDS - set(cell):
            raise PublicInputError("cells have missing required fields")
        normalized = {field: _public_string(cell[field]) for field in _CELL_FIELDS}
        if normalized["status"] not in {"complete", "complete_failure"}:
            raise PublicInputError("cell status must be complete or complete_failure")
        identity = (normalized["model"], normalized["backend"], normalized["method"])
        if identity in identities:
            raise PublicInputError("duplicate cell identity")
        identities.add(identity)
        copied.append(normalized)
    return copied


def _public_representative(point: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if point is None:
        return None
    return {field: point[field] for field in _PUBLIC_REPRESENTATIVE_FIELDS}


def select_representative_points(
    measurements: Sequence[Mapping[str, Any]], cells: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Select a Stage6 representative or explicit failure for every requested cell.

    The public layer only normalizes record identity and rejects malformed input.
    It calls the frozen feasibility/window/energy selection helpers unchanged.
    """
    points = _validated_measurements(measurements)
    requested_cells = _validated_cells(cells)
    results: list[dict[str, Any]] = []
    for cell in requested_cells:
        model, backend, method = cell["model"], cell["backend"], cell["method"]
        baselines = [
            point
            for point in points
            if point["model"] == model
            and point["backend"] == backend
            and point["method"] == "original_default"
        ]
        if len(baselines) != 1 or not _trusted(baselines[0]):
            results.append(
                {
                    "cell": dict(cell),
                    "status": "no_trusted_baseline",
                    "outcome": "baseline_not_trusted",
                    "representative": None,
                    "ap_constraint_violated": False,
                }
            )
            continue
        arm_points = sorted(
            (
                point
                for point in points
                if point["model"] == model
                and point["backend"] == backend
                and point["method"] == method
            ),
            key=lambda point: point["evidence_id"],
        )
        floor = _finite(baselines[0]["AP70"]) - PUBLIC_DELTA_AP70
        selected = _select(arm_points, floor)
        representative = selected or _fastest_trusted(arm_points)
        if selected is not None:
            status, outcome, violation = "selected", "selected", False
        elif cell["status"] == "complete_failure":
            status, outcome, violation = "feasibility_failure", "feasibility_failure:unspecified", None
        elif representative is None:
            status, outcome, violation = "no_trusted_point", "no_point_satisfies_ap_floor", False
        else:
            status, outcome, violation = "no_feasible_point", "no_point_satisfies_ap_floor", True
        results.append(
            {
                "cell": dict(cell),
                "status": status,
                "outcome": outcome,
                "representative": _public_representative(representative),
                "ap_constraint_violated": violation,
                "delta_ap_max": PUBLIC_DELTA_AP70,
                "ap70_floor": floor,
            }
        )
    return {
        "schema_version": "stage6_representative_points_v1",
        "latency_close_fraction": LATENCY_CLOSE_FRACTION,
        "cells": results,
    }
