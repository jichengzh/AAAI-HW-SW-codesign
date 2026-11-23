#!/usr/bin/env python3
"""Finalize formal, evidence-verified F-Cooper five-arm Stage6 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "S5-FCO-TRT-V2"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"
BASE_WIDTH = (64, 128, 256, 128, 256)
SUCCESS = "measured_success_gold"
TERMINAL_FAILURES = {"feasibility_failure", "numerical_feasibility_failure"}
METRICS = ("ap30", "ap50", "ap70", "latency_ms", "energy_j")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.single_target_search_v2 import (  # noqa: E402
    FROZEN_GOLD176_GRAPH_FEATURES_SHA256,
    FROZEN_GOLD176_ROWS_SHA256,
)
from scripts.fcooper_execute_measurement_row_v2 import (  # noqa: E402
    validate_recovery_training_evidence,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path, field: str = "rows") -> list[dict[str, Any]]:
    payload = _read(path)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(row) for row in payload[field]]
    raise ValueError(f"expected rows in {path}")


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_pilot(value: Any, *, context: str) -> None:
    if isinstance(value, str):
        if PILOT_FRAGMENT in value:
            raise ValueError(f"{context} contains forbidden pilot path fragment")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_pilot(key, context=context)
            _reject_pilot(item, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _reject_pilot(item, context=context)


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite measured {name}: {value!r}")
    return number


def _resolve_path(value: Any, *, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing evidence path: {field}")
    _reject_pilot(value, context=field)
    path = Path(value)
    return path if path.is_absolute() else root / path


def _verify_file(
    row: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
    root: Path,
) -> Path:
    path = _resolve_path(row.get(path_field), root=root, field=path_field)
    expected = row.get(sha_field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing evidence SHA: {sha_field}")
    if not path.is_file():
        raise ValueError(f"evidence file missing: {path}")
    actual = _file_sha(path)
    if actual != expected:
        raise ValueError(f"evidence SHA mismatch for {sha_field}: {path}")
    return path


def verify_success_evidence(
    source: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Verify every byte-bound artifact required by a successful formal row."""
    row = dict(source)
    _reject_pilot(row, context="successful feedback row")
    if row.get("terminal_status") != SUCCESS:
        raise ValueError("success evidence verifier received a non-success row")
    for metric in METRICS:
        _finite(row.get(metric), name=metric)

    verified_paths = [
        _verify_file(
            row,
            path_field="performance_result_json",
            sha_field="performance_result_sha256",
            root=root,
        ),
        _verify_file(
            row,
            path_field="ap_report_path",
            sha_field="ap_report_sha256",
            root=root,
        ),
        _verify_file(
            row,
            path_field="materialized_source_evidence_path",
            sha_field="materialized_source_evidence_sha256",
            root=root,
        ),
        _verify_file(
            row,
            path_field="engine_path",
            sha_field="engine_sha256",
            root=root,
        ),
    ]
    graph = row.get("graph_features")
    if not isinstance(graph, Mapping):
        raise ValueError("successful row is missing graph features")
    if _canonical_sha(graph) != row.get("materialized_graph_features_sha256"):
        raise ValueError("successful row graph feature SHA mismatch")

    recorded_row_sha = row.pop("actual_feedback_row_sha256", None)
    if recorded_row_sha is not None and recorded_row_sha != _canonical_sha(row):
        raise ValueError("successful actual feedback row SHA mismatch")

    source_evidence = _read(verified_paths[2])
    if not isinstance(source_evidence, Mapping):
        raise ValueError("materialized source evidence must be an object")
    _reject_pilot(source_evidence, context="materialized source evidence")
    checkpoint = _verify_file(
        source_evidence,
        path_field="checkpoint_path",
        sha_field="checkpoint_sha256",
        root=root,
    )
    if source_evidence["checkpoint_sha256"] != row.get("checkpoint_sha256"):
        raise ValueError("feedback/source checkpoint SHA mismatch")
    verified_paths.append(checkpoint)

    width = tuple(int(value) for value in row.get("width") or ())
    if len(width) != 5:
        raise ValueError("successful F-Cooper row must expose five widths")
    recovery_sha = row.get("recovery_training_report_sha256")
    if width != BASE_WIDTH:
        recovery = _verify_file(
            source_evidence,
            path_field="recovery_training_report_path",
            sha_field="recovery_training_report_sha256",
            root=root,
        )
        if source_evidence["recovery_training_report_sha256"] != recovery_sha:
            raise ValueError("feedback/source recovery-training SHA mismatch")
        recovery_report = _read(recovery)
        recovery_audit = validate_recovery_training_evidence(
            report_path=recovery,
            recovery_contract_path=Path(
                str(recovery_report.get("recovery_contract_path") or "")
            ),
            config_path=Path(str(recovery_report.get("config_path") or "")),
            initial_checkpoint_path=Path(
                str(recovery_report.get("initial_checkpoint_path") or "")
            ),
            recovered_checkpoint_path=Path(
                str(recovery_report.get("recovered_checkpoint_path") or "")
            ),
        )
        if recovery_audit["recovered_checkpoint_sha256"] != row.get(
            "checkpoint_sha256"
        ):
            raise ValueError("recovery-training report checkpoint drift")
        verified_paths.append(recovery)
    elif recovery_sha not in {None, source_evidence.get("recovery_training_report_sha256")}:
        raise ValueError("unpruned recovery-training SHA drift")

    return {
        "verified": True,
        "row_id": str(row.get("row_id") or row.get("manifest_job_id") or ""),
        "verified_paths": [str(path.resolve()) for path in verified_paths],
        "verified_sha256": [_file_sha(path) for path in verified_paths],
    }


def normalize_terminal_row(source: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    status = row.get("terminal_status")
    if status == SUCCESS:
        return {
            **row,
            **{metric: _finite(row.get(metric), name=metric) for metric in METRICS},
            "credible_terminal_status": True,
        }
    if status not in TERMINAL_FAILURES:
        raise ValueError(f"non-credible terminal status: {status!r}")
    reason = str(row.get("failure_reason") or "").strip()
    if not reason:
        raise ValueError("terminal feasibility failure is missing failure_reason")
    return {
        **row,
        **{metric: None for metric in METRICS},
        "credible_terminal_status": True,
        "selection_status": row.get("selection_status")
        or "credible_terminal_failure",
    }


def _select_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    terminal = [normalize_terminal_row(row) for row in rows]
    successful = [row for row in terminal if row["terminal_status"] == SUCCESS]
    if not successful:
        failure = min(
            terminal,
            key=lambda row: str(row.get("row_id") or row.get("manifest_job_id") or ""),
        )
        return {**failure, "selection_status": "all_candidates_terminal_failure"}
    floor = float(ap70_ref) - float(max_ap_drop)
    feasible = [row for row in successful if float(row["ap70"]) >= floor]
    pool = feasible or successful
    minimum_latency = min(float(row["latency_ms"]) for row in pool)
    tied = [
        row
        for row in pool
        if float(row["latency_ms"]) <= minimum_latency * 1.01
    ]
    selected = min(
        tied,
        key=lambda row: (
            float(row["energy_j"]),
            float(row["latency_ms"]),
            str(row.get("row_id") or row.get("manifest_job_id") or ""),
        ),
    )
    return {
        **selected,
        "ap70_floor": floor,
        "ap_constraint_satisfied": bool(feasible),
        "selection_status": (
            "selected_feasible"
            if feasible
            else "selected_fastest_ap_constraint_violation"
        ),
    }


def select_arm_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("arm has no terminal candidates")
    return _select_candidate(rows, ap70_ref=ap70_ref, max_ap_drop=max_ap_drop)


def select_gear_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    """Select GEAR strictly from the 16 formal T16 online observations."""
    formal = [dict(row) for row in rows]
    identities = [
        str(row.get("row_id") or row.get("manifest_job_id") or "") for row in formal
    ]
    if (
        len(formal) != 16
        or len(set(identities)) != 16
        or any(not identity for identity in identities)
    ):
        raise ValueError("GEAR requires exactly 16 unique formal T16 online rows")
    for row in formal:
        _reject_pilot(row, context="GEAR T16 row")
        if (
            row.get("task_id") != TASK_ID
            or row.get("training_source") != "online_feedback"
        ):
            raise ValueError("GEAR candidates must be formal T16 online feedback only")
    return _select_candidate(
        formal, ap70_ref=ap70_ref, max_ap_drop=max_ap_drop
    )


def repeat_paths_for_row(repeat_root: Path, row_id: str) -> list[Path]:
    """Resolve repeats under the selected identity, never a fixed incumbent lane."""
    _reject_pilot(str(repeat_root), context="repeat root")
    _reject_pilot(row_id, context="selected row_id")
    row_root = repeat_root / row_id
    performance = sorted(
        row_root.glob("same_gpu7_repeat_*/performance.json")
    )
    direct = sorted(row_root.glob("same_gpu7_repeat_*.json"))
    return performance or direct


def _median_performance(
    paths: Sequence[Path],
    *,
    expected_onnx_sha256: str,
    expected_precision: str,
    expected_builder_level: int,
) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError(f"expected three GPU7 performance repeats, got {len(paths)}")
    reports = [_read(path) for path in paths]
    engine_shas = []
    for report in reports:
        if int(report.get("gpu_abs", -1)) != 7:
            raise ValueError("independent repeat is not bound to GPU7")
        if report.get("precision") != expected_precision:
            raise ValueError("independent repeat precision drift")
        if int(report.get("builder_optimization_level", -1)) != expected_builder_level:
            raise ValueError("independent repeat builder level drift")
        artifacts = report.get("artifact_sha256")
        if not isinstance(artifacts, Mapping):
            raise ValueError("independent repeat artifact SHA mapping missing")
        if artifacts.get("source_onnx") != expected_onnx_sha256:
            raise ValueError("independent repeat ONNX SHA drift")
        engine_sha = artifacts.get("compiled_engine")
        if not isinstance(engine_sha, str) or len(engine_sha) != 64:
            raise ValueError("independent repeat engine SHA missing")
        engine_shas.append(engine_sha)
    return {
        "latency_ms": statistics.median(
            _finite(report.get("lat_p50_ms"), name="repeat latency")
            for report in reports
        ),
        "energy_j": statistics.median(
            _finite(report.get("energy_j"), name="repeat energy")
            for report in reports
        ),
        "repeat_paths": [str(path.resolve()) for path in paths],
        "repeat_sha256": [_file_sha(path) for path in paths],
        "repeat_engine_sha256": engine_shas,
    }


def _median_native(
    paths: Sequence[Path],
    *,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError(f"expected three GPU7 native repeats, got {len(paths)}")
    reports = [_read(path) for path in paths]
    for report in reports:
        if int(report.get("gpu_abs", -1)) != 7:
            raise ValueError("native independent repeat is not bound to GPU7")
        if report.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError("native repeat checkpoint SHA drift")
        if report.get("config_sha256") != expected_config_sha256:
            raise ValueError("native repeat config SHA drift")
    return {
        "latency_ms": statistics.median(
            _finite(report.get("latency_ms"), name="native repeat latency")
            for report in reports
        ),
        "energy_j": statistics.median(
            _finite(report.get("energy_j"), name="native repeat energy")
            for report in reports
        ),
        "repeat_paths": [str(path.resolve()) for path in paths],
        "repeat_sha256": [_file_sha(path) for path in paths],
    }


def _validate_independent_ap(
    path: Path, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    report = _read(path)
    if (
        report.get("status") != "success_full"
        or int(report.get("dataset_samples") or 0) != 2170
        or int(report.get("processed_samples") or 0) != 2170
        or int(report.get("fallback_samples", -1)) != 0
    ):
        raise ValueError("independent AP report does not satisfy the full-2170 contract")
    for metric in ("ap30", "ap50", "ap70"):
        if abs(
            _finite(report.get(metric), name=f"independent {metric}")
            - _finite(expected.get(metric), name=f"selected {metric}")
        ) > 1.0e-8:
            raise ValueError(f"independent AP metric drift: {metric}")
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha(path),
        "processed_samples": 2170,
        "fallback_samples": 0,
        "ap30": float(report["ap30"]),
        "ap50": float(report["ap50"]),
        "ap70": float(report["ap70"]),
    }


def _validate_terminal_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence_root: Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} terminal rows, got {len(rows)}")
    normalized = []
    for row in rows:
        item = normalize_terminal_row(row)
        if item["terminal_status"] == SUCCESS:
            item["success_evidence_audit"] = verify_success_evidence(
                row, root=evidence_root
            )
        normalized.append(item)
    return normalized


def _bind_repeats(
    selected: Mapping[str, Any],
    *,
    repeat_root: Path,
    builder_level: int,
) -> dict[str, Any]:
    row = dict(selected)
    if row["terminal_status"] != SUCCESS:
        return {
            **row,
            "repeat_paths": [],
            "repeat_sha256": [],
            "ap_constraint_satisfied": False,
        }
    row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
    graph = row.get("graph_features") or {}
    metrics = _median_performance(
        repeat_paths_for_row(repeat_root, row_id),
        expected_onnx_sha256=str(graph.get("onnx_sha256") or ""),
        expected_precision=str(row["q_mode"]),
        expected_builder_level=builder_level,
    )
    ap_recheck = _validate_independent_ap(
        repeat_root / row_id / "independent_ap_report.json",
        expected=row,
    )
    return {**row, **metrics, "independent_ap_recheck": ap_recheck}


def _verify_contract_file(
    contract: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
    root: Path,
) -> Path:
    return _verify_file(
        contract, path_field=path_field, sha_field=sha_field, root=root
    )


def _original_row(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    for path_field, sha_field in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("config_path", "config_sha256"),
        ("ap_reference_report_path", "ap_reference_report_sha256"),
    ):
        _verify_contract_file(
            contract, path_field=path_field, sha_field=sha_field, root=root
        )
    row_id = "fcooper-original-default"
    repeats = repeat_paths_for_row(
        root / "controls/original_default/independent_repeats", row_id
    )
    metrics = _median_native(
        repeats,
        expected_checkpoint_sha256=str(contract["checkpoint_sha256"]),
        expected_config_sha256=str(contract["config_sha256"]),
    )
    return {
        "row_id": row_id,
        "terminal_status": SUCCESS,
        "credible_terminal_status": True,
        "width": list(BASE_WIDTH),
        "q_mode": "fp32",
        "ap30": _finite(contract["ap30_ref"], name="ap30_ref"),
        "ap50": _finite(contract["ap50_ref"], name="ap50_ref"),
        "ap70": _finite(contract["ap70_ref"], name="ap70_ref"),
        "ap_report_sha256": contract["ap_reference_report_sha256"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        **metrics,
        "ap_constraint_satisfied": True,
        "selection_status": "fixed_original_default",
    }


def _configuration(row: Mapping[str, Any]) -> str | None:
    width = row.get("width")
    q_mode = row.get("q_mode")
    if not isinstance(width, Sequence) or isinstance(width, str) or not q_mode:
        return None
    return f"({','.join(map(str, width))},{q_mode})"


def _csv_row(method: str, row: Mapping[str, Any], budget: Any) -> dict[str, Any]:
    return {
        "method": method,
        "terminal_status": row["terminal_status"],
        "selection_status": row.get("selection_status"),
        "ap_constraint_satisfied": (
            row.get("ap_constraint_satisfied")
            if row["terminal_status"] == SUCCESS
            else None
        ),
        "configuration": _configuration(row),
        **{metric: row.get(metric) for metric in METRICS},
        "budget": budget,
        "failure_rate": 0.0 if row["terminal_status"] == SUCCESS else 1.0,
    }


def finalize(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    _reject_pilot(str(root), context="formal root")
    _reject_pilot(str(output), context="output root")
    contract_path = root / "contracts/frozen_contract.json"
    contract = _read(contract_path)
    _reject_pilot(contract, context="formal frozen contract")
    ap70_ref = _finite(contract["ap70_ref"], name="ap70_ref")
    max_drop = 0.10

    original_failure = root / "controls/original_default/terminal_failure.json"
    original = (
        normalize_terminal_row(_read(original_failure))
        if original_failure.is_file()
        else _original_row(root, contract)
    )

    schedule_rows = _validate_terminal_pool(
        _rows(root / "controls/schedule_only/feedback_history.json"),
        evidence_root=root,
        expected_count=1,
    )
    schedule = schedule_rows[0]
    if tuple(schedule.get("width") or ()) != BASE_WIDTH or schedule.get("q_mode") != "fp32":
        raise ValueError("schedule-only must use the fixed original FP32 structure")
    schedule["selection_status"] = "fixed_schedule_only"
    schedule["ap_constraint_satisfied"] = (
        schedule["terminal_status"] == SUCCESS
        and float(schedule["ap70"]) >= ap70_ref - max_drop
    )
    schedule = _bind_repeats(
        schedule,
        repeat_root=root / "controls/schedule_only/independent_repeats",
        builder_level=5,
    )

    compression_rows = _validate_terminal_pool(
        _rows(root / "controls/compression_only/feedback_history.json"),
        evidence_root=root,
        expected_count=16,
    )
    plan = _read(root / "controls/five_arm_plan/stage6_fcooper_five_arm_plan.json")
    arms = plan.get("arms") or {}
    coldstart_contract = plan.get("coldstart_contract") or {}
    graph_contract = plan.get("observed_graph_evidence_contract") or {}
    if (
        plan.get("schema_version") != "stage6_fcooper_five_arm_plan_v2"
        or plan.get("task_id") != TASK_ID
        or coldstart_contract.get("source")
        != "byte_frozen_standalone_gold176_only"
        or coldstart_contract.get("rows_sha256")
        != FROZEN_GOLD176_ROWS_SHA256
        or coldstart_contract.get("graph_features_sha256")
        != FROZEN_GOLD176_GRAPH_FEATURES_SHA256
    ):
        raise ValueError("five-arm plan does not bind the frozen Gold176 contract")
    graph_evidence_path = _resolve_path(
        graph_contract.get("path"),
        root=root,
        field="observed_graph_evidence_contract.path",
    )
    if (
        not graph_evidence_path.is_file()
        or _file_sha(graph_evidence_path) != graph_contract.get("sha256")
    ):
        raise ValueError("five-arm plan observed graph evidence SHA drift")
    graph_evidence = _read(graph_evidence_path)
    graph_bindings = graph_evidence.get("t16_graph_bindings") or []
    graph_rows = graph_evidence.get("graph_features") or []
    if (
        len(graph_bindings) != 16
        or len(graph_rows) != 17
        or len({str(row.get("row_id") or "") for row in graph_bindings}) != 16
        or any(
            binding.get("materialized_graph_features_sha256")
            != binding.get("graph_payload_sha256")
            or binding.get("graph_payload_sha256")
            != _canonical_sha(graph_rows[index])
            for index, binding in enumerate(graph_bindings)
        )
    ):
        raise ValueError("five-arm plan observed graph row binding drift")
    compression_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in compression_rows
    ]
    if compression_ids != list(
        (arms.get("compression_only") or {}).get("selected_row_ids") or []
    ):
        raise ValueError("compression feedback does not match five-arm plan")
    compression = _bind_repeats(
        select_arm_candidate(
            compression_rows, ap70_ref=ap70_ref, max_ap_drop=max_drop
        ),
        repeat_root=root / "controls/compression_only/independent_repeats",
        builder_level=0,
    )

    screen_rows = _validate_terminal_pool(
        _rows(root / "controls/compress_then_tune/screen_feedback_history.json"),
        evidence_root=root,
        expected_count=12,
    )
    screen_ids_ordered = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in screen_rows
    ]
    if screen_ids_ordered != list(
        (arms.get("compress_then_tune") or {}).get("screen_row_ids") or []
    ):
        raise ValueError("screen feedback does not match five-arm plan")
    tuned_rows = _validate_terminal_pool(
        _rows(root / "controls/compress_then_tune/tuned_feedback_history.json"),
        evidence_root=root,
        expected_count=4,
    )
    screen_ids = {
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in screen_rows
    }
    tuned_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in tuned_rows
    ]
    if len(set(tuned_ids)) != 4 or not set(tuned_ids).issubset(screen_ids):
        raise ValueError("tuned four must be distinct members of the observed screen")
    tune_plan = plan.get("arms", {}).get("compress_then_tune", {})
    if (
        plan.get("task_id") != TASK_ID
        or tune_plan.get("phase_status") != "tuned_remeasurement_ready"
        or list(tune_plan.get("tuned_row_ids") or []) != tuned_ids
    ):
        raise ValueError("compress-then-tune plan does not bind the observed tuned four")
    compress_then_tune = _bind_repeats(
        select_arm_candidate(tuned_rows, ap70_ref=ap70_ref, max_ap_drop=max_drop),
        repeat_root=root
        / "controls/compress_then_tune/independent_repeats",
        builder_level=5,
    )

    online_path = root / f"search/{TASK_ID}/feedback_history_final_t16.json"
    online_rows = _validate_terminal_pool(
        _rows(online_path), evidence_root=root, expected_count=16
    )
    gear = _bind_repeats(
        select_gear_candidate(
            online_rows, ap70_ref=ap70_ref, max_ap_drop=max_drop
        ),
        repeat_root=root / f"search/{TASK_ID}/independent_repeats",
        builder_level=5,
    )
    gear["selection_source"] = "formal_t16_online_feedback"

    arm_rows = [
        ("Original/default", original, 1),
        ("Compression only", compression, 16),
        ("Schedule only", schedule, 1),
        ("Compress -> Tune", compress_then_tune, "12+4"),
        ("GEAR (Ours)", gear, 16),
    ]
    csv_rows = [_csv_row(method, row, budget) for method, row, budget in arm_rows]

    integrity_paths = [
        root / f"search/{TASK_ID}/online_file_integrity_audit.json",
        root / f"search/{TASK_ID}/final_feedback_integrity_audit.json",
        root / "controls/compression_only/file_integrity_audit.json",
        root / "controls/compress_then_tune/file_integrity_audit.json",
    ]
    integrity = [_read(path) for path in integrity_paths]
    integrity_passed = all(payload.get("passed") is True for payload in integrity)
    all_credible_terminal = all(
        row.get("credible_terminal_status") is True for _, row, _ in arm_rows
    )
    successful_repeats_verified = all(
        row["terminal_status"] != SUCCESS or len(row.get("repeat_paths") or []) == 3
        for _, row, _ in arm_rows
    )

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "fcooper_stage6_trt_delta_ap_0.10_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    audit = {
        "schema_version": "stage6_fcooper_five_arm_audit_v2",
        "task_id": TASK_ID,
        "target_model": "fcooper",
        "hardware_id": "h800",
        "backend": "trt",
        "performance_gpu_index": 7,
        "ap70_ref": ap70_ref,
        "ap70_floor": ap70_ref - max_drop,
        "online_search_budget_consumed": len(online_rows),
        "gear_candidate_source": "formal_t16_online_feedback_only",
        "gear_probe_rows_used": False,
        "all_five_arms_credible_terminal": all_credible_terminal,
        "successful_repeats_verified": successful_repeats_verified,
        "paper_ready": (
            integrity_passed
            and all_credible_terminal
            and successful_repeats_verified
            and len(csv_rows) == 5
        ),
        "integrity_audits": [
            {
                "path": str(path.resolve()),
                "sha256": _file_sha(path),
                "passed": payload.get("passed"),
                "row_count": payload.get("row_count"),
            }
            for path, payload in zip(integrity_paths, integrity)
        ],
        "rows": csv_rows,
        "selected_evidence": {
            method: {
                "row_id": row.get("row_id"),
                "terminal_status": row["terminal_status"],
                "failure_reason": row.get("failure_reason"),
                "engine_sha256": row.get("engine_sha256"),
                "performance_result_sha256": row.get(
                    "performance_result_sha256"
                ),
                "ap_report_sha256": row.get("ap_report_sha256"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "recovery_training_report_sha256": row.get(
                    "recovery_training_report_sha256"
                ),
                "repeat_paths": row.get("repeat_paths") or [],
                "repeat_sha256": row.get("repeat_sha256") or [],
                "repeat_engine_sha256": row.get("repeat_engine_sha256"),
            }
            for method, row, _ in arm_rows
        },
    }
    audit_path = output / "fcooper_stage6_five_arm_audit_v2.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle = {
        "schema_version": "fcooper_stage6_evidence_bundle_v2",
        "task_id": TASK_ID,
        "audit_path": str(audit_path),
        "audit_sha256": _file_sha(audit_path),
        "csv_path": str(csv_path),
        "csv_sha256": _file_sha(csv_path),
        "frozen_contract_path": str(contract_path),
        "frozen_contract_sha256": _file_sha(contract_path),
        "selected_evidence": audit["selected_evidence"],
    }
    bundle_path = output / "fcooper_stage6_evidence_bundle_v2.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = finalize(args.root, args.output_dir)
    print(
        json.dumps(
            {"paper_ready": audit["paper_ready"], "rows": audit["rows"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
