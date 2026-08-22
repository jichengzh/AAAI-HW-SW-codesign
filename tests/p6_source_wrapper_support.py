"""Shared canonical request fixture for source-wrapper integration tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def source_bridge_request(
    binding: Mapping[str, Any],
    *,
    source_contract_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal hash-consistent request with one nested training binding."""
    contract = {
        **copy.deepcopy(dict(source_contract_fields or {})),
        "external_training_binding": copy.deepcopy(dict(binding)),
    }
    task_sha = "d" * 64
    row = {
        "schema_version": "stage5_candidate_row_v2",
        "task_id": "P6-H800-PYRAMID",
        "task_sha256": task_sha,
        "row_id": "row-0",
        "manifest_job_id": "row-0",
        "group_id": "pyramid|16x32x64",
        "model": "pyramid",
        "width": [16, 32, 64],
        "width_schema": ["w0", "w1", "w2"],
        "structure_widths": {"w0": 16, "w1": 32, "w2": 64},
        "genome": [16, 32, 64, "fp16"],
        "strategy_id": "q=fp16",
        "q_mode": "fp16",
        "hardware_id": "h800",
        "capability_profile_id": "h800-tvm-auto",
        "capability_digest": "e" * 64,
        "dispatch_key": "tvm_auto",
        "source_status": "materializable",
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_materialized",
        "source_contract": contract,
        "source_contract_sha256": _sha(contract),
        "source_evidence_sha256": "c" * 64,
        "graph_features": {
            "group_id": "pyramid|16x32x64",
            "model": "pyramid",
            "width": [16, 32, 64],
        },
    }
    body = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "P6-H800-PYRAMID",
        "task_sha256": task_sha,
        "round_index": 0,
        "batch_size": 1,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "rows": [row],
        "row_sha256": {"row-0": _sha(row)},
    }
    return {**body, "measurement_request_sha256": _sha(body)}


def source_bridge_output_paths(
    artifact: Path, *, config_at_checkpoint: bool = False
) -> dict[str, str]:
    """Return the complete eleven-path recipe-v2 output fixture."""
    checkpoint = artifact / "checkpoint"
    config = (
        checkpoint / "config.yaml"
        if config_at_checkpoint
        else artifact / "config" / "source-config.json"
    )
    return {
        "checkpoint_path": str(checkpoint / "model.ckpt"),
        "checkpoint_dir": str(checkpoint),
        "config_path": str(config),
        "training_done_marker": str(artifact / "markers" / "training.done"),
        "onnx_path": str(artifact / "onnx" / "model.onnx"),
        "onnx_report_path": str(artifact / "onnx" / "report.json"),
        "calibration_root": str(artifact / "calibration"),
        "calibration_npz": str(artifact / "calibration" / "cache.npz"),
        "calibration_summary": str(artifact / "calibration" / "summary.json"),
        "trt_calibration_dir": str(artifact / "trt-calibration"),
        "source_done_marker": str(artifact / "markers" / "source.done"),
    }


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
