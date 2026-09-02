from __future__ import annotations

import copy
import hashlib

import pytest

from framework.stage6.p6_capability_artifacts_v1 import (
    P6CapabilityArtifactError,
    build_probe_artifact_blobs,
    validate_probe_artifact_blobs,
)
from framework.stage6.p6_capability_probe_specs_v1 import RECORD_SCHEMA_VERSION


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(probe_id: str, q_mode: str, *, success: bool = True) -> dict:
    onnx = f"onnx:{probe_id}:{q_mode}".encode()
    compiler = f"ir:{probe_id}:{q_mode}".encode()
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "probe_id": probe_id,
        "q_mode": q_mode,
        "onnx_sha256": _sha(onnx),
        "build_success": success,
        "observation_status": ("observed_build_success" if success else "observed_build_failure"),
        "compiler_ir_sha256": _sha(compiler) if success else None,
        "int8_propagated_ops": 1 if q_mode == "int8" and success else None,
        "precision_eligible_ops": 2 if q_mode == "int8" and success else None,
        "qdq_folded_pairs": 1 if q_mode == "int8" and success else None,
        "qdq_pairs": 1 if q_mode == "int8" and success else None,
        "reformat_ops": 0 if success else None,
        "total_ops": 2 if success else None,
        "fused_ops": 0 if success else None,
        "fusible_ops": 1 if success else None,
    }


def _inputs() -> tuple[list[dict], dict[tuple[str, str], bytes], dict[tuple[str, str], bytes]]:
    records = [_record("P1", "fp16"), _record("P1", "int8")]
    onnx = {
        (row["probe_id"], row["q_mode"]): f"onnx:{row['probe_id']}:{row['q_mode']}".encode()
        for row in records
    }
    outputs = {
        (row["probe_id"], row["q_mode"]): f"ir:{row['probe_id']}:{row['q_mode']}".encode()
        for row in records
    }
    return records, onnx, outputs


def test_artifact_blobs_round_trip_without_paths() -> None:
    records, onnx, outputs = _inputs()
    blobs, manifest_sha256 = build_probe_artifact_blobs(
        family="neutral", records=records, onnx_by_cell=onnx, output_by_cell=outputs
    )

    validated, rebuilt_sha256 = validate_probe_artifact_blobs(
        blobs, family="neutral", records=records
    )

    assert list(validated) == blobs
    assert rebuilt_sha256 == manifest_sha256
    assert "/" not in str(blobs)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "extra", "onnx", "ir", "path", "wrong-kind"],
)
def test_artifact_blobs_reject_partition_or_byte_drift(mutation: str) -> None:
    records, onnx, outputs = _inputs()
    blobs, _ = build_probe_artifact_blobs(
        family="neutral", records=records, onnx_by_cell=onnx, output_by_cell=outputs
    )
    changed = copy.deepcopy(blobs)
    if mutation == "missing":
        changed.pop()
    elif mutation == "duplicate":
        changed[-1] = copy.deepcopy(changed[0])
    elif mutation == "extra":
        changed.append({**copy.deepcopy(changed[0]), "probe_id": "P2"})
    elif mutation == "onnx":
        changed[0]["onnx_base64"] = changed[1]["onnx_base64"]
    elif mutation == "ir":
        changed[0]["compiler_output_base64"] = changed[1]["compiler_output_base64"]
    elif mutation == "path":
        changed[0]["compiler_path"] = "/private/tvm"
    else:
        changed[0]["compiler_output_kind"] = "error"

    with pytest.raises(P6CapabilityArtifactError):
        validate_probe_artifact_blobs(changed, family="neutral", records=records)


def test_failed_observation_requires_sealed_error_bytes() -> None:
    records = [_record("P1", "fp16", success=False)]
    onnx = {("P1", "fp16"): b"onnx:P1:fp16"}
    outputs = {("P1", "fp16"): b"RuntimeError: compiler rejected probe"}

    blobs, _ = build_probe_artifact_blobs(
        family="neutral", records=records, onnx_by_cell=onnx, output_by_cell=outputs
    )

    assert blobs[0]["compiler_output_kind"] == "error"
    validate_probe_artifact_blobs(blobs, family="neutral", records=records)
