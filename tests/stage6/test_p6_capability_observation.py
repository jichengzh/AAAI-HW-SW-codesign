from __future__ import annotations

import base64
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from framework.stage6.p6_capability_observation_v1 import rebuild_probe_records
from framework.stage6.p6_capability_probe_worker_v1 import P6CompilerRejection


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _blob(*, q_mode: str, onnx: bytes, kind: str, output: bytes) -> dict:
    return {
        "schema_version": "p6_compiler_capability_artifact_v1",
        "family": "neutral",
        "probe_id": "P1",
        "q_mode": q_mode,
        "onnx_base64": base64.b64encode(onnx).decode("ascii"),
        "compiler_output_kind": kind,
        "compiler_output_base64": base64.b64encode(output).decode("ascii"),
        "compiler_output_sha256": _sha(output),
    }


@pytest.fixture
def fake_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("onnx")

    def load(payload: bytes):
        names = json.loads(payload)
        return SimpleNamespace(
            graph=SimpleNamespace(node=[SimpleNamespace(op_type=name) for name in names])
        )

    module.load_from_string = load
    monkeypatch.setitem(sys.modules, "onnx", module)


def test_raw_onnx_and_compiler_ir_independently_rebuild_numeric_records(
    fake_onnx: None,
) -> None:
    onnx = json.dumps(["Conv", "QuantizeLinear", "DequantizeLinear"]).encode()
    ir = b"@T.prim_func\ndef conv_int8():\n    layout_transform()\n"
    blobs = [
        _blob(q_mode="fp16", onnx=onnx, kind="compiler_ir", output=ir),
        _blob(q_mode="int8", onnx=onnx, kind="compiler_ir", output=ir),
    ]

    records = rebuild_probe_records(blobs, family="neutral", probe_ids=("P1",))

    assert records[1] == {
        "schema_version": "p6_compiler_capability_probe_record_v1",
        "probe_id": "P1",
        "q_mode": "int8",
        "onnx_sha256": _sha(onnx),
        "build_success": True,
        "observation_status": "observed_build_success",
        "compiler_ir_sha256": _sha(ir),
        "int8_propagated_ops": 1,
        "precision_eligible_ops": 1,
        "qdq_folded_pairs": 1,
        "qdq_pairs": 1,
        "reformat_ops": 1,
        "total_ops": 1,
        "fused_ops": 0,
        "fusible_ops": 1,
    }
    assert all(
        records[0][key] is None
        for key in (
            "int8_propagated_ops",
            "precision_eligible_ops",
            "qdq_folded_pairs",
            "qdq_pairs",
        )
    )


def test_raw_error_evidence_requires_allowlisted_canonical_category(
    fake_onnx: None,
) -> None:
    onnx = json.dumps(["Conv"]).encode()
    error = P6CompilerRejection("private compiler detail").evidence_bytes()
    ir = b"@T.prim_func\ndef conv():\n    pass\n"
    blobs = [
        _blob(q_mode="fp16", onnx=onnx, kind="compiler_ir", output=ir),
        _blob(q_mode="int8", onnx=onnx, kind="error", output=error),
    ]

    records = rebuild_probe_records(blobs, family="neutral", probe_ids=("P1",))
    assert records[1]["observation_status"] == "observed_build_failure"

    malformed = json.loads(error)
    malformed["category"] = "runtime_infrastructure"
    blobs[1] = _blob(
        q_mode="int8",
        onnx=onnx,
        kind="error",
        output=json.dumps(malformed, sort_keys=True, separators=(",", ":")).encode(),
    )
    with pytest.raises(ValueError, match="artifact observation"):
        rebuild_probe_records(blobs, family="neutral", probe_ids=("P1",))
