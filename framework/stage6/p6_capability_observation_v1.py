"""Rebuild capability records only from retained ONNX/compiler output bytes."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any

from framework.stage6.p6_capability_probe_specs_v1 import (
    COUNT_FIELDS,
    RECORD_SCHEMA_VERSION,
    validate_probe_partition,
)
from framework.stage6.p6_capability_probe_worker_v1 import (
    COMPILER_REJECTION_SCHEMA_VERSION,
)


_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "family",
        "probe_id",
        "q_mode",
        "onnx_base64",
        "compiler_output_kind",
        "compiler_output_base64",
        "compiler_output_sha256",
    }
)
_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode(raw: object) -> bytes:
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact observation invalid")
    try:
        payload = base64.b64decode(raw, validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("artifact observation invalid") from error
    if not payload or base64.b64encode(payload).decode("ascii") != raw:
        raise ValueError("artifact observation invalid")
    return payload


def _onnx_operator_counts(payload: bytes) -> dict[str, int]:
    import onnx

    model = onnx.load_from_string(payload)
    checker = getattr(onnx, "checker", None)
    if checker is not None:
        checker.check_model(model)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        name = str(node.op_type)
        counts[name] = counts.get(name, 0) + 1
    return counts


def derive_successful_counts(
    onnx_payload: bytes, compiler_ir: bytes, q_mode: str
) -> dict[str, int | None]:
    """Apply the existing TVM structural-probe semantics to raw retained bytes."""
    if q_mode not in {"fp16", "int8"}:
        raise ValueError("artifact observation invalid")
    try:
        script = compiler_ir.decode("utf-8", errors="strict")
        counts = _onnx_operator_counts(onnx_payload)
    except (AttributeError, UnicodeError, ValueError) as error:
        raise ValueError("artifact observation invalid") from error
    if not script or "@T.prim_func" not in script:
        raise ValueError("artifact observation invalid")
    conv_count = counts.get("Conv", 0)
    qdq_pairs = min(counts.get("QuantizeLinear", 0), counts.get("DequantizeLinear", 0))
    sections = script.split("    @T.prim_func")
    int8_convs = sum(
        "int8" in item.lower() for item in sections if "def conv" in item[:300].lower()
    )
    result: dict[str, int | None] = {
        "int8_propagated_ops": min(conv_count, int8_convs) if qdq_pairs else 0,
        "precision_eligible_ops": conv_count,
        "qdq_folded_pairs": qdq_pairs if "quantize" not in script.lower() else 0,
        "qdq_pairs": qdq_pairs,
        "reformat_ops": script.lower().count("layout_transform"),
        "total_ops": max(1, script.count("@T.prim_func")),
        "fused_ops": 0,
        "fusible_ops": conv_count,
    }
    if q_mode == "fp16":
        for key in COUNT_FIELDS[:4]:
            result[key] = None
    return result


def _rejection_evidence(output: bytes) -> None:
    try:
        payload = json.loads(output)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact observation invalid") from error
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "category", "detail_sha256"}
        or payload.get("schema_version") != COMPILER_REJECTION_SCHEMA_VERSION
        or payload.get("category") != "tvm_compiler_rejection"
        or not isinstance(payload.get("detail_sha256"), str)
        or _SHA_PATTERN.fullmatch(payload["detail_sha256"]) is None
    ):
        raise ValueError("artifact observation invalid")


def _record_from_blob(blob: Mapping[str, Any]) -> dict[str, Any]:
    onnx = _decode(blob["onnx_base64"])
    output = _decode(blob["compiler_output_base64"])
    if _sha(output) != blob["compiler_output_sha256"]:
        raise ValueError("artifact observation invalid")
    kind = blob["compiler_output_kind"]
    success = kind == "compiler_ir"
    if kind == "error":
        _rejection_evidence(output)
        counts: dict[str, int | None] = {name: None for name in COUNT_FIELDS}
    elif success:
        counts = derive_successful_counts(onnx, output, str(blob["q_mode"]))
    else:
        raise ValueError("artifact observation invalid")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "probe_id": blob["probe_id"],
        "q_mode": blob["q_mode"],
        "onnx_sha256": _sha(onnx),
        "build_success": success,
        "observation_status": "observed_build_success" if success else "observed_build_failure",
        "compiler_ir_sha256": _sha(output) if success else None,
        **counts,
    }


def rebuild_probe_records(
    raw: object, *, family: str, probe_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    """Require the exact raw partition and reconstruct every numeric record."""
    if not isinstance(raw, list):
        raise ValueError("artifact observation invalid")
    records = []
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or set(item) != _ARTIFACT_KEYS
            or item.get("schema_version") != "p6_compiler_capability_artifact_v1"
            or item.get("family") != family
        ):
            raise ValueError("artifact observation invalid")
        records.append(_record_from_blob(item))
    return validate_probe_partition(records, probe_ids=probe_ids)


__all__ = ["derive_successful_counts", "rebuild_probe_records"]
