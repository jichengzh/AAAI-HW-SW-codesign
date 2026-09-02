"""Execute fixed compiler probes and retain path-free structural evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from framework.stage6.p6_capability_artifacts_v1 import build_probe_artifact_blobs
from framework.stage6.p6_capability_probe_specs_v1 import (
    COUNT_FIELDS,
    RECORD_SCHEMA_VERSION,
)


ModelBuilder = Callable[[str, str], bytes]
Compiler = Callable[[bytes, str], tuple[Mapping[str, Any], bytes]]
COMPILER_REJECTION_SCHEMA_VERSION = "p6_compiler_rejection_evidence_v1"


class P6CompilerRejection(Exception):
    """One allowlisted compiler rejection with path-free retained evidence."""

    def __init__(self, detail: str) -> None:
        if not isinstance(detail, str) or not detail:
            raise ValueError("compiler rejection detail invalid")
        self.detail_sha256 = hashlib.sha256(detail.encode("utf-8")).hexdigest()
        super().__init__("tvm_compiler_rejection")

    def evidence_bytes(self) -> bytes:
        return json.dumps(
            {
                "schema_version": COMPILER_REJECTION_SCHEMA_VERSION,
                "category": "tvm_compiler_rejection",
                "detail_sha256": self.detail_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")


@dataclass(frozen=True)
class ProbeFamilyResult:
    records: tuple[dict[str, Any], ...]
    artifact_blobs: tuple[dict[str, Any], ...]
    manifest_sha256: str


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _failed_counts() -> dict[str, None]:
    return {name: None for name in COUNT_FIELDS}


def _successful_record(
    *, probe_id: str, q_mode: str, onnx: bytes, counts: Mapping[str, Any], ir: bytes
) -> dict[str, Any]:
    if set(counts) != set(COUNT_FIELDS) or not ir:
        raise ValueError("compiler observation invalid")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "probe_id": probe_id,
        "q_mode": q_mode,
        "onnx_sha256": _sha(onnx),
        "build_success": True,
        "observation_status": "observed_build_success",
        "compiler_ir_sha256": _sha(ir),
        **dict(counts),
    }


def _failed_record(*, probe_id: str, q_mode: str, onnx: bytes) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "probe_id": probe_id,
        "q_mode": q_mode,
        "onnx_sha256": _sha(onnx),
        "build_success": False,
        "observation_status": "observed_build_failure",
        "compiler_ir_sha256": None,
        **_failed_counts(),
    }


def _observe_cell(
    probe_id: str,
    q_mode: str,
    *,
    model_builder: ModelBuilder,
    compiler: Compiler,
) -> tuple[dict[str, Any], bytes, bytes]:
    onnx = model_builder(probe_id, q_mode)
    if not isinstance(onnx, bytes) or not onnx:
        raise ValueError("probe model bytes invalid")
    try:
        counts, compiler_output = compiler(onnx, q_mode)
    except P6CompilerRejection as error:
        record = _failed_record(probe_id=probe_id, q_mode=q_mode, onnx=onnx)
        compiler_output = error.evidence_bytes()
    else:
        record = _successful_record(
            probe_id=probe_id,
            q_mode=q_mode,
            onnx=onnx,
            counts=counts,
            ir=compiler_output,
        )
    return record, onnx, compiler_output


def run_probe_family(
    *,
    family: str,
    probe_ids: Sequence[str],
    model_builder: ModelBuilder,
    compiler: Compiler,
) -> ProbeFamilyResult:
    """Run every requested probe exactly once in each canonical precision."""
    if family not in {"neutral", "pruning"} or not probe_ids:
        raise ValueError("probe family invalid")
    if any(not isinstance(probe_id, str) or not probe_id for probe_id in probe_ids):
        raise ValueError("probe identities invalid")
    if len(set(probe_ids)) != len(probe_ids):
        raise ValueError("probe identities invalid")
    records = []
    onnx_by_cell = {}
    output_by_cell = {}
    for probe_id in probe_ids:
        for q_mode in ("fp16", "int8"):
            record, onnx, output = _observe_cell(
                probe_id, q_mode, model_builder=model_builder, compiler=compiler
            )
            cell = (probe_id, q_mode)
            records.append(record)
            onnx_by_cell[cell] = onnx
            output_by_cell[cell] = output
    blobs, manifest_sha256 = build_probe_artifact_blobs(
        family=family,
        records=records,
        onnx_by_cell=onnx_by_cell,
        output_by_cell=output_by_cell,
    )
    return ProbeFamilyResult(tuple(records), tuple(blobs), manifest_sha256)


__all__ = [
    "COMPILER_REJECTION_SCHEMA_VERSION",
    "P6CompilerRejection",
    "ProbeFamilyResult",
    "run_probe_family",
]
