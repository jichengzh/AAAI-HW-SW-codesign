"""Path-free, content-addressed ONNX and compiler evidence for RTX probes."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


ARTIFACT_SCHEMA_VERSION = "p6_compiler_capability_artifact_v1"
_FAMILIES = frozenset({"neutral", "pruning"})
_KINDS = frozenset({"compiler_ir", "error"})
_KEYS = frozenset(
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


class P6CapabilityArtifactError(ValueError):
    """Stable path-free rejection for invalid embedded evidence."""

    def __init__(self) -> None:
        super().__init__("capability_artifact_invalid")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encode(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _decode(payload: object) -> bytes:
    if not isinstance(payload, str) or not payload:
        raise P6CapabilityArtifactError()
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as error:
        raise P6CapabilityArtifactError() from error
    if not decoded or _encode(decoded) != payload:
        raise P6CapabilityArtifactError()
    return decoded


def _cell(raw: Mapping[str, Any]) -> tuple[str, str]:
    probe_id = raw.get("probe_id")
    q_mode = raw.get("q_mode")
    if not isinstance(probe_id, str) or not probe_id or q_mode not in {"fp16", "int8"}:
        raise P6CapabilityArtifactError()
    return probe_id, str(q_mode)


def _record_index(records: object) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise P6CapabilityArtifactError()
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise P6CapabilityArtifactError()
        record = dict(raw)
        cell = _cell(record)
        if cell in index:
            raise P6CapabilityArtifactError()
        index[cell] = record
    if not index:
        raise P6CapabilityArtifactError()
    return index


def _validated_blob(
    raw: object,
    *,
    family: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _KEYS:
        raise P6CapabilityArtifactError()
    blob = dict(raw)
    onnx = _decode(blob.get("onnx_base64"))
    output = _decode(blob.get("compiler_output_base64"))
    expected_kind = "compiler_ir" if record.get("build_success") is True else "error"
    if (
        blob.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or blob.get("family") != family
        or _cell(blob) != _cell(record)
        or blob.get("compiler_output_kind") != expected_kind
        or _sha(onnx) != record.get("onnx_sha256")
        or _sha(output) != blob.get("compiler_output_sha256")
        or expected_kind == "compiler_ir"
        and _sha(output) != record.get("compiler_ir_sha256")
    ):
        raise P6CapabilityArtifactError()
    return blob


def _manifest_sha256(
    blobs: Sequence[Mapping[str, Any]],
    records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    rows = [
        {
            "schema_version": blob["schema_version"],
            "family": blob["family"],
            "probe_id": blob["probe_id"],
            "q_mode": blob["q_mode"],
            "onnx_sha256": _sha(_decode(blob["onnx_base64"])),
            "compiler_output_kind": blob["compiler_output_kind"],
            "compiler_output_sha256": blob["compiler_output_sha256"],
            "record": dict(records[_cell(blob)]),
        }
        for blob in blobs
    ]
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return _sha(payload)


def validate_probe_artifact_blobs(
    raw: object, *, family: str, records: object
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Rebuild every byte digest and require one artifact per probe cell."""
    if family not in _FAMILIES or not isinstance(raw, list):
        raise P6CapabilityArtifactError()
    record_by_cell = _record_index(records)
    if len(raw) != len(record_by_cell):
        raise P6CapabilityArtifactError()
    blobs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise P6CapabilityArtifactError()
        cell = _cell(item)
        if cell in seen or cell not in record_by_cell:
            raise P6CapabilityArtifactError()
        seen.add(cell)
        blobs.append(_validated_blob(item, family=family, record=record_by_cell[cell]))
    if seen != set(record_by_cell):
        raise P6CapabilityArtifactError()
    ordered = tuple(sorted(blobs, key=lambda item: (item["probe_id"], item["q_mode"])))
    return ordered, _manifest_sha256(ordered, record_by_cell)


def build_probe_artifact_blobs(
    *,
    family: str,
    records: object,
    onnx_by_cell: Mapping[tuple[str, str], bytes],
    output_by_cell: Mapping[tuple[str, str], bytes],
) -> tuple[list[dict[str, Any]], str]:
    """Embed raw evidence so downstream validation never trusts a private path."""
    record_by_cell = _record_index(records)
    if set(onnx_by_cell) != set(record_by_cell) or set(output_by_cell) != set(record_by_cell):
        raise P6CapabilityArtifactError()
    blobs = []
    for cell, record in record_by_cell.items():
        output = output_by_cell[cell]
        blobs.append(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "family": family,
                "probe_id": cell[0],
                "q_mode": cell[1],
                "onnx_base64": _encode(onnx_by_cell[cell]),
                "compiler_output_kind": (
                    "compiler_ir" if record.get("build_success") is True else "error"
                ),
                "compiler_output_base64": _encode(output),
                "compiler_output_sha256": _sha(output),
            }
        )
    validated, digest = validate_probe_artifact_blobs(
        blobs, family=family, records=list(record_by_cell.values())
    )
    return list(validated), digest


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "P6CapabilityArtifactError",
    "build_probe_artifact_blobs",
    "validate_probe_artifact_blobs",
]
