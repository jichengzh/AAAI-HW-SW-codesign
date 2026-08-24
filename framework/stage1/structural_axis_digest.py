"""Canonical digests and immutable mappings for structural-axis evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


SCANNER_AXIS_PROVENANCE_SOURCE = "stage1.graph_scan.build_structural_axis_inputs"


def canonical_digest(value: Any) -> str:
    primitive = _canonical_json_primitive(value)
    encoded = json.dumps(
        primitive,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _immutable_value(value)


def mutable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return _mutable_value(value)


def scanner_structural_axes_digest(axes: Any) -> str:
    """Seal the canonical scanner-axis payload without digest self-reference."""

    if not isinstance(axes, (list, tuple)):
        raise TypeError("scanner structural axes must be a sequence")
    ordered = sorted(
        (dict(axis) for axis in axes), key=lambda axis: str(axis.get("axis_id") or "")
    )
    return canonical_digest(ordered)


def _immutable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _immutable_value(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_value(child) for child in value)
    return value


def _mutable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_value(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_mutable_value(child) for child in value)
    return value


def _canonical_json_primitive(value: Any, path: str = "$") -> Any:
    if isinstance(value, Mapping):
        normalized = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            normalized[key] = _canonical_json_primitive(child, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_primitive(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"{path} must contain only JSON primitive values")


__all__ = [
    "SCANNER_AXIS_PROVENANCE_SOURCE",
    "canonical_digest",
    "immutable_mapping",
    "mutable_mapping",
    "scanner_structural_axes_digest",
]
