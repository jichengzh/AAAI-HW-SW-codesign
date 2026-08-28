"""Canonical immutable evidence derived from model-native configuration objects."""

from __future__ import annotations

from collections.abc import Mapping
import math
from types import MappingProxyType
from typing import Any

import numpy as np


def canonicalize_model_config(value: Any) -> Mapping[str, Any]:
    """Normalize one trusted model-native config to immutable JSON primitives."""

    normalized = _canonical_value(value, "$", set())
    if not isinstance(normalized, Mapping):
        raise TypeError("model config root must be a mapping")
    return normalized


def _canonical_value(value: Any, path: str, active: set[int]) -> Any:
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist(), path, active)
    if isinstance(value, np.generic):
        return _canonical_value(value.item(), path, active)
    if isinstance(value, Mapping):
        return _canonical_mapping(value, path, active)
    if isinstance(value, (list, tuple)):
        return _canonical_sequence(value, path, active)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"model config {path} must contain finite numbers")
        return value
    raise TypeError(f"model config {path} contains an unsupported value")


def _canonical_mapping(value: Mapping[Any, Any], path: str, active: set[int]) -> Mapping[str, Any]:
    _enter_container(value, path, active)
    try:
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"model config {path} mapping keys must be strings")
        return MappingProxyType(
            {key: _canonical_value(value[key], f"{path}.{key}", active) for key in sorted(value)}
        )
    finally:
        active.remove(id(value))


def _canonical_sequence(value: Any, path: str, active: set[int]) -> tuple[Any, ...]:
    _enter_container(value, path, active)
    try:
        return tuple(
            _canonical_value(child, f"{path}[{index}]", active) for index, child in enumerate(value)
        )
    finally:
        active.remove(id(value))


def _enter_container(value: Any, path: str, active: set[int]) -> None:
    identity = id(value)
    if identity in active:
        raise ValueError(f"model config {path} contains a recursive container")
    active.add(identity)


__all__ = ["canonicalize_model_config"]
