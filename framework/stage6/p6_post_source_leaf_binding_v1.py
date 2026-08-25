"""Validate the private P6 post-source historical leaf binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from framework.stage6.p6_history_execution_closure_v1 import (
    P6ValidatedExecutionClosure,
)


LEAF_BINDING_SCHEMA_VERSION = "p6_post_source_leaf_binding_v1"
POST_SOURCE_LEAF_NAMES: tuple[str, ...] = (
    "quant_contract",
    "performance_plan",
    "performance_execute",
    "ap_plan",
    "ap_execute",
    "feedback_finalize",
    "feedback_promote",
)
LEAF_BINDING_KEYS = frozenset({"schema_version", "leaves"})
LEAF_KEYS = frozenset({"closure_id", "entrypoint_relative_path"})


@dataclass(frozen=True)
class BoundPostSourceLeaf:
    name: str
    closure_id: str
    entrypoint_relative_path: Path
    source_root: Path
    destination_relative_root: Path


@dataclass(frozen=True)
class ValidatedPostSourceLeafBinding:
    schema_version: Literal["p6_post_source_leaf_binding_v1"]
    leaves: tuple[BoundPostSourceLeaf, ...]


class P6PostSourceLeafBindingError(ValueError):
    """Stable path-free post-source leaf binding failure."""

    def __init__(self) -> None:
        super().__init__("history_normalization_invalid")


def validate_post_source_leaf_binding(
    raw: object,
    *,
    execution_closure: P6ValidatedExecutionClosure,
) -> ValidatedPostSourceLeafBinding:
    """Bind exactly seven historical leaves to declared execution closure roots."""
    if not isinstance(execution_closure, P6ValidatedExecutionClosure):
        _invalid()
    roots = {root.closure_id: root for root in execution_closure.roots}
    if (
        not isinstance(raw, Mapping)
        or set(raw) != LEAF_BINDING_KEYS
        or raw.get("schema_version") != LEAF_BINDING_SCHEMA_VERSION
    ):
        _invalid()
    raw_leaves = raw.get("leaves")
    if not isinstance(raw_leaves, Mapping) or set(raw_leaves) != set(
        POST_SOURCE_LEAF_NAMES
    ):
        _invalid()
    leaves = tuple(_validated_leaf(name, raw_leaves[name], roots) for name in POST_SOURCE_LEAF_NAMES)
    resolved = tuple(leaf.source_root / leaf.entrypoint_relative_path for leaf in leaves)
    if len(set(resolved)) != len(resolved):
        _invalid()
    return ValidatedPostSourceLeafBinding(LEAF_BINDING_SCHEMA_VERSION, leaves)


def _validated_leaf(
    name: str,
    raw: object,
    roots: Mapping[str, Any],
) -> BoundPostSourceLeaf:
    if not isinstance(raw, Mapping) or set(raw) != LEAF_KEYS:
        _invalid()
    closure_id = raw.get("closure_id")
    relative = _relative_path(raw.get("entrypoint_relative_path"))
    root = roots.get(closure_id)
    if not isinstance(closure_id, str) or root is None:
        _invalid()
    source = root.source_root / relative
    if not _is_executable_file(source):
        _invalid()
    return BoundPostSourceLeaf(
        name=name,
        closure_id=closure_id,
        entrypoint_relative_path=relative,
        source_root=root.source_root,
        destination_relative_root=root.destination_relative_root,
    )


def _relative_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw or "//" in raw:
        _invalid()
    path = Path(raw)
    if (
        path.as_posix() != raw
        or path.is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or path == Path(".")
        or ".." in path.parts
        or ".git" in path.parts
    ):
        _invalid()
    return path


def _is_executable_file(path: Path) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return path.is_file() and stat.st_nlink == 1 and os.access(path, os.X_OK)


def _invalid() -> None:
    raise P6PostSourceLeafBindingError()
