"""Static security contract for Stage1 checkpoint deserialization."""

from __future__ import annotations

import ast
from pathlib import Path


CHECKPOINT_LOADERS = (
    Path("framework/stage1/adapters.py"),
    Path("framework/stage1/auto_trace.py"),
    Path("framework/stage1/formal_scan_evidence.py"),
    Path("tools/configurable/depgraph_pyramid.py"),
    Path("tools/configurable/depgraph_v2xvit.py"),
)


def _torch_load_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]


def test_all_shared_stage1_checkpoint_loads_are_weights_only() -> None:
    insecure = []
    for path in CHECKPOINT_LOADERS:
        for call in _torch_load_calls(path):
            keyword = next(
                (item for item in call.keywords if item.arg == "weights_only"), None
            )
            if not isinstance(getattr(keyword, "value", None), ast.Constant) or (
                keyword.value.value is not True
            ):
                insecure.append(f"{path}:{call.lineno}")

    assert insecure == []


def test_formal_config_evidence_never_deserializes_model_native_yaml() -> None:
    path = Path("framework/stage1/formal_scan_evidence.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "yaml"
        and node.func.attr in {"load", "unsafe_load"}
    ]

    assert forbidden_calls == []
