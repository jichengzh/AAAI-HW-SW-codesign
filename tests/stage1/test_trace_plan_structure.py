from __future__ import annotations

import ast
from pathlib import Path

from framework.stage1 import trace_plan


_PUBLIC_CALLABLES = (
    "ModuleRecord",
    "TraceCandidate",
    "ModuleTreeScanner",
    "HeuristicTagger",
    "DensePathFinder",
    "GeneratedTraceWrapper",
    "WrapperSynthesizer",
    "BoundaryValidator",
    "TraceBoundaryDetector",
    "attach_runtime_validation",
    "legacy_trace_plan_from_manifest",
)


def test_trace_plan_facade_keeps_public_api_ownership() -> None:
    assert trace_plan.__all__ == [
        "TRACE_PLAN_SCHEMA",
        *_PUBLIC_CALLABLES,
    ]
    assert {
        name: getattr(trace_plan, name).__module__ for name in _PUBLIC_CALLABLES
    } == {
        name: "framework.stage1.trace_plan" for name in _PUBLIC_CALLABLES
    }


def test_trace_plan_production_modules_stay_within_size_limits() -> None:
    stage1_dir = Path(trace_plan.__file__).resolve().parent
    modules = sorted(stage1_dir.glob("trace_plan*.py"))

    assert modules
    for module in modules:
        source = module.read_text(encoding="utf-8")
        assert len(source.splitlines()) < 800, module.name
        tree = ast.parse(source)
        oversized = {
            node.name: node.end_lineno - node.lineno + 1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.end_lineno is not None
            and node.end_lineno - node.lineno + 1 >= 50
        }
        assert oversized == {}, f"{module.name}: {oversized}"
