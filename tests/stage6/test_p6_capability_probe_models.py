from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.stage6.p6_capability_probe_models_v1 import build_probe_onnx
from framework.stage6 import p6_capability_probe_models_v1 as models
from framework.stage6.p6_capability_probe_specs_v1 import (
    NEUTRAL_PROBE_IDS,
    PRUNING_PROBE_IDS,
)


def _onnx_package():
    root = Path(__file__).resolve().parents[2]
    loaded = sys.modules.pop("onnx", None)
    original = list(sys.path)
    try:
        sys.path[:] = [item for item in sys.path if item and Path(item).resolve() != root]
        try:
            return importlib.import_module("onnx")
        except ModuleNotFoundError:
            pytest.skip("ONNX is validated under the approved compiler Python")
    finally:
        sys.path[:] = original
        if loaded is not None and not hasattr(loaded, "helper"):
            sys.modules["onnx"] = loaded


def test_fixed_probe_models_are_checked_deterministic_and_unique() -> None:
    onnx = _onnx_package()
    digests = set()
    for family, probe_ids in (
        ("neutral", NEUTRAL_PROBE_IDS),
        ("pruning", PRUNING_PROBE_IDS),
    ):
        for probe_id in probe_ids:
            for q_mode in ("fp16", "int8"):
                payload = build_probe_onnx(family, probe_id, q_mode)
                assert payload == build_probe_onnx(family, probe_id, q_mode)
                onnx.checker.check_model(onnx.load_from_string(payload))
                digests.add(hashlib.sha256(payload).hexdigest())

    assert len(digests) == 24


class _FakeModel:
    def __init__(self, graph):
        self.graph = graph
        self.ir_version = 9

    def SerializeToString(self, deterministic: bool = False) -> bytes:
        assert deterministic is True
        return repr(self.graph).encode()


def _fake_onnx():
    helper = SimpleNamespace(
        make_node=lambda kind, inputs, outputs, **kwargs: (
            kind,
            tuple(inputs),
            tuple(outputs),
            tuple(sorted(kwargs.items())),
        ),
        make_tensor_value_info=lambda name, dtype, shape: (name, dtype, tuple(shape)),
        make_graph=lambda nodes, name, inputs, outputs, initializer: (
            tuple(nodes),
            name,
            tuple(inputs),
            tuple(outputs),
            tuple(map(repr, initializer)),
        ),
        make_model=lambda graph, **kwargs: _FakeModel((graph, tuple(sorted(kwargs.items())))),
        make_opsetid=lambda domain, version: (domain, version),
    )
    return SimpleNamespace(
        TensorProto=SimpleNamespace(FLOAT16=10, FLOAT=1),
        numpy_helper=SimpleNamespace(from_array=lambda values, name: (name, values.tolist())),
        helper=helper,
        checker=SimpleNamespace(check_model=lambda model: model),
    )


def test_all_model_shapes_and_qdq_branches_build_without_runtime_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "_onnx_package", _fake_onnx)
    payloads = {
        build_probe_onnx(family, probe_id, q_mode)
        for family, probe_ids in (
            ("neutral", NEUTRAL_PROBE_IDS),
            ("pruning", PRUNING_PROBE_IDS),
        )
        for probe_id in probe_ids
        for q_mode in ("fp16", "int8")
    }
    assert len(payloads) == 24
    with pytest.raises(ValueError, match="precision"):
        build_probe_onnx("neutral", "P1", "fp32")
    with pytest.raises(ValueError, match="identity"):
        build_probe_onnx("alternate", "P1", "fp16")
