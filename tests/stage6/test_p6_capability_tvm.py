from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from framework.stage6.p6_capability_tvm_v1 import (
    collect_tvm_runtime_identity,
    compile_tvm_probe,
    require_cuda_sm89,
)


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Target(_Context):
    kind = SimpleNamespace(name="cuda")

    @classmethod
    def from_device(cls, device):
        assert device.exist
        return cls()

    def __str__(self) -> str:
        return "cuda -arch=sm_89"


class _Lowered:
    def script(self, show_meta: bool) -> str:
        assert show_meta is True
        return "@T.prim_func\ndef conv_int8():\n    layout_transform()\n"


def _fake_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    site = tmp_path / "site"
    package = site / "tvm"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    library = package / "libtvm.so"
    init.write_text("# tvm\n", encoding="utf-8")
    library.write_bytes(b"compiler-library")
    dimensions = [SimpleNamespace(dim_value=value) for value in (1, 16, 8, 8)]
    model = SimpleNamespace(
        graph=SimpleNamespace(
            input=[
                SimpleNamespace(
                    name="input",
                    type=SimpleNamespace(
                        tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dimensions))
                    ),
                )
            ],
            node=[
                SimpleNamespace(op_type="Conv"),
                SimpleNamespace(op_type="QuantizeLinear"),
                SimpleNamespace(op_type="DequantizeLinear"),
            ],
        )
    )
    onnx = ModuleType("onnx")
    onnx.load_from_string = lambda payload: model
    tvm = ModuleType("tvm")
    tvm.__file__ = str(init)
    tvm.__version__ = "0.20.dev0"
    tvm.cuda = lambda index: SimpleNamespace(exist=index == 0)
    tvm.target = SimpleNamespace(Target=_Target)
    tvm.transform = SimpleNamespace(
        Sequential=lambda passes: lambda module: _Lowered(),
        PassContext=lambda opt_level: _Context(),
    )
    tvm.compile = lambda module, target: (module, target)
    relax = ModuleType("tvm.relax")
    relax.transform = SimpleNamespace(
        LegalizeOps=lambda: "legalize",
        AnnotateTIROpPattern=lambda: "annotate",
        FuseOps=lambda: "fuse-ops",
        FuseTIR=lambda: "fuse-tir",
    )
    frontend = ModuleType("tvm.relax.frontend")
    frontend_onnx = ModuleType("tvm.relax.frontend.onnx")
    frontend_onnx.from_onnx = lambda model, **kwargs: (model, kwargs)
    ffi = ModuleType("tvm._ffi")
    ffi.libinfo = SimpleNamespace(find_lib_path=lambda: [str(library)])
    tvm.relax = relax
    for name, module in {
        "onnx": onnx,
        "tvm": tvm,
        "tvm.relax": relax,
        "tvm.relax.frontend": frontend,
        "tvm.relax.frontend.onnx": frontend_onnx,
        "tvm._ffi": ffi,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return site, model


def test_tvm_compile_observes_structural_counts_and_raw_ir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_modules(tmp_path, monkeypatch)

    counts, ir = compile_tvm_probe(b"onnx", "int8")
    fp16, _ = compile_tvm_probe(b"onnx", "fp16")

    assert counts == {
        "int8_propagated_ops": 1,
        "precision_eligible_ops": 1,
        "qdq_folded_pairs": 1,
        "qdq_pairs": 1,
        "reformat_ops": 1,
        "total_ops": 1,
        "fused_ops": 0,
        "fusible_ops": 1,
    }
    assert ir.startswith(b"@T.prim_func")
    assert all(
        fp16[key] is None
        for key in (
            "int8_propagated_ops",
            "precision_eligible_ops",
            "qdq_folded_pairs",
            "qdq_pairs",
        )
    )


def test_tvm_runtime_identity_is_path_free_and_byte_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")

    identity = collect_tvm_runtime_identity(
        tvm_site=site,
        nvlibs_file=nvlibs,
        support_root_sha256="a" * 64,
    )

    assert identity["tvm_arch"] == "sm89"
    assert len(identity["compiler_file_sha256"]) == 2
    assert "/" not in json.dumps(identity)
    assert require_cuda_sm89()[1].kind.name == "cuda"


def test_tvm_runtime_fails_closed_without_cuda_or_with_outside_compiler_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    sys.modules["tvm"].cuda = lambda index: SimpleNamespace(exist=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        require_cuda_sm89()

    _fake_modules(tmp_path / "fresh", monkeypatch)
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")
    sys.modules["tvm.bad"] = SimpleNamespace(__file__=str(outside))
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        collect_tvm_runtime_identity(
            tvm_site=tmp_path / "fresh" / "site",
            nvlibs_file=nvlibs,
            support_root_sha256="a" * 64,
        )
