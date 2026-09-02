from __future__ import annotations

import json
import os
from pathlib import Path
import py_compile
import sys
from types import ModuleType, SimpleNamespace

import pytest

from framework.stage6.p6_capability_tvm_v1 import (
    collect_tvm_runtime_identity,
    compile_tvm_probe,
    require_cuda_sm89,
)
from framework.stage6.p6_capability_probe_worker_v1 import P6CompilerRejection


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


class _TVMError(Exception):
    pass


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
    tvm.cuda = lambda index: SimpleNamespace(exist=index == 0, compute_version="8.9")
    tvm.target = SimpleNamespace(Target=_Target)
    tvm.transform = SimpleNamespace(
        Sequential=lambda passes: lambda module: _Lowered(),
        PassContext=lambda opt_level: _Context(),
    )
    tvm.compile = lambda module, target: (module, target)
    tvm.error = SimpleNamespace(TVMError=_TVMError)
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


def test_tvm_runtime_fails_closed_without_cuda_or_with_invalid_compiler_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    sys.modules["tvm"].cuda = lambda index: SimpleNamespace(exist=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        require_cuda_sm89()

    _fake_modules(tmp_path / "fresh", monkeypatch)
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")
    (tmp_path / "fresh" / "site" / "tvm" / "invalid.py").symlink_to(outside)
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        collect_tvm_runtime_identity(
            tvm_site=tmp_path / "fresh" / "site",
            nvlibs_file=nvlibs,
            support_root_sha256="a" * 64,
        )


def test_tvm_compile_only_translates_allowlisted_tvm_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_modules(tmp_path, monkeypatch)
    sys.modules["tvm"].compile = lambda module, target: (_ for _ in ()).throw(
        _TVMError("Unsupported instruction for CUDA target")
    )
    with pytest.raises(P6CompilerRejection) as captured:
        compile_tvm_probe(b"onnx", "int8")
    assert json.loads(captured.value.evidence_bytes())["category"] == (
        "tvm_codegen_compiler_rejection"
    )

    sys.modules["tvm"].compile = lambda module, target: (_ for _ in ()).throw(
        OSError("runtime infrastructure")
    )
    with pytest.raises(OSError, match="infrastructure"):
        compile_tvm_probe(b"onnx", "int8")


def test_tvm_runtime_manifest_is_import_state_independent_and_tree_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")
    baseline = collect_tvm_runtime_identity(
        tvm_site=site,
        nvlibs_file=nvlibs,
        support_root_sha256="a" * 64,
    )

    transient = ModuleType("tvm.transient")
    transient.__file__ = str(site / "tvm" / "__init__.py")
    monkeypatch.setitem(sys.modules, "tvm.transient", transient)
    imported = collect_tvm_runtime_identity(
        tvm_site=site,
        nvlibs_file=nvlibs,
        support_root_sha256="a" * 64,
    )
    assert imported == baseline

    extra = site / "tvm" / "unimported_compiler.py"
    extra.write_text("# compiler authority\n", encoding="utf-8")
    drifted = collect_tvm_runtime_identity(
        tvm_site=site,
        nvlibs_file=nvlibs,
        support_root_sha256="a" * 64,
    )
    assert drifted["compiler_fingerprint"] != baseline["compiler_fingerprint"]
    extra.unlink()
    assert (
        collect_tvm_runtime_identity(
            tvm_site=site,
            nvlibs_file=nvlibs,
            support_root_sha256="a" * 64,
        )
        == baseline
    )
    (site / "tvm" / "libtvm.so").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        collect_tvm_runtime_identity(
            tvm_site=site,
            nvlibs_file=nvlibs,
            support_root_sha256="a" * 64,
        )


def test_tvm_runtime_manifest_binds_sourceless_top_level_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")
    baseline = collect_tvm_runtime_identity(
        tvm_site=site,
        nvlibs_file=nvlibs,
        support_root_sha256="a" * 64,
    )
    source = site / "tvm" / "sourceless_authority.py"
    bytecode = site / "tvm" / "sourceless_authority.pyc"
    source.write_text("AUTHORITY = 'changed'\n", encoding="utf-8")
    py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
    source.unlink()

    changed = collect_tvm_runtime_identity(
        tvm_site=site,
        nvlibs_file=nvlibs,
        support_root_sha256="a" * 64,
    )

    assert changed["compiler_fingerprint"] != baseline["compiler_fingerprint"]


def test_tvm_runtime_manifest_binds_native_library_under_bytecode_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")
    baseline = collect_tvm_runtime_identity(
        tvm_site=site,
        nvlibs_file=nvlibs,
        support_root_sha256="a" * 64,
    )
    cache = site / "tvm" / "__pycache__"
    cache.mkdir()
    (cache / "native_compiler.so").write_bytes(b"native-compiler-authority")

    changed = collect_tvm_runtime_identity(
        tvm_site=site,
        nvlibs_file=nvlibs,
        support_root_sha256="a" * 64,
    )

    assert changed["compiler_fingerprint"] != baseline["compiler_fingerprint"]


def test_tvm_runtime_rejects_symlink_named_bytecode_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "native_compiler.so").write_bytes(b"outside-authority")
    (site / "tvm" / "__pycache__").symlink_to(outside, target_is_directory=True)
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="compiler file invalid"):
        collect_tvm_runtime_identity(
            tvm_site=site,
            nvlibs_file=nvlibs,
            support_root_sha256="a" * 64,
        )


def test_tvm_runtime_rejects_symlinked_site_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    alias = tmp_path / "site-alias"
    alias.symlink_to(site, target_is_directory=True)
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="compiler site invalid"):
        collect_tvm_runtime_identity(
            tvm_site=alias,
            nvlibs_file=nvlibs,
            support_root_sha256="a" * 64,
        )


def test_tvm_runtime_rejects_special_file_with_ignored_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, _ = _fake_modules(tmp_path, monkeypatch)
    os.mkfifo(site / "tvm" / "special.pyc")
    nvlibs = tmp_path / "nvlibs.json"
    nvlibs.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="compiler file invalid"):
        collect_tvm_runtime_identity(
            tvm_site=site,
            nvlibs_file=nvlibs,
            support_root_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "detail",
    (
        "no CUDA-capable device is detected",
        "CUDA_ERROR_NO_DEVICE",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "CUDA out of memory",
        "CUDA driver initialization failed",
        "invalid device ordinal",
        "nvcc not found",
        "No such file or directory",
        "cannot open shared object file",
        "ModuleNotFoundError: runtime package missing",
        "ImportError: binary ABI mismatch",
        "Read-only file system",
        "CUDA runtime failure",
        "InternalError: invariant violated",
        "compiler detail",
    ),
)
def test_tvmerror_infrastructure_categories_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detail: str
) -> None:
    _fake_modules(tmp_path, monkeypatch)
    sys.modules["tvm"].compile = lambda module, target: (_ for _ in ()).throw(_TVMError(detail))

    with pytest.raises(_TVMError, match=detail):
        compile_tvm_probe(b"onnx", "int8")


def test_tvm_compiler_rejection_evidence_binds_exact_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_modules(tmp_path, monkeypatch)
    sys.modules["tvm.relax.frontend.onnx"].from_onnx = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(_TVMError("Unsupported ONNX operator"))

    with pytest.raises(P6CompilerRejection) as captured:
        compile_tvm_probe(b"onnx", "int8")
    evidence = json.loads(captured.value.evidence_bytes())
    assert evidence["category"] == "tvm_frontend_compiler_rejection"


def test_tvm_lowering_compiler_rejection_binds_lowering_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_modules(tmp_path, monkeypatch)
    sys.modules["tvm"].transform.Sequential = lambda passes: (
        lambda module: (_ for _ in ()).throw(_TVMError("Cannot legalize operator nn.conv2d"))
    )

    with pytest.raises(P6CompilerRejection) as captured:
        compile_tvm_probe(b"onnx", "int8")

    evidence = json.loads(captured.value.evidence_bytes())
    assert evidence["category"] == "tvm_lowering_compiler_rejection"


def test_tvm_rejection_message_is_allowlisted_for_its_exact_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_modules(tmp_path, monkeypatch)
    sys.modules["tvm"].compile = lambda module, target: (_ for _ in ()).throw(
        _TVMError("Unsupported ONNX operator")
    )

    with pytest.raises(_TVMError, match="Unsupported ONNX operator"):
        compile_tvm_probe(b"onnx", "int8")
