"""Real TVM/CUDA structural compiler observations for RTX capability probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any

from framework.stage6.p6_capability_context_v1 import (
    canonical_cuda_target,
    compiler_fingerprint,
)
from framework.stage6.p6_capability_observation_v1 import derive_successful_counts
from framework.stage6.p6_capability_probe_worker_v1 import P6CompilerRejection


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not path.is_symlink()


def _loaded_tvm_files(tvm_site: Path) -> tuple[Path, ...]:
    files: set[Path] = set()
    root = tvm_site.resolve(strict=True)
    for name, module in sys.modules.items():
        if name != "tvm" and not name.startswith("tvm."):
            continue
        value = getattr(module, "__file__", None)
        if not value:
            continue
        path = Path(value).resolve(strict=True)
        path.relative_to(root)
        if not _regular_file(path):
            raise ValueError("TVM compiler file invalid")
        files.add(path)
    try:
        from tvm._ffi import libinfo

        for value in libinfo.find_lib_path():
            path = Path(value).resolve(strict=True)
            path.relative_to(root)
            if not _regular_file(path):
                raise ValueError("TVM compiler file invalid")
            files.add(path)
    except ImportError as error:
        raise ValueError("TVM compiler library unavailable") from error
    if not files:
        raise ValueError("TVM compiler files unavailable")
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _site_digest(tvm_site: Path, files: tuple[Path, ...]) -> str:
    root = tvm_site.resolve(strict=True)
    payload = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha_file(path)} for path in files
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_cuda_sm89() -> tuple[Any, Any]:
    """Require the selected TVM runtime to expose one live CUDA SM89 device."""
    import tvm

    device = tvm.cuda(0)
    if not device.exist:
        raise RuntimeError("TVM CUDA device unavailable")
    target = tvm.target.Target.from_device(device)
    target_text = str(target)
    if target.kind.name != "cuda" or re.search(r"(?:sm_|sm)(?:89)\b", target_text) is None:
        raise RuntimeError("TVM CUDA target is not SM89")
    return device, target


def compile_tvm_probe(payload: bytes, q_mode: str) -> tuple[dict[str, int | None], bytes]:
    """Compile one ONNX probe with the live CUDA target and return raw lowered IR."""
    import onnx
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    model = onnx.load_from_string(payload)
    shapes = {
        item.name: tuple(int(dim.dim_value) for dim in item.type.tensor_type.shape.dim)
        for item in model.graph.input
    }
    try:
        module = from_onnx(model, shape_dict=shapes, keep_params_in_input=False)
        _, target = require_cuda_sm89()
        passes = tvm.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )
        with target, tvm.transform.PassContext(opt_level=3):
            lowered = passes(module)
            tvm.compile(lowered, target=target)
    except tvm.error.TVMError as error:
        raise P6CompilerRejection(f"{type(error).__name__}:{error}") from error
    script = lowered.script(show_meta=True)
    raw_ir = script.encode("utf-8")
    return derive_successful_counts(payload, raw_ir, q_mode), raw_ir


def collect_tvm_runtime_identity(
    *,
    tvm_site: Path,
    nvlibs_file: Path,
    support_root_sha256: str,
) -> dict[str, Any]:
    """Collect normalized versions and compiler-byte identities after compilation."""
    import tvm

    device, _ = require_cuda_sm89()
    files = _loaded_tvm_files(tvm_site)
    if not _regular_file(nvlibs_file):
        raise ValueError("NVLIBS authority invalid")
    python = Path(sys.executable).resolve(strict=True)
    if not _regular_file(python):
        raise ValueError("Python runtime invalid")
    cuda_compute_version = str(getattr(device, "compute_version", "")).strip()
    if cuda_compute_version != "8.9":
        raise ValueError("CUDA compute version invalid")
    identity = {
        "schema_version": "p6_tvm_compiler_runtime_identity_v1",
        "tvm_version": str(tvm.__version__),
        "python_version": platform.python_version(),
        "python_executable_sha256": _sha_file(python),
        "target": canonical_cuda_target("sm89"),
        "tvm_arch": "sm89",
        "cuda_compute_version": cuda_compute_version,
        "support_root_sha256": support_root_sha256,
        "tvm_site_sha256": _site_digest(tvm_site, files),
        "nvlibs_file_sha256": _sha_file(nvlibs_file),
        "compiler_file_sha256": sorted({_sha_file(path) for path in files}),
    }
    return {**identity, "compiler_fingerprint": compiler_fingerprint(identity)}


__all__ = [
    "collect_tvm_runtime_identity",
    "compile_tvm_probe",
    "require_cuda_sm89",
]
