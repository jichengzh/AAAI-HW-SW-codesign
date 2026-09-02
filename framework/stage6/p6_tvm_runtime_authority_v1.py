"""Canonical normalized authority for formal P6 TVM execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


TVM_SUPPORT_CLOSURE_ID = "tvm-support"
TVM_RUNTIME_CONTRACT_RELATIVE_PATH = Path(
    "framework/stage5/tvm_runtime_contract_v1.py"
)
_HEX_DIGEST = frozenset("0123456789abcdef")


class P6TvmRuntimeAuthorityError(ValueError):
    """Stable path-free formal TVM authority failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


def canonical_tvm_support_tree_sha256(root: Path) -> str:
    """Rebuild the byte-only support-tree digest used by the parent runtime."""
    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or _contains_symlink_component(root)
        or not root.is_dir()
    ):
        raise P6TvmRuntimeAuthorityError()
    entries: list[dict[str, object]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        try:
            info = path.lstat()
        except OSError:
            raise P6TvmRuntimeAuthorityError() from None
        if stat.S_ISDIR(info.st_mode):
            entries.append({"kind": "directory", "path": relative})
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            entries.append(
                {
                    "kind": "regular_file",
                    "path": relative,
                    "sha256": _sha256_file(path),
                    "size": info.st_size,
                }
            )
        else:
            raise P6TvmRuntimeAuthorityError()
    encoded = json.dumps(
        entries,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def formal_tvm_code_digest(
    implementation: Path,
    runtime_contract: Path,
    support_root_sha256: str,
) -> str:
    """Independently rebuild the parent formal leaf/code/support identity."""
    if (
        not _regular_single_link_file(implementation)
        or not _regular_single_link_file(runtime_contract)
        or not _is_sha256(support_root_sha256)
    ):
        raise P6TvmRuntimeAuthorityError()
    payload = {
        "implementation_sha256": _sha256_file(implementation),
        "runtime_contract_sha256": _sha256_file(runtime_contract),
        "support_root_sha256": support_root_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_single_link_file(path: object) -> bool:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or _contains_symlink_component(path)
    ):
        return False
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size > 0


def _contains_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX_DIGEST)
    )
