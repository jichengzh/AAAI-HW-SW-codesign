from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from framework.stage6.p6_tvm_runtime_authority_v1 import (
    FP16_FORMAL_ENERGY_HELPER_RELATIVE_PATH,
    P6TvmRuntimeAuthorityError,
    formal_fp16_tvm_code_digest,
)


def test_formal_fp16_digest_binds_exact_energy_helper_payload(tmp_path: Path) -> None:
    implementation, runtime_contract, helper = _write_fp16_authority(tmp_path)
    support_root_sha256 = "a" * 64
    payload = {
        "implementation_sha256": _sha256_file(implementation),
        "runtime_contract_sha256": _sha256_file(runtime_contract),
        "support_root_sha256": support_root_sha256,
        "fp16_energy_helpers": [
            {
                "path": FP16_FORMAL_ENERGY_HELPER_RELATIVE_PATH.as_posix(),
                "sha256": _sha256_file(helper),
                "size": helper.stat().st_size,
            }
        ],
    }

    assert formal_fp16_tvm_code_digest(
        implementation, runtime_contract, support_root_sha256
    ) == _canonical_sha256(payload)

    helper.write_text("HELPER_VERSION = 2\n", encoding="utf-8")
    assert formal_fp16_tvm_code_digest(
        implementation, runtime_contract, support_root_sha256
    ) != _canonical_sha256(payload)


@pytest.mark.parametrize("mutation", ("missing", "directory", "symlink", "hardlink"))
def test_formal_fp16_digest_rejects_invalid_energy_helper(
    tmp_path: Path, mutation: str
) -> None:
    implementation, runtime_contract, helper = _write_fp16_authority(tmp_path)
    original = helper.with_name("original-helper.py")
    if mutation == "missing":
        helper.unlink()
    elif mutation == "directory":
        helper.unlink()
        helper.mkdir()
    elif mutation == "symlink":
        helper.rename(original)
        helper.symlink_to(original)
    else:
        helper.rename(original)
        os.link(original, helper)

    with pytest.raises(P6TvmRuntimeAuthorityError):
        formal_fp16_tvm_code_digest(implementation, runtime_contract, "a" * 64)


@pytest.mark.parametrize("mutation", ("wrong-leaf", "symlink", "hardlink"))
def test_formal_fp16_digest_rejects_invalid_runner_leaf(
    tmp_path: Path, mutation: str
) -> None:
    implementation, runtime_contract, _helper = _write_fp16_authority(tmp_path)
    if mutation == "wrong-leaf":
        implementation = implementation.rename(
            implementation.with_name("alternate_fp16_runner.py")
        )
    else:
        original = implementation.with_name("original-fp16-runner.py")
        implementation.rename(original)
        if mutation == "symlink":
            implementation.symlink_to(original)
        else:
            os.link(original, implementation)

    with pytest.raises(P6TvmRuntimeAuthorityError):
        formal_fp16_tvm_code_digest(implementation, runtime_contract, "a" * 64)


def _write_fp16_authority(tmp_path: Path) -> tuple[Path, Path, Path]:
    code_root = tmp_path / "code-root"
    implementation = code_root / "scripts/stage2_route_b_fp16_auto_runner.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("RUNNER_VERSION = 1\n", encoding="utf-8")
    runtime_contract = code_root / "framework/stage5/tvm_runtime_contract_v1.py"
    runtime_contract.parent.mkdir(parents=True)
    runtime_contract.write_text("RUNTIME_VERSION = 1\n", encoding="utf-8")
    helper = code_root / FP16_FORMAL_ENERGY_HELPER_RELATIVE_PATH
    helper.parent.mkdir(parents=True)
    helper.write_text("HELPER_VERSION = 1\n", encoding="utf-8")
    return implementation, runtime_contract, helper


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
