"""Rebuild RTX capability authority under the normalized formal TVM runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

from framework.stage6.p6_capability_context_v1 import compiler_fingerprint
from framework.stage6.p6_capability_probe_specs_v1 import (
    NEUTRAL_PROBE_IDS,
    PRUNING_PROBE_IDS,
    validate_probe_partition,
)
from framework.stage6.p6_gpu_policy_v1 import parse_gpu_indices_csv
from framework.stage6.hardware_execution_profile_v1 import validate_profile_gpu_policy
from framework.stage6.p6_history_binding_v1 import (
    EXPECTED_HISTORY_ENV_KEYS,
    FORMAL_TVM_ENVIRONMENT_SPEC,
    FORMAL_TVM_ENV_KEYS,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    load_post_source_adapter_profile,
    require_post_source_adapter_profile_v4,
)
from framework.stage6.p6_python_runtime_v1 import validate_adapter_python
from framework.stage6.p6_runner_template_validator_v1 import (
    validate_pre_provision_runner_template,
)


AUTHORITY_SCHEMA_VERSION = "p6_normalized_capability_authority_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HELPER_RELATIVE_PATH = Path("tools/release/rebuild_p6_rtx_capability_authority.py")


class P6CapabilityRuntimeAuthorityError(ValueError):
    """Stable path-free rejection from external runtime reconstruction."""

    def __init__(self) -> None:
        super().__init__("capability_runtime_authority_invalid")


@dataclass(frozen=True)
class NormalizedCapabilityAuthority:
    runtime_identity: dict[str, Any]
    probe_records: dict[str, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class _NormalizedRuntimeInputs:
    python: Path
    tvm_site: Path
    nvlibs_file: Path
    support_root: Path
    support_root_sha256: str
    dependency_root: Path
    gpu_indices: tuple[int, ...]


def _regular_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not path.is_symlink()


def _validated_context_path(path: Path, private_root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        root = private_root.resolve(strict=True)
        info = path.lstat()
    except OSError as error:
        raise P6CapabilityRuntimeAuthorityError() from error
    if (
        resolved != path
        or resolved.parent != root / "inputs"
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or path.is_symlink()
    ):
        raise P6CapabilityRuntimeAuthorityError()
    return resolved


def _runtime_value(values: Mapping[str, Any], key: str, kind: str) -> str:
    entry = values.get(key)
    if (
        not isinstance(entry, Mapping)
        or set(entry) != {"kind", "value"}
        or entry.get("kind") != kind
        or not isinstance(entry.get("value"), str)
        or not entry["value"]
    ):
        raise P6CapabilityRuntimeAuthorityError()
    return str(entry["value"])


def _runtime_declarations(
    *, profile: Any, runner: Any, expected_gpu_indices: tuple[int, ...] | None
) -> tuple[dict[str, str], tuple[int, ...]]:
    values = runner.execution_interface["environment"]["values"]
    if not isinstance(values, Mapping) or set(values) != set(
        (*EXPECTED_HISTORY_ENV_KEYS, *FORMAL_TVM_ENV_KEYS)
    ):
        raise ValueError
    declared = {
        key: _runtime_value(values, key, FORMAL_TVM_ENVIRONMENT_SPEC[key][0])
        for key in FORMAL_TVM_ENV_KEYS
    }
    indices = parse_gpu_indices_csv(
        _runtime_value(values, "CUDA_VISIBLE_DEVICES", "literal")
    )
    indices = validate_profile_gpu_policy(profile.hardware_profile, indices)
    if expected_gpu_indices is not None and indices != expected_gpu_indices:
        raise ValueError
    return declared, indices


def _normalized_runtime_inputs(
    *, private_root: Path, expected_gpu_indices: tuple[int, ...] | None
) -> _NormalizedRuntimeInputs:
    try:
        profile = load_post_source_adapter_profile(
            private_root / "post-source-adapter-profile.yaml",
            private_root=private_root,
        )
        require_post_source_adapter_profile_v4(profile)
        runner = validate_pre_provision_runner_template(
            private_root / "runner-template.yaml",
            private_root,
            require_exact_history_environment=True,
        )
        declared, indices = _runtime_declarations(
            profile=profile,
            runner=runner,
            expected_gpu_indices=expected_gpu_indices,
        )
        python = validate_adapter_python(declared["P6_TVM_PYTHON"])
        tvm_site = Path(declared["P6_TVM_SITE"])
        nvlibs = Path(declared["P6_TVM_NVLIBS_FILE"])
        support = Path(declared["P6_TVM_SUPPORT_ROOT"])
        if (
            profile.hardware_profile.profile_id != "rtx4090"
            or python != profile.adapter_python
            or profile.adapter_dependency_root is None
            or support != profile.tvm_support_root
            or declared["P6_TVM_SUPPORT_ROOT_SHA256"]
            != profile.tvm_support_root_sha256
            or not tvm_site.is_dir()
            or tvm_site.is_symlink()
            or not _regular_file(nvlibs)
        ):
            raise ValueError
        return _NormalizedRuntimeInputs(
            python,
            tvm_site,
            nvlibs,
            support,
            declared["P6_TVM_SUPPORT_ROOT_SHA256"],
            profile.adapter_dependency_root,
            indices,
        )
    except Exception as error:
        if isinstance(error, P6CapabilityRuntimeAuthorityError):
            raise
        raise P6CapabilityRuntimeAuthorityError() from error


def _helper_environment(inputs: _NormalizedRuntimeInputs) -> dict[str, str]:
    try:
        nvlibs = inputs.nvlibs_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise P6CapabilityRuntimeAuthorityError() from error
    library_paths = [
        str(inputs.tvm_site / "nvidia" / "cuda_runtime" / "lib"),
        str(inputs.tvm_site / "tvm" / "lib"),
    ]
    if nvlibs:
        library_paths.append(nvlibs)
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, inputs.gpu_indices)),
        "LD_LIBRARY_PATH": os.pathsep.join(library_paths),
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _run_authority_helper(
    *, inputs: _NormalizedRuntimeInputs, capability_context_path: Path
) -> str:
    helper = REPOSITORY_ROOT / HELPER_RELATIVE_PATH
    if not _regular_file(helper):
        raise P6CapabilityRuntimeAuthorityError()
    argv = (
        str(inputs.python),
        "-I",
        str(helper),
        "--context",
        str(capability_context_path),
        "--tvm-site",
        str(inputs.tvm_site),
        "--nvlibs-file",
        str(inputs.nvlibs_file),
        "--support-root",
        str(inputs.support_root),
        "--support-root-sha256",
        inputs.support_root_sha256,
        "--dependency-root",
        str(inputs.dependency_root),
    )
    try:
        completed = subprocess.run(
            argv,
            cwd=REPOSITORY_ROOT,
            env=_helper_environment(inputs),
            capture_output=True,
            text=True,
            shell=False,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise P6CapabilityRuntimeAuthorityError() from error
    if completed.returncode != 0 or len(completed.stdout.encode("utf-8")) > 4 * 1024 * 1024:
        raise P6CapabilityRuntimeAuthorityError()
    return completed.stdout


def _parse_authority_output(raw: str) -> NormalizedCapabilityAuthority:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "runtime_identity",
            "probe_records",
        }:
            raise ValueError
        runtime = dict(payload["runtime_identity"])
        identity = {key: value for key, value in runtime.items() if key != "compiler_fingerprint"}
        if runtime.get("compiler_fingerprint") != compiler_fingerprint(identity):
            raise ValueError
        raw_records = payload["probe_records"]
        if not isinstance(raw_records, Mapping) or set(raw_records) != {
            "neutral",
            "pruning",
        }:
            raise ValueError
        records = {
            "neutral": validate_probe_partition(
                raw_records["neutral"], probe_ids=NEUTRAL_PROBE_IDS
            ),
            "pruning": validate_probe_partition(
                raw_records["pruning"], probe_ids=PRUNING_PROBE_IDS
            ),
        }
        if payload.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
            raise ValueError
        return NormalizedCapabilityAuthority(runtime, records)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise P6CapabilityRuntimeAuthorityError() from error


def probe_normalized_capability_authority(
    *,
    private_root: Path,
    capability_context_path: Path,
    expected_gpu_indices: tuple[int, ...] | None = None,
) -> NormalizedCapabilityAuthority:
    """Rebuild compiler identity and records outside the context payload."""
    try:
        inputs = _normalized_runtime_inputs(
            private_root=private_root,
            expected_gpu_indices=expected_gpu_indices,
        )
        context = _validated_context_path(capability_context_path, private_root)
        return _parse_authority_output(
            _run_authority_helper(inputs=inputs, capability_context_path=context)
        )
    except Exception as error:
        if isinstance(error, P6CapabilityRuntimeAuthorityError):
            raise
        raise P6CapabilityRuntimeAuthorityError() from error


__all__ = [
    "NormalizedCapabilityAuthority",
    "P6CapabilityRuntimeAuthorityError",
    "probe_normalized_capability_authority",
]
