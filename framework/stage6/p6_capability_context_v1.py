"""Measured RTX capability context kept separate from historical Gold evidence."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from framework.stage2.canonical_search_v3 import (
    build_capability_profile,
    validate_capability_profile,
)
from framework.stage6.hardware_execution_profile_v1 import HardwareExecutionProfile
from framework.stage6.p6_capability_artifacts_v1 import (
    P6CapabilityArtifactError,
    validate_probe_artifact_blobs,
)
from framework.stage6.p6_capability_probe_specs_v1 import (
    FEATURE_NAMES,
    aggregate_capability_features,
)


CONTEXT_SCHEMA_VERSION = "p6_rtx_capability_context_v1"
EVIDENCE_SCHEMA_VERSION = "p6_rtx_compiler_capability_evidence_v1"
RUNTIME_SCHEMA_VERSION = "p6_tvm_compiler_runtime_identity_v1"
ACTIVE_PROFILE_ID = "rtx4090-tvm-auto-probe-v1"
HISTORICAL_LEDGER_RELATIVE_PATH = (
    "artifacts/verified/stage4/cost_model_selection_report.json"
)
PROBE_CODE_RELATIVE_PATHS = (
    HISTORICAL_LEDGER_RELATIVE_PATH,
    "framework/stage6/p6_capability_artifacts_v1.py",
    "framework/stage6/p6_capability_context_v1.py",
    "framework/stage6/p6_capability_observation_v1.py",
    "framework/stage6/p6_capability_probe_models_v1.py",
    "framework/stage6/p6_capability_probe_producer_v1.py",
    "framework/stage6/p6_capability_probe_specs_v1.py",
    "framework/stage6/p6_capability_probe_worker_v1.py",
    "framework/stage6/p6_capability_runtime_authority_v1.py",
    "framework/stage6/p6_capability_tvm_v1.py",
    "framework/stage6/p6_tvm_runtime_authority_v1.py",
    "tools/release/probe_p6_rtx_capability.py",
    "tools/release/rebuild_p6_rtx_capability_authority.py",
)
_HISTORICAL_DISPATCH_KEYS = frozenset({"tvm_auto", "trt_engine"})
_METRIC_KEYS = frozenset({"latency_ms", "energy_j", "ap30", "ap50", "ap70"})
_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TVM_ARCH_PATTERN = re.compile(r"^sm([0-9]{2})$")
_CONTEXT_KEYS = frozenset(
    {
        "schema_version",
        "historical_source_base64",
        "historical_profiles",
        "active_profile",
        "measurement_evidence",
        "context_digest",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "hardware_profile",
        "hardware_target",
        "dispatch_key",
        "tvm_arch",
        "verified_gpu_count",
        "hardware_capability_sha256",
        "environment_contract_sha256",
        "runtime_identity",
        "probe_code_sha256",
        "neutral_probe_manifest_sha256",
        "pruning_probe_manifest_sha256",
        "neutral_records",
        "pruning_records",
        "artifact_blobs",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "tvm_version",
        "python_version",
        "python_executable_sha256",
        "target",
        "tvm_arch",
        "cuda_compute_version",
        "support_root_sha256",
        "tvm_site_sha256",
        "nvlibs_file_sha256",
        "compiler_file_sha256",
        "compiler_fingerprint",
    }
)
_RUNTIME_IDENTITY_KEYS = _RUNTIME_KEYS - {"compiler_fingerprint"}


class P6CapabilityContextError(ValueError):
    """Stable path-free failure for invalid measured capability context."""

    def __init__(self) -> None:
        super().__init__("capability_context_invalid")


@dataclass(frozen=True)
class ValidatedP6CapabilityContext:
    historical_source_base64: str
    historical_profiles: tuple[dict[str, Any], ...]
    active_profile: dict[str, Any]
    evidence: dict[str, Any]
    context_digest: str


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_capability_source_sha256(repository_root: Path) -> str:
    """Read the reviewed Gold capability-source digest from its tracked ledger."""
    try:
        root = repository_root.resolve(strict=True)
        path = root.joinpath(*Path(HISTORICAL_LEDGER_RELATIVE_PATH).parts)
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError
        report = json.loads(path.read_text(encoding="utf-8"))
        digest = report["source_data_content_hashes"]["capability_profiles_sha256"]
        if report.get("schema_version") != "stage4_cost_model_selection_v1" or not _is_sha(
            digest
        ):
            raise ValueError
        return digest
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise P6CapabilityContextError() from error


def canonical_probe_code_sha256(repository_root: Path) -> str:
    """Rebuild the tracked producer-code digest using path-independent labels."""
    try:
        root = repository_root.resolve(strict=True)
        rows = []
        for relative in PROBE_CODE_RELATIVE_PATHS:
            path = root.joinpath(*Path(relative).parts)
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
                raise ValueError
            rows.append({"path": relative, "sha256": _file_sha(path)})
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise P6CapabilityContextError() from error
    return _sha(rows)


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA_PATTERN.fullmatch(value) is not None


def compiler_fingerprint(runtime_identity: Mapping[str, Any]) -> str:
    """Hash only canonical observations of the actual compiler/runtime bytes."""
    if not isinstance(runtime_identity, Mapping) or set(runtime_identity) != _RUNTIME_IDENTITY_KEYS:
        raise P6CapabilityContextError()
    if any(
        not _is_sha(runtime_identity.get(key))
        for key in (
            "python_executable_sha256",
            "support_root_sha256",
            "tvm_site_sha256",
            "nvlibs_file_sha256",
        )
    ):
        raise P6CapabilityContextError()
    compiler_files = runtime_identity.get("compiler_file_sha256")
    if (
        not isinstance(compiler_files, list)
        or not compiler_files
        or compiler_files != sorted(set(compiler_files))
        or any(not _is_sha(item) for item in compiler_files)
    ):
        raise P6CapabilityContextError()
    if any(
        not isinstance(runtime_identity.get(key), str)
        or not runtime_identity[key]
        or "/" in runtime_identity[key]
        for key in (
            "tvm_version",
            "python_version",
            "target",
            "tvm_arch",
            "cuda_compute_version",
        )
    ):
        raise P6CapabilityContextError()
    return _sha(dict(runtime_identity))


def canonical_cuda_target(tvm_arch: str) -> str:
    """Derive the one normalized CUDA target admitted by a registry architecture."""
    match = _TVM_ARCH_PATTERN.fullmatch(tvm_arch) if isinstance(tvm_arch, str) else None
    if match is None:
        raise P6CapabilityContextError()
    return f"cuda -arch=sm_{match.group(1)}"


def _validated_runtime(
    raw: object, *, expected_support_root_sha256: str, profile: HardwareExecutionProfile
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _RUNTIME_KEYS:
        raise P6CapabilityContextError()
    runtime = copy.deepcopy(dict(raw))
    identity = {key: runtime[key] for key in _RUNTIME_IDENTITY_KEYS}
    if (
        runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION
        or runtime.get("tvm_arch") != profile.tvm_arch
        or runtime.get("cuda_compute_version") != _cuda_compute_version(profile.tvm_arch)
        or runtime.get("target") != canonical_cuda_target(profile.tvm_arch)
        or runtime.get("support_root_sha256") != expected_support_root_sha256
        or runtime.get("compiler_fingerprint") != compiler_fingerprint(identity)
    ):
        raise P6CapabilityContextError()
    return runtime


def _cuda_compute_version(tvm_arch: str) -> str:
    match = _TVM_ARCH_PATTERN.fullmatch(tvm_arch) if isinstance(tvm_arch, str) else None
    if match is None:
        raise P6CapabilityContextError()
    digits = match.group(1)
    return f"{digits[0]}.{digits[1]}"


def _declared_input_digests(
    profile: HardwareExecutionProfile, repository_root: Path
) -> tuple[str, str]:
    try:
        root = repository_root.resolve(strict=True)
        hardware = root.joinpath(*profile.hardware_capability_path.parts).resolve(strict=True)
        environment = root.joinpath(*profile.environment_contract_path.parts).resolve(strict=True)
        hardware.relative_to(root)
        environment.relative_to(root)
        if not hardware.is_file() or not environment.is_file():
            raise ValueError
        return _file_sha(hardware), _file_sha(environment)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise P6CapabilityContextError() from error


def _validated_historical_profiles(raw: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise P6CapabilityContextError()
    try:
        profiles = tuple(validate_capability_profile(item) for item in raw)
    except ValueError as error:
        raise P6CapabilityContextError() from error
    if (
        {profile["dispatch_key"] for profile in profiles} != _HISTORICAL_DISPATCH_KEYS
        or any(profile["hardware_target"] != "h800" for profile in profiles)
        or len({profile["capability_profile_id"] for profile in profiles}) != 2
        or any(set(profile["features"]) != set(FEATURE_NAMES) for profile in profiles)
    ):
        raise P6CapabilityContextError()
    return profiles


def _historical_source_bytes(raw: object) -> bytes:
    if not isinstance(raw, str) or not raw:
        raise P6CapabilityContextError()
    try:
        payload = base64.b64decode(raw, validate=True)
    except (TypeError, ValueError) as error:
        raise P6CapabilityContextError() from error
    if not payload or base64.b64encode(payload).decode("ascii") != raw:
        raise P6CapabilityContextError()
    return payload


def _profiles_from_historical_source(
    payload: bytes, *, expected_sha256: str
) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, bytes) or not payload or _file_bytes_sha(payload) != expected_sha256:
        raise P6CapabilityContextError()
    try:
        raw = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise P6CapabilityContextError() from error
    return _validated_historical_profiles(raw)


def _file_bytes_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_evidence_header(
    evidence: Mapping[str, Any],
    *,
    profile: HardwareExecutionProfile,
    repository_root: Path,
    expected_probe_code_sha256: str,
) -> None:
    hardware_sha, environment_sha = _declared_input_digests(profile, repository_root)
    if (
        profile.profile_id != "rtx4090"
        or evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or evidence.get("hardware_profile") != profile.profile_id
        or evidence.get("hardware_target") != profile.target_hardware_id
        or evidence.get("dispatch_key") != "tvm_auto"
        or evidence.get("tvm_arch") != profile.tvm_arch
        or evidence.get("verified_gpu_count") != profile.required_gpu_count
        or evidence.get("hardware_capability_sha256") != hardware_sha
        or evidence.get("environment_contract_sha256") != environment_sha
        or evidence.get("probe_code_sha256") != expected_probe_code_sha256
        or any(
            not _is_sha(evidence.get(key))
            for key in ("neutral_probe_manifest_sha256", "pruning_probe_manifest_sha256")
        )
    ):
        raise P6CapabilityContextError()


def _validated_artifacts(evidence: dict[str, Any]) -> None:
    blobs = evidence["artifact_blobs"]
    if not isinstance(blobs, list):
        raise P6CapabilityContextError()
    try:
        validated = {}
        for family in ("neutral", "pruning"):
            validated[family] = validate_probe_artifact_blobs(
                [
                    item
                    for item in blobs
                    if isinstance(item, Mapping) and item.get("family") == family
                ],
                family=family,
                records=evidence[f"{family}_records"],
            )
    except P6CapabilityArtifactError as error:
        raise P6CapabilityContextError() from error
    neutral, neutral_sha = validated["neutral"]
    pruning, pruning_sha = validated["pruning"]
    if (
        len(blobs) != len(neutral) + len(pruning)
        or evidence["neutral_probe_manifest_sha256"] != neutral_sha
        or evidence["pruning_probe_manifest_sha256"] != pruning_sha
    ):
        raise P6CapabilityContextError()
    evidence["artifact_blobs"] = [*neutral, *pruning]


def _validated_probe_records(
    evidence: Mapping[str, Any], trusted: object
) -> tuple[dict[str, float | int], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if not isinstance(trusted, Mapping) or set(trusted) != {"neutral", "pruning"}:
        raise P6CapabilityContextError()
    try:
        features, neutral, pruning = aggregate_capability_features(
            list(trusted["neutral"]), list(trusted["pruning"])
        )
        _, declared_neutral, declared_pruning = aggregate_capability_features(
            evidence["neutral_records"], evidence["pruning_records"]
        )
    except ValueError as error:
        raise P6CapabilityContextError() from error
    if declared_neutral != neutral or declared_pruning != pruning:
        raise P6CapabilityContextError()
    return features, neutral, pruning


def _validated_evidence(
    raw: object,
    *,
    profile: HardwareExecutionProfile,
    repository_root: Path,
    expected_probe_code_sha256: str,
    expected_support_root_sha256: str,
    trusted_runtime_identity: object,
    trusted_probe_records: object,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    if not isinstance(raw, Mapping) or set(raw) != _EVIDENCE_KEYS:
        raise P6CapabilityContextError()
    evidence = copy.deepcopy(dict(raw))
    _validate_evidence_header(
        evidence,
        profile=profile,
        repository_root=repository_root,
        expected_probe_code_sha256=expected_probe_code_sha256,
    )
    runtime = _validated_runtime(
        evidence["runtime_identity"],
        expected_support_root_sha256=expected_support_root_sha256,
        profile=profile,
    )
    trusted_runtime = _validated_runtime(
        trusted_runtime_identity,
        expected_support_root_sha256=expected_support_root_sha256,
        profile=profile,
    )
    if runtime != trusted_runtime:
        raise P6CapabilityContextError()
    evidence["runtime_identity"] = runtime
    features, neutral, pruning = _validated_probe_records(
        evidence, trusted_probe_records
    )
    evidence["neutral_records"] = list(neutral)
    evidence["pruning_records"] = list(pruning)
    _validated_artifacts(evidence)
    return evidence, features


def _rebuilt_active_profile(
    evidence: Mapping[str, Any],
    features: Mapping[str, float | int],
    feature_order: Sequence[str],
) -> dict[str, Any]:
    runtime = evidence["runtime_identity"]
    return build_capability_profile(
        capability_profile_id=ACTIVE_PROFILE_ID,
        hardware_target=str(evidence["hardware_target"]),
        compiler_fingerprint=str(runtime["compiler_fingerprint"]),
        dispatch_key=str(evidence["dispatch_key"]),
        features={name: features[name] for name in feature_order},
    )


def _context_payload(
    historical_source_base64: str,
    historical_profiles: Sequence[Mapping[str, Any]],
    active_profile: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "historical_source_base64": historical_source_base64,
        "historical_profiles": copy.deepcopy(list(historical_profiles)),
        "active_profile": copy.deepcopy(dict(active_profile)),
        "measurement_evidence": copy.deepcopy(dict(evidence)),
    }


def build_rtx_capability_context(
    *,
    historical_source_bytes: bytes,
    evidence: object,
    profile: HardwareExecutionProfile,
    repository_root: Path,
    expected_probe_code_sha256: str,
    expected_support_root_sha256: str,
    expected_historical_source_sha256: str,
    trusted_runtime_identity: object,
    trusted_probe_records: object,
) -> ValidatedP6CapabilityContext:
    """Build a canonical context without modifying its historical profiles."""
    historical = _profiles_from_historical_source(
        historical_source_bytes, expected_sha256=expected_historical_source_sha256
    )
    historical_base64 = base64.b64encode(historical_source_bytes).decode("ascii")
    measured, features = _validated_evidence(
        evidence,
        profile=profile,
        repository_root=repository_root,
        expected_probe_code_sha256=expected_probe_code_sha256,
        expected_support_root_sha256=expected_support_root_sha256,
        trusted_runtime_identity=trusted_runtime_identity,
        trusted_probe_records=trusted_probe_records,
    )
    active = _rebuilt_active_profile(
        measured, features, tuple(historical[0]["features"])
    )
    payload = _context_payload(historical_base64, historical, active, measured)
    return ValidatedP6CapabilityContext(
        historical_base64, historical, active, measured, _sha(payload)
    )


def capability_context_to_mapping(
    context: ValidatedP6CapabilityContext,
) -> dict[str, Any]:
    if not isinstance(context, ValidatedP6CapabilityContext):
        raise P6CapabilityContextError()
    payload = _context_payload(
        context.historical_source_base64,
        context.historical_profiles,
        context.active_profile,
        context.evidence,
    )
    if _sha(payload) != context.context_digest:
        raise P6CapabilityContextError()
    return {**payload, "context_digest": context.context_digest}


def validate_rtx_capability_context(
    raw: object,
    *,
    profile: HardwareExecutionProfile,
    repository_root: Path,
    expected_probe_code_sha256: str,
    expected_support_root_sha256: str,
    expected_historical_source_sha256: str,
    trusted_runtime_identity: object,
    trusted_probe_records: object,
) -> ValidatedP6CapabilityContext:
    """Independently rebuild the measured profile and every canonical digest."""
    if not isinstance(raw, Mapping) or set(raw) != _CONTEXT_KEYS:
        raise P6CapabilityContextError()
    rebuilt = build_rtx_capability_context(
        historical_source_bytes=_historical_source_bytes(raw["historical_source_base64"]),
        evidence=raw["measurement_evidence"],
        profile=profile,
        repository_root=repository_root,
        expected_probe_code_sha256=expected_probe_code_sha256,
        expected_support_root_sha256=expected_support_root_sha256,
        expected_historical_source_sha256=expected_historical_source_sha256,
        trusted_runtime_identity=trusted_runtime_identity,
        trusted_probe_records=trusted_probe_records,
    )
    if (
        raw.get("schema_version") != CONTEXT_SCHEMA_VERSION
        or raw.get("historical_profiles") != list(rebuilt.historical_profiles)
        or raw.get("active_profile") != rebuilt.active_profile
        or raw.get("context_digest") != rebuilt.context_digest
        or _contains_forbidden_metric(raw)
    ):
        raise P6CapabilityContextError()
    return rebuilt


def _contains_forbidden_metric(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(_METRIC_KEYS & set(value)) or any(
            _contains_forbidden_metric(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_metric(item) for item in value)
    return False


__all__ = [
    "ACTIVE_PROFILE_ID",
    "CONTEXT_SCHEMA_VERSION",
    "P6CapabilityContextError",
    "ValidatedP6CapabilityContext",
    "build_rtx_capability_context",
    "canonical_cuda_target",
    "canonical_probe_code_sha256",
    "capability_context_to_mapping",
    "compiler_fingerprint",
    "historical_capability_source_sha256",
    "validate_rtx_capability_context",
]
