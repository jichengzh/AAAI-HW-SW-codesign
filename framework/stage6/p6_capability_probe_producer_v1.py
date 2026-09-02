"""Fail-closed orchestration helpers for the one real RTX capability probe."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Any, Mapping

from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    validate_profile_gpu_policy,
    validate_profile_gpu_records,
)
from framework.stage6.p6_capability_probe_worker_v1 import ProbeFamilyResult


class P6CapabilityProbeProducerError(ValueError):
    """Stable path-free producer rejection."""

    def __init__(self) -> None:
        super().__init__("capability_probe_invalid")


def _identity(records: Sequence[object]) -> tuple[tuple[int, str, str], ...]:
    if any(
        isinstance(getattr(record, "index", None), bool)
        or not isinstance(getattr(record, "index", None), int)
        or not isinstance(getattr(record, "uuid", None), str)
        or not getattr(record, "uuid", "")
        or not isinstance(getattr(record, "model_name", None), str)
        or not getattr(record, "model_name", "")
        for record in records
    ):
        raise P6CapabilityProbeProducerError()
    return tuple(
        (
            int(getattr(record, "index")),
            str(getattr(record, "uuid")),
            str(getattr(record, "model_name")),
        )
        for record in records
    )


def validate_live_gpu_snapshots(
    *,
    profile: HardwareExecutionProfile,
    indices: tuple[int, ...],
    first: Sequence[object],
    second: Sequence[object],
) -> int:
    """Require two admitted snapshots with stable order and immutable identity."""
    try:
        canonical = validate_profile_gpu_policy(profile, indices)
        validate_profile_gpu_records(profile, first, canonical)
        validate_profile_gpu_records(profile, second, canonical)
    except ValueError as error:
        raise P6CapabilityProbeProducerError() from error
    if (
        profile.profile_id != "rtx4090"
        or tuple(record.index for record in first) != canonical
        or tuple(record.index for record in second) != canonical
        or _identity(first) != _identity(second)
    ):
        raise P6CapabilityProbeProducerError()
    return len(canonical)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_probe_evidence(
    *,
    profile: HardwareExecutionProfile,
    repository_root: Path,
    verified_gpu_count: int,
    runtime_identity: Mapping[str, Any],
    probe_code_sha256: str,
    neutral: ProbeFamilyResult,
    pruning: ProbeFamilyResult,
) -> dict[str, Any]:
    """Assemble the exact path-free evidence consumed by the context validator."""
    try:
        hardware = repository_root.joinpath(*profile.hardware_capability_path.parts)
        environment = repository_root.joinpath(*profile.environment_contract_path.parts)
        if profile.profile_id != "rtx4090" or verified_gpu_count != profile.required_gpu_count:
            raise ValueError
        return {
            "schema_version": "p6_rtx_compiler_capability_evidence_v1",
            "hardware_profile": profile.profile_id,
            "hardware_target": profile.target_hardware_id,
            "dispatch_key": "tvm_auto",
            "tvm_arch": profile.tvm_arch,
            "verified_gpu_count": verified_gpu_count,
            "hardware_capability_sha256": _sha_file(hardware),
            "environment_contract_sha256": _sha_file(environment),
            "runtime_identity": dict(runtime_identity),
            "probe_code_sha256": probe_code_sha256,
            "neutral_probe_manifest_sha256": neutral.manifest_sha256,
            "pruning_probe_manifest_sha256": pruning.manifest_sha256,
            "neutral_records": list(neutral.records),
            "pruning_records": list(pruning.records),
            "artifact_blobs": [*neutral.artifact_blobs, *pruning.artifact_blobs],
        }
    except (OSError, TypeError, ValueError) as error:
        raise P6CapabilityProbeProducerError() from error


__all__ = [
    "P6CapabilityProbeProducerError",
    "build_probe_evidence",
    "validate_live_gpu_snapshots",
]
