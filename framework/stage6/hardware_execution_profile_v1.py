"""Immutable, repository-declared hardware profiles for P6 execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from framework.stage6.p6_gpu_policy_v1 import canonical_gpu_indices


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HardwareExecutionProfile:
    """The hardware constraints shared by P6 execution boundaries."""

    profile_id: str
    target_hardware_id: str
    hardware_capability_path: PurePosixPath
    environment_contract_path: PurePosixPath
    allowed_normalized_gpu_models: frozenset[str]
    backend_scope: frozenset[str]
    tvm_arch: str
    tvm_cache_namespace: str
    required_gpu_count: int | None
    maximum_occupancy: float


_PROFILES = MappingProxyType({
    "h800": HardwareExecutionProfile(
        profile_id="h800",
        target_hardware_id="h800",
        hardware_capability_path=PurePosixPath("configs/hardware/h800.yaml"),
        environment_contract_path=PurePosixPath("configs/environment/h800.yaml"),
        allowed_normalized_gpu_models=frozenset(
            {"H800", "NVIDIAH800", "NVIDIAH80080GBHBM3"}
        ),
        backend_scope=frozenset({"tvm_auto"}),
        tvm_arch="sm90",
        tvm_cache_namespace="h800-sm90",
        required_gpu_count=None,
        maximum_occupancy=0.05,
    ),
    "rtx4090": HardwareExecutionProfile(
        profile_id="rtx4090",
        target_hardware_id="rtx4090",
        hardware_capability_path=PurePosixPath("configs/hardware/rtx4090.yaml"),
        environment_contract_path=PurePosixPath("configs/environment/rtx4090.yaml"),
        allowed_normalized_gpu_models=frozenset(
            {"RTX4090", "NVIDIARTX4090", "NVIDIAGEFORCERTX4090"}
        ),
        backend_scope=frozenset({"tvm_auto"}),
        tvm_arch="sm89",
        tvm_cache_namespace="rtx4090-sm89",
        required_gpu_count=4,
        maximum_occupancy=0.05,
    ),
})


def load_hardware_execution_profile(profile_id: str) -> HardwareExecutionProfile:
    """Load one supported profile after checking its declared public inputs."""
    if not isinstance(profile_id, str) or profile_id not in _PROFILES:
        raise ValueError("unknown hardware execution profile")
    profile = _PROFILES[profile_id]
    _validate_declared_yaml_path(profile.hardware_capability_path)
    _validate_declared_yaml_path(profile.environment_contract_path)
    return profile


def default_hardware_execution_profile() -> HardwareExecutionProfile:
    """Return the H800 profile used by legacy P6 contracts."""
    return load_hardware_execution_profile("h800")


def validate_profile_gpu_policy(
    profile: HardwareExecutionProfile, indices: tuple[int, ...]
) -> tuple[int, ...]:
    """Require canonical indices and the selected profile's cardinality."""
    _validate_profile(profile)
    canonical_indices = canonical_gpu_indices(indices)
    if (
        profile.required_gpu_count is not None
        and len(canonical_indices) != profile.required_gpu_count
    ):
        raise ValueError(
            f"{profile.profile_id} requires exactly {profile.required_gpu_count} GPU indices"
        )
    return canonical_indices


def validate_profile_gpu_records(
    profile: HardwareExecutionProfile,
    records: Sequence[object],
    indices: tuple[int, ...],
) -> None:
    """Validate injected GPU model and occupancy records for a profile."""
    canonical_indices = validate_profile_gpu_policy(profile, indices)
    if isinstance(records, (str, bytes)) or len(records) != len(canonical_indices):
        raise ValueError("GPU records do not match the profile policy")
    try:
        by_index = {record.index: record for record in records}  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        raise ValueError("GPU records do not match the profile policy") from None
    if set(by_index) != set(canonical_indices) or len(by_index) != len(records):
        raise ValueError("GPU records do not match the profile policy")
    for index in canonical_indices:
        record = by_index[index]
        if _normalize_model(getattr(record, "model_name", None)) not in profile.allowed_normalized_gpu_models:
            raise ValueError("GPU model does not match the hardware execution profile")
        occupancy = getattr(record, "occupancy", None)
        if (
            isinstance(occupancy, bool)
            or not isinstance(occupancy, (int, float))
            or not math.isfinite(occupancy)
            or occupancy < 0.0
            or occupancy > profile.maximum_occupancy
        ):
            raise ValueError("GPU occupancy does not match the hardware execution profile")


def _validate_profile(profile: HardwareExecutionProfile) -> None:
    if not isinstance(profile, HardwareExecutionProfile):
        raise ValueError("hardware execution profile is required")


def _validate_declared_yaml_path(path: PurePosixPath) -> None:
    if path.is_absolute() or ".." in path.parts or path.suffix != ".yaml":
        raise ValueError("hardware profile YAML path must be repository-relative")
    resolved_path = _REPOSITORY_ROOT.joinpath(*path.parts)
    if not resolved_path.is_file():
        raise ValueError("hardware profile YAML path is missing")


def _normalize_model(model_name: object) -> str:
    if not isinstance(model_name, str):
        return ""
    return "".join(character for character in model_name.upper() if character.isalnum())
