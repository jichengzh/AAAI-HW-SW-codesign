"""Contract tests for immutable P6 hardware execution profiles."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import PurePosixPath

import pytest

from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    default_hardware_execution_profile,
    load_hardware_execution_profile,
    validate_profile_gpu_policy,
    validate_profile_gpu_records,
)
from framework.stage6.p6_history_binding_v1 import GpuRecord


def _records(
    indices: tuple[int, ...], *, model_name: str
) -> tuple[GpuRecord, ...]:
    return tuple(
        GpuRecord(
            index=index,
            uuid=f"GPU-fixture-{index}",
            model_name=model_name,
            occupancy=0.0,
        )
        for index in indices
    )


def test_profiles_are_frozen_and_have_only_the_supported_ids() -> None:
    """Catches a mutable or silently extensible profile registry."""
    h800 = load_hardware_execution_profile("h800")
    rtx4090 = load_hardware_execution_profile("rtx4090")

    assert isinstance(h800, HardwareExecutionProfile)
    assert {h800.profile_id, rtx4090.profile_id} == {"h800", "rtx4090"}
    with pytest.raises(FrozenInstanceError):
        h800.target_hardware_id = "rtx4090"  # type: ignore[misc]


def test_default_profile_preserves_h800_and_profile_values() -> None:
    """Catches legacy H800 callers selecting a different target or backend."""
    h800 = default_hardware_execution_profile()
    rtx4090 = load_hardware_execution_profile("rtx4090")

    assert h800.profile_id == "h800"
    assert h800.target_hardware_id == "h800"
    assert h800.hardware_capability_path == PurePosixPath("configs/hardware/h800.yaml")
    assert h800.environment_contract_path == PurePosixPath("configs/environment/h800.yaml")
    assert h800.tvm_arch == "sm90"
    assert h800.backend_scope == frozenset({"tvm_auto"})
    assert h800.required_gpu_count is None
    assert h800.maximum_occupancy == 0.05

    assert rtx4090.target_hardware_id == "rtx4090"
    assert rtx4090.hardware_capability_path == PurePosixPath(
        "configs/hardware/rtx4090.yaml"
    )
    assert rtx4090.environment_contract_path == PurePosixPath(
        "configs/environment/rtx4090.yaml"
    )
    assert rtx4090.tvm_arch == "sm89"
    assert rtx4090.tvm_cache_namespace == "rtx4090-sm89"
    assert rtx4090.backend_scope == frozenset({"tvm_auto"})
    assert rtx4090.required_gpu_count == 4
    assert rtx4090.maximum_occupancy == 0.05


def test_unknown_profile_is_rejected() -> None:
    """Catches callers falling back from an unrecognized profile identifier."""
    with pytest.raises(ValueError, match="unknown hardware execution profile"):
        load_hardware_execution_profile("a100")


@pytest.mark.parametrize("indices", [(0,), (0, 1, 2), (0, 1, 2, 3, 4)])
def test_rtx4090_requires_exactly_four_gpu_indices(indices: tuple[int, ...]) -> None:
    """Catches RTX policies admitting a non-four-card allocation."""
    profile = load_hardware_execution_profile("rtx4090")

    with pytest.raises(ValueError, match="exactly 4"):
        validate_profile_gpu_policy(profile, indices)


def test_h800_accepts_existing_two_card_fixture() -> None:
    """Catches a new cardinality rule breaking legacy nonempty H800 policies."""
    profile = default_hardware_execution_profile()
    indices = (17, 19)
    records = _records(indices, model_name="NVIDIA H800 80GB HBM3")

    assert validate_profile_gpu_policy(profile, indices) == indices
    assert validate_profile_gpu_records(profile, records, indices) is None


def test_profile_gpu_records_reject_model_from_another_profile() -> None:
    """Catches an RTX policy admitting an H800 GPU model."""
    profile = load_hardware_execution_profile("rtx4090")
    indices = (0, 1, 2, 3)
    records = _records(indices, model_name="NVIDIA H800 80GB HBM3")

    with pytest.raises(ValueError, match="profile"):
        validate_profile_gpu_records(profile, records, indices)
