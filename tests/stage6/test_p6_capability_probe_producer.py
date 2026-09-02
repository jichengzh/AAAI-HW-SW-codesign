from __future__ import annotations

from dataclasses import replace

import pytest

from framework.stage6.hardware_execution_profile_v1 import load_hardware_execution_profile
from framework.stage6.p6_capability_probe_producer_v1 import (
    P6CapabilityProbeProducerError,
    validate_live_gpu_snapshots,
)
from framework.stage6.p6_history_binding_v1 import GpuRecord


def _records() -> tuple[GpuRecord, ...]:
    return tuple(
        GpuRecord(
            index=index,
            uuid=f"GPU-test-{index}",
            model_name="NVIDIA GeForce RTX 4090",
            occupancy=0.01,
        )
        for index in range(4)
    )


def test_live_probe_requires_two_stable_ordered_rtx_snapshots() -> None:
    profile = load_hardware_execution_profile("rtx4090")
    assert (
        validate_live_gpu_snapshots(
            profile=profile,
            indices=(0, 1, 2, 3),
            first=_records(),
            second=_records(),
        )
        == 4
    )


@pytest.mark.parametrize("mutation", ["uuid", "order", "model", "occupancy"])
def test_live_probe_rejects_identity_or_admission_drift(mutation: str) -> None:
    profile = load_hardware_execution_profile("rtx4090")
    changed = list(_records())
    if mutation == "uuid":
        changed[0] = replace(changed[0], uuid="GPU-alternate")
    elif mutation == "order":
        changed[0], changed[1] = changed[1], changed[0]
    elif mutation == "model":
        changed[0] = replace(changed[0], model_name="NVIDIA H800 80GB HBM3")
    else:
        changed[0] = replace(changed[0], occupancy=0.06)

    with pytest.raises(P6CapabilityProbeProducerError):
        validate_live_gpu_snapshots(
            profile=profile,
            indices=(0, 1, 2, 3),
            first=_records(),
            second=tuple(changed),
        )
