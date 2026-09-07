from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from framework.stage6.hardware_execution_profile_v1 import load_hardware_execution_profile
from framework.stage6.p6_capability_probe_producer_v1 import (
    P6CapabilityProbeProducerError,
    build_probe_evidence,
    validate_live_gpu_snapshots,
)
from framework.stage6.p6_capability_probe_worker_v1 import ProbeFamilyResult
from framework.stage6.p6_history_binding_v1 import GpuRecord


def _records(indices: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[GpuRecord, ...]:
    return tuple(
        GpuRecord(
            index=index,
            uuid=f"GPU-test-{index}",
            model_name="NVIDIA GeForce RTX 4090",
            occupancy=0.01,
        )
        for index in indices
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


def test_live_probe_accepts_single_gpu_snapshot() -> None:
    profile = load_hardware_execution_profile("rtx4090")

    assert validate_live_gpu_snapshots(
        profile=profile,
        indices=(7,),
        first=_records((7,)),
        second=_records((7,)),
    ) == 1


@pytest.mark.parametrize("verified_gpu_count", [1, 3, 7])
def test_probe_evidence_accepts_any_positive_verified_gpu_count(
    verified_gpu_count: int,
) -> None:
    empty_family = ProbeFamilyResult((), (), "a" * 64)

    evidence = build_probe_evidence(
        profile=load_hardware_execution_profile("rtx4090"),
        repository_root=Path(__file__).resolve().parents[2],
        verified_gpu_count=verified_gpu_count,
        runtime_identity={},
        probe_code_sha256="b" * 64,
        neutral=empty_family,
        pruning=empty_family,
    )

    assert evidence["verified_gpu_count"] == verified_gpu_count


@pytest.mark.parametrize("verified_gpu_count", [True, 0, -1])
def test_probe_evidence_rejects_nonpositive_or_boolean_gpu_count(
    verified_gpu_count: object,
) -> None:
    empty_family = ProbeFamilyResult((), (), "a" * 64)

    with pytest.raises(P6CapabilityProbeProducerError):
        build_probe_evidence(
            profile=load_hardware_execution_profile("rtx4090"),
            repository_root=Path(__file__).resolve().parents[2],
            verified_gpu_count=verified_gpu_count,  # type: ignore[arg-type]
            runtime_identity={},
            probe_code_sha256="b" * 64,
            neutral=empty_family,
            pruning=empty_family,
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
