"""Contract tests for the explicit runtime GPU pool."""

from __future__ import annotations

import pytest

from framework.stage6.p6_gpu_policy_v1 import parse_runtime_gpu_pool


@pytest.mark.parametrize("gpu_count", [1, 3, 7])
def test_runtime_gpu_pool_reads_a_strict_positive_count(gpu_count: int) -> None:
    environment = {
        "GPU_POOL": str(gpu_count),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    }

    assert parse_runtime_gpu_pool(environment) == gpu_count


def test_runtime_gpu_pool_never_falls_back_to_cuda_visible_devices() -> None:
    with pytest.raises(ValueError, match="GPU_POOL"):
        parse_runtime_gpu_pool({"CUDA_VISIBLE_DEVICES": "7"})


@pytest.mark.parametrize(
    "environment",
    [
        None,
        {},
        {"GPU_POOL": ""},
        {"GPU_POOL": "0"},
        {"GPU_POOL": "-1"},
        {"GPU_POOL": "01"},
        {"GPU_POOL": " 7"},
        {"GPU_POOL": "7 "},
        {"GPU_POOL": "7,2"},
        {"GPU_POOL": 7},
    ],
)
def test_runtime_gpu_pool_rejects_noncanonical_inputs(environment: object) -> None:
    with pytest.raises(ValueError, match="GPU_POOL"):
        parse_runtime_gpu_pool(environment)  # type: ignore[arg-type]
