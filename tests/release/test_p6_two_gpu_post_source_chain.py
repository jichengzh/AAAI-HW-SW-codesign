from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.release.p6_post_source_adapter_chain_fixture import (
    build_adapter_measurement_request,
)
from tests.release.test_p6_history_execution_adapters import (
    SYNTHETIC_GPU_INDICES,
    _run_measurement_cli,
    _synthetic_history_binding,
    _write_json,
)


RUNTIME_GPU_POOLS = (
    SYNTHETIC_GPU_INDICES[:1],
    SYNTHETIC_GPU_INDICES[:2],
    SYNTHETIC_GPU_INDICES,
)


def _with_gpu_pool(
    binding: dict[str, object], gpu_indices: tuple[int, ...]
) -> dict[str, object]:
    interface = binding["execution_interface"]
    assert isinstance(interface, dict)
    environment = interface["environment"]
    assert isinstance(environment, dict)
    values = environment["values"]
    assert isinstance(values, dict)
    cuda = values["CUDA_VISIBLE_DEVICES"]
    assert isinstance(cuda, dict)
    gpu_csv = ",".join(str(index) for index in gpu_indices)
    runtime_interface = {
        **interface,
        "environment": {
            **environment,
            "values": {
                **values,
                "CUDA_VISIBLE_DEVICES": {**cuda, "value": gpu_csv},
            },
        },
    }
    runtime_policy = {
        "indices": list(gpu_indices),
        "uuid_by_index": {
            str(index): f"GPU-fixture-{index}" for index in gpu_indices
        },
        "model": "h800",
        "maximum_occupancy": 0.05,
    }
    return {
        **binding,
        "execution_interface": runtime_interface,
        "gpu_policy": runtime_policy,
    }


def _write_gpu_probe(path: Path, gpu_indices: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = " ".join(
        f"'{index}, GPU-fixture-{index}, NVIDIA H800 80GB HBM3, 0, 100'"
        for index in gpu_indices
    )
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' {records}\n", encoding="utf-8")
    path.chmod(0o700)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _run_gpu_pool_chain(
    tmp_path: Path, gpu_indices: tuple[int, ...]
) -> tuple[Path, Path]:
    binding = _with_gpu_pool(
        _synthetic_history_binding(tmp_path, adapter_chain=True), gpu_indices
    )
    private_root = Path(str(binding["private_root"]))
    controller_root = private_root / "controller-round"
    controller_root.mkdir()
    binding_path = _write_json(controller_root / "binding.json", binding)
    request = build_adapter_measurement_request(private_root)
    request_path = _write_json(controller_root / "request.json", request)
    feedback_path = controller_root / "feedback.json"
    fake_bin = tmp_path / "fake-bin"
    _write_gpu_probe(fake_bin / "nvidia-smi", gpu_indices)

    result = _run_measurement_cli(
        binding_path,
        request_path,
        feedback_path,
        controller_root,
        fake_bin,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "measurement_feedback_written\n"
    round_root = private_root / "private-runs/0"
    return round_root, feedback_path


def _assert_source_and_quantization(
    round_root: Path, gpu_indices: tuple[int, ...]
) -> None:
    stages = _read_jsonl(round_root / "executed-stages.log")
    sources = [row for row in stages if row["stage"] == "stage5_materialize_round_sources_v1.sh"]
    sources_by_group = sorted(sources, key=lambda row: _option(row["argv"], "--group-id"))
    assert [_option(row["argv"], "--gpu") for row in sources_by_group] == [
        str(gpu_indices[index % len(gpu_indices)]) for index in range(4)
    ]
    quant = _read_jsonl(round_root / "quant-leaf.log")
    assert [row["cuda"] for row in quant] == [
        str(gpu_indices[index % len(gpu_indices)]) for index in range(len(quant))
    ]


def _assert_performance_and_ap(
    round_root: Path, gpu_indices: tuple[int, ...]
) -> None:
    gpu_csv = ",".join(str(index) for index in gpu_indices)
    performance = _read_jsonl(round_root / "performance-leaves.log")
    assert all(_option(row["argv"], "--gpus") == gpu_csv for row in performance)
    execute = next(row for row in performance if row["leaf"] == "execute")
    assert _option(execute["argv"], "--max-workers") == str(len(gpu_indices))
    assert sorted(path.name for path in (round_root / "ap").glob("ap_plan_shard_*.jsonl")) == [
        f"ap_plan_shard_{index}.jsonl" for index in range(len(gpu_indices))
    ]
    ap = _read_jsonl(round_root / "ap-leaves.log")
    ap_execute = [row for row in ap if row["leaf"] == "execute"]
    assert sorted((row["cuda"], _option(row["argv"], "--stage")) for row in ap_execute) == [
        (str(index), stage)
        for index in gpu_indices
        for stage in ("full", "sanity")
    ]


def _assert_finalization(
    round_root: Path, feedback_path: Path, gpu_indices: tuple[int, ...]
) -> None:
    gpu_csv = ",".join(str(index) for index in gpu_indices)
    finalization = _read_jsonl(round_root / "finalization-leaves.log")
    assert [row["leaf"] for row in finalization] == [
        "finalize",
        "promote",
    ]
    assert [row["cuda"] for row in finalization] == [gpu_csv] * 2
    assert json.loads(feedback_path.read_text(encoding="utf-8"))["rows"]


@pytest.mark.parametrize("gpu_indices", RUNTIME_GPU_POOLS)
def test_runtime_gpu_pool_completes_all_post_source_adapters(
    tmp_path: Path, gpu_indices: tuple[int, ...]
) -> None:
    """One batch reaches finalization with fan-out derived from its runtime pool."""
    round_root, feedback_path = _run_gpu_pool_chain(tmp_path, gpu_indices)

    _assert_source_and_quantization(round_root, gpu_indices)
    _assert_performance_and_ap(round_root, gpu_indices)
    _assert_finalization(round_root, feedback_path, gpu_indices)
