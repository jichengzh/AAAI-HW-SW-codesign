from __future__ import annotations

import json
from pathlib import Path

from tests.release.p6_post_source_adapter_chain_fixture import (
    build_adapter_measurement_request,
)
from tests.release.test_p6_history_execution_adapters import (
    SYNTHETIC_GPU_INDICES,
    _run_measurement_cli,
    _synthetic_history_binding,
    _write_json,
)


TWO_GPU_INDICES = SYNTHETIC_GPU_INDICES[:2]
TWO_GPU_CSV = ",".join(str(index) for index in TWO_GPU_INDICES)


def _with_two_gpus(binding: dict[str, object]) -> dict[str, object]:
    interface = binding["execution_interface"]
    assert isinstance(interface, dict)
    environment = interface["environment"]
    assert isinstance(environment, dict)
    values = environment["values"]
    assert isinstance(values, dict)
    cuda = values["CUDA_VISIBLE_DEVICES"]
    assert isinstance(cuda, dict)
    two_gpu_interface = {
        **interface,
        "environment": {
            **environment,
            "values": {
                **values,
                "CUDA_VISIBLE_DEVICES": {**cuda, "value": TWO_GPU_CSV},
            },
        },
    }
    two_gpu_policy = {
        "indices": list(TWO_GPU_INDICES),
        "uuid_by_index": {
            str(index): f"GPU-fixture-{index}" for index in TWO_GPU_INDICES
        },
        "model": "h800",
        "maximum_occupancy": 0.05,
    }
    return {
        **binding,
        "execution_interface": two_gpu_interface,
        "gpu_policy": two_gpu_policy,
    }


def _write_two_gpu_probe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = " ".join(
        f"'{index}, GPU-fixture-{index}, NVIDIA H800 80GB HBM3, 0, 100'"
        for index in TWO_GPU_INDICES
    )
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' {records}\n", encoding="utf-8")
    path.chmod(0o700)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _run_two_gpu_chain(tmp_path: Path) -> tuple[Path, Path]:
    binding = _with_two_gpus(
        _synthetic_history_binding(tmp_path, adapter_chain=True)
    )
    private_root = Path(str(binding["private_root"]))
    controller_root = private_root / "controller-round"
    controller_root.mkdir()
    binding_path = _write_json(controller_root / "binding.json", binding)
    request = build_adapter_measurement_request(private_root)
    request_path = _write_json(controller_root / "request.json", request)
    feedback_path = controller_root / "feedback.json"
    fake_bin = tmp_path / "fake-bin"
    _write_two_gpu_probe(fake_bin / "nvidia-smi")

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


def _assert_source_and_quantization(round_root: Path) -> None:
    stages = _read_jsonl(round_root / "executed-stages.log")
    sources = [row for row in stages if row["stage"] == "stage5_materialize_round_sources_v1.sh"]
    assert [_option(row["argv"], "--gpu") for row in sources] == [
        str(TWO_GPU_INDICES[index % len(TWO_GPU_INDICES)]) for index in range(4)
    ]
    quant = _read_jsonl(round_root / "quant-leaf.log")
    assert [row["cuda"] for row in quant] == [str(index) for index in TWO_GPU_INDICES]


def _assert_performance_and_ap(round_root: Path) -> None:
    performance = _read_jsonl(round_root / "performance-leaves.log")
    assert all(_option(row["argv"], "--gpus") == TWO_GPU_CSV for row in performance)
    execute = next(row for row in performance if row["leaf"] == "execute")
    assert _option(execute["argv"], "--max-workers") == "2"
    assert sorted(path.name for path in (round_root / "ap").glob("ap_plan_shard_*.jsonl")) == [
        "ap_plan_shard_0.jsonl",
        "ap_plan_shard_1.jsonl",
    ]
    ap = _read_jsonl(round_root / "ap-leaves.log")
    ap_execute = [row for row in ap if row["leaf"] == "execute"]
    assert sorted((row["cuda"], _option(row["argv"], "--stage")) for row in ap_execute) == [
        (str(index), stage)
        for index in TWO_GPU_INDICES
        for stage in ("full", "sanity")
    ]


def _assert_finalization(round_root: Path, feedback_path: Path) -> None:
    finalization = _read_jsonl(round_root / "finalization-leaves.log")
    assert [row["leaf"] for row in finalization] == [
        "finalize",
        "promote",
    ]
    assert [row["cuda"] for row in finalization] == [TWO_GPU_CSV] * 2
    assert json.loads(feedback_path.read_text(encoding="utf-8"))["rows"]


def test_two_gpu_policy_completes_all_post_source_adapters(tmp_path: Path) -> None:
    """One two-card batch reaches finalization with two-worker/two-shard fan-out."""
    round_root, feedback_path = _run_two_gpu_chain(tmp_path)

    _assert_source_and_quantization(round_root)
    _assert_performance_and_ap(round_root)
    _assert_finalization(round_root, feedback_path)
