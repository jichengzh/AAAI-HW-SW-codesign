"""CPU-only behavioral checks for Stage1 latency sidecar generation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from framework.stage1 import latency_profile


class _Adapter:
    name = "unit"

    def semantic_bucket(self, name: str) -> str:
        return "encoder" if name.startswith("encoder") else "head"

    def typed_skipped_subgraphs(self) -> list[dict[str, str]]:
        return [{"name": "external_attention"}]


class _TwoInputAdd(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


def test_profile_latency_records_real_cpu_leaf_timing_and_sidecar(tmp_path: Path, monkeypatch) -> None:
    """A small real torch graph produces deterministic contract fields on CPU."""
    monkeypatch.setattr(latency_profile, "_RESULTS", tmp_path / "results")
    net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

    report = latency_profile.profile_latency(
        net, torch.ones(2, 4), _Adapter(), device="cpu", warmup=1, measure=2
    )

    assert report["status"] == "estimated_cpu"
    assert report["provenance"]["method"] == "perf_counter_forward_hook"
    assert report["e2e_forward_ms"] >= 0.0
    assert {row["layer"] for row in report["top_layers"]} == {"0", "1", "2"}
    assert report["by_b2_quant_unit"]
    assert report["by_b1_search_knob"]
    assert report["coverage"]["n_skipped_subgraphs"] == 1
    assert "CPU" in report["provenance"]["note"]
    sidecar = Path(report["full_detail_sidecar"])
    assert sidecar.is_file()
    assert '"model": "unit"' in sidecar.read_text(encoding="utf-8")


def test_latency_profile_supports_multi_input_networks_and_write_failure(tmp_path: Path, monkeypatch) -> None:
    """The profiler accepts tuple inputs and returns a diagnostic if sidecar persistence fails."""
    blocked_output = tmp_path / "not-a-directory"
    blocked_output.write_text("blocked", encoding="utf-8")
    monkeypatch.setattr(latency_profile, "_RESULTS", blocked_output)

    report = latency_profile.profile_latency(
        _TwoInputAdd(), (torch.ones(1, 2), torch.ones(1, 2)), _Adapter(), "cpu", measure=1
    )

    assert report["status"] == "estimated_cpu"
    assert report["top_layers"] == []
    assert report["full_detail_sidecar"].startswith("write_fail:")


def test_gpu_snapshot_reports_parseable_output_and_safe_cpu_shortcut(monkeypatch) -> None:
    """GPU probing parses tool output without needing an actual accelerator."""
    class _Result:
        stdout = "Demo GPU, 2, 20\n"

    monkeypatch.setattr(latency_profile.subprocess, "run", lambda *args, **kwargs: _Result())

    assert latency_profile._gpu_snapshot("cpu") == {"device": "cpu"}
    assert latency_profile._gpu_snapshot("cuda:3") == {
        "device": "cuda:3",
        "gpu_index": 3,
        "gpu_name": "Demo GPU",
        "util_pct_at_start": 2,
        "mem_used_mib_at_start": 20,
        "idle_ok": True,
    }
