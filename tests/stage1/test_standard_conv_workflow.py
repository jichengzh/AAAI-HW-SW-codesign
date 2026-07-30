"""End-to-end static Stage1 standard-convolution census and planning contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from framework.stage1.standard_conv_census import analyze_manifest, build_report, write_markdown as write_census_markdown
from framework.stage1.standard_conv_probe_plan import build_probe_plan, write_markdown as write_plan_markdown


def _manifest() -> dict:
    return {
        "model": "fcooper",
        "scan_status": "ok",
        "trace": {"entry_shape": [1, 64, 8, 8], "skipped_modules": ["attention.fusion"]},
        "view_b1_prune_groups": [
            {
                "group_id": "backbone.s0",
                "root_layer": "backbone.blocks.0.conv",
                "member_layers": ["backbone.blocks.0.conv", "neck.up"],
                "grouped_conv_g": 1,
            },
            {
                "group_id": "unknown.group",
                "root_layer": "fusion.unknown",
                "member_layers": ["fusion.unknown"],
            },
        ],
        "view_b1_search_groups": [
            {
                "search_group_id": "backbone.s0",
                "bucket": "backbone",
                "member_b1_groups": ["backbone.s0"],
                "widths": [32, 64],
                "round_to": 16,
                "int8_buildable_align": 32,
            },
            {
                "search_group_id": "unknown",
                "bucket": "fusion",
                "member_b1_groups": ["missing"],
                "widths": [17],
                "round_to": 1,
                "int8_buildable_align": 1,
            },
        ],
        "view_d_routing": [
            {"node": "backbone.blocks.0.conv", "op_type": "Conv2d"},
            {"node": "neck.up", "op_type": "ConvTranspose2d"},
        ],
    }


def test_census_selects_risky_standard_conv_candidates_and_writes_reports(tmp_path: Path) -> None:
    """A real manifest drives static candidate selection without executing a model."""
    manifest_path = tmp_path / "fcooper.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest()), encoding="utf-8")

    model = analyze_manifest(manifest_path)
    report = build_report([manifest_path, tmp_path / "missing.yaml"])
    markdown_path = tmp_path / "census.md"
    write_census_markdown(report, markdown_path)

    standard, unknown = model["candidates"]
    assert model["n_candidates"] == 2
    assert standard["is_standard_conv_candidate"] is True
    assert standard["contains_convtranspose"] is True
    assert standard["anchor_priority"] == "high"
    assert "attention_or_transformer" in standard["skipped_subgraph_types"]
    assert unknown["is_standard_conv_candidate"] is False
    assert unknown["missing_member_ids"] == ["missing"]
    assert report["summary"] == {
        "models": 1,
        "search_groups": 2,
        "standard_conv_candidates": 1,
        "high_priority_anchors": 1,
        "medium_priority_anchors": 0,
    }
    assert "Standard Conv Coupling Census" in markdown_path.read_text(encoding="utf-8")


def test_probe_plan_binds_census_queue_to_safe_runner_policy(tmp_path: Path) -> None:
    """The generated plan keeps the no-enumeration and no-hardware boundaries explicit."""
    manifest_path = tmp_path / "fcooper.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest()), encoding="utf-8")
    census = build_report([manifest_path])

    plan = build_probe_plan(census)
    markdown_path = tmp_path / "probe-plan.md"
    write_plan_markdown(plan, markdown_path)

    assert plan["schema"] == "standard_conv_probe_queue_v1"
    assert len(plan["probes"]) == len(census["recommended_probe_queue"])
    assert "CUDA_import_or_execution" in plan["policy"]["forbidden"]
    assert plan["probes"][0]["enumeration_policy"]
    assert "groups=1 cannot directly classify" in plan["policy"]["groups_1_verdict_boundary"]
    assert "Standard Conv Probe Queue" in markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "census, message",
    [
        ({}, "recommended_probe_queue"),
        ({"recommended_probe_queue": [{"probe_id": "unknown", "priority": "low"}]}, "no S2 template"),
    ],
)
def test_probe_plan_rejects_invalid_or_unknown_census_queue(census: dict, message: str) -> None:
    """A corrupt census cannot silently produce a measurement plan."""
    with pytest.raises(ValueError, match=message):
        build_probe_plan(census)
