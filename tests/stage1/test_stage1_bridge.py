"""Contract tests for the Stage1-to-Stage2 search-space boundary."""

from __future__ import annotations

from pathlib import Path
from math import prod

import pytest
import yaml

from framework.stage1.graph_scan import scan
from framework.stage1_bridge import SpaceSpec, load_stage2_search_space
from tests.stage1.test_trace_graph_adapter_workflow import _ToyAdapter, _hardware


def _manifest(*, int8_buildable_align: int = 64) -> dict:
    return {
        "schema": "stage1_partition_manifest_demo_v1",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {"name": "h800_tvm_demo"},
        "view_b1_search_groups": [
            {
                "search_group_id": "pyramid_group",
                "bucket": "pyramid_backbone",
                "widths": [16, 32, 48, 64],
                "round_to": 16,
                "int8_buildable_align": int8_buildable_align,
                "max_rate": 0.75,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": ["pyramid_group"],
            }
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": ["pyramid_group"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _paper_manifest(model: str, axes: list[list[int]]) -> dict:
    groups = []
    units = []
    for index, widths in enumerate(axes):
        axis_id = (
            f"backbone.s{index}"
            if model != "fcooper" or index < 3
            else ("neck.deblock" if index == 3 else "neck.output")
        )
        group_id = f"{model}.{axis_id}"
        dense_stage = (
            f"stage{index + 1}"
            if model in {"pyramid_lidar", "codriving"} and index < 3
            else axis_id
        )
        groups.append(
            {
                "search_group_id": group_id,
                "bucket": axis_id.split(".", 1)[0],
                "widths": [max(widths)],
                "canonical_base_width": max(widths),
                "round_to": widths[1] - widths[0] if len(widths) > 1 else widths[0],
                "int8_buildable_align": widths[1] - widths[0]
                if len(widths) > 1
                else widths[0],
                "max_rate": round(1.0 - min(widths) / max(widths), 6),
                "grouped_conv": model == "pyramid_lidar",
                "criterion_pool": ["L1"],
                "member_b1_groups": [group_id],
                "formal_axis": {
                    "axis_id": axis_id,
                    "dense_stage": dense_stage,
                    "base_width": max(widths),
                    "legal_widths": widths,
                    "provenance": "paper_fixture_scanned_axis",
                },
            }
        )
        units.append(
            {
                "unit": axis_id,
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": [group_id],
            }
        )
    return {
        "schema": "stage1_partition_manifest_v1",
        "model": model,
        "scan_status": "ok",
        "hw_capability": {
            "name": "h800",
            "legal_bits": ["FP16", "INT8"],
            "int8_align": 32,
            "fp16_align": 8,
        },
        "view_b1_search_groups": groups,
        "view_b2_quant_units": units,
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def test_load_stage2_search_space_rejects_missing_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest path"):
        load_stage2_search_space(tmp_path / "missing.yaml")


def test_load_stage2_search_space_rejects_nonpositive_int8_alignment(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "invalid.yaml", _manifest(int8_buildable_align=0))

    with pytest.raises(ValueError, match="int8_buildable_align"):
        load_stage2_search_space(manifest_path)


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("invalid-syntax.yaml", "model: [\n", "invalid manifest YAML"),
        ("not-an-object.yaml", "- item\n", "manifest must decode to an object"),
    ],
)
def test_load_stage2_search_space_validates_yaml_boundary(
    tmp_path: Path, filename: str, contents: str, message: str
) -> None:
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"scan_status": "failed"}), "scan_status"),
        (lambda data: data.update({"hw_capability": []}), "hw_capability"),
        (lambda data: data.update({"view_b1_search_groups": {}}), "search sections"),
        (lambda data: data["view_b1_search_groups"][0].update({"round_to": True}), "round_to"),
        (lambda data: data["view_b1_search_groups"][0].update({"widths": [15]}), "divisible"),
        (lambda data: data["view_b1_search_groups"][0].update({"max_rate": 1.0}), "max_rate"),
    ],
)
def test_stage1_manifest_validation_rejects_invalid_search_contracts(
    tmp_path: Path, mutate: object, message: str
) -> None:
    payload = _manifest()
    mutate(payload)  # type: ignore[operator]
    manifest_path = _write_manifest(tmp_path / "invalid-contract.yaml", payload)

    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(manifest_path)


def test_illegal_int8_widths_remain_diagnostic_only(tmp_path: Path) -> None:
    search_space = load_stage2_search_space(_write_manifest(tmp_path / "valid.yaml", _manifest()))

    points = search_space["software_candidates"][0]["software_points"]
    assert any(point["quant_policy"] == "int8" and not point["buildable"] for point in points)
    assert all(
        point["status"] == "diagnostic_only"
        for point in points
        if point["quant_policy"] == "int8" and not point["buildable"]
    )


def test_legacy_manifest_without_int8_alignment_remains_deterministic(tmp_path: Path) -> None:
    payload = _manifest()
    del payload["view_b1_search_groups"][0]["int8_buildable_align"]
    manifest_path = _write_manifest(tmp_path / "legacy.yaml", payload)

    first = load_stage2_search_space(manifest_path)
    second = load_stage2_search_space(manifest_path)
    spec = SpaceSpec.from_manifest(manifest_path)

    assert first == second
    assert first["model_search_policy"]["selected"] == "serial"
    assert spec.coupling_summary()["n_serial"] == 1


def test_stage2_loader_accepts_schema_tagged_real_scan_manifest(tmp_path: Path) -> None:
    manifest = scan(_ToyAdapter(), _hardware(), device="cpu", profile_latency_mode="off")
    path = _write_manifest(tmp_path / "partition.yaml", manifest)

    assert load_stage2_search_space(path)["schema"] == "stage2_search_space_v1"


@pytest.mark.parametrize(
    ("model", "axes", "structure_count"),
    [
        (
            "pyramid_lidar",
            [
                [16, 24, 32, 40, 48, 56, 64],
                [32, 48, 64, 80, 96, 112, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            343,
        ),
        (
            "codriving",
            [
                [16, 24, 32, 40, 48, 56, 64],
                [32, 48, 64, 80, 96, 112, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            343,
        ),
        (
            "fcooper",
            [
                [32, 64],
                [32, 64, 96, 128],
                [32, 64, 96, 128, 160, 192, 224, 256],
                [32, 64, 96, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            1792,
        ),
    ],
)
def test_stage2_loader_exposes_scan_derived_paper_formal_axes(
    tmp_path: Path,
    model: str,
    axes: list[list[int]],
    structure_count: int,
) -> None:
    """Catches diagnostic anchors replacing the paper's formal scan-derived space."""
    search_space = load_stage2_search_space(
        _write_manifest(tmp_path / f"{model}.yaml", _paper_manifest(model, axes))
    )

    formal_axes = search_space["structural_axes"]

    assert [axis["legal_widths"] for axis in formal_axes] == axes
    assert [axis["base_width"] for axis in formal_axes] == [max(axis) for axis in axes]
    assert prod(len(axis["legal_widths"]) for axis in formal_axes) == structure_count
    assert search_space["formal_q_modes"] == ["fp16", "int8"]
    assert search_space["formal_candidate_policy"] == {
        "enumeration": "structural_axes_x_formal_q_modes",
        "diagnostic_anchors_drive_formal_space": False,
    }
