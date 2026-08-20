from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from framework.stage6.p6_stage1_bridge_v1 import (
    P6Stage1BridgeError,
    build_p6_stage1_partition_manifest,
)


def _valid_stage1_manifest() -> dict[str, Any]:
    return {
        "schema": "stage1_partition_manifest_v1",
        "stage": "stage1_partition",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {"name": "h800"},
        "view_b1_search_groups": [
            {
                "search_group_id": "pyramid_group.s0",
                "bucket": "pyramid_backbone",
                "widths": [224],
                "round_to": 32,
                "int8_buildable_align": 128,
                "max_rate": 0.875,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": ["pyramid_group.s0"],
            }
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": ["pyramid_group.s0"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def _hardware_path(tmp_path: Path) -> Path:
    path = tmp_path / "h800.yaml"
    path.write_text("basic:\n  name: h800\n", encoding="utf-8")
    return path


def test_bridge_calls_scanner_and_writes_only_valid_json_manifest(
    tmp_path: Path,
) -> None:
    manifest = _valid_stage1_manifest()
    calls: list[tuple[str, Path, str, Mapping[str, str]]] = []

    def scanner(
        model_name: str,
        hardware_path: Path,
        device: str,
        environment: Mapping[str, str],
    ) -> Mapping[str, Any]:
        calls.append((model_name, hardware_path, device, environment))
        return manifest

    output_path = tmp_path / "manifest.json"
    hardware_path = _hardware_path(tmp_path)

    result = build_p6_stage1_partition_manifest(
        output_path,
        hardware_path,
        "cuda:0",
        {"STAGE1_REPO_ROOT": "/private/stage1"},
        scanner,
    )

    assert result == manifest
    assert result is not manifest
    assert calls == [
        (
            "pyramid_lidar",
            hardware_path,
            "cuda:0",
            {"STAGE1_REPO_ROOT": "/private/stage1"},
        )
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize(
    "invalid_manifest",
    [
        {"scan_status": "partial"},
        {
            **_valid_stage1_manifest(),
            "model": "codriving",
        },
        {
            **_valid_stage1_manifest(),
            "view_b1_search_groups": [],
        },
    ],
)
def test_bridge_rejects_partial_wrong_or_empty_manifest_without_output(
    tmp_path: Path,
    invalid_manifest: Mapping[str, Any],
) -> None:
    output_path = tmp_path / "manifest.json"

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            output_path,
            _hardware_path(tmp_path),
            "cuda",
            {},
            lambda *_args: invalid_manifest,
        )

    assert not output_path.exists()


def test_bridge_rejects_unsafe_paths_before_scanner_runs(tmp_path: Path) -> None:
    calls: list[str] = []

    for output_path in (tmp_path / "missing-parent" / "manifest.json", tmp_path):
        with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
            build_p6_stage1_partition_manifest(
                output_path,
                _hardware_path(tmp_path),
                "cuda",
                {},
                lambda *_args: calls.append("called") or _valid_stage1_manifest(),
            )

    assert calls == []
    assert not (tmp_path / "missing-parent" / "manifest.json").exists()
