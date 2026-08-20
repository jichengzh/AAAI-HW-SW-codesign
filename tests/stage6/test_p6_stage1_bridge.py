from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
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


def _non_h800_hardware_path(tmp_path: Path) -> Path:
    path = tmp_path / "a100.yaml"
    path.write_text("basic:\n  name: a100\n", encoding="utf-8")
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
        {
            **_valid_stage1_manifest(),
            "view_b1_search_groups": [
                {
                    "search_group_id": "pyramid_group.s0",
                    "bucket": "pyramid_backbone",
                    "widths": [224],
                    "int8_buildable_align": 128,
                    "max_rate": 0.875,
                    "grouped_conv": True,
                    "criterion_pool": ["L1"],
                    "member_b1_groups": ["pyramid_group.s0"],
                }
            ],
        },
        {
            **_valid_stage1_manifest(),
            "view_b1_search_groups": [
                {
                    "search_group_id": "pyramid_group.s0",
                    "bucket": "pyramid_backbone",
                    "widths": [224],
                    "round_to": 32,
                    "max_rate": 0.875,
                    "grouped_conv": True,
                    "criterion_pool": ["L1"],
                    "member_b1_groups": ["pyramid_group.s0"],
                }
            ],
        },
        {
            **_valid_stage1_manifest(),
            "view_b2_quant_units": [
                {
                    "unit": "pyramid_backbone",
                    "quantizable": True,
                    "member_groups": ["pyramid_group.s0"],
                }
            ],
        },
        {
            **_valid_stage1_manifest(),
            "view_b2_quant_units": [
                {
                    "unit": "pyramid_backbone",
                    "legal_bits": ["FP16", "INT8"],
                    "member_groups": ["pyramid_group.s0"],
                }
            ],
        },
        {
            **_valid_stage1_manifest(),
            "view_d_routing_segments": {"segments": [{"n_nodes": 1}]},
        },
        {
            **_valid_stage1_manifest(),
            "hw_capability": {"name": "a100"},
        },
    ],
)
def test_bridge_rejects_incomplete_or_wrong_manifest_without_output(
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


def test_bridge_rejects_non_h800_hardware_yaml_before_scanner_runs(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    output_path = tmp_path / "manifest.json"

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            output_path,
            _non_h800_hardware_path(tmp_path),
            "cuda",
            {},
            lambda *_args: calls.append("called") or _valid_stage1_manifest(),
        )

    assert calls == []
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


def test_bridge_cleans_temporary_file_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = Path.replace

    def fail_manifest_replace(self: Path, target: os.PathLike[str] | str) -> Path:
        if Path(target) == tmp_path / "manifest.json":
            raise OSError("simulated replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            tmp_path / "manifest.json",
            _hardware_path(tmp_path),
            "cuda",
            {},
            lambda *_args: _valid_stage1_manifest(),
        )

    assert not (tmp_path / "manifest.json").exists()
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_bridge_cleans_temporary_file_when_atomic_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_named_temporary_file = tempfile.NamedTemporaryFile

    class FailingTemporaryFile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._handle = original_named_temporary_file(*args, **kwargs)
            self.name = self._handle.name

        def __enter__(self) -> "FailingTemporaryFile":
            self._handle.__enter__()
            Path(self.name).write_text("partial", encoding="utf-8")
            return self

        def __exit__(self, *args: object) -> object:
            return self._handle.__exit__(*args)

        def write(self, value: str) -> int:
            del value
            raise OSError("simulated write failure")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", FailingTemporaryFile)

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            tmp_path / "manifest.json",
            _hardware_path(tmp_path),
            "cuda",
            {},
            lambda *_args: _valid_stage1_manifest(),
        )

    assert not (tmp_path / "manifest.json").exists()
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []
