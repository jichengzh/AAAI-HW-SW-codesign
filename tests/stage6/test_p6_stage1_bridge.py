from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
from typing import Any, Mapping

import pytest
from torch import nn

from framework.stage1.graph_scan import extract_quant_units
from framework.stage1.hardware_scan import HwCapability
import framework.stage6.p6_stage1_bridge_v1 as stage1_bridge
from framework.stage1.structural_axis_digest import scanner_structural_axes_digest
from framework.stage6.p6_stage1_bridge_v1 import (
    P6Stage1BridgeError,
    build_p6_stage1_partition_manifest,
)
from framework.stage6.hardware_execution_profile_v1 import (
    load_hardware_execution_profile,
)
from framework.stage1_bridge import load_stage2_search_space
from framework.stage6.pyramid_search_space_adapter_v1 import (
    build_pyramid_candidate_plan,
)
from tests.stage6.pyramid_formal_space_support import (
    scanner_owned_pyramid_stage1_manifest,
)


def _valid_stage1_manifest(*, hardware_name: str = "h800") -> dict[str, Any]:
    manifest = scanner_owned_pyramid_stage1_manifest()
    manifest["hw_capability"]["name"] = hardware_name
    manifest["formal_scan"] = {"status": "derived"}
    return manifest


def _hardware_path(tmp_path: Path, *, hardware_name: str = "h800") -> Path:
    path = tmp_path / "h800.yaml"
    path.write_text(f"basic:\n  name: {hardware_name}\n", encoding="utf-8")
    return path


def _scenario_path(tmp_path: Path) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text("hardware_precisions: [FP16, INT8]\n", encoding="utf-8")
    return path


def _malform_axis_and_resign(manifest: dict[str, Any]) -> None:
    manifest["scanner_structural_axes"][0].pop("provenance")
    manifest["scanner_structural_axes_digest"] = scanner_structural_axes_digest(
        manifest["scanner_structural_axes"]
    )


def _non_h800_hardware_path(tmp_path: Path) -> Path:
    path = tmp_path / "a100.yaml"
    path.write_text("basic:\n  name: a100\n", encoding="utf-8")
    return path


def test_h800_profile_bridge_calls_scanner_and_writes_only_valid_json_manifest(
    tmp_path: Path,
) -> None:
    manifest = _valid_stage1_manifest()
    calls: list[tuple[str, Path, str, Mapping[str, str], Path]] = []

    def scanner(
        model_name: str,
        hardware_path: Path,
        device: str,
        environment: Mapping[str, str],
        scenario_path: Path,
    ) -> Mapping[str, Any]:
        calls.append((model_name, hardware_path, device, environment, scenario_path))
        return manifest

    output_path = tmp_path / "manifest.json"
    hardware_path = _hardware_path(tmp_path)

    result = build_p6_stage1_partition_manifest(
        output_path,
        hardware_path,
        "cuda:0",
        {"STAGE1_REPO_ROOT": "/private/stage1"},
        scanner,
        profile=load_hardware_execution_profile("h800"),
        scenario_path=_scenario_path(tmp_path),
    )

    assert result == manifest
    assert result is not manifest
    assert calls == [
        (
            "pyramid_lidar",
            hardware_path,
            "cuda:0",
            {"STAGE1_REPO_ROOT": "/private/stage1"},
            tmp_path / "scenario.yaml",
        )
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize("hardware_name", ["NVIDIA H800", "  nvidia   h800  "])
def test_bridge_accepts_tracked_h800_vendor_name(
    tmp_path: Path,
    hardware_name: str,
) -> None:
    manifest = _valid_stage1_manifest(hardware_name=hardware_name)
    output_path = tmp_path / "manifest.json"

    result = build_p6_stage1_partition_manifest(
        output_path,
        _hardware_path(tmp_path, hardware_name=hardware_name),
        "cuda:0",
        {},
        lambda *_args: manifest,
        scenario_path=_scenario_path(tmp_path),
    )

    assert result == manifest
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest


def test_vendor_h800_manifest_flows_through_strict_stage2_to_pyramid_plan(
    tmp_path: Path,
) -> None:
    manifest = _valid_stage1_manifest(hardware_name="NVIDIA H800")
    output_path = tmp_path / "manifest.json"

    build_p6_stage1_partition_manifest(
        output_path,
        _hardware_path(tmp_path, hardware_name="NVIDIA H800"),
        "cuda:0",
        {},
        lambda *_args: manifest,
        scenario_path=_scenario_path(tmp_path),
    )

    search_space = load_stage2_search_space(output_path)
    plan = build_pyramid_candidate_plan(search_space)

    assert search_space["hardware_target"]["name"] == "NVIDIA H800"
    assert plan["hardware_target"] == "h800"
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest


def test_bridge_accepts_fp16_locked_nonquantizable_heads_without_member_groups(
    tmp_path: Path,
) -> None:
    manifest = _valid_stage1_manifest()
    manifest["view_b2_quant_units"].append(
        {
            "unit": "heads",
            "quantizable": False,
            "legal_bits": ["FP16"],
            "member_groups": [],
        }
    )
    output_path = tmp_path / "manifest.json"

    result = build_p6_stage1_partition_manifest(
        output_path,
        _hardware_path(tmp_path),
        "cuda:0",
        {},
        lambda *_args: manifest,
        scenario_path=_scenario_path(tmp_path),
    )

    assert result == manifest
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest


def test_bridge_accepts_real_h800_quant_unit_order_from_stage1_extractor(
    tmp_path: Path,
) -> None:
    hardware = HwCapability(
        {
            "basic": {"name": "NVIDIA H800"},
            "ips": {"gpu": {"precisions": ["FP16", "INT8"]}},
            "quant_constraints": {"bit_widths_w": [8, 16]},
        },
        tmp_path / "h800.yaml",
        True,
    )
    network = nn.Module()
    network.add_module("backbone", nn.Linear(8, 8))
    quant_units = extract_quant_units(
        network,
        [{"bucket": "pyramid_backbone", "group_id": "pyramid_group.s0"}],
        hardware,
        SimpleNamespace(semantic_bucket=lambda _name: "pyramid_backbone"),
    )
    assert quant_units[0]["legal_bits"] == ["INT8", "FP16"]

    manifest = _valid_stage1_manifest(hardware_name=hardware.name)
    manifest["view_b2_quant_units"] = quant_units

    result = build_p6_stage1_partition_manifest(
        tmp_path / "manifest.json",
        _hardware_path(tmp_path, hardware_name=hardware.name),
        "cuda:0",
        {},
        lambda *_args: manifest,
        scenario_path=_scenario_path(tmp_path),
    )

    assert result == manifest


@pytest.mark.parametrize(
    "quant_unit",
    [
        {
            "unit": "pyramid_backbone",
            "quantizable": True,
            "legal_bits": ["FP16", "INT8"],
            "member_groups": [],
        },
        {
            "unit": "pyramid_backbone",
            "quantizable": True,
            "legal_bits": ["FP16"],
            "member_groups": ["pyramid_group.s0"],
        },
        {
            "unit": "pyramid_backbone",
            "quantizable": True,
            "legal_bits": ["FP16", "INT8", "BF16"],
            "member_groups": ["pyramid_group.s0"],
        },
        {
            "unit": "pyramid_backbone",
            "quantizable": True,
            "legal_bits": ["FP16", "INT8", "INT8"],
            "member_groups": ["pyramid_group.s0"],
        },
        {
            "unit": "heads",
            "quantizable": False,
            "legal_bits": ["FP16", "INT8"],
            "member_groups": [],
        },
        {
            "unit": "heads",
            "quantizable": False,
            "legal_bits": ["FP16"],
            "member_groups": ["pyramid_group.s0"],
        },
    ],
)
def test_bridge_rejects_malformed_quant_unit_combinations(
    tmp_path: Path,
    quant_unit: Mapping[str, Any],
) -> None:
    manifest = _valid_stage1_manifest()
    manifest["view_b2_quant_units"] = [dict(quant_unit)]

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            tmp_path / "manifest.json",
            _hardware_path(tmp_path),
            "cuda:0",
            {},
            lambda *_args: manifest,
            scenario_path=_scenario_path(tmp_path),
        )


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
        {
            **_valid_stage1_manifest(),
            "hw_capability": {"name": "not-h800-compatible"},
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
            scenario_path=_scenario_path(tmp_path),
        )

    assert not output_path.exists()


def test_bridge_rejects_absolute_path_without_leaking_it(tmp_path: Path) -> None:
    manifest = _valid_stage1_manifest()
    private_path = "/private/model/config.yaml"
    manifest["legacy_provenance"] = {"config": private_path}
    output_path = tmp_path / "manifest.json"

    with pytest.raises(P6Stage1BridgeError) as caught:
        build_p6_stage1_partition_manifest(
            output_path,
            _hardware_path(tmp_path),
            "cuda:0",
            {},
            lambda *_args: manifest,
            scenario_path=_scenario_path(tmp_path),
        )

    assert str(caught.value) == "stage1_scan_invalid"
    assert private_path not in str(caught.value)
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("private_path", "nested"),
    [
        ("/private/model/config.yaml", False),
        ("/private/model/config.yaml", True),
        (r"C:\private\model\config.yaml", False),
        (r"C:\private\model\config.yaml", True),
    ],
    ids=("posix-top", "posix-nested", "windows-top", "windows-nested"),
)
def test_bridge_rejects_absolute_path_mapping_keys_without_leaking_them(
    tmp_path: Path,
    private_path: str,
    nested: bool,
) -> None:
    manifest = _valid_stage1_manifest()
    path_mapping = {private_path: "private provenance"}
    manifest["legacy_provenance"] = (
        {"nested": path_mapping} if nested else path_mapping
    )
    output_path = tmp_path / "manifest.json"

    with pytest.raises(P6Stage1BridgeError) as caught:
        build_p6_stage1_partition_manifest(
            output_path,
            _hardware_path(tmp_path),
            "cuda:0",
            {},
            lambda *_args: manifest,
            scenario_path=_scenario_path(tmp_path),
        )

    assert str(caught.value) == "stage1_scan_invalid"
    assert private_path not in str(caught.value)
    assert not output_path.exists()


def test_bridge_accepts_semantic_selector_mapping_key(tmp_path: Path) -> None:
    manifest = _valid_stage1_manifest()
    manifest["selector_evidence"] = {"model.backbone.width": 12}
    output_path = tmp_path / "manifest.json"

    result = build_p6_stage1_partition_manifest(
        output_path,
        _hardware_path(tmp_path),
        "cuda:0",
        {},
        lambda *_args: manifest,
        scenario_path=_scenario_path(tmp_path),
    )

    assert result == manifest
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest


def test_bridge_rejects_non_string_mapping_key(tmp_path: Path) -> None:
    manifest = _valid_stage1_manifest()
    manifest["legacy_provenance"] = {1: "ambiguous JSON key"}
    output_path = tmp_path / "manifest.json"

    with pytest.raises(P6Stage1BridgeError, match="^stage1_scan_invalid$"):
        build_p6_stage1_partition_manifest(
            output_path,
            _hardware_path(tmp_path),
            "cuda:0",
            {},
            lambda *_args: manifest,
            scenario_path=_scenario_path(tmp_path),
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
            scenario_path=_scenario_path(tmp_path),
        )

    assert calls == []
    assert not output_path.exists()


def test_bridge_rejects_hardware_yaml_with_incidental_h800_substring(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            tmp_path / "manifest.json",
            _hardware_path(tmp_path, hardware_name="not-h800-compatible"),
            "cuda",
            {},
            lambda *_args: calls.append("called") or _valid_stage1_manifest(),
            scenario_path=_scenario_path(tmp_path),
        )

    assert calls == []
    assert not (tmp_path / "manifest.json").exists()


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
                scenario_path=_scenario_path(tmp_path),
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
            scenario_path=_scenario_path(tmp_path),
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
            scenario_path=_scenario_path(tmp_path),
        )

    assert not (tmp_path / "manifest.json").exists()
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.pop("scanner_structural_axes"),
        lambda manifest: manifest.pop("scanner_structural_axes_digest"),
        lambda manifest: manifest.update({"scanner_structural_axes_digest": "0" * 64}),
        lambda manifest: manifest.update({"structural_axes": manifest.pop("scanner_structural_axes")}),
        _malform_axis_and_resign,
    ],
)
def test_bridge_rejects_missing_drifted_or_handwritten_formal_axes(
    tmp_path: Path,
    mutation: Any,
) -> None:
    manifest = _valid_stage1_manifest()
    mutation(manifest)
    output_path = tmp_path / "manifest.json"

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            output_path,
            _hardware_path(tmp_path),
            "cuda:0",
            {},
            lambda *_args: manifest,
            scenario_path=_scenario_path(tmp_path),
        )

    assert not output_path.exists()


def test_bridge_requires_explicit_scenario_before_scanner_runs(tmp_path: Path) -> None:
    calls: list[str] = []

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            tmp_path / "manifest.json",
            _hardware_path(tmp_path),
            "cuda:0",
            {},
            lambda *_args: calls.append("called") or _valid_stage1_manifest(),
            scenario_path=tmp_path / "missing-scenario.yaml",
        )

    assert calls == []


def _real_scan_environment(tmp_path: Path) -> dict[str, str]:
    paths = {
        "STAGE1_REPO_ROOT": tmp_path / "stage1",
        "HEAL_ROOT": tmp_path / "heal",
        "HEAL_CKPT_ROOT": tmp_path / "checkpoints",
    }
    for path in paths.values():
        path.mkdir()
    return {key: str(path) for key, path in paths.items()}


def _install_real_scan_import_fakes(
    monkeypatch: pytest.MonkeyPatch,
    environment: Mapping[str, str],
    scanner_fails: bool,
) -> str:
    stage1_root = Path(environment["STAGE1_REPO_ROOT"])
    scanner_added = "scanner-owned-path-entry"
    adapter = SimpleNamespace(
        formal_scenario_root=stage1_root / "configs" / "stage1"
    )

    class _Capability:
        legal_bits = ["FP16"]

        @classmethod
        def from_yaml(cls, _path: Path) -> "_Capability":
            return cls()

    def scan(
        _adapter,
        _hardware,
        *,
        device: str,
        scenario: object,
        profile_latency_mode: str,
    ):
        del _adapter, _hardware, device, scenario
        assert profile_latency_mode == "off"
        assert {key: os.environ[key] for key in environment} == environment
        assert sys.path[0] == str(stage1_root)
        sys.path.append(scanner_added)
        if scanner_fails:
            raise RuntimeError("private scanner failure")
        return {"scan_status": "ok"}

    modules = {
        "framework.stage1.adapters": SimpleNamespace(
            get_adapter=lambda _name: adapter
        ),
        "framework.stage1.graph_scan": SimpleNamespace(scan=scan),
        "framework.stage1.hardware_scan": SimpleNamespace(HwCapability=_Capability),
        "framework.stage1.formal_scan_evidence": SimpleNamespace(
            load_scan_scenario=lambda _path, **_kwargs: object(),
            validate_scenario_hardware=lambda _scenario, _hardware: None,
        ),
    }
    monkeypatch.setattr(stage1_bridge, "_supplied_stage1_imports", lambda _root: nullcontext())
    monkeypatch.setattr(
        stage1_bridge.importlib, "import_module", lambda name: modules[name]
    )
    return scanner_added


def _invoke_real_scan(
    tmp_path: Path, environment: Mapping[str, str], scanner_fails: bool
) -> None:
    if scanner_fails:
        with pytest.raises(P6Stage1BridgeError) as caught:
            stage1_bridge.run_real_stage1_scan(
                "pyramid_lidar", tmp_path / "hardware.yaml", "cpu", environment,
                tmp_path / "scenario.yaml",
            )
        assert str(caught.value) == "stage1_scan_invalid"
        assert str(tmp_path) not in str(caught.value)
        return
    assert stage1_bridge.run_real_stage1_scan(
        "pyramid_lidar", tmp_path / "hardware.yaml", "cpu", environment,
        tmp_path / "scenario.yaml",
    ) == {"scan_status": "ok"}


@pytest.mark.parametrize("scanner_fails", [False, True], ids=["success", "failure"])
def test_real_stage1_scan_restores_only_its_process_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scanner_fails: bool,
) -> None:
    environment = _real_scan_environment(tmp_path)
    monkeypatch.setenv("STAGE1_REPO_ROOT", "preexisting-stage1")
    monkeypatch.delenv("HEAL_ROOT", raising=False)
    monkeypatch.setenv("HEAL_CKPT_ROOT", "preexisting-checkpoints")
    original_sys_path = list(sys.path)
    sys.path[:] = ["preexisting-first", *original_sys_path]
    before_sys_path = list(sys.path)
    scanner_added = _install_real_scan_import_fakes(
        monkeypatch, environment, scanner_fails
    )

    try:
        _invoke_real_scan(tmp_path, environment, scanner_fails)
        assert os.environ["STAGE1_REPO_ROOT"] == "preexisting-stage1"
        assert "HEAL_ROOT" not in os.environ
        assert os.environ["HEAL_CKPT_ROOT"] == "preexisting-checkpoints"
        assert sys.path == [*before_sys_path, scanner_added]
    finally:
        sys.path[:] = original_sys_path
