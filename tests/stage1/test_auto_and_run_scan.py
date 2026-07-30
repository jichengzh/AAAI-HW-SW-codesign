"""CPU-only public contracts for automatic trace and scan CLI adapters."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from framework.stage1 import run_scan
from framework.stage1.adapters import TraceAdapter
import framework.stage1.auto_trace as auto_trace
from framework.stage1.auto_trace import (
    AutoTraceAdapter,
    _CoDrivingTraceNet,
    _HeterBaselineTraceNet,
    _infer_heter_bev_shape,
    _infer_heter_model_key,
    get_auto_adapter,
)
from framework.stage1.hardware_scan import HwCapability


class _HeadedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(32, 32, 3, padding=1)
        self.cls_head = nn.Conv2d(32, 2, 1)
        self.reg_head = nn.Conv2d(32, 4, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.backbone(value)
        return self.cls_head(feature), self.reg_head(feature)


class _DictionaryBackbone(nn.Module):
    """Tiny dict-I/O backbone matching the external model wrappers' contract."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"spatial_features_2d": self.conv(inputs["spatial_features"])}


class _CoDrivingFull(nn.Module):
    def __init__(self, *, shrink: bool) -> None:
        super().__init__()
        self.backbone = _DictionaryBackbone()
        self.shrink_flag = shrink
        self.shrink_conv = nn.Conv2d(32, 32, 1)
        self.cls_head = nn.Conv2d(32, 2, 1)
        self.reg_head = nn.Conv2d(32, 4, 1)


class _HeterFull(nn.Module):
    def __init__(
        self, *, use_shrinker: bool, include_direction: bool, use_legacy_shrink: bool = False
    ) -> None:
        super().__init__()
        self.backbone_m2 = _DictionaryBackbone()
        self.shrink_flag = use_legacy_shrink
        if use_shrinker:
            self.shrinker_m2 = nn.Conv2d(32, 32, 1)
        if use_legacy_shrink:
            self.shrink_conv = nn.Conv2d(32, 32, 1)
        self.cls_head = nn.Conv2d(32, 2, 1)
        self.reg_head = nn.Conv2d(32, 4, 1)
        if include_direction:
            self.dir_head = nn.Conv2d(32, 1, 1)


class _NestedHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Conv2d(32, 32, 1)
        self.heads = nn.ModuleList([nn.Conv2d(32, 2, 1), nn.Conv2d(32, 4, 1)])
        self.single_head_branch = nn.Sequential(nn.Conv2d(32, 1, 1), nn.ReLU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.single_head_branch(self.feature(value))


class _SmallScanAdapter(TraceAdapter):
    name = "small"

    def build_trace_net(self, device: str) -> tuple[nn.Module, torch.Tensor]:
        return _HeadedNet().to(device).eval(), torch.ones(1, 32, 8, 8, device=device)

    def ignored_layers(self, net: nn.Module) -> list[nn.Module]:
        return [net.cls_head, net.reg_head]  # type: ignore[attr-defined]


def _hardware() -> HwCapability:
    return HwCapability(
        {"name": "unit", "ips": {"gpu": {"precisions": ["INT8", "FP16"]}}, "alignment": {"int8_channel": 32}},
        "unit.yaml",
        True,
    )


def test_auto_trace_adapter_builds_cpu_input_detects_heads_and_generates_plan() -> None:
    """AutoTraceAdapter preserves the build function's net and exposes audited skip boundaries."""
    adapter = AutoTraceAdapter(
        name="unit",
        model_class="UnitNet",
        config_path="unit.yaml",
        ckpt_path="missing.pt",
        build_fn=lambda _config, _checkpoint, _device: _HeadedNet().eval(),
        bev_shape=(1, 32, 8, 8),
        ckpt_status="missing_architecture_scan_only",
        skipped_desc=["attention_fusion (untraced)"],
    )

    net, value = adapter.build_trace_net("cpu")
    plan = adapter.build_trace_plan()

    assert tuple(value.shape) == (1, 32, 8, 8)
    assert len(adapter.ignored_layers(net)) == 2
    assert plan["ckpt_status"] == "missing_architecture_scan_only"
    assert plan["skipped_subgraphs"][0]["full_model_verdict_blocker"] is True
    assert [list(item.shape) for item in net(value)] == [[1, 2, 8, 8], [1, 4, 8, 8]]


def test_auto_trace_registry_and_model_key_inference_fail_closed_for_unknown_models() -> None:
    """Known adapters keep documented metadata while unknown keys do not silently resolve."""
    assert get_auto_adapter("where2comm").ckpt_status == "missing_architecture_scan_only"
    assert _infer_heter_model_key("configs/lidar.yaml", {"fusion_method": "AttFusion"}) == "attfuse"
    assert _infer_heter_model_key("configs/disconet.yaml") == "disconet"
    with pytest.raises(KeyError, match="unknown model"):
        get_auto_adapter("not-a-model")


def test_auto_trace_head_resolution_and_external_wrappers_preserve_tensor_contracts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Automatic and explicit head selection keep only boundary modules, not their children."""
    explicit = AutoTraceAdapter(
        name="nested",
        model_class="NestedHeads",
        config_path="nested.yaml",
        ckpt_path="",
        build_fn=lambda *_args: _NestedHeads(),
        bev_shape=(1, 32, 4, 4),
        ignored_attr_names=["heads.0", "heads.5", "missing"],
    )
    nested = _NestedHeads()
    assert explicit.ignored_layers(nested) == [nested.heads[0]]
    assert "ignored_attr 'heads.5' not found" in capsys.readouterr().out

    detected = AutoTraceAdapter(
        name="detected",
        model_class="NestedHeads",
        config_path="detected.yaml",
        ckpt_path="",
        build_fn=lambda *_args: _NestedHeads(),
        bev_shape=(1, 32, 4, 4),
    )
    assert detected.ignored_layers(nested) == [nested.single_head_branch]

    input_value = torch.ones(1, 32, 4, 4)
    for shrink in (False, True):
        codriving_outputs = _CoDrivingTraceNet(_CoDrivingFull(shrink=shrink))(input_value)
        assert [tuple(output.shape) for output in codriving_outputs] == [(1, 2, 4, 4), (1, 4, 4, 4)]

    heter_outputs = _HeterBaselineTraceNet(
        _HeterFull(use_shrinker=True, include_direction=True)
    )(input_value)
    assert [tuple(output.shape) for output in heter_outputs] == [(1, 2, 4, 4), (1, 4, 4, 4), (1, 1, 4, 4)]
    no_shrinker_outputs = _HeterBaselineTraceNet(
        _HeterFull(use_shrinker=False, include_direction=False)
    )(input_value)
    assert [tuple(output.shape) for output in no_shrinker_outputs] == [(1, 2, 4, 4), (1, 4, 4, 4)]
    legacy_shrinker_outputs = _HeterBaselineTraceNet(
        _HeterFull(use_shrinker=False, include_direction=False, use_legacy_shrink=True)
    )(input_value)
    assert [tuple(output.shape) for output in legacy_shrinker_outputs] == [(1, 2, 4, 4), (1, 4, 4, 4)]
    with pytest.raises(ValueError, match="backbone_m"):
        _HeterBaselineTraceNet(nn.Module())


def test_heter_bev_shape_inference_uses_lidar_then_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shape inference treats lidar modality metadata and legacy preprocess metadata explicitly."""
    package = types.ModuleType("opencood")
    package.__path__ = []  # type: ignore[attr-defined]
    hypes_package = types.ModuleType("opencood.hypes_yaml")
    hypes_package.__path__ = []  # type: ignore[attr-defined]
    yaml_utils = types.ModuleType("opencood.hypes_yaml.yaml_utils")
    configurations = {
        "lidar.yaml": {
            "heter": {
                "modality_setting": {
                    "m1": {
                        "sensor_type": "lidar",
                        "encoder_args": {
                            "pillar_vfe": {"num_filters": [32, 96]},
                            "lidar_range": [-4, -2, -3, 8, 6, 1],
                            "voxel_size": [0.5, 1.0, 4],
                        },
                    }
                }
            }
        },
        "fallback.yaml": {
            "model": {
                "args": {
                    "encoder_args": {
                        "pillar_vfe": {"num_filters": [48]},
                        "voxel_size": [1.0, 2.0, 4],
                    }
                }
            },
            "preprocess": {
                "args": {
                    "cav_lidar_range": [-10, -10, -3, 10, 10, 1],
                    "voxel_size": [1.0, 2.0, 4],
                }
            },
        },
    }
    yaml_utils.load_yaml = lambda path: configurations[path]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opencood", package)
    monkeypatch.setitem(sys.modules, "opencood.hypes_yaml", hypes_package)
    monkeypatch.setitem(sys.modules, "opencood.hypes_yaml.yaml_utils", yaml_utils)
    monkeypatch.setattr(auto_trace, "_add_path", lambda _path: None)

    assert _infer_heter_bev_shape("lidar.yaml") == (1, 96, 8, 24)
    assert _infer_heter_bev_shape("fallback.yaml") == (1, 48, 10, 20)


def test_heter_builder_records_a_trace_plan_for_present_or_unavailable_shape_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestration layer keeps loader status and shape failures visible in the trace plan."""
    model = _HeterFull(use_shrinker=False, include_direction=False)
    calls: list[dict[str, object]] = []

    class _Detector:
        def detect(self, _model: nn.Module, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {"selected_candidate": {"name": "unit"}, "marker": kwargs["ckpt_status"]}

    class _Synthesizer:
        def synthesize(self, _model: nn.Module, candidate: dict[str, object]) -> nn.Module:
            assert candidate == {"name": "unit"}
            return nn.Identity()

    monkeypatch.setattr(auto_trace, "_load_heter_baseline_full_model", lambda *_args: (model, {}, "unit"))
    monkeypatch.setattr(auto_trace, "TraceBoundaryDetector", _Detector)
    monkeypatch.setattr(auto_trace, "WrapperSynthesizer", _Synthesizer)
    monkeypatch.setattr(auto_trace, "_infer_heter_bev_shape", lambda _path: (1, 32, 4, 4))
    wrapper = auto_trace._build_heter_baseline("unit.yaml", "", "cpu")
    assert wrapper._stage1_trace_plan["marker"] == "missing_architecture_scan_only"  # type: ignore[attr-defined]
    assert calls[-1]["input_shape"] == (1, 32, 4, 4)

    def _raise_shape_error(_path: str) -> tuple[int, ...]:
        raise ValueError("missing metadata")

    monkeypatch.setattr(auto_trace, "_infer_heter_bev_shape", _raise_shape_error)
    wrapper = auto_trace._build_heter_baseline("unit.yaml", "", "cpu")
    assert wrapper._stage1_trace_plan["marker"] == "missing_architecture_scan_only"  # type: ignore[attr-defined]
    assert calls[-1]["input_shape"] is None


def test_heter_loader_keeps_missing_checkpoints_as_explicit_architecture_scan_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The external-model loader does not pretend an absent checkpoint was loaded."""
    package = types.ModuleType("opencood")
    package.__path__ = []  # type: ignore[attr-defined]
    hypes_package = types.ModuleType("opencood.hypes_yaml")
    hypes_package.__path__ = []  # type: ignore[attr-defined]
    yaml_utils = types.ModuleType("opencood.hypes_yaml.yaml_utils")
    yaml_utils.load_yaml = lambda _path: {"model": {"args": {"fusion_method": "fcooper"}}}  # type: ignore[attr-defined]
    models_package = types.ModuleType("opencood.models")
    models_package.__path__ = []  # type: ignore[attr-defined]
    model_module = types.ModuleType("opencood.models.heter_model_baseline")
    model_module.HeterModelBaseline = lambda _args: _HeterFull(  # type: ignore[attr-defined]
        use_shrinker=False, include_direction=False
    )
    monkeypatch.setitem(sys.modules, "opencood", package)
    monkeypatch.setitem(sys.modules, "opencood.hypes_yaml", hypes_package)
    monkeypatch.setitem(sys.modules, "opencood.hypes_yaml.yaml_utils", yaml_utils)
    monkeypatch.setitem(sys.modules, "opencood.models", models_package)
    monkeypatch.setitem(sys.modules, "opencood.models.heter_model_baseline", model_module)
    monkeypatch.setattr(auto_trace, "_use_heal_opencood", lambda: None)
    monkeypatch.setattr(auto_trace.os, "chdir", lambda _path: None)

    loaded, hypes, model_key = auto_trace._load_heter_baseline_full_model(
        "fcooper.yaml", "missing.pt", "cpu"
    )
    assert isinstance(loaded, _HeterFull)
    assert hypes["model"]["args"]["fusion_method"] == "fcooper"
    assert model_key == "fcooper"


def test_opencood_source_switches_clear_cached_modules_and_prioritize_the_selected_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HEAL and V2Xverse import switches cannot reuse a cached package from the other tree."""
    heal_root = Path("/tmp/anonymous-heal")
    v2xverse_root = Path("/tmp/anonymous-v2xverse")
    monkeypatch.setattr(auto_trace, "_HEAL", heal_root)
    monkeypatch.setattr(auto_trace, "_V2XVERSE", v2xverse_root)
    monkeypatch.setattr(sys, "path", [str(v2xverse_root), str(heal_root), "other"])
    monkeypatch.setitem(sys.modules, "opencood", types.ModuleType("opencood"))
    monkeypatch.setitem(sys.modules, "opencood.models", types.ModuleType("opencood.models"))

    auto_trace._use_heal_opencood()
    assert "opencood" not in sys.modules
    assert sys.path[0] == str(heal_root)

    monkeypatch.setitem(sys.modules, "opencood", types.ModuleType("opencood"))
    monkeypatch.setitem(sys.modules, "opencood.models", types.ModuleType("opencood.models"))
    auto_trace._use_v2xverse_opencood()
    assert "opencood.models" not in sys.modules
    assert sys.path[0] == str(v2xverse_root)


def test_run_scan_writes_success_and_failure_manifests_without_hiding_errors(tmp_path: Path, monkeypatch) -> None:
    """The CLI adapter persists either an actual scan or its explicit failure record."""
    monkeypatch.setattr(run_scan, "get_scan_adapter", lambda _model: _SmallScanAdapter())
    status = run_scan.run_one("unit", _hardware(), "cpu", tmp_path, profile_latency="off")
    success = yaml.safe_load((tmp_path / "unit_partition.yaml").read_text(encoding="utf-8"))

    assert status == "ok"
    assert success["scan_status"] == "ok"
    assert success["view_b1_search_groups"]

    def _raise_missing_adapter(_model: str) -> TraceAdapter:
        raise KeyError("missing adapter")

    monkeypatch.setattr(run_scan, "get_scan_adapter", _raise_missing_adapter)
    failed_status = run_scan.run_one("missing", _hardware(), "cpu", tmp_path, profile_latency="off")
    failure = yaml.safe_load((tmp_path / "missing_partition.yaml").read_text(encoding="utf-8"))

    assert failed_status == "fail"
    assert failure["scan_status"] == "fail"
    assert "KeyError" in failure["error"]


def test_run_scan_main_executes_selected_or_all_models_with_explicit_latency_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI parsing passes the selected models and profiling controls to each independent scan."""
    hardware = _hardware()
    calls: list[tuple[str, str, int, int]] = []
    monkeypatch.setattr(run_scan, "ALL_MODELS", ["first", "second"])
    monkeypatch.setattr(
        run_scan.HwCapability,
        "from_yaml",
        staticmethod(lambda _path: hardware),
    )

    def _record(
        model: str,
        _hardware_value: HwCapability,
        _device: str,
        _out_dir: Path,
        profile_latency: str,
        lat_warmup: int,
        lat_measure: int,
    ) -> str:
        calls.append((model, profile_latency, lat_warmup, lat_measure))
        return "ok"

    monkeypatch.setattr(run_scan, "run_one", _record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scan.py",
            "--model",
            "all",
            "--out-dir",
            str(tmp_path),
            "--profile-latency",
            "on",
            "--lat-warmup",
            "1",
            "--lat-measure",
            "2",
        ],
    )
    run_scan.main()
    assert calls == [("first", "on", 1, 2), ("second", "on", 1, 2)]

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["run_scan.py", "--model", "first", "--out-dir", str(tmp_path)])
    run_scan.main()
    assert calls == [("first", "auto", 30, 100)]
    assert "Stage1 scan 汇总" in capsys.readouterr().out
