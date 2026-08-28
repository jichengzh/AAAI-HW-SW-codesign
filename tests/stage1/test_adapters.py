"""Contracts for formal Stage1 trace evidence and adapter selectors."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import inspect
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
import torch.nn as nn

from framework.stage1.adapters import (
    CoDrivingAdapter,
    FCooperAdapter,
    MaterializerParameterSource,
    PyramidCameraAdapter,
    PyramidLidarAdapter,
    REGISTRY,
    ScanScenario,
    TraceAdapter,
    TraceContext,
    V2XViTAdapter,
    get_adapter,
)
from framework.stage1.auto_trace import AutoTraceAdapter
from framework.stage1.formal_checkpoint_snapshot import (
    scanner_owned_checkpoint_snapshot,
)


@contextmanager
def _checkpoint_authority(tmp_path: Path):
    checkpoint = tmp_path / "authority.pth"
    checkpoint.write_bytes(b"scanner-owned-checkpoint")

    class _AuthorityAdapter:
        ckpt_path = str(checkpoint)
        formal_checkpoint_root = tmp_path

    with scanner_owned_checkpoint_snapshot(_AuthorityAdapter()) as authority:
        yield authority


def test_trace_context_contains_evidence_but_no_width_or_count_oracle() -> None:
    net = nn.Sequential(nn.Conv2d(8, 16, 1))
    source = MaterializerParameterSource(
        axis_id="encoder.output",
        config_selector="model.encoder.width",
        mutation_kind="out_channels",
        module_root_selector="0",
        allowed_roles=("output",),
        provenance={"adapter": "unit", "declaration": "encoder output"},
    )

    context = TraceContext(
        net=net,
        example_inputs=(torch.ones(1, 8, 2, 2),),
        full_model=net,
        loaded_config={"model": {"encoder": {"width": 16}}},
        checkpoint_evidence={"digest": "a" * 64, "module_widths": {"0": 16}},
        materializer_sources=(source,),
        trace_modules=dict(net.named_modules()),
        dataflow_relations=(
            {"module_root_selector": "0", "canonical_axis_id": "encoder.output"},
        ),
    )

    assert context.materializer_sources == (source,)
    assert not hasattr(source, "legal_widths")
    assert not hasattr(source, "axis_count")
    assert not hasattr(source, "candidate_count")
    with pytest.raises(FrozenInstanceError):
        source.axis_id = "paper-oracle"  # type: ignore[misc]


def test_scan_scenario_is_explicit_and_frozen() -> None:
    quant_policy = {"encoder": ["FP16"]}
    alignment = {"default_round_to": 8, "default_min_width": 8}
    scenario = ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16",),
        compression_modes=("fp16", "int8"),
        graph_quant_unit_policy=quant_policy,
        alignment=alignment,
    )

    assert scenario.backend_precisions == ("FP16",)
    assert scenario.graph_quant_unit_policy == {"encoder": ("FP16",)}
    quant_policy["encoder"].append("INT8")
    alignment["default_round_to"] = 32
    assert scenario.graph_quant_unit_policy == {"encoder": ("FP16",)}
    assert scenario.alignment["default_round_to"] == 8
    with pytest.raises(TypeError):
        scenario.alignment["default_round_to"] = 16  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        scenario.compression_modes = ("fp16",)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hardware_precisions", "FP16"),
        ("hardware_precisions", ("",)),
        ("backend_precisions", ()),
        ("compression_modes", ("fp16", 8)),
        ("graph_quant_unit_policy", {"encoder": "FP16"}),
        ("graph_quant_unit_policy", {"encoder": ("",)}),
        ("alignment", {"default_round_to": True}),
        ("alignment", {"default_round_to": 0}),
        ("alignment", {"default_round_to": "8"}),
    ],
)
def test_scan_scenario_rejects_invalid_runtime_types(field: str, value: object) -> None:
    kwargs = {
        "hardware_precisions": ("FP16",),
        "backend_precisions": ("FP16",),
        "compression_modes": ("fp16",),
        "graph_quant_unit_policy": {"encoder": ("FP16",)},
        "alignment": {"default_round_to": 8},
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError)):
        ScanScenario(**kwargs)  # type: ignore[arg-type]


class _SelectorAdapter(TraceAdapter):
    def materializer_parameter_sources(
        self, loaded_config: dict
    ) -> tuple[MaterializerParameterSource, ...]:
        assert loaded_config["model"]["width"] == 16
        return (
            MaterializerParameterSource(
                axis_id="encoder.output",
                config_selector="model.width",
                mutation_kind="out_channels",
                module_root_selector="0",
                allowed_roles=("output",),
                provenance={"adapter": "selector"},
            ),
        )


def test_trace_adapter_context_copies_and_deeply_freezes_scanner_evidence() -> None:
    net = nn.Sequential(nn.Conv2d(8, 16, 1))
    loaded_config = {"model": {"width": 16, "metadata": ["original"]}}
    checkpoint = {
        "digest": "d" * 64,
        "module_widths": {"0": 16},
    }

    context = _SelectorAdapter().build_trace_context(
        net,
        (torch.ones(1, 8, 2, 2),),
        loaded_config,
        checkpoint,
    )
    loaded_config["model"]["width"] = 99
    loaded_config["model"]["metadata"].append("forged")
    checkpoint["module_widths"]["0"] = 99

    assert context.loaded_config["model"]["width"] == 16
    assert context.loaded_config["model"]["metadata"] == ("original",)
    assert context.checkpoint_evidence["module_widths"]["0"] == 16
    with pytest.raises(TypeError):
        context.loaded_config["model"]["width"] = 24  # type: ignore[index]
    with pytest.raises(TypeError):
        context.materializer_sources[0].provenance["adapter"] = "forged"  # type: ignore[index]


def _codriving_config(filters: list[int]) -> dict:
    return {
        "model": {
            "args": {
                "base_bev_backbone": {"num_filters": filters},
                "shrink_header": {"dim": [filters[-1]]},
            }
        }
    }


def _pyramid_config(filters: list[int]) -> dict:
    return {
        "model": {
            "args": {
                "fusion_backbone": {"num_filters": filters},
                "shrink_header": {"dim": [filters[-1]]},
            }
        }
    }


def _fcooper_config(filters: list[int]) -> dict:
    return {
        "model": {
            "args": {
                "m1": {
                    "backbone_args": {
                        "num_filters": filters,
                        "num_upsample_filter": [filters[0] for _ in filters],
                    },
                    "shrink_header": {"dim": [filters[-1]]},
                }
            }
        }
    }


@pytest.mark.parametrize(
    ("adapter", "config", "filters", "sequence_selector"),
    [
        (
            CoDrivingAdapter(),
            _codriving_config([11, 22]),
            [11, 22],
            "model.args.base_bev_backbone.num_filters",
        ),
        (
            PyramidLidarAdapter(),
            _pyramid_config([11, 22, 33, 44]),
            [11, 22, 33, 44],
            "model.args.fusion_backbone.num_filters",
        ),
        (
            FCooperAdapter(),
            _fcooper_config([11, 22]),
            [11, 22],
            "model.args.m1.backbone_args.num_filters",
        ),
    ],
)
def test_paper_adapters_expand_backbone_selectors_from_config_sequence(
    adapter: TraceAdapter,
    config: dict,
    filters: list[int],
    sequence_selector: str,
) -> None:
    sources = adapter.materializer_parameter_sources(config)
    backbone = tuple(source for source in sources if source.axis_id.startswith("backbone."))
    expected = tuple(
        f"{sequence_selector}[{index}]"
        for index, _ in enumerate(filters)
    )

    assert tuple(source.config_selector for source in backbone) == expected
    assert all(not hasattr(source, "legal_widths") for source in sources)
    assert all(source.provenance for source in sources)


def test_fcooper_is_registered_and_declares_config_driven_neck_sources() -> None:
    adapter = get_adapter("fcooper")
    sources = adapter.materializer_parameter_sources(_fcooper_config([11, 22, 33, 44]))

    assert isinstance(adapter, FCooperAdapter)
    assert {
        source.config_selector
        for source in sources
        if source.axis_id.startswith("neck.")
    } == {
        "model.args.m1.backbone_args.num_upsample_filter[0]",
        "model.args.m1.shrink_header.dim[0]",
    }
    by_axis = {source.axis_id: source for source in sources}
    assert tuple(
        (target.selector, target.transform, target.reference_selector)
        for target in by_axis["neck.deblock"].write_targets
    ) == (
        (
            "model.args.m1.backbone_args.num_upsample_filter[*]",
            "repeat_to_reference_length",
            "model.args.m1.backbone_args.num_upsample_filter",
        ),
        (
            "model.args.m1.shrink_header.input_dim",
            "multiply_by_reference_length",
            "model.args.m1.backbone_args.num_upsample_filter",
        ),
    )
    assert tuple(
        (target.selector, target.transform)
        for target in by_axis["neck.output"].write_targets
    ) == (("model.args.in_head", "identity"),)


@pytest.mark.parametrize(
    "adapter",
    [adapter_type() for adapter_type in REGISTRY.values()],
    ids=tuple(REGISTRY),
)
def test_paper_adapter_contract_accepts_one_checkpoint_authority(
    adapter: TraceAdapter,
) -> None:
    signature = inspect.signature(adapter.build_trace_net)
    formal_signature = inspect.signature(adapter.build_formal_trace_net)

    signature.bind("cpu", checkpoint_authority=None)
    formal_signature.bind(
        "cpu", checkpoint_authority=None, loaded_model_config=None
    )


@pytest.mark.parametrize("adapter_name", tuple(REGISTRY))
def test_registered_adapter_formal_dispatch_reuses_both_authorities(
    adapter_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = get_adapter(adapter_name)
    checkpoint_authority = object()
    loaded_model_config = {"model": {"args": {}}}
    calls = []

    def build_trace_net(
        device,
        checkpoint_authority=None,
        loaded_model_config=None,
    ):
        calls.append((device, checkpoint_authority, loaded_model_config))
        return nn.Conv2d(4, 4, 1), torch.ones(1, 4, 2, 2)

    monkeypatch.setattr(adapter, "build_trace_net", build_trace_net)

    _net, _inputs, returned_config = adapter.build_formal_trace_net(
        "cpu",
        checkpoint_authority=checkpoint_authority,
        loaded_model_config=loaded_model_config,
    )

    assert calls == [("cpu", checkpoint_authority, loaded_model_config)]
    assert returned_config is loaded_model_config


def test_auto_trace_adapter_uses_supplied_checkpoint_authority(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def build(config_path: str, checkpoint_path: str, device: str) -> nn.Module:
        calls.append((config_path, checkpoint_path, device))
        return nn.Conv2d(4, 4, 1)

    adapter = AutoTraceAdapter(
        name="paper",
        model_class="PaperModel",
        config_path="/config.yaml",
        ckpt_path="/default.pth",
        build_fn=build,
        bev_shape=(1, 4, 2, 2),
    )

    with _checkpoint_authority(tmp_path) as authority:
        expected_path = str(authority.loader_path)
        adapter.build_trace_net("cpu", checkpoint_authority=authority)

    assert calls == [("/config.yaml", expected_path, "cpu")]


def test_codriving_loads_supplied_resolved_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 1)

        def forward(self, payload):
            return {"spatial_features_2d": self.conv(payload["spatial_features"])}

    class _Full(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _Backbone()
            self.shrink_flag = False
            self.cls_head = nn.Conv2d(4, 2, 1)
            self.reg_head = nn.Conv2d(4, 4, 1)

    opencood = ModuleType("opencood")
    opencood.__path__ = []  # type: ignore[attr-defined]
    hypes_yaml = ModuleType("opencood.hypes_yaml")
    hypes_yaml.__path__ = []  # type: ignore[attr-defined]
    yaml_utils = ModuleType("opencood.hypes_yaml.yaml_utils")
    yaml_utils.load_yaml = lambda _path: pytest.fail("config must be reused")  # type: ignore[attr-defined]
    models = ModuleType("opencood.models")
    models.__path__ = []  # type: ignore[attr-defined]
    centerpoint = ModuleType("opencood.models.center_point_codriving")
    centerpoint.centerpointcodriving = lambda _args: _Full()  # type: ignore[attr-defined]
    for name, module in {
        "opencood": opencood,
        "opencood.hypes_yaml": hypes_yaml,
        "opencood.hypes_yaml.yaml_utils": yaml_utils,
        "opencood.models": models,
        "opencood.models.center_point_codriving": centerpoint,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def load_checkpoint(path: Path, *, map_location: str, weights_only: bool):
        assert map_location == "cpu"
        assert weights_only is True
        calls.append(path)
        return {}

    monkeypatch.setattr(torch, "load", load_checkpoint)
    loaded_model_config = {"model": {"args": {}}}

    with _checkpoint_authority(tmp_path) as authority:
        expected_path = authority.loader_path
        _, _, returned_config = CoDrivingAdapter().build_formal_trace_net(
            "cpu",
            checkpoint_authority=authority,
            loaded_model_config=loaded_model_config,
        )

    assert calls == [expected_path]
    assert returned_config is loaded_model_config


def test_pyramid_loads_supplied_resolved_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class _PyramidTrace(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone_m1 = nn.Sequential(nn.Conv2d(4, 4, 1))

    dependency = ModuleType("tools.configurable.depgraph_pyramid")

    def build_full(
        config_path: str,
        checkpoint_path: str,
        device: str,
        *,
        loaded_hypes: object | None = None,
    ) -> object:
        del config_path, device
        calls.append((checkpoint_path, loaded_hypes))
        return object()

    dependency.build_full = build_full  # type: ignore[attr-defined]
    dependency.PyramidFullTraceNet = lambda _full: _PyramidTrace()  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "tools.configurable.depgraph_pyramid", dependency
    )

    loaded_model_config = {"model": {"args": {}}}
    with _checkpoint_authority(tmp_path) as authority:
        expected_path = str(authority.loader_path)
        _, _, returned_config = PyramidLidarAdapter().build_formal_trace_net(
            "cpu",
            checkpoint_authority=authority,
            loaded_model_config=loaded_model_config,
        )

    assert calls == [(expected_path, loaded_model_config)]
    assert returned_config is loaded_model_config


def test_fcooper_delegates_supplied_checkpoint_authority_to_auto_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []

    class _Auto:
        def build_trace_net(
            self,
            device,
            checkpoint_authority=None,
            loaded_model_config=None,
        ):
            del device
            calls.append((checkpoint_authority, loaded_model_config))
            return nn.Conv2d(4, 4, 1), torch.ones(1, 4, 2, 2)

    monkeypatch.setattr(
        "framework.stage1.auto_trace.get_auto_adapter", lambda _name: _Auto()
    )

    loaded_model_config = {"model": {"args": {}}}
    with _checkpoint_authority(tmp_path) as authority:
        _, _, returned_config = FCooperAdapter().build_formal_trace_net(
            "cpu",
            checkpoint_authority=authority,
            loaded_model_config=loaded_model_config,
        )

    assert calls == [(authority, loaded_model_config)]
    assert returned_config is loaded_model_config


def test_pyramid_camera_formal_build_uses_only_scanner_owned_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 1)

        def forward(self, payload):
            return {"spatial_features_2d": self.conv(payload["spatial_features"])}

    class _Pyramid(nn.Module):
        num_levels = 0

        def forward_single(self, value):
            return value, []

    class _Full(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone_m2 = _Backbone()
            self.aligner_m2 = nn.Identity()
            self.pyramid_backbone = _Pyramid()
            self.shrink_flag = False
            self.cls_head = nn.Conv2d(4, 2, 1)
            self.reg_head = nn.Conv2d(4, 4, 1)
            self.dir_head = nn.Conv2d(4, 2, 1)

    yaml_utils = ModuleType("opencood.hypes_yaml.yaml_utils")
    yaml_utils.load_yaml = lambda _path: pytest.fail("ambient config fallback")  # type: ignore[attr-defined]
    pyramid_model = ModuleType("opencood.models.heter_pyramid_single")
    model_args = {"camera": "scanner-owned"}
    pyramid_model.HeterPyramidSingle = (  # type: ignore[attr-defined]
        lambda args: calls.append(("config", args)) or _Full()
    )
    monkeypatch.setitem(sys.modules, "opencood.hypes_yaml.yaml_utils", yaml_utils)
    monkeypatch.setitem(
        sys.modules, "opencood.models.heter_pyramid_single", pyramid_model
    )
    monkeypatch.setattr("framework.stage1.adapters.os.chdir", lambda _path: None)

    def load_checkpoint(path, *, map_location, weights_only):
        calls.append((path, (map_location, weights_only)))
        return {}

    monkeypatch.setattr(torch, "load", load_checkpoint)
    loaded_model_config = {"model": {"args": model_args}}
    with _checkpoint_authority(tmp_path) as authority:
        expected_path = authority.loader_path
        _net, _inputs, returned_config = (
            PyramidCameraAdapter().build_formal_trace_net(
                "cpu",
                checkpoint_authority=authority,
                loaded_model_config=loaded_model_config,
            )
        )

    assert calls == [
        ("config", model_args),
        (expected_path, ("cpu", True)),
    ]
    assert returned_config is loaded_model_config


def test_v2xvit_formal_build_uses_only_scanner_owned_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class _Full(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone_m1 = nn.Sequential(nn.Conv2d(4, 4, 1))
            self.shrinker_m1 = nn.Identity()
            self.shrink_flag = False
            self.cls_head = nn.Conv2d(4, 2, 1)
            self.reg_head = nn.Conv2d(4, 4, 1)
            self.dir_head = nn.Conv2d(4, 2, 1)

        def load_state_dict(self, state_dict, strict=True):
            calls.append(("state", (state_dict, strict)))
            return (), ()

    class _Trace(nn.Module):
        def __init__(self, full: _Full) -> None:
            super().__init__()
            self.backbone_m1 = full.backbone_m1
            self.shrinker_m1 = full.shrinker_m1
            self.has_shrink = False
            self.cls_head = full.cls_head
            self.reg_head = full.reg_head
            self.dir_head = full.dir_head

    dependency = ModuleType("tools.configurable.depgraph_v2xvit")
    dependency.build_model = lambda _device: pytest.fail("ambient model fallback")  # type: ignore[attr-defined]
    dependency.V2XViTBackboneTraceNet = _Trace  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "tools.configurable.depgraph_v2xvit", dependency
    )
    yaml_utils = ModuleType("opencood.hypes_yaml.yaml_utils")

    def load_general_params(config):
        calls.append(("config", config))
        return config

    yaml_utils.load_general_params = load_general_params  # type: ignore[attr-defined]
    train_utils = ModuleType("opencood.tools.train_utils")
    train_utils.create_model = (  # type: ignore[attr-defined]
        lambda config: calls.append(("model", config)) or _Full()
    )
    opencood_tools = ModuleType("opencood.tools")
    opencood_tools.train_utils = train_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opencood.hypes_yaml.yaml_utils", yaml_utils)
    monkeypatch.setitem(sys.modules, "opencood.tools", opencood_tools)
    monkeypatch.setitem(sys.modules, "opencood.tools.train_utils", train_utils)

    def load_checkpoint(path, *, map_location, weights_only):
        calls.append(("checkpoint", (path, map_location, weights_only)))
        return {"model_state_dict": {"scanner": "owned"}}

    monkeypatch.setattr(torch, "load", load_checkpoint)
    loaded_model_config = {
        "model": {"args": {}},
        "test_dir": "scanner-test-dir",
    }
    with _checkpoint_authority(tmp_path) as authority:
        expected_path = authority.loader_path
        _net, _inputs, returned_config = V2XViTAdapter().build_formal_trace_net(
            "cpu",
            checkpoint_authority=authority,
            loaded_model_config=loaded_model_config,
        )

    assert calls == [
        ("config", loaded_model_config),
        ("model", loaded_model_config),
        ("checkpoint", (expected_path, "cpu", True)),
        ("state", ({"scanner": "owned"}, True)),
    ]
    assert loaded_model_config["validate_dir"] == "scanner-test-dir"
    assert returned_config is loaded_model_config
