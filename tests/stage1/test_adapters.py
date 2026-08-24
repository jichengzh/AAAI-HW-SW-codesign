"""Contracts for formal Stage1 trace evidence and adapter selectors."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn as nn

from framework.stage1.adapters import (
    CoDrivingAdapter,
    FCooperAdapter,
    MaterializerParameterSource,
    PyramidLidarAdapter,
    ScanScenario,
    TraceAdapter,
    TraceContext,
    get_adapter,
)


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
