"""Tests for scanner-owned structural-axis derivation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from framework.stage1.adapters import (
    ConfigWriteTarget,
    MaterializerParameterSource,
    ScanScenario,
    TraceContext,
)
from framework.stage1.structural_axes import (
    axis_to_scanner_dict,
    build_structural_axis_inputs,
    derive_structural_axes,
)


def _selector_context(
    *,
    sources: tuple[MaterializerParameterSource, ...] | None = None,
    loaded_config: dict | None = None,
    checkpoint_width: int = 32,
    module_selector: str = "backbone.0",
) -> TraceContext:
    net = nn.Module()
    net.backbone = nn.Sequential(nn.Conv2d(16, 32, 1))
    declared = sources
    if declared is None:
        declared = (
            MaterializerParameterSource(
                axis_id="backbone.output",
                config_selector="model.encoder.width",
                mutation_kind="out_channels",
                module_root_selector=module_selector,
                allowed_roles=("output",),
                provenance={"adapter": "unit", "declaration": "encoder output"},
            ),
        )
    return TraceContext(
        net=net,
        example_inputs=(torch.ones(1, 16, 2, 2),),
        full_model=net,
        loaded_config=loaded_config or {"model": {"encoder": {"width": 32}}},
        checkpoint_evidence={
            "digest": "b" * 64,
            "module_widths": {"backbone.0": checkpoint_width},
        },
        materializer_sources=declared,
        trace_modules=dict(net.named_modules()),
        dataflow_relations=(
            {
                "module_root_selector": "backbone.0",
                "canonical_axis_id": "backbone.output",
            },
        ),
    )


def _selector_groups(root_layer: str = "backbone.0") -> list[dict]:
    return [
        {
            "group_id": "g0",
            "root_layer": root_layer,
            "cur_width": 32,
            "round_to_default": 8,
            "width_floor": 8,
        }
    ]


def _selector_scenario() -> ScanScenario:
    return ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16", "INT8"),
        compression_modes=("fp16", "int8"),
        graph_quant_unit_policy={"backbone": ("FP16", "INT8")},
        alignment={"default_round_to": 8, "default_min_width": 8},
    )


def test_build_inputs_resolves_config_module_checkpoint_and_depgraph_evidence() -> None:
    inputs = build_structural_axis_inputs(
        trace_context=_selector_context(),
        prune_groups=_selector_groups(),
        scenario=_selector_scenario(),
    )

    assert inputs["base_widths"] == [
        {
            "axis_id": "backbone.output",
            "width": 32,
            "config_path": "model.encoder.width",
        }
    ]
    assert inputs["materializer_bindings"][0]["b1_group_id"] == "g0"
    assert inputs["materializer_bindings"][0]["param"] == "model.encoder.width"
    assert inputs["provenance"]["checkpoint_digest"] == "b" * 64
    assert len(inputs["digest"]) == 64


def test_build_inputs_preserves_generic_materializer_writeback_transforms() -> None:
    source = MaterializerParameterSource(
        axis_id="backbone.output",
        config_selector="model.encoder.width",
        mutation_kind="out_channels",
        module_root_selector="backbone.0",
        allowed_roles=("output",),
        provenance={"adapter": "unit"},
        write_targets=(
            ConfigWriteTarget(
                selector="model.encoder.aliases[*]",
                transform="repeat_to_reference_length",
                reference_selector="model.encoder.aliases",
            ),
        ),
    )

    inputs = build_structural_axis_inputs(
        trace_context=_selector_context(sources=(source,)),
        prune_groups=_selector_groups(),
        scenario=_selector_scenario(),
    )

    assert inputs["materializer_bindings"][0]["write_targets"] == [
        {
            "selector": "model.encoder.aliases[*]",
            "transform": "repeat_to_reference_length",
            "reference_selector": "model.encoder.aliases",
        }
    ]


def test_canonical_digests_ignore_mapping_insertion_order_after_freeze() -> None:
    first = _selector_context(
        loaded_config={
            "metadata": {"second": 2, "first": 1},
            "model": {"encoder": {"label": "unit", "width": 32}},
        }
    )
    second = _selector_context(
        loaded_config={
            "model": {"encoder": {"width": 32, "label": "unit"}},
            "metadata": {"first": 1, "second": 2},
        }
    )

    first_inputs = build_structural_axis_inputs(
        trace_context=first,
        prune_groups=_selector_groups(),
        scenario=_selector_scenario(),
    )
    second_inputs = build_structural_axis_inputs(
        trace_context=second,
        prune_groups=_selector_groups(),
        scenario=_selector_scenario(),
    )
    first_axis = derive_structural_axes(
        {"structural_axis_inputs": first_inputs}
    ).axes[0]
    second_axis = derive_structural_axes(
        {"structural_axis_inputs": second_inputs}
    ).axes[0]

    assert first_inputs["provenance"]["config_digest"] == second_inputs[
        "provenance"
    ]["config_digest"]
    assert first_inputs["digest"] == second_inputs["digest"]
    assert axis_to_scanner_dict(first_axis, first_inputs) == axis_to_scanner_dict(
        second_axis, second_inputs
    )


def test_canonical_digest_rejects_unsupported_config_values() -> None:
    context = _selector_context(
        loaded_config={
            "model": {"encoder": {"width": 32}},
            "unsupported": object(),
        }
    )

    with pytest.raises(TypeError, match="JSON primitive"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_two_axis_ids_for_same_physical_coordinate() -> None:
    first = _selector_context().materializer_sources[0]
    second = MaterializerParameterSource(
        axis_id="alias.output",
        config_selector=first.config_selector,
        mutation_kind=first.mutation_kind,
        module_root_selector=first.module_root_selector,
        allowed_roles=first.allowed_roles,
        provenance={"adapter": "alias"},
    )
    base = _selector_context()
    context = TraceContext(
        net=base.net,
        example_inputs=base.example_inputs,
        full_model=base.full_model,
        loaded_config=base.loaded_config,
        checkpoint_evidence=base.checkpoint_evidence,
        materializer_sources=(first, second),
        trace_modules=base.trace_modules,
        dataflow_relations=(
            *base.dataflow_relations,
            {
                "module_root_selector": "backbone.0",
                "canonical_axis_id": "alias.output",
            },
        ),
    )

    with pytest.raises(ValueError, match="duplicate physical axis"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_missing_adapter_binding() -> None:
    with pytest.raises(ValueError, match="materializer binding"):
        build_structural_axis_inputs(
            trace_context=_selector_context(sources=()),
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_missing_canonical_base() -> None:
    context = _selector_context(
        loaded_config={"model": {"encoder": {"channels": 32}}}
    )
    with pytest.raises(ValueError, match="canonical base"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_binding_without_depgraph_group() -> None:
    with pytest.raises(ValueError, match="DepGraph group"):
        build_structural_axis_inputs(
            trace_context=_selector_context(),
            prune_groups=_selector_groups(root_layer="somewhere.else"),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_non_unique_config_path() -> None:
    source = MaterializerParameterSource(
        axis_id="backbone.output",
        config_selector="model.*.width",
        mutation_kind="out_channels",
        module_root_selector="backbone.0",
        allowed_roles=("output",),
        provenance={"adapter": "unit"},
    )
    context = _selector_context(
        sources=(source,),
        loaded_config={"model": {"first": {"width": 32}, "second": {"width": 32}}},
    )
    with pytest.raises(ValueError, match="unique config path"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_ambiguous_module_selector() -> None:
    source = MaterializerParameterSource(
        axis_id="backbone.output",
        config_selector="model.encoder.width",
        mutation_kind="out_channels",
        module_root_selector="backbone*",
        allowed_roles=("output",),
        provenance={"adapter": "unit"},
    )
    with pytest.raises(ValueError, match="unique trace module"):
        build_structural_axis_inputs(
            trace_context=_selector_context(sources=(source,)),
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_checkpoint_module_width_mismatch() -> None:
    with pytest.raises(ValueError, match="checkpoint evidence"):
        build_structural_axis_inputs(
            trace_context=_selector_context(checkpoint_width=24),
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_depgraph_module_width_mismatch() -> None:
    groups = _selector_groups()
    groups[0]["cur_width"] = 24
    with pytest.raises(ValueError, match="DepGraph evidence"):
        build_structural_axis_inputs(
            trace_context=_selector_context(),
            prune_groups=groups,
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_selector_without_provenance() -> None:
    source = MaterializerParameterSource(
        axis_id="backbone.output",
        config_selector="model.encoder.width",
        mutation_kind="out_channels",
        module_root_selector="backbone.0",
        allowed_roles=("output",),
        provenance={},
    )
    with pytest.raises(ValueError, match="provenance"):
        build_structural_axis_inputs(
            trace_context=_selector_context(sources=(source,)),
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_duplicate_axis_declarations() -> None:
    source = _selector_context().materializer_sources[0]
    with pytest.raises(ValueError, match="duplicate materializer axis"):
        build_structural_axis_inputs(
            trace_context=_selector_context(sources=(source, source)),
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def _scan(
    prune_groups: list[dict],
    dataflow_relations: list[dict],
    materializer_bindings: list[dict],
    base_widths: list[dict],
    backend_constraints: list[dict],
) -> dict:
    return {
        "structural_axis_inputs": {
            "prune_groups": prune_groups,
            "dataflow_relations": dataflow_relations,
            "materializer_bindings": materializer_bindings,
            "base_widths": base_widths,
            "backend_constraints": backend_constraints,
        }
    }


def _pyramid_stage_with_output_c_and_internal_2c() -> dict:
    return _scan(
        prune_groups=[
            {
                "id": "stage3.out",
                "module_path": "backbone.stage3.out",
                "cur_width": 256,
                "dataflow_group_id": "stage3",
            },
            {
                "id": "stage3.inner",
                "module_path": "backbone.stage3.inner",
                "cur_width": 512,
                "dataflow_group_id": "stage3",
            },
        ],
        dataflow_relations=[
            {"group_id": "stage3", "canonical_axis_id": "backbone.stage3"}
        ],
        materializer_bindings=[
            {
                "b1_group_id": "stage3.out",
                "axis_id": "backbone.stage3",
                "param": "stage3",
                "role": "output",
            },
            {
                "b1_group_id": "stage3.inner",
                "axis_id": "backbone.stage3",
                "param": "stage3",
                "role": "internal",
            },
        ],
        base_widths=[{"axis_id": "backbone.stage3", "width": 256}],
        backend_constraints=[
            {"axis_id": "backbone.stage3", "round_to": 32, "min_width": 64}
        ],
    )


def _stage_with_non_integral_member_width() -> dict:
    bad = _pyramid_stage_with_output_c_and_internal_2c()
    bad["structural_axis_inputs"]["prune_groups"][1]["cur_width"] = 384
    bad["structural_axis_inputs"]["backend_constraints"][0]["min_width"] = 96
    return bad


def _codriving_scan_with_neck_binding() -> dict:
    return _scan(
        prune_groups=[
            {"id": "s1.out", "module_path": "backbone.stage1", "cur_width": 64, "dataflow_group_id": "s1"},
            {"id": "s2.out", "module_path": "backbone.stage2", "cur_width": 128, "dataflow_group_id": "s2"},
            {"id": "s3.out", "module_path": "backbone.stage3", "cur_width": 256, "dataflow_group_id": "s3"},
            {"id": "neck.out", "module_path": "neck.output", "cur_width": 256, "dataflow_group_id": "s3"},
        ],
        dataflow_relations=[
            {"group_id": "s1", "canonical_axis_id": "backbone.stage1"},
            {"group_id": "s2", "canonical_axis_id": "backbone.stage2"},
            {"group_id": "s3", "canonical_axis_id": "backbone.stage3"},
        ],
        materializer_bindings=[
            {"b1_group_id": "s1.out", "axis_id": "backbone.stage1", "param": "stage1", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "s2.out", "axis_id": "backbone.stage2", "param": "stage2", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "s3.out", "axis_id": "backbone.stage3", "param": "stage3", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "neck.out", "axis_id": "neck.output", "param": "stage3", "role": "derived_neck", "axis_kind": "fixed_derived", "derived_from": "backbone.stage3"},
        ],
        base_widths=[
            {"axis_id": "backbone.stage1", "width": 64},
            {"axis_id": "backbone.stage2", "width": 128},
            {"axis_id": "backbone.stage3", "width": 256},
            {"axis_id": "neck.output", "width": 256},
        ],
        backend_constraints=[
            {"axis_id": "backbone.stage1", "round_to": 8, "min_width": 16},
            {"axis_id": "backbone.stage2", "round_to": 16, "min_width": 32},
            {"axis_id": "backbone.stage3", "round_to": 32, "min_width": 64},
            {"axis_id": "neck.output", "round_to": 32, "min_width": 64},
        ],
    )


def _fcooper_scan_with_independent_neck_bindings() -> dict:
    return _scan(
        prune_groups=[
            {"id": "b0.out", "module_path": "backbone.s0", "cur_width": 64, "dataflow_group_id": "b0"},
            {"id": "b1.out", "module_path": "backbone.s1", "cur_width": 128, "dataflow_group_id": "b1"},
            {"id": "b2.out", "module_path": "backbone.s2", "cur_width": 256, "dataflow_group_id": "b2"},
            {"id": "neck.deblock", "module_path": "neck.deblock", "cur_width": 128, "dataflow_group_id": "neck_deblock"},
            {"id": "neck.output", "module_path": "neck.output", "cur_width": 256, "dataflow_group_id": "neck_output"},
        ],
        dataflow_relations=[
            {"group_id": "b0", "canonical_axis_id": "backbone.s0"},
            {"group_id": "b1", "canonical_axis_id": "backbone.s1"},
            {"group_id": "b2", "canonical_axis_id": "backbone.s2"},
            {"group_id": "neck_deblock", "canonical_axis_id": "neck.deblock", "independent_interface": True},
            {"group_id": "neck_output", "canonical_axis_id": "neck.output", "independent_interface": True},
        ],
        materializer_bindings=[
            {"b1_group_id": "b0.out", "axis_id": "backbone.s0", "param": "backbone.s0", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "b1.out", "axis_id": "backbone.s1", "param": "backbone.s1", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "b2.out", "axis_id": "backbone.s2", "param": "backbone.s2", "role": "output", "axis_kind": "free"},
            {"b1_group_id": "neck.deblock", "axis_id": "neck.deblock", "param": "neck.deblock", "role": "neck_interface", "axis_kind": "free"},
            {"b1_group_id": "neck.output", "axis_id": "neck.output", "param": "neck.output", "role": "neck_interface", "axis_kind": "free"},
        ],
        base_widths=[
            {"axis_id": "backbone.s0", "width": 64},
            {"axis_id": "backbone.s1", "width": 128},
            {"axis_id": "backbone.s2", "width": 256},
            {"axis_id": "neck.deblock", "width": 128},
            {"axis_id": "neck.output", "width": 256},
        ],
        backend_constraints=[
            {"axis_id": "backbone.s0", "round_to": 32, "min_width": 32},
            {"axis_id": "backbone.s1", "round_to": 32, "min_width": 32},
            {"axis_id": "backbone.s2", "round_to": 32, "min_width": 32},
            {"axis_id": "neck.deblock", "round_to": 32, "min_width": 32},
            {"axis_id": "neck.output", "round_to": 32, "min_width": 64},
        ],
    )


def test_derive_structural_axis_uses_canonical_base_not_max_internal_width() -> None:
    bundle = derive_structural_axes(_pyramid_stage_with_output_c_and_internal_2c())
    axis = bundle.free_axes[0]

    assert axis.axis_id == "backbone.stage3"
    assert axis.base_width == 256
    assert axis.legal_widths == (64, 96, 128, 160, 192, 224, 256)
    assert [
        (member.canonical_to_member_num, member.canonical_to_member_den)
        for member in axis.member_b1_groups
    ] == [(1, 1), (2, 1)]


def test_derive_structural_axes_rejects_non_integral_member_ratio() -> None:
    with pytest.raises(ValueError, match="integer ratio"):
        derive_structural_axes(_stage_with_non_integral_member_width())


def test_derive_structural_axes_rejects_missing_canonical_base_width() -> None:
    scan = _pyramid_stage_with_output_c_and_internal_2c()
    scan["structural_axis_inputs"]["base_widths"] = []

    with pytest.raises(ValueError, match="canonical base"):
        derive_structural_axes(scan)


def test_derive_structural_axes_rejects_missing_backend_constraint() -> None:
    scan = _pyramid_stage_with_output_c_and_internal_2c()
    scan["structural_axis_inputs"]["backend_constraints"] = []

    with pytest.raises(ValueError, match="backend constraint"):
        derive_structural_axes(scan)


def test_derive_structural_axes_rejects_mixed_axis_kind_bindings() -> None:
    scan = _codriving_scan_with_neck_binding()
    scan["structural_axis_inputs"]["materializer_bindings"].append(
        {
            "b1_group_id": "s3.out",
            "axis_id": "neck.output",
            "param": "stage3",
            "role": "bad_free_mix",
            "axis_kind": "free",
        }
    )

    with pytest.raises(ValueError, match="mixes free and fixed-derived"):
        derive_structural_axes(scan)


def test_codriving_neck_is_fixed_derived_not_free_axis() -> None:
    bundle = derive_structural_axes(_codriving_scan_with_neck_binding())

    assert [axis.axis_id for axis in bundle.free_axes] == [
        "backbone.stage1",
        "backbone.stage2",
        "backbone.stage3",
    ]
    assert [axis.axis_id for axis in bundle.fixed_axes] == ["neck.output"]
    assert bundle.fixed_axes[0].provenance["derived_from"] == "backbone.stage3"


def test_fcooper_neck_interfaces_are_free_when_bindings_are_independent() -> None:
    bundle = derive_structural_axes(_fcooper_scan_with_independent_neck_bindings())

    assert [axis.axis_id for axis in bundle.free_axes] == [
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    ]
