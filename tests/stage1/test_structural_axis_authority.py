"""Focused structural-axis lattice and scanner-contract tests."""

from __future__ import annotations

import pytest

from framework.stage1.adapters import MaterializerParameterSource, ScanScenario, TraceContext
from framework.stage1.structural_axes import (
    axis_to_scanner_dict,
    build_structural_axis_inputs,
    derive_structural_axes,
)
from framework.stage1.structural_axis_widths import (
    derive_legal_widths,
    scenario_axis_constraint,
    validated_width_policy,
)
from tests.stage1.structural_axis_selector_test_support import (
    built_selector_inputs,
    scenario_evidence,
    selector_context,
    selector_group_manifest,
    selector_groups,
    selector_scenario,
)
from tests.stage1.structural_axis_test_support import resign_structural_inputs


def _trusted_selector_relations():
    return selector_context().dataflow_relations


def _derive_selector_inputs(inputs: dict):
    return derive_structural_axes(
        {"structural_axis_inputs": inputs},
        trusted_source_relation_declarations=_trusted_selector_relations(),
    )


def test_legal_widths_use_max_rate_and_alignment_not_probe_anchors() -> None:
    """Diagnostic probe samples must not truncate the formal pruning lattice."""
    inputs = built_selector_inputs()
    constraint = inputs["backend_constraints"][0]
    constraint["probe_anchors"] = [8, 32]
    resign_structural_inputs(inputs)

    first = _derive_selector_inputs(inputs).free_axes[0]
    constraint["probe_anchors"] = [16, 24]
    resign_structural_inputs(inputs)
    second = _derive_selector_inputs(inputs).free_axes[0]

    assert first.legal_widths == (8, 16, 24, 32)
    assert second.legal_widths == first.legal_widths


def test_formal_max_rate_comes_from_scenario_policy_not_depgraph_diagnostic() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16", "INT8"),
        compression_modes=("fp16", "int8"),
        graph_quant_unit_policy={"backbone": ("FP16", "INT8")},
        alignment={
            "default_round_to": 8,
            "default_hardware_alignment": 8,
            "default_max_rate_numerator": 3,
            "default_max_rate_denominator": 4,
        },
    )
    groups = [{**selector_groups()[0], "max_rate": 0.5}]

    inputs = build_structural_axis_inputs(
        trace_context=selector_context(),
        prune_groups=groups,
        scenario=scenario,
        group_manifest=selector_group_manifest(groups),
    )
    constraint = inputs["backend_constraints"][0]
    axis = _derive_selector_inputs(inputs).free_axes[0]

    assert "max_rate" not in constraint
    assert constraint["max_rate_numerator"] == 3
    assert constraint["max_rate_denominator"] == 4
    assert constraint["max_rate_source"] == "scan_scenario.backend_policy"
    assert constraint["scenario_digest"] == inputs["provenance"]["scenario_digest"]
    assert axis.legal_widths == (8, 16, 24, 32)


@pytest.mark.parametrize(
    "constraint_update",
    [
        {"max_rate": "3/4", "max_rate_source": "scan_scenario.backend_policy"},
        {
            "max_rate_numerator": 3,
            "max_rate_denominator": 4,
            "max_rate_source": "depgraph.rollup",
        },
    ],
)
def test_derive_rejects_unverified_formal_rate_policy(
    constraint_update: dict,
) -> None:
    inputs = built_selector_inputs()
    original = inputs["backend_constraints"][0]
    inputs["backend_constraints"][0] = {
        "axis_id": original["axis_id"],
        "round_to": original["round_to"],
        "hardware_alignment": original["hardware_alignment"],
        "scenario_digest": inputs["provenance"]["scenario_digest"],
        **constraint_update,
    }
    resign_structural_inputs(inputs)

    with pytest.raises(ValueError, match="formal max_rate policy"):
        _derive_selector_inputs(inputs)


def test_derive_rejects_constraint_tamper_against_scanner_evidence_seal() -> None:
    inputs = built_selector_inputs()
    inputs["backend_constraints"][0]["max_rate_numerator"] = 1
    resign_structural_inputs(inputs)

    with pytest.raises(ValueError, match="scanner evidence|formal max_rate policy"):
        _derive_selector_inputs(inputs)


def test_formal_axis_fails_closed_without_scenario_max_rate_policy() -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16",),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 8, "default_hardware_alignment": 8},
    )

    with pytest.raises(ValueError, match="formal max_rate policy"):
        build_structural_axis_inputs(
            trace_context=selector_context(),
            prune_groups=selector_groups(),
            scenario=scenario,
            group_manifest=selector_group_manifest(scenario=scenario),
        )


@pytest.mark.parametrize("numerator", [True, -1, 4, "not-a-rate"])
def test_legal_widths_reject_malformed_max_rate(numerator: object) -> None:
    scenario = scenario_evidence(selector_scenario())
    constraint = scenario_axis_constraint("backbone.output", scenario)
    constraint["max_rate_numerator"] = numerator
    with pytest.raises(ValueError, match="max_rate"):
        validated_width_policy(constraint, "backbone.output", scenario)


def test_legal_widths_reject_misaligned_base_checkpoint_width() -> None:
    scenario = scenario_evidence(selector_scenario())
    policy = validated_width_policy(
        scenario_axis_constraint("backbone.output", scenario),
        "backbone.output",
        scenario,
    )
    with pytest.raises(ValueError, match="base width"):
        derive_legal_widths(
            base_width=60,
            policy=policy,
            round_to=8,
            hardware_alignment=8,
        )


def test_canonical_digests_ignore_mapping_insertion_order_after_freeze() -> None:
    first = selector_context(
        loaded_config={
            "metadata": {"second": 2, "first": 1},
            "model": {"encoder": {"label": "unit", "width": 32}},
        }
    )
    second = selector_context(
        loaded_config={
            "model": {"encoder": {"width": 32, "label": "unit"}},
            "metadata": {"first": 1, "second": 2},
        }
    )

    first_inputs = build_structural_axis_inputs(
        trace_context=first,
        prune_groups=selector_groups(),
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(),
    )
    second_inputs = build_structural_axis_inputs(
        trace_context=second,
        prune_groups=selector_groups(),
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(),
    )
    first_axis = derive_structural_axes(
        {"structural_axis_inputs": first_inputs},
        trusted_source_relation_declarations=first.dataflow_relations,
    ).axes[0]
    second_axis = derive_structural_axes(
        {"structural_axis_inputs": second_inputs},
        trusted_source_relation_declarations=second.dataflow_relations,
    ).axes[0]

    assert first_inputs["provenance"]["config_digest"] == second_inputs[
        "provenance"
    ]["config_digest"]
    assert first_inputs["digest"] == second_inputs["digest"]
    assert axis_to_scanner_dict(
        first_axis,
        first_inputs,
        trusted_source_relation_declarations=first.dataflow_relations,
    ) == axis_to_scanner_dict(
        second_axis,
        second_inputs,
        trusted_source_relation_declarations=second.dataflow_relations,
    )


def test_canonical_digest_rejects_unsupported_config_values() -> None:
    context = selector_context(
        loaded_config={
            "model": {"encoder": {"width": 32}},
            "unsupported": object(),
        }
    )

    with pytest.raises(TypeError, match="JSON primitive"):
        build_structural_axis_inputs(
                trace_context=context,
                prune_groups=selector_groups(),
                scenario=selector_scenario(),
                group_manifest=selector_group_manifest(
                    source_relations=[
                        dict(row) for row in context.dataflow_relations
                    ]
                ),
        )


def test_build_inputs_rejects_two_axis_ids_for_same_physical_coordinate() -> None:
    first = selector_context().materializer_sources[0]
    second = MaterializerParameterSource(
        axis_id="alias.output",
        config_selector=first.config_selector,
        mutation_kind=first.mutation_kind,
        module_root_selector=first.module_root_selector,
        allowed_roles=first.allowed_roles,
        provenance={"adapter": "alias"},
    )
    base = selector_context()
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
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(
                source_relations=[dict(row) for row in context.dataflow_relations]
            ),
        )


def test_build_inputs_rejects_missing_adapter_binding() -> None:
    with pytest.raises(ValueError, match="materializer binding"):
        build_structural_axis_inputs(
            trace_context=selector_context(sources=()),
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(),
        )


def test_build_inputs_rejects_missing_canonical_base() -> None:
    context = selector_context(
        loaded_config={"model": {"encoder": {"channels": 32}}}
    )
    with pytest.raises(ValueError, match="canonical base"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(),
        )


def test_build_inputs_rejects_binding_without_depgraph_group() -> None:
    groups = selector_groups(root_layer="somewhere.else")
    with pytest.raises(ValueError, match="DepGraph group"):
        build_structural_axis_inputs(
            trace_context=selector_context(),
            prune_groups=groups,
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(groups),
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
    context = selector_context(
        sources=(source,),
        loaded_config={"model": {"first": {"width": 32}, "second": {"width": 32}}},
    )
    with pytest.raises(ValueError, match="unique config path"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(),
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
            trace_context=selector_context(sources=(source,)),
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(),
        )


def test_build_inputs_rejects_checkpoint_module_width_mismatch() -> None:
    with pytest.raises(ValueError, match="checkpoint evidence"):
        build_structural_axis_inputs(
            trace_context=selector_context(checkpoint_width=24),
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(),
        )


def test_build_inputs_rejects_depgraph_module_width_mismatch() -> None:
    groups = selector_groups()
    groups[0]["cur_width"] = 24
    with pytest.raises(ValueError, match="DepGraph evidence"):
        build_structural_axis_inputs(
            trace_context=selector_context(),
            prune_groups=groups,
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(groups),
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
            trace_context=selector_context(sources=(source,)),
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(),
        )


def test_build_inputs_rejects_duplicate_axis_declarations() -> None:
    source = selector_context().materializer_sources[0]
    with pytest.raises(ValueError, match="duplicate materializer axis"):
        build_structural_axis_inputs(
            trace_context=selector_context(sources=(source, source)),
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(),
        )
