"""Tests for scanner-owned structural-axis derivation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch.nn as nn

from framework.stage1.adapters import (
    ConfigWriteTarget,
    MaterializerParameterSource,
    TraceContext,
)
from framework.stage1.structural_axes import (
    build_structural_axis_inputs,
    derive_structural_axes,
)
from framework.stage1.structural_axis_contract import (
    canonical_source_dataflow_relations,
    seal_dataflow_relation,
    seal_materializer_binding,
    source_relation_authority_payload,
    validate_dataflow_relations,
)
from framework.stage1.structural_axis_digest import canonical_digest
from tests.stage1.structural_axis_selector_test_support import (
    built_cross_interface_inputs,
    built_selector_inputs,
    selector_context,
    selector_group_manifest,
    selector_groups,
    selector_scenario,
)
from tests.stage1.structural_axis_test_support import (
    resign_structural_inputs,
)


def _trusted_selector_relations() -> tuple[dict, ...]:
    return tuple(dict(row) for row in selector_context().dataflow_relations)


def _trusted_cross_interface_relations() -> tuple[dict, ...]:
    return (
        {
            "module_root_selector": "backbone.0",
            "canonical_axis_id": "backbone.output",
            "independent_interface": True,
        },
    )


def _derive_selector_inputs(inputs: dict):
    return derive_structural_axes(
        {"structural_axis_inputs": inputs},
        trusted_source_relation_declarations=_trusted_selector_relations(),
    )


def _attach_source_dataflow_relations(inputs: dict, relations: list[dict]) -> None:
    canonical = sorted(relations, key=canonical_digest)
    inputs["source_dataflow_relations"] = canonical
    inputs["scanner_evidence"]["source_dataflow_relations"] = canonical
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    resign_structural_inputs(inputs)


def test_build_inputs_resolves_config_module_checkpoint_and_depgraph_evidence() -> None:
    inputs = build_structural_axis_inputs(
        trace_context=selector_context(),
        prune_groups=selector_groups(),
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(),
    )

    expected = [
        {
            "axis_id": "backbone.output",
            "width": 32,
            "config_path": "model.encoder.width",
            "config_width": 32,
            "checkpoint_module_path": "backbone.0",
            "checkpoint_width": 32,
            "canonical_group_id": "g0",
            "canonical_group_width": 32,
        }
    ]
    assert [
        {key: row[key] for key in expected[0]} for row in inputs["base_widths"]
    ] == expected
    assert len(inputs["base_widths"][0]["source_relation_digest"]) == 64
    assert len(inputs["base_widths"][0]["materializer_binding_digest"]) == 64
    assert inputs["materializer_bindings"][0]["b1_group_id"] == "g0"
    assert inputs["materializer_bindings"][0]["param"] == "model.encoder.width"
    assert inputs["provenance"]["checkpoint_digest"] == "b" * 64
    assert len(inputs["digest"]) == 64


def test_derive_requires_retained_source_dataflow_relations() -> None:
    inputs = built_selector_inputs()
    inputs.pop("source_dataflow_relations")
    inputs["scanner_evidence"].pop("source_dataflow_relations")
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    resign_structural_inputs(inputs)

    with pytest.raises(ValueError, match="source dataflow relations"):
        _derive_selector_inputs(inputs)


def test_derive_rejects_duplicate_source_dataflow_relation_identity() -> None:
    inputs = built_selector_inputs()
    source = dict(selector_context().dataflow_relations[0])
    _attach_source_dataflow_relations(inputs, [source, source])

    with pytest.raises(ValueError, match="duplicate source dataflow relation"):
        _derive_selector_inputs(inputs)


def test_derive_rejects_self_signed_independent_interface_tamper() -> None:
    inputs = built_selector_inputs()
    source = dict(selector_context().dataflow_relations[0])
    _attach_source_dataflow_relations(inputs, [source])
    derived = {
        key: value
        for key, value in inputs["dataflow_relations"][0].items()
        if key not in {"relation_digest", "relation_provenance"}
    }
    derived["independent_interface"] = True
    inputs["dataflow_relations"][0] = seal_dataflow_relation(derived, derived)
    resign_structural_inputs(inputs)

    with pytest.raises(
        ValueError,
        match="retained source relation|interface proof|binding relation authority",
    ):
        _derive_selector_inputs(inputs)


def test_derive_rejects_derived_relation_without_matching_raw_identity() -> None:
    inputs = built_selector_inputs()
    unrelated = {
        "module_root_selector": "backbone.0",
        "canonical_axis_id": "backbone.unrelated",
    }
    _attach_source_dataflow_relations(inputs, [unrelated])

    with pytest.raises(
        ValueError, match="retained source relation|binding relation authority"
    ):
        _derive_selector_inputs(inputs)


def test_build_inputs_requires_scanner_group_manifest_provenance() -> None:
    with pytest.raises(ValueError, match="group manifest"):
        build_structural_axis_inputs(
            trace_context=selector_context(),
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
        )


def test_build_inputs_rejects_group_manifest_without_scan_digest() -> None:
    manifest = selector_group_manifest()
    manifest.pop("scan_manifest_digest")

    with pytest.raises(ValueError, match="scan manifest digest"):
        build_structural_axis_inputs(
            trace_context=selector_context(),
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=manifest,
        )


def test_build_rejects_retained_group_tamper_against_scanner_evidence_seal() -> None:
    original = selector_groups()
    manifest = selector_group_manifest(original)
    tampered = [{**original[0], "max_rate": 0.5}]

    with pytest.raises(ValueError, match="scanner evidence seal"):
        build_structural_axis_inputs(
            trace_context=selector_context(),
            prune_groups=tampered,
            scenario=selector_scenario(),
            group_manifest=manifest,
        )


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("inputs", "scanner_input_digest"),
        ("inputs", "structural_evidence_digest"),
        ("provenance", "config_digest"),
        ("provenance", "checkpoint_digest"),
        ("provenance", "scenario_digest"),
        ("provenance", "scan_manifest_digest"),
        ("provenance", "structural_evidence_digest"),
        ("provenance", "digest_sources"),
    ],
)
def test_derive_rejects_missing_scanner_owned_digest(
    location: str, field: str
) -> None:
    inputs = built_selector_inputs()
    target = inputs if location == "inputs" else inputs["provenance"]
    target.pop(field, None)

    with pytest.raises(ValueError, match=field):
        _derive_selector_inputs(inputs)


def test_derived_axis_retains_verified_scanner_digest_provenance() -> None:
    inputs = built_selector_inputs()

    axis = _derive_selector_inputs(inputs).free_axes[0]

    for field in (
        "scanner_input_digest",
        "config_digest",
        "checkpoint_digest",
        "scenario_digest",
        "scan_manifest_digest",
        "structural_evidence_digest",
    ):
        expected = inputs.get(field, inputs["provenance"].get(field))
        assert axis.provenance[field] == expected
    assert axis.provenance["digest_sources"] == inputs["provenance"][
        "digest_sources"
    ]


def test_build_preserves_verified_independent_interface_relation() -> None:
    relation = built_cross_interface_inputs()["dataflow_relations"][0]

    assert relation["independent_interface"] is True
    assert relation["relation_provenance"]["source"] == (
        "trace_context.dataflow_relations"
    )
    assert len(relation["relation_provenance"]["source_relation_digest"]) == 64
    assert len(relation["relation_digest"]) == 64


@pytest.mark.parametrize(
    "missing_field",
    ["independent_interface", "relation_provenance", "relation_digest"],
)
def test_derive_rejects_free_cross_interface_without_verified_relation(
    missing_field: str,
) -> None:
    inputs = built_cross_interface_inputs()
    inputs["dataflow_relations"][0].pop(missing_field, None)
    resign_structural_inputs(inputs)

    with pytest.raises(ValueError, match="independent interface|relation provenance|relation digest"):
        derive_structural_axes(
            {"structural_axis_inputs": inputs},
            trusted_source_relation_declarations=_trusted_cross_interface_relations(),
        )


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
        trace_context=selector_context(sources=(source,)),
        prune_groups=selector_groups(),
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(),
    )

    assert inputs["materializer_bindings"][0]["write_targets"] == [
        {
            "selector": "model.encoder.aliases[*]",
            "transform": "repeat_to_reference_length",
            "reference_selector": "model.encoder.aliases",
        }
    ]


def _exact_depgraph_member_inputs() -> dict:
    base = selector_context()
    internal = nn.Conv2d(32, 64, 1)
    context = TraceContext(
        net=base.net,
        example_inputs=base.example_inputs,
        full_model=base.full_model,
        loaded_config=base.loaded_config,
        checkpoint_evidence={
            "digest": "b" * 64,
            "module_widths": {"backbone.0": 32, "backbone.1": 64},
        },
        materializer_sources=base.materializer_sources,
        trace_modules={**dict(base.trace_modules), "backbone.1": internal},
        dataflow_relations=(
            {
                "module_root_selector": "backbone.0",
                "canonical_axis_id": "backbone.output",
                "member_relations": (
                    {"group_id": "g0", "role": "output"},
                    {"group_id": "g1", "role": "internal"},
                ),
                "declared_member_group_ids": ("g0", "g1"),
                "member_relations_digest": canonical_digest(("g0", "g1")),
            },
        ),
    )
    groups = [
        {**selector_groups()[0], "max_rate": 0.75},
        {
            "group_id": "g1",
            "root_layer": "backbone.1",
            "cur_width": 64,
            "round_to_default": 8,
            "width_floor": 16,
            "max_rate": 0.75,
        },
    ]

    return build_structural_axis_inputs(
        trace_context=context,
        prune_groups=groups,
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(
            groups,
            source_relations=[dict(row) for row in context.dataflow_relations],
        ),
    )


def test_build_inputs_expands_exact_depgraph_member_relations() -> None:
    """Dropping a related 2C group would make materialization structurally incomplete."""
    inputs = _exact_depgraph_member_inputs()

    assert [row["group_id"] for row in inputs["prune_groups"]] == ["g0", "g1"]
    assert [
        (row["b1_group_id"], row["role"])
        for row in inputs["materializer_bindings"]
    ] == [("g0", "output"), ("g1", "internal")]


def test_explicit_member_relations_fail_when_manifest_declaration_is_incomplete() -> None:
    base = selector_context()
    relation = {
        "module_root_selector": "backbone.0",
        "canonical_axis_id": "backbone.output",
        "member_relations": ({"group_id": "g0", "role": "output"},),
        "declared_member_group_ids": ("g0", "g1"),
        "member_relations_digest": canonical_digest(("g0", "g1")),
    }
    context = TraceContext(
        net=base.net,
        example_inputs=base.example_inputs,
        full_model=base.full_model,
        loaded_config=base.loaded_config,
        checkpoint_evidence=base.checkpoint_evidence,
        materializer_sources=base.materializer_sources,
        trace_modules=base.trace_modules,
        dataflow_relations=(relation,),
    )

    with pytest.raises(ValueError, match="declared member groups"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=selector_groups(),
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(
                source_relations=[dict(row) for row in context.dataflow_relations]
            ),
        )


def test_build_inputs_infers_members_from_exact_depgraph_layer_relations() -> None:
    """Live adapter relations declare seeds; DepGraph edges supply their members."""
    base = selector_context()
    internal = nn.Conv2d(32, 64, 1)
    context = TraceContext(
        net=base.net,
        example_inputs=base.example_inputs,
        full_model=base.full_model,
        loaded_config=base.loaded_config,
        checkpoint_evidence={
            "digest": "b" * 64,
            "module_widths": {"backbone.0": 32, "backbone.1": 64},
        },
        materializer_sources=base.materializer_sources,
        trace_modules={**dict(base.trace_modules), "backbone.1": internal},
        dataflow_relations=base.dataflow_relations,
    )
    groups = [
        {
            **selector_groups()[0],
            "member_layers": ["backbone.0", "backbone.1"],
        },
        {
            "group_id": "g1",
            "root_layer": "backbone.1",
            "member_layers": ["backbone.1"],
            "cur_width": 64,
            "round_to_default": 8,
            "width_floor": 16,
            "max_rate": 0.75,
        },
    ]

    inputs = build_structural_axis_inputs(
        trace_context=context,
        prune_groups=groups,
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(groups),
    )

    assert [row["group_id"] for row in inputs["prune_groups"]] == ["g0", "g1"]


def _chain_axis_fixture(
    *, group_count: int = 4, right_index: int = 3
) -> tuple[TraceContext, list[dict], tuple[dict, ...]]:
    base = selector_context()
    sources = tuple(
        MaterializerParameterSource(
            axis_id=axis_id,
            config_selector=f"model.{config_key}",
            mutation_kind="out_channels",
            module_root_selector=module_path,
            allowed_roles=("output",),
            provenance={"adapter": "unit", "declaration": "chain endpoint"},
        )
        for axis_id, config_key, module_path in (
            ("chain.left", "left", "backbone.0"),
            ("chain.right", "right", f"backbone.{right_index}"),
        )
    )
    modules = {
        f"backbone.{index}": nn.Conv2d(32, 32, 1)
        for index in range(group_count)
    }
    relations = tuple(
        {
            "module_root_selector": source.module_root_selector,
            "canonical_axis_id": source.axis_id,
        }
        for source in sources
    )
    context = TraceContext(
        net=base.net,
        example_inputs=base.example_inputs,
        full_model=base.full_model,
        loaded_config={"model": {"left": 32, "right": 32}},
        checkpoint_evidence={
            "digest": "b" * 64,
            "module_widths": {
                module_path: 32 for module_path in modules
            },
        },
        materializer_sources=sources,
        trace_modules=modules,
        dataflow_relations=relations,
    )
    groups = [
        {
            **selector_groups()[0],
            "group_id": f"g{index}",
            "root_layer": f"backbone.{index}",
            "member_layers": [
                f"backbone.{index}",
                *(
                    [f"backbone.{index + 1}"]
                    if index < right_index
                    else []
                ),
            ],
        }
        for index in range(group_count)
    ]
    return context, groups, relations


def test_inferred_members_stop_at_the_nearest_materializer_axis() -> None:
    """A shared path belongs to its nearest declared axis, not every axis."""
    context, groups, relations = _chain_axis_fixture()

    inputs = build_structural_axis_inputs(
        trace_context=context,
        prune_groups=groups,
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(
            groups,
            source_relations=[dict(row) for row in relations],
        ),
    )

    assert [
        (row["b1_group_id"], row["axis_id"])
        for row in inputs["materializer_bindings"]
    ] == [
        ("g0", "chain.left"),
        ("g1", "chain.left"),
        ("g3", "chain.right"),
        ("g2", "chain.right"),
    ]


def _resealed_wrong_chain_partition(inputs: dict) -> tuple[list[dict], list[dict]]:
    sources = deepcopy(inputs["source_dataflow_relations"])
    members_by_axis = {
        "chain.left": (
            {"group_id": "g0", "role": "output"},
        ),
        "chain.right": (
            {"group_id": "g1", "role": "internal"},
            {"group_id": "g2", "role": "internal"},
            {"group_id": "g3", "role": "output"},
        ),
    }
    derived = []
    for source in sources:
        members = members_by_axis[source["canonical_axis_id"]]
        member_ids = tuple(sorted(row["group_id"] for row in members))
        source["member_relations"] = members
        source["declared_member_group_ids"] = member_ids
        source["member_relations_digest"] = canonical_digest(member_ids)
        derived.extend(
            seal_dataflow_relation(
                {
                    "group_id": row["group_id"],
                    "canonical_axis_id": source["canonical_axis_id"],
                },
                source,
            )
            for row in members
        )
    return sources, derived


def _built_chain_inputs() -> tuple[dict, list[dict]]:
    context, groups, relations = _chain_axis_fixture()
    inputs = build_structural_axis_inputs(
        trace_context=context,
        prune_groups=groups,
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(
            groups,
            source_relations=[dict(row) for row in relations],
        ),
    )
    return inputs, groups


def _fully_rebuilt_chain_inputs(
    *, selector_override: bool = False, declared_downgrade: bool = False
) -> tuple[dict, tuple[dict, ...]]:
    context, groups, trusted_relations = _chain_axis_fixture()
    sources = context.materializer_sources
    if selector_override:
        sources = (
            sources[0],
            replace(sources[1], module_root_selector="backbone.1"),
        )
    declarations = tuple(
        {
            "module_root_selector": source.module_root_selector,
            "canonical_axis_id": source.axis_id,
        }
        for source in sources
    )
    if declared_downgrade:
        declared_members = {
            "chain.left": ("g0", "g1"),
            "chain.right": ("g2", "g3"),
        }
        declarations = tuple(
            {
                **relation,
                "member_relations": [
                    {
                        "group_id": group_id,
                        "role": (
                            "output"
                            if group_id in {"g0", "g3"}
                            else "internal"
                        ),
                    }
                    for group_id in declared_members[relation["canonical_axis_id"]]
                ],
                "declared_member_group_ids": declared_members[
                    relation["canonical_axis_id"]
                ],
                "member_relations_digest": canonical_digest(
                    declared_members[relation["canonical_axis_id"]]
                ),
            }
            for relation in declarations
        )
        groups = [
            {key: value for key, value in group.items() if key != "member_layers"}
            for group in groups
        ]
    alternate_context = TraceContext(
        net=context.net,
        example_inputs=context.example_inputs,
        full_model=context.full_model,
        loaded_config=context.loaded_config,
        checkpoint_evidence=context.checkpoint_evidence,
        materializer_sources=sources,
        trace_modules=context.trace_modules,
        dataflow_relations=declarations,
    )
    inputs = build_structural_axis_inputs(
        trace_context=alternate_context,
        prune_groups=groups,
        scenario=selector_scenario(),
        group_manifest=selector_group_manifest(
            groups,
            source_relations=[dict(row) for row in declarations],
        ),
    )
    return inputs, trusted_relations


def _fully_resealed_wrong_chain_inputs(
    *, boundary_override: bool = False, declared_downgrade: bool = False
) -> dict:
    inputs, _ = _built_chain_inputs()
    members_by_axis = {
        "chain.left": ("g0",),
        "chain.right": ("g1", "g2", "g3"),
    }
    bindings = deepcopy(inputs["materializer_bindings"])
    moved = next(row for row in bindings if row["b1_group_id"] == "g1")
    moved.update(axis_id="chain.right", param="model.right")
    moved = seal_materializer_binding(moved)
    bindings = [
        moved if row["b1_group_id"] == "g1" else row for row in bindings
    ]
    binding_by_group = {row["b1_group_id"]: row for row in bindings}
    sources = deepcopy(inputs["source_dataflow_relations"])
    for source in sources:
        axis_id = source["canonical_axis_id"]
        member_ids = members_by_axis[axis_id]
        if boundary_override and axis_id == "chain.right":
            source["group_id"] = "g1"
        if declared_downgrade:
            source["member_relations_source"] = "adapter_declared_v1"
        source["member_relations"] = tuple(
            {
                "group_id": group_id,
                "role": binding_by_group[group_id]["role"],
            }
            for group_id in member_ids
        )
        source["declared_member_group_ids"] = member_ids
        source["member_relations_digest"] = canonical_digest(member_ids)
        source["materializer_binding_projections"] = [
            binding_by_group[group_id] for group_id in member_ids
        ]
    sources = canonical_source_dataflow_relations(sources)
    source_by_axis = {row["canonical_axis_id"]: row for row in sources}
    derived = [
        seal_dataflow_relation(
            {
                "group_id": group_id,
                "canonical_axis_id": axis_id,
                "materializer_binding_digest": binding_by_group[group_id][
                    "materializer_binding_digest"
                ],
            },
            source_by_axis[axis_id],
        )
        for axis_id, member_ids in members_by_axis.items()
        for group_id in member_ids
    ]
    bases = deepcopy(inputs["base_widths"])
    for row in bases:
        source = source_by_axis[row["axis_id"]]
        row["source_relation_digest"] = canonical_digest(source)
        row["materializer_binding_digest"] = binding_by_group[
            row["canonical_group_id"]
        ]["materializer_binding_digest"]
    if declared_downgrade:
        for group in inputs["prune_groups"]:
            group.pop("member_layers")
        inputs["scanner_evidence"]["prune_groups"] = deepcopy(
            inputs["prune_groups"]
        )
    inputs.update(
        source_dataflow_relations=sources,
        dataflow_relations=derived,
        materializer_bindings=bindings,
        base_widths=bases,
    )
    inputs["scanner_evidence"].update(
        source_dataflow_relations=sources,
        materializer_bindings=bindings,
        base_widths=bases,
    )
    if boundary_override:
        authority = deepcopy(
            inputs["scanner_evidence"]["source_relation_authority"]
        )
        declaration = next(
            row
            for row in authority["source_relations"]
            if row["canonical_axis_id"] == "chain.right"
        )
        declaration["group_id"] = "g1"
        inputs["scanner_evidence"]["source_relation_authority"] = (
            source_relation_authority_payload(
                inputs["prune_groups"],
                inputs["scanner_evidence"]["group_manifest"],
                inputs["scanner_evidence"]["scenario"],
                authority["source_relations"],
            )
        )
        inputs["provenance"]["source_group_manifest_digest"] = inputs[
            "scanner_evidence"
        ]["source_relation_authority"]["group_manifest_digest"]
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    resign_structural_inputs(inputs)
    return inputs


def test_validator_recomputes_inferred_partition_after_complete_reseal() -> None:
    inputs, groups = _built_chain_inputs()
    sources, derived = _resealed_wrong_chain_partition(inputs)

    with pytest.raises(ValueError, match="canonical inferred member partition"):
        validate_dataflow_relations(
            derived,
            sources,
            groups,
            inputs["provenance"]["source_relation_authority_schema"],
        )


def test_graph_partition_cannot_be_downgraded_to_declared_members() -> None:
    inputs, groups = _built_chain_inputs()
    sources, derived = _resealed_wrong_chain_partition(inputs)
    for source in sources:
        source["member_relations_source"] = "adapter_declared_v1"

    with pytest.raises(ValueError, match="member relation authority"):
        validate_dataflow_relations(
            derived,
            sources,
            groups,
            inputs["provenance"]["source_relation_authority_schema"],
        )


def test_full_reseal_cannot_override_selector_resolved_boundary() -> None:
    inputs = _fully_resealed_wrong_chain_inputs(boundary_override=True)

    with pytest.raises(
        ValueError,
        match="canonical group.*selector|trusted source relation declarations",
    ):
        derive_structural_axes(
            {"structural_axis_inputs": inputs},
            trusted_source_relation_declarations=_chain_axis_fixture()[2],
        )


def test_full_reseal_cannot_delete_graph_and_downgrade_to_declared() -> None:
    inputs = _fully_resealed_wrong_chain_inputs(declared_downgrade=True)

    with pytest.raises(ValueError, match="source relation authority"):
        derive_structural_axes(
            {"structural_axis_inputs": inputs},
            trusted_source_relation_declarations=_chain_axis_fixture()[2],
        )


def test_formal_derivation_requires_trusted_source_relation_declarations() -> None:
    inputs, _ = _built_chain_inputs()

    with pytest.raises(ValueError, match="trusted source relation declarations"):
        derive_structural_axes({"structural_axis_inputs": inputs})


def test_full_reseal_cannot_replace_code_side_selector_partition() -> None:
    inputs, trusted_relations = _fully_rebuilt_chain_inputs(
        selector_override=True
    )

    with pytest.raises(ValueError, match="trusted source relation declarations"):
        derive_structural_axes(
            {"structural_axis_inputs": inputs},
            trusted_source_relation_declarations=trusted_relations,
        )


def test_full_reseal_cannot_downgrade_code_side_inferred_schema() -> None:
    inputs, trusted_relations = _fully_rebuilt_chain_inputs(
        declared_downgrade=True
    )

    with pytest.raises(ValueError, match="trusted source relation declarations"):
        derive_structural_axes(
            {"structural_axis_inputs": inputs},
            trusted_source_relation_declarations=trusted_relations,
        )


def test_inferred_partition_rejects_disconnected_retained_group() -> None:
    context, groups, relations = _chain_axis_fixture(group_count=5)

    with pytest.raises(ValueError, match="retained group.*materializer axis"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=groups,
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(
                groups,
                source_relations=[dict(row) for row in relations],
            ),
        )


def test_inferred_partition_rejects_equidistant_retained_group() -> None:
    context, groups, relations = _chain_axis_fixture(
        group_count=3, right_index=2
    )

    with pytest.raises(ValueError, match="equidistant"):
        build_structural_axis_inputs(
            trace_context=context,
            prune_groups=groups,
            scenario=selector_scenario(),
            group_manifest=selector_group_manifest(
                groups,
                source_relations=[dict(row) for row in relations],
            ),
        )
