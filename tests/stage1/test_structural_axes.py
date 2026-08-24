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
    build_structural_axis_inputs,
    derive_structural_axes,
)
from framework.stage1.structural_axis_contract import (
    canonical_source_dataflow_relations,
    seal_dataflow_relation,
    seal_materializer_binding,
)
from framework.stage1.structural_axis_digest import canonical_digest


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
            "max_rate": 0.75,
        }
    ]


def _selector_scenario() -> ScanScenario:
    return ScanScenario(
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


def _scenario_evidence(scenario: ScanScenario) -> dict:
    return {
        "hardware_precisions": list(scenario.hardware_precisions),
        "backend_precisions": list(scenario.backend_precisions),
        "compression_modes": list(scenario.compression_modes),
        "graph_quant_unit_policy": {
            key: list(value)
            for key, value in sorted(scenario.graph_quant_unit_policy.items())
        },
        "alignment": dict(sorted(scenario.alignment.items())),
    }


def _scanner_evidence_seal(
    groups: list[dict],
    manifest: dict,
    scenario_evidence: dict,
    source_relations: list[dict],
) -> str:
    declaration = {
        key: manifest[key]
        for key in (
            "source_group_count",
            "declared_relevant_group_ids",
            "relevant_groups_digest",
        )
    }
    return canonical_digest(
        {
            "prune_groups": sorted(groups, key=lambda group: group["group_id"]),
            "source_dataflow_relations": canonical_source_dataflow_relations(
                source_relations
            ),
            "group_manifest": declaration,
            "scenario": scenario_evidence,
        }
    )


def _selector_group_manifest(
    groups: list[dict] | None = None,
    scenario: ScanScenario | None = None,
    source_relations: list[dict] | None = None,
) -> dict:
    evidence = groups or _selector_groups()
    group_ids = tuple(sorted(group["group_id"] for group in evidence))
    manifest = {
        "source_group_count": len(evidence),
        "declared_relevant_group_ids": group_ids,
        "relevant_groups_digest": canonical_digest(group_ids),
        "structural_evidence_digest": canonical_digest(evidence),
    }
    manifest["scan_manifest_digest"] = _scanner_evidence_seal(
        evidence,
        manifest,
        _scenario_evidence(scenario or _selector_scenario()),
        source_relations
        or [dict(row) for row in _selector_context().dataflow_relations],
    )
    return manifest


def _built_selector_inputs() -> dict:
    groups = _selector_groups()
    return build_structural_axis_inputs(
        trace_context=_selector_context(),
        prune_groups=groups,
        scenario=_selector_scenario(),
        group_manifest=_selector_group_manifest(groups),
    )


def _built_cross_interface_inputs() -> dict:
    source = MaterializerParameterSource(
        axis_id="backbone.output",
        config_selector="model.encoder.width",
        mutation_kind="out_channels",
        module_root_selector="backbone.0",
        allowed_roles=("output",),
        provenance={"adapter": "unit", "declaration": "cross-interface output"},
        write_targets=(
            ConfigWriteTarget(selector="model.neck.width", transform="identity"),
        ),
    )
    base = _selector_context(sources=(source,))
    context = TraceContext(
        net=base.net,
        example_inputs=base.example_inputs,
        full_model=base.full_model,
        loaded_config=base.loaded_config,
        checkpoint_evidence=base.checkpoint_evidence,
        materializer_sources=base.materializer_sources,
        trace_modules=base.trace_modules,
        dataflow_relations=(
            {
                "module_root_selector": "backbone.0",
                "canonical_axis_id": "backbone.output",
                "independent_interface": True,
            },
        ),
    )
    groups = _selector_groups()
    return build_structural_axis_inputs(
        trace_context=context,
        prune_groups=groups,
        scenario=_selector_scenario(),
        group_manifest=_selector_group_manifest(
            groups,
            source_relations=[dict(row) for row in context.dataflow_relations],
        ),
    )


def _resign_inputs(inputs: dict) -> None:
    inputs["structural_evidence_digest"] = canonical_digest(
        {
            key: inputs.get(key)
            for key in (
                "prune_groups",
                "source_dataflow_relations",
                "dataflow_relations",
                "materializer_bindings",
                "base_widths",
                "backend_constraints",
            )
        }
    )
    inputs["provenance"]["structural_evidence_digest"] = inputs[
        "structural_evidence_digest"
    ]
    unsigned = {key: value for key, value in inputs.items() if key not in {"digest", "scanner_input_digest"}}
    inputs["scanner_input_digest"] = canonical_digest(unsigned)
    inputs["digest"] = inputs["scanner_input_digest"]


def _attach_source_dataflow_relations(inputs: dict, relations: list[dict]) -> None:
    canonical = sorted(relations, key=canonical_digest)
    inputs["source_dataflow_relations"] = canonical
    inputs["scanner_evidence"]["source_dataflow_relations"] = canonical
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    _resign_inputs(inputs)


def _set_scanner_owned_group_width(
    inputs: dict, group_id: str, width: int
) -> None:
    outer_group = next(
        group
        for group in inputs["prune_groups"]
        if group.get("group_id", group.get("id")) == group_id
    )
    outer_group["cur_width"] = width
    sealed_group = next(
        group
        for group in inputs["scanner_evidence"]["prune_groups"]
        if group.get("group_id", group.get("id")) == group_id
    )
    sealed_group["cur_width"] = width
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    _resign_inputs(inputs)


def _set_scanner_owned_binding_fields(
    inputs: dict, index: int, **changes: object
) -> None:
    outer = inputs["materializer_bindings"][index]
    group_id = outer["b1_group_id"]
    sealed = next(
        row
        for row in inputs["scanner_evidence"]["materializer_bindings"]
        if row["b1_group_id"] == group_id
    )
    binding = seal_materializer_binding({**sealed, **changes})
    outer.clear()
    outer.update(binding)
    sealed.clear()
    sealed.update(binding)
    canonical_sources = _update_scanner_binding_relation(
        inputs, group_id, binding
    )
    _update_base_binding_seals(inputs, group_id, binding, canonical_sources)
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    _resign_inputs(inputs)


def _update_scanner_binding_relation(
    inputs: dict, group_id: str, binding: dict
) -> list[dict]:
    source = next(
        row
        for row in inputs["source_dataflow_relations"]
        if group_id in row["declared_member_group_ids"]
    )
    source.update(
        {
            "canonical_axis_id": binding["axis_id"],
            "axis_kind": binding["axis_kind"],
            "materializer_binding_projections": [binding],
        }
    )
    source["member_relations"] = [
        {"group_id": group_id, "role": binding["role"]}
    ]
    if "derived_from" in binding:
        source["derived_from"] = binding["derived_from"]
    else:
        source.pop("derived_from", None)
    canonical_sources = canonical_source_dataflow_relations(
        inputs["source_dataflow_relations"]
    )
    inputs["source_dataflow_relations"] = canonical_sources
    inputs["scanner_evidence"]["source_dataflow_relations"] = canonical_sources
    derived = next(
        row for row in inputs["dataflow_relations"] if row["group_id"] == group_id
    )
    unsigned = {
        key: value
        for key, value in derived.items()
        if key not in {"relation_digest", "relation_provenance"}
    }
    unsigned.update(
        canonical_axis_id=binding["axis_id"],
        axis_kind=binding["axis_kind"],
        materializer_binding_digest=binding["materializer_binding_digest"],
    )
    if "derived_from" in binding:
        unsigned["derived_from"] = binding["derived_from"]
    else:
        unsigned.pop("derived_from", None)
    derived.clear()
    derived.update(seal_dataflow_relation(unsigned, source))
    return canonical_sources


def _update_base_binding_seals(
    inputs: dict,
    group_id: str,
    binding: dict,
    canonical_sources: list[dict],
) -> None:
    canonical_source = next(
        row for row in canonical_sources if row["canonical_group_id"] == group_id
    )
    for rows in (
        inputs["base_widths"],
        inputs["scanner_evidence"]["base_widths"],
    ):
        base = next(row for row in rows if row["canonical_group_id"] == group_id)
        base["source_relation_digest"] = canonical_digest(canonical_source)
        base["materializer_binding_digest"] = binding[
            "materializer_binding_digest"
        ]


def test_build_inputs_resolves_config_module_checkpoint_and_depgraph_evidence() -> None:
    inputs = build_structural_axis_inputs(
        trace_context=_selector_context(),
        prune_groups=_selector_groups(),
        scenario=_selector_scenario(),
        group_manifest=_selector_group_manifest(),
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
    inputs = _built_selector_inputs()
    inputs.pop("source_dataflow_relations")
    inputs["scanner_evidence"].pop("source_dataflow_relations")
    inputs["provenance"]["scan_manifest_digest"] = canonical_digest(
        inputs["scanner_evidence"]
    )
    _resign_inputs(inputs)

    with pytest.raises(ValueError, match="source dataflow relations"):
        derive_structural_axes({"structural_axis_inputs": inputs})


def test_derive_rejects_duplicate_source_dataflow_relation_identity() -> None:
    inputs = _built_selector_inputs()
    source = dict(_selector_context().dataflow_relations[0])
    _attach_source_dataflow_relations(inputs, [source, source])

    with pytest.raises(ValueError, match="duplicate source dataflow relation"):
        derive_structural_axes({"structural_axis_inputs": inputs})


def test_derive_rejects_self_signed_independent_interface_tamper() -> None:
    inputs = _built_selector_inputs()
    source = dict(_selector_context().dataflow_relations[0])
    _attach_source_dataflow_relations(inputs, [source])
    derived = {
        key: value
        for key, value in inputs["dataflow_relations"][0].items()
        if key not in {"relation_digest", "relation_provenance"}
    }
    derived["independent_interface"] = True
    inputs["dataflow_relations"][0] = seal_dataflow_relation(derived, derived)
    _resign_inputs(inputs)

    with pytest.raises(
        ValueError,
        match="retained source relation|interface proof|binding relation authority",
    ):
        derive_structural_axes({"structural_axis_inputs": inputs})


def test_derive_rejects_derived_relation_without_matching_raw_identity() -> None:
    inputs = _built_selector_inputs()
    unrelated = {
        "module_root_selector": "backbone.0",
        "canonical_axis_id": "backbone.unrelated",
    }
    _attach_source_dataflow_relations(inputs, [unrelated])

    with pytest.raises(
        ValueError, match="retained source relation|binding relation authority"
    ):
        derive_structural_axes({"structural_axis_inputs": inputs})


def test_build_inputs_requires_scanner_group_manifest_provenance() -> None:
    with pytest.raises(ValueError, match="group manifest"):
        build_structural_axis_inputs(
            trace_context=_selector_context(),
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
        )


def test_build_inputs_rejects_group_manifest_without_scan_digest() -> None:
    manifest = _selector_group_manifest()
    manifest.pop("scan_manifest_digest")

    with pytest.raises(ValueError, match="scan manifest digest"):
        build_structural_axis_inputs(
            trace_context=_selector_context(),
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
            group_manifest=manifest,
        )


def test_build_rejects_retained_group_tamper_against_scanner_evidence_seal() -> None:
    original = _selector_groups()
    manifest = _selector_group_manifest(original)
    tampered = [{**original[0], "max_rate": 0.5}]

    with pytest.raises(ValueError, match="scanner evidence seal"):
        build_structural_axis_inputs(
            trace_context=_selector_context(),
            prune_groups=tampered,
            scenario=_selector_scenario(),
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
    inputs = _built_selector_inputs()
    target = inputs if location == "inputs" else inputs["provenance"]
    target.pop(field, None)

    with pytest.raises(ValueError, match=field):
        derive_structural_axes({"structural_axis_inputs": inputs})


def test_derived_axis_retains_verified_scanner_digest_provenance() -> None:
    inputs = _built_selector_inputs()

    axis = derive_structural_axes({"structural_axis_inputs": inputs}).free_axes[0]

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
    relation = _built_cross_interface_inputs()["dataflow_relations"][0]

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
    inputs = _built_cross_interface_inputs()
    inputs["dataflow_relations"][0].pop(missing_field, None)
    _resign_inputs(inputs)

    with pytest.raises(ValueError, match="independent interface|relation provenance|relation digest"):
        derive_structural_axes({"structural_axis_inputs": inputs})


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
        group_manifest=_selector_group_manifest(),
    )

    assert inputs["materializer_bindings"][0]["write_targets"] == [
        {
            "selector": "model.encoder.aliases[*]",
            "transform": "repeat_to_reference_length",
            "reference_selector": "model.encoder.aliases",
        }
    ]


def _exact_depgraph_member_inputs() -> dict:
    base = _selector_context()
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
        {**_selector_groups()[0], "max_rate": 0.75},
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
        scenario=_selector_scenario(),
        group_manifest=_selector_group_manifest(
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
    base = _selector_context()
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
            prune_groups=_selector_groups(),
            scenario=_selector_scenario(),
            group_manifest=_selector_group_manifest(
                source_relations=[dict(row) for row in context.dataflow_relations]
            ),
        )


def test_build_inputs_infers_members_from_exact_depgraph_layer_relations() -> None:
    """Live adapter relations declare seeds; DepGraph edges supply their members."""
    base = _selector_context()
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
            **_selector_groups()[0],
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
        scenario=_selector_scenario(),
        group_manifest=_selector_group_manifest(groups),
    )

    assert [row["group_id"] for row in inputs["prune_groups"]] == ["g0", "g1"]
