"""Paper-model scanner evidence and structural authority workflows."""

from __future__ import annotations

import pytest
import yaml

from framework.stage1.structural_axes import axis_to_scanner_dict, derive_structural_axes
from framework.stage1.structural_axis_contract import (
    formal_scanner_evidence_digest,
    scanner_group_evidence_payload,
    seal_dataflow_relation,
    seal_scanner_inputs,
    validate_scanner_inputs,
)
from framework.stage1.structural_axis_digest import canonical_digest
from tests.stage1.test_trace_graph_adapter_workflow import (
    _paper_axis_bundle,
    _paper_scanner_evidence,
)

@pytest.mark.parametrize(
    ("name", "free_axis_ids", "legal_widths", "fixed_axis_ids"),
    [
        (
            "pyramid",
            ["backbone.s0", "backbone.s1", "backbone.s2"],
            [
                [16, 24, 32, 40, 48, 56, 64],
                [32, 48, 64, 80, 96, 112, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            ["neck.output"],
        ),
        (
            "codriving",
            ["backbone.s0", "backbone.s1", "backbone.s2"],
            [
                [16, 24, 32, 40, 48, 56, 64],
                [32, 48, 64, 80, 96, 112, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            ["neck.output"],
        ),
        (
            "fcooper",
            [
                "backbone.s0",
                "backbone.s1",
                "backbone.s2",
                "neck.deblock",
                "neck.output",
            ],
            [
                [32, 64],
                [32, 64, 96, 128],
                [32, 64, 96, 128, 160, 192, 224, 256],
                [32, 64, 96, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            [],
        ),
    ],
)
def test_paper_scanner_evidence_derives_formal_axes(
    name: str,
    free_axis_ids: list[str],
    legal_widths: list[list[int]],
    fixed_axis_ids: list[str],
) -> None:
    evidence, inputs, bundle = _paper_axis_bundle(name)
    _assert_paper_axes(evidence, inputs, bundle, free_axis_ids, legal_widths, fixed_axis_ids)
    _assert_paper_base_authority(inputs)
    assert len(inputs["digest"]) == 64


def _assert_paper_axes(
    evidence: dict,
    inputs: dict,
    bundle: object,
    free_axis_ids: list[str],
    legal_widths: list[list[int]],
    fixed_axis_ids: list[str],
) -> None:

    assert [axis.axis_id for axis in bundle.free_axes] == free_axis_ids
    assert [list(axis.legal_widths) for axis in bundle.free_axes] == legal_widths
    assert [axis.axis_id for axis in bundle.fixed_axes] == fixed_axis_ids
    member_ids = [
        member.b1_group_id for axis in bundle.axes for member in axis.member_b1_groups
    ]
    evidence_ids = [group["group_id"] for group in evidence["prune_groups"]]
    assert len(member_ids) == len(evidence_ids)
    assert set(member_ids) == set(evidence_ids)
    scanner_axes = [axis_to_scanner_dict(axis, inputs) for axis in bundle.axes]
    assert all(axis["provenance"]["input_digest"] == inputs["digest"] for axis in scanner_axes)
    scanner_evidence = inputs["scanner_evidence"]
    assert inputs["materializer_bindings"] == scanner_evidence[
        "materializer_bindings"
    ]
    assert inputs["base_widths"] == scanner_evidence["base_widths"]
    assert inputs["provenance"]["scan_manifest_digest"] == (
        formal_scanner_evidence_digest(scanner_evidence)
    )
    assert scanner_evidence["base_width_provenance"] == {
        "source": "trace_context.config_checkpoint_depgraph",
        "config_digest": inputs["provenance"]["config_digest"],
        "checkpoint_digest": inputs["provenance"]["checkpoint_digest"],
    }


def _assert_paper_base_authority(inputs: dict) -> None:
    assert all(
        len(
            {
                row["width"],
                row["config_width"],
                row["checkpoint_width"],
                row["canonical_group_width"],
            }
        )
        == 1
        for row in inputs["base_widths"]
    )
    sources = {
        row["canonical_axis_id"]: row
        for row in inputs["source_dataflow_relations"]
    }
    groups = {row["group_id"]: row for row in inputs["prune_groups"]}
    bindings = {
        row["b1_group_id"]: row for row in inputs["materializer_bindings"]
    }
    for row in inputs["base_widths"]:
        source = sources[row["axis_id"]]
        group_id = row["canonical_group_id"]
        assert group_id == source["canonical_group_id"]
        assert row["checkpoint_module_path"] == groups[group_id]["root_layer"]
        assert row["source_relation_digest"] == canonical_digest(source)
        assert row["config_path"] == bindings[group_id]["param"]
        assert row["materializer_binding_digest"] == bindings[group_id][
            "materializer_binding_digest"
        ]


@pytest.mark.parametrize("name", ["pyramid", "codriving", "fcooper"])
def test_paper_scanner_fixtures_contain_only_purified_input_evidence(name: str) -> None:
    evidence = _paper_scanner_evidence(name)
    serialized = yaml.safe_dump(evidence)

    assert evidence["schema"] == "paper_scanner_evidence_v1"
    assert "/home/" not in serialized
    assert "legal_widths" not in serialized
    assert "expected_" not in serialized
    assert "formal_axis" not in serialized
    assert "scanner_structural_axes" not in serialized


@pytest.mark.parametrize("name", ["pyramid", "codriving", "fcooper"])
def test_paper_scanner_fixture_digest_seals_retained_formal_evidence(name: str) -> None:
    evidence = _paper_scanner_evidence(name)
    manifest = evidence["source_provenance"]
    declaration = {
        key: manifest[key]
        for key in (
            "source_group_count",
            "declared_relevant_group_ids",
            "relevant_groups_digest",
        )
    }
    payload = scanner_group_evidence_payload(
        evidence["prune_groups"],
        declaration,
        evidence["scan_scenario"],
        evidence["dataflow_relations"],
    )

    assert manifest["scan_manifest_digest"] == formal_scanner_evidence_digest(payload)


def test_paper_group_manifest_rejects_incomplete_purified_evidence() -> None:
    evidence = _paper_scanner_evidence("codriving")
    evidence["prune_groups"].pop()

    with pytest.raises(ValueError, match="declared relevant groups"):
        _paper_axis_bundle("codriving", evidence)


def test_scanner_axis_bundle_keeps_paper_provenance_immutable() -> None:
    _, _, bundle = _paper_axis_bundle("pyramid")

    with pytest.raises(TypeError):
        bundle.axes[0].provenance["source"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        bundle.diagnostics["axis_count"] = 0  # type: ignore[index]


def test_fcooper_neck_requires_graph_independence_evidence() -> None:
    evidence = _paper_scanner_evidence("fcooper")
    evidence["dataflow_relations"][3].pop("independent_interface")

    with pytest.raises(ValueError, match="independent interface|scanner evidence seal"):
        _paper_axis_bundle("fcooper", evidence)


def test_formal_axis_identity_ignores_scanner_evidence_serialization_order() -> None:
    evidence = _paper_scanner_evidence("pyramid")
    _, first_inputs, first_bundle = _paper_axis_bundle("pyramid", evidence)
    evidence["prune_groups"].reverse()
    evidence["dataflow_relations"].reverse()
    for relation in evidence["dataflow_relations"]:
        relation["member_relations"].reverse()

    _, second_inputs, second_bundle = _paper_axis_bundle("pyramid", evidence)

    assert second_bundle == first_bundle
    assert second_inputs["digest"] == first_inputs["digest"]
    assert second_inputs["prune_groups"] == first_inputs["prune_groups"]
    assert first_inputs.get("source_dataflow_relations")
    assert second_inputs["source_dataflow_relations"] == first_inputs[
        "source_dataflow_relations"
    ]


def test_derive_rejects_group_outside_matched_raw_relation_members() -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    derived = inputs["dataflow_relations"][0]
    source_digest = derived["relation_provenance"]["source_relation_digest"]
    source = next(
        relation
        for relation in inputs["source_dataflow_relations"]
        if canonical_digest(relation) == source_digest
    )
    unsigned = {
        key: value
        for key, value in derived.items()
        if key not in {"relation_digest", "relation_provenance"}
    }
    unsigned["group_id"] = "not-a-retained-member"
    inputs["dataflow_relations"][0] = seal_dataflow_relation(unsigned, source)
    with pytest.raises(
        ValueError,
        match="member group|retained prune group|binding relation authority",
    ):
        derive_structural_axes(
            {"structural_axis_inputs": _resign_structural_inputs(inputs)}
        )


def test_derive_rejects_retained_group_owned_by_another_source_relation() -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    derived = inputs["dataflow_relations"][0]
    source_digest = derived["relation_provenance"]["source_relation_digest"]
    source = next(
        relation
        for relation in inputs["source_dataflow_relations"]
        if canonical_digest(relation) == source_digest
    )
    foreign_group_id = next(
        group["group_id"]
        for group in inputs["prune_groups"]
        if group["group_id"] not in source["declared_member_group_ids"]
    )
    unsigned = {
        key: value
        for key, value in derived.items()
        if key not in {"relation_digest", "relation_provenance"}
    }
    inputs["dataflow_relations"][0] = seal_dataflow_relation(
        {**unsigned, "group_id": foreign_group_id}, source
    )

    with pytest.raises(
        ValueError, match="source member group|binding relation authority"
    ):
        derive_structural_axes(
            {"structural_axis_inputs": _resign_structural_inputs(inputs)}
        )


@pytest.mark.parametrize("field", ["cur_width", "module_path"])
def test_derive_rejects_outer_prune_group_field_drift(field: str) -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    group = inputs["prune_groups"][0]
    group[field] = (
        int(group[field]) * 2 if field == "cur_width" else "tampered.module"
    )

    with pytest.raises(ValueError, match="retained prune group evidence"):
        derive_structural_axes(
            {"structural_axis_inputs": _resign_structural_inputs(inputs)}
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_validate_rejects_outer_prune_group_membership_drift(mutation: str) -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    if mutation == "missing":
        inputs["prune_groups"].pop()
    else:
        inputs["prune_groups"].append(dict(inputs["prune_groups"][0]))

    with pytest.raises(ValueError, match="retained prune group (evidence|identity)"):
        validate_scanner_inputs(_resign_structural_inputs(inputs))


def test_outer_prune_group_order_is_not_an_authority_difference() -> None:
    _, inputs, expected = _paper_axis_bundle("pyramid")
    inputs["prune_groups"].reverse()

    actual = derive_structural_axes(
        {"structural_axis_inputs": _resign_structural_inputs(inputs)}
    )

    assert actual == expected


def test_derive_rejects_self_signed_noncanonical_outer_group_order() -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    inputs["prune_groups"].reverse()

    with pytest.raises(ValueError, match="canonical retained prune group order"):
        derive_structural_axes(
            {
                "structural_axis_inputs": _resign_preserving_group_order(
                    inputs
                )
            }
        )


@pytest.mark.parametrize(
    "field", ["axis_id", "axis_kind", "role", "param", "write_targets"]
)
def test_validate_rejects_binding_tamper_against_relation_authority(
    field: str,
) -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    binding = inputs["materializer_bindings"][0]
    sealed = inputs["scanner_evidence"]["materializer_bindings"][0]
    if field == "write_targets":
        value = [
            {
                "selector": binding["param"],
                "transform": "tampered",
                "reference_selector": None,
            }
        ]
    elif field == "axis_id":
        value = "backbone.s1"
    elif field == "axis_kind":
        value = "fixed_derived"
    else:
        value = f"{binding[field]}.tampered"
    binding[field] = value
    sealed[field] = value
    inputs["provenance"]["scan_manifest_digest"] = (
        formal_scanner_evidence_digest(inputs["scanner_evidence"])
    )

    with pytest.raises(ValueError, match="binding relation authority"):
        validate_scanner_inputs(
            _resign_structural_inputs(inputs)
        )


def test_validate_rejects_missing_binding_to_relation_identity() -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    inputs["materializer_bindings"].pop()
    inputs["scanner_evidence"]["materializer_bindings"].pop()
    inputs["provenance"]["scan_manifest_digest"] = (
        formal_scanner_evidence_digest(inputs["scanner_evidence"])
    )

    with pytest.raises(ValueError, match="binding relation authority"):
        validate_scanner_inputs(_resign_structural_inputs(inputs))


def test_derive_rejects_self_signed_base_width_tamper() -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    inputs["base_widths"][0]["width"] //= 2

    with pytest.raises(ValueError, match="base width authority"):
        derive_structural_axes(
            {"structural_axis_inputs": _resign_structural_inputs(inputs)}
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "same_axis_internal_member",
        "checkpoint_path",
        "config_path",
        "source_digest",
        "binding_digest",
    ],
)
def test_validate_rejects_base_proof_coordinate_tamper(mutation: str) -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    row = inputs["base_widths"][0]
    sealed = inputs["scanner_evidence"]["base_widths"][0]
    changes: dict[str, object]
    if mutation == "same_axis_internal_member":
        internal = next(
            group
            for group in inputs["prune_groups"]
            if group["group_id"] == "g9"
        )
        changes = {
            "canonical_group_id": internal["group_id"],
            "checkpoint_module_path": internal["root_layer"],
            "width": internal["cur_width"],
            "config_width": internal["cur_width"],
            "checkpoint_width": internal["cur_width"],
            "canonical_group_width": internal["cur_width"],
        }
    elif mutation == "checkpoint_path":
        changes = {"checkpoint_module_path": "tampered.module"}
    elif mutation == "config_path":
        changes = {"config_path": "tampered.config"}
    elif mutation == "source_digest":
        changes = {"source_relation_digest": "a" * 64}
    else:
        changes = {"materializer_binding_digest": "b" * 64}
    row.update(changes)
    sealed.update(changes)
    inputs["provenance"]["scan_manifest_digest"] = (
        formal_scanner_evidence_digest(inputs["scanner_evidence"])
    )

    with pytest.raises(ValueError, match="base width authority"):
        validate_scanner_inputs(_resign_structural_inputs(inputs))


def test_validate_rejects_foreign_member_as_canonical_base() -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    row = inputs["base_widths"][0]
    sealed = inputs["scanner_evidence"]["base_widths"][0]
    foreign = next(
        group for group in inputs["prune_groups"] if group["group_id"] == "g15"
    )
    for target in (row, sealed):
        target.update(
            canonical_group_id=foreign["group_id"],
            checkpoint_module_path=foreign["root_layer"],
            width=foreign["cur_width"],
            config_width=foreign["cur_width"],
            checkpoint_width=foreign["cur_width"],
            canonical_group_width=foreign["cur_width"],
        )
    inputs["provenance"]["scan_manifest_digest"] = (
        formal_scanner_evidence_digest(inputs["scanner_evidence"])
    )

    with pytest.raises(ValueError, match="base width authority"):
        validate_scanner_inputs(_resign_structural_inputs(inputs))


@pytest.mark.parametrize("collection", ["materializer_bindings", "base_widths"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order"])
def test_validate_rejects_structural_authority_alias_drift(
    collection: str, mutation: str
) -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    rows = inputs[collection]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows.reverse()

    with pytest.raises(
        ValueError, match="materializer binding authority|base width authority"
    ):
        validate_scanner_inputs(_resign_structural_inputs(inputs))


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_derive_requires_one_relation_per_declared_source_member(mutation: str) -> None:
    _, inputs, _ = _paper_axis_bundle("pyramid")
    source = next(
        relation
        for relation in inputs["source_dataflow_relations"]
        if len(relation["declared_member_group_ids"]) > 1
    )
    source_digest = canonical_digest(source)
    indexes = [
        index
        for index, relation in enumerate(inputs["dataflow_relations"])
        if relation["relation_provenance"]["source_relation_digest"]
        == source_digest
    ]
    if mutation == "missing":
        inputs["dataflow_relations"].pop(indexes[-1])
    else:
        inputs["dataflow_relations"].append(
            dict(inputs["dataflow_relations"][indexes[-1]])
        )

    with pytest.raises(ValueError, match="complete|duplicate"):
        derive_structural_axes(
            {"structural_axis_inputs": _resign_structural_inputs(inputs)}
        )


def test_pyramid_multi_member_relation_keeps_exact_declared_group_set() -> None:
    _, inputs, bundle = _paper_axis_bundle("pyramid")
    source = next(
        relation
        for relation in inputs["source_dataflow_relations"]
        if relation["canonical_axis_id"] == "backbone.s0"
    )
    source_digest = canonical_digest(source)
    expected = {"g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14"}
    emitted = {
        relation["group_id"]
        for relation in inputs["dataflow_relations"]
        if relation["relation_provenance"]["source_relation_digest"]
        == source_digest
    }
    axis = next(axis for axis in bundle.axes if axis.axis_id == "backbone.s0")

    assert set(source["declared_member_group_ids"]) == expected
    assert emitted == expected
    assert set(axis.provenance["dataflow_group_ids"]) == expected


def _resign_structural_inputs(inputs: dict) -> dict:
    payload = {
        key: value
        for key, value in inputs.items()
        if key not in {"digest", "scanner_input_digest", "structural_evidence_digest"}
    }
    return seal_scanner_inputs(payload)


def _resign_preserving_group_order(inputs: dict) -> dict:
    payload = {
        key: value
        for key, value in inputs.items()
        if key not in {"digest", "scanner_input_digest", "structural_evidence_digest"}
    }
    structural = {
        key: payload.get(key)
        for key in (
            "prune_groups",
            "source_dataflow_relations",
            "dataflow_relations",
            "materializer_bindings",
            "base_widths",
            "backend_constraints",
        )
    }
    structural_digest = canonical_digest(structural)
    payload["structural_evidence_digest"] = structural_digest
    payload["provenance"] = {
        **payload["provenance"],
        "structural_evidence_digest": structural_digest,
    }
    scanner_input_digest = canonical_digest(payload)
    return {
        **payload,
        "scanner_input_digest": scanner_input_digest,
        "digest": scanner_input_digest,
    }
