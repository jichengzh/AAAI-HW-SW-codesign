"""Integrity checks for scanner-owned structural-axis evidence."""

from __future__ import annotations

import fnmatch
from typing import Any, Mapping, Sequence

from framework.stage1.structural_axis_binding_contract import (
    canonical_materializer_binding,
    seal_materializer_binding,
    validate_binding_relation_authority,
)
from framework.stage1.structural_axis_base_contract import (
    validate_base_width_authority,
)
from framework.stage1.structural_axis_digest import (
    SCANNER_AXIS_PROVENANCE_SOURCE,
    canonical_digest,
)
from framework.stage1.structural_axis_partition import canonical_member_partition

_STRUCTURAL_KEYS = (
    "prune_groups",
    "source_dataflow_relations",
    "dataflow_relations",
    "materializer_bindings",
    "base_widths",
    "backend_constraints",
)
_PROVENANCE_DIGESTS = (
    "config_digest",
    "checkpoint_digest",
    "scenario_digest",
    "scan_manifest_digest",
    "structural_evidence_digest",
)
_INFERRED_MEMBER_SOURCE = "scanner_inferred_nearest_boundary_v1"
_DECLARED_MEMBER_SOURCE = "adapter_declared_v1"
_INFERRED_AUTHORITY_SCHEMA = "scanner_retained_graph_inference_v1"
_DECLARED_AUTHORITY_SCHEMA = "adapter_declared_member_relations_v1"
_DIGEST_SOURCES = {
    "config_digest": "trace_context.loaded_config",
    "checkpoint_digest": "trace_context.checkpoint_evidence",
    "scenario_digest": "scan_scenario",
    "scan_manifest_digest": "scanner_group_manifest",
    "structural_evidence_digest": "canonical_structural_axis_inputs",
}
_BASE_WIDTH_AUTHORITY_SOURCE = "trace_context.config_checkpoint_depgraph"


def scanner_digest_sources() -> dict[str, str]:
    return {
        field: source
        for field, source in _DIGEST_SOURCES.items()
        if field != "structural_evidence_digest"
    }


def axis_provenance(
    axis_id: str,
    axis_kind: str,
    bindings: Sequence[Mapping[str, Any]],
    dataflow_group_ids: tuple[str, ...],
    scanner_provenance: Mapping[str, Any],
    scanner_input_digest: str,
) -> dict[str, Any]:
    provenance = {
        "source": scanner_provenance["source"],
        "scanner_input_digest": scanner_input_digest,
        "input_digest": scanner_input_digest,
        "dataflow_group_ids": dataflow_group_ids,
        **{
            field: value
            for field, value in scanner_provenance.items()
            if field.endswith("_digest")
            and field != "source_group_manifest_digest"
        },
        "digest_sources": scanner_provenance["digest_sources"],
    }
    if axis_kind != "fixed_derived":
        return provenance
    derived_from = {
        _required_text(binding.get("derived_from"), "fixed-derived derived_from")
        for binding in bindings
    }
    if len(derived_from) != 1:
        raise ValueError(f"fixed-derived axis {axis_id} must have one derived_from source")
    return {**provenance, "derived_from": next(iter(derived_from))}


def validate_axis_provenance(
    axis: Mapping[str, Any], inputs: Mapping[str, Any]
) -> None:
    source = validate_scanner_inputs(inputs)
    if axis.get("scanner_input_digest") != inputs["scanner_input_digest"]:
        raise ValueError("axis scanner_input_digest disagrees with scanner inputs")
    if any(
        axis.get(field) != value
        for field, value in source.items()
        if field.endswith("_digest")
        and field != "source_group_manifest_digest"
    ):
        raise ValueError("axis provenance disagrees with scanner inputs")
    if axis.get("digest_sources") != source["digest_sources"]:
        raise ValueError("axis digest_sources disagree with scanner inputs")


def scanner_scenario_payload(scenario: Any) -> dict[str, Any]:
    """Return the exact path-free scenario evidence used by formal derivation."""

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


def formal_scanner_evidence_payload(
    groups: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    scenario: Mapping[str, Any],
    source_relations: Sequence[Mapping[str, Any]],
    materializer_bindings: Sequence[Mapping[str, Any]],
    base_widths: Sequence[Mapping[str, Any]],
    base_width_provenance: Mapping[str, Any],
    source_relation_declarations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Canonical payload sealed by ``scan_manifest_digest``.

    The seal covers only retained scanner evidence that drives formal axes; it is
    deliberately independent of diagnostic anchors and filesystem metadata.
    """

    payload = scanner_group_evidence_payload(
        groups, manifest, scenario, source_relations
    )
    return {
        **payload,
        "materializer_bindings": [dict(row) for row in materializer_bindings],
        "base_widths": [dict(row) for row in base_widths],
        "base_width_provenance": dict(base_width_provenance),
        "source_relation_authority": source_relation_authority_payload(
            groups, manifest, scenario, source_relation_declarations
        ),
    }


def scanner_group_evidence_payload(
    groups: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    scenario: Mapping[str, Any],
    source_relations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return pre-resolution evidence used to authenticate a group manifest."""

    declaration = {
        field: manifest[field]
        for field in (
            "source_group_count",
            "declared_relevant_group_ids",
            "relevant_groups_digest",
        )
    }
    return {
        "prune_groups": sorted(
            (dict(group) for group in groups),
            key=lambda group: str(group.get("group_id") or group.get("id") or ""),
        ),
        "source_dataflow_relations": canonical_source_dataflow_relations(
            source_relations
        ),
        "group_manifest": declaration,
        "scenario": dict(scenario),
    }


def formal_scanner_evidence_digest(evidence: Mapping[str, Any]) -> str:
    """Recompute the formal scanner-evidence authority seal."""

    return canonical_digest(evidence)


def scanner_group_manifest(
    groups: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
    source_relations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Declare and seal the exact scanner group set used for formal derivation."""

    group_ids = tuple(sorted(_required_text(row.get("group_id"), "group_id") for row in groups))
    manifest = {
        "source_group_count": len(groups),
        "declared_relevant_group_ids": group_ids,
        "relevant_groups_digest": canonical_digest(group_ids),
    }
    evidence = scanner_group_evidence_payload(
        groups, manifest, scenario, source_relations
    )
    return {**manifest, "scan_manifest_digest": formal_scanner_evidence_digest(evidence)}


def source_relation_authority_payload(
    groups: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    scenario: Mapping[str, Any],
    source_relations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain the pre-resolution relation schema sealed by the group manifest."""

    declarations = canonical_source_dataflow_relations(source_relations)
    declared = tuple(row.get("member_relations") is not None for row in declarations)
    if any(declared) and not all(declared):
        raise ValueError("declared and inferred source relations cannot mix")
    schema = (
        _DECLARED_AUTHORITY_SCHEMA if all(declared) else _INFERRED_AUTHORITY_SCHEMA
    )
    return {
        "schema": schema,
        "source_relations": declarations,
        "group_manifest_digest": formal_scanner_evidence_digest(
            scanner_group_evidence_payload(groups, manifest, scenario, declarations)
        ),
    }


def seal_scanner_inputs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach canonical structural and whole-input digests to scanner evidence."""

    sealed = _canonical_scanner_payload(payload)
    _validate_structural_authority_aliases(sealed)
    structural_digest = _structural_digest(sealed)
    provenance = dict(_required_mapping(sealed.get("provenance"), "provenance"))
    provenance["structural_evidence_digest"] = structural_digest
    sources = dict(_required_mapping(provenance.get("digest_sources"), "digest_sources"))
    sources["structural_evidence_digest"] = _DIGEST_SOURCES[
        "structural_evidence_digest"
    ]
    provenance["digest_sources"] = sources
    sealed["provenance"] = provenance
    sealed["structural_evidence_digest"] = structural_digest
    input_digest = canonical_digest(_canonical_scanner_payload(sealed))
    sealed["scanner_input_digest"] = input_digest
    sealed["digest"] = input_digest
    return sealed


def validate_scanner_inputs(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed unless all scanner provenance and digests are canonical."""

    provenance = _required_mapping(inputs.get("provenance"), "provenance")
    if provenance.get("source") != SCANNER_AXIS_PROVENANCE_SOURCE:
        raise ValueError("scanner provenance source is invalid")
    for field in _PROVENANCE_DIGESTS:
        _sha256(provenance.get(field), field)
    if provenance.get("digest_sources") != _DIGEST_SOURCES:
        raise ValueError("digest_sources provenance is invalid")
    evidence = _required_mapping(inputs.get("scanner_evidence"), "scanner_evidence")
    retained_groups = validated_retained_groups(inputs)
    bindings = validated_materializer_bindings(inputs)
    validated_base_widths(inputs, retained_groups, bindings)
    source_authority = _validated_source_relation_authority(
        evidence, retained_groups, provenance
    )
    _validate_retained_source_relations(
        inputs,
        evidence,
        source_authority["source_relations"],
        source_authority["schema"],
    )
    if formal_scanner_evidence_digest(evidence) != provenance["scan_manifest_digest"]:
        raise ValueError("formal scanner evidence seal mismatch")
    scenario = _required_mapping(evidence.get("scenario"), "scanner evidence scenario")
    if canonical_digest(scenario) != provenance["scenario_digest"]:
        raise ValueError("scanner evidence scenario digest mismatch")
    structural_digest = _sha256(
        inputs.get("structural_evidence_digest"), "structural_evidence_digest"
    )
    if structural_digest != provenance["structural_evidence_digest"]:
        raise ValueError("structural_evidence_digest provenance mismatch")
    if structural_digest != _structural_digest(inputs):
        raise ValueError("structural_evidence_digest is not canonical")
    scanner_digest = _sha256(inputs.get("scanner_input_digest"), "scanner_input_digest")
    if inputs.get("digest") != scanner_digest:
        raise ValueError("scanner_input_digest alias mismatch")
    unsigned = {key: value for key, value in inputs.items() if key not in {"digest", "scanner_input_digest"}}
    if scanner_digest != canonical_digest(_canonical_scanner_payload(unsigned)):
        raise ValueError("scanner_input_digest is not canonical")
    return provenance


def validated_retained_groups(
    inputs: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return the canonical sealed groups after validating the outer alias."""

    evidence = _required_mapping(inputs.get("scanner_evidence"), "scanner_evidence")
    raw_outer = inputs.get("prune_groups")
    outer = _canonical_retained_groups(raw_outer, "structural inputs")
    if tuple(dict(row) for row in raw_outer) != outer:
        raise ValueError("outer alias must use canonical retained prune group order")
    sealed = _canonical_retained_groups(
        evidence.get("prune_groups"), "sealed scanner evidence"
    )
    if outer != sealed:
        raise ValueError("retained prune group evidence disagrees with structural inputs")
    return sealed


def validated_materializer_bindings(
    inputs: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    evidence = _required_mapping(inputs.get("scanner_evidence"), "scanner_evidence")
    outer = _ordered_authority_rows(
        inputs.get("materializer_bindings"),
        "materializer binding authority",
        "b1_group_id",
    )
    sealed = _ordered_authority_rows(
        evidence.get("materializer_bindings"),
        "sealed materializer binding authority",
        "b1_group_id",
    )
    if outer != sealed:
        raise ValueError("materializer binding authority alias mismatch")
    sources = canonical_source_dataflow_relations(
        _required_sequence(
            inputs.get("source_dataflow_relations"), "source dataflow relations"
        )
    )
    validate_binding_relation_authority(
        sources,
        _required_sequence(inputs.get("dataflow_relations"), "derived relations"),
        sealed,
    )
    return sealed


def _required_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} are required")
    return value


def validated_base_widths(
    inputs: Mapping[str, Any],
    retained_groups: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    evidence = _required_mapping(inputs.get("scanner_evidence"), "scanner_evidence")
    outer = _ordered_authority_rows(
        inputs.get("base_widths"), "base width authority", "axis_id"
    )
    sealed = _ordered_authority_rows(
        evidence.get("base_widths"), "sealed base width authority", "axis_id"
    )
    if outer != sealed:
        raise ValueError("base width authority alias mismatch")
    provenance = _required_mapping(
        evidence.get("base_width_provenance"), "base width authority provenance"
    )
    scanner_provenance = _required_mapping(inputs.get("provenance"), "provenance")
    _validate_base_width_provenance(provenance, scanner_provenance)
    sources = canonical_source_dataflow_relations(
        _required_sequence(
            inputs.get("source_dataflow_relations"), "source dataflow relations"
        )
    )
    validate_base_width_authority(sealed, retained_groups, bindings, sources)
    return sealed


def seal_dataflow_relation(
    row: Mapping[str, Any], source_relation: Mapping[str, Any]
) -> dict[str, Any]:
    sealed = dict(row)
    sealed["relation_provenance"] = {
        "source": "trace_context.dataflow_relations",
        "source_relation_digest": canonical_digest(
            _canonical_source_relation(source_relation)
        ),
    }
    sealed["relation_digest"] = canonical_digest(sealed)
    return sealed


def validate_dataflow_relations(
    rows: Sequence[Any],
    source_rows: Sequence[Mapping[str, Any]],
    retained_groups: Sequence[Mapping[str, Any]],
    source_authority_schema: str,
) -> None:
    sources = _source_relation_index(source_rows)
    groups = _retained_group_index(retained_groups)
    allowed_by_source = _canonical_source_member_partition(
        sources, groups, source_authority_schema
    )
    emitted = {digest: [] for digest in sources}
    for row in rows:
        relation = _required_mapping(row, "dataflow relation")
        digest = _sha256(relation.get("relation_digest"), "relation digest")
        unsigned = {key: value for key, value in relation.items() if key != "relation_digest"}
        if digest != canonical_digest(unsigned):
            raise ValueError("relation digest is not canonical")
        provenance = _required_mapping(
            relation.get("relation_provenance"), "relation provenance"
        )
        if provenance.get("source") != "trace_context.dataflow_relations":
            raise ValueError("relation provenance source is invalid")
        source_digest = _sha256(
            provenance.get("source_relation_digest"), "source relation digest"
        )
        source = sources.get(source_digest)
        if source is None:
            raise ValueError("derived relation has no retained source relation identity")
        emitted[source_digest].append(
            _validate_derived_relation_source(
                relation, source, groups, allowed_by_source[source_digest]
            )
        )
    _validate_source_member_coverage(emitted, allowed_by_source)


def require_free_interface_proof(
    axis_id: str,
    bindings: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
) -> None:
    if not any(_binding_crosses_interface(binding) for binding in bindings):
        return
    related = [row for row in relations if row.get("canonical_axis_id") == axis_id]
    if not related or any(row.get("independent_interface") is not True for row in related):
        raise ValueError(f"free axis {axis_id} requires independent interface proof")


def canonical_source_dataflow_relations(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Canonicalize path-free raw relation authority evidence."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("source dataflow relations are required")
    normalized = [_canonical_source_relation(row) for row in rows]
    identities = [canonical_digest(relation) for relation in normalized]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate source dataflow relation identity")
    return sorted(normalized, key=canonical_digest)


def _validate_retained_source_relations(
    inputs: Mapping[str, Any],
    evidence: Mapping[str, Any],
    declarations: Sequence[Mapping[str, Any]],
    authority_schema: str,
) -> None:
    raw = inputs.get("source_dataflow_relations")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("source dataflow relations are required")
    canonical = canonical_source_dataflow_relations(raw)
    if list(raw) != canonical:
        raise ValueError("source dataflow relations must be canonical")
    if evidence.get("source_dataflow_relations") != canonical:
        raise ValueError("scanner evidence source dataflow relations mismatch")
    _validate_source_declaration_projection(
        canonical, declarations, authority_schema
    )


def _source_relation_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for relation in canonical_source_dataflow_relations(rows):
        digest = canonical_digest(relation)
        indexed[digest] = relation
    return indexed


def _validate_derived_relation_source(
    derived: Mapping[str, Any],
    source: Mapping[str, Any],
    groups: Mapping[str, Mapping[str, Any]],
    allowed_group_ids: frozenset[str],
) -> str:
    fields = {
        "canonical_axis_id": (derived.get("canonical_axis_id"), source.get("canonical_axis_id")),
        "axis_kind": (derived.get("axis_kind", "free"), source.get("axis_kind", "free")),
        "independent_interface": (
            derived.get("independent_interface") is True,
            source.get("independent_interface") is True,
        ),
    }
    if any(left != right for left, right in fields.values()):
        raise ValueError("derived relation disagrees with retained source relation")
    group_id = _required_text(derived.get("group_id"), "derived relation group_id")
    if group_id not in groups:
        raise ValueError("derived relation group_id is not a retained prune group")
    if group_id not in allowed_group_ids:
        raise ValueError("derived relation group_id is not a source member group")
    return group_id


def _validate_source_member_coverage(
    emitted: Mapping[str, list[str]],
    allowed_by_source: Mapping[str, frozenset[str]],
) -> None:
    all_group_ids = [group_id for ids in emitted.values() for group_id in ids]
    if len(all_group_ids) != len(set(all_group_ids)):
        raise ValueError("duplicate derived relation member group")
    for digest, allowed in allowed_by_source.items():
        if set(emitted[digest]) != allowed:
            raise ValueError("derived relation member groups are not complete")


def _retained_group_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group = _required_mapping(row, "retained prune group")
        group_id = _required_text(
            group.get("group_id", group.get("id")), "retained prune group_id"
        )
        if group_id in indexed:
            raise ValueError("duplicate retained prune group identity")
        indexed[group_id] = group
    if not indexed:
        raise ValueError("retained prune groups are required")
    return indexed


def _canonical_retained_groups(
    value: Any, source: str
) -> tuple[dict[str, Any], ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
    ):
        raise ValueError(f"{source} retained prune group evidence is required")
    indexed = _retained_group_index(value)
    return tuple(dict(indexed[group_id]) for group_id in sorted(indexed))


def _ordered_authority_rows(
    value: Any, description: str, identity_field: str
) -> tuple[dict[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{description} rows are required")
    rows = tuple(dict(_required_mapping(row, description)) for row in value)
    identities = [
        _required_text(row.get(identity_field), f"{description} {identity_field}")
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate {description} identity")
    return rows


def _validate_base_width_provenance(
    authority: Mapping[str, Any], scanner: Mapping[str, Any]
) -> None:
    if authority.get("source") != _BASE_WIDTH_AUTHORITY_SOURCE:
        raise ValueError("base width authority source is invalid")
    for field in ("config_digest", "checkpoint_digest"):
        if _sha256(authority.get(field), field) != scanner.get(field):
            raise ValueError(f"base width authority {field} mismatch")


def _validate_structural_authority_aliases(inputs: Mapping[str, Any]) -> None:
    groups = validated_retained_groups(inputs)
    bindings = validated_materializer_bindings(inputs)
    validated_base_widths(inputs, groups, bindings)


def _canonical_source_member_partition(
    sources: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Mapping[str, Any]],
    authority_schema: str,
) -> dict[str, frozenset[str]]:
    boundary_by_source = {
        digest: _canonical_source_group_id(source, groups)
        for digest, source in sources.items()
    }
    modes = {
        _required_text(
            source.get("member_relations_source"), "member relations source"
        )
        for source in sources.values()
    }
    expected_mode = {
        _INFERRED_AUTHORITY_SCHEMA: _INFERRED_MEMBER_SOURCE,
        _DECLARED_AUTHORITY_SCHEMA: _DECLARED_MEMBER_SOURCE,
    }.get(authority_schema)
    if expected_mode is None or modes != {expected_mode}:
        raise ValueError("member relation authority disagrees with source schema")
    if expected_mode == _INFERRED_MEMBER_SOURCE:
        partition = canonical_member_partition(
            tuple(groups.values()), tuple(boundary_by_source.values())
        )
        allowed = {
            digest: _validated_source_member_partition(
                source, partition[boundary_by_source[digest]]
            )
            for digest, source in sources.items()
        }
    else:
        allowed = {
            digest: _validated_source_member_partition(source, None)
            for digest, source in sources.items()
        }
    _validate_exact_retained_group_ownership(allowed, groups)
    return allowed


def _validated_source_relation_authority(
    evidence: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _required_mapping(
        evidence.get("source_relation_authority"), "source relation authority"
    )
    if set(authority) != {"schema", "source_relations", "group_manifest_digest"}:
        raise ValueError("source relation authority fields are invalid")
    declarations = canonical_source_dataflow_relations(
        _required_sequence(
            authority.get("source_relations"), "source relation declarations"
        )
    )
    expected = source_relation_authority_payload(
        groups,
        _required_mapping(evidence.get("group_manifest"), "group manifest"),
        _required_mapping(evidence.get("scenario"), "scanner evidence scenario"),
        declarations,
    )
    if dict(authority) != expected:
        raise ValueError("source relation authority is not canonical")
    if provenance.get("source_relation_authority_schema") != expected["schema"]:
        raise ValueError("source relation authority provenance mismatch")
    if provenance.get("source_group_manifest_digest") != expected[
        "group_manifest_digest"
    ]:
        raise ValueError("source relation authority group manifest mismatch")
    return expected


def _validate_source_declaration_projection(
    sources: Sequence[Mapping[str, Any]],
    declarations: Sequence[Mapping[str, Any]],
    authority_schema: str,
) -> None:
    declared_by_axis = {
        _required_text(row.get("canonical_axis_id"), "declared canonical_axis_id"): row
        for row in declarations
    }
    if len(declared_by_axis) != len(declarations):
        raise ValueError("duplicate source relation declaration axis")
    if len(sources) != len(declared_by_axis):
        raise ValueError("resolved source relations disagree with declarations")
    generated = {
        "canonical_group_id",
        "materializer_binding_projections",
        "member_relations_source",
    }
    if authority_schema == _INFERRED_AUTHORITY_SCHEMA:
        generated |= {
            "member_relations",
            "declared_member_group_ids",
            "member_relations_digest",
        }
    for source in sources:
        axis_id = _required_text(source.get("canonical_axis_id"), "canonical_axis_id")
        declaration = declared_by_axis.get(axis_id)
        if declaration is None:
            raise ValueError("resolved source relation has no declaration authority")
        if any(source.get(key) != value for key, value in declaration.items()):
            raise ValueError("resolved source relation disagrees with declaration authority")
        if set(source) - set(declaration) != generated:
            raise ValueError("resolved source relation fields exceed declaration authority")


def _validated_source_member_partition(
    source: Mapping[str, Any], expected_group_ids: Sequence[str] | None
) -> frozenset[str]:
    members = _required_sequence(
        source.get("member_relations"), "source member relations"
    )
    member_rows = tuple(
        dict(_required_mapping(row, "source member relation")) for row in members
    )
    if any(set(row) != {"group_id", "role"} for row in member_rows):
        raise ValueError("source member relation fields are invalid")
    member_ids = _declared_source_member_ids(source, member_rows)
    declared_ids = tuple(sorted(member_ids))
    expected_ids = (
        declared_ids
        if expected_group_ids is None
        else tuple(sorted(expected_group_ids))
    )
    if declared_ids != expected_ids:
        raise ValueError("source disagrees with canonical inferred member partition")
    roles = {
        _required_text(row.get("b1_group_id"), "binding projection group_id"):
        _required_text(row.get("role"), "binding projection role")
        for row in _required_sequence(
            source.get("materializer_binding_projections"),
            "materializer binding projections",
        )
    }
    expected_rows = tuple(
        sorted(
            ({"group_id": group_id, "role": roles.get(group_id)}
             for group_id in expected_ids),
            key=canonical_digest,
        )
    )
    if any(row["role"] is None for row in expected_rows):
        raise ValueError("canonical inferred member projection is incomplete")
    if tuple(sorted(member_rows, key=canonical_digest)) != expected_rows:
        raise ValueError("source member rows disagree with canonical inferred partition")
    return frozenset(expected_ids)


def _validate_exact_retained_group_ownership(
    allowed_by_source: Mapping[str, frozenset[str]],
    groups: Mapping[str, Mapping[str, Any]],
) -> None:
    owned = tuple(
        group_id
        for group_ids in allowed_by_source.values()
        for group_id in group_ids
    )
    if len(owned) != len(set(owned)):
        raise ValueError("retained group belongs to multiple materializer axes")
    if set(owned) != set(groups):
        raise ValueError("retained groups must all belong to one materializer axis")


def _declared_source_member_ids(
    source: Mapping[str, Any], members: Any
) -> tuple[str, ...]:
    if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
        raise ValueError("source member relations must be a sequence")
    member_ids = tuple(
        _required_text(_required_mapping(row, "source member relation").get("group_id"),
                       "source member group_id")
        for row in members
    )
    declared = source.get("declared_member_group_ids")
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ValueError("source declared member groups are required")
    declared_ids = tuple(_required_text(item, "source declared member group") for item in declared)
    if len(member_ids) != len(set(member_ids)) or len(declared_ids) != len(set(declared_ids)):
        raise ValueError("duplicate source member group identity")
    if set(member_ids) != set(declared_ids):
        raise ValueError("source member groups disagree with declared member groups")
    digest = _sha256(
        source.get("member_relations_digest"), "source member relations digest"
    )
    if digest != canonical_digest(tuple(sorted(declared_ids))):
        raise ValueError("source member relations digest mismatch")
    return member_ids


def _canonical_source_group_id(
    source: Mapping[str, Any], groups: Mapping[str, Mapping[str, Any]]
) -> str:
    selector = _required_text(
        source.get("module_root_selector"), "source module_root_selector"
    )
    matches = [
        group_id
        for group_id, group in groups.items()
        if fnmatch.fnmatchcase(
            str(group.get("root_layer", group.get("module_path")) or ""), selector
        )
    ]
    if len(matches) != 1:
        raise ValueError("source relation must resolve one canonical retained group")
    resolved = matches[0]
    canonical = _required_text(
        source.get("canonical_group_id"), "source canonical_group_id"
    )
    if canonical != resolved:
        raise ValueError("source canonical group disagrees with selector")
    raw_group_id = source.get("group_id")
    if raw_group_id is not None and _required_text(
        raw_group_id, "source group_id"
    ) != resolved:
        raise ValueError("source group_id disagrees with canonical group and selector")
    return resolved


def _binding_crosses_interface(binding: Mapping[str, Any]) -> bool:
    parent = _selector_parent(_required_text(binding.get("param"), "binding param"))
    targets = binding.get("write_targets") or ()
    return any(
        isinstance(target, Mapping)
        and _selector_parent(_required_text(target.get("selector"), "write target")) != parent
        for target in targets
    )


def _canonical_source_relation(relation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(relation, Mapping):
        raise ValueError("source dataflow relation must be an object")
    if any(key in relation for key in ("relation_digest", "relation_provenance")):
        raise ValueError("source dataflow relation cannot contain derived seal fields")
    normalized = dict(relation)
    members = normalized.get("member_relations")
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        if any(not isinstance(row, Mapping) for row in members):
            raise ValueError("source member relations must be objects")
        normalized["member_relations"] = sorted(
            (dict(row) for row in members), key=canonical_digest
        )
    elif members is not None:
        raise ValueError("source member relations must be a sequence")
    declared = normalized.get("declared_member_group_ids")
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        normalized["declared_member_group_ids"] = sorted(str(item) for item in declared)
    projections = normalized.get("materializer_binding_projections")
    if isinstance(projections, Sequence) and not isinstance(
        projections, (str, bytes)
    ):
        normalized["materializer_binding_projections"] = sorted(
            (dict(_required_mapping(row, "materializer binding projection")) for row in projections),
            key=lambda row: str(row.get("b1_group_id") or ""),
        )
    elif projections is not None:
        raise ValueError("materializer binding projections must be a sequence")
    return normalized


def _selector_parent(selector: str) -> str:
    normalized = selector.replace("[*]", "").split("[", 1)[0]
    return normalized.rsplit(".", 1)[0]


def _structural_digest(inputs: Mapping[str, Any]) -> str:
    evidence = {key: inputs.get(key) for key in _STRUCTURAL_KEYS}
    return canonical_digest(_canonical_scanner_payload(evidence))


def _canonical_scanner_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if normalized.get("prune_groups") is not None:
        normalized["prune_groups"] = list(
            _canonical_retained_groups(
                normalized["prune_groups"], "structural inputs"
            )
        )
    if normalized.get("source_dataflow_relations") is not None:
        normalized["source_dataflow_relations"] = (
            canonical_source_dataflow_relations(
                normalized["source_dataflow_relations"]
            )
        )
    return normalized


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} is required")
    return value


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _sha256(value: Any, field: str) -> str:
    digest = _required_text(value, field)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return digest


__all__ = [
    "axis_provenance",
    "canonical_materializer_binding",
    "canonical_source_dataflow_relations",
    "formal_scanner_evidence_digest",
    "formal_scanner_evidence_payload",
    "require_free_interface_proof",
    "scanner_group_manifest",
    "scanner_group_evidence_payload",
    "scanner_digest_sources",
    "scanner_scenario_payload",
    "source_relation_authority_payload",
    "seal_dataflow_relation",
    "seal_materializer_binding",
    "seal_scanner_inputs",
    "validate_dataflow_relations",
    "validate_axis_provenance",
    "validated_base_widths",
    "validated_materializer_bindings",
    "validated_retained_groups",
    "validate_scanner_inputs",
]
