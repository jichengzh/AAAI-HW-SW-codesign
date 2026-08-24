"""Contract tests for generic scanner-derived formal enumeration."""

from __future__ import annotations

from copy import deepcopy

import pytest

from framework.stage1.structural_axis_digest import (
    canonical_digest,
    scanner_structural_axes_digest,
)
from framework.stage2.formal_software_space_v1 import build_formal_software_plan


def _axis(
    axis_id: str,
    widths: list[int],
    *,
    axis_kind: str = "free",
    derived_from: str | None = None,
) -> dict[str, object]:
    provenance = {
        "source": "stage1.graph_scan.build_structural_axis_inputs",
        "scanner_input_digest": "a" * 64,
        "input_digest": "a" * 64,
        "dataflow_group_ids": [f"{axis_id}.g0"],
        "config_digest": "b" * 64,
        "checkpoint_digest": "c" * 64,
        "scenario_digest": "d" * 64,
        "scan_manifest_digest": "e" * 64,
        "relevant_groups_digest": "f" * 64,
        "structural_evidence_digest": "1" * 64,
        "digest_sources": {
            "config_digest": "trace_context.loaded_config",
            "checkpoint_digest": "trace_context.checkpoint_evidence",
            "scenario_digest": "scan_scenario",
            "scan_manifest_digest": "scanner_group_manifest",
            "structural_evidence_digest": "canonical_structural_axis_inputs",
        },
    }
    if derived_from is not None:
        provenance["derived_from"] = derived_from
    return {
        "axis_id": axis_id,
        "dense_stage": None,
        "axis_kind": axis_kind,
        "base_width": widths[-1],
        "legal_widths": widths,
        "member_b1_groups": [
            {
                "b1_group_id": f"{axis_id}.g0",
                "module_path": axis_id,
                "canonical_to_member_num": 1,
                "canonical_to_member_den": 1,
                "materializer_param": f"model.{axis_id}",
                "role": "output",
            }
        ],
        "round_to": 8,
        "provenance": provenance,
    }


def _space(
    free_axes: list[dict[str, object]] | None = None,
    q_modes: list[str] | None = None,
    fixed_axes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    axes = free_axes if free_axes is not None else [_axis("backbone.a0", [16, 24])]
    fixed = fixed_axes or []
    modes = q_modes if q_modes is not None else ["fp16", "int8"]
    hardware_target = {
        "name": "unit-hardware",
        "int8_align": 32,
        "fp16_align": 8,
        "int8_pack_factor": 4,
        "alignment_enforcement": "hard",
    }
    return {
        "schema": "stage2_search_space_v1",
        "model": "unit_model",
        "hardware_target": hardware_target,
        "scanner_structural_axes_digest": scanner_structural_axes_digest(
            axes + fixed
        ),
        "axis_schema": {
            "free_axes": axes,
            "fixed_derived_axes": fixed,
        },
        "formal_q_modes": modes,
        "formal_q_mode_provenance": _q_mode_provenance(hardware_target, modes),
        "formal_candidate_policy": {
            "enumeration": "axis_schema.free_axes_x_formal_q_modes",
            "legacy_views": "diagnostic_only",
            "diagnostic_anchors_drive_formal_space": False,
            "source": "scanner_structural_axes",
        },
    }


def _q_mode_provenance(
    hardware_target: dict[str, object], modes: list[str]
) -> dict[str, object]:
    payload = {
        "schema": "formal_q_mode_provenance_v1",
        "hardware_target": deepcopy(hardware_target),
        "sources": {
            "hardware": list(modes),
            "backend": list(modes),
            "configured": list(modes),
            "graph": list(modes),
        },
        "formal_q_modes": list(modes),
    }
    return {**payload, "digest": canonical_digest(payload)}


def _resign_q_mode_provenance(search_space: dict[str, object]) -> None:
    provenance = search_space["formal_q_mode_provenance"]
    unsigned = {key: value for key, value in provenance.items() if key != "digest"}
    provenance["digest"] = canonical_digest(unsigned)


def test_formal_software_plan_enumerates_ordered_free_axes_and_q_modes() -> None:
    plan = build_formal_software_plan(_space())

    assert plan["structure_count"] == 2
    assert plan["candidate_count"] == 4
    assert [row["width_tuple"] for row in plan["candidates"]] == [
        [16],
        [16],
        [24],
        [24],
    ]
    assert [row["q_mode"] for row in plan["candidates"]] == [
        "fp16",
        "int8",
        "fp16",
        "int8",
    ]


@pytest.mark.parametrize("dense_stage", ["", " ", 0, False, []])
def test_formal_plan_rejects_invalid_dense_stage(dense_stage: object) -> None:
    search_space = _space()
    search_space["axis_schema"]["free_axes"][0]["dense_stage"] = dense_stage
    search_space["scanner_structural_axes_digest"] = scanner_structural_axes_digest(
        search_space["axis_schema"]["free_axes"]
    )

    with pytest.raises(ValueError, match="dense_stage"):
        build_formal_software_plan(search_space)


def test_candidate_identity_binds_axis_order_width_q_mode_and_scanner_source() -> None:
    fixed = _axis(
        "neck.output",
        [32, 64],
        axis_kind="fixed_derived",
        derived_from="backbone.a0",
    )
    search_space = _space(fixed_axes=[fixed])

    first = build_formal_software_plan(search_space)
    second = build_formal_software_plan(deepcopy(search_space))

    assert first == second
    assert first["free_axis_count"] == 1
    assert first["axis_schema"]["fixed_derived_axes"] == [fixed]
    assert first["source_provenance"] == {
        "scanner_structural_axes_digest": search_space[
            "scanner_structural_axes_digest"
        ],
        "formal_candidate_policy": search_space["formal_candidate_policy"],
        "formal_q_mode_provenance": search_space["formal_q_mode_provenance"],
        "axis_provenance": [
            {
                "axis_id": "backbone.a0",
                "provenance": search_space["axis_schema"]["free_axes"][0][
                    "provenance"
                ],
            },
            {
                "axis_id": "neck.output",
                "provenance": fixed["provenance"],
            },
        ],
    }
    identities = [row["identity"] for row in first["candidates"]]
    assert identities[0] == {
        "axis_schema": first["axis_schema"],
        "axis_order": ["backbone.a0"],
        "width_tuple": [16],
        "q_mode": "fp16",
        "source_provenance": first["source_provenance"],
    }
    candidate_ids = [row["candidate_id"] for row in first["candidates"]]
    assert len(candidate_ids) == len(set(candidate_ids))


def test_candidate_identities_are_isolated_from_returned_plan_aliases() -> None:
    plan = build_formal_software_plan(_space())
    identities = deepcopy([candidate["identity"] for candidate in plan["candidates"]])

    plan["axis_schema"]["free_axes"][0]["base_width"] = 999
    plan["source_provenance"]["formal_candidate_policy"]["source"] = "mutated"
    plan["candidates"][0]["axis_order"].append("mutated.order")

    assert [candidate["identity"] for candidate in plan["candidates"]] == identities
    assert plan["candidates"][1]["axis_order"] == ["backbone.a0"]
    assert all(
        candidate["candidate_id"] == canonical_digest(candidate["identity"])
        for candidate in plan["candidates"]
    )


def test_candidate_identity_mutation_does_not_reach_another_candidate() -> None:
    plan = build_formal_software_plan(_space())
    second = plan["candidates"][1]
    second_identity = deepcopy(second["identity"])

    first_identity = plan["candidates"][0]["identity"]
    first_identity["axis_schema"]["free_axes"][0]["base_width"] = 999
    first_identity["source_provenance"]["formal_candidate_policy"]["source"] = "mutated"

    assert second["identity"] == second_identity
    assert second["candidate_id"] == canonical_digest(second["identity"])


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda space: space.pop("axis_schema"), "axis_schema"),
        (lambda space: space["axis_schema"].update({"free_axes": []}), "free_axes"),
        (
            lambda space: space["axis_schema"]["free_axes"][0].update(
                {"legal_widths": []}
            ),
            "legal_widths",
        ),
        (
            lambda space: space["axis_schema"]["free_axes"].append(
                deepcopy(space["axis_schema"]["free_axes"][0])
            ),
            "duplicate axis_id",
        ),
        (
            lambda space: space["axis_schema"]["free_axes"][0].update(
                {"legal_widths": [16, 16, 24]}
            ),
            "sorted and unique",
        ),
        (
            lambda space: space["axis_schema"]["free_axes"][0].update(
                {"legal_widths": [24, 16]}
            ),
            "sorted and unique",
        ),
        (
            lambda space: space["axis_schema"]["free_axes"][0].update(
                {"legal_widths": [16, 32]}
            ),
            "base_width",
        ),
        (
            lambda space: space["axis_schema"]["free_axes"][0].update(
                {"legal_widths": [True, 24]}
            ),
            "positive integer",
        ),
        (
            lambda space: space["axis_schema"]["free_axes"][0].update(
                {"base_width": 0}
            ),
            "positive integer",
        ),
    ],
    ids=[
        "missing-axis-schema",
        "empty-free-axes",
        "empty-widths",
        "duplicate-axis-id",
        "duplicate-width",
        "noncanonical-width-order",
        "over-base-width",
        "boolean-width",
        "nonpositive-base",
    ],
)
def test_formal_plan_rejects_malformed_axis_universe(mutation, error: str) -> None:
    search_space = _space()
    mutation(search_space)

    with pytest.raises(ValueError, match=error):
        build_formal_software_plan(search_space)


@pytest.mark.parametrize(
    "q_modes",
    [
        [],
        ["fp16", "fp16"],
        ["int8", "fp16"],
        ["fp16", "int4"],
        ["fp16", True],
    ],
    ids=["empty", "duplicate", "noncanonical", "unsupported", "boolean"],
)
def test_formal_plan_rejects_invalid_q_modes(q_modes: list[object]) -> None:
    with pytest.raises(ValueError, match="formal_q_modes"):
        build_formal_software_plan(_space(q_modes=q_modes))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target",
    [
        "missing-provenance",
        "unknown-provenance-field",
        "unknown-source",
        "missing-source",
        "noncanonical-source",
        "source-intersection-drift",
        "unknown-hardware-target-field",
        "hardware-target-drift",
        "digest-drift",
        "mode-alias-drift",
    ],
)
def test_formal_plan_rejects_invalid_q_mode_provenance(target: str) -> None:
    search_space = _space()
    provenance = search_space["formal_q_mode_provenance"]
    if target == "missing-provenance":
        search_space.pop("formal_q_mode_provenance")
    elif target == "unknown-provenance-field":
        provenance["expected_candidates"] = 4
        _resign_q_mode_provenance(search_space)
    elif target == "unknown-source":
        provenance["sources"]["paper"] = ["fp16", "int8"]
        _resign_q_mode_provenance(search_space)
    elif target == "missing-source":
        provenance["sources"].pop("graph")
        _resign_q_mode_provenance(search_space)
    elif target == "noncanonical-source":
        provenance["sources"]["hardware"] = ["int8", "fp16"]
        _resign_q_mode_provenance(search_space)
    elif target == "source-intersection-drift":
        provenance["sources"]["graph"] = ["fp16"]
        _resign_q_mode_provenance(search_space)
    elif target == "unknown-hardware-target-field":
        search_space["hardware_target"]["expected_speedup"] = 2
        provenance["hardware_target"]["expected_speedup"] = 2
        _resign_q_mode_provenance(search_space)
    elif target == "hardware-target-drift":
        search_space["hardware_target"]["name"] = "different-hardware"
    elif target == "digest-drift":
        provenance["digest"] = "9" * 64
    else:
        provenance["formal_q_modes"] = ["fp16"]
        _resign_q_mode_provenance(search_space)

    with pytest.raises(ValueError, match="formal_q_mode_provenance"):
        build_formal_software_plan(search_space)


@pytest.mark.parametrize(
    "target",
    ["top-level", "axis-schema", "axis", "member", "policy"],
)
def test_formal_plan_rejects_unknown_contract_fields(target: str) -> None:
    search_space = _space()
    if target == "top-level":
        search_space["candidate_count"] = 4
    elif target == "axis-schema":
        search_space["axis_schema"]["oracle_axes"] = []
    elif target == "axis":
        search_space["axis_schema"]["free_axes"][0]["expected_count"] = 2
    elif target == "member":
        search_space["axis_schema"]["free_axes"][0]["member_b1_groups"][0][
            "unknown"
        ] = True
    else:
        search_space["formal_candidate_policy"]["fallback"] = "legacy"

    with pytest.raises(ValueError, match="unknown fields"):
        build_formal_software_plan(search_space)


def test_formal_plan_rejects_scanner_digest_drift() -> None:
    search_space = _space()
    search_space["scanner_structural_axes_digest"] = "c" * 64

    with pytest.raises(ValueError, match="digest mismatch"):
        build_formal_software_plan(search_space)


@pytest.mark.parametrize("alias", [[None], ["axis"], [[]]])
def test_formal_plan_rejects_nonmapping_structural_axis_alias(
    alias: list[object],
) -> None:
    search_space = _space()
    search_space["structural_axes"] = alias

    with pytest.raises(ValueError, match="structural_axes entries must be objects"):
        build_formal_software_plan(search_space)


def test_formal_plan_rejects_resigned_non_scanner_provenance() -> None:
    search_space = _space()
    axis = search_space["axis_schema"]["free_axes"][0]
    axis["provenance"]["source"] = "handwritten"
    search_space["scanner_structural_axes_digest"] = scanner_structural_axes_digest(
        search_space["axis_schema"]["free_axes"]
    )

    with pytest.raises(ValueError, match="provenance.source"):
        build_formal_software_plan(search_space)


def test_formal_plan_rejects_resigned_unknown_axis_provenance_field() -> None:
    search_space = _space()
    axis = search_space["axis_schema"]["free_axes"][0]
    axis["provenance"]["expected_count"] = 343
    search_space["scanner_structural_axes_digest"] = scanner_structural_axes_digest(
        search_space["axis_schema"]["free_axes"]
    )

    with pytest.raises(ValueError, match="provenance.*unknown fields"):
        build_formal_software_plan(search_space)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda provenance: provenance.pop("input_digest"), "missing fields"),
        (
            lambda provenance: provenance.update({"derived_from": "backbone.a0"}),
            "unknown fields",
        ),
    ],
    ids=["missing-required-field", "derived-from-on-free-axis"],
)
def test_formal_plan_rejects_noncanonical_axis_provenance(
    mutation, error: str
) -> None:
    search_space = _space()
    axis = search_space["axis_schema"]["free_axes"][0]
    mutation(axis["provenance"])
    search_space["scanner_structural_axes_digest"] = scanner_structural_axes_digest(
        search_space["axis_schema"]["free_axes"]
    )

    with pytest.raises(ValueError, match=error):
        build_formal_software_plan(search_space)
