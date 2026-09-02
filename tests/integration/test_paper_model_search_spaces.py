"""CPU-only reproduction gate for the three CoptV2X paper spaces."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from framework.reproduction.coptv2x_paper_space_v1 import (
    build_stage1_scanner_manifest,
    build_paper_space_reproduction,
    load_paper_scanner_evidence,
)
from framework.stage1.structural_axis_contract import scanner_group_manifest
from framework.stage1.structural_axis_digest import canonical_digest
from framework.stage1_bridge import load_stage2_search_space
from framework.stage2.formal_software_space_v1 import build_formal_software_plan
from framework.stage6.pyramid_search_space_adapter_v1 import (
    build_pyramid_candidate_plan,
)


_PAPER_ORACLES = [
    (
        "pyramid",
        343,
        686,
        [
            [16, 24, 32, 40, 48, 56, 64],
            [32, 48, 64, 80, 96, 112, 128],
            [64, 96, 128, 160, 192, 224, 256],
        ],
    ),
    (
        "codriving",
        343,
        686,
        [
            [16, 24, 32, 40, 48, 56, 64],
            [32, 48, 64, 80, 96, 112, 128],
            [64, 96, 128, 160, 192, 224, 256],
        ],
    ),
    (
        "fcooper",
        1792,
        3584,
        [
            [32, 64],
            [32, 64, 96, 128],
            [32, 64, 96, 128, 160, 192, 224, 256],
            [32, 64, 96, 128],
            [64, 96, 128, 160, 192, 224, 256],
        ],
    ),
]


def _production_like_inferred_pyramid_evidence() -> dict:
    evidence = load_paper_scanner_evidence("pyramid")
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "paper_spaces"
        / "pyramid_source_bound_retained_graph.yaml"
    )
    graph = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    assert graph["schema"] == "pyramid_source_bound_retained_graph_v1"
    assert graph["source_group_count"] == 43
    assert graph["groups_digest"] == canonical_digest(graph["groups"])
    paper_groups = {row["group_id"]: row for row in evidence["prune_groups"]}
    retained_groups = [
        {**paper_groups[row["group_id"]], **row} for row in graph["groups"]
    ]
    inferred_relations = [
        {
            key: source[key]
            for key in (
                "canonical_axis_id",
                "module_root_selector",
                "axis_kind",
                "derived_from",
            )
            if key in source
        }
        for source in evidence["dataflow_relations"]
    ]
    provenance = scanner_group_manifest(
        retained_groups,
        evidence["scan_scenario"],
        inferred_relations,
    )
    return {
        **evidence,
        "source_provenance": {
            **provenance,
            "purification": evidence["source_provenance"]["purification"],
        },
        "prune_groups": retained_groups,
        "dataflow_relations": inferred_relations,
    }


def test_production_like_43_group_pyramid_inference_reaches_formal_plan(
    tmp_path: Path,
) -> None:
    evidence = _production_like_inferred_pyramid_evidence()
    assert len(evidence["prune_groups"]) == 43
    assert all(
        "member_relations" not in relation
        for relation in evidence["dataflow_relations"]
    )

    manifest = build_stage1_scanner_manifest("pyramid", evidence)
    manifest_path = tmp_path / "inferred-pyramid.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    search_space = load_stage2_search_space(manifest_path)
    formal_plan = build_formal_software_plan(search_space)
    p6_plan = build_pyramid_candidate_plan(search_space)

    scanner_groups = [
        member["b1_group_id"]
        for axis in manifest["scanner_structural_axes"]
        for member in axis["member_b1_groups"]
    ]
    assert len(scanner_groups) == len(set(scanner_groups)) == 43
    assert set(scanner_groups) == {
        group["group_id"] for group in evidence["prune_groups"]
    }
    assert formal_plan["structure_count"] == p6_plan["structure_count"] == 343
    assert formal_plan["candidate_count"] == p6_plan["candidate_count"] == 686


@pytest.mark.parametrize(
    ("name", "expected_structures", "expected_candidates", "expected_widths"),
    _PAPER_ORACLES,
)
def test_scanner_evidence_reproduces_paper_formal_search_spaces(
    name: str,
    expected_structures: int,
    expected_candidates: int,
    expected_widths: list[list[int]],
) -> None:
    search_space = build_paper_space_reproduction(name).mutable_search_space()
    plan = build_formal_software_plan(search_space)

    axes = plan["axis_schema"]["free_axes"]
    assert [axis["legal_widths"] for axis in axes] == expected_widths
    assert plan["structure_count"] == expected_structures
    assert plan["candidate_count"] == expected_candidates
    assert plan["q_modes"] == ["fp16", "int8"]
    assert plan["source_provenance"]["scanner_structural_axes_digest"] == (
        search_space["scanner_structural_axes_digest"]
    )
    assert plan["source_provenance"]["formal_candidate_policy"]["source"] == (
        "scanner_structural_axes"
    )
    assert plan["source_provenance"]["formal_q_mode_provenance"] == (
        search_space["formal_q_mode_provenance"]
    )

    candidates = plan["candidates"]
    assert len({row["candidate_id"] for row in candidates}) == expected_candidates
    assert [row["structure_index"] for row in candidates[:4]] == [0, 0, 1, 1]
    assert candidates[0]["q_mode"] == "fp16"
    assert candidates[1]["q_mode"] == "int8"
    assert candidates[0]["width_tuple"] == [widths[0] for widths in expected_widths]
    assert candidates[-1]["width_tuple"] == [widths[-1] for widths in expected_widths]
    assert all(
        width <= axis["base_width"]
        for row in candidates
        for width, axis in zip(row["width_tuple"], axes, strict=True)
    )
    assert all(row["identity"]["axis_schema"] == plan["axis_schema"] for row in candidates)


def test_candidate_ids_bind_revalidated_q_capability_provenance(
) -> None:
    first_space = build_paper_space_reproduction("pyramid").mutable_search_space()
    second_space = deepcopy(first_space)
    second_space["hardware_target"]["name"] = "alternate-h800-profile"
    provenance = deepcopy(second_space["formal_q_mode_provenance"])
    provenance["hardware_target"] = deepcopy(second_space["hardware_target"])
    unsigned = {key: value for key, value in provenance.items() if key != "digest"}

    provenance["digest"] = canonical_digest(unsigned)
    second_space["formal_q_mode_provenance"] = provenance

    assert first_space["formal_q_modes"] == second_space["formal_q_modes"]
    assert first_space["formal_q_mode_provenance"]["digest"] != (
        second_space["formal_q_mode_provenance"]["digest"]
    )
    first_ids = [
        row["candidate_id"] for row in build_formal_software_plan(first_space)["candidates"]
    ]
    second_ids = [
        row["candidate_id"] for row in build_formal_software_plan(second_space)["candidates"]
    ]

    assert first_ids != second_ids
