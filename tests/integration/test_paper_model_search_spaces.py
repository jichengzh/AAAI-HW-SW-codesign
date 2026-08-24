"""CPU-only reproduction gate for the three CoptV2X paper spaces."""

from __future__ import annotations

from copy import deepcopy

import pytest

from framework.reproduction.coptv2x_paper_space_v1 import (
    build_paper_space_reproduction,
)
from framework.stage1.structural_axis_digest import canonical_digest
from framework.stage2.formal_software_space_v1 import build_formal_software_plan


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
