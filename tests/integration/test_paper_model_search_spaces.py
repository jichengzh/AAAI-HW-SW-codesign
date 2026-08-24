"""CPU-only reproduction gate for the three CoptV2X paper spaces."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from framework.stage1.structural_axes import axis_to_scanner_dict
from framework.stage1.structural_axis_digest import scanner_structural_axes_digest
from framework.stage1_bridge import load_stage2_search_space
from framework.stage2.formal_software_space_v1 import build_formal_software_plan
from tests.stage1.structural_axis_test_support import paper_axis_bundle


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


def _scanner_manifest(name: str) -> dict[str, object]:
    evidence, inputs, bundle = paper_axis_bundle(name)
    scenario = evidence["scan_scenario"]
    scanner_axes = [axis_to_scanner_dict(axis, inputs) for axis in bundle.axes]
    return {
        "schema": "stage1_scanner_contract_v1",
        "model": evidence["model"],
        "scan_status": "ok",
        "hw_capability": {
            "name": "h800",
            "ips": {"gpu": {"precisions": scenario["hardware_precisions"]}},
        },
        "backend_support": {"precisions": scenario["backend_precisions"]},
        "compression_modes": scenario["compression_modes"],
        "quant_units": [
            {"id": unit, "legal_precisions": precisions}
            for unit, precisions in scenario["graph_quant_unit_policy"].items()
        ],
        "scanner_structural_axes": scanner_axes,
        "scanner_structural_axes_digest": scanner_structural_axes_digest(
            scanner_axes
        ),
        "view_b1_search_groups": [],
        "view_b2_quant_units": [],
        "view_d_routing_segments": {"segments": []},
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("name", "expected_structures", "expected_candidates", "expected_widths"),
    _PAPER_ORACLES,
)
def test_scanner_evidence_reproduces_paper_formal_search_spaces(
    tmp_path: Path,
    name: str,
    expected_structures: int,
    expected_candidates: int,
    expected_widths: list[list[int]],
) -> None:
    manifest_path = _write_manifest(tmp_path / f"{name}.yaml", _scanner_manifest(name))
    search_space = load_stage2_search_space(manifest_path)
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
    tmp_path: Path,
) -> None:
    first_manifest = _scanner_manifest("pyramid")
    second_manifest = deepcopy(first_manifest)
    second_manifest["hw_capability"]["name"] = "alternate-h800-profile"
    first_space = load_stage2_search_space(
        _write_manifest(tmp_path / "first.yaml", first_manifest)
    )
    second_space = load_stage2_search_space(
        _write_manifest(tmp_path / "second.yaml", second_manifest)
    )

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
