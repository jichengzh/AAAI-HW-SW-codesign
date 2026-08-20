from __future__ import annotations

import json
from itertools import product
from typing import Any

import pytest

from framework.stage6.pyramid_search_space_adapter_v1 import (
    PyramidSearchSpaceAdapterError,
    build_pyramid_candidate_plan,
    parse_pyramid_stage_provenance_token,
)


def _candidate(
    stage: str,
    widths: list[int],
    q_modes: list[str],
    widths_by_q_mode: dict[str, list[int]] | None = None,
    *,
    group_id: str | None = None,
) -> dict[str, Any]:
    source_group_id = group_id or stage
    return {
        "id": source_group_id,
        "dense_stage": stage,
        "software_points": [
            {
                "id": f"{source_group_id}-{q_mode}-{width}",
                "width": width,
                "quant_policy": q_mode,
                "buildable": True,
                "status": "active",
            }
            for q_mode in q_modes
            for width in (widths_by_q_mode or {}).get(q_mode, widths)
        ],
    }


def _space(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": "stage2_search_space_v1",
        "model": "pyramid_lidar",
        "hardware_target": {"name": "h800"},
        "software_candidates": [
            _candidate("stage1", [16, 32], ["fp16", "int8"], {"int8": [16]}),
            _candidate("stage2", [32, 64], ["fp16", "int8"], {"int8": [32]}),
            _candidate("stage3", [64, 96], ["fp16", "int8"], {"int8": [64]}),
        ],
        "hardware_candidates": [
            {
                "id": "tvm_metaschedule_candidate",
                "backend_scope": "measured_h800_tvm",
                "hardware": "h800",
                "schedule_policy": "tuned",
            }
        ],
    }
    return {**payload, **overrides}


def test_build_pyramid_candidate_plan_uses_independent_q_mode_products() -> None:
    plan = build_pyramid_candidate_plan(_space())

    assert plan["schema_version"] == "p6_pyramid_candidate_plan_v2"
    assert plan["source_schema"] == "stage2_search_space_v1"
    assert plan["target_model"] == "pyramid"
    assert plan["hardware_target"] == "h800"
    assert plan["execution_backend"] == "tvm_auto"
    assert plan["candidate_source_mode"] == "framework_stage2_search_space"
    assert plan["structure_count"] == 8
    assert plan["candidate_count"] == 9
    assert {(tuple(row["width"]), row["q_mode"]) for row in plan["candidates"]} == {
        *((width, "fp16") for width in product((16, 32), (32, 64), (64, 96))),
        ((16, 32, 64), "int8"),
    }
    assert all(row["q_mode"] in {"fp16", "int8"} for row in plan["candidates"])


def test_build_pyramid_candidate_plan_allows_missing_int8_counterparts() -> None:
    space = _space()
    for stage in space["software_candidates"]:
        stage["software_points"] = [
            point
            for point in stage["software_points"]
            if point["quant_policy"] == "fp16"
        ]
    plan = build_pyramid_candidate_plan(space)
    assert plan["candidate_count"] == 8
    assert all(row["q_mode"] == "fp16" for row in plan["candidates"])


def test_build_pyramid_candidate_plan_merges_real_same_stage_group_provenance() -> None:
    payload = _space(
        software_candidates=[
            _candidate(
                "stage1", [32], ["fp16"], group_id="stage1/group,A"
            ),
            _candidate(
                "stage1", [32], ["fp16"], group_id="stage1 group+B%"
            ),
            _candidate("stage2", [64], ["fp16"]),
            _candidate("stage3", [96], ["fp16"]),
        ]
    )

    plan = build_pyramid_candidate_plan(payload)

    assert plan["candidate_count"] == 1
    candidate = plan["candidates"][0]
    assert candidate["width"] == [32, 64, 96]
    assert len(candidate["source_point_ids"]) == 3
    assert parse_pyramid_stage_provenance_token(candidate["source_point_ids"][0]) == (
        "stage1 group+B%-fp16-32",
        "stage1/group,A-fp16-32",
    )
    assert candidate["source_point_ids"][1:] == [
        "stage2-fp16-64",
        "stage3-fp16-96",
    ]


def test_build_pyramid_candidate_plan_intersects_widths_across_stage_groups() -> None:
    payload = _space(
        software_candidates=[
            _candidate("stage1", [16, 32], ["fp16"], group_id="stage1-a"),
            _candidate("stage1", [32, 64], ["fp16"], group_id="stage1-b"),
            _candidate("stage2", [32, 64], ["fp16"]),
            _candidate("stage3", [64, 96], ["fp16"]),
        ]
    )

    plan = build_pyramid_candidate_plan(payload)

    assert plan["candidate_count"] == 4
    assert {row["width"][0] for row in plan["candidates"]} == {32}


def test_build_pyramid_candidate_plan_preserves_single_group_source_ids() -> None:
    plan = build_pyramid_candidate_plan(_space())

    candidate = next(
        row
        for row in plan["candidates"]
        if row["width"] == [16, 32, 64] and row["q_mode"] == "fp16"
    )
    assert candidate["source_point_ids"] == [
        "stage1-fp16-16",
        "stage2-fp16-32",
        "stage3-fp16-64",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"schema": "bad"}), "schema"),
        (lambda data: data.update({"model": "codriving"}), "model"),
        (lambda data: data.update({"hardware_target": {"name": "orin"}}), "H800"),
        (lambda data: data.update({"hardware_candidates": []}), "TVM"),
        (
            lambda data: data["hardware_candidates"][0].update({"backend_scope": "cuda"}),
            "TVM",
        ),
        (lambda data: data.update({"software_candidates": data["software_candidates"][:-1]}), "stage"),
    ],
)
def test_build_pyramid_candidate_plan_rejects_unsupported_inputs(
    mutate: Any, message: str
) -> None:
    payload = _space()
    mutate(payload)
    with pytest.raises(PyramidSearchSpaceAdapterError, match=message):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_unknown_quant_policy() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"][0]["quant_policy"] = "fp32"
    with pytest.raises(PyramidSearchSpaceAdapterError, match="quant"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_active_non_buildable_point() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"][0]["buildable"] = False
    with pytest.raises(PyramidSearchSpaceAdapterError, match="active.*buildable"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_non_active_buildable_point() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"][0].update(
        {"status": "diagnostic", "buildable": True}
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match="unsupported"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_ignores_non_active_non_buildable_diagnostics() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"].append(
        {
            "id": "diagnostic",
            "width": 128,
            "quant_policy": "int8",
            "buildable": False,
            "status": "diagnostic",
        }
    )
    assert build_pyramid_candidate_plan(payload)["candidate_count"] == 9


def test_build_pyramid_candidate_plan_rejects_non_diagnostic_non_buildable_point() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"].append(
        {
            "id": "retired",
            "width": 128,
            "quant_policy": "int8",
            "buildable": False,
            "status": "retired",
        }
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match="unsupported"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_duplicate_point_identity() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"].append(
        payload["software_candidates"][0]["software_points"][0].copy()
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match="duplicate"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_source_id_across_shapes() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"][1]["id"] = (
        payload["software_candidates"][0]["software_points"][0]["id"]
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match="different shapes"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_reserved_token_prefix_collision() -> None:
    payload = _space()
    payload["software_candidates"][0]["software_points"][0]["id"] = (
        "p6-stage-provenance-v1:raw-source-id"
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match="reserved provenance"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_no_complete_q_mode_product() -> None:
    payload = _space()
    payload["software_candidates"][1]["software_points"] = [{
        "id": "diagnostic-stage2",
        "width": 128,
        "quant_policy": "fp16",
        "buildable": False,
        "status": "diagnostic",
    }]
    with pytest.raises(PyramidSearchSpaceAdapterError, match="q-mode"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_rejects_stage_groups_without_common_q_mode() -> None:
    payload = _space(
        software_candidates=[
            _candidate("stage1", [16], ["fp16", "int8"], group_id="stage1-a"),
            _candidate("stage1", [32], ["fp16", "int8"], group_id="stage1-b"),
            _candidate("stage2", [64], ["fp16", "int8"]),
            _candidate("stage3", [96], ["fp16", "int8"]),
        ]
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match="q-mode"):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_is_canonical_when_inputs_reverse() -> None:
    first = _space()
    second = _space()
    first["software_candidates"].append(
        _candidate(
            "stage1",
            [16, 32],
            ["fp16", "int8"],
            {"int8": [16]},
            group_id="stage1-b",
        )
    )
    second["software_candidates"].append(
        _candidate(
            "stage1",
            [16, 32],
            ["fp16", "int8"],
            {"int8": [16]},
            group_id="stage1-b",
        )
    )
    second["software_candidates"] = list(reversed(second["software_candidates"]))
    for candidate in second["software_candidates"]:
        candidate["software_points"] = list(reversed(candidate["software_points"]))
    second["hardware_candidates"] = list(reversed(second["hardware_candidates"]))
    assert json.dumps(build_pyramid_candidate_plan(first), sort_keys=True) == json.dumps(
        build_pyramid_candidate_plan(second), sort_keys=True
    )


@pytest.mark.parametrize(
    "token",
    [
        "raw-source-id",
        "p6-stage-provenance-v1:only-one",
        "p6-stage-provenance-v1:b,a",
        "p6-stage-provenance-v1:a,a",
        "p6-stage-provenance-v1:a,%61",
        "p6-stage-provenance-v1:a,%ZZ",
        "p6-stage-provenance-v1:a,",
    ],
)
def test_parse_pyramid_stage_provenance_token_rejects_noncanonical_tokens(
    token: str,
) -> None:
    with pytest.raises(PyramidSearchSpaceAdapterError, match="provenance token"):
        parse_pyramid_stage_provenance_token(token)
