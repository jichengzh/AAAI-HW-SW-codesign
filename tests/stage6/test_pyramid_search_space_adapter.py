from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from framework.stage1.structural_axis_digest import (
    canonical_digest,
    scanner_structural_axes_digest,
)
from framework.stage6.hardware_execution_profile_v1 import (
    load_hardware_execution_profile,
)
from framework.stage6.p6_formal_plan_contract_v1 import validate_p6_candidate_plan
from framework.stage6.pyramid_search_space_adapter_v1 import (
    PyramidSearchSpaceAdapterError,
    build_pyramid_candidate_plan,
    parse_pyramid_stage_provenance_token,
)


def _candidate(
    stage: str,
    widths: list[int],
    q_modes: list[str],
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
            for width in widths
        ],
    }


def _space(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": "stage2_search_space_v1",
        "model": "pyramid_lidar",
        "hardware_target": {
            "name": "h800",
            "int8_align": 32,
            "fp16_align": 8,
            "int8_pack_factor": 4,
            "alignment_enforcement": "hard",
        },
        "software_candidates": [
            _candidate("stage1", [64], ["fp16"], group_id="legacy-stage1"),
            _candidate("stage2", [128], ["fp16"], group_id="legacy-stage2"),
            _candidate("stage3", [256], ["fp16"], group_id="legacy-stage3"),
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


def _scanner_axis(axis_id: str, dense_stage: str, widths: list[int]) -> dict[str, Any]:
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
    return {
        "axis_id": axis_id,
        "dense_stage": dense_stage,
        "axis_kind": "free",
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


def _q_mode_provenance(
    hardware_target: dict[str, Any], modes: list[str]
) -> dict[str, Any]:
    payload = {
        "schema": "formal_q_mode_provenance_v1",
        "hardware_target": copy.deepcopy(hardware_target),
        "sources": {
            "hardware": list(modes),
            "backend": list(modes),
            "configured": list(modes),
            "graph": list(modes),
        },
        "formal_q_modes": list(modes),
    }
    return {**payload, "digest": canonical_digest(payload)}


def _pyramid_space_with_axis_schema() -> dict[str, Any]:
    axes = [
        _scanner_axis("backbone.s0", "stage1", [16, 24, 32, 40, 48, 56, 64]),
        _scanner_axis("backbone.s1", "stage2", [32, 48, 64, 80, 96, 112, 128]),
        _scanner_axis("backbone.s2", "stage3", [64, 96, 128, 160, 192, 224, 256]),
    ]
    hardware_target = _space()["hardware_target"]
    return _space(
        scanner_structural_axes_digest=scanner_structural_axes_digest(axes),
        axis_schema={"free_axes": axes, "fixed_derived_axes": []},
        formal_q_modes=["fp16", "int8"],
        formal_q_mode_provenance=_q_mode_provenance(
            hardware_target, ["fp16", "int8"]
        ),
        formal_candidate_policy={
            "enumeration": "axis_schema.free_axes_x_formal_q_modes",
            "legacy_views": "diagnostic_only",
            "diagnostic_anchors_drive_formal_space": False,
            "source": "scanner_structural_axes",
        },
    )


def _profile_space(hardware_name: str) -> dict[str, Any]:
    payload = _pyramid_space_with_axis_schema()
    hardware_target = copy.deepcopy(payload["hardware_target"])
    hardware_target["name"] = hardware_name
    payload["hardware_target"] = hardware_target
    payload["formal_q_mode_provenance"] = _q_mode_provenance(
        hardware_target, ["fp16", "int8"]
    )
    payload["hardware_candidates"][0]["hardware"] = hardware_name
    return payload


def test_build_pyramid_candidate_plan_uses_generic_formal_plan() -> None:
    space = _pyramid_space_with_axis_schema()

    plan = build_pyramid_candidate_plan(space)

    assert plan["schema_version"] == "p6_pyramid_candidate_plan_v2"
    assert plan["source_schema"] == "stage2_search_space_v1"
    assert plan["target_model"] == "pyramid"
    assert plan["hardware_target"] == "h800"
    assert plan["execution_backend"] == "tvm_auto"
    assert plan["candidate_source_mode"] == "framework_stage2_search_space"
    assert plan["structure_count"] == 343
    assert plan["candidate_count"] == 686
    assert plan["candidates"][0]["width"] == [16, 32, 64]
    assert plan["candidates"][-1]["width"] == [64, 128, 256]
    assert {row["q_mode"] for row in plan["candidates"]} == {"fp16", "int8"}
    assert all("formal_candidate_id" in row for row in plan["candidates"])
    assert all("formal_identity" in row for row in plan["candidates"])


def test_rtx_profile_builds_scanner_owned_343_by_686_candidate_plan() -> None:
    profile = load_hardware_execution_profile("rtx4090")

    plan = build_pyramid_candidate_plan(
        _profile_space("NVIDIA RTX 4090"), profile=profile
    )
    mapping = validate_p6_candidate_plan(plan, profile=profile)

    assert plan["hardware_target"] == "rtx4090"
    assert plan["execution_backend"] == "tvm_auto"
    assert plan["structure_count"] == 343
    assert plan["candidate_count"] == 686
    assert len(mapping) == 686


def test_h800_profile_normalizes_tracked_vendor_name_to_canonical_target() -> None:
    profile = load_hardware_execution_profile("h800")

    plan = build_pyramid_candidate_plan(
        _profile_space("NVIDIA H800"), profile=profile
    )

    assert plan["hardware_target"] == "h800"
    validate_p6_candidate_plan(plan, profile=profile)


@pytest.mark.parametrize("hardware_name", ["h800_custom", "NVIDIA H800 custom"])
def test_h800_profile_rejects_fuzzy_free_form_hardware_names(
    hardware_name: str,
) -> None:
    with pytest.raises(PyramidSearchSpaceAdapterError, match="hardware profile"):
        build_pyramid_candidate_plan(
            _profile_space(hardware_name),
            profile=load_hardware_execution_profile("h800"),
        )


def test_build_pyramid_candidate_plan_ignores_legacy_candidates_for_formal_space() -> None:
    space = _pyramid_space_with_axis_schema()
    space["software_candidates"] = [
        _candidate("stage1", [64], ["fp16"], group_id="legacy-stage1"),
        _candidate("stage2", [128], ["fp16"], group_id="legacy-stage2"),
        _candidate("stage3", [256], ["fp16"], group_id="legacy-stage3"),
    ]

    plan = build_pyramid_candidate_plan(space)

    assert plan["structure_count"] == 343
    assert plan["candidate_count"] == 686
    assert plan["candidates"][0]["width"] == [16, 32, 64]
    assert all(
        "legacy-stage" not in json.dumps(row, sort_keys=True)
        for row in plan["candidates"]
    )


def test_build_pyramid_candidate_plan_preserves_formal_identity_and_order() -> None:
    plan = build_pyramid_candidate_plan(_pyramid_space_with_axis_schema())
    first = plan["candidates"][0]

    assert first["source_point_ids"] == [
        "backbone.s0:w16",
        "backbone.s1:w32",
        "backbone.s2:w64",
    ]
    assert first["axis_order"] == ["backbone.s0", "backbone.s1", "backbone.s2"]
    assert first["axis_values"] == {
        "backbone.s0": 16,
        "backbone.s1": 32,
        "backbone.s2": 64,
    }
    assert first["formal_identity"]["source_provenance"] == plan["source_provenance"]
    assert [row["formal_candidate_index"] for row in plan["candidates"]] == list(
        range(plan["candidate_count"])
    )


def test_build_pyramid_candidate_plan_is_canonical_when_diagnostics_change() -> None:
    first = _pyramid_space_with_axis_schema()
    second = _pyramid_space_with_axis_schema()
    second["software_candidates"] = list(reversed(second["software_candidates"]))
    for candidate in second["software_candidates"]:
        candidate["software_points"] = list(reversed(candidate["software_points"]))

    assert json.dumps(build_pyramid_candidate_plan(first), sort_keys=True) == json.dumps(
        build_pyramid_candidate_plan(second), sort_keys=True
    )


def test_build_pyramid_candidate_plan_requires_scanner_owned_axis_schema() -> None:
    with pytest.raises(PyramidSearchSpaceAdapterError, match="axis_schema"):
        build_pyramid_candidate_plan(_space())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"schema": "bad"}), "schema"),
        (lambda data: data.update({"model": "codriving"}), "model"),
        (
            lambda data: data.update({"hardware_target": {"name": "orin"}}),
            "hardware profile",
        ),
        (lambda data: data.update({"hardware_candidates": []}), "TVM"),
        (
            lambda data: data["hardware_candidates"][0].update({"backend_scope": "cuda"}),
            "TVM",
        ),
    ],
)
def test_build_pyramid_candidate_plan_rejects_unsupported_envelope(
    mutate: Any, message: str
) -> None:
    payload = _pyramid_space_with_axis_schema()
    mutate(payload)
    with pytest.raises(PyramidSearchSpaceAdapterError, match=message):
        build_pyramid_candidate_plan(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["axis_schema"]["free_axes"][0].update(
                {"dense_stage": None}
            ),
            "stage1, stage2, and stage3",
        ),
        (
            lambda data: data["axis_schema"]["free_axes"][0].update(
                {"dense_stage": "stage2"}
            ),
            "duplicate",
        ),
        (
            lambda data: data["axis_schema"]["free_axes"].append(
                _scanner_axis("backbone.s3", "stage4", [8, 16])
            ),
            "stage1, stage2, and stage3",
        ),
    ],
)
def test_build_pyramid_candidate_plan_rejects_non_pyramid_formal_axis_mapping(
    mutate: Any, message: str
) -> None:
    payload = _pyramid_space_with_axis_schema()
    mutate(payload)
    payload["scanner_structural_axes_digest"] = scanner_structural_axes_digest(
        payload["axis_schema"]["free_axes"]
    )
    with pytest.raises(PyramidSearchSpaceAdapterError, match=message):
        build_pyramid_candidate_plan(payload)


def test_build_pyramid_candidate_plan_accepts_exact_scanner_stage_alias_trio() -> None:
    payload = _pyramid_space_with_axis_schema()
    for axis, dense_stage in zip(
        payload["axis_schema"]["free_axes"],
        ("stage0", "stage1", "stage2"),
        strict=True,
    ):
        axis["dense_stage"] = dense_stage
    payload["scanner_structural_axes_digest"] = scanner_structural_axes_digest(
        payload["axis_schema"]["free_axes"]
    )

    plan = build_pyramid_candidate_plan(payload)

    assert plan["structure_count"] == 343
    assert plan["candidate_count"] == 686
    assert plan["candidates"][0]["width"] == [16, 32, 64]
    assert plan["candidates"][-1]["width"] == [64, 128, 256]


@pytest.mark.parametrize(
    "dense_stages",
    [
        ("stage0", "stage1", "stage3"),
        ("stage0", "stage1", "stage1"),
        ("stage0", "stage1", "stage2", "stage3"),
    ],
)
def test_build_pyramid_candidate_plan_rejects_partial_scanner_stage_aliases(
    dense_stages: tuple[str, ...],
) -> None:
    payload = _pyramid_space_with_axis_schema()
    axes = payload["axis_schema"]["free_axes"]
    for axis, dense_stage in zip(axes, dense_stages, strict=False):
        axis["dense_stage"] = dense_stage
    if len(dense_stages) > len(axes):
        axes.append(_scanner_axis("backbone.s3", dense_stages[-1], [8, 16]))
    payload["scanner_structural_axes_digest"] = scanner_structural_axes_digest(axes)

    with pytest.raises(PyramidSearchSpaceAdapterError):
        build_pyramid_candidate_plan(payload)


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


def test_parse_pyramid_stage_provenance_token_accepts_canonical_token() -> None:
    assert parse_pyramid_stage_provenance_token(
        "p6-stage-provenance-v1:a,b%2Fc"
    ) == ("a", "b/c")
