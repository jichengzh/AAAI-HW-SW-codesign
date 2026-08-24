"""Contract tests for the Stage1-to-Stage2 search-space boundary."""

from __future__ import annotations

from pathlib import Path
from math import prod

import pytest
import yaml

from framework.stage1.graph_scan import scan
from framework.stage1_bridge import SpaceSpec, load_stage2_search_space
from tests.stage1.test_trace_graph_adapter_workflow import _ToyAdapter, _hardware


SCANNER_DIGEST = "a" * 64


def _manifest(*, int8_buildable_align: int = 64) -> dict:
    payload = {
        "schema": "stage1_partition_manifest_demo_v1",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {
            "name": "h800_tvm_demo",
            "ips": {"gpu": {"precisions": ["FP16", "INT8"]}},
        },
        "backend_support": {"precisions": ["FP16", "INT8"]},
        "compression_modes": ["fp16", "int8"],
        "quant_units": [{"id": "all", "legal_precisions": ["FP16", "INT8"]}],
        "view_b1_search_groups": [
            {
                "search_group_id": "pyramid_group",
                "bucket": "pyramid_backbone",
                "widths": [16, 32, 48, 64],
                "round_to": 16,
                "int8_buildable_align": int8_buildable_align,
                "max_rate": 0.75,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": ["pyramid_group"],
            }
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": ["pyramid_group"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }
    payload["scanner_structural_axes"] = [
        _paper_axis("pyramid_group", 64, [16, 32, 48, 64])
    ]
    payload["scanner_structural_axes_digest"] = SCANNER_DIGEST
    return payload


def _write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _paper_axis(axis_id: str, base_width: int, legal_widths: list[int]) -> dict:
    return {
        "axis_id": axis_id,
        "axis_kind": "free",
        "base_width": base_width,
        "legal_widths": legal_widths,
        "member_b1_groups": [
            {
                "b1_group_id": axis_id,
                "module_path": axis_id,
                "canonical_to_member_num": 1,
                "canonical_to_member_den": 1,
                "materializer_param": axis_id,
                "role": "output",
            }
        ],
        "round_to": 8,
        "provenance": {"source": "scanner_contract_test"},
    }


def _legacy_search_group_with_widths() -> dict:
    return {"search_group_id": "legacy.stage1", "bucket": "legacy", "widths": [16, 24, 32], "round_to": 8}


def _legacy_software_candidate() -> dict:
    return {"id": "legacy.stage1", "software_points": [{"id": "p0", "width": 16}]}


def _paper_manifest(model: str, axes: list[list[int]]) -> dict:
    units = []
    scanner_axes = []
    for index, widths in enumerate(axes):
        axis_id = (
            f"backbone.s{index}"
            if model != "fcooper" or index < 3
            else ("neck.deblock" if index == 3 else "neck.output")
        )
        group_id = f"{model}.{axis_id}"
        scanner_axes.append(_paper_axis(axis_id, max(widths), widths))
        units.append(
            {
                "unit": axis_id,
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": [group_id],
            }
        )
    return {
        "schema": "stage1_partition_manifest_v1",
        "model": model,
        "scan_status": "ok",
        "hw_capability": {
            "name": "h800",
            "legal_bits": ["FP16", "INT8"],
            "int8_align": 32,
            "fp16_align": 8,
        },
        "backend_support": {"precisions": ["FP16", "INT8"]},
        "compression_modes": ["fp16", "int8"],
        "quant_units": [{"id": "all", "legal_precisions": ["FP16", "INT8"]}],
        "scanner_structural_axes": scanner_axes,
        "scanner_structural_axes_digest": SCANNER_DIGEST,
        "view_b1_search_groups": [],
        "view_b2_quant_units": units,
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def test_load_stage2_search_space_rejects_missing_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest path"):
        load_stage2_search_space(tmp_path / "missing.yaml")


def test_load_stage2_search_space_rejects_nonpositive_int8_alignment(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "invalid.yaml", _manifest(int8_buildable_align=0))

    with pytest.raises(ValueError, match="int8_buildable_align"):
        load_stage2_search_space(manifest_path)


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("invalid-syntax.yaml", "model: [\n", "invalid manifest YAML"),
        ("not-an-object.yaml", "- item\n", "manifest must decode to an object"),
    ],
)
def test_load_stage2_search_space_validates_yaml_boundary(
    tmp_path: Path, filename: str, contents: str, message: str
) -> None:
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update({"scan_status": "failed"}), "scan_status"),
        (lambda data: data.update({"hw_capability": []}), "hw_capability"),
        (lambda data: data.update({"view_b1_search_groups": {}}), "search sections"),
        (lambda data: data["view_b1_search_groups"][0].update({"round_to": True}), "round_to"),
        (lambda data: data["view_b1_search_groups"][0].update({"widths": [15]}), "divisible"),
        (lambda data: data["view_b1_search_groups"][0].update({"max_rate": 1.0}), "max_rate"),
    ],
)
def test_stage1_manifest_validation_rejects_invalid_search_contracts(
    tmp_path: Path, mutate: object, message: str
) -> None:
    payload = _manifest()
    mutate(payload)  # type: ignore[operator]
    manifest_path = _write_manifest(tmp_path / "invalid-contract.yaml", payload)

    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(manifest_path)


def test_illegal_int8_widths_remain_diagnostic_only(tmp_path: Path) -> None:
    search_space = load_stage2_search_space(_write_manifest(tmp_path / "valid.yaml", _manifest()))

    points = search_space["software_candidates"][0]["software_points"]
    assert any(point["quant_policy"] == "int8" and not point["buildable"] for point in points)
    assert all(
        point["status"] == "diagnostic_only"
        for point in points
        if point["quant_policy"] == "int8" and not point["buildable"]
    )


def test_legacy_manifest_without_int8_alignment_remains_deterministic(tmp_path: Path) -> None:
    payload = _manifest()
    del payload["view_b1_search_groups"][0]["int8_buildable_align"]
    manifest_path = _write_manifest(tmp_path / "legacy.yaml", payload)

    first = load_stage2_search_space(manifest_path)
    second = load_stage2_search_space(manifest_path)
    spec = SpaceSpec.from_manifest(manifest_path)

    assert first == second
    assert first["model_search_policy"]["selected"] == "serial"
    assert spec.coupling_summary()["n_serial"] == 1


def test_stage2_loader_accepts_schema_tagged_real_scan_manifest(tmp_path: Path) -> None:
    manifest = scan(_ToyAdapter(), _hardware(), device="cpu", profile_latency_mode="off")
    path = _write_manifest(tmp_path / "partition.yaml", manifest)

    assert load_stage2_search_space(path)["schema"] == "stage2_search_space_v1"


@pytest.mark.parametrize(
    ("model", "axes", "structure_count"),
    [
        (
            "pyramid_lidar",
            [
                [16, 24, 32, 40, 48, 56, 64],
                [32, 48, 64, 80, 96, 112, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            343,
        ),
        (
            "codriving",
            [
                [16, 24, 32, 40, 48, 56, 64],
                [32, 48, 64, 80, 96, 112, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            343,
        ),
        (
            "fcooper",
            [
                [32, 64],
                [32, 64, 96, 128],
                [32, 64, 96, 128, 160, 192, 224, 256],
                [32, 64, 96, 128],
                [64, 96, 128, 160, 192, 224, 256],
            ],
            1792,
        ),
    ],
)
def test_stage2_loader_exposes_scan_derived_paper_formal_axes(
    tmp_path: Path,
    model: str,
    axes: list[list[int]],
    structure_count: int,
) -> None:
    """Catches diagnostic anchors replacing the paper's formal scan-derived space."""
    search_space = load_stage2_search_space(
        _write_manifest(tmp_path / f"{model}.yaml", _paper_manifest(model, axes))
    )

    formal_axes = search_space["structural_axes"]

    assert [axis["legal_widths"] for axis in formal_axes] == axes
    assert [axis["base_width"] for axis in formal_axes] == [max(axis) for axis in axes]
    assert prod(len(axis["legal_widths"]) for axis in formal_axes) == structure_count
    assert search_space["formal_q_modes"] == ["fp16", "int8"]
    assert search_space["formal_candidate_policy"] == {
        "enumeration": "axis_schema.free_axes_x_formal_q_modes",
        "legacy_views": "diagnostic_only",
        "diagnostic_anchors_drive_formal_space": False,
    }


def test_formal_q_modes_require_backend_support_config_and_graph_units(tmp_path: Path) -> None:
    """Catches q-mode construction that reads only hardware capability."""
    manifest = _manifest()
    manifest["hw_capability"] = {
        "name": "h800",
        "ips": {"gpu": {"precisions": ["FP16", "INT8"]}},
        "quant_constraints": {"bit_widths_w": [8, 16]},
    }
    manifest["backend_support"] = {"precisions": ["FP16"]}
    manifest["compression_modes"] = ["fp16", "int8"]
    manifest["quant_units"] = [{"id": "all", "legal_precisions": ["FP16", "INT8"]}]
    manifest["scanner_structural_axes"] = [
        _paper_axis("stage1", 64, [16, 24, 32, 40, 48, 56, 64])
    ]
    manifest["scanner_structural_axes_digest"] = SCANNER_DIGEST
    path = _write_manifest(tmp_path / "backend-fp16.yaml", manifest)

    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]


def test_formal_q_modes_fail_when_intersection_is_empty(tmp_path: Path) -> None:
    """Catches silent fallback to fp16/int8 when one q-mode source rejects all modes."""
    manifest = _manifest()
    manifest["hw_capability"] = {
        "name": "h800",
        "ips": {"gpu": {"precisions": ["FP16"]}},
        "quant_constraints": {"bit_widths_w": [16]},
    }
    manifest["backend_support"] = {"precisions": ["INT8"]}
    manifest["compression_modes"] = ["int8"]
    manifest["quant_units"] = [{"id": "all", "legal_precisions": ["INT8"]}]
    manifest["scanner_structural_axes"] = [
        _paper_axis("stage1", 64, [16, 24, 32, 40, 48, 56, 64])
    ]
    manifest["scanner_structural_axes_digest"] = SCANNER_DIGEST
    path = _write_manifest(tmp_path / "empty-q.yaml", manifest)

    with pytest.raises(ValueError, match="formal q modes"):
        load_stage2_search_space(path)


def test_formal_q_modes_require_every_quantizable_graph_unit(tmp_path: Path) -> None:
    """Catches a union across graph units admitting a globally illegal mode."""
    manifest = _manifest()
    manifest["quant_units"] = [
        {"id": "backbone", "quantizable": True, "legal_precisions": ["FP16", "INT8"]},
        {"id": "neck", "quantizable": True, "legal_precisions": ["FP16"]},
    ]
    path = _write_manifest(tmp_path / "graph-units.yaml", manifest)

    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]


def test_stage2_search_space_rejects_missing_scanner_structural_axes(tmp_path: Path) -> None:
    """Catches legacy search groups being promoted into formal axes."""
    manifest = _manifest()
    manifest.pop("scanner_structural_axes", None)
    manifest.pop("scanner_structural_axes_digest", None)
    manifest["view_b1_search_groups"] = [_legacy_search_group_with_widths()]
    manifest["software_candidates"] = [_legacy_software_candidate()]
    path = _write_manifest(tmp_path / "legacy-only.yaml", manifest)

    with pytest.raises(ValueError, match="scanner_structural_axes"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_handwritten_structural_axes(tmp_path: Path) -> None:
    """Catches top-level oracle axes being accepted without scanner provenance."""
    manifest = _manifest()
    manifest["structural_axes"] = manifest.pop("scanner_structural_axes")
    manifest.pop("scanner_structural_axes_digest")
    path = _write_manifest(tmp_path / "handwritten-axes.yaml", manifest)

    with pytest.raises(ValueError, match="handwritten structural_axes"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_noncanonical_scanner_digest(tmp_path: Path) -> None:
    """Catches arbitrary provenance labels masquerading as a SHA-256 digest."""
    manifest = _manifest()
    manifest["scanner_structural_axes_digest"] = "z" * 64
    path = _write_manifest(tmp_path / "bad-digest.yaml", manifest)

    with pytest.raises(ValueError, match="provenance digest"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_axis_width_off_alignment(tmp_path: Path) -> None:
    """Catches malformed scanner axes bypassing their declared hardware step."""
    manifest = _manifest()
    manifest["scanner_structural_axes"][0]["legal_widths"] = [16, 18, 64]
    path = _write_manifest(tmp_path / "off-alignment.yaml", manifest)

    with pytest.raises(ValueError, match="divisible by round_to"):
        load_stage2_search_space(path)


def test_stage2_search_space_requires_scanner_provenance_source(tmp_path: Path) -> None:
    """Catches structurally valid axes with no auditable scanner source."""
    manifest = _manifest()
    manifest["scanner_structural_axes"][0]["provenance"] = {"note": "handwritten"}
    path = _write_manifest(tmp_path / "missing-source.yaml", manifest)

    with pytest.raises(ValueError, match="provenance.source"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_nonintegral_member_width_mapping(
    tmp_path: Path,
) -> None:
    """Catches a member ratio that cannot materialize every legal canonical width."""
    manifest = _manifest()
    axis = manifest["scanner_structural_axes"][0]
    axis["base_width"] = 24
    axis["legal_widths"] = [16, 24]
    member = axis["member_b1_groups"][0]
    member["canonical_to_member_num"] = 2
    member["canonical_to_member_den"] = 3
    path = _write_manifest(tmp_path / "nonintegral-member.yaml", manifest)

    with pytest.raises(ValueError, match="integer member width"):
        load_stage2_search_space(path)


@pytest.mark.parametrize("required_field", ["axis_kind", "round_to"])
def test_stage2_search_space_requires_explicit_axis_contract_fields(
    tmp_path: Path,
    required_field: str,
) -> None:
    """Catches missing scanner-owned axis semantics being silently defaulted."""
    manifest = _manifest()
    manifest["scanner_structural_axes"][0].pop(required_field)
    path = _write_manifest(tmp_path / f"missing-{required_field}.yaml", manifest)

    with pytest.raises(ValueError, match=required_field):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_duplicate_scanner_axis_ids(tmp_path: Path) -> None:
    """Catches duplicate formal identities being multiplied as separate axes."""
    manifest = _manifest()
    manifest["scanner_structural_axes"].append(
        _paper_axis("pyramid_group", 128, [32, 64, 96, 128])
    )
    path = _write_manifest(tmp_path / "duplicate-axis-id.yaml", manifest)

    with pytest.raises(ValueError, match="duplicate.*axis_id"):
        load_stage2_search_space(path)


def test_stage2_search_space_canonicalizes_scanner_axis_input_order(tmp_path: Path) -> None:
    """Catches input serialization order changing formal candidate identity."""
    first = _manifest()
    first["scanner_structural_axes"].append(
        _paper_axis("stage2", 128, [32, 64, 96, 128])
    )
    reversed_input = _manifest()
    reversed_input["scanner_structural_axes"].append(
        _paper_axis("stage2", 128, [32, 64, 96, 128])
    )
    reversed_input["scanner_structural_axes"].reverse()

    ordered_space = load_stage2_search_space(
        _write_manifest(tmp_path / "ordered.yaml", first)
    )
    reversed_space = load_stage2_search_space(
        _write_manifest(tmp_path / "reversed.yaml", reversed_input)
    )

    assert reversed_space["structural_axes"] == ordered_space["structural_axes"]
    assert reversed_space["axis_schema"] == ordered_space["axis_schema"]


def test_stage2_search_space_rejects_axis_without_base_width_baseline(
    tmp_path: Path,
) -> None:
    """Catches a formal axis that omits the unpruned checkpoint baseline."""
    manifest = _manifest()
    manifest["scanner_structural_axes"][0]["legal_widths"] = [16, 32, 48]
    path = _write_manifest(tmp_path / "missing-baseline.yaml", manifest)

    with pytest.raises(ValueError, match="end at base_width"):
        load_stage2_search_space(path)


def test_formal_q_modes_reject_nonboolean_quantizable_flag(tmp_path: Path) -> None:
    """Catches truthy strings activating graph quant units through bool coercion."""
    manifest = _manifest()
    manifest["quant_units"] = [
        {
            "id": "backbone",
            "quantizable": "false",
            "legal_precisions": ["FP16", "INT8"],
        }
    ]
    path = _write_manifest(tmp_path / "string-quantizable.yaml", manifest)

    with pytest.raises(ValueError, match="quantizable.*boolean"):
        load_stage2_search_space(path)


def test_formal_q_modes_reject_nonboolean_legacy_quant_unit_flag(tmp_path: Path) -> None:
    """Catches legacy graph units coercing a string quantizable flag to true."""
    manifest = _manifest()
    manifest.pop("quant_units")
    manifest["view_b2_quant_units"][0]["quantizable"] = "false"
    path = _write_manifest(tmp_path / "legacy-string-quantizable.yaml", manifest)

    with pytest.raises(ValueError, match="quantizable.*boolean"):
        load_stage2_search_space(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest.update({"structural_axis_inputs": {"source": "raw"}}),
            "scanner_structural_axes",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"].__setitem__(0, "axis"),
            "entries must be objects",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0].update({"axis_id": ""}),
            "axis_id is required",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0].update(
                {"axis_kind": "derived"}
            ),
            "axis_kind is invalid",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0].update(
                {"legal_widths": []}
            ),
            "legal_widths must be a non-empty list",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0].update(
                {"legal_widths": [32, 16, 64]}
            ),
            "sorted and unique",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0].update(
                {"legal_widths": [16, 32, 80]}
            ),
            "exceed base_width",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0].update(
                {"member_b1_groups": []}
            ),
            "member_b1_groups must be a non-empty list",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0].update(
                {"member_b1_groups": [None]}
            ),
            "entries must be objects",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0][
                "member_b1_groups"
            ][0].update({"role": ""}),
            "entries are incomplete",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0][
                "member_b1_groups"
            ][0].update(
                {"canonical_to_member_num": 2, "canonical_to_member_den": 3}
            ),
            "axis base must map to an integer member width",
        ),
        (
            lambda manifest: manifest["scanner_structural_axes"][0].update(
                {"provenance": {}}
            ),
            "provenance must be a non-empty object",
        ),
    ],
)
def test_scanner_axis_validation_rejects_malformed_contract_branches(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """Each malformed scanner-owned field fails at the Stage1/Stage2 trust boundary."""
    manifest = _manifest()
    mutate(manifest)  # type: ignore[operator]
    path = _write_manifest(tmp_path / "malformed-scanner-axis.yaml", manifest)

    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest.update(
                {"backend_support": {"precisions": "FP16"}}
            ),
            "source must be a sequence",
        ),
        (
            lambda manifest: manifest.update({"compression_modes": {"fp16": True}}),
            "source must be a sequence",
        ),
        (
            lambda manifest: manifest.update({"quant_units": {"all": "FP16"}}),
            "quant_units must be a list",
        ),
        (
            lambda manifest: manifest.update({"quant_units": [None]}),
            "entries must be objects",
        ),
    ],
)
def test_formal_q_mode_sources_reject_malformed_boundaries(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """Malformed q-mode sources are rejected instead of silently normalized."""
    manifest = _manifest()
    mutate(manifest)  # type: ignore[operator]
    path = _write_manifest(tmp_path / "malformed-q-source.yaml", manifest)

    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(path)


def test_nonquantizable_graph_units_reduce_formal_mode_to_fp16(tmp_path: Path) -> None:
    """No active graph quant unit means the graph contributes its fixed FP16 mode."""
    manifest = _manifest()
    manifest["quant_units"] = [
        {"id": "fixed", "quantizable": False, "legal_precisions": ["INT8"]}
    ]
    path = _write_manifest(tmp_path / "fixed-graph-unit.yaml", manifest)

    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]


def test_formal_q_modes_use_legacy_graph_units_when_raw_units_are_absent(
    tmp_path: Path,
) -> None:
    """A valid legacy B2 manifest still supplies graph legality deterministically."""
    manifest = _manifest()
    manifest.pop("quant_units")
    path = _write_manifest(tmp_path / "legacy-graph-units.yaml", manifest)

    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16", "int8"]
