"""Contract tests for the Stage1-to-Stage2 search-space boundary."""

from __future__ import annotations

from pathlib import Path
from math import prod

import pytest

from framework.stage1.adapters import ScanScenario
from framework.stage1.graph_scan import scan
from framework.stage1.structural_axis_digest import (
    canonical_digest,
    scanner_structural_axes_digest,
)
from framework.stage1_bridge import (
    SpaceSpec,
    _normalized_precisions,
    load_stage2_search_space,
)
from tests.stage1.stage1_bridge_test_support import (
    bridge_manifest,
    paper_axis,
    seal_scanner_axes,
    write_manifest,
)
from tests.stage1.trace_graph_test_support import (
    ToyAdapter,
    formal_evidence,
    hardware_capability,
)


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
        scanner_axes.append(paper_axis(axis_id, max(widths), widths))
        units.append(
            {
                "unit": axis_id,
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": [group_id],
            }
        )
    manifest = {
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
        "view_b1_search_groups": [],
        "view_b2_quant_units": units,
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }
    seal_scanner_axes(manifest)
    return manifest


def test_load_stage2_search_space_rejects_missing_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest path"):
        load_stage2_search_space(tmp_path / "missing.yaml")


def test_load_stage2_search_space_rejects_nonpositive_int8_alignment(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path / "invalid.yaml", bridge_manifest(int8_buildable_align=0))
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
    payload = bridge_manifest()
    mutate(payload)  # type: ignore[operator]
    manifest_path = write_manifest(tmp_path / "invalid-contract.yaml", payload)
    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(manifest_path)


def test_illegal_int8_widths_remain_diagnostic_only(tmp_path: Path) -> None:
    search_space = load_stage2_search_space(write_manifest(tmp_path / "valid.yaml", bridge_manifest()))
    points = search_space["software_candidates"][0]["software_points"]
    assert any(point["quant_policy"] == "int8" and not point["buildable"] for point in points)
    assert all(
        point["status"] == "diagnostic_only"
        for point in points
        if point["quant_policy"] == "int8" and not point["buildable"]
    )


def test_legacy_manifest_without_int8_alignment_remains_deterministic(tmp_path: Path) -> None:
    payload = bridge_manifest()
    del payload["view_b1_search_groups"][0]["int8_buildable_align"]
    manifest_path = write_manifest(tmp_path / "legacy.yaml", payload)
    first = load_stage2_search_space(manifest_path)
    second = load_stage2_search_space(manifest_path)
    spec = SpaceSpec.from_manifest(manifest_path)
    assert first == second
    assert first["model_search_policy"]["selected"] == "serial"
    assert spec.coupling_summary()["n_serial"] == 1


def test_stage2_loader_accepts_schema_tagged_real_scan_manifest(tmp_path: Path) -> None:
    scenario = ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16", "INT8"),
        compression_modes=("fp16", "int8"),
        graph_quant_unit_policy={
            "backbone": ("FP16", "INT8"),
            "heads": ("FP16", "INT8"),
        },
        alignment={
            "default_round_to": 32,
            "default_max_rate_numerator": 3,
            "default_max_rate_denominator": 4,
        },
    )
    loaded_config, checkpoint = formal_evidence()
    manifest = scan(
        ToyAdapter(),
        hardware_capability(),
        device="cpu",
        profile_latency_mode="off",
        scenario=scenario,
        loaded_config=loaded_config,
        checkpoint_evidence=checkpoint,
    )
    path = write_manifest(tmp_path / "partition.yaml", manifest)
    search_space = load_stage2_search_space(path)
    assert search_space["schema"] == "stage2_search_space_v1"
    assert search_space["structural_axes"][0]["provenance"]["input_digest"] == (
        manifest["formal_scan"]["structural_axis_inputs_digest"]
    )
    assert search_space["scanner_structural_axes_digest"] == (
        manifest["scanner_structural_axes_digest"]
    )
    assert search_space["formal_candidate_policy"]["source"] == (
        "scanner_structural_axes"
    )
    assert manifest["scanner_structural_axes_digest"] == (
        scanner_structural_axes_digest(manifest["scanner_structural_axes"])
    )


@pytest.mark.parametrize("dense_stage", ["", " ", 0, False, []])
def test_stage2_loader_rejects_invalid_scanner_axis_dense_stage(
    tmp_path: Path, dense_stage: object
) -> None:
    manifest = bridge_manifest()
    manifest["scanner_structural_axes"][0]["dense_stage"] = dense_stage
    seal_scanner_axes(manifest)
    with pytest.raises(ValueError, match="dense_stage"):
        load_stage2_search_space(
            write_manifest(tmp_path / "invalid-dense-stage.yaml", manifest)
        )


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
        write_manifest(tmp_path / f"{model}.yaml", _paper_manifest(model, axes))
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
        "source": "scanner_structural_axes",
    }


def test_formal_q_modes_require_backend_support_config_and_graph_units(tmp_path: Path) -> None:
    """Catches q-mode construction that reads only hardware capability."""
    manifest = bridge_manifest()
    manifest["hw_capability"] = {
        "name": "h800",
        "ips": {"gpu": {"precisions": ["FP16", "INT8"]}},
        "quant_constraints": {"bit_widths_w": [8, 16]},
    }
    manifest["backend_support"] = {"precisions": ["FP16"]}
    manifest["compression_modes"] = ["fp16", "int8"]
    manifest["quant_units"] = [{"id": "all", "legal_precisions": ["FP16", "INT8"]}]
    manifest["scanner_structural_axes"] = [
        paper_axis("stage1", 64, [16, 24, 32, 40, 48, 56, 64])
    ]
    seal_scanner_axes(manifest)
    path = write_manifest(tmp_path / "backend-fp16.yaml", manifest)
    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]


def test_formal_q_modes_emit_canonical_four_source_provenance(tmp_path: Path) -> None:
    search_space = load_stage2_search_space(
        write_manifest(tmp_path / "q-provenance.yaml", bridge_manifest())
    )
    payload = {
        "schema": "formal_q_mode_provenance_v1",
        "hardware_target": search_space["hardware_target"],
        "sources": {source: ["fp16", "int8"] for source in ("hardware", "backend", "configured", "graph")},
        "formal_q_modes": search_space["formal_q_modes"],
    }
    assert search_space["formal_q_mode_provenance"] == {
        **payload,
        "digest": canonical_digest(payload),
    }


def test_formal_q_modes_fail_when_intersection_is_empty(tmp_path: Path) -> None:
    """Catches silent fallback to fp16/int8 when one q-mode source rejects all modes."""
    manifest = bridge_manifest()
    manifest["hw_capability"] = {
        "name": "h800",
        "ips": {"gpu": {"precisions": ["FP16"]}},
        "quant_constraints": {"bit_widths_w": [16]},
    }
    manifest["backend_support"] = {"precisions": ["INT8"]}
    manifest["compression_modes"] = ["int8"]
    manifest["quant_units"] = [{"id": "all", "legal_precisions": ["INT8"]}]
    manifest["scanner_structural_axes"] = [
        paper_axis("stage1", 64, [16, 24, 32, 40, 48, 56, 64])
    ]
    seal_scanner_axes(manifest)
    path = write_manifest(tmp_path / "empty-q.yaml", manifest)
    with pytest.raises(ValueError, match="formal q modes"):
        load_stage2_search_space(path)


def test_formal_q_modes_require_every_quantizable_graph_unit(tmp_path: Path) -> None:
    """Catches a union across graph units admitting a globally illegal mode."""
    manifest = bridge_manifest()
    manifest["quant_units"] = [
        {"id": "backbone", "quantizable": True, "legal_precisions": ["FP16", "INT8"]},
        {"id": "neck", "quantizable": True, "legal_precisions": ["FP16"]},
    ]
    path = write_manifest(tmp_path / "graph-units.yaml", manifest)
    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]


def test_stage2_search_space_rejects_missing_scanner_structural_axes(tmp_path: Path) -> None:
    """Catches legacy search groups being promoted into formal axes."""
    manifest = bridge_manifest()
    manifest.pop("scanner_structural_axes", None)
    manifest.pop("scanner_structural_axes_digest", None)
    manifest["view_b1_search_groups"] = [_legacy_search_group_with_widths()]
    manifest["software_candidates"] = [_legacy_software_candidate()]
    path = write_manifest(tmp_path / "legacy-only.yaml", manifest)
    with pytest.raises(ValueError, match="scanner_structural_axes"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_handwritten_structural_axes(tmp_path: Path) -> None:
    """Catches top-level oracle axes being accepted without scanner provenance."""
    manifest = bridge_manifest()
    manifest["structural_axes"] = manifest.pop("scanner_structural_axes")
    manifest.pop("scanner_structural_axes_digest")
    path = write_manifest(tmp_path / "handwritten-axes.yaml", manifest)
    with pytest.raises(ValueError, match="handwritten structural_axes"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_noncanonical_scanner_digest(tmp_path: Path) -> None:
    """Catches arbitrary provenance labels masquerading as a SHA-256 digest."""
    manifest = bridge_manifest()
    manifest["scanner_structural_axes_digest"] = "z" * 64
    path = write_manifest(tmp_path / "bad-digest.yaml", manifest)
    with pytest.raises(ValueError, match="provenance digest"):
        load_stage2_search_space(path)


def test_stage2_rejects_handwritten_axes_with_fake_digest_and_source(
    tmp_path: Path,
) -> None:
    manifest = bridge_manifest()
    manifest["scanner_structural_axes_digest"] = "a" * 64
    manifest["scanner_structural_axes"][0]["provenance"]["source"] = "handwritten"
    path = write_manifest(tmp_path / "fake-axis-authority.yaml", manifest)
    with pytest.raises(ValueError, match="provenance.source|canonical scanner structural axes"):
        load_stage2_search_space(path)


def test_stage2_rejects_axis_tamper_without_updated_canonical_digest(
    tmp_path: Path,
) -> None:
    manifest = bridge_manifest()
    manifest["scanner_structural_axes"][0]["dense_stage"] = "tampered"
    path = write_manifest(tmp_path / "tampered-axis.yaml", manifest)
    with pytest.raises(ValueError, match="canonical scanner structural axes"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_axis_width_off_alignment(tmp_path: Path) -> None:
    """Catches malformed scanner axes bypassing their declared hardware step."""
    manifest = bridge_manifest()
    manifest["scanner_structural_axes"][0]["legal_widths"] = [16, 18, 64]
    path = write_manifest(tmp_path / "off-alignment.yaml", manifest)
    with pytest.raises(ValueError, match="divisible by round_to"):
        load_stage2_search_space(path)


def test_stage2_search_space_requires_scanner_provenance_source(tmp_path: Path) -> None:
    """Catches structurally valid axes with no auditable scanner source."""
    manifest = bridge_manifest()
    manifest["scanner_structural_axes"][0]["provenance"]["source"] = "handwritten"
    path = write_manifest(tmp_path / "missing-source.yaml", manifest)
    with pytest.raises(ValueError, match="provenance.source"):
        load_stage2_search_space(path)


@pytest.mark.parametrize("input_digest", [None, "z" * 64], ids=["missing", "invalid"])
def test_stage2_search_space_requires_canonical_axis_input_digest(
    tmp_path: Path,
    input_digest: str | None,
) -> None:
    manifest = bridge_manifest()
    provenance = manifest["scanner_structural_axes"][0]["provenance"]
    if input_digest is None:
        provenance.pop("input_digest")
    else:
        provenance["input_digest"] = input_digest
    seal_scanner_axes(manifest)
    path = write_manifest(tmp_path / "invalid-input-digest.yaml", manifest)
    with pytest.raises(ValueError, match="input_digest"):
        load_stage2_search_space(path)


@pytest.mark.parametrize("target", ["axis", "member", "provenance"])
def test_stage2_search_space_rejects_unknown_scanner_axis_fields(
    tmp_path: Path,
    target: str,
) -> None:
    manifest = bridge_manifest()
    axis = manifest["scanner_structural_axes"][0]
    if target == "axis":
        axis["expected_count"] = 343
    elif target == "member":
        axis["member_b1_groups"][0]["paper_widths"] = [16, 32, 64]
    else:
        axis["provenance"]["expected_count"] = 343
        seal_scanner_axes(manifest)
    path = write_manifest(tmp_path / f"unknown-{target}.yaml", manifest)
    with pytest.raises(ValueError, match="unknown fields"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_nonintegral_member_width_mapping(
    tmp_path: Path,
) -> None:
    """Catches a member ratio that cannot materialize every legal canonical width."""
    manifest = bridge_manifest()
    axis = manifest["scanner_structural_axes"][0]
    axis["base_width"] = 24
    axis["legal_widths"] = [16, 24]
    member = axis["member_b1_groups"][0]
    member["canonical_to_member_num"] = 2
    member["canonical_to_member_den"] = 3
    path = write_manifest(tmp_path / "nonintegral-member.yaml", manifest)
    with pytest.raises(ValueError, match="integer member width"):
        load_stage2_search_space(path)


@pytest.mark.parametrize("required_field", ["axis_kind", "round_to"])
def test_stage2_search_space_requires_explicit_axis_contract_fields(
    tmp_path: Path,
    required_field: str,
) -> None:
    """Catches missing scanner-owned axis semantics being silently defaulted."""
    manifest = bridge_manifest()
    manifest["scanner_structural_axes"][0].pop(required_field)
    path = write_manifest(tmp_path / f"missing-{required_field}.yaml", manifest)
    with pytest.raises(ValueError, match=required_field):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_duplicate_scanner_axis_ids(tmp_path: Path) -> None:
    """Catches duplicate formal identities being multiplied as separate axes."""
    manifest = bridge_manifest()
    manifest["scanner_structural_axes"].append(
        paper_axis("pyramid_group", 128, [32, 64, 96, 128])
    )
    path = write_manifest(tmp_path / "duplicate-axis-id.yaml", manifest)
    with pytest.raises(ValueError, match="duplicate.*axis_id"):
        load_stage2_search_space(path)


def test_stage2_search_space_rejects_fixed_axis_without_free_source(
    tmp_path: Path,
) -> None:
    manifest = bridge_manifest()
    fixed = paper_axis("neck.output", 64, [16, 32, 48, 64])
    fixed["axis_kind"] = "fixed_derived"
    fixed["provenance"]["derived_from"] = "missing.free.axis"
    manifest["scanner_structural_axes"].append(fixed)
    seal_scanner_axes(manifest)

    with pytest.raises(ValueError, match="derived_from.*free axis"):
        load_stage2_search_space(
            write_manifest(tmp_path / "orphan-fixed-axis.yaml", manifest)
        )


def test_stage2_search_space_canonicalizes_scanner_axis_input_order(tmp_path: Path) -> None:
    """Catches input serialization order changing formal candidate identity."""
    first = bridge_manifest()
    first["scanner_structural_axes"].append(
        paper_axis("stage2", 128, [32, 64, 96, 128])
    )
    reversed_input = bridge_manifest()
    reversed_input["scanner_structural_axes"].append(
        paper_axis("stage2", 128, [32, 64, 96, 128])
    )
    reversed_input["scanner_structural_axes"].reverse()
    seal_scanner_axes(first)
    seal_scanner_axes(reversed_input)

    ordered_space = load_stage2_search_space(
        write_manifest(tmp_path / "ordered.yaml", first)
    )
    reversed_space = load_stage2_search_space(
        write_manifest(tmp_path / "reversed.yaml", reversed_input)
    )
    assert reversed_space["structural_axes"] == ordered_space["structural_axes"]
    assert reversed_space["axis_schema"] == ordered_space["axis_schema"]


def test_stage2_search_space_rejects_axis_without_base_width_baseline(
    tmp_path: Path,
) -> None:
    """Catches a formal axis that omits the unpruned checkpoint baseline."""
    manifest = bridge_manifest()
    manifest["scanner_structural_axes"][0]["legal_widths"] = [16, 32, 48]
    path = write_manifest(tmp_path / "missing-baseline.yaml", manifest)
    with pytest.raises(ValueError, match="end at base_width"):
        load_stage2_search_space(path)


def test_formal_q_modes_reject_nonboolean_quantizable_flag(tmp_path: Path) -> None:
    """Catches truthy strings activating graph quant units through bool coercion."""
    manifest = bridge_manifest()
    manifest["quant_units"] = [
        {
            "id": "backbone",
            "quantizable": "false",
            "legal_precisions": ["FP16", "INT8"],
        }
    ]
    path = write_manifest(tmp_path / "string-quantizable.yaml", manifest)
    with pytest.raises(ValueError, match="quantizable.*boolean"):
        load_stage2_search_space(path)


def test_formal_q_modes_reject_nonboolean_legacy_quant_unit_flag(tmp_path: Path) -> None:
    """Catches legacy graph units coercing a string quantizable flag to true."""
    manifest = bridge_manifest()
    manifest.pop("quant_units")
    manifest["view_b2_quant_units"][0]["quantizable"] = "false"
    path = write_manifest(tmp_path / "legacy-string-quantizable.yaml", manifest)
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
    manifest = bridge_manifest()
    mutate(manifest)  # type: ignore[operator]
    path = write_manifest(tmp_path / "malformed-scanner-axis.yaml", manifest)
    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update(
            {"backend_support": {"precisions": "FP16"}}
        ), "source must be a sequence"),
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
        (lambda manifest: manifest["hw_capability"]["ips"]["gpu"].update(
            {"precisions": [True, "INT8"]}
        ), "precision entries must be strings"),
        (lambda manifest: manifest.update(
            {"backend_support": {"precisions": [True, "INT8"]}}
        ), "precision entries must be strings"),
        (lambda manifest: manifest.update(
            {"compression_modes": [True, "int8"]}
        ), "precision entries must be strings"),
        (lambda manifest: manifest["quant_units"][0].update(
            {"legal_precisions": [True, "INT8"]}
        ), "precision entries must be strings"),
        (lambda manifest: manifest["hw_capability"].update({"legal_bits": True}), "hardware precision source must be a sequence"),
        (lambda manifest: manifest["hw_capability"].update({"legal_bits": 8}), "hardware precision source must be a sequence"),
        (lambda manifest: manifest["hw_capability"]["ips"]["gpu"].update({"precisions": True}), "hardware precision source must be a sequence"),
        (lambda manifest: manifest["hw_capability"]["ips"]["gpu"].update({"precisions": 8}), "hardware precision source must be a sequence"),
        (lambda manifest: manifest["hw_capability"].update({"legal_bits": "INT8"}), "hardware precision source must be a sequence"),
        (lambda manifest: manifest["hw_capability"]["ips"]["gpu"].update({"precisions": "INT8"}), "hardware precision source must be a sequence"),
        (lambda manifest: manifest["hw_capability"].update({"legal_bits": b"INT8"}), "hardware precision source must be a sequence"),
        (lambda manifest: manifest["hw_capability"]["ips"]["gpu"].update({"precisions": b"INT8"}), "hardware precision source must be a sequence"),
    ],
)
def test_formal_q_mode_sources_reject_malformed_boundaries(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    """Malformed q-mode sources are rejected instead of silently normalized."""
    manifest = bridge_manifest()
    mutate(manifest)  # type: ignore[operator]
    path = write_manifest(tmp_path / "malformed-q-source.yaml", manifest)
    with pytest.raises(ValueError, match=message):
        load_stage2_search_space(path)


@pytest.mark.parametrize("source", ["hardware", "backend", "configured", "graph"])
@pytest.mark.parametrize(
    "malformed",
    [None, {"precision": "INT8"}, ["INT8"], b"INT8", 8.0, ""],
    ids=["none", "mapping", "nested-list", "bytes", "float", "empty-string"],
)
def test_formal_q_mode_sources_reject_malformed_precision_members(
    tmp_path: Path,
    source: str,
    malformed: object,
) -> None:
    manifest = bridge_manifest()
    values = [malformed, "INT8"]
    if source == "hardware":
        manifest["hw_capability"]["legal_bits"] = values
    elif source == "backend":
        manifest["backend_support"]["precisions"] = values
    elif source == "configured":
        manifest["compression_modes"] = values
    else:
        manifest["quant_units"][0]["legal_precisions"] = values
    with pytest.raises(ValueError, match="precision entries"):
        load_stage2_search_space(
            write_manifest(tmp_path / f"malformed-{source}.yaml", manifest)
        )


@pytest.mark.parametrize("malformed", [("INT8",), object()], ids=["tuple", "object"])
def test_precision_normalizer_rejects_non_scalar_members(malformed: object) -> None:
    with pytest.raises(ValueError, match="precision entries"):
        _normalized_precisions([malformed, "INT8"])


def test_precision_normalizer_preserves_legal_string_and_integer_aliases() -> None:
    assert _normalized_precisions(
        ["FP16", "float16", "16", 16, "INT8", "8", 8, 32, "int4"]
    ) == {"fp16", "int8"}


def test_nonquantizable_graph_units_reduce_formal_mode_to_fp16(tmp_path: Path) -> None:
    """No active graph quant unit means the graph contributes its fixed FP16 mode."""
    manifest = bridge_manifest()
    manifest["quant_units"] = [
        {"id": "fixed", "quantizable": False, "legal_precisions": ["INT8"]}
    ]
    path = write_manifest(tmp_path / "fixed-graph-unit.yaml", manifest)
    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16"]


def test_formal_q_modes_use_legacy_graph_units_when_raw_units_are_absent(
    tmp_path: Path,
) -> None:
    """A valid legacy B2 manifest still supplies graph legality deterministically."""
    manifest = bridge_manifest()
    manifest.pop("quant_units")
    path = write_manifest(tmp_path / "legacy-graph-units.yaml", manifest)
    assert load_stage2_search_space(path)["formal_q_modes"] == ["fp16", "int8"]
