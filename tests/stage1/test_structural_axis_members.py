"""Focused structural-axis member-transform and fixed-axis tests."""

from __future__ import annotations

import pytest

from framework.stage1.structural_axes import derive_structural_axes
from tests.stage1.structural_axis_selector_test_support import (
    set_scanner_owned_binding_fields,
)
from tests.stage1.structural_axis_test_support import (
    codriving_scan_with_neck_binding,
    fcooper_scan_with_independent_neck_bindings,
    pyramid_stage_with_output_c_and_internal_2c,
    stage_with_non_integral_member_width,
    resign_structural_inputs,
    set_scanner_owned_group_width,
)

def test_derive_structural_axis_uses_canonical_base_not_max_internal_width() -> None:
    bundle = derive_structural_axes(pyramid_stage_with_output_c_and_internal_2c())
    axis = bundle.free_axes[0]

    assert axis.axis_id == "backbone.stage3"
    assert axis.base_width == 256
    assert axis.legal_widths == (64, 96, 128, 160, 192, 224, 256)
    assert [
        (member.canonical_to_member_num, member.canonical_to_member_den)
        for member in axis.member_b1_groups
    ] == [(1, 1), (2, 1)]


def test_derive_structural_axes_rejects_non_integral_member_ratio() -> None:
    with pytest.raises(ValueError, match="integer widths"):
        derive_structural_axes(stage_with_non_integral_member_width())


def test_derive_structural_axes_preserves_reduced_rational_member_ratio() -> None:
    scan = pyramid_stage_with_output_c_and_internal_2c()
    set_scanner_owned_group_width(
        scan["structural_axis_inputs"], "stage3.inner", 384
    )

    axis = derive_structural_axes(scan).free_axes[0]

    assert (
        axis.member_b1_groups[1].canonical_to_member_num,
        axis.member_b1_groups[1].canonical_to_member_den,
    ) == (3, 2)


def test_derive_structural_axes_rejects_missing_canonical_base_width() -> None:
    scan = pyramid_stage_with_output_c_and_internal_2c()
    scan["structural_axis_inputs"]["base_widths"] = []
    resign_structural_inputs(scan["structural_axis_inputs"])

    with pytest.raises(ValueError, match="base width authority"):
        derive_structural_axes(scan)


def test_derive_structural_axes_rejects_missing_backend_constraint() -> None:
    scan = pyramid_stage_with_output_c_and_internal_2c()
    scan["structural_axis_inputs"]["backend_constraints"] = []
    resign_structural_inputs(scan["structural_axis_inputs"])

    with pytest.raises(ValueError, match="backend constraint"):
        derive_structural_axes(scan)


def test_derive_structural_axes_rejects_mixed_axis_kind_bindings() -> None:
    scan = codriving_scan_with_neck_binding()
    set_scanner_owned_binding_fields(
        scan["structural_axis_inputs"],
        0,
        axis_id="neck.output",
        role="bad_free_mix",
    )

    with pytest.raises(
        ValueError,
        match=(
            "mixes free and fixed-derived|base width authority.*binding authority|"
            "base width authority duplicate raw source relation"
        ),
    ):
        derive_structural_axes(scan)


def test_codriving_neck_is_fixed_derived_not_free_axis() -> None:
    bundle = derive_structural_axes(codriving_scan_with_neck_binding())

    assert [axis.axis_id for axis in bundle.free_axes] == [
        "backbone.stage1",
        "backbone.stage2",
        "backbone.stage3",
    ]
    assert [axis.axis_id for axis in bundle.fixed_axes] == ["neck.output"]
    assert bundle.fixed_axes[0].provenance["derived_from"] == "backbone.stage3"


def test_fixed_derived_axis_rejects_unknown_source_axis() -> None:
    scan = codriving_scan_with_neck_binding()
    set_scanner_owned_binding_fields(
        scan["structural_axis_inputs"], -1, derived_from="backbone.unknown"
    )

    with pytest.raises(ValueError, match="derived_from.*existing free structural axis"):
        derive_structural_axes(scan)


def test_fixed_derived_axis_rejects_self_reference() -> None:
    scan = codriving_scan_with_neck_binding()
    set_scanner_owned_binding_fields(
        scan["structural_axis_inputs"], -1, derived_from="neck.output"
    )

    with pytest.raises(ValueError, match="derived_from.*existing free structural axis"):
        derive_structural_axes(scan)


def test_fixed_derived_axis_rejects_fixed_source_axis() -> None:
    scan = codriving_scan_with_neck_binding()
    set_scanner_owned_binding_fields(
        scan["structural_axis_inputs"],
        -2,
        axis_kind="fixed_derived",
        derived_from="backbone.stage2",
    )

    with pytest.raises(ValueError, match="derived_from.*existing free structural axis"):
        derive_structural_axes(scan)


def test_fixed_derived_axis_rejects_cycle() -> None:
    scan = codriving_scan_with_neck_binding()
    set_scanner_owned_binding_fields(
        scan["structural_axis_inputs"],
        -2,
        axis_kind="fixed_derived",
        derived_from="neck.output",
    )

    with pytest.raises(ValueError, match="derived_from.*existing free structural axis"):
        derive_structural_axes(scan)


def test_fcooper_neck_interfaces_are_free_when_bindings_are_independent() -> None:
    bundle = derive_structural_axes(fcooper_scan_with_independent_neck_bindings())

    assert [axis.axis_id for axis in bundle.free_axes] == [
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    ]
