"""Test-only migration helpers for scanner-owned Stage2 inputs."""

from __future__ import annotations

from copy import deepcopy

from framework.stage1.structural_axes import axis_to_scanner_dict
from framework.stage1.structural_axis_digest import scanner_structural_axes_digest
from tests.stage1.structural_axis_test_support import paper_axis_bundle


def with_scanner_owned_contract(
    manifest: dict[str, object], paper_name: str
) -> dict[str, object]:
    """Return a demo manifest migrated to the current scanner-owned contract."""
    evidence, inputs, bundle = paper_axis_bundle(paper_name)
    scenario = evidence["scan_scenario"]
    axes = [axis_to_scanner_dict(axis, inputs) for axis in bundle.axes]
    return {
        **deepcopy(manifest),
        "backend_support": {"precisions": scenario["backend_precisions"]},
        "compression_modes": scenario["compression_modes"],
        "quant_units": [
            {"id": unit, "legal_precisions": precisions}
            for unit, precisions in scenario["graph_quant_unit_policy"].items()
        ],
        "scanner_structural_axes": axes,
        "scanner_structural_axes_digest": scanner_structural_axes_digest(axes),
    }
