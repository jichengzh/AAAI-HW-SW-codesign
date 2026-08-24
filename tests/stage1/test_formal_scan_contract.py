"""Focused contracts for Stage1 formal-scan orchestration helpers."""

from __future__ import annotations

import inspect

import pytest

from framework.stage1.adapters import ScanScenario
from framework.stage1.formal_scan_contract import (
    formal_manifest_fields,
    validate_formal_request,
)
from framework.stage1.graph_scan import extract_prune_groups, scan
from framework.stage1.structural_axes import derive_structural_axes


def _scenario() -> ScanScenario:
    return ScanScenario(
        hardware_precisions=("FP16", "INT8"),
        backend_precisions=("FP16",),
        compression_modes=("fp16",),
        graph_quant_unit_policy={"backbone": ("FP16",)},
        alignment={"default_round_to": 16, "default_min_width": 16},
    )


def test_validate_formal_request_requires_both_scanner_evidence_mappings() -> None:
    with pytest.raises(ValueError, match="loaded_config.*checkpoint_evidence"):
        validate_formal_request(_scenario(), {}, None)

    assert validate_formal_request(None, None, None) is False
    assert validate_formal_request(_scenario(), {}, {}) is True


def test_validate_formal_request_rejects_non_scenario_objects() -> None:
    with pytest.raises(TypeError, match="ScanScenario"):
        validate_formal_request(  # type: ignore[arg-type]
            {"hardware_precisions": ["FP16"]}, {}, {}
        )


def test_formal_manifest_fields_do_not_fallback_missing_graph_policy() -> None:
    fields = formal_manifest_fields(
        _scenario(),
        [
            {"unit": "backbone", "quantizable": True},
            {"unit": "heads", "quantizable": True},
        ],
    )

    assert fields["quant_units"] == [
        {"id": "backbone", "legal_precisions": ["FP16"], "quantizable": True},
        {"id": "heads", "legal_precisions": [], "quantizable": True},
    ]


@pytest.mark.parametrize(
    "function",
    [scan, extract_prune_groups, derive_structural_axes],
)
def test_changed_stage1_functions_stay_within_50_lines(function: object) -> None:
    assert len(inspect.getsource(function).splitlines()) <= 50
