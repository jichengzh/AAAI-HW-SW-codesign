from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from framework.stage6.p6_external_training_binding_v1 import (
    bind_external_training_contract,
    validate_external_training_binding,
)
from framework.stage6.p6_history_recipe_profiles_v1 import RECIPE_V2, SHARED_SOURCE_PATH_KEYS
from framework.stage6.p6_history_training_contract_v1 import (
    P6HistoryTrainingContractError,
    public_safe_contract_projection,
    validate_recipe_v2_group_training_contract,
    validate_recipe_v2_training_template,
)


def _recipe() -> dict[str, Any]:
    return {
        "schema_version": RECIPE_V2,
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
        "group_id_template": "pyramid|{stage1_width}x{stage2_width}x{stage3_width}",
        "artifact_id_template": "pyramid-{stage1_width}-{stage2_width}-{stage3_width}",
        "shared_source_path_templates": {
            key: f"materialized/{{artifact_id}}/{key}" for key in SHARED_SOURCE_PATH_KEYS
        },
    }


def _bound_contract(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    code = tmp_path / "code"
    code.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    assets = tmp_path / "operator-assets"
    assets.mkdir()
    dataset = assets / "dataset"
    dataset.mkdir()
    checkpoint = assets / "base.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    config = assets / "pyramid.py"
    config.write_text(
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )
    raw = {
        "schema_version": "p6_external_training_binding_v1",
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
        "dataset_root": str(dataset),
        "base_checkpoint_path": str(checkpoint),
        "base_checkpoint_sha256": None,
        "pyramid_config_path": str(config),
        "pyramid_config_sha256": None,
        "training_parameters": {
            "training_mode": "finetune_selected_width",
            "epochs": 2,
            "target_epoch": 9,
            "seed": 1,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "batch_size": 1,
            "dataset_split": "trainval",
            "checkpoint_selection": "best",
            "freeze_policy": "partial",
            "groups": 3,
            "width_per_group": 5,
            "base_stage_widths": [3, 5, 7],
        },
    }
    binding = validate_external_training_binding(
        raw, code_toolchain_root=code, local_output_root=output, reserved_paths=()
    )
    contract = bind_external_training_contract(
        {
            "schema_version": "stage5_source_contract_v1",
            "group_id": "pyramid|16x32x64",
            "model": "pyramid",
            "width": [16, 32, 64],
            "artifact_id": "pyramid-16-32-64",
            "source_status": "ready",
            "source_evidence_sha256": "a" * 64,
            "dynamic_materialization_recipe": _recipe(),
        },
        binding,
    )
    return contract, code, output


def test_template_consumes_nested_external_binding_outside_code_root(tmp_path: Path) -> None:
    contract, code, _ = _bound_contract(tmp_path)
    validated = validate_recipe_v2_training_template(contract, private_root=code)
    assert validated == contract and validated is not contract
    assert "external_training_binding" in validated
    assert "dataset_root" not in validated


def test_template_rejects_flat_or_invalid_external_binding(tmp_path: Path) -> None:
    contract, code, _ = _bound_contract(tmp_path)
    for mutation in ("flat", "missing", "unknown"):
        candidate = copy.deepcopy(contract)
        if mutation == "flat":
            candidate["dataset_root"] = "/private/dataset"
        elif mutation == "missing":
            candidate.pop("external_training_binding")
        else:
            candidate["external_training_binding"]["unknown"] = "x"
        with pytest.raises(P6HistoryTrainingContractError) as captured:
            validate_recipe_v2_training_template(candidate, private_root=code)
        assert captured.value.category == "history_execution_invalid"


def test_group_contract_delegates_nested_binding_and_checks_outputs(tmp_path: Path) -> None:
    contract, code, output = _bound_contract(tmp_path)
    contract.pop("dynamic_materialization_recipe")
    contract["stage_widths"] = {"stage1_width": 16, "stage2_width": 32, "stage3_width": 64}
    contract["shared_source_paths"] = {
        key: str(output / "materialized" / key) for key in SHARED_SOURCE_PATH_KEYS
    }
    assert (
        validate_recipe_v2_group_training_contract(
            contract, private_root=code, local_output_root=output, group_id="pyramid|16x32x64"
        )
        == contract
    )
    contract["shared_source_paths"]["checkpoint_path"] = str(tmp_path / "outside")
    with pytest.raises(P6HistoryTrainingContractError):
        validate_recipe_v2_group_training_contract(
            contract, private_root=code, local_output_root=output, group_id="pyramid|16x32x64"
        )


def test_public_projection_excludes_external_binding(tmp_path: Path) -> None:
    contract, _, _ = _bound_contract(tmp_path)
    projection = public_safe_contract_projection(contract)
    assert "external_training_binding" not in projection
    assert set(projection) == {
        "schema_version",
        "group_id",
        "model",
        "width",
        "artifact_id",
        "source_status",
        "source_evidence_sha256",
    }
