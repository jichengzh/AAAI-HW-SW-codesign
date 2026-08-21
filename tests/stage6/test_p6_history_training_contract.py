from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from framework.stage6.p6_history_recipe_profiles_v1 import (
    RECIPE_V2,
    SHARED_SOURCE_PATH_KEYS,
)
from framework.stage6.p6_history_training_contract_v1 import (
    REQUIRED_TRAINING_PARAMETER_KEYS,
    P6HistoryTrainingContractError,
    public_safe_contract_projection,
    validate_recipe_v2_group_training_contract,
    validate_recipe_v2_training_template,
)


def _sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _recipe_v2() -> dict[str, Any]:
    return {
        "schema_version": RECIPE_V2,
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
        "group_id_template": "pyramid|{stage1_width}x{stage2_width}x{stage3_width}",
        "artifact_id_template": "pyramid-{stage1_width}-{stage2_width}-{stage3_width}",
        "shared_source_path_templates": {
            key: f"materialized/{{artifact_id}}/{key}" for key in SHARED_SOURCE_PATH_KEYS
        },
    }


def _private_root_with_training_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "private-history"
    (root / "checkpoints").mkdir(parents=True)
    (root / "datasets" / "coptv2x").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "checkpoints" / "base.ckpt").write_text("base\n", encoding="utf-8")
    (root / "configs" / "pyramid.py").write_text("config\n", encoding="utf-8")
    return root


def _training_template(private_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "stage5_source_contract_v1",
        "group_id": "pyramid|16x32x64",
        "model": "pyramid",
        "width": [16, 32, 64],
        "artifact_id": "pyramid-16-32-64",
        "source_status": "ready",
        "source_evidence_sha256": _sha({"evidence": "training"}),
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
        "base_checkpoint_path": str(private_root / "checkpoints" / "base.ckpt"),
        "dataset_root": str(private_root / "datasets" / "coptv2x"),
        "pyramid_config_path": str(private_root / "configs" / "pyramid.py"),
        "training_parameters": {
            "training_mode": "finetune_selected_width",
            "epochs": 2,
            "seed": 20260821,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "batch_size": 1,
            "dataset_split": "trainval_coptv2x",
            "checkpoint_selection": "best_ap70",
            "freeze_policy": "pyramid_backbone_partial",
        },
        "dynamic_materialization_recipe": _recipe_v2(),
    }


def _mutated_training_template(private_root: Path, mutation: str) -> dict[str, Any]:
    template = _training_template(private_root)
    if mutation == "missing_training_required":
        template.pop("training_required")
    elif mutation == "false_training_required":
        template["training_required"] = False
    elif mutation == "missing_training_source_kind":
        template.pop("training_source_kind")
    elif mutation == "missing_base_checkpoint_path":
        template.pop("base_checkpoint_path")
    elif mutation == "missing_dataset_root":
        template.pop("dataset_root")
    elif mutation == "missing_pyramid_config_path":
        template.pop("pyramid_config_path")
    elif mutation == "missing_training_parameters":
        template.pop("training_parameters")
    elif mutation == "empty_training_parameters":
        template["training_parameters"] = {}
    elif mutation == "missing_training_mode":
        template["training_parameters"].pop("training_mode")
    elif mutation == "zero_epochs":
        template["training_parameters"]["epochs"] = 0
    elif mutation == "negative_learning_rate":
        template["training_parameters"]["learning_rate"] = -0.1
    elif mutation == "zero_batch_size":
        template["training_parameters"]["batch_size"] = 0
    elif mutation == "outside_private_root":
        template["base_checkpoint_path"] = str(private_root.parent / "outside.ckpt")
    elif mutation == "symlinked_training_input":
        link = private_root / "checkpoints" / "linked.ckpt"
        link.symlink_to(private_root / "checkpoints" / "base.ckpt")
        template["base_checkpoint_path"] = str(link)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    return template


def test_training_template_requires_real_training_and_all_static_fields(tmp_path: Path) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    template = _training_template(private_root)

    validated = validate_recipe_v2_training_template(template, private_root=private_root)

    assert validated["training_required"] is True
    assert validated["training_source_kind"] == "selected_candidate_finetune"
    assert set(validated["training_parameters"]) == set(REQUIRED_TRAINING_PARAMETER_KEYS)
    assert validated["training_parameters"]["epochs"] == 2
    assert validated["training_parameters"]["batch_size"] == 1
    assert validated == template
    assert validated is not template
    assert validated["training_parameters"] is not template["training_parameters"]


@pytest.mark.parametrize("source_kind", ["selected_candidate_finetune"])
def test_training_template_accepts_canonical_selected_candidate_training_kind(
    tmp_path: Path, source_kind: str
) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    template = _training_template(private_root)
    template["training_source_kind"] = source_kind

    validated = validate_recipe_v2_training_template(template, private_root=private_root)

    assert validated["training_source_kind"] == source_kind


@pytest.mark.parametrize(
    "source_kind",
    [
        "checkpoint_already_exists",
        "pretrained_checkpoint_reuse",
        "reuse_pretrained_checkpoint",
        "selected_candidate_checkpoint_reuse",
        ["selected_candidate_finetune"],
    ],
)
def test_training_template_rejects_checkpoint_reuse_source_kind(
    tmp_path: Path, source_kind: object
) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    template = _training_template(private_root)
    template["training_source_kind"] = source_kind

    with pytest.raises(P6HistoryTrainingContractError) as captured:
        validate_recipe_v2_training_template(template, private_root=private_root)

    assert captured.value.category == "history_execution_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_training_required",
        "false_training_required",
        "missing_training_source_kind",
        "missing_base_checkpoint_path",
        "missing_dataset_root",
        "missing_pyramid_config_path",
        "missing_training_parameters",
        "empty_training_parameters",
        "missing_training_mode",
        "zero_epochs",
        "negative_learning_rate",
        "zero_batch_size",
        "outside_private_root",
        "symlinked_training_input",
    ],
)
def test_training_template_rejects_incomplete_or_unsafe_static_contract(
    tmp_path: Path, mutation: str
) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    template = _mutated_training_template(private_root, mutation)

    with pytest.raises(P6HistoryTrainingContractError) as captured:
        validate_recipe_v2_training_template(template, private_root=private_root)

    assert captured.value.category == "history_execution_invalid"


def test_group_contract_requires_rendered_private_and_local_paths(tmp_path: Path) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    contract = _training_template(private_root)
    contract.pop("dynamic_materialization_recipe")
    contract["stage_widths"] = {
        "stage1_width": 16,
        "stage2_width": 32,
        "stage3_width": 64,
    }
    contract["shared_source_paths"] = {
        key: str(local_output_root / "materialized" / key)
        for key in SHARED_SOURCE_PATH_KEYS
    }

    validated = validate_recipe_v2_group_training_contract(
        contract,
        private_root=private_root,
        local_output_root=local_output_root,
        group_id="pyramid|16x32x64",
    )

    assert validated == contract
    assert validated is not contract
    assert validated["shared_source_paths"] is not contract["shared_source_paths"]


@pytest.mark.parametrize("mutation", ["outside_output", "missing_output", "wrong_width"])
def test_group_contract_rejects_unsafe_rendered_fields(tmp_path: Path, mutation: str) -> None:
    private_root = _private_root_with_training_inputs(tmp_path)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    contract = _training_template(private_root)
    contract.pop("dynamic_materialization_recipe")
    contract["stage_widths"] = {
        "stage1_width": 16,
        "stage2_width": 32,
        "stage3_width": 64,
    }
    contract["shared_source_paths"] = {
        key: str(local_output_root / "materialized" / key)
        for key in SHARED_SOURCE_PATH_KEYS
    }
    if mutation == "outside_output":
        contract["shared_source_paths"]["checkpoint_path"] = str(tmp_path / "outside")
    elif mutation == "missing_output":
        contract["shared_source_paths"].pop("checkpoint_path")
    else:
        contract["stage_widths"]["stage1_width"] = 99

    with pytest.raises(P6HistoryTrainingContractError) as captured:
        validate_recipe_v2_group_training_contract(
            contract,
            private_root=private_root,
            local_output_root=local_output_root,
            group_id="pyramid|16x32x64",
        )

    assert captured.value.category == "history_execution_invalid"


def test_public_safe_contract_projection_excludes_private_training_values(tmp_path: Path) -> None:
    template = _training_template(_private_root_with_training_inputs(tmp_path))

    projection = public_safe_contract_projection(copy.deepcopy(template))

    assert set(projection) == {
        "schema_version",
        "group_id",
        "model",
        "width",
        "artifact_id",
        "source_status",
        "source_evidence_sha256",
    }
    assert not {
        "base_checkpoint_path",
        "dataset_root",
        "pyramid_config_path",
        "training_parameters",
        "training_required",
        "training_source_kind",
    }.intersection(projection)
