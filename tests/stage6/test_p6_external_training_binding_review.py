from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from framework.stage6.p6_external_training_binding_v1 import (
    P6ExternalTrainingBindingError,
    bind_external_training_contract,
    external_training_binding_from_contract,
    load_external_training_binding,
    validate_external_training_binding,
)
from framework.stage6.p6_history_training_contract_v1 import (
    P6HistoryTrainingContractError,
    validate_recipe_v2_group_training_contract,
    validate_recipe_v2_training_template,
)
from framework.stage6.p6_history_recipe_profiles_v1 import RECIPE_V2, SHARED_SOURCE_PATH_KEYS


def _fixture(tmp_path: Path) -> dict[str, Any]:
    code = tmp_path / "code"
    output = tmp_path / "output"
    dataset = tmp_path / "dataset"
    stable = tmp_path / "stable"
    for root in (code, output, dataset, stable):
        root.mkdir()
    checkpoint = stable / "base.ckpt"
    config = stable / "pyramid.py"
    checkpoint.write_bytes(b"checkpoint")
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
            "training_mode": "finetune",
            "epochs": 1,
            "target_epoch": 9,
            "seed": 1,
            "optimizer": "adamw",
            "learning_rate": 0.1,
            "batch_size": 1,
            "dataset_split": "train",
            "checkpoint_selection": "best",
            "freeze_policy": "none",
            "groups": 3,
            "width_per_group": 5,
            "base_stage_widths": [3, 5, 7],
        },
    }
    return {
        "raw": raw,
        "code": code,
        "output": output,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "config": config,
    }


def _binding(fixture: dict[str, Any]) -> Any:
    return validate_external_training_binding(
        fixture["raw"],
        code_toolchain_root=fixture["code"],
        local_output_root=fixture["output"],
        reserved_paths=(),
    )


def _contract(fixture: dict[str, Any]) -> dict[str, Any]:
    return bind_external_training_contract(
        {
            "schema_version": "stage5_source_contract_v1",
            "group_id": "pyramid|16x32x64",
            "width": [16, 32, 64],
            "stage_widths": {
                "stage1_width": 16,
                "stage2_width": 32,
                "stage3_width": 64,
            },
            "shared_source_paths": {
                key: str(fixture["output"] / key) for key in SHARED_SOURCE_PATH_KEYS
            },
            "dynamic_materialization_recipe": {"schema_version": RECIPE_V2},
        },
        _binding(fixture),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "required_false",
        "wrong_source",
        "null_digest",
        "mismatched_digest",
        "bool_seed",
        "bool_learning_rate",
        "zero_batch_size",
        "base_width_bool",
    ),
)
def test_nested_binding_rejects_semantically_invalid_normalized_values(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    contract = _contract(fixture)
    nested = contract["external_training_binding"]
    if mutation == "required_false":
        nested["training_required"] = False
    elif mutation == "wrong_source":
        nested["training_source_kind"] = "reuse"
    elif mutation == "null_digest":
        nested["base_checkpoint_sha256"] = None
    elif mutation == "mismatched_digest":
        nested["pyramid_config_sha256"] = "0" * 64
    elif mutation == "bool_seed":
        nested["training_parameters"]["seed"] = True
    elif mutation == "bool_learning_rate":
        nested["training_parameters"]["learning_rate"] = True
    elif mutation == "base_width_bool":
        nested["training_parameters"]["base_stage_widths"] = [True, 5, 7]
    else:
        nested["training_parameters"]["batch_size"] = 0

    with pytest.raises(P6ExternalTrainingBindingError):
        external_training_binding_from_contract(contract)
    with pytest.raises(P6HistoryTrainingContractError):
        validate_recipe_v2_training_template(contract, private_root=fixture["code"])
    contract.pop("dynamic_materialization_recipe")
    with pytest.raises(P6HistoryTrainingContractError):
        validate_recipe_v2_group_training_contract(
            contract,
            private_root=fixture["code"],
            local_output_root=fixture["output"],
            group_id="pyramid|16x32x64",
        )


def test_nested_binding_accepts_valid_normalized_mapping(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract = _contract(fixture)

    extracted = external_training_binding_from_contract(copy.deepcopy(contract))

    assert extracted["base_checkpoint_sha256"]
    assert validate_recipe_v2_training_template(contract, private_root=fixture["code"])
    grouped = copy.deepcopy(contract)
    grouped.pop("dynamic_materialization_recipe")
    assert validate_recipe_v2_group_training_contract(
        grouped,
        private_root=fixture["code"],
        local_output_root=fixture["output"],
        group_id="pyramid|16x32x64",
    )


def test_dataset_at_unignored_git_worktree_root_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    repository = tmp_path / "dataset-repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    fixture["raw"]["dataset_root"] = str(repository)

    with pytest.raises(P6ExternalTrainingBindingError):
        _binding(fixture)


def test_ignored_dataset_root_is_accepted(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    repository = tmp_path / "operator-repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    (repository / ".gitignore").write_text("assets/\n", encoding="utf-8")
    dataset = repository / "assets"
    dataset.mkdir()
    fixture["raw"]["dataset_root"] = str(dataset)

    assert _binding(fixture).dataset_root == dataset


@pytest.mark.parametrize(
    "mutation",
    (
        "null_checkpoint",
        "null_config",
        "missing_dataset",
        "missing_checkpoint",
        "missing_config",
        "config_is_directory",
        "checkpoint_equals_config",
        "config_ancestor_symlink",
    ),
)
def test_additional_external_path_matrix(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path)
    raw = fixture["raw"]
    if mutation == "null_checkpoint":
        raw["base_checkpoint_path"] = None
    elif mutation == "null_config":
        raw["pyramid_config_path"] = None
    elif mutation == "missing_dataset":
        raw["dataset_root"] = str(tmp_path / "missing-dataset")
    elif mutation == "missing_checkpoint":
        raw["base_checkpoint_path"] = str(tmp_path / "missing-checkpoint")
    elif mutation == "missing_config":
        raw["pyramid_config_path"] = str(tmp_path / "missing-config")
    elif mutation == "config_is_directory":
        raw["pyramid_config_path"] = str(fixture["dataset"])
    elif mutation == "checkpoint_equals_config":
        raw["pyramid_config_path"] = raw["base_checkpoint_path"]
    else:
        linked = tmp_path / "linked-stable"
        linked.symlink_to(fixture["config"].parent, target_is_directory=True)
        raw["pyramid_config_path"] = str(linked / fixture["config"].name)

    with pytest.raises(P6ExternalTrainingBindingError):
        _binding(fixture)


@pytest.mark.parametrize("key", ("dataset_root", "base_checkpoint_path", "pyramid_config_path"))
def test_each_unignored_git_asset_is_rejected(tmp_path: Path, key: str) -> None:
    fixture = _fixture(tmp_path)
    repository = tmp_path / f"unignored-{key}"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    if key == "dataset_root":
        asset = repository / "dataset"
        asset.mkdir()
    else:
        asset = repository / f"{key}.txt"
        asset.write_bytes(b"asset")
    fixture["raw"][key] = str(asset)

    with pytest.raises(P6ExternalTrainingBindingError):
        _binding(fixture)


@pytest.mark.parametrize(
    "mutation",
    (
        "relative",
        "symlink",
        "symlink_parent",
        "directory",
        "invalid_utf8",
        "oversize",
        "tracked",
    ),
)
def test_loader_rejects_private_or_unsafe_inputs(tmp_path: Path, mutation: str) -> None:
    if mutation == "relative":
        path = Path("relative.yaml")
    elif mutation == "directory":
        path = tmp_path / "directory"
        path.mkdir()
    else:
        path = tmp_path / "binding.yaml"
        if mutation == "symlink":
            target = tmp_path / "target.yaml"
            target.write_text("schema: x\n", encoding="utf-8")
            path.symlink_to(target)
        elif mutation == "symlink_parent":
            target_parent = tmp_path / "target-parent"
            target_parent.mkdir()
            (target_parent / "binding.yaml").write_text("schema: x\n", encoding="utf-8")
            linked_parent = tmp_path / "linked-parent"
            linked_parent.symlink_to(target_parent, target_is_directory=True)
            path = linked_parent / "binding.yaml"
        elif mutation == "invalid_utf8":
            path.write_bytes(b"\xff")
        elif mutation == "oversize":
            path.write_bytes(b"x" * (1024 * 1024 + 1))
        elif mutation == "tracked":
            repository = tmp_path / "tracked-repository"
            repository.mkdir()
            subprocess.run(("git", "init", "-q", str(repository)), check=True)
            path = repository / "binding.yaml"
            path.write_text("schema: x\n", encoding="utf-8")
            subprocess.run(("git", "add", "binding.yaml"), cwd=repository, check=True)
        else:
            raise AssertionError(mutation)

    with pytest.raises(P6ExternalTrainingBindingError):
        load_external_training_binding(path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read denied files")
def test_loader_rejects_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "unreadable.yaml"
    path.write_text("schema: x\n", encoding="utf-8")
    path.chmod(0)
    try:
        with pytest.raises(P6ExternalTrainingBindingError):
            load_external_training_binding(path)
    finally:
        path.chmod(0o600)


@pytest.mark.parametrize("asset", ("dataset", "checkpoint", "config"))
def test_external_binding_rejects_unreadable_assets(tmp_path: Path, asset: str) -> None:
    fixture = _fixture(tmp_path)
    path = fixture[asset]
    path.chmod(0)
    required_access = os.R_OK | (os.X_OK if asset == "dataset" else 0)
    if os.geteuid() == 0 and os.access(path, required_access):
        path.chmod(0o700 if asset == "dataset" else 0o600)
        pytest.skip("root can read denied test asset")
    try:
        with pytest.raises(P6ExternalTrainingBindingError):
            _binding(fixture)
    finally:
        path.chmod(0o700 if asset == "dataset" else 0o600)
