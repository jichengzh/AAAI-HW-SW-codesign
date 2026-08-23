from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from framework.stage6.p6_external_training_binding_v1 import (
    P6ExternalTrainingBindingError,
    bind_external_training_contract,
    external_training_binding_from_contract,
    external_training_binding_to_mapping,
    load_external_training_binding,
    public_safe_external_training_projection,
    validate_external_training_binding,
)
from framework.stage6.p6_history_training_contract_v1 import (
    REQUIRED_TRAINING_PARAMETER_KEYS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_TOKEN = "PRIVATE-EXTERNAL-TOKEN"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Any]:
    code = tmp_path / "code-root"
    output = tmp_path / "output-root"
    dataset = tmp_path / "operator-dataset"
    stable = tmp_path / "operator-stable-files"
    for path in (code, output, dataset, stable):
        path.mkdir()
    checkpoint = stable / "base.ckpt"
    config = stable / "pyramid.py"
    checkpoint.write_bytes(b"checkpoint\x00")
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
            "seed": 0,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "batch_size": 1,
            "dataset_split": "trainval_coptv2x",
            "checkpoint_selection": "best_ap70",
            "freeze_policy": "pyramid_backbone_partial",
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
        "reserved": [output / "result.json"],
    }


def _validate(fixture: dict[str, Any]) -> Any:
    return validate_external_training_binding(
        fixture["raw"],
        code_toolchain_root=fixture["code"],
        local_output_root=fixture["output"],
        reserved_paths=fixture["reserved"],
    )


def test_external_binding_accepts_unrelated_roots_and_computes_null_digests(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    binding = _validate(fixture)
    normalized = external_training_binding_to_mapping(binding)

    assert binding.dataset_root == fixture["dataset"].resolve(strict=True)
    assert binding.base_checkpoint_path == fixture["checkpoint"].resolve(strict=True)
    assert binding.pyramid_config_path == fixture["config"].resolve(strict=True)
    assert normalized["base_checkpoint_sha256"] == _sha(fixture["checkpoint"])
    assert normalized["pyramid_config_sha256"] == _sha(fixture["config"])
    assert tuple(key for key, _ in binding.training_parameters) == REQUIRED_TRAINING_PARAMETER_KEYS
    assert not binding.dataset_root.is_relative_to(fixture["code"])


def test_external_binding_requires_structural_pruning_parameters(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["raw"]["training_parameters"].update(
        groups=3,
        width_per_group=5,
    )

    binding = _validate(fixture)

    assert dict(binding.training_parameters)["groups"] == 3
    assert dict(binding.training_parameters)["width_per_group"] == 5


def test_external_binding_requires_distinct_target_epoch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["raw"]["training_parameters"]["epochs"] = 8
    fixture["raw"]["training_parameters"]["target_epoch"] = 9

    binding = _validate(fixture)

    parameters = dict(binding.training_parameters)
    assert parameters["epochs"] == 8
    assert parameters["target_epoch"] == 9


def test_external_binding_requires_explicit_prune_base_stage_widths(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["config"].write_text(
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )
    fixture["raw"]["training_parameters"]["base_stage_widths"] = [3, 5, 7]

    binding = _validate(fixture)

    parameters = dict(binding.training_parameters)
    assert parameters["base_stage_widths"] == (3, 5, 7)
    assert external_training_binding_to_mapping(binding)["training_parameters"][
        "base_stage_widths"
    ] == [3, 5, 7]


def test_external_binding_reads_base_widths_without_constructing_legacy_noise(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["config"].write_text(
        "model:\n"
        "  args:\n"
        "    noise_setting: !!python/object/apply:collections.OrderedDict\n"
        "      - - [alpha, 1]\n"
        "        - [beta, true]\n"
        "    numpy_noise: !!python/object/apply:numpy.core.multiarray._reconstruct\n"
        "      - ignored\n"
        "    fusion_backbone:\n"
        "      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )

    binding = _validate(fixture)

    assert dict(binding.training_parameters)["base_stage_widths"] == (3, 5, 7)


@pytest.mark.parametrize(
    "payload",
    (
        "model:\n  args:\n    fusion_backbone: !!python/object/apply:unsafe\n"
        "      - [num_filters, [3, 5, 7]]\n",
        "defaults: &filters [3, 5, 7]\nmodel:\n  args:\n"
        "    fusion_backbone:\n      num_filters: *filters\n",
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n"
        "model:\n  args: {}\n",
        "model:\n  args:\n"
        "    noise: !!python/object/apply:collections.OrderedDict\n"
        "      - - [alpha, 1]\n        - [alpha, 2]\n"
        "    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        "model:\n  args:\n"
        "    noise: !!python/object/apply:collections.OrderedDict\n"
        "      - [alpha, 1]\n"
        "    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
    ),
)
def test_external_binding_rejects_unsafe_or_ambiguous_pyramid_nodes(
    tmp_path: Path, payload: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture["config"].write_text(payload, encoding="utf-8")

    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


@pytest.mark.parametrize("value", ((3, 5, 7), [True, 5, 7], [3, 5]))
def test_external_binding_rejects_noncanonical_prune_base_stage_widths(
    tmp_path: Path, value: object
) -> None:
    fixture = _fixture(tmp_path)
    fixture["raw"]["training_parameters"]["base_stage_widths"] = value

    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


def test_external_binding_rejects_missing_prune_base_stage_widths(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["raw"]["training_parameters"].pop("base_stage_widths")

    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


def test_external_binding_rejects_prune_base_widths_not_matching_config(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["raw"]["training_parameters"]["base_stage_widths"] = [3, 5, 9]

    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


def test_external_binding_rejects_malformed_pyramid_config(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["config"].write_text("model: [\n", encoding="utf-8")

    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


def test_prune_base_widths_are_distinct_from_candidate_widths(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate_stage_widths = (11, 13, 17)

    binding = _validate(fixture)

    assert dict(binding.training_parameters)["base_stage_widths"] == (3, 5, 7)
    assert dict(binding.training_parameters)["base_stage_widths"] != candidate_stage_widths


@pytest.mark.parametrize("key", ("target_epoch", "groups", "width_per_group"))
def test_external_binding_rejects_missing_structural_training_parameter(
    tmp_path: Path, key: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture["raw"]["training_parameters"].pop(key)

    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("groups", True),
        ("groups", 0),
        ("width_per_group", False),
        ("width_per_group", 0),
        ("target_epoch", True),
        ("target_epoch", 0),
    ),
)
def test_external_binding_rejects_invalid_structural_pruning_parameters(
    tmp_path: Path, key: str, value: object
) -> None:
    fixture = _fixture(tmp_path)
    fixture["raw"]["training_parameters"].update(groups=3, width_per_group=5)
    fixture["raw"]["training_parameters"][key] = value

    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown",
        "missing",
        "required",
        "source",
        "null_path",
        "unknown_parameter",
        "missing_parameter",
        "bool_epochs",
        "zero_epochs",
        "negative_seed",
        "nan_rate",
        "zero_rate",
        "bool_batch",
        "blank_string",
        "padded_string",
        "uppercase_digest",
        "short_digest",
        "bad_digest",
    ),
)
def test_external_binding_rejects_noncanonical_schema_or_values(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    raw = fixture["raw"]
    if mutation == "unknown":
        raw["unknown"] = PRIVATE_TOKEN
    elif mutation == "missing":
        raw.pop("dataset_root")
    elif mutation == "required":
        raw["training_required"] = False
    elif mutation == "source":
        raw["training_source_kind"] = "reuse"
    elif mutation == "null_path":
        raw["dataset_root"] = None
    elif mutation == "unknown_parameter":
        raw["training_parameters"]["unknown"] = "x"
    elif mutation == "missing_parameter":
        raw["training_parameters"].pop("epochs")
    elif mutation == "bool_epochs":
        raw["training_parameters"]["epochs"] = True
    elif mutation == "zero_epochs":
        raw["training_parameters"]["epochs"] = 0
    elif mutation == "negative_seed":
        raw["training_parameters"]["seed"] = -1
    elif mutation == "nan_rate":
        raw["training_parameters"]["learning_rate"] = float("nan")
    elif mutation == "zero_rate":
        raw["training_parameters"]["learning_rate"] = 0
    elif mutation == "bool_batch":
        raw["training_parameters"]["batch_size"] = False
    elif mutation == "blank_string":
        raw["training_parameters"]["optimizer"] = ""
    elif mutation == "padded_string":
        raw["training_parameters"]["optimizer"] = " adamw "
    elif mutation == "uppercase_digest":
        raw["base_checkpoint_sha256"] = "A" * 64
    elif mutation == "short_digest":
        raw["pyramid_config_sha256"] = "a" * 63
    else:
        raw["base_checkpoint_sha256"] = "0" * 64

    with pytest.raises(P6ExternalTrainingBindingError) as captured:
        _validate(fixture)
    assert captured.value.category == "history_execution_invalid"
    assert PRIVATE_TOKEN not in str(captured.value)


@pytest.mark.parametrize(
    "mutation",
    (
        "relative",
        "dot",
        "missing",
        "dataset_file",
        "checkpoint_directory",
        "leaf_symlink",
        "parent_symlink",
        "code_overlap",
        "output_overlap",
        "reserved_overlap",
        "file_alias",
    ),
)
def test_external_binding_rejects_unsafe_paths_and_aliases(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path)
    raw = fixture["raw"]
    if mutation == "relative":
        raw["dataset_root"] = "relative"
    elif mutation == "dot":
        raw["base_checkpoint_path"] = (
            f"{fixture['checkpoint'].parent}/./{fixture['checkpoint'].name}"
        )
    elif mutation == "missing":
        raw["pyramid_config_path"] = str(tmp_path / "missing")
    elif mutation == "dataset_file":
        raw["dataset_root"] = str(fixture["checkpoint"])
    elif mutation == "checkpoint_directory":
        raw["base_checkpoint_path"] = str(fixture["dataset"])
    elif mutation == "leaf_symlink":
        link = tmp_path / "dataset-link"
        link.symlink_to(fixture["dataset"], target_is_directory=True)
        raw["dataset_root"] = str(link)
    elif mutation == "parent_symlink":
        link = tmp_path / "stable-link"
        link.symlink_to(fixture["checkpoint"].parent, target_is_directory=True)
        raw["base_checkpoint_path"] = str(link / fixture["checkpoint"].name)
    elif mutation == "code_overlap":
        raw["dataset_root"] = str(fixture["code"])
    elif mutation == "output_overlap":
        raw["base_checkpoint_path"] = str(fixture["output"] / "file")
    elif mutation == "reserved_overlap":
        raw["base_checkpoint_path"] = str(fixture["reserved"][0])
    else:
        alias = fixture["output"] / "result.json"
        os.link(fixture["checkpoint"], alias)
        raw["base_checkpoint_path"] = str(alias)

    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


def test_external_binding_requires_ignored_assets_inside_git_worktree(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    repo = tmp_path / "operator-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("assets/\n", encoding="utf-8")
    assets = repo / "assets"
    assets.mkdir()
    dataset = assets / "dataset"
    dataset.mkdir()
    checkpoint = assets / "base.ckpt"
    checkpoint.write_bytes(b"x")
    config = assets / "config.py"
    config.write_text(
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )
    fixture["raw"].update(
        dataset_root=str(dataset),
        base_checkpoint_path=str(checkpoint),
        pyramid_config_path=str(config),
    )
    assert _validate(fixture).dataset_root == dataset
    (repo / ".gitignore").write_text("", encoding="utf-8")
    with pytest.raises(P6ExternalTrainingBindingError):
        _validate(fixture)


def test_loader_example_and_contract_helpers_are_private_safe(tmp_path: Path) -> None:
    example = load_external_training_binding(
        REPOSITORY_ROOT / "configs/execution/p6_external_training_binding.example.yaml"
    )
    assert example["dataset_root"] is None
    assert all(value is None for value in example["training_parameters"].values())
    with pytest.raises(P6ExternalTrainingBindingError):
        validate_external_training_binding(
            example,
            code_toolchain_root=tmp_path,
            local_output_root=tmp_path / "out",
            reserved_paths=(),
        )
    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    fixture = _fixture(valid_root)
    binding = _validate(fixture)
    contract = {"flat": "preserved", "dataset_root": "old", "training_parameters": {"old": "x"}}
    bound = bind_external_training_contract(contract, binding)
    assert contract != bound and "external_training_binding" in bound
    assert "dataset_root" not in bound and "training_parameters" not in bound
    extracted = external_training_binding_from_contract(bound)
    extracted["dataset_root"] = "changed"
    assert external_training_binding_from_contract(bound)["dataset_root"] != "changed"
    assert public_safe_external_training_projection(extracted) == {
        "schema_version": "p6_external_training_binding_v1",
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
    }


@pytest.mark.parametrize(
    "payload",
    (
        "a: 1\na: 2\n",
        "[not-a-mapping]",
        "bad: [",
    ),
)
def test_loader_rejects_invalid_or_duplicate_yaml_without_leaking_path(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / PRIVATE_TOKEN
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(P6ExternalTrainingBindingError) as captured:
        load_external_training_binding(path)
    assert PRIVATE_TOKEN not in str(captured.value)
