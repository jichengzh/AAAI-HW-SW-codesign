"""Private static training-contract validation for recipe-v2 Pyramid sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
from pathlib import Path
from typing import Any

from framework.stage6.p6_history_recipe_profiles_v1 import (
    RECIPE_V2,
    SHARED_SOURCE_PATH_KEYS,
)


REQUIRED_TRAINING_CONTRACT_KEYS: tuple[str, ...] = (
    "training_required",
    "training_source_kind",
    "base_checkpoint_path",
    "dataset_root",
    "pyramid_config_path",
    "training_parameters",
    "stage_widths",
)
REQUIRED_TRAINING_PARAMETER_KEYS: tuple[str, ...] = (
    "training_mode",
    "epochs",
    "seed",
    "optimizer",
    "learning_rate",
    "batch_size",
    "dataset_split",
    "checkpoint_selection",
    "freeze_policy",
)
STATIC_TRAINING_INPUT_PATH_KEYS: tuple[str, ...] = (
    "base_checkpoint_path",
    "dataset_root",
    "pyramid_config_path",
)
ALLOWED_TRAINING_SOURCE_KINDS = frozenset({"selected_candidate_finetune"})
_PUBLIC_SAFE_CONTRACT_KEYS = (
    "schema_version",
    "group_id",
    "model",
    "width",
    "artifact_id",
    "source_status",
    "source_evidence_sha256",
)


class P6HistoryTrainingContractError(ValueError):
    """Stable failure category for private training-contract validation."""

    def __init__(self, detail: str) -> None:
        self.category = "history_execution_invalid"
        super().__init__(f"{self.category}: {detail}")


def validate_recipe_v2_training_template(
    template: Mapping[str, Any], *, private_root: Path
) -> dict[str, Any]:
    """Return a detached recipe-v2 template after validating static inputs."""
    contract = _detached_mapping(template)
    _require_recipe_v2(contract)
    _require_training_required_true(contract)
    _require_training_source_kind(contract)
    _require_training_parameters(contract)
    _validate_static_training_paths(contract, private_root)
    return contract


def validate_recipe_v2_group_training_contract(
    contract: Mapping[str, Any],
    *,
    private_root: Path,
    local_output_root: Path,
    group_id: str,
) -> dict[str, Any]:
    """Return a detached materialized contract after validating private paths."""
    canonical = _detached_mapping(contract)
    _require_training_required_true(canonical)
    _require_training_source_kind(canonical)
    _require_training_parameters(canonical)
    _validate_static_training_paths(canonical, private_root)
    _validate_group_identity(canonical, group_id)
    _validate_stage_widths(canonical)
    _validate_shared_source_paths(canonical, local_output_root)
    return canonical


def validate_projected_training_contract(
    contract: Mapping[str, Any],
    *,
    group_id: str,
) -> dict[str, Any]:
    """Return a detached projected contract after lexical-only validation."""
    canonical = _detached_mapping(contract)
    _require_training_required_true(canonical)
    _require_training_source_kind(canonical)
    _require_training_parameters(canonical)
    _validate_group_identity(canonical, group_id)
    _validate_stage_widths(canonical)
    _require_lexical_absolute_paths(canonical, STATIC_TRAINING_INPUT_PATH_KEYS)
    _require_lexical_absolute_paths(
        canonical,
        ("training_done_marker", "source_done_marker"),
    )
    return canonical


def public_safe_contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Project only fixed public contract labels; never copy private training data."""
    canonical = _detached_mapping(contract)
    return {
        key: copy.deepcopy(canonical[key])
        for key in _PUBLIC_SAFE_CONTRACT_KEYS
        if key in canonical
    }


def _detached_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        _invalid("training contract is invalid")
    try:
        return copy.deepcopy(dict(raw))
    except (TypeError, ValueError) as error:
        raise P6HistoryTrainingContractError("training contract is invalid") from error


def _require_recipe_v2(contract: Mapping[str, Any]) -> None:
    recipe = contract.get("dynamic_materialization_recipe")
    if not isinstance(recipe, Mapping) or recipe.get("schema_version") != RECIPE_V2:
        _invalid("training recipe is invalid")


def _require_training_required_true(contract: Mapping[str, Any]) -> None:
    if contract.get("training_required") is not True:
        _invalid("training requirement is invalid")


def _require_training_source_kind(contract: Mapping[str, Any]) -> None:
    value = contract.get("training_source_kind")
    if not isinstance(value, str) or value not in ALLOWED_TRAINING_SOURCE_KINDS:
        _invalid("training source kind is invalid")


def _require_training_parameters(contract: Mapping[str, Any]) -> None:
    parameters = contract.get("training_parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != set(
        REQUIRED_TRAINING_PARAMETER_KEYS
    ):
        _invalid("training parameters are invalid")
    if (
        not _nonempty_string(parameters["training_mode"])
        or not _positive_int(parameters["epochs"])
        or not _nonnegative_int(parameters["seed"])
        or not _nonempty_string(parameters["optimizer"])
        or not _positive_float(parameters["learning_rate"])
        or not _positive_int(parameters["batch_size"])
        or not _nonempty_string(parameters["dataset_split"])
        or not _nonempty_string(parameters["checkpoint_selection"])
        or not _nonempty_string(parameters["freeze_policy"])
    ):
        _invalid("training parameters are invalid")


def _validate_static_training_paths(contract: Mapping[str, Any], private_root: Path) -> None:
    root = _resolve_root(private_root, "private training root")
    for key in STATIC_TRAINING_INPUT_PATH_KEYS:
        expected_kind = "directory" if key == "dataset_root" else "file"
        _resolve_private_input(contract.get(key), root, expected_kind)


def _validate_group_identity(contract: Mapping[str, Any], group_id: str) -> None:
    if not isinstance(group_id, str) or not group_id or contract.get("group_id") != group_id:
        _invalid("training group identity is invalid")


def _validate_stage_widths(contract: Mapping[str, Any]) -> None:
    widths = contract.get("width")
    stage_widths = contract.get("stage_widths")
    expected_keys = ("stage1_width", "stage2_width", "stage3_width")
    if (
        not isinstance(widths, Sequence)
        or isinstance(widths, (str, bytes))
        or len(widths) != len(expected_keys)
        or any(not _positive_int(value) for value in widths)
        or not isinstance(stage_widths, Mapping)
        or set(stage_widths) != set(expected_keys)
        or any(stage_widths[key] != widths[index] for index, key in enumerate(expected_keys))
    ):
        _invalid("training stage widths are invalid")


def _validate_shared_source_paths(contract: Mapping[str, Any], local_output_root: Path) -> None:
    root = _resolve_root(local_output_root, "local output root")
    paths = contract.get("shared_source_paths")
    if not isinstance(paths, Mapping) or set(paths) != set(SHARED_SOURCE_PATH_KEYS):
        _invalid("shared training outputs are invalid")
    resolved_paths = [_resolve_local_output(paths[key], root) for key in SHARED_SOURCE_PATH_KEYS]
    if len(set(resolved_paths)) != len(resolved_paths):
        _invalid("shared training outputs are invalid")


def _require_lexical_absolute_paths(
    contract: Mapping[str, Any], keys: Sequence[str]
) -> None:
    for key in keys:
        raw_path = contract.get(key)
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or any(character in raw_path for character in ("\x00", "\r", "\n"))
        ):
            _invalid("training path is invalid")
        path = Path(raw_path)
        if not path.is_absolute() or ".." in path.parts:
            _invalid("training path is invalid")


def _resolve_root(raw_path: Path, label: str) -> Path:
    if not isinstance(raw_path, Path) or not raw_path.is_absolute() or raw_path.is_symlink():
        _invalid(f"{label} is invalid")
    try:
        resolved = raw_path.resolve(strict=True)
    except OSError as error:
        raise P6HistoryTrainingContractError(f"{label} is unavailable") from error
    if not resolved.is_dir():
        _invalid(f"{label} is invalid")
    return resolved


def _resolve_private_input(raw_path: object, root: Path, expected_kind: str) -> Path:
    path = _absolute_path_beneath_root(raw_path, root, strict=True)
    if (expected_kind == "file" and not path.is_file()) or (
        expected_kind == "directory" and not path.is_dir()
    ):
        _invalid("training input is invalid")
    return path


def _resolve_local_output(raw_path: object, root: Path) -> Path:
    return _absolute_path_beneath_root(raw_path, root, strict=False)


def _absolute_path_beneath_root(raw_path: object, root: Path, *, strict: bool) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        _invalid("training path is invalid")
    path = Path(raw_path)
    if not path.is_absolute() or not _is_relative_to(path, root):
        _invalid("training path is invalid")
    _reject_symlink_components(path, root)
    try:
        resolved = path.resolve(strict=strict)
    except OSError as error:
        raise P6HistoryTrainingContractError("training path is unavailable") from error
    if not _is_relative_to(resolved, root) or resolved == root:
        _invalid("training path is invalid")
    return resolved


def _reject_symlink_components(path: Path, root: Path) -> None:
    current = path
    while current != root:
        if current.is_symlink():
            _invalid("training path is invalid")
        parent = current.parent
        if parent == current:
            _invalid("training path is invalid")
        current = parent
    if root.is_symlink():
        _invalid("training path is invalid")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_float(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _invalid(detail: str) -> None:
    raise P6HistoryTrainingContractError(detail)
