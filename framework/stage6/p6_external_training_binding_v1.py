"""Validated, private binding for the external P6 training inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Literal, TypeAlias

import yaml

from framework.stage6.p6_history_training_contract_v1 import REQUIRED_TRAINING_PARAMETER_KEYS


ExternalTrainingParameter: TypeAlias = str | int | float
_SCHEMA_VERSION = "p6_external_training_binding_v1"
_SOURCE_KIND = "selected_candidate_finetune"
_PATH_KEYS = ("dataset_root", "base_checkpoint_path", "pyramid_config_path")
_DIGEST_KEYS = ("base_checkpoint_sha256", "pyramid_config_sha256")
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "training_required",
        "training_source_kind",
        *_PATH_KEYS,
        *_DIGEST_KEYS,
        "training_parameters",
    }
)
_FLAT_KEYS = frozenset(
    {"training_required", "training_source_kind", *_PATH_KEYS, *_DIGEST_KEYS, "training_parameters"}
)
_MAX_BYTES = 1024 * 1024


@dataclass(frozen=True)
class P6ExternalTrainingBinding:
    schema_version: Literal["p6_external_training_binding_v1"]
    training_required: Literal[True]
    training_source_kind: Literal["selected_candidate_finetune"]
    dataset_root: Path
    base_checkpoint_path: Path
    base_checkpoint_sha256: str
    pyramid_config_path: Path
    pyramid_config_sha256: str
    training_parameters: tuple[tuple[str, ExternalTrainingParameter], ...]


class P6ExternalTrainingBindingError(ValueError):
    """Stable, redacted external-training binding error."""

    category: Literal["history_execution_invalid"] = "history_execution_invalid"

    def __init__(self) -> None:
        super().__init__(self.category)


def load_external_training_binding(path: Path) -> dict[str, Any]:
    """Load a bounded, private YAML mapping without exposing its location."""
    try:
        if not isinstance(path, Path) or not path.is_absolute() or not _is_lexical_path(str(path)):
            _invalid()
        _reject_symlink_components(path)
        info = path.stat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > _MAX_BYTES
            or not os.access(path, os.R_OK)
        ):
            _invalid()
        _require_ignored_if_in_git(path, allow_public_example=True)
        payload = path.read_bytes()
        if len(payload) > _MAX_BYTES:
            _invalid()
        loaded = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(loaded, Mapping):
            _invalid()
        return copy.deepcopy(dict(loaded))
    except P6ExternalTrainingBindingError:
        raise
    except (OSError, UnicodeDecodeError, yaml.YAMLError, TypeError, ValueError):
        _invalid()


def validate_external_training_binding(
    raw: Mapping[str, Any],
    *,
    code_toolchain_root: Path,
    local_output_root: Path,
    reserved_paths: Sequence[Path],
) -> P6ExternalTrainingBinding:
    """Validate external immutable inputs and return only canonical detached state."""
    try:
        if not isinstance(raw, Mapping) or set(raw) != _REQUIRED_KEYS:
            _invalid()
        if (
            type(raw.get("schema_version")) is not str
            or raw.get("schema_version") != _SCHEMA_VERSION
            or raw.get("training_required") is not True
            or type(raw.get("training_source_kind")) is not str
            or raw.get("training_source_kind") != _SOURCE_KIND
        ):
            _invalid()
        code_root = _canonical_boundary_root(code_toolchain_root)
        output_root = _canonical_boundary_root(local_output_root)
        dataset = _external_path(raw["dataset_root"], directory=True)
        checkpoint = _external_path(raw["base_checkpoint_path"], directory=False)
        config = _external_path(raw["pyramid_config_path"], directory=False)
        parameters = _parameters(raw["training_parameters"])
        checkpoint_digest = _digest(checkpoint, raw["base_checkpoint_sha256"])
        config_digest = _digest(config, raw["pyramid_config_sha256"])
        reserved = tuple(_canonical_planned_path(path) for path in reserved_paths)
        external = (dataset, checkpoint, config)
        boundaries = (code_root, output_root, *reserved)
        if checkpoint == config or any(
            _overlaps(left, right) for left in external for right in boundaries
        ):
            _invalid()
        for file_path in (checkpoint, config):
            for destination in boundaries:
                if (
                    destination.exists()
                    and destination.is_file()
                    and _same_inode(file_path, destination)
                ):
                    _invalid()
        return P6ExternalTrainingBinding(
            _SCHEMA_VERSION,
            True,
            _SOURCE_KIND,
            dataset,
            checkpoint,
            checkpoint_digest,
            config,
            config_digest,
            parameters,
        )
    except P6ExternalTrainingBindingError:
        raise
    except (OSError, TypeError, ValueError, KeyError):
        _invalid()


def external_training_binding_to_mapping(binding: P6ExternalTrainingBinding) -> dict[str, Any]:
    """Return a detached canonical mapping suitable for private request hashing."""
    if not isinstance(binding, P6ExternalTrainingBinding):
        _invalid()
    return {
        "schema_version": binding.schema_version,
        "training_required": binding.training_required,
        "training_source_kind": binding.training_source_kind,
        "dataset_root": str(binding.dataset_root),
        "base_checkpoint_path": str(binding.base_checkpoint_path),
        "base_checkpoint_sha256": binding.base_checkpoint_sha256,
        "pyramid_config_path": str(binding.pyramid_config_path),
        "pyramid_config_sha256": binding.pyramid_config_sha256,
        "training_parameters": dict(binding.training_parameters),
    }


def bind_external_training_contract(
    contract: Mapping[str, Any], binding: P6ExternalTrainingBinding
) -> dict[str, Any]:
    """Replace legacy flat private fields with one complete detached binding."""
    if not isinstance(contract, Mapping):
        _invalid()
    result = copy.deepcopy(dict(contract))
    for key in _FLAT_KEYS:
        result.pop(key, None)
    result["external_training_binding"] = external_training_binding_to_mapping(binding)
    return result


def external_training_binding_from_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the sole nested training binding without aliasing caller state."""
    if not isinstance(contract, Mapping) or set(contract).intersection(_FLAT_KEYS):
        _invalid()
    binding = contract.get("external_training_binding")
    if not isinstance(binding, Mapping) or set(binding) != _REQUIRED_KEYS:
        _invalid()
    return copy.deepcopy(dict(binding))


def public_safe_external_training_projection(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return the three non-sensitive training labels only."""
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema_version") != _SCHEMA_VERSION
        or binding.get("training_required") is not True
        or binding.get("training_source_kind") != _SOURCE_KIND
    ):
        _invalid()
    return {
        "schema_version": _SCHEMA_VERSION,
        "training_required": True,
        "training_source_kind": _SOURCE_KIND,
    }


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.YAMLError("duplicate key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _external_path(raw: object, *, directory: bool) -> Path:
    if not isinstance(raw, str) or not _is_lexical_path(raw):
        _invalid()
    path = Path(raw)
    _reject_symlink_components(path)
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError:
        _invalid()
    if (directory and not stat.S_ISDIR(mode)) or (not directory and not stat.S_ISREG(mode)):
        _invalid()
    if not os.access(resolved, os.R_OK | (os.X_OK if directory else 0)):
        _invalid()
    _require_ignored_if_in_git(resolved, allow_public_example=False)
    return resolved


def _parameters(raw: object) -> tuple[tuple[str, ExternalTrainingParameter], ...]:
    if not isinstance(raw, Mapping) or set(raw) != set(REQUIRED_TRAINING_PARAMETER_KEYS):
        _invalid()
    five_strings = (
        "training_mode",
        "optimizer",
        "dataset_split",
        "checkpoint_selection",
        "freeze_policy",
    )
    if (
        not all(_trimmed_string(raw[key]) for key in five_strings)
        or not _positive_int(raw["epochs"])
        or not _nonnegative_int(raw["seed"])
        or not _positive_number(raw["learning_rate"])
        or not _positive_int(raw["batch_size"])
    ):
        _invalid()
    return tuple((key, raw[key]) for key in REQUIRED_TRAINING_PARAMETER_KEYS)


def _digest(path: Path, supplied: object) -> str:
    if supplied is not None and (
        not isinstance(supplied, str)
        or len(supplied) != 64
        or any(character not in "0123456789abcdef" for character in supplied)
    ):
        _invalid()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_MAX_BYTES), b""):
                digest.update(chunk)
    except OSError:
        _invalid()
    actual = digest.hexdigest()
    if supplied is not None and supplied != actual:
        _invalid()
    return actual


def _canonical_boundary_root(path: object) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or not _is_lexical_path(str(path)):
        _invalid()
    resolved = _canonical_planned_path(path)
    if path.exists() and not resolved.is_dir():
        _invalid()
    return resolved


def _canonical_planned_path(path: Path) -> Path:
    _reject_symlink_components(path, allow_missing=True)
    ancestor = path
    suffix: list[str] = []
    while not ancestor.exists():
        suffix.append(ancestor.name)
        if ancestor.parent == ancestor:
            _invalid()
        ancestor = ancestor.parent
    if suffix and not ancestor.is_dir():
        _invalid()
    try:
        resolved = ancestor.resolve(strict=True)
    except OSError:
        _invalid()
    return resolved.joinpath(*reversed(suffix))


def _reject_symlink_components(path: Path, *, allow_missing: bool = False) -> None:
    if not path.is_absolute():
        _invalid()
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                _invalid()
        except FileNotFoundError:
            if allow_missing:
                return
            _invalid()
        except OSError:
            _invalid()


def _require_ignored_if_in_git(path: Path, *, allow_public_example: bool) -> None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(path.parent), "rev-parse", "--show-toplevel"),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        _invalid()
    if completed.returncode != 0:
        return
    root = Path(completed.stdout.strip())
    if allow_public_example and path == _public_example_path():
        return
    try:
        relative = path.relative_to(root)
    except ValueError:
        _invalid()
    checked = subprocess.run(
        ("git", "-C", str(root), "check-ignore", "-q", "--", str(relative)),
        capture_output=True,
        check=False,
        timeout=5,
    )
    if checked.returncode != 0:
        _invalid()


def _public_example_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs/execution/p6_external_training_binding.example.yaml"
    )


def _is_lexical_path(raw: str) -> bool:
    path = Path(raw)
    return (
        bool(raw)
        and not any(character in raw for character in ("\x00", "\r", "\n"))
        and path.anchor == "/"
        and path.is_absolute()
        and str(path) == raw
        and all(part not in {"", ".", ".."} for part in raw.split("/")[1:])
    )


def _overlaps(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        try:
            right.relative_to(left)
            return True
        except ValueError:
            return False


def _same_inode(left: Path, right: Path) -> bool:
    return os.path.samestat(left.stat(), right.stat())


def _trimmed_string(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _invalid() -> None:
    raise P6ExternalTrainingBindingError()
