"""Private output helpers for the P6 full-chain bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from framework.stage6.coptv2x_h800_search_v2 import (
    P6CoptV2XContractError,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_history_binding_v1 import (
    P6HistoryBindingError,
    validate_history_execution_binding,
)


ErrorFactory = Callable[[str, str], Exception]


def validate_rendered_pair(
    binding: Mapping[str, Any],
    config: Mapping[str, Any],
    local_output_root: Path,
    *,
    public_contract_path: Path,
    error_factory: ErrorFactory,
) -> None:
    try:
        validate_history_execution_binding(binding)
        contract = load_public_contract(public_contract_path)
        descriptor, raw_path = tempfile.mkstemp(
            dir=local_output_root,
            prefix=".p6-full-chain-local.",
            suffix=".json",
        )
        path = Path(raw_path)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(config, handle, ensure_ascii=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            load_local_config(path, contract)
        finally:
            path.unlink(missing_ok=True)
    except P6HistoryBindingError as error:
        raise error_factory(
            "execution_interface_unavailable", "rendered binding is invalid"
        ) from error
    except (OSError, P6CoptV2XContractError, TypeError, ValueError) as error:
        raise error_factory(
            "local_config_invalid", "rendered local config is invalid"
        ) from error


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_output_root(raw_root: Path, error_factory: ErrorFactory) -> Path:
    try:
        root = raw_root.resolve(strict=True)
    except OSError as error:
        raise error_factory(
            "unsafe_destination", "private output root is unavailable"
        ) from error
    if not root.is_dir():
        raise error_factory("unsafe_destination", "private output root is invalid")
    return root


def _resolve_output_path(
    raw_path: Path, root: Path, error_factory: ErrorFactory
) -> Path:
    if raw_path.is_symlink():
        raise error_factory("unsafe_destination", "private output path is unsafe")
    try:
        parent = raw_path.parent.resolve(strict=True)
        destination = (parent / raw_path.name).resolve(strict=False)
    except OSError as error:
        raise error_factory(
            "unsafe_destination", "private output path is unavailable"
        ) from error
    if not _is_relative_to(destination, root) or destination == root:
        raise error_factory("unsafe_destination", "private output escapes its root")
    return destination


def resolve_private_outputs(
    raw_root: Path,
    raw_binding: Path,
    raw_config: Path,
    *,
    error_factory: ErrorFactory,
) -> tuple[Path, Path, Path]:
    paths = (raw_root, raw_binding, raw_config)
    if (
        not all(isinstance(path, Path) and path.is_absolute() for path in paths)
        or raw_root.is_symlink()
    ):
        raise error_factory(
            "unsafe_destination", "private output paths are invalid"
        )
    root = _resolve_output_root(raw_root, error_factory)
    binding = _resolve_output_path(raw_binding, root, error_factory)
    config = _resolve_output_path(raw_config, root, error_factory)
    if binding == config:
        raise error_factory(
            "unsafe_destination", "private output paths must differ"
        )
    if root / "stage1_partition_manifest.json" in (binding, config):
        raise error_factory("unsafe_destination", "private output path is reserved")
    return root, binding, config
