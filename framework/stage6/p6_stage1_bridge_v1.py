"""Bridge the real Stage1 scanner into a private P6 JSON manifest."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any


MODEL_NAME = "pyramid_lidar"
SCHEMA = "stage1_partition_manifest_v1"
STAGE = "stage1_partition"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class P6Stage1BridgeError(ValueError):
    """Stable categorized failure for P6 Stage1 bridge operations."""

    def __init__(self, category: str = "stage1_scan_invalid") -> None:
        self.category = category
        super().__init__(category)


def build_p6_stage1_partition_manifest(
    output_path: Path,
    hardware_path: Path,
    device: str,
    environment: Mapping[str, str],
    scanner: Callable[[str, Path, str, Mapping[str, str]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run Stage1 scanning, validate the P6 manifest contract, and write JSON."""
    try:
        _validate_output_path(output_path)
        _validate_input_file(hardware_path)
        manifest = scanner(
            MODEL_NAME,
            hardware_path,
            device,
            copy.deepcopy(dict(environment)),
        )
        if not _is_p6_stage1_manifest(manifest):
            raise P6Stage1BridgeError()
        owned_manifest = copy.deepcopy(dict(manifest))
        _atomic_write_json(output_path, owned_manifest)
    except P6Stage1BridgeError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise P6Stage1BridgeError() from error
    return copy.deepcopy(owned_manifest)


def run_real_stage1_scan(
    model_name: str,
    hardware_path: Path,
    device: str,
    environment: Mapping[str, str],
) -> Mapping[str, Any]:
    """Adapter that invokes the repository Stage1 graph scanner."""
    stage1_root = _validate_directory_env(environment, "STAGE1_REPO_ROOT")
    heal_root = _validate_directory_env(environment, "HEAL_ROOT")
    heal_checkpoint_root = _validate_directory_env(environment, "HEAL_CKPT_ROOT")
    os.environ.update(
        {
            "STAGE1_REPO_ROOT": str(stage1_root),
            "HEAL_ROOT": str(heal_root),
            "HEAL_CKPT_ROOT": str(heal_checkpoint_root),
        }
    )
    if str(stage1_root) not in sys.path:
        sys.path.insert(0, str(stage1_root))

    try:
        adapters = _load_stage1_module(stage1_root, "adapters")
        graph_scan = _load_stage1_module(stage1_root, "graph_scan")
        from framework.stage1.hardware_scan import HwCapability

        hardware = HwCapability.from_yaml(hardware_path)
        return graph_scan.scan(adapters.get_adapter(model_name), hardware, device=device)
    except P6Stage1BridgeError:
        raise
    except Exception as error:  # noqa: BLE001 - CLI must expose only stable category.
        raise P6Stage1BridgeError() from error


def _load_stage1_module(stage1_root: Path, module_name: str) -> ModuleType:
    module_path = stage1_root / "framework" / "stage1" / f"{module_name}.py"
    _validate_input_file(module_path)
    spec = importlib.util.spec_from_file_location(f"_p6_stage1_bridge_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise P6Stage1BridgeError()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_p6_stage1_manifest(manifest: object) -> bool:
    if not isinstance(manifest, Mapping):
        return False
    required = {
        "schema",
        "stage",
        "model",
        "scan_status",
        "hw_capability",
        "view_b1_search_groups",
        "view_b2_quant_units",
        "view_d_routing_segments",
    }
    if not required.issubset(manifest):
        return False
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("stage") != STAGE
        or manifest.get("model") != MODEL_NAME
        or manifest.get("scan_status") != "ok"
    ):
        return False
    return (
        isinstance(manifest.get("hw_capability"), Mapping)
        and _non_empty_list(manifest.get("view_b1_search_groups"))
        and _non_empty_list(manifest.get("view_b2_quant_units"))
        and isinstance(manifest.get("view_d_routing_segments"), Mapping)
    )


def _non_empty_list(value: object) -> bool:
    return isinstance(value, list) and len(value) > 0


def _validate_directory_env(environment: Mapping[str, str], name: str) -> Path:
    value = environment.get(name)
    if not isinstance(value, str):
        raise P6Stage1BridgeError()
    path = Path(value)
    if not path.is_absolute() or _contains_symlink_component(path):
        raise P6Stage1BridgeError()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6Stage1BridgeError() from error
    if not resolved.is_dir():
        raise P6Stage1BridgeError()
    return resolved


def _validate_input_file(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or _contains_symlink_component(path):
        raise P6Stage1BridgeError()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6Stage1BridgeError() from error
    if not resolved.is_file():
        raise P6Stage1BridgeError()
    return resolved


def _validate_output_path(path: Path) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or _contains_symlink_component(path)
        or not path.parent.is_dir()
        or path.is_dir()
    ):
        raise P6Stage1BridgeError()
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise P6Stage1BridgeError() from error
    if not parent.is_dir() or path.exists() and path.is_symlink():
        raise P6Stage1BridgeError()


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink() for component in (path, *path.parents)
    )


def _atomic_write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise P6Stage1BridgeError()
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(encoded)
        handle.write("\n")
    try:
        temporary_path.replace(output_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "P6Stage1BridgeError",
    "build_p6_stage1_partition_manifest",
    "run_real_stage1_scan",
]
