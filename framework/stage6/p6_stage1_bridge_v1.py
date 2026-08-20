"""Bridge the real Stage1 scanner into a private P6 JSON manifest."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml


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
        _validate_h800_hardware_yaml(hardware_path)
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
        with _supplied_stage1_imports(stage1_root):
            adapters = importlib.import_module("framework.stage1.adapters")
            graph_scan = importlib.import_module("framework.stage1.graph_scan")
            hardware_scan = importlib.import_module("framework.stage1.hardware_scan")
            hardware = hardware_scan.HwCapability.from_yaml(hardware_path)
            return graph_scan.scan(adapters.get_adapter(model_name), hardware, device=device)
    except P6Stage1BridgeError:
        raise
    except Exception as error:  # noqa: BLE001 - CLI must expose only stable category.
        raise P6Stage1BridgeError() from error


@contextmanager
def _supplied_stage1_imports(stage1_root: Path) -> Any:
    framework_dir = stage1_root / "framework"
    stage1_dir = framework_dir / "stage1"
    for module_name in ("__init__", "adapters", "graph_scan", "hardware_scan"):
        _validate_input_file(stage1_dir / f"{module_name}.py")

    framework_package = importlib.import_module("framework")
    original_path = list(getattr(framework_package, "__path__", []))
    saved_stage1_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "framework.stage1" or name.startswith("framework.stage1.")
    }
    for name in tuple(saved_stage1_modules):
        sys.modules.pop(name, None)
    framework_package.__path__ = [str(framework_dir), *original_path]
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "framework.stage1" or name.startswith("framework.stage1."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_stage1_modules)
        framework_package.__path__ = original_path
        importlib.invalidate_caches()


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
        _is_h800_capability(manifest.get("hw_capability"))
        and _valid_search_groups(manifest.get("view_b1_search_groups"))
        and _valid_quant_units(manifest.get("view_b2_quant_units"))
        and _valid_routing_segments(manifest.get("view_d_routing_segments"))
    )


def _is_h800_capability(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return _is_h800_name(value.get("name"))


def _valid_search_groups(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_valid_search_group(item) for item in value)
    )


def _valid_search_group(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        _non_empty_string(value.get("search_group_id"))
        and _non_empty_string(value.get("bucket"))
        and _positive_int_list(value.get("widths"))
        and _positive_int(value.get("round_to"))
        and _positive_int(value.get("int8_buildable_align"))
        and _rate(value.get("max_rate"))
        and isinstance(value.get("grouped_conv"), bool)
        and _string_list(value.get("criterion_pool"))
        and _string_list(value.get("member_b1_groups"))
    )


def _valid_quant_units(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_valid_quant_unit(item) for item in value)
    )


def _valid_quant_unit(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    legal_bits = value.get("legal_bits")
    return (
        _non_empty_string(value.get("unit"))
        and isinstance(value.get("quantizable"), bool)
        and _string_list(legal_bits)
        and {"FP16", "INT8"}.issubset(set(legal_bits))
        and _string_list(value.get("member_groups"))
    )


def _valid_routing_segments(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    segments = value.get("segments")
    return (
        isinstance(segments, list)
        and len(segments) > 0
        and all(
            isinstance(segment, Mapping)
            and _non_empty_string(segment.get("device"))
            and _positive_int(segment.get("n_nodes"))
            for segment in segments
        )
    )


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_non_empty_string(item) for item in value)
    )


def _positive_int_list(value: object) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(_positive_int(item) for item in value)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _rate(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) < 1.0
    )


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


def _validate_h800_hardware_yaml(path: Path) -> None:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise P6Stage1BridgeError() from error
    if not isinstance(raw, Mapping):
        raise P6Stage1BridgeError()
    basic = raw.get("basic")
    name = basic.get("name") if isinstance(basic, Mapping) else raw.get("name")
    if not _is_h800_name(name):
        raise P6Stage1BridgeError()


def _is_h800_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = "_".join(value.casefold().strip().split())
    if normalized.startswith("nvidia_"):
        normalized = normalized.removeprefix("nvidia_")
    return normalized == "h800" or normalized.startswith("h800_")


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
    temporary_path: Path | None = None
    try:
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
        temporary_path.replace(output_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "P6Stage1BridgeError",
    "build_p6_stage1_partition_manifest",
    "run_real_stage1_scan",
]
