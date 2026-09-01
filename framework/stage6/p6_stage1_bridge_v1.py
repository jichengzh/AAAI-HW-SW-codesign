"""Bridge the real Stage1 scanner into a private P6 JSON manifest."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import copy
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import tempfile
from typing import Any

import yaml

from framework.stage1.structural_axis_digest import scanner_structural_axes_digest
from framework.stage1_bridge import load_stage2_search_space
from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    default_hardware_execution_profile,
    load_hardware_execution_profile,
    validate_profile_backend,
)


MODEL_NAME = "pyramid_lidar"
SCHEMA = "stage1_partition_manifest_v1"
STAGE = "stage1_partition"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_STAGE1_ENV_KEYS = ("STAGE1_REPO_ROOT", "HEAL_ROOT", "HEAL_CKPT_ROOT")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
    scanner: Callable[
        [str, Path, str, Mapping[str, str], Path], Mapping[str, Any]
    ],
    *,
    profile: HardwareExecutionProfile | None = None,
    scenario_path: Path,
) -> dict[str, Any]:
    """Run Stage1 scanning, validate the P6 manifest contract, and write JSON."""
    try:
        _validate_output_path(output_path)
        hardware_path = _validate_input_file(hardware_path)
        profile = _selected_profile(profile, hardware_path)
        scenario_path = _validate_input_file(scenario_path)
        _validate_profile_hardware_yaml(hardware_path, profile)
        manifest = scanner(
            MODEL_NAME,
            hardware_path,
            device,
            copy.deepcopy(dict(environment)),
            scenario_path,
        )
        if not _is_p6_stage1_manifest(manifest, profile):
            raise P6Stage1BridgeError()
        owned_manifest = _profile_target_manifest(manifest, profile)
        _validate_strict_stage2_contract(owned_manifest, output_path.parent)
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
    scenario_path: Path,
) -> Mapping[str, Any]:
    """Adapter that invokes the repository Stage1 graph scanner."""
    stage1_root = _validate_directory_env(environment, "STAGE1_REPO_ROOT")
    heal_root = _validate_directory_env(environment, "HEAL_ROOT")
    heal_checkpoint_root = _validate_directory_env(environment, "HEAL_CKPT_ROOT")
    try:
        with _temporary_stage1_process_state(
            stage1_root, heal_root, heal_checkpoint_root
        ):
            with _supplied_stage1_imports(stage1_root):
                adapters = importlib.import_module("framework.stage1.adapters")
                graph_scan = importlib.import_module("framework.stage1.graph_scan")
                hardware_scan = importlib.import_module("framework.stage1.hardware_scan")
                formal_evidence = importlib.import_module(
                    "framework.stage1.formal_scan_evidence"
                )
                hardware = hardware_scan.HwCapability.from_yaml(hardware_path)
                adapter = adapters.get_adapter(model_name)
                scenario = formal_evidence.load_scan_scenario(
                    scenario_path,
                    trusted_root=adapter.formal_scenario_root,
                )
                formal_evidence.validate_scenario_hardware(scenario, hardware)
                return graph_scan.scan(
                    adapter,
                    hardware,
                    device=device,
                    scenario=scenario,
                    profile_latency_mode="off",
                )
    except P6Stage1BridgeError:
        raise
    except Exception as error:  # noqa: BLE001 - CLI must expose only stable category.
        raise P6Stage1BridgeError() from error


@contextmanager
def _temporary_stage1_process_state(
    stage1_root: Path,
    heal_root: Path,
    heal_checkpoint_root: Path,
) -> Any:
    """Restore only process state temporarily owned by this real scan."""
    values = {
        "STAGE1_REPO_ROOT": str(stage1_root),
        "HEAL_ROOT": str(heal_root),
        "HEAL_CKPT_ROOT": str(heal_checkpoint_root),
    }
    prior_values = {key: os.environ.get(key) for key in _STAGE1_ENV_KEYS}
    missing_keys = {key for key in _STAGE1_ENV_KEYS if key not in os.environ}
    inserted_path = None
    try:
        os.environ.update(values)
        if str(stage1_root) not in sys.path:
            inserted_path = str(stage1_root)
            sys.path.insert(0, inserted_path)
        yield
    finally:
        for key in _STAGE1_ENV_KEYS:
            if key in missing_keys:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior_values[key] or ""
        if inserted_path is not None:
            for index, value in enumerate(sys.path):
                if value is inserted_path:
                    sys.path.pop(index)
                    break


@contextmanager
def _supplied_stage1_imports(stage1_root: Path) -> Any:
    framework_dir = stage1_root / "framework"
    stage1_dir = framework_dir / "stage1"
    for module_name in (
        "__init__",
        "adapters",
        "formal_scan_evidence",
        "graph_scan",
        "hardware_scan",
    ):
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


def _is_p6_stage1_manifest(
    manifest: object, profile: HardwareExecutionProfile
) -> bool:
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
        "scanner_structural_axes",
        "scanner_structural_axes_digest",
        "formal_scan",
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
        _is_profile_capability(manifest.get("hw_capability"), profile)
        and _valid_search_groups(manifest.get("view_b1_search_groups"))
        and _valid_quant_units(manifest.get("view_b2_quant_units"))
        and _valid_routing_segments(manifest.get("view_d_routing_segments"))
        and _valid_formal_axes(manifest)
        and not _contains_absolute_manifest_path(manifest)
    )


def _contains_absolute_manifest_path(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            not isinstance(key, str)
            or _is_absolute_path(key)
            or _contains_absolute_manifest_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_manifest_path(item) for item in value)
    return isinstance(value, str) and _is_absolute_path(value)


def _is_absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _valid_formal_axes(manifest: Mapping[str, Any]) -> bool:
    if "structural_axes" in manifest or "structural_axis_inputs" in manifest:
        return False
    axes = manifest.get("scanner_structural_axes")
    digest = manifest.get("scanner_structural_axes_digest")
    formal_scan = manifest.get("formal_scan")
    if (
        not isinstance(axes, list)
        or not axes
        or not isinstance(digest, str)
        or not isinstance(formal_scan, Mapping)
        or formal_scan.get("status") != "derived"
    ):
        return False
    try:
        return digest == scanner_structural_axes_digest(axes)
    except (TypeError, ValueError):
        return False


def _validate_strict_stage2_contract(
    manifest: Mapping[str, Any], directory: Path
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".p6-stage1-strict.",
            suffix=".json",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
        load_stage2_search_space(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _is_profile_capability(
    value: object, profile: HardwareExecutionProfile
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if profile.profile_id == "h800":
        return _is_h800_name(value.get("name"))
    return _normalize_hardware_name(value.get("name")) in (
        profile.allowed_normalized_gpu_models
    )


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
    member_groups = value.get("member_groups")
    if not (
        _non_empty_string(value.get("unit"))
        and isinstance(value.get("quantizable"), bool)
        and _string_list(legal_bits)
        and isinstance(member_groups, list)
    ):
        return False
    if value["quantizable"]:
        return (
            len(legal_bits) == 2
            and set(legal_bits) == {"FP16", "INT8"}
            and _string_list(member_groups)
        )
    return legal_bits == ["FP16"] and member_groups == []


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


def _selected_profile(
    profile: HardwareExecutionProfile | None, hardware_path: Path
) -> HardwareExecutionProfile:
    if profile is None:
        rtx_profile = load_hardware_execution_profile("rtx4090")
        selected = (
            rtx_profile
            if hardware_path == _declared_hardware_path(rtx_profile)
            else default_hardware_execution_profile()
        )
    else:
        selected = profile
    validate_profile_backend(selected, "tvm_auto")
    return selected


def _declared_hardware_path(profile: HardwareExecutionProfile) -> Path:
    return _validate_input_file(
        _REPOSITORY_ROOT.joinpath(*profile.hardware_capability_path.parts)
    )


def _validate_profile_hardware_yaml(
    path: Path, profile: HardwareExecutionProfile
) -> None:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise P6Stage1BridgeError() from error
    if not isinstance(raw, Mapping):
        raise P6Stage1BridgeError()
    basic = raw.get("basic")
    name = basic.get("name") if isinstance(basic, Mapping) else raw.get("name")
    if profile.profile_id == "h800":
        if not _is_h800_name(name):
            raise P6Stage1BridgeError()
        return
    declared_path = _declared_hardware_path(profile)
    architecture = raw.get("arch")
    if (
        path != declared_path
        or _normalize_hardware_name(name)
        not in profile.allowed_normalized_gpu_models
        or not isinstance(architecture, Mapping)
        or architecture.get("sm") != profile.tvm_arch
    ):
        raise P6Stage1BridgeError()


def _profile_target_manifest(
    manifest: Mapping[str, Any], profile: HardwareExecutionProfile
) -> dict[str, Any]:
    owned = copy.deepcopy(dict(manifest))
    if profile.profile_id == "h800":
        return owned
    capability = owned["hw_capability"]
    return {
        **owned,
        "hw_capability": {
            **dict(capability),
            "name": profile.target_hardware_id,
        },
    }


def _is_h800_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = "_".join(value.casefold().strip().split())
    if normalized.startswith("nvidia_"):
        normalized = normalized.removeprefix("nvidia_")
    return normalized == "h800" or normalized.startswith("h800_")


def _normalize_hardware_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.upper() if character.isalnum())


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
