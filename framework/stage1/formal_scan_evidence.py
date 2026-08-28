"""Scanner-owned config, checkpoint, module-width, and scenario evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from framework.stage1.adapters import ScanScenario
from framework.stage1.formal_checkpoint_snapshot import FormalCheckpointSnapshot
from framework.stage1.formal_config_evidence import canonicalize_model_config
from framework.stage1.formal_config_snapshot import FormalConfigSnapshot


_SCENARIO_FIELDS = frozenset(
    {
        "hardware_precisions",
        "backend_precisions",
        "compression_modes",
        "graph_quant_unit_policy",
        "alignment",
    }
)


def load_scan_scenario(path: Path, *, trusted_root: Path) -> ScanScenario:
    """Load one explicit, exact-schema formal scan scenario."""
    try:
        raw = _yaml_mapping(
            _trusted_file(path, trusted_root), "formal scan scenario"
        )
        if set(raw) != _SCENARIO_FIELDS:
            raise ValueError("formal scan scenario fields are invalid")
        return ScanScenario(**dict(raw))
    except (OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise ValueError("formal scan scenario is invalid") from error


def validate_scenario_hardware(scenario: ScanScenario, hardware: Any) -> None:
    """Require every requested hardware precision to be scanner-capability legal."""
    requested = {_precision(value) for value in scenario.hardware_precisions}
    supported = {_precision(value) for value in hardware.legal_bits}
    if not requested or not requested.issubset(supported):
        raise ValueError("formal scenario hardware precisions are incompatible")


def load_formal_scan_evidence(
    adapter: Any,
    traced_net: Any,
    *,
    checkpoint_authority: FormalCheckpointSnapshot,
    config_authority: FormalConfigSnapshot,
    loaded_model_config: Any,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Bind real config and checkpoint bytes to all trace module output widths."""
    try:
        if not isinstance(checkpoint_authority, FormalCheckpointSnapshot):
            raise TypeError("formal checkpoint authority is required")
        if not isinstance(config_authority, FormalConfigSnapshot):
            raise TypeError("formal config authority is required")
        loaded_config = canonicalize_model_config(loaded_model_config)
        checkpoint_state = _checkpoint_state_dict(checkpoint_authority.loader_path)
        module_widths = _checkpoint_module_widths(traced_net, checkpoint_state)
        if not module_widths:
            raise ValueError("trace has no output channel evidence")
        return loaded_config, {
            "digest": checkpoint_authority.digest,
            "config_file_digest": config_authority.digest,
            "module_widths": module_widths,
        }
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"formal scan evidence is invalid: {error}") from error


def resolve_formal_checkpoint_path(adapter: Any) -> Path:
    """Resolve one checkpoint identity before building the formal trace."""
    try:
        resolver = getattr(adapter, "resolved_checkpoint_path", None)
        path = Path(resolver() if callable(resolver) else adapter.ckpt_path)
        return _trusted_file(
            path,
            _declared_root(adapter, "formal_checkpoint_root"),
        )
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ValueError("formal scan checkpoint identity is invalid") from error


def resolve_formal_config_path(adapter: Any) -> Path:
    """Resolve one trusted model config before invoking its native loader."""
    try:
        return _trusted_file(
            Path(adapter.config_path),
            _declared_root(adapter, "formal_config_root"),
        )
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ValueError("formal scan config identity is invalid") from error


def _checkpoint_module_widths(
    net: Any, checkpoint_state: Mapping[str, Any]
) -> dict[str, int]:
    modules = dict(net.named_modules())
    key_map = _validated_checkpoint_module_key_map(net, modules)
    widths = {}
    for name, module in modules.items():
        width = getattr(module, "out_channels", None)
        if isinstance(width, int) and not isinstance(width, bool) and width > 0:
            source_name = str(name)
            if key_map is not None:
                if source_name not in key_map:
                    raise ValueError("checkpoint module mapping is missing a selected module")
                source_name = key_map[source_name]
            tensor = _selected_weight(checkpoint_state, source_name, module)
            checkpoint_width = _tensor_out_width(module, tensor)
            if checkpoint_width != width:
                raise ValueError("selected checkpoint tensor output width disagrees")
            widths[str(name)] = checkpoint_width
    return widths


def _validated_checkpoint_module_key_map(
    net: Any, modules: Mapping[str, Any]
) -> dict[str, str] | None:
    raw = getattr(net, "checkpoint_module_key_map", None)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint module mapping must be a mapping")
    normalized: dict[str, str] = {}
    for wrapper_name, source_name in raw.items():
        if not isinstance(wrapper_name, str) or not wrapper_name:
            raise ValueError("checkpoint module mapping has an invalid wrapper name")
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("checkpoint module mapping has an invalid source name")
        normalized[wrapper_name] = source_name
    if set(normalized) - set(modules):
        raise ValueError("checkpoint module mapping contains an unknown wrapper module")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("checkpoint module mapping contains duplicate source modules")
    return normalized


def _selected_weight(
    checkpoint_state: Mapping[str, Any], module_name: str, module: Any
) -> torch.Tensor:
    key = f"{module_name}.weight" if module_name else "weight"
    tensor = checkpoint_state.get(key)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"selected checkpoint tensor is missing: {key}")
    live_weight = getattr(module, "weight", None)
    if not isinstance(live_weight, torch.Tensor) or tensor.shape != live_weight.shape:
        raise ValueError(f"selected checkpoint tensor shape is incompatible: {key}")
    return tensor


def _tensor_out_width(module: Any, tensor: torch.Tensor) -> int:
    transposed = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    if isinstance(module, transposed):
        return int(tensor.shape[1]) * int(module.groups)
    return int(tensor.shape[0])


def _checkpoint_state_dict(path: Path) -> Mapping[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    state = raw.get("model_state_dict", raw)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")
    return state


def _declared_root(adapter: Any, attribute: str) -> Path:
    root = getattr(adapter, attribute, None)
    if not isinstance(root, Path):
        raise ValueError("formal artifact path is invalid")
    return root


def _trusted_file(path: Path, trusted_root: Path) -> Path:
    """Resolve an exact regular file inside one explicit, non-symlink root."""
    try:
        if (
            not isinstance(path, Path)
            or not isinstance(trusted_root, Path)
            or not path.is_absolute()
            or not trusted_root.is_absolute()
            or _contains_symlink_component(path)
            or _contains_symlink_component(trusted_root)
        ):
            raise ValueError
        resolved_root = trusted_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        if not resolved_root.is_dir() or not resolved.is_file():
            raise ValueError
        return resolved
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("formal artifact path is invalid") from error


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )


def _yaml_mapping(path: Path, field: str) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return raw


def _precision(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("hardware precision must be a non-empty string")
    return value.strip().upper()


__all__ = [
    "load_formal_scan_evidence",
    "load_scan_scenario",
    "resolve_formal_config_path",
    "resolve_formal_checkpoint_path",
    "validate_scenario_hardware",
]
