"""Strict, side-effect-free parsing for P6 H800 search execution contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import yaml


PUBLIC_SCHEMA_VERSION = "p6_h800_search_contract_v1"
LOCAL_SCHEMA_VERSION = "p6_h800_search_local_v1"
_PUBLIC_KEYS = frozenset(
    {
        "schema_version",
        "search_id",
        "target",
        "target_model",
        "seed",
        "max_rounds",
        "batch_size",
        "configuration_label",
        "assets",
        "metric_names",
        "candidate_space_label",
    }
)
_LOCAL_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "asset_paths",
        "stage5_input_paths",
        "steps",
        "result_step",
        "result_path_template",
        "local_output_root",
    }
)
_ASSET_KEYS = frozenset({"label", "version", "license_status"})
_STEP_KEYS = frozenset({"name", "argv"})
_ALLOWED_TEMPLATE_TOKENS = frozenset(
    {"{candidate_request}", "{result_json}", "{local_output_root}"}
)
_PUBLIC_RESTRICTED_VALUE_PATTERNS = (
    re.compile(r"(?:^|[-_\s])host(?:name)?(?:[-_\s]|\d|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])raw[-_\s]?logs?(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])checkpoint(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])candidate(?:[-_\s]*id)?[-_\s]*\d+(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])(?:sha(?:1|224|256|384|512)?|hash)(?:[:=_-]|$)", re.IGNORECASE),
    re.compile(r"^(?:python(?:\d+(?:\.\d+)?)?|bash|sh|zsh|cmd(?:\.exe)?|powershell|pwsh)\b", re.IGNORECASE),
)
_RAW_DIGEST_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}|[0-9a-f]{128}", re.IGNORECASE)


class H800SearchContractError(ValueError):
    """Raised when a P6 public or local search contract is unsafe or incomplete."""


@dataclass(frozen=True)
class RegisteredAsset:
    label: str
    version: str
    license_status: str


@dataclass(frozen=True)
class PublicH800SearchContract:
    search_id: str
    target: str
    target_model: str
    seed: int
    max_rounds: int
    batch_size: int
    configuration_label: str
    assets: tuple[RegisteredAsset, ...]
    metric_names: tuple[str, ...]
    candidate_space_label: str


@dataclass(frozen=True)
class LocalExecutionStep:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class LocalH800SearchConfig:
    asset_paths: Mapping[str, Path]
    stage5_input_paths: Mapping[str, Path]
    steps: tuple[LocalExecutionStep, ...]
    result_step: str
    result_path_template: Path
    local_output_root: Path


def load_public_contract(path: Path) -> PublicH800SearchContract:
    """Load a public, location-free H800 search declaration without side effects."""
    raw_contract = _load_mapping(path, "public contract")
    _reject_public_execution_details(raw_contract)
    _require_exact_keys(raw_contract, _PUBLIC_KEYS, "public contract")

    if raw_contract["schema_version"] != PUBLIC_SCHEMA_VERSION:
        raise H800SearchContractError("public schema_version is invalid")
    if raw_contract["target"] != "h800":
        raise H800SearchContractError("target must be h800")

    assets = _parse_assets(raw_contract["assets"])
    metric_names = _parse_string_list(raw_contract["metric_names"], "metric_names")
    return PublicH800SearchContract(
        search_id=_require_public_identifier(raw_contract["search_id"], "search_id"),
        target="h800",
        target_model=_require_public_identifier(raw_contract["target_model"], "target_model"),
        seed=_require_positive_integer(raw_contract["seed"], "seed"),
        max_rounds=_require_positive_integer(raw_contract["max_rounds"], "max_rounds"),
        batch_size=_require_positive_integer(raw_contract["batch_size"], "batch_size"),
        configuration_label=_require_public_identifier(
            raw_contract["configuration_label"], "configuration_label"
        ),
        assets=assets,
        metric_names=metric_names,
        candidate_space_label=_require_public_identifier(
            raw_contract["candidate_space_label"], "candidate_space_label"
        ),
    )


def load_local_config(path: Path, contract: PublicH800SearchContract) -> LocalH800SearchConfig:
    """Load local execution details without touching referenced assets or outputs."""
    raw_config = _load_mapping(path, "local config")
    _require_exact_keys(raw_config, _LOCAL_KEYS, "local config")

    if raw_config["schema_version"] != LOCAL_SCHEMA_VERSION:
        raise H800SearchContractError("local schema_version is invalid")
    if raw_config["target"] != contract.target:
        raise H800SearchContractError("local target must match the public contract")

    asset_paths = _parse_path_mapping(raw_config["asset_paths"], "asset_paths")
    registered_labels = {asset.label for asset in contract.assets}
    if set(asset_paths) != registered_labels:
        raise H800SearchContractError("local asset labels must exactly match public asset labels")

    stage5_input_paths = _parse_path_mapping(raw_config["stage5_input_paths"], "stage5_input_paths")
    steps = _parse_steps(raw_config["steps"])
    result_step = _require_nonempty_string(raw_config["result_step"], "result_step")
    if result_step != steps[-1].name:
        raise H800SearchContractError("result_step must reference the last step")
    if "{result_json}" not in steps[-1].argv:
        raise H800SearchContractError("result_step must produce {result_json}")

    return LocalH800SearchConfig(
        asset_paths=MappingProxyType(asset_paths),
        stage5_input_paths=MappingProxyType(stage5_input_paths),
        steps=steps,
        result_step=result_step,
        result_path_template=_parse_absolute_path(
            raw_config["result_path_template"], "result_path_template"
        ),
        local_output_root=_parse_absolute_path(
            raw_config["local_output_root"], "local_output_root"
        ),
    )


def _load_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise H800SearchContractError(f"could not load {description}") from error
    if not isinstance(loaded, Mapping):
        raise H800SearchContractError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise H800SearchContractError(f"{description} keys must be strings")
    return loaded


def _reject_public_execution_details(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _is_forbidden_public_key(key):
                raise H800SearchContractError("public contract contains forbidden execution detail")
            _reject_public_execution_details(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _reject_public_execution_details(nested_value)


def _is_forbidden_public_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower().replace("-", "_")
    return (
        "path" in normalized
        or "command" in normalized
        or "cmd" in normalized
        or "host" in normalized
        or normalized == "candidate_id"
        or "hash" in normalized
        or normalized.startswith("sha")
        or normalized in {"raw_log", "raw_logs", "checkpoint"}
    )


def _require_exact_keys(value: Mapping[str, Any], allowed: frozenset[str], description: str) -> None:
    keys = set(value)
    unknown = keys - allowed
    missing = allowed - keys
    if unknown:
        raise H800SearchContractError(f"{description} contains unknown keys: {sorted(unknown)}")
    if missing:
        raise H800SearchContractError(f"{description} is missing required keys: {sorted(missing)}")


def _parse_assets(value: Any) -> tuple[RegisteredAsset, ...]:
    if not isinstance(value, list) or not value:
        raise H800SearchContractError("assets must be a non-empty list")

    assets: list[RegisteredAsset] = []
    labels: set[str] = set()
    for raw_asset in value:
        if not isinstance(raw_asset, Mapping) or not all(isinstance(key, str) for key in raw_asset):
            raise H800SearchContractError("each asset must be a mapping")
        _require_exact_keys(raw_asset, _ASSET_KEYS, "asset")
        asset = RegisteredAsset(
            label=_require_public_identifier(raw_asset["label"], "asset label"),
            version=_require_public_identifier(raw_asset["version"], "asset version"),
            license_status=_require_public_identifier(
                raw_asset["license_status"], "asset license_status"
            ),
        )
        if asset.label in labels:
            raise H800SearchContractError("asset labels must be unique")
        labels.add(asset.label)
        assets.append(asset)
    return tuple(assets)


def _parse_string_list(value: Any, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise H800SearchContractError(f"{description} must be a non-empty list")
    parsed = tuple(_require_public_identifier(item, description) for item in value)
    if len(set(parsed)) != len(parsed):
        raise H800SearchContractError(f"{description} must contain unique values")
    return parsed


def _parse_path_mapping(value: Any, description: str) -> dict[str, Path]:
    if not isinstance(value, Mapping) or not value:
        raise H800SearchContractError(f"{description} must be a non-empty mapping")
    parsed: dict[str, Path] = {}
    for key, raw_path in value.items():
        label = _require_nonempty_string(key, f"{description} label")
        parsed[label] = _parse_absolute_path(raw_path, f"{description}[{label}]")
    return parsed


def _parse_steps(value: Any) -> tuple[LocalExecutionStep, ...]:
    if not isinstance(value, list) or not value:
        raise H800SearchContractError("steps must be a non-empty list")

    steps: list[LocalExecutionStep] = []
    names: set[str] = set()
    for raw_step in value:
        if not isinstance(raw_step, Mapping) or not all(isinstance(key, str) for key in raw_step):
            raise H800SearchContractError("each step must be a mapping")
        _require_exact_keys(raw_step, _STEP_KEYS, "step")
        name = _require_nonempty_string(raw_step["name"], "step name")
        if name in names:
            raise H800SearchContractError("step names must be unique")
        names.add(name)
        steps.append(LocalExecutionStep(name=name, argv=_parse_argv(raw_step["argv"])))
    return tuple(steps)


def _parse_argv(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise H800SearchContractError("step argv must be a non-empty list")
    argv = tuple(_require_nonempty_string(token, "argv token") for token in value)
    for token in argv:
        if ("{" in token or "}" in token) and token not in _ALLOWED_TEMPLATE_TOKENS:
            raise H800SearchContractError("argv contains an unknown or embedded placeholder")
    return argv


def _parse_absolute_path(value: Any, description: str) -> Path:
    path = Path(_require_nonempty_string(value, description))
    if not path.is_absolute():
        raise H800SearchContractError(f"{description} must be an absolute path")
    return path


def _require_nonempty_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise H800SearchContractError(f"{description} must be a non-empty string")
    return value


def _require_public_identifier(value: Any, description: str) -> str:
    identifier = _require_nonempty_string(value, description)
    if "/" in identifier or "\\" in identifier:
        raise H800SearchContractError(f"{description} contains restricted public information")
    if _RAW_DIGEST_PATTERN.fullmatch(identifier) or any(
        pattern.search(identifier) for pattern in _PUBLIC_RESTRICTED_VALUE_PATTERNS
    ):
        raise H800SearchContractError(f"{description} contains restricted public information")
    return identifier


def _require_positive_integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise H800SearchContractError(f"{description} must be a positive integer")
    return value
