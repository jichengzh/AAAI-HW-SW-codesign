"""Side-effect-free contracts for the local P6 CoptV2X H800/TVM search."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import yaml


PUBLIC_SCHEMA_VERSION = "p6_h800_coptv2x_search_contract_v2"
LOCAL_SCHEMA_VERSION = "p6_h800_coptv2x_local_v2"
METRIC_NAMES = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
FIXED_TARGET = "h800"
FIXED_MODEL = "pyramid"
FIXED_BACKEND = "tvm_auto"
FIXED_SAMPLE_BUDGET = 16
FIXED_BATCH_SIZE = 4
FIXED_ROUND_COUNT = 4
LOCAL_INPUT_NAMES = frozenset(
    {"gold176_rows", "gold176_graph_features", "capability_profiles", "closure"}
)
ALLOWED_TEMPLATE_TOKENS = frozenset(
    {
        "{local_output_root}",
        "{source_registry_json}",
        "{measurement_request}",
        "{feedback_json}",
        "{round_output_root}",
    }
)
PUBLIC_KEYS = frozenset(
    {
        "schema_version",
        "search_id",
        "target",
        "target_model",
        "execution_backend",
        "seed",
        "sample_budget",
        "batch_size",
        "round_count",
        "configuration_label",
        "candidate_space_label",
        "metric_names",
        "assets",
    }
)
LOCAL_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "asset_paths",
        "local_input_paths",
        "source_registry_step",
        "measurement_step",
        "local_output_root",
    }
)

_ASSET_KEYS = frozenset({"label", "version", "license_status"})
_STEP_KEYS = frozenset({"name", "argv"})
_SHELL_EXECUTABLES = frozenset(
    {
        "bash",
        "bash.exe",
        "cmd",
        "cmd.exe",
        "csh",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
        "sh.exe",
        "tcsh",
        "zsh",
    }
)
_COMMAND_WRAPPER_EXECUTABLES = frozenset({"env", "env.exe"})
_PUBLIC_RESTRICTED_VALUE_PATTERNS = (
    re.compile(r"(?:^|[-_\s])host(?:name)?(?:[-_\s]|\d|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])raw[-_\s]?logs?(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[-_\s])checkpoint(?:[-_\s]|$)", re.IGNORECASE),
    re.compile(
        r"(?:^|[-_\s])candidate(?:[-_\s]+id)?[-_\s]+[a-z0-9]+(?:[-_\s]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|[-_\s])(?:sha(?:1|224|256|384|512)?|hash)(?:[:=_-]|$)", re.IGNORECASE),
    re.compile(
        r"^(?:python(?:\d+(?:\.\d+)?)?|bash|sh|zsh|cmd(?:\.exe)?|powershell|pwsh)\b",
        re.IGNORECASE,
    ),
)
_RAW_DIGEST_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}|[0-9a-f]{128}", re.IGNORECASE)


class P6CoptV2XContractError(ValueError):
    """Raised when a P6 public or local CoptV2X contract is invalid."""


@dataclass(frozen=True)
class RegisteredAsset:
    label: str
    version: str
    license_status: str


@dataclass(frozen=True)
class PublicP6CoptV2XContract:
    search_id: str
    target: str
    target_model: str
    execution_backend: str
    seed: int
    sample_budget: int
    batch_size: int
    round_count: int
    configuration_label: str
    candidate_space_label: str
    assets: tuple[RegisteredAsset, ...]
    metric_names: tuple[str, ...]


@dataclass(frozen=True)
class LocalExecutionStep:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class LocalP6CoptV2XConfig:
    asset_paths: Mapping[str, Path]
    local_input_paths: Mapping[str, Path]
    source_registry_step: LocalExecutionStep
    measurement_step: LocalExecutionStep
    local_output_root: Path


def load_public_contract(path: Path) -> PublicP6CoptV2XContract:
    """Load the fixed, non-executable public P6 CoptV2X search contract."""
    payload = _load_mapping(path, "public contract")
    _reject_public_execution_details(payload)
    _require_exact_keys(payload, PUBLIC_KEYS, "public contract")
    if payload["schema_version"] != PUBLIC_SCHEMA_VERSION:
        raise P6CoptV2XContractError("public schema_version is invalid")
    if _require_public_identifier(payload["target"], "target") != FIXED_TARGET:
        raise P6CoptV2XContractError("target must be h800")
    if _require_public_identifier(payload["target_model"], "target_model") != FIXED_MODEL:
        raise P6CoptV2XContractError("target_model must be pyramid")
    if _require_public_identifier(payload["execution_backend"], "execution_backend") != FIXED_BACKEND:
        raise P6CoptV2XContractError("execution_backend must be tvm_auto")
    if _require_positive_integer(payload["sample_budget"], "sample_budget") != FIXED_SAMPLE_BUDGET:
        raise P6CoptV2XContractError("sample_budget must be 16")
    if _require_positive_integer(payload["batch_size"], "batch_size") != FIXED_BATCH_SIZE:
        raise P6CoptV2XContractError("batch_size must be 4")
    if _require_positive_integer(payload["round_count"], "round_count") != FIXED_ROUND_COUNT:
        raise P6CoptV2XContractError("round_count must be 4")
    metric_names = _parse_string_list(payload["metric_names"], "metric_names")
    if metric_names != METRIC_NAMES:
        raise P6CoptV2XContractError("metric_names must exactly match runtime metrics")
    return PublicP6CoptV2XContract(
        search_id=_require_public_identifier(payload["search_id"], "search_id"),
        target=FIXED_TARGET,
        target_model=FIXED_MODEL,
        execution_backend=FIXED_BACKEND,
        seed=_require_positive_integer(payload["seed"], "seed"),
        sample_budget=FIXED_SAMPLE_BUDGET,
        batch_size=FIXED_BATCH_SIZE,
        round_count=FIXED_ROUND_COUNT,
        configuration_label=_require_public_identifier(
            payload["configuration_label"], "configuration_label"
        ),
        candidate_space_label=_require_public_identifier(
            payload["candidate_space_label"], "candidate_space_label"
        ),
        assets=_parse_assets(payload["assets"]),
        metric_names=metric_names,
    )


def load_local_config(path: Path, contract: PublicP6CoptV2XContract) -> LocalP6CoptV2XConfig:
    """Load private, argv-only execution details without publishing them."""
    if not isinstance(contract, PublicP6CoptV2XContract):
        raise P6CoptV2XContractError("public contract is invalid")
    payload = _load_mapping(path, "local config")
    _require_exact_keys(payload, LOCAL_KEYS, "local config")
    if payload["schema_version"] != LOCAL_SCHEMA_VERSION:
        raise P6CoptV2XContractError("local schema_version is invalid")
    if _require_nonempty_string(payload["target"], "local target") != contract.target:
        raise P6CoptV2XContractError("local target must match the public contract")
    asset_paths = _parse_path_mapping(payload["asset_paths"], "asset_paths")
    if set(asset_paths) != {asset.label for asset in contract.assets}:
        raise P6CoptV2XContractError("local asset labels must exactly match public asset labels")
    local_input_paths = _parse_path_mapping(payload["local_input_paths"], "local_input_paths")
    if set(local_input_paths) != LOCAL_INPUT_NAMES:
        raise P6CoptV2XContractError("local input labels are invalid")
    return LocalP6CoptV2XConfig(
        asset_paths=MappingProxyType(asset_paths),
        local_input_paths=MappingProxyType(local_input_paths),
        source_registry_step=_load_step(
            payload["source_registry_step"],
            expected_name="build_source_registry",
            required_tokens={"{local_output_root}", "{source_registry_json}"},
        ),
        measurement_step=_load_step(
            payload["measurement_step"],
            expected_name="measure_batch",
            required_tokens={"{measurement_request}", "{feedback_json}", "{round_output_root}"},
        ),
        local_output_root=_parse_absolute_path(payload["local_output_root"], "local_output_root"),
    )


def _load_step(
    payload: object, *, expected_name: str, required_tokens: set[str]
) -> LocalExecutionStep:
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise P6CoptV2XContractError("step keys invalid")
    if set(payload) != _STEP_KEYS:
        raise P6CoptV2XContractError("step keys invalid")
    if payload["name"] != expected_name:
        raise P6CoptV2XContractError("step name invalid")
    argv = _validate_argv(payload["argv"])
    present = {token for arg in argv for token in ALLOWED_TEMPLATE_TOKENS if token in arg}
    if present != required_tokens:
        raise P6CoptV2XContractError("template tokens invalid")
    return LocalExecutionStep(name=expected_name, argv=argv)


def _validate_argv(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise P6CoptV2XContractError("step argv must be a non-empty list")
    argv = tuple(_require_nonempty_string(token, "argv token") for token in value)
    executable = argv[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if executable in _SHELL_EXECUTABLES:
        raise P6CoptV2XContractError("step argv must not invoke a shell executable")
    if executable in _COMMAND_WRAPPER_EXECUTABLES:
        raise P6CoptV2XContractError("step argv must not invoke a command wrapper")
    for token in argv:
        if ("{" in token or "}" in token) and token not in ALLOWED_TEMPLATE_TOKENS:
            raise P6CoptV2XContractError("argv template contains an unknown or embedded placeholder")
    return argv


def _load_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise P6CoptV2XContractError(f"could not load {description}") from error
    if not isinstance(loaded, Mapping):
        raise P6CoptV2XContractError(f"{description} must be a mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise P6CoptV2XContractError(f"{description} keys must be strings")
    return loaded


def _reject_public_execution_details(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _is_forbidden_public_key(key):
                raise P6CoptV2XContractError("public contract contains forbidden execution detail")
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
        or normalized in {"raw_log", "raw_logs", "checkpoint", "measurements"}
    )


def _require_exact_keys(value: Mapping[str, Any], allowed: frozenset[str], description: str) -> None:
    keys = set(value)
    unknown = keys - allowed
    missing = allowed - keys
    if unknown:
        raise P6CoptV2XContractError(f"{description} contains unknown keys: {sorted(unknown)}")
    if missing:
        raise P6CoptV2XContractError(f"{description} is missing required keys: {sorted(missing)}")


def _parse_assets(value: object) -> tuple[RegisteredAsset, ...]:
    if not isinstance(value, list) or not value:
        raise P6CoptV2XContractError("assets must be a non-empty list")
    assets: list[RegisteredAsset] = []
    labels: set[str] = set()
    for raw_asset in value:
        if not isinstance(raw_asset, Mapping) or not all(isinstance(key, str) for key in raw_asset):
            raise P6CoptV2XContractError("each asset must be a mapping")
        _require_exact_keys(raw_asset, _ASSET_KEYS, "asset")
        asset = RegisteredAsset(
            label=_require_public_identifier(raw_asset["label"], "asset label"),
            version=_require_public_identifier(raw_asset["version"], "asset version"),
            license_status=_require_public_identifier(raw_asset["license_status"], "asset license_status"),
        )
        if asset.label in labels:
            raise P6CoptV2XContractError("asset labels must be unique")
        labels.add(asset.label)
        assets.append(asset)
    return tuple(assets)


def _parse_string_list(value: object, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise P6CoptV2XContractError(f"{description} must be a non-empty list")
    parsed = tuple(_require_public_identifier(item, description) for item in value)
    if len(set(parsed)) != len(parsed):
        raise P6CoptV2XContractError(f"{description} must contain unique values")
    return parsed


def _parse_path_mapping(value: object, description: str) -> dict[str, Path]:
    if not isinstance(value, Mapping) or not value:
        raise P6CoptV2XContractError(f"{description} must be a non-empty mapping")
    return {
        _require_nonempty_string(key, f"{description} label"): _parse_absolute_path(
            raw_path, f"{description}[{key}]"
        )
        for key, raw_path in value.items()
    }


def _parse_absolute_path(value: object, description: str) -> Path:
    path = Path(_require_nonempty_string(value, description))
    if not path.is_absolute():
        raise P6CoptV2XContractError(f"{description} must be an absolute path")
    return path


def _require_nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P6CoptV2XContractError(f"{description} must be a non-empty string")
    return value


def _require_public_identifier(value: object, description: str) -> str:
    identifier = _require_nonempty_string(value, description)
    if "/" in identifier or "\\" in identifier or any(character.isspace() for character in identifier):
        raise P6CoptV2XContractError(f"{description} contains restricted public information")
    if _RAW_DIGEST_PATTERN.search(identifier) or any(
        pattern.search(identifier) for pattern in _PUBLIC_RESTRICTED_VALUE_PATTERNS
    ):
        raise P6CoptV2XContractError(f"{description} contains restricted public information")
    return identifier


def _require_positive_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise P6CoptV2XContractError(f"{description} must be a positive integer")
    return value
