"""Explicit, redacted environment input loaders.

This module deliberately only parses caller-provided files.  It does not inspect
the host environment, invoke subprocesses, or initialize runtime frameworks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from framework.capability_schema import HardwareCapability


class EnvironmentContractError(ValueError):
    """Raised when an explicit public environment input is invalid."""


@dataclass(frozen=True)
class RuntimeConstraint:
    python: str
    cuda: str | None
    driver: str | None
    framework_name: str
    framework_version: str


@dataclass(frozen=True)
class EnvironmentContract:
    target: str
    hardware_capability: Path
    requires_gpu: bool
    runtime: RuntimeConstraint


@dataclass(frozen=True)
class RuntimeObservation:
    python: str
    cuda: str | None
    driver: str | None
    framework_name: str
    framework_version: str


@dataclass(frozen=True)
class EnvironmentObservation:
    target: str
    runtime: RuntimeObservation
    gpu_count: int


_CONTRACT_FIELDS = {"target", "hardware_capability", "requires_gpu", "runtime"}
_RUNTIME_FIELDS = {"python", "cuda", "driver", "framework_name", "framework_version"}
_OBSERVATION_FIELDS = {"target", "runtime", "gpu_count", "note"}
_Runtime = TypeVar("_Runtime", RuntimeConstraint, RuntimeObservation)


def load_environment_contract(path: Path, *, repository_root: Path) -> EnvironmentContract:
    """Load a declared environment contract without examining the local machine."""
    document = _load_yaml_object(path)
    _require_exact_fields(document, _CONTRACT_FIELDS, "contract")

    target = _require_string(document, "target", "contract")
    hardware_path = _require_string(document, "hardware_capability", "contract")
    requires_gpu = document["requires_gpu"]
    if type(requires_gpu) is not bool:
        raise EnvironmentContractError("contract.requires_gpu must be a boolean")
    runtime = _parse_runtime(document["runtime"], RuntimeConstraint, "contract.runtime")
    _validate_runtime_for_target(target, requires_gpu, runtime.cuda, runtime.driver)

    relative_capability = Path(hardware_path)
    if relative_capability.is_absolute() or ".." in relative_capability.parts:
        raise EnvironmentContractError("hardware_capability must be a repository-relative path")
    root = repository_root.resolve()
    resolved_capability = (root / relative_capability).resolve()
    if not resolved_capability.is_relative_to(root):
        raise EnvironmentContractError("hardware_capability must remain within repository_root")
    if relative_capability.stem != target:
        raise EnvironmentContractError("target must match the hardware capability filename")
    try:
        HardwareCapability.from_yaml(resolved_capability)
    except Exception as exc:
        raise EnvironmentContractError("hardware_capability is not a valid capability YAML") from exc

    return EnvironmentContract(
        target=target,
        hardware_capability=relative_capability,
        requires_gpu=requires_gpu,
        runtime=runtime,
    )


def load_environment_observation(path: Path) -> EnvironmentObservation:
    """Load an explicit observation, dropping the optional human-readable note."""
    document = _load_json_object(path)
    _require_allowed_fields(document, _OBSERVATION_FIELDS, "observation")
    target = _require_string(document, "target", "observation")
    runtime = _parse_runtime(document.get("runtime"), RuntimeObservation, "observation.runtime")
    gpu_count = document.get("gpu_count")
    if type(gpu_count) is not int or gpu_count < 0:
        raise EnvironmentContractError("observation.gpu_count must be a non-negative integer")
    _validate_runtime_for_target(target, target != "cpu", runtime.cuda, runtime.driver)
    if target == "cpu" and gpu_count != 0:
        raise EnvironmentContractError("CPU observations must report gpu_count as 0")
    return EnvironmentObservation(target=target, runtime=runtime, gpu_count=gpu_count)


def _load_yaml_object(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EnvironmentContractError("contract YAML could not be loaded") from exc
    return _require_object(document, "contract")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentContractError("observation JSON could not be loaded") from exc
    return _require_object(document, "observation")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise EnvironmentContractError(f"{label} must be an object")
    return value


def _require_exact_fields(document: dict[str, Any], fields: set[str], label: str) -> None:
    _require_allowed_fields(document, fields, label)
    if set(document) != fields:
        raise EnvironmentContractError(f"{label} is missing required fields")


def _require_allowed_fields(document: dict[str, Any], fields: set[str], label: str) -> None:
    if not set(document).issubset(fields):
        raise EnvironmentContractError(f"{label} contains unknown fields")


def _require_string(document: dict[str, Any], field: str, label: str) -> str:
    value = document.get(field)
    if type(value) is not str or not value:
        raise EnvironmentContractError(f"{label}.{field} must be a non-empty string")
    return value


def _parse_runtime(
    value: Any, runtime_type: type[_Runtime], label: str
) -> _Runtime:
    runtime = _require_object(value, label)
    _require_exact_fields(runtime, _RUNTIME_FIELDS, label)
    python = _require_string(runtime, "python", label)
    framework_name = _require_string(runtime, "framework_name", label)
    framework_version = _require_string(runtime, "framework_version", label)
    cuda = runtime["cuda"]
    driver = runtime["driver"]
    if cuda is not None and (type(cuda) is not str or not cuda):
        raise EnvironmentContractError(f"{label}.cuda must be a non-empty string or null")
    if driver is not None and (type(driver) is not str or not driver):
        raise EnvironmentContractError(f"{label}.driver must be a non-empty string or null")
    return runtime_type(python, cuda, driver, framework_name, framework_version)


def _validate_runtime_for_target(
    target: str, requires_gpu: bool, cuda: str | None, driver: str | None
) -> None:
    if target == "cpu" and requires_gpu:
        raise EnvironmentContractError("CPU targets cannot require a GPU")
    if requires_gpu and (cuda is None or driver is None):
        raise EnvironmentContractError("GPU targets require CUDA and driver versions")
    if not requires_gpu and (cuda is not None or driver is not None):
        raise EnvironmentContractError("CPU targets cannot declare CUDA or a driver")


__all__ = [
    "EnvironmentContractError",
    "RuntimeConstraint",
    "EnvironmentContract",
    "RuntimeObservation",
    "EnvironmentObservation",
    "load_environment_contract",
    "load_environment_observation",
]
