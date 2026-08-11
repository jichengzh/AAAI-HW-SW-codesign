"""Explicit, redacted environment input loaders.

This module deliberately only parses caller-provided files.  It does not inspect
the host environment, invoke subprocesses, or initialize runtime frameworks.
"""

from __future__ import annotations

import json
import re
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


@dataclass(frozen=True)
class ValidationFailure:
    code: str
    field: str


@dataclass(frozen=True)
class ValidationResult:
    target: str
    passed: bool
    failures: tuple[ValidationFailure, ...]


_CONTRACT_FIELDS = {"schema", "target", "hardware_capability", "requires_gpu", "runtime"}
_RUNTIME_FIELDS = {"python", "cuda", "driver", "framework_name", "framework_version"}
_OBSERVATION_FIELDS = {"schema", "target", "runtime", "gpu_count", "note"}
_OBSERVATION_REQUIRED_FIELDS = _OBSERVATION_FIELDS - {"note"}
_CPU_TARGET = "cpu_reference"
_TARGET_REQUIRES_GPU = {_CPU_TARGET: False, "rtx4090": True, "h800": True}
_Runtime = TypeVar("_Runtime", RuntimeConstraint, RuntimeObservation)
_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_VERSION_OPERATORS = (">=", "<=", ">", "<")


def version_satisfies(actual: str, expression: str) -> bool:
    """Return whether a numeric dotted version meets every declared constraint."""
    actual_parts = _version_parts(actual)
    if type(expression) is not str or not expression:
        raise EnvironmentContractError("version expression must be a non-empty string")

    for clause in expression.split(","):
        operator = next((item for item in _VERSION_OPERATORS if clause.startswith(item)), "")
        required = clause[len(operator) :] if operator else clause
        comparison = _compare_versions(actual_parts, _version_parts(required))
        if not _comparison_satisfies(comparison, operator):
            return False
    return True


def validate_environment(
    contract: EnvironmentContract, observation: EnvironmentObservation
) -> ValidationResult:
    """Validate an explicit observation against a contract without host inspection."""
    failures: list[ValidationFailure] = []
    if observation.target != contract.target:
        failures.append(ValidationFailure("observation.target.mismatch", "target"))
    if contract.requires_gpu and observation.gpu_count <= 0:
        failures.append(ValidationFailure("runtime.gpu_count.required", "gpu_count"))
    if not contract.requires_gpu and observation.gpu_count != 0:
        failures.append(ValidationFailure("runtime.gpu_count.unexpected", "gpu_count"))

    runtime = contract.runtime
    observed_runtime = observation.runtime
    _append_version_failure(
        failures, "runtime.python.out_of_range", "runtime.python", observed_runtime.python, runtime.python
    )
    _append_optional_version_failure(
        failures, "runtime.cuda.out_of_range", "runtime.cuda", observed_runtime.cuda, runtime.cuda
    )
    _append_optional_version_failure(
        failures, "runtime.driver.out_of_range", "runtime.driver", observed_runtime.driver, runtime.driver
    )
    if observed_runtime.framework_name != runtime.framework_name:
        failures.append(ValidationFailure("runtime.framework.name.mismatch", "runtime.framework_name"))
    _append_version_failure(
        failures,
        "runtime.framework.version.out_of_range",
        "runtime.framework_version",
        observed_runtime.framework_version,
        runtime.framework_version,
    )
    return ValidationResult(contract.target, not failures, tuple(failures))


def _version_parts(version: str) -> tuple[int, ...]:
    if type(version) is not str or _VERSION_PATTERN.fullmatch(version) is None:
        raise EnvironmentContractError("version must use numeric dotted parts")
    return tuple(int(part) for part in version.split("."))


def _compare_versions(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    padded_left = left + (0,) * (width - len(left))
    padded_right = right + (0,) * (width - len(right))
    return (padded_left > padded_right) - (padded_left < padded_right)


def _comparison_satisfies(comparison: int, operator: str) -> bool:
    if operator == ">=":
        return comparison >= 0
    if operator == "<=":
        return comparison <= 0
    if operator == ">":
        return comparison > 0
    if operator == "<":
        return comparison < 0
    return comparison == 0


def _append_version_failure(
    failures: list[ValidationFailure], code: str, field: str, actual: str, expression: str
) -> None:
    if not version_satisfies(actual, expression):
        failures.append(ValidationFailure(code, field))


def _append_optional_version_failure(
    failures: list[ValidationFailure],
    code: str,
    field: str,
    actual: str | None,
    expression: str | None,
) -> None:
    if actual is None and expression is None:
        return
    if actual is None or expression is None or not version_satisfies(actual, expression):
        failures.append(ValidationFailure(code, field))


def load_environment_contract(path: Path, *, repository_root: Path) -> EnvironmentContract:
    """Load a declared environment contract without examining the local machine."""
    document = _load_yaml_object(path)
    _require_exact_fields(document, _CONTRACT_FIELDS, "contract")
    _require_schema(document, "environment_contract_v1", "contract")

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
    _require_required_fields(document, _OBSERVATION_REQUIRED_FIELDS, "observation")
    _require_schema(document, "environment_observation_v1", "observation")
    target = _require_string(document, "target", "observation")
    runtime = _parse_runtime(document.get("runtime"), RuntimeObservation, "observation.runtime")
    gpu_count = document.get("gpu_count")
    if type(gpu_count) is not int or gpu_count < 0:
        raise EnvironmentContractError("observation.gpu_count must be a non-negative integer")
    requires_gpu = _requires_gpu_for_target(target)
    _validate_runtime_for_target(target, requires_gpu, runtime.cuda, runtime.driver)
    if requires_gpu and gpu_count <= 0:
        raise EnvironmentContractError("GPU observations must report gpu_count greater than 0")
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
    _require_required_fields(document, fields, label)


def _require_allowed_fields(document: dict[str, Any], fields: set[str], label: str) -> None:
    if not set(document).issubset(fields):
        raise EnvironmentContractError(f"{label} contains unknown fields")


def _require_required_fields(document: dict[str, Any], fields: set[str], label: str) -> None:
    if not fields.issubset(document):
        raise EnvironmentContractError(f"{label} is missing required fields")


def _require_schema(document: dict[str, Any], expected: str, label: str) -> None:
    if _require_string(document, "schema", label) != expected:
        raise EnvironmentContractError(f"{label}.schema must be {expected}")


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
    target_requires_gpu = _requires_gpu_for_target(target)
    if requires_gpu != target_requires_gpu:
        raise EnvironmentContractError("requires_gpu must match the formal target class")
    if target_requires_gpu and (cuda is None or driver is None):
        raise EnvironmentContractError("GPU targets require CUDA and driver versions")
    if not target_requires_gpu and (cuda is not None or driver is not None):
        raise EnvironmentContractError("CPU targets cannot declare CUDA or a driver")


def _requires_gpu_for_target(target: str) -> bool:
    try:
        return _TARGET_REQUIRES_GPU[target]
    except KeyError as exc:
        raise EnvironmentContractError("target must be cpu_reference, rtx4090, or h800") from exc


__all__ = [
    "EnvironmentContractError",
    "RuntimeConstraint",
    "EnvironmentContract",
    "RuntimeObservation",
    "EnvironmentObservation",
    "ValidationFailure",
    "ValidationResult",
    "load_environment_contract",
    "load_environment_observation",
    "version_satisfies",
    "validate_environment",
]
