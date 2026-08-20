"""Private, offline binding of documented Stage5 history for the P6 search."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Protocol
from types import MappingProxyType

from framework.stage5.production_search_v1 import validate_source_contract


BINDING_SCHEMA_VERSION = "p6_history_binding_v1"
PUBLIC_SCHEMA_VERSION = "p6_history_binding_public_v1"
SOURCE_REGISTRY_SCHEMA_VERSION = "stage5_candidate_source_registry_v1"
EXECUTION_INTERFACE_SCHEMA_VERSION = "p6_history_runner_interface_v1"
MAX_GPU_OCCUPANCY = 0.05
ALLOWED_NORMALIZED_H800_MODELS = frozenset(
    {
        "H800",
        "NVIDIAH800",
        "NVIDIAH80080GBHBM3",
    }
)
TARGET = {
    "model": "pyramid",
    "hardware": "h800",
    "backend": "tvm_auto",
}
COMPONENT_MARKERS = {
    "controller": ("stage5_task_round_controller_v3.sh", "v3"),
    "source_materializer": ("stage5_materialize_round_sources_v1.sh", "v1"),
    "performance_plan": ("stage5_build_performance_plan_v2.py", "v2"),
    "finalizer": ("stage5_finalize_feedback_v2.py", "v2"),
}
LOCAL_INPUT_NAMES = (
    "gold176_rows",
    "gold176_graph_features",
    "capability_profiles",
    "closure",
)
EXECUTION_STAGES = (
    "source_materialization",
    "quantization",
    "performance",
    "ap",
    "finalization",
)
EXECUTION_PLACEHOLDERS = frozenset(
    {
        "{measurement_request}",
        "{round_output_root}",
        "{p6_row_id}",
        "{task_state}",
        "{actual_feedback}",
        "{actual_receipt}",
        "{finalization_barrier}",
    }
)
REQUIRED_STAGE_PLACEHOLDERS = {
    "source_materialization": (
        "{measurement_request}",
        "{round_output_root}",
    ),
    "quantization": ("{task_state}", "{round_output_root}"),
    "performance": ("{task_state}", "{round_output_root}"),
    "ap": ("{task_state}", "{round_output_root}"),
    "finalization": (
        "{measurement_request}",
        "{task_state}",
        "{actual_feedback}",
        "{actual_receipt}",
        "{finalization_barrier}",
        "{round_output_root}",
    ),
}
EXPECTED_ROW_FIELD_NAMES = {
    "rows_key": "rows",
    "row_id_key": "row_id",
    "row_hash_key": "row_sha256",
    "source_evidence_key": "source_evidence_sha256",
    "status_key": "terminal_status",
}
ALLOWED_TERMINAL_STATUSES = (
    "measured_success_gold",
    "feasibility_failure",
    "numerical_feasibility_failure",
)
METRIC_KEYS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
WRAPPER_ENVIRONMENT_SPEC = {
    "CUDA_VISIBLE_DEVICES": ("literal", None),
    "P6_HISTORY_RUN_MODE": ("literal", "bound"),
    "P6_HISTORY_PRIVATE_ROOT": ("private_path", None),
    "P6_HISTORY_TASK_STATE": ("placeholder", "{task_state}"),
    "P6_HISTORY_ROUND_OUTPUT_ROOT": ("placeholder", "{round_output_root}"),
}
INTERFACE_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "controller",
        "execution_chain",
        "environment",
        "output_layout",
        "actual_feedback",
    }
)
SHELL_TOKENS = frozenset({"$", "`", ";", "|", "&", "<", ">", "\n", "\r"})
FORBIDDEN_PUBLIC_KEY_TOKENS = (
    "path",
    "command",
    "argv",
    "uuid",
    "identifier",
    "request",
    "feedback",
    "output",
    "checkpoint",
    "metric",
    "candidate",
    "result",
    "secret",
)


class P6HistoryBindingError(ValueError):
    """Stable categorized failure for private history binding operations."""

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")


@dataclass(frozen=True)
class GpuRecord:
    """Immutable GPU admission record supplied by an injected offline probe."""

    index: int
    uuid: str
    model_name: str
    occupancy: float


class GpuProbe(Protocol):
    """Probe boundary; production GPU access belongs to a later execution adapter."""

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]: ...


IgnorePredicate = Callable[[Path], bool]


def discover_history_binding(
    history_root: str | Path,
    gpu_probe: GpuProbe,
) -> dict[str, Any]:
    """Discover and validate one private Stage5 history binding beneath ``root``."""
    root = _resolve_history_root(history_root)
    component_paths = {
        role: str(_discover_one_named_path(root, marker))
        for role, (marker, _) in COMPONENT_MARKERS.items()
    }
    execution_interface = _discover_execution_interface(root, component_paths)
    local_input_paths = {
        name: str(_discover_one_named_path(root, f"{name}.json", category="local_inputs"))
        for name in LOCAL_INPUT_NAMES
    }
    return build_history_binding(
        root,
        component_paths=component_paths,
        execution_interface=execution_interface,
        local_input_paths=local_input_paths,
        gpu_probe=gpu_probe,
    )


def build_history_binding(
    history_root: str | Path,
    *,
    component_paths: Mapping[str, str],
    execution_interface: Mapping[str, Any],
    local_input_paths: Mapping[str, str],
    gpu_probe: GpuProbe,
) -> dict[str, Any]:
    """Build one binding from explicit paths and an already selected interface."""
    root = _resolve_history_root(history_root)
    canonical_components = _binding_component_paths(component_paths, root)
    canonical_inputs = _binding_local_input_paths(local_input_paths, root)
    canonical_interface = _freeze_mapping(
        _validate_execution_interface(
            _json_compatible(execution_interface), root, canonical_components
        )
    )
    gpu_indices = _private_gpu_indices(canonical_interface)
    registry_path, source_contract = _discover_source_contract(root)
    first_snapshot = _probe_snapshot(gpu_probe, gpu_indices)
    second_snapshot = _probe_snapshot(gpu_probe, gpu_indices)
    first_uuid_map = _validate_gpu_snapshot(first_snapshot, gpu_indices)
    second_uuid_map = _validate_gpu_snapshot(second_snapshot, gpu_indices)
    if first_uuid_map != second_uuid_map:
        raise P6HistoryBindingError("gpu_drift", "GPU UUIDs changed between snapshots")

    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "target": copy.deepcopy(TARGET),
        "private_root": str(root),
        "component_paths": canonical_components,
        "component_versions": {
            role: version for role, (_, version) in COMPONENT_MARKERS.items()
        },
        "execution_interface": canonical_interface,
        "source_registry_path": str(registry_path),
        "source_contract_template": copy.deepcopy(source_contract),
        "local_input_paths": canonical_inputs,
        "gpu_policy": {
            "indices": list(gpu_indices),
            "uuid_by_index": first_uuid_map,
            "model": "h800",
            "maximum_occupancy": MAX_GPU_OCCUPANCY,
        },
        "status": "validated",
    }


def public_binding_projection(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed safe-label projection; private binding content is never copied."""
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION:
        raise P6HistoryBindingError("public_projection", "unexpected binding schema")
    if binding.get("target") != TARGET:
        raise P6HistoryBindingError("public_projection", "unexpected binding target")
    projection = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "binding_schema_version": BINDING_SCHEMA_VERSION,
        "target": copy.deepcopy(TARGET),
        "component_versions": {
            role: version for role, (_, version) in COMPONENT_MARKERS.items()
        },
        "status": "validated",
    }
    _validate_public_keys(projection)
    return projection


def validate_history_execution_binding(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a canonical immutable execution interface from a private binding."""
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or binding.get("target") != TARGET
    ):
        raise _execution_interface_error()
    root = _binding_private_root(binding.get("private_root"))
    component_paths = _binding_component_paths(binding.get("component_paths"), root)
    interface = binding.get("execution_interface")
    if not isinstance(interface, Mapping):
        raise _execution_interface_error()
    canonical_interface = _validate_execution_interface(
        _json_compatible(interface), root, component_paths
    )
    _private_gpu_indices(canonical_interface)
    return _freeze_mapping(canonical_interface)


def validate_binding_recipe_consistency(
    binding: Mapping[str, Any],
    expected_recipe: Mapping[str, Any] | None,
) -> None:
    """Reject a provisioned binding whose recipe differs from pre-provision input."""
    if expected_recipe is None:
        return
    source_contract = binding.get("source_contract_template")
    actual_recipe = (
        source_contract.get("dynamic_materialization_recipe")
        if isinstance(source_contract, Mapping)
        else None
    )
    try:
        actual = json.dumps(
            actual_recipe,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected = json.dumps(
            expected_recipe,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise P6HistoryBindingError(
            "history_recipe_derivation_invalid",
            "binding recipe is not canonical JSON",
        ) from error
    if actual != expected:
        raise P6HistoryBindingError(
            "history_recipe_derivation_invalid",
            "binding recipe differs from the pre-provision recipe",
        )


def write_private_binding_pair(
    binding: Mapping[str, Any],
    local_config: Mapping[str, Any],
    binding_path: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    *,
    ignore_predicate: IgnorePredicate | None = None,
) -> None:
    """Prevalidate, stage, and replace a private JSON pair with failure compensation."""
    serialized = (
        _serialize_json(binding, "binding"),
        _serialize_json(local_config, "local config"),
    )
    resolved_destinations = prevalidate_private_binding_pair_destinations(
        binding_path,
        config_path,
        repo_root,
        ignore_predicate=ignore_predicate,
    )

    original_payloads = tuple(
        _capture_destination_payload(destination) for destination in resolved_destinations
    )

    temporary_paths: list[Path] = []
    replaced_destinations: list[tuple[Path, bytes, bytes | None]] = []
    try:
        for destination, payload in zip(resolved_destinations, serialized, strict=True):
            temporary_paths.append(_write_temporary_sibling(destination, payload))
        for temporary, destination, payload, previous_payload in zip(
            temporary_paths,
            resolved_destinations,
            serialized,
            original_payloads,
            strict=True,
        ):
            try:
                os.replace(temporary, destination)
            except OSError as error:
                _restore_replaced_destinations(replaced_destinations)
                raise P6HistoryBindingError(
                    "persistence", f"private pair replacement failed: {error}"
                ) from error
            replaced_destinations.append((destination, payload, previous_payload))
        for parent in {path.parent for path in resolved_destinations}:
            _fsync_directory(parent)
    except P6HistoryBindingError:
        raise
    except OSError as error:
        raise P6HistoryBindingError(
            "persistence", f"private pair replacement failed: {error}"
        ) from error
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def prevalidate_private_binding_pair_destinations(
    binding_path: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    *,
    ignore_predicate: IgnorePredicate | None = None,
) -> tuple[Path, Path]:
    """Return two safe pair destinations without serializing or writing payloads."""
    repository = _resolve_directory(repo_root, "repository root")
    destinations = (Path(binding_path), Path(config_path))
    if destinations[0].absolute() == destinations[1].absolute():
        raise P6HistoryBindingError(
            "unsafe_destination", "binding and config destinations must differ"
        )
    resolved_destinations = tuple(
        _prevalidate_destination(path, repository, ignore_predicate)
        for path in destinations
    )
    return resolved_destinations


def _resolve_history_root(history_root: str | Path) -> Path:
    try:
        root = Path(history_root).resolve(strict=True)
    except OSError as error:
        raise P6HistoryBindingError("history_root", "history root does not exist") from error
    if not root.is_dir():
        raise P6HistoryBindingError("history_root", "history root is not a directory")
    return root


def _binding_private_root(raw: object) -> Path:
    if not isinstance(raw, str):
        raise _execution_interface_error()
    try:
        root = Path(raw).resolve(strict=True)
    except OSError as error:
        raise _execution_interface_error() from error
    if not root.is_dir():
        raise _execution_interface_error()
    return root


def _binding_component_paths(raw: object, root: Path) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != set(COMPONENT_MARKERS):
        raise _execution_interface_error()
    component_paths: dict[str, str] = {}
    for role, (marker, _) in COMPONENT_MARKERS.items():
        value = raw.get(role)
        if not isinstance(value, str) or Path(value).name != marker:
            raise _execution_interface_error()
        component_paths[role] = str(_resolve_interface_executable(value, root))
    return component_paths


def _binding_local_input_paths(raw: object, root: Path) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != set(LOCAL_INPUT_NAMES):
        raise P6HistoryBindingError("local_inputs", "local input mapping is invalid")
    local_input_paths: dict[str, str] = {}
    for name in LOCAL_INPUT_NAMES:
        value = raw.get(name)
        if not isinstance(value, str) or Path(value).name != f"{name}.json":
            raise P6HistoryBindingError("local_inputs", "local input mapping is invalid")
        local_input_paths[name] = str(_resolve_beneath_root(Path(value), root))
    return local_input_paths


def _discover_one_named_path(
    root: Path,
    name: str,
    *,
    category: str = "component_discovery",
) -> Path:
    candidates = list(root.rglob(name))
    if len(candidates) != 1:
        raise P6HistoryBindingError(
            category,
            f"expected exactly one {name}, found {len(candidates)}",
        )
    return _resolve_beneath_root(candidates[0], root)


def _resolve_beneath_root(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6HistoryBindingError("path_escape", f"discovered path is unavailable: {path.name}") from error
    if not _is_relative_to(resolved, root):
        raise P6HistoryBindingError(
            "path_escape", f"discovered path escapes history root: {path.name}"
        )
    if not resolved.is_file():
        raise P6HistoryBindingError("path_escape", f"discovered path is not a file: {path.name}")
    return resolved


def _discover_source_contract(root: Path) -> tuple[Path, dict[str, Any]]:
    registries: list[tuple[Path, Mapping[str, Any]]] = []
    for raw_path in root.rglob("*.json"):
        path = _resolve_beneath_root(raw_path, root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        schema = payload.get("schema_version")
        if isinstance(schema, str) and schema.startswith("stage5_candidate_source_registry_"):
            registries.append((path, payload))
    if len(registries) != 1:
        raise P6HistoryBindingError(
            "source_registry",
            f"expected exactly one candidate source registry, found {len(registries)}",
        )
    registry_path, registry = registries[0]
    if registry.get("schema_version") != SOURCE_REGISTRY_SCHEMA_VERSION:
        raise P6HistoryBindingError("source_registry", "incompatible source registry version")
    groups = registry.get("groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise P6HistoryBindingError("source_registry", "candidate source groups are absent")
    valid_contracts: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        if group.get("materialization_kind") != "local_pyramid_tvm":
            continue
        try:
            contract = validate_source_contract(group)
        except ValueError:
            continue
        if contract.get("model") == "pyramid" and contract.get("source_status") == "ready":
            valid_contracts.append(contract)
    if len(valid_contracts) != 1:
        raise P6HistoryBindingError(
            "source_registry",
            f"expected exactly one complete Pyramid/TVM template, found {len(valid_contracts)}",
        )
    return registry_path, copy.deepcopy(valid_contracts[0])


def _discover_execution_interface(
    root: Path,
    component_paths: Mapping[str, str],
) -> Mapping[str, Any]:
    manifests: list[tuple[Path, Mapping[str, Any]]] = []
    for raw_path in root.rglob("*.json"):
        path = _resolve_beneath_root(raw_path, root)
        try:
            text = path.read_text(encoding="utf-8")
            preliminary = json.loads(text)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(preliminary, Mapping):
            continue
        if preliminary.get("schema_version") != EXECUTION_INTERFACE_SCHEMA_VERSION:
            continue
        try:
            payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise _execution_interface_error() from error
        if not isinstance(payload, Mapping):
            raise _execution_interface_error()
        manifests.append((path, payload))
    if len(manifests) != 1:
        raise P6HistoryBindingError(
            "execution_interface", "history runner interface selection failed"
        )
    _, manifest = manifests[0]
    canonical = _validate_execution_interface(manifest, root, component_paths)
    return _freeze_mapping(canonical)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate key")
        payload[key] = value
    return payload


def _validate_execution_interface(
    manifest: Mapping[str, Any],
    root: Path,
    component_paths: Mapping[str, str],
) -> dict[str, Any]:
    _require_exact_keys(manifest, INTERFACE_TOP_LEVEL_KEYS)
    if manifest.get("schema_version") != EXECUTION_INTERFACE_SCHEMA_VERSION:
        raise _execution_interface_error()
    controller = _validate_component_argv(
        manifest.get("controller"), root, component_paths["controller"]
    )
    chain = _validate_execution_chain(
        manifest.get("execution_chain"), root, component_paths
    )
    environment = _validate_environment(manifest.get("environment"), root)
    output_layout = _validate_output_layout(manifest.get("output_layout"), root)
    actual_feedback = _validate_actual_feedback(manifest.get("actual_feedback"), root)
    return {
        "schema_version": EXECUTION_INTERFACE_SCHEMA_VERSION,
        "controller": {"argv": controller},
        "execution_chain": chain,
        "environment": environment,
        "output_layout": output_layout,
        "actual_feedback": actual_feedback,
    }


def _validate_component_argv(
    raw: object,
    root: Path,
    expected_path: str,
) -> list[str]:
    if not isinstance(raw, Mapping):
        raise _execution_interface_error()
    _require_exact_keys(raw, {"argv"})
    argv = _validate_argv(raw.get("argv"), root)
    if argv != [expected_path]:
        raise _execution_interface_error()
    return argv


def _validate_execution_chain(
    raw: object,
    root: Path,
    component_paths: Mapping[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != len(EXECUTION_STAGES):
        raise _execution_interface_error()
    chain: list[dict[str, Any]] = []
    expected_component_paths = {
        "source_materialization": component_paths["source_materializer"],
        "performance": component_paths["performance_plan"],
        "finalization": component_paths["finalizer"],
    }
    for expected_stage, entry in zip(EXECUTION_STAGES, raw, strict=True):
        if not isinstance(entry, Mapping):
            raise _execution_interface_error()
        _require_exact_keys(entry, {"stage", "argv", "required_placeholders"})
        if entry.get("stage") != expected_stage:
            raise _execution_interface_error()
        argv = _validate_argv(entry.get("argv"), root)
        required_placeholders = REQUIRED_STAGE_PLACEHOLDERS[expected_stage]
        if entry.get("required_placeholders") != list(required_placeholders):
            raise _execution_interface_error()
        actual_placeholders = tuple(
            token for token in argv if token in EXECUTION_PLACEHOLDERS
        )
        if actual_placeholders != required_placeholders:
            raise _execution_interface_error()
        component_path = expected_component_paths.get(expected_stage)
        if component_path is not None and argv[0] != component_path:
            raise _execution_interface_error()
        chain.append(
            {
                "stage": expected_stage,
                "argv": argv,
                "required_placeholders": list(required_placeholders),
            }
        )
    return chain


def _validate_argv(raw: object, root: Path) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise _execution_interface_error()
    argv: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value or _contains_shell_token(value):
            raise _execution_interface_error()
        if value in EXECUTION_PLACEHOLDERS:
            if index == 0:
                raise _execution_interface_error()
            argv.append(value)
            continue
        if "{" in value or "}" in value:
            raise _execution_interface_error()
        if index == 0:
            argv.append(str(_resolve_interface_executable(value, root)))
            continue
        if Path(value).is_absolute():
            argv.append(str(_resolve_interface_executable(value, root)))
            continue
        if "/" in value or "\\" in value:
            raise _execution_interface_error()
        argv.append(value)
    return argv


def _resolve_interface_executable(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        raise _execution_interface_error()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise _execution_interface_error() from error
    if (
        not _is_relative_to(resolved, root)
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
    ):
        raise _execution_interface_error()
    return resolved


def _validate_environment(raw: object, root: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _execution_interface_error()
    _require_exact_keys(raw, {"values", "activation_argv"})
    values = raw.get("values")
    if not isinstance(values, Mapping) or set(values) != set(WRAPPER_ENVIRONMENT_SPEC):
        raise _execution_interface_error()
    normalized_values: dict[str, dict[str, str]] = {}
    for key, expected in WRAPPER_ENVIRONMENT_SPEC.items():
        value = values.get(key)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"kind", "value"}
            or value.get("kind") != expected[0]
            or not isinstance(value.get("value"), str)
            or not value["value"]
        ):
            raise _execution_interface_error()
        normalized_values[key] = _validate_environment_value(
            key, expected, value["value"], root
        )
    activation_argv = _validate_argv(raw.get("activation_argv"), root)
    if any(token in EXECUTION_PLACEHOLDERS for token in activation_argv):
        raise _execution_interface_error()
    return {
        "values": normalized_values,
        "activation_argv": activation_argv,
    }


def _validate_environment_value(
    key: str,
    expected: tuple[str, str | None],
    value: str,
    root: Path,
) -> dict[str, str]:
    kind, expected_value = expected
    if _contains_shell_token(value):
        raise _execution_interface_error()
    if kind == "private_path":
        try:
            resolved = Path(value).resolve(strict=True)
        except OSError as error:
            raise _execution_interface_error() from error
        if not Path(value).is_absolute() or not _is_relative_to(resolved, root):
            raise _execution_interface_error()
        return {"kind": kind, "value": str(resolved)}
    if (
        (expected_value is not None and value != expected_value)
        or "/" in value
        or "\\" in value
    ):
        raise _execution_interface_error()
    return {"kind": kind, "value": value}


def _validate_output_layout(raw: object, root: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _execution_interface_error()
    _require_exact_keys(raw, {"round_root_template", "task_state"})
    round_root = _validate_private_template(raw.get("round_root_template"), root)
    task_state = raw.get("task_state")
    if not isinstance(task_state, Mapping):
        raise _execution_interface_error()
    _require_exact_keys(
        task_state,
        {
            "path_template",
            "format",
            "rows_key",
            "row_id_key",
            "row_hash_key",
            "source_evidence_key",
            "stage_key",
            "status_key",
            "row_count",
            "allowed_terminal_statuses",
            "stage_order",
        },
    )
    if (
        task_state.get("format") != "json"
        or task_state.get("stage_key") != "stage"
        or task_state.get("row_count") != 4
        or task_state.get("allowed_terminal_statuses")
        != list(ALLOWED_TERMINAL_STATUSES)
    ):
        raise _execution_interface_error()
    if not _has_expected_row_field_names(task_state):
        raise _execution_interface_error()
    if task_state.get("stage_order") != list(EXECUTION_STAGES):
        raise _execution_interface_error()
    return {
        "round_root_template": round_root,
        "task_state": {
            "path_template": _validate_private_template(
                task_state.get("path_template"), root
            ),
            "format": "json",
            **EXPECTED_ROW_FIELD_NAMES,
            "stage_key": "stage",
            "row_count": 4,
            "allowed_terminal_statuses": list(ALLOWED_TERMINAL_STATUSES),
            "stage_order": list(EXECUTION_STAGES),
        },
    }


def _validate_actual_feedback(raw: object, root: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _execution_interface_error()
    _require_exact_keys(raw, {"result", "receipt", "finalization_barrier"})
    return {
        "result": _validate_actual_result_schema(raw.get("result"), root),
        "receipt": _validate_feedback_validation_location(raw.get("receipt"), root),
        "finalization_barrier": _validate_feedback_validation_location(
            raw.get("finalization_barrier"), root
        ),
    }


def _validate_actual_result_schema(raw: object, root: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _execution_interface_error()
    _require_exact_keys(
        raw,
        {
            "path_template",
            "format",
            "rows_key",
            "row_count",
            "row_id_key",
            "row_hash_key",
            "source_evidence_key",
            "status_key",
            "allowed_terminal_statuses",
            "metric_keys",
        },
    )
    if (
        raw.get("format") != "json"
        or raw.get("row_count") != 4
        or raw.get("allowed_terminal_statuses") != list(ALLOWED_TERMINAL_STATUSES)
        or raw.get("metric_keys") != list(METRIC_KEYS)
        or not _has_expected_row_field_names(raw)
    ):
        raise _execution_interface_error()
    return {
        "path_template": _validate_private_template(raw.get("path_template"), root),
        "format": "json",
        "row_count": 4,
        **EXPECTED_ROW_FIELD_NAMES,
        "allowed_terminal_statuses": list(ALLOWED_TERMINAL_STATUSES),
        "metric_keys": list(METRIC_KEYS),
    }


def _validate_feedback_validation_location(raw: object, root: Path) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise _execution_interface_error()
    _require_exact_keys(
        raw,
        {
            "path_template",
            "format",
            "request_sha256_key",
            "row_hashes_key",
            "source_evidence_key",
        },
    )
    if (
        raw.get("format") != "json"
        or raw.get("request_sha256_key") != "measurement_request_sha256"
        or raw.get("row_hashes_key") != "row_sha256"
        or raw.get("source_evidence_key") != "source_evidence_sha256"
    ):
        raise _execution_interface_error()
    return {
        "path_template": _validate_private_template(raw.get("path_template"), root),
        "format": "json",
        "request_sha256_key": "measurement_request_sha256",
        "row_hashes_key": "row_sha256",
        "source_evidence_key": "source_evidence_sha256",
    }


def _has_expected_row_field_names(payload: Mapping[str, Any]) -> bool:
    return all(payload.get(key) == value for key, value in EXPECTED_ROW_FIELD_NAMES.items())


def _validate_private_template(raw: object, root: Path) -> str:
    if not isinstance(raw, str) or not raw or _contains_shell_token(raw):
        raise _execution_interface_error()
    remainder = raw.replace("{round_id}", "")
    if raw.count("{round_id}") != 1 or "{" in remainder or "}" in remainder:
        raise _execution_interface_error()
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise _execution_interface_error()
    resolved = (root / raw.replace("{round_id}", "round")).resolve(strict=False)
    if not _is_relative_to(resolved, root):
        raise _execution_interface_error()
    return raw


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str] | frozenset[str]) -> None:
    if set(payload) != set(expected):
        raise _execution_interface_error()


def _contains_shell_token(value: str) -> bool:
    return any(token in value for token in SHELL_TOKENS)


def _execution_interface_error() -> P6HistoryBindingError:
    return P6HistoryBindingError("execution_interface", "invalid private execution interface")


def _freeze_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            frozen[key] = _freeze_mapping(value)
        elif isinstance(value, list):
            frozen[key] = tuple(
                _freeze_mapping(item) if isinstance(item, Mapping) else item
                for item in value
            )
        else:
            frozen[key] = value
    return MappingProxyType(frozen)


def _private_gpu_indices(interface: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        value = interface["environment"]["values"]["CUDA_VISIBLE_DEVICES"][
            "value"
        ]
        parts = tuple(value.split(","))
        indices = tuple(int(part) for part in parts)
    except (KeyError, TypeError, ValueError):
        raise _execution_interface_error() from None
    if (
        len(indices) != 3
        or tuple(sorted(indices)) != indices
        or len(set(indices)) != 3
        or any(index < 0 for index in indices)
    ):
        raise _execution_interface_error()
    return (indices[0], indices[1], indices[2])


def _probe_snapshot(
    gpu_probe: GpuProbe, indices: tuple[int, int, int]
) -> tuple[GpuRecord, ...]:
    try:
        snapshot = gpu_probe.snapshot(indices)
    except Exception as error:
        raise P6HistoryBindingError("gpu_admission", "GPU probe failed closed") from error
    if not isinstance(snapshot, tuple):
        raise P6HistoryBindingError("gpu_admission", "GPU snapshot must be immutable")
    return snapshot


def _validate_gpu_snapshot(
    snapshot: tuple[GpuRecord, ...], indices: tuple[int, int, int]
) -> dict[str, str]:
    if len(snapshot) != len(indices) or any(
        not isinstance(record, GpuRecord) for record in snapshot
    ):
        raise P6HistoryBindingError("gpu_admission", "GPU snapshot is incomplete")
    by_index = {record.index: record for record in snapshot}
    if set(by_index) != set(indices) or len(by_index) != len(snapshot):
        raise P6HistoryBindingError("gpu_admission", "GPU index set does not match policy")
    ordered = [by_index[index] for index in indices]
    if any(not isinstance(record.uuid, str) for record in ordered):
        raise P6HistoryBindingError("gpu_admission", "GPU UUIDs must be strings")
    uuids = [record.uuid.strip() for record in ordered]
    if any(not uuid for uuid in uuids) or len(set(uuids)) != len(uuids):
        raise P6HistoryBindingError("gpu_admission", "GPU UUIDs must be non-empty and unique")
    if any(
        _normalize_model(record.model_name) not in ALLOWED_NORMALIZED_H800_MODELS
        for record in ordered
    ):
        raise P6HistoryBindingError("gpu_admission", "all admitted GPUs must be H800 models")
    if any(
        isinstance(record.occupancy, bool)
        or not isinstance(record.occupancy, (int, float))
        or not math.isfinite(record.occupancy)
        or record.occupancy < 0.0
        or record.occupancy > MAX_GPU_OCCUPANCY
        for record in ordered
    ):
        raise P6HistoryBindingError("gpu_admission", "GPU occupancy is incompatible")
    return {str(record.index): record.uuid.strip() for record in ordered}


def _normalize_model(model_name: str) -> str:
    if not isinstance(model_name, str):
        return ""
    return "".join(character for character in model_name.upper() if character.isalnum())


def _validate_public_keys(payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        lowered = str(key).lower()
        if any(token in lowered for token in FORBIDDEN_PUBLIC_KEY_TOKENS):
            raise P6HistoryBindingError("public_projection", "private semantic key denied")
        if isinstance(value, Mapping):
            _validate_public_keys(value)


def _serialize_json(payload: Mapping[str, Any], label: str) -> bytes:
    if not isinstance(payload, Mapping):
        raise P6HistoryBindingError("persistence", f"{label} must be a mapping")
    try:
        text = json.dumps(
            _json_compatible(payload),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise P6HistoryBindingError("persistence", f"{label} is not valid JSON") from error
    return f"{text}\n".encode("utf-8")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(child) for child in value]
    if isinstance(value, list):
        return [_json_compatible(child) for child in value]
    return value


def _resolve_directory(path: str | Path, label: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as error:
        raise P6HistoryBindingError("unsafe_destination", f"{label} does not exist") from error
    if not resolved.is_dir():
        raise P6HistoryBindingError("unsafe_destination", f"{label} is not a directory")
    return resolved


def _prevalidate_destination(
    raw_path: Path,
    repository: Path,
    ignore_predicate: IgnorePredicate | None,
) -> Path:
    path = raw_path.absolute()
    if path.is_symlink():
        raise P6HistoryBindingError("unsafe_destination", "destination cannot be a symlink")
    if path.exists() and not path.is_file():
        raise P6HistoryBindingError("unsafe_destination", "destination must be a regular file")
    parent = _resolve_directory(path.parent, "destination parent")
    if path.parent.is_symlink():
        raise P6HistoryBindingError("unsafe_destination", "destination parent cannot be a symlink")
    destination = parent / path.name
    if _is_relative_to(destination, repository):
        predicate = ignore_predicate or (
            lambda candidate: _git_check_ignored(repository, candidate)
        )
        try:
            ignored = predicate(destination)
        except Exception as error:
            raise P6HistoryBindingError(
                "unsafe_destination", "ignore predicate failed closed"
            ) from error
        if ignored is not True:
            raise P6HistoryBindingError(
                "unsafe_destination", "repository destination is not git-ignored"
            )
    return destination


def _capture_destination_payload(destination: Path) -> bytes | None:
    if destination.is_symlink():
        raise P6HistoryBindingError("unsafe_destination", "destination cannot be a symlink")
    try:
        return destination.read_bytes()
    except FileNotFoundError:
        return None


def _restore_replaced_destinations(
    replacements: Sequence[tuple[Path, bytes, bytes | None]],
) -> None:
    for destination, published_payload, original_payload in reversed(replacements):
        try:
            if not _destination_has_payload(destination, published_payload):
                continue
            if original_payload is None:
                destination.unlink()
            else:
                temporary = _write_temporary_sibling(destination, original_payload)
                try:
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
        except OSError:
            continue


def _destination_has_payload(destination: Path, expected_payload: bytes) -> bool:
    if destination.is_symlink():
        return False
    try:
        return destination.read_bytes() == expected_payload
    except OSError:
        return False


def _git_check_ignored(repository: Path, destination: Path) -> bool:
    relative = destination.relative_to(repository)
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(relative)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _write_temporary_sibling(destination: Path, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
