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

from framework.stage5.production_search_v1 import validate_source_contract


BINDING_SCHEMA_VERSION = "p6_history_binding_v1"
PUBLIC_SCHEMA_VERSION = "p6_history_binding_public_v1"
SOURCE_REGISTRY_SCHEMA_VERSION = "stage5_candidate_source_registry_v1"
EXPECTED_GPU_INDICES = (5, 6, 7)
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
    registry_path, source_contract = _discover_source_contract(root)
    local_input_paths = {
        name: str(_discover_one_named_path(root, f"{name}.json", category="local_inputs"))
        for name in LOCAL_INPUT_NAMES
    }
    first_snapshot = _probe_snapshot(gpu_probe)
    second_snapshot = _probe_snapshot(gpu_probe)
    first_uuid_map = _validate_gpu_snapshot(first_snapshot)
    second_uuid_map = _validate_gpu_snapshot(second_snapshot)
    if first_uuid_map != second_uuid_map:
        raise P6HistoryBindingError("gpu_drift", "GPU UUIDs changed between snapshots")

    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "target": copy.deepcopy(TARGET),
        "private_root": str(root),
        "component_paths": component_paths,
        "component_versions": {
            role: version for role, (_, version) in COMPONENT_MARKERS.items()
        },
        "source_registry_path": str(registry_path),
        "source_contract_template": copy.deepcopy(source_contract),
        "local_input_paths": local_input_paths,
        "gpu_policy": {
            "indices": list(EXPECTED_GPU_INDICES),
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


def write_private_binding_pair(
    binding: Mapping[str, Any],
    local_config: Mapping[str, Any],
    binding_path: str | Path,
    config_path: str | Path,
    repo_root: str | Path,
    *,
    ignore_predicate: IgnorePredicate | None = None,
) -> None:
    """Prevalidate, stage, and independently replace a private JSON pair.

    The two sibling temporary files are complete before replacement begins. Filesystems
    provide no portable transaction spanning two ``os.replace`` calls: if the second
    replace fails after the first succeeds, the first destination remains replaced.
    """
    serialized = (
        _serialize_json(binding, "binding"),
        _serialize_json(local_config, "local config"),
    )
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

    temporary_paths: list[Path] = []
    try:
        for destination, payload in zip(resolved_destinations, serialized, strict=True):
            temporary_paths.append(_write_temporary_sibling(destination, payload))
        for temporary, destination in zip(
            temporary_paths, resolved_destinations, strict=True
        ):
            os.replace(temporary, destination)
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


def _resolve_history_root(history_root: str | Path) -> Path:
    try:
        root = Path(history_root).resolve(strict=True)
    except OSError as error:
        raise P6HistoryBindingError("history_root", "history root does not exist") from error
    if not root.is_dir():
        raise P6HistoryBindingError("history_root", "history root is not a directory")
    return root


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


def _probe_snapshot(gpu_probe: GpuProbe) -> tuple[GpuRecord, ...]:
    try:
        snapshot = gpu_probe.snapshot(EXPECTED_GPU_INDICES)
    except Exception as error:
        raise P6HistoryBindingError("gpu_admission", "GPU probe failed closed") from error
    if not isinstance(snapshot, tuple):
        raise P6HistoryBindingError("gpu_admission", "GPU snapshot must be immutable")
    return snapshot


def _validate_gpu_snapshot(snapshot: tuple[GpuRecord, ...]) -> dict[str, str]:
    if len(snapshot) != len(EXPECTED_GPU_INDICES) or any(
        not isinstance(record, GpuRecord) for record in snapshot
    ):
        raise P6HistoryBindingError("gpu_admission", "GPU snapshot is incomplete")
    by_index = {record.index: record for record in snapshot}
    if set(by_index) != set(EXPECTED_GPU_INDICES) or len(by_index) != len(snapshot):
        raise P6HistoryBindingError(
            "gpu_admission", "GPU index set must be exactly [5, 6, 7]"
        )
    ordered = [by_index[index] for index in EXPECTED_GPU_INDICES]
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
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise P6HistoryBindingError("persistence", f"{label} is not valid JSON") from error
    return f"{text}\n".encode("utf-8")


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
