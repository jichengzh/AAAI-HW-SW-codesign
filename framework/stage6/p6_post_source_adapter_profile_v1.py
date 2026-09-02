"""Build and load the ignored private P6 post-source adapter profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Literal

import yaml

from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    default_hardware_execution_profile,
    load_hardware_execution_profile,
    validate_profile_backend,
)
from framework.stage6.p6_history_execution_closure_v1 import (
    P6ValidatedExecutionClosure,
)
from framework.stage6.p6_post_source_leaf_binding_v1 import (
    POST_SOURCE_LEAF_NAMES,
    ValidatedPostSourceLeafBinding,
)
from framework.stage6.p6_python_runtime_v1 import (
    P6PythonRuntimeError,
    validate_adapter_python,
)
from framework.stage6.p6_tvm_runtime_authority_v1 import (
    P6TvmRuntimeAuthorityError,
    TVM_SUPPORT_CLOSURE_ID,
    canonical_tvm_support_tree_sha256,
)


PROFILE_SCHEMA_VERSION = "p6_post_source_adapter_profile_v1"
PROFILE_SCHEMA_VERSION_V2 = "p6_post_source_adapter_profile_v2"
PROFILE_SCHEMA_VERSION_V3 = "p6_post_source_adapter_profile_v3"
PROFILE_SCHEMA_VERSION_V4 = "p6_post_source_adapter_profile_v4"
RUNNER_INTERFACE_SCHEMA_VERSION = "p6_history_runner_interface_v1"
POST_SOURCE_ADAPTER_STAGES: tuple[str, ...] = (
    "quantization",
    "performance",
    "ap",
    "finalization",
)
TARGET_PROFILE = {"model": "pyramid", "hardware": "h800", "backend": "tvm_auto"}
PROFILE_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "runner_interface_schema_version",
        "project_python",
        "adapters",
        "leaves",
    }
)
PROFILE_V2_KEYS = PROFILE_KEYS | {"adapter_python"}
PROFILE_V3_KEYS = PROFILE_V2_KEYS | {"adapter_dependency_root_relative_path"}
PROFILE_V4_KEYS = PROFILE_V3_KEYS | {
    "hardware_profile",
    "tvm_support_root_relative_path",
    "tvm_support_root_sha256",
}
IMPLEMENTATION_KEYS = frozenset(
    {"implementation_relative_path", "implementation_cwd_relative_path"}
)
LEAF_KEYS = IMPLEMENTATION_KEYS | {"sha256"}
MAX_PROFILE_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PostSourceLeaf:
    name: str
    implementation: Path
    implementation_cwd: Path
    sha256: str


@dataclass(frozen=True)
class PostSourceAdapter:
    stage: str
    implementation: Path
    implementation_cwd: Path


@dataclass(frozen=True)
class ValidatedPostSourceAdapterProfile:
    schema_version: Literal[
        "p6_post_source_adapter_profile_v1",
        "p6_post_source_adapter_profile_v2",
        "p6_post_source_adapter_profile_v3",
        "p6_post_source_adapter_profile_v4",
    ]
    private_root: Path
    project_python: Path
    adapters: tuple[PostSourceAdapter, ...]
    leaves: tuple[PostSourceLeaf, ...]
    adapter_python: Path | None = None
    adapter_dependency_root: Path | None = None
    hardware_profile: HardwareExecutionProfile = field(
        default_factory=default_hardware_execution_profile
    )
    tvm_support_root: Path | None = None
    tvm_support_root_sha256: str | None = None


class P6PostSourceAdapterProfileError(ValueError):
    """Stable path-free post-source adapter profile failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    keys: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            _invalid()
        keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def build_post_source_adapter_profile(
    *,
    private_root: Path,
    project_python: Path,
    copied_role_paths: Mapping[str, Path],
    execution_closure: P6ValidatedExecutionClosure,
    leaf_binding: ValidatedPostSourceLeafBinding,
    adapter_python: Path | None = None,
    adapter_dependency_root_relative_path: Path | None = None,
    hardware_profile: HardwareExecutionProfile | None = None,
) -> ValidatedPostSourceAdapterProfile:
    """Build the immutable normalized profile from copied closure paths."""
    root = _private_root(private_root)
    python = _project_python(project_python)
    closure_roots = _normalized_closure_roots(root, execution_closure)
    adapters = tuple(
        _adapter(stage, copied_role_paths, closure_roots, execution_closure)
        for stage in POST_SOURCE_ADAPTER_STAGES
    )
    leaves = tuple(_leaf(leaf, closure_roots) for leaf in leaf_binding.leaves)
    if len({leaf.implementation for leaf in leaves}) != len(leaves):
        _invalid()
    validated_adapter_python = _adapter_python(adapter_python)
    dependency_root = _adapter_dependency_root(
        root, adapter_dependency_root_relative_path
    )
    if dependency_root is not None and validated_adapter_python is None:
        _invalid()
    selected_hardware_profile = (
        default_hardware_execution_profile()
        if hardware_profile is None
        else _hardware_profile(hardware_profile)
    )
    if hardware_profile is not None and dependency_root is None:
        _invalid()
    support_root, support_root_sha256 = _tvm_support_authority(
        root,
        closure_roots,
        execution_closure,
        required=hardware_profile is not None,
    )
    return ValidatedPostSourceAdapterProfile(
        schema_version=(
            PROFILE_SCHEMA_VERSION_V4
            if hardware_profile is not None
            else PROFILE_SCHEMA_VERSION_V3
            if dependency_root is not None
            else PROFILE_SCHEMA_VERSION_V2
            if validated_adapter_python is not None
            else PROFILE_SCHEMA_VERSION
        ),
        private_root=root,
        project_python=python,
        adapters=adapters,
        leaves=leaves,
        adapter_python=validated_adapter_python,
        adapter_dependency_root=dependency_root,
        hardware_profile=selected_hardware_profile,
        tvm_support_root=support_root,
        tvm_support_root_sha256=support_root_sha256,
    )


def post_source_adapter_profile_to_mapping(
    profile: ValidatedPostSourceAdapterProfile,
) -> dict[str, Any]:
    """Serialize a validated profile without private absolute implementation paths."""
    if not isinstance(profile, ValidatedPostSourceAdapterProfile):
        _invalid()
    hardware_profile = _hardware_profile(profile.hardware_profile)
    if (
        profile.schema_version != PROFILE_SCHEMA_VERSION_V4
        and hardware_profile is not default_hardware_execution_profile()
    ):
        _invalid()
    payload = {
        "schema_version": profile.schema_version,
        "target": _target_profile(hardware_profile),
        "runner_interface_schema_version": RUNNER_INTERFACE_SCHEMA_VERSION,
        "project_python": str(profile.project_python),
        "adapters": {
            adapter.stage: _entry_to_mapping(adapter, profile.private_root)
            for adapter in profile.adapters
        },
        "leaves": {
            leaf.name: {
                **_entry_to_mapping(leaf, profile.private_root),
                "sha256": leaf.sha256,
            }
            for leaf in profile.leaves
        },
    }
    if profile.schema_version == PROFILE_SCHEMA_VERSION:
        if (
            profile.adapter_python is not None
            or profile.adapter_dependency_root is not None
        ):
            _invalid()
        return payload
    if profile.schema_version not in {
        PROFILE_SCHEMA_VERSION_V2,
        PROFILE_SCHEMA_VERSION_V3,
        PROFILE_SCHEMA_VERSION_V4,
    }:
        _invalid()
    adapter_python = _adapter_python(profile.adapter_python)
    if adapter_python is None:
        _invalid()
    runtime_payload = {**payload, "adapter_python": str(adapter_python)}
    if profile.schema_version == PROFILE_SCHEMA_VERSION_V2:
        if profile.adapter_dependency_root is not None:
            _invalid()
        return runtime_payload
    dependency_payload = _v3_profile_mapping(profile, runtime_payload)
    if profile.schema_version == PROFILE_SCHEMA_VERSION_V3:
        return dependency_payload
    support_root, support_root_sha256 = _profile_tvm_support_authority(profile)
    return {
        **dependency_payload,
        "hardware_profile": hardware_profile.profile_id,
        "tvm_support_root_relative_path": support_root.relative_to(
            profile.private_root
        ).as_posix(),
        "tvm_support_root_sha256": support_root_sha256,
    }


def _v3_profile_mapping(
    profile: ValidatedPostSourceAdapterProfile,
    runtime_payload: Mapping[str, Any],
) -> dict[str, Any]:
    dependency_root = _profile_dependency_root(profile)
    return {
        **runtime_payload,
        "adapter_dependency_root_relative_path": dependency_root.relative_to(
            profile.private_root
        ).as_posix(),
    }


def load_post_source_adapter_profile(
    path: Path,
    *,
    private_root: Path,
) -> ValidatedPostSourceAdapterProfile:
    """Load one ignored normalized post-source adapter profile."""
    root = _private_root(private_root)
    payload = _load_profile_payload(path, root)
    schema_version = payload.get("schema_version")
    expected_keys = (
        PROFILE_KEYS
        if schema_version == PROFILE_SCHEMA_VERSION
        else PROFILE_V2_KEYS
        if schema_version == PROFILE_SCHEMA_VERSION_V2
        else PROFILE_V3_KEYS
        if schema_version == PROFILE_SCHEMA_VERSION_V3
        else PROFILE_V4_KEYS
        if schema_version == PROFILE_SCHEMA_VERSION_V4
        else frozenset()
    )
    hardware_profile = (
        _loaded_hardware_profile(payload.get("hardware_profile"))
        if schema_version == PROFILE_SCHEMA_VERSION_V4
        else default_hardware_execution_profile()
    )
    if (
        set(payload) != expected_keys
        or payload.get("target") != _target_profile(hardware_profile)
        or payload.get("runner_interface_schema_version") != RUNNER_INTERFACE_SCHEMA_VERSION
    ):
        _invalid()
    project_python = _project_python(payload.get("project_python"))
    adapter_python = (
        _adapter_python(payload.get("adapter_python"))
        if schema_version
        in {
            PROFILE_SCHEMA_VERSION_V2,
            PROFILE_SCHEMA_VERSION_V3,
            PROFILE_SCHEMA_VERSION_V4,
        }
        else None
    )
    dependency_root = (
        _adapter_dependency_root(
            root, _relative_path(payload.get("adapter_dependency_root_relative_path"))
        )
        if schema_version in {PROFILE_SCHEMA_VERSION_V3, PROFILE_SCHEMA_VERSION_V4}
        else None
    )
    support_root = (
        _private_path(
            root,
            _relative_path(payload.get("tvm_support_root_relative_path")),
            want_dir=True,
        )
        if schema_version == PROFILE_SCHEMA_VERSION_V4
        else None
    )
    support_root_sha256 = (
        _validated_tvm_support_digest(
            support_root, payload.get("tvm_support_root_sha256")
        )
        if support_root is not None
        else None
    )
    adapters = _load_adapters(payload.get("adapters"), root)
    leaves = _load_leaves(payload.get("leaves"), root)
    return ValidatedPostSourceAdapterProfile(
        schema_version,
        root,
        project_python,
        adapters,
        leaves,
        adapter_python,
        dependency_root,
        hardware_profile,
        support_root,
        support_root_sha256,
    )


def require_post_source_adapter_profile_v2(
    profile: ValidatedPostSourceAdapterProfile,
) -> None:
    """Require the schema-honest public-adapter runtime contract."""
    if (
        not isinstance(profile, ValidatedPostSourceAdapterProfile)
        or profile.schema_version != PROFILE_SCHEMA_VERSION_V2
        or profile.adapter_python is None
    ):
        _invalid()


def require_post_source_adapter_profile_v3(
    profile: ValidatedPostSourceAdapterProfile,
) -> None:
    """Require the public-adapter dependency-overlay runtime contract."""
    if (
        not isinstance(profile, ValidatedPostSourceAdapterProfile)
        or profile.schema_version != PROFILE_SCHEMA_VERSION_V3
        or profile.adapter_python is None
        or profile.adapter_dependency_root is None
    ):
        _invalid()


def require_post_source_adapter_profile_v4(
    profile: ValidatedPostSourceAdapterProfile,
) -> None:
    """Require the hardware-bound dependency-overlay runtime contract."""
    if (
        not isinstance(profile, ValidatedPostSourceAdapterProfile)
        or profile.schema_version != PROFILE_SCHEMA_VERSION_V4
        or profile.adapter_python is None
        or profile.adapter_dependency_root is None
        or profile.tvm_support_root is None
        or profile.tvm_support_root_sha256 is None
    ):
        _invalid()
    _hardware_profile(profile.hardware_profile)
    _profile_tvm_support_authority(profile)


def _loaded_hardware_profile(raw: object) -> HardwareExecutionProfile:
    try:
        return load_hardware_execution_profile(raw)  # type: ignore[arg-type]
    except ValueError:
        _invalid()


def _hardware_profile(raw: object) -> HardwareExecutionProfile:
    if not isinstance(raw, HardwareExecutionProfile):
        _invalid()
    try:
        validate_profile_backend(raw, "tvm_auto")
    except ValueError:
        _invalid()
    return raw


def _target_profile(profile: HardwareExecutionProfile) -> dict[str, str]:
    _hardware_profile(profile)
    return {
        "model": "pyramid",
        "hardware": profile.target_hardware_id,
        "backend": "tvm_auto",
    }


def _adapter(
    stage: str,
    copied_role_paths: Mapping[str, Path],
    closure_roots: Mapping[str, Path],
    execution_closure: P6ValidatedExecutionClosure,
) -> PostSourceAdapter:
    implementation = copied_role_paths.get(stage)
    role = next(item for item in execution_closure.roles if item.role == stage)
    cwd = closure_roots[role.closure_id]
    if not isinstance(implementation, Path) or not _executable_file(implementation):
        _invalid()
    return PostSourceAdapter(stage=stage, implementation=implementation, implementation_cwd=cwd)


def _leaf(raw_leaf: Any, closure_roots: Mapping[str, Path]) -> PostSourceLeaf:
    cwd = closure_roots[raw_leaf.closure_id]
    implementation = cwd / raw_leaf.entrypoint_relative_path
    if not _executable_file(implementation):
        _invalid()
    return PostSourceLeaf(
        name=raw_leaf.name,
        implementation=implementation,
        implementation_cwd=cwd,
        sha256=_file_sha256(implementation),
    )


def _load_adapters(raw: object, root: Path) -> tuple[PostSourceAdapter, ...]:
    if not isinstance(raw, Mapping) or set(raw) != set(POST_SOURCE_ADAPTER_STAGES):
        _invalid()
    return tuple(_load_adapter(stage, raw[stage], root) for stage in POST_SOURCE_ADAPTER_STAGES)


def _load_adapter(stage: str, raw: object, root: Path) -> PostSourceAdapter:
    implementation, cwd, _ = _load_implementation(raw, root, with_sha=False)
    return PostSourceAdapter(stage, implementation, cwd)


def _load_leaves(raw: object, root: Path) -> tuple[PostSourceLeaf, ...]:
    if not isinstance(raw, Mapping) or set(raw) != set(POST_SOURCE_LEAF_NAMES):
        _invalid()
    leaves = tuple(_load_leaf(name, raw[name], root) for name in POST_SOURCE_LEAF_NAMES)
    if len({leaf.implementation for leaf in leaves}) != len(leaves):
        _invalid()
    return leaves


def _load_leaf(name: str, raw: object, root: Path) -> PostSourceLeaf:
    implementation, cwd, sha256 = _load_implementation(raw, root, with_sha=True)
    assert sha256 is not None
    if _file_sha256(implementation) != sha256:
        _invalid()
    return PostSourceLeaf(name, implementation, cwd, sha256)


def _load_implementation(
    raw: object,
    root: Path,
    *,
    with_sha: bool,
) -> tuple[Path, Path, str | None]:
    expected_keys = LEAF_KEYS if with_sha else IMPLEMENTATION_KEYS
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        _invalid()
    implementation = _private_path(
        root, _relative_path(raw.get("implementation_relative_path")), want_dir=False
    )
    cwd = _private_path(
        root, _relative_path(raw.get("implementation_cwd_relative_path")), want_dir=True
    )
    if not _is_relative_to(implementation, cwd):
        _invalid()
    sha256 = raw.get("sha256") if with_sha else None
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
        _invalid()
    return implementation, cwd, sha256


def _load_profile_payload(path: Path, root: Path) -> dict[str, Any]:
    if not isinstance(path, Path) or not path.is_absolute() or _contains_symlink(path):
        _invalid()
    try:
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_file()
            or resolved.stat().st_size > MAX_PROFILE_SIZE
            or not _is_relative_to(resolved, root)
        ):
            _invalid()
        payload = yaml.load(resolved.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except P6PostSourceAdapterProfileError:
        raise
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise P6PostSourceAdapterProfileError() from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        _invalid()
    return dict(payload)


def _entry_to_mapping(entry: PostSourceAdapter | PostSourceLeaf, root: Path) -> dict[str, str]:
    return {
        "implementation_relative_path": entry.implementation.relative_to(root).as_posix(),
        "implementation_cwd_relative_path": entry.implementation_cwd.relative_to(root).as_posix(),
    }


def _normalized_closure_roots(
    root: Path, execution_closure: P6ValidatedExecutionClosure
) -> dict[str, Path]:
    if not isinstance(execution_closure, P6ValidatedExecutionClosure):
        _invalid()
    return {
        item.closure_id: _private_path(root, item.destination_relative_root, want_dir=True)
        for item in execution_closure.roots
    }


def _tvm_support_authority(
    root: Path,
    closure_roots: Mapping[str, Path],
    execution_closure: P6ValidatedExecutionClosure,
    *,
    required: bool,
) -> tuple[Path | None, str | None]:
    if not required:
        return None, None
    declarations = tuple(
        item
        for item in execution_closure.roots
        if item.closure_id == TVM_SUPPORT_CLOSURE_ID
    )
    if len(declarations) != 1 or any(
        role.closure_id == TVM_SUPPORT_CLOSURE_ID
        for role in execution_closure.roles
    ):
        _invalid()
    support_root = closure_roots.get(TVM_SUPPORT_CLOSURE_ID)
    if not isinstance(support_root, Path):
        _invalid()
    declared_digest = declarations[0].sha256
    if _validated_tvm_support_digest(support_root, declared_digest) != declared_digest:
        _invalid()
    try:
        support_root.relative_to(root)
    except ValueError:
        _invalid()
    return support_root, declared_digest


def _profile_tvm_support_authority(
    profile: ValidatedPostSourceAdapterProfile,
) -> tuple[Path, str]:
    support_root = profile.tvm_support_root
    if not isinstance(support_root, Path):
        _invalid()
    try:
        relative = support_root.relative_to(profile.private_root)
    except ValueError:
        _invalid()
    validated_root = _private_path(profile.private_root, relative, want_dir=True)
    digest = _validated_tvm_support_digest(
        validated_root, profile.tvm_support_root_sha256
    )
    return validated_root, digest


def _validated_tvm_support_digest(root: Path, raw_digest: object) -> str:
    if (
        not isinstance(raw_digest, str)
        or len(raw_digest) != 64
        or set(raw_digest) - set("0123456789abcdef")
    ):
        _invalid()
    try:
        observed = canonical_tvm_support_tree_sha256(root)
    except P6TvmRuntimeAuthorityError:
        _invalid()
    if observed != raw_digest:
        _invalid()
    return raw_digest


def _private_root(raw: Path) -> Path:
    if not isinstance(raw, Path) or not raw.is_absolute() or _contains_symlink(raw):
        _invalid()
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise P6PostSourceAdapterProfileError() from error
    if not root.is_dir() or _git_root_for(root) != root:
        _invalid()
    return root


def _private_path(root: Path, relative: Path, *, want_dir: bool) -> Path:
    candidate = root / relative
    if _contains_symlink(candidate):
        _invalid()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise P6PostSourceAdapterProfileError() from error
    if not _is_relative_to(resolved, root):
        _invalid()
    if want_dir:
        if not resolved.is_dir():
            _invalid()
    elif not _executable_file(resolved):
        _invalid()
    return resolved


def _project_python(raw: object) -> Path:
    path = raw if isinstance(raw, Path) else Path(raw) if isinstance(raw, str) else None
    if path is None or not path.is_absolute() or _contains_symlink(path):
        _invalid()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6PostSourceAdapterProfileError() from error
    if resolved != path or not _executable_file(resolved):
        _invalid()
    return resolved


def _adapter_python(raw: object) -> Path | None:
    if raw is None:
        return None
    try:
        return validate_adapter_python(raw)
    except P6PythonRuntimeError as error:
        raise P6PostSourceAdapterProfileError() from error


def _adapter_dependency_root(root: Path, raw: object) -> Path | None:
    if raw is None:
        return None
    relative = _relative_path(raw.as_posix()) if isinstance(raw, Path) else _relative_path(raw)
    return _private_path(root, relative, want_dir=True)


def _profile_dependency_root(profile: ValidatedPostSourceAdapterProfile) -> Path:
    root = profile.adapter_dependency_root
    if not isinstance(root, Path):
        _invalid()
    try:
        relative = root.relative_to(profile.private_root)
    except ValueError:
        _invalid()
    validated = _adapter_dependency_root(profile.private_root, relative)
    if validated is None:
        _invalid()
    return validated


def _relative_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw or "//" in raw:
        _invalid()
    path = Path(raw)
    if path.is_absolute() or path == Path(".") or ".." in path.parts or ".git" in path.parts:
        _invalid()
    return path


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise P6PostSourceAdapterProfileError() from error


def _executable_file(path: Path) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False
    return path.is_file() and info.st_nlink == 1 and os.access(path, os.X_OK)


def _git_root_for(path: Path) -> Path:
    working_directory = path if path.is_dir() else path.parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(working_directory), "rev-parse", "--show-toplevel"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise P6PostSourceAdapterProfileError() from error
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) != 1 or not lines[0]:
        _invalid()
    try:
        return Path(lines[0]).resolve(strict=True)
    except OSError as error:
        raise P6PostSourceAdapterProfileError() from error


def _contains_symlink(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(component != anchor and component.is_symlink() for component in (path, *path.parents))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _invalid() -> None:
    raise P6PostSourceAdapterProfileError()
