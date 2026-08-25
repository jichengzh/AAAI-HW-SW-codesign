"""Build and load the ignored private P6 post-source adapter profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Literal

import yaml

from framework.stage6.p6_history_execution_closure_v1 import (
    P6ValidatedExecutionClosure,
)
from framework.stage6.p6_post_source_leaf_binding_v1 import (
    POST_SOURCE_LEAF_NAMES,
    ValidatedPostSourceLeafBinding,
)


PROFILE_SCHEMA_VERSION = "p6_post_source_adapter_profile_v1"
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
    schema_version: Literal["p6_post_source_adapter_profile_v1"]
    private_root: Path
    project_python: Path
    adapters: tuple[PostSourceAdapter, ...]
    leaves: tuple[PostSourceLeaf, ...]


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
    return ValidatedPostSourceAdapterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        private_root=root,
        project_python=python,
        adapters=adapters,
        leaves=leaves,
    )


def post_source_adapter_profile_to_mapping(
    profile: ValidatedPostSourceAdapterProfile,
) -> dict[str, Any]:
    """Serialize a validated profile without private absolute implementation paths."""
    if not isinstance(profile, ValidatedPostSourceAdapterProfile):
        _invalid()
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "target": dict(TARGET_PROFILE),
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


def load_post_source_adapter_profile(
    path: Path,
    *,
    private_root: Path,
) -> ValidatedPostSourceAdapterProfile:
    """Load one ignored normalized post-source adapter profile."""
    root = _private_root(private_root)
    payload = _load_profile_payload(path, root)
    if (
        set(payload) != PROFILE_KEYS
        or payload.get("schema_version") != PROFILE_SCHEMA_VERSION
        or payload.get("target") != TARGET_PROFILE
        or payload.get("runner_interface_schema_version") != RUNNER_INTERFACE_SCHEMA_VERSION
    ):
        _invalid()
    project_python = _project_python(payload.get("project_python"))
    adapters = _load_adapters(payload.get("adapters"), root)
    leaves = _load_leaves(payload.get("leaves"), root)
    return ValidatedPostSourceAdapterProfile(
        PROFILE_SCHEMA_VERSION,
        root,
        project_python,
        adapters,
        leaves,
    )


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
