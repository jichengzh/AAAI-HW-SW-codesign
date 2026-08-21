"""Render and validate a deterministic private P6 source marker wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Literal

import yaml

from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_runner_template_validator_v1 import (
    ValidatedRunnerTemplate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILE_SCHEMA_VERSION = "p6_private_source_wrapper_profile_v1"
WRAPPER_KIND = "repo_cwd_exec_v1"
SOURCE_MARKER_BASENAME = "stage5_materialize_round_sources_v1.sh"
PROFILE_KEYS = frozenset(
    {
        "schema_version",
        "wrapper_kind",
        "destination_relative_path",
        "implementation_relative_path",
        "implementation_cwd_relative_path",
    }
)
EXPECTED_EXECUTION_STAGES = (
    "source_materialization",
    "quantization",
    "performance",
    "ap",
    "finalization",
)
SOURCE_TEMPLATE_ARGV_TAIL = ("{measurement_request}", "{round_output_root}")
SOURCE_RUNTIME_ARGV_SHAPE = (
    "--request",
    "<absolute-private-request-json>",
    "--model",
    "pyramid",
    "--group-id",
    "<canonical-pyramid-group-id>",
    "--gpu",
    "<validated-binding-gpu-index>",
)
MAX_PROFILE_SIZE = 1024 * 1024
SAFE_RELATIVE_PATH = re.compile(r"\A[A-Za-z0-9._/-]+\Z")


class P6SourceWrapperProfileError(ValueError):
    """Stable, path-free source-wrapper validation failure."""

    category = "history_execution_invalid"

    def __init__(self) -> None:
        super().__init__(self.category)


@dataclass(frozen=True)
class ValidatedSourceWrapper:
    """Validated private source marker and its fixed runtime argv contract."""

    executable: Path
    argv_shape: tuple[str, ...]
    marker_basename: str


@dataclass(frozen=True)
class SourceWrapperRenderPlan:
    """Immutable deterministic deployment plan derived from a private profile."""

    destination: Path
    wrapper_kind: Literal["repo_cwd_exec_v1"]
    implementation: Path
    implementation_cwd: Path
    expected_bytes: bytes
    expected_sha256: str


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


def load_source_wrapper_profile(path: Path) -> dict[str, Any]:
    """Load one absolute, non-symlinked private profile with duplicate-key checks."""
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or _contains_symlink_component(path)
    ):
        _invalid()
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > MAX_PROFILE_SIZE:
            _invalid()
        _validate_profile_privacy(resolved)
        payload = yaml.load(
            resolved.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except P6SourceWrapperProfileError:
        raise
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise P6SourceWrapperProfileError() from error
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        _invalid()
    return dict(payload)


def _validate_profile_privacy(path: Path) -> None:
    try:
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if not _is_relative_to(path, repository):
        return
    relative = path.relative_to(repository)
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(relative)],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise P6SourceWrapperProfileError() from error
    if completed.returncode != 0:
        _invalid()


def expected_wrapper_bytes_from_profile(
    source_wrapper_profile: Path,
    *,
    history_root: Path,
) -> bytes:
    """Recompute the exact wrapper bytes from an ignored private profile."""
    profile = load_source_wrapper_profile(source_wrapper_profile)
    return _build_render_plan(profile, history_root=history_root).expected_bytes


def render_self_contained_source_wrapper(
    profile: Mapping[str, Any],
    *,
    history_root: Path,
) -> ValidatedSourceWrapper:
    """Create an absent marker or accept an exact existing generated marker."""
    plan = _build_render_plan(profile, history_root=history_root)
    _write_wrapper_script(plan)
    return _validate_rendered_wrapper(plan)


def validate_self_contained_source_wrapper(
    validated_template: ValidatedRunnerTemplate,
    *,
    source_wrapper_profile: Path,
) -> ValidatedSourceWrapper:
    """Bind a validated source stage to the exact bytes specified by its profile."""
    if not isinstance(validated_template, ValidatedRunnerTemplate):
        _invalid()
    plan = _build_render_plan(
        load_source_wrapper_profile(source_wrapper_profile),
        history_root=validated_template.history_root,
    )
    if tuple(validated_template.stage_argv) != EXPECTED_EXECUTION_STAGES:
        _invalid()
    source_argv = tuple(validated_template.stage_argv["source_materialization"])
    if (
        len(source_argv) != 3
        or tuple(source_argv[1:]) != SOURCE_TEMPLATE_ARGV_TAIL
        or Path(source_argv[0]) != plan.destination
    ):
        _invalid()
    source_component = validated_template.component_paths.get("source_materializer")
    if source_component != plan.destination:
        _invalid()
    environment = validated_template.execution_interface.get("environment")
    values = environment.get("values") if isinstance(environment, Mapping) else None
    if not isinstance(values, Mapping) or set(values) != set(
        EXPECTED_HISTORY_ENV_KEYS
    ):
        _invalid()
    return _validate_rendered_wrapper(plan)


def _build_render_plan(
    profile: Mapping[str, Any],
    *,
    history_root: Path,
) -> SourceWrapperRenderPlan:
    root = _resolve_history_root(history_root)
    if not isinstance(profile, Mapping) or set(profile) != PROFILE_KEYS:
        _invalid()
    if (
        profile.get("schema_version") != PROFILE_SCHEMA_VERSION
        or profile.get("wrapper_kind") != WRAPPER_KIND
    ):
        _invalid()
    destination_relative = _validated_relative_path(
        profile.get("destination_relative_path")
    )
    implementation_relative = _validated_relative_path(
        profile.get("implementation_relative_path")
    )
    implementation_cwd_relative = _validated_relative_path(
        profile.get("implementation_cwd_relative_path")
    )
    if (
        destination_relative.name != SOURCE_MARKER_BASENAME
        or implementation_relative.name == SOURCE_MARKER_BASENAME
    ):
        _invalid()
    destination = _resolve_destination(root, destination_relative)
    implementation = _resolve_existing_private_path(
        root, implementation_relative, require_directory=False
    )
    implementation_cwd = _resolve_existing_private_path(
        root, implementation_cwd_relative, require_directory=True
    )
    if not _is_relative_to(implementation, implementation_cwd):
        _invalid()
    expected_bytes = _wrapper_bytes(
        implementation_relative,
        implementation_cwd_relative,
    )
    return SourceWrapperRenderPlan(
        destination=destination,
        wrapper_kind=WRAPPER_KIND,
        implementation=implementation,
        implementation_cwd=implementation_cwd,
        expected_bytes=expected_bytes,
        expected_sha256=hashlib.sha256(expected_bytes).hexdigest(),
    )


def _validated_relative_path(raw: object) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or not SAFE_RELATIVE_PATH.fullmatch(raw)
        or "//" in raw
        or "\\" in raw
    ):
        _invalid()
    path = Path(raw)
    if (
        path.is_absolute()
        or path == Path(".")
        or ".." in path.parts
        or ".git" in path.parts
    ):
        _invalid()
    return path


def _resolve_history_root(raw: Path) -> Path:
    if (
        not isinstance(raw, Path)
        or not raw.is_absolute()
        or _contains_symlink_component(raw)
    ):
        _invalid()
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if not root.is_dir() or _git_root_for(root) != root:
        _invalid()
    return root


def _resolve_existing_private_path(
    root: Path,
    relative: Path,
    *,
    require_directory: bool,
) -> Path:
    candidate = root / relative
    if _contains_symlink_component(candidate):
        _invalid()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if not _is_relative_to(resolved, root) or _git_root_for(resolved) != root:
        _invalid()
    if require_directory:
        if not resolved.is_dir():
            _invalid()
    elif not resolved.is_file() or not os.access(resolved, os.X_OK):
        _invalid()
    return resolved


def _resolve_destination(root: Path, relative: Path) -> Path:
    candidate = root / relative
    if _contains_symlink_component(candidate):
        _invalid()
    try:
        destination = candidate.resolve(strict=False)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if not _is_relative_to(destination, root) or destination == root:
        _invalid()
    if candidate.exists() and (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve(strict=True) != destination
    ):
        _invalid()
    return destination


def _wrapper_bytes(implementation: Path, implementation_cwd: Path) -> bytes:
    body = (
        "#!/bin/sh\n"
        "set -eu\n"
        ': "${P6_HISTORY_PRIVATE_ROOT:?}"\n'
        'root="$(cd "${P6_HISTORY_PRIVATE_ROOT}" && pwd -P)"\n'
        f'implementation="${{root}}/{implementation.as_posix()}"\n'
        f'implementation_cwd="${{root}}/{implementation_cwd.as_posix()}"\n'
        '[ -f "$implementation" ] && [ -x "$implementation" ]\n'
        'cd "$implementation_cwd"\n'
        'implementation_cwd="$(pwd -P)"\n'
        'exec "$implementation" "$@"\n'
    )
    return body.encode("utf-8")


def _write_wrapper_script(plan: SourceWrapperRenderPlan) -> None:
    destination = plan.destination
    if destination.exists() or destination.is_symlink():
        _validate_rendered_wrapper(plan)
        destination.chmod(0o700)
        return
    try:
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if _contains_symlink_component(destination):
        _invalid()
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o700,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(plan.expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        destination.chmod(0o700)
    except FileExistsError:
        _validate_rendered_wrapper(plan)
    except OSError as error:
        if created:
            destination.unlink(missing_ok=True)
        raise P6SourceWrapperProfileError() from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_rendered_wrapper(
    plan: SourceWrapperRenderPlan,
) -> ValidatedSourceWrapper:
    destination = plan.destination
    if (
        destination.is_symlink()
        or not destination.is_file()
        or not os.access(destination, os.X_OK)
        or destination.name != SOURCE_MARKER_BASENAME
    ):
        _invalid()
    try:
        actual_bytes = destination.read_bytes()
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    actual_sha256 = hashlib.sha256(actual_bytes).hexdigest()
    if actual_sha256 != plan.expected_sha256 or actual_bytes != plan.expected_bytes:
        _invalid()
    return ValidatedSourceWrapper(
        executable=destination,
        argv_shape=SOURCE_RUNTIME_ARGV_SHAPE,
        marker_basename=SOURCE_MARKER_BASENAME,
    )


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
        raise P6SourceWrapperProfileError() from error
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) != 1 or not lines[0]:
        _invalid()
    try:
        return Path(lines[0]).resolve(strict=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _invalid() -> None:
    raise P6SourceWrapperProfileError()
