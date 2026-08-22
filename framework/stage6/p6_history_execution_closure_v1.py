"""Validate and copy the explicit P6 execution-code closure."""

from __future__ import annotations

import ast
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Literal

from framework.stage6.p6_external_training_binding_v1 import P6ExternalTrainingBinding
from framework.stage6.p6_history_binding_v1 import REQUIRED_STAGE_PLACEHOLDERS
from framework.stage6.p6_runner_template_validator_v1 import (
    ValidatedRunnerTemplate,
    validate_pre_provision_runner_template,
)


EXECUTION_CLOSURE_ROLES: tuple[str, ...] = (
    "stage1_scan",
    "controller",
    "source_materializer",
    "quantization",
    "performance",
    "ap",
    "finalization",
    "activation",
)
_SCHEMA_VERSION = "p6_execution_code_closure_v1"
_ROOT_KEYS = frozenset({"closure_id", "source_root", "destination_relative_root", "sha256"})
_ROLE_KEYS = frozenset({"closure_id", "entrypoint_relative_path"})
_CANONICAL_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HEX_DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")
_SOURCE_WRAPPER = Path("documented-stage5-chain/stage5_materialize_round_sources_v1.sh")
_ROLE_BY_STAGE = {
    "source_materialization": "source_materializer",
    "quantization": "quantization",
    "performance": "performance",
    "ap": "ap",
    "finalization": "finalization",
}
_APPROVED_SYSTEM_IMPORTS = frozenset(sys.builtin_module_names) | frozenset(sys.stdlib_module_names)


@dataclass(frozen=True)
class P6ExecutionClosureRoot:
    closure_id: str
    source_root: Path
    destination_relative_root: Path
    sha256: str


@dataclass(frozen=True)
class P6ExecutionClosureRole:
    role: str
    closure_id: str
    entrypoint_relative_path: Path


@dataclass(frozen=True)
class P6ValidatedExecutionClosure:
    schema_version: Literal["p6_execution_code_closure_v1"]
    roots: tuple[P6ExecutionClosureRoot, ...]
    roles: tuple[P6ExecutionClosureRole, ...]


class P6ExecutionClosureError(ValueError):
    """Stable path-free closure validation error."""

    category: Literal["history_normalization_invalid"] = "history_normalization_invalid"

    def __init__(self) -> None:
        super().__init__(self.category)


def validate_execution_closure_manifest(
    raw: Mapping[str, Any],
    *,
    source_history_root: Path,
    external_training: P6ExternalTrainingBinding,
) -> P6ValidatedExecutionClosure:
    """Validate the exact all-role manifest and every declared source byte."""
    try:
        history_root = _existing_directory(source_history_root)
        if _git_root_for(history_root) != history_root:
            _invalid()
        if not isinstance(external_training, P6ExternalTrainingBinding):
            _invalid()
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"schema_version", "roots", "roles"}
            or raw.get("schema_version") != _SCHEMA_VERSION
        ):
            _invalid()
        roots = _validated_roots(raw.get("roots"), history_root, external_training)
        roles = _validated_roles(raw.get("roles"), roots)
        _validate_source_import_siblings(roles, roots)
        return P6ValidatedExecutionClosure(_SCHEMA_VERSION, roots, roles)
    except P6ExecutionClosureError:
        raise
    except (OSError, TypeError, ValueError, SyntaxError):
        _invalid()


def copy_execution_closure(
    closure: P6ValidatedExecutionClosure,
    *,
    staged_private_root: Path,
) -> Mapping[str, Path]:
    """Copy each closure root once and publish only a completely verified tree."""
    temporary: Path | None = None
    try:
        root = _planned_or_existing_directory(staged_private_root)
        if not isinstance(closure, P6ValidatedExecutionClosure):
            _invalid()
        destination_root = root / "execution-closure"
        if destination_root.exists() or destination_root.is_symlink():
            _invalid()
        temporary = Path(tempfile.mkdtemp(dir=root, prefix=".execution-closure.", suffix=".tmp"))
        copied_roots: dict[str, Path] = {}
        for declared in closure.roots:
            relative = declared.destination_relative_root.relative_to("execution-closure")
            destination = temporary / relative
            _copy_tree(declared.source_root, destination)
            if _tree_digest(destination) != declared.sha256:
                _invalid()
            copied_roots[declared.closure_id] = destination
        copied_roles = {
            role.role: copied_roots[role.closure_id] / role.entrypoint_relative_path
            for role in closure.roles
        }
        if set(copied_roles) != set(EXECUTION_CLOSURE_ROLES) or any(
            not _single_link_executable(path) for path in copied_roles.values()
        ):
            _invalid()
        os.replace(temporary, destination_root)
        temporary = None
        published = _expected_normalized_role_paths(closure, root)
        if any(not _single_link_executable(path) for path in published.values()):
            _invalid()
        return published
    except P6ExecutionClosureError:
        raise
    except (OSError, StopIteration, TypeError, ValueError):
        _invalid()
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def render_normalized_runner_template(
    source_template: ValidatedRunnerTemplate,
    *,
    normalized_private_root: Path,
    copied_role_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Rewrite every repository-local executable to its normalized role path."""
    if not isinstance(source_template, ValidatedRunnerTemplate) or set(copied_role_paths) != set(
        EXECUTION_CLOSURE_ROLES
    ):
        _invalid()
    root = _planned_or_existing_directory(normalized_private_root)
    canonical = {role: _role_relative_path(path, root) for role, path in copied_role_paths.items()}
    _reject_path_argv_tails(source_template)
    interface = copy.deepcopy(dict(source_template.execution_interface))
    interface["controller"]["argv"][0] = canonical["controller"].as_posix()
    for entry in interface["execution_chain"]:
        role = _ROLE_BY_STAGE[entry["stage"]]
        entry["argv"][0] = str(
            _SOURCE_WRAPPER if role == "source_materializer" else canonical[role]
        )
        entry["required_placeholders"] = list(REQUIRED_STAGE_PLACEHOLDERS[entry["stage"]])
    interface["environment"]["activation_argv"][0] = canonical["activation"].as_posix()
    return {
        "schema_version": "p6_history_runner_template_v1",
        "stage1_scan": {
            "name": source_template.stage1_name,
            "argv": [
                canonical["stage1_scan"].as_posix(),
                *source_template.stage1_argv[1:],
            ],
        },
        "execution_interface": interface,
    }


def validate_normalized_runner_closure(
    runner_template_path: Path,
    *,
    normalized_private_root: Path,
    expected_closure: P6ValidatedExecutionClosure,
) -> ValidatedRunnerTemplate:
    """Revalidate the runner and prove all eight executable roles are declared."""
    try:
        root = _existing_directory(normalized_private_root)
        validated = validate_pre_provision_runner_template(
            runner_template_path,
            root,
            require_exact_history_environment=True,
        )
        expected = _expected_normalized_role_paths(expected_closure, root)
        observed = _runner_role_paths(validated)
        if observed != {**expected, "source_materializer": root / _SOURCE_WRAPPER}:
            _invalid()
        return validated
    except P6ExecutionClosureError:
        raise
    except Exception:
        _invalid()


def _validated_roots(
    raw: object,
    history_root: Path,
    external: P6ExternalTrainingBinding,
) -> tuple[P6ExecutionClosureRoot, ...]:
    if not isinstance(raw, list) or not raw:
        _invalid()
    roots: list[P6ExecutionClosureRoot] = []
    seen_ids: set[str] = set()
    seen_destinations: set[Path] = set()
    external_paths = (
        external.dataset_root,
        external.base_checkpoint_path,
        external.pyramid_config_path,
    )
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != _ROOT_KEYS:
            _invalid()
        closure_id = item.get("closure_id")
        if not isinstance(closure_id, str) or not _CANONICAL_ID.fullmatch(closure_id):
            _invalid()
        raw_source = item.get("source_root")
        if not isinstance(raw_source, str) or str(Path(raw_source)) != raw_source:
            _invalid()
        source = _existing_directory(Path(raw_source))
        destination = _relative_path(item.get("destination_relative_root"))
        digest = item.get("sha256")
        if (
            closure_id in seen_ids
            or source == history_root
            or not source.is_relative_to(history_root)
            or _git_root_for(source) != history_root
            or destination.parts[0] != "execution-closure"
            or len(destination.parts) < 2
            or destination in seen_destinations
            or not isinstance(digest, str)
            or not _HEX_DIGEST.fullmatch(digest)
            or any(_overlaps(source, path) for path in external_paths)
        ):
            _invalid()
        if _tree_digest(source) != digest:
            _invalid()
        seen_ids.add(closure_id)
        seen_destinations.add(destination)
        roots.append(P6ExecutionClosureRoot(closure_id, source, destination, digest))
    if any(
        _overlaps(left.destination_relative_root, right.destination_relative_root)
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        _invalid()
    return tuple(roots)


def _validated_roles(
    raw: object, roots: tuple[P6ExecutionClosureRoot, ...]
) -> tuple[P6ExecutionClosureRole, ...]:
    if not isinstance(raw, Mapping) or set(raw) != set(EXECUTION_CLOSURE_ROLES):
        _invalid()
    by_id = {root.closure_id: root for root in roots}
    roles: list[P6ExecutionClosureRole] = []
    for role in EXECUTION_CLOSURE_ROLES:
        item = raw[role]
        if not isinstance(item, Mapping) or set(item) != _ROLE_KEYS:
            _invalid()
        closure_id = item.get("closure_id")
        relative = _relative_path(item.get("entrypoint_relative_path"))
        if closure_id not in by_id:
            _invalid()
        entrypoint = by_id[closure_id].source_root / relative
        if not _single_link_executable(entrypoint):
            _invalid()
        roles.append(P6ExecutionClosureRole(role, closure_id, relative))
    return tuple(roles)


def _validate_source_import_siblings(
    roles: tuple[P6ExecutionClosureRole, ...],
    roots: tuple[P6ExecutionClosureRoot, ...],
) -> None:
    role = next(item for item in roles if item.role == "source_materializer")
    root = next(item for item in roots if item.closure_id == role.closure_id)
    entrypoint = root.source_root / role.entrypoint_relative_path
    if entrypoint.suffix != ".py":
        return
    _validate_module_imports(entrypoint, root.source_root, frozenset())


def _validate_module_imports(module: Path, closure_root: Path, visited: frozenset[Path]) -> None:
    if module in visited:
        return
    next_visited = visited | {module}
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        for name, relative_level, imported_members in _import_names(node):
            base = _import_base(module, closure_root, relative_level)
            local_modules = _local_import_modules(base, name)
            if local_modules:
                for local in local_modules:
                    _validate_module_imports(local, closure_root, next_visited)
                _validate_imported_members(
                    local_modules[-1],
                    imported_members,
                    closure_root,
                    next_visited,
                )
            elif relative_level or name.split(".", 1)[0] not in _APPROVED_SYSTEM_IMPORTS:
                _invalid()


def _import_names(node: ast.AST) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    if isinstance(node, ast.Import):
        return tuple((alias.name, 0, ()) for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        if node.module:
            return (
                (
                    node.module,
                    node.level,
                    tuple(alias.name for alias in node.names),
                ),
            )
        return tuple((alias.name, node.level, ()) for alias in node.names)
    return ()


def _validate_imported_members(
    module: Path,
    members: tuple[str, ...],
    closure_root: Path,
    visited: frozenset[Path],
) -> None:
    if not members:
        return
    exported = _module_bound_names(module)
    for member in members:
        if member == "*" or member in exported:
            continue
        if module.name != "__init__.py":
            _invalid()
        submodules = _local_import_modules(module.parent, member)
        if not submodules:
            _invalid()
        for submodule in submodules:
            _validate_module_imports(submodule, closure_root, visited)


def _module_bound_names(module: Path) -> frozenset[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(statement.name)
        elif isinstance(statement, ast.Assign):
            targets = statement.targets
            names.update(_assignment_names(targets))
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets = (statement.target,)
            names.update(_assignment_names(targets))
        elif isinstance(statement, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in statement.names)
    return frozenset(names)


def _assignment_names(targets: tuple[ast.expr, ...] | list[ast.expr]) -> set[str]:
    return {
        node.id for target in targets for node in ast.walk(target) if isinstance(node, ast.Name)
    }


def _import_base(module: Path, closure_root: Path, relative_level: int) -> Path:
    base = closure_root if relative_level == 0 else module.parent
    for _ in range(max(0, relative_level - 1)):
        base = base.parent
    if not base.is_relative_to(closure_root):
        _invalid()
    return base


def _local_import_modules(base: Path, name: str) -> tuple[Path, ...]:
    parts = name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        _invalid()
    modules: list[Path] = []
    cursor = base
    for part in parts[:-1]:
        cursor /= part
        package = cursor / "__init__.py"
        if not package.is_file():
            return ()
        modules.append(package)
    local_file = cursor / f"{parts[-1]}.py"
    local_package = cursor / parts[-1] / "__init__.py"
    if local_file.is_file():
        return (*modules, local_file)
    if local_package.is_file():
        return (*modules, local_package)
    return ()


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, mode=0o700)
    for entry in _walk(source):
        relative = entry.relative_to(source)
        target = destination / relative
        info = entry.lstat()
        if stat.S_ISDIR(info.st_mode):
            target.mkdir(mode=stat.S_IMODE(info.st_mode))
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry, target, follow_symlinks=False)
            os.chmod(target, stat.S_IMODE(info.st_mode), follow_symlinks=False)
        else:
            _invalid()


def _tree_digest(root: Path) -> str:
    entries: list[dict[str, object]] = []
    for path in _walk(root):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            entries.append({"kind": "directory", "path": relative})
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append(
                {
                    "kind": "regular_file",
                    "path": relative,
                    "sha256": digest.hexdigest(),
                    "size": info.st_size,
                }
            )
        else:
            _invalid()
    encoded = json.dumps(
        entries,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _walk(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child in children:
            path = Path(child.path)
            paths.append(path)
            if child.is_dir(follow_symlinks=False):
                visit(path)

    visit(root)
    return tuple(paths)


def _expected_normalized_role_paths(
    closure: P6ValidatedExecutionClosure, root: Path
) -> dict[str, Path]:
    roots = {item.closure_id: root / item.destination_relative_root for item in closure.roots}
    return {
        role.role: roots[role.closure_id] / role.entrypoint_relative_path for role in closure.roles
    }


def _runner_role_paths(template: ValidatedRunnerTemplate) -> dict[str, Path]:
    return {
        "stage1_scan": Path(template.stage1_argv[0]),
        "controller": Path(template.execution_interface["controller"]["argv"][0]),
        **{_ROLE_BY_STAGE[stage]: Path(argv[0]) for stage, argv in template.stage_argv.items()},
        "activation": Path(template.activation_argv[0]),
    }


def validate_source_runner_roles(
    template: ValidatedRunnerTemplate,
    closure: P6ValidatedExecutionClosure,
) -> None:
    """Prove the source runner and closure name the same executable roles."""
    try:
        expected = {
            role.role: next(
                root.source_root for root in closure.roots if root.closure_id == role.closure_id
            )
            / role.entrypoint_relative_path
            for role in closure.roles
        }
        observed = _runner_role_paths(template)
        compared = set(EXECUTION_CLOSURE_ROLES) - {"source_materializer"}
        if any(observed[role] != expected[role] for role in compared):
            _invalid()
        _reject_path_argv_tails(template)
    except (KeyError, StopIteration, TypeError, ValueError):
        _invalid()


def _reject_path_argv_tails(template: ValidatedRunnerTemplate) -> None:
    interface = template.execution_interface
    argv_groups = (
        template.stage1_argv,
        tuple(interface["controller"]["argv"]),
        *(tuple(entry["argv"]) for entry in interface["execution_chain"]),
        template.activation_argv,
    )
    if any(
        Path(token).is_absolute() or "/" in token or "\\" in token
        for argv in argv_groups
        for token in argv[1:]
    ):
        _invalid()


def _role_relative_path(raw: object, root: Path) -> Path:
    if not isinstance(raw, Path) or not raw.is_absolute():
        _invalid()
    try:
        relative = raw.relative_to(root)
    except ValueError:
        _invalid()
    if relative == Path(".") or ".." in relative.parts:
        _invalid()
    if raw.exists() and not _single_link_executable(raw):
        _invalid()
    return relative


def _relative_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw or "//" in raw:
        _invalid()
    path = Path(raw)
    if path.as_posix() != raw:
        _invalid()
    if (
        path.is_absolute()
        or path == Path(".")
        or any(part in {"", ".", "..", ".git"} for part in path.parts)
    ):
        _invalid()
    return path


def _existing_directory(path: object) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        _invalid()
    _reject_symlink_components(path)
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_dir():
        _invalid()
    return resolved


def _planned_or_existing_directory(path: object) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        _invalid()
    _reject_symlink_components(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        _invalid()
    return resolved


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                _invalid()
        except FileNotFoundError:
            _invalid()


def _single_link_executable(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and os.access(path, os.X_OK)


def _git_root_for(path: Path) -> Path:
    try:
        completed = subprocess.run(
            ("git", "-C", str(path), "rev-parse", "--show-toplevel"),
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        _invalid()
    if completed.returncode != 0:
        _invalid()
    return Path(completed.stdout.strip()).resolve(strict=True)


def _overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _invalid() -> None:
    raise P6ExecutionClosureError()
