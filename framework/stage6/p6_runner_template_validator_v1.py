"""Validate the private runner template before P6 binding provisioning."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_TEMPLATE_SCHEMA_VERSION = "p6_history_runner_template_v1"
RUNNER_TEMPLATE_KEYS = frozenset(
    {"schema_version", "stage1_scan", "execution_interface"}
)
EXECUTION_INTERFACE_KEYS = frozenset(
    {
        "schema_version",
        "controller",
        "execution_chain",
        "environment",
        "output_layout",
        "actual_feedback",
    }
)
COMPONENT_MARKERS = {
    "controller": "stage5_task_round_controller_v3.sh",
    "source_materializer": "stage5_materialize_round_sources_v1.sh",
    "performance_plan": "stage5_build_performance_plan_v2.py",
    "finalizer": "stage5_finalize_feedback_v2.py",
}
EXECUTION_STAGES = (
    "source_materialization",
    "quantization",
    "performance",
    "ap",
    "finalization",
)
COMPONENT_STAGE_ROLES = {
    "source_materialization": "source_materializer",
    "performance": "performance_plan",
    "finalization": "finalizer",
}
SHELL_TOKENS = frozenset({"$", "`", ";", "|", "&", "<", ">", "\n", "\r"})
MAX_PRIVATE_INPUT_SIZE = 4 * 1024 * 1024


class RunnerTemplateValidationError(ValueError):
    """Stable categorized failure from pre-provision template validation."""

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")


@dataclass(frozen=True)
class ValidatedRunnerTemplate:
    """A private-copy, path-validated runner interface for one history root."""

    history_root: Path
    component_paths: Mapping[str, Path]
    execution_interface: Mapping[str, Any]
    stage_argv: Mapping[str, tuple[str, ...]]


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
            raise ValueError("duplicate mapping key")
        keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def validate_pre_provision_runner_template(
    runner_template_path: Path,
    history_root: Path,
) -> ValidatedRunnerTemplate:
    """Return a detached, validated runner interface for one private Git root."""
    root = _resolve_history_root(history_root)
    _validate_runner_template_privacy(runner_template_path)
    payload = _load_runner_template(runner_template_path)
    stage1_scan = payload["stage1_scan"]
    interface = payload["execution_interface"]
    _validate_stage1_scan(stage1_scan, root)
    component_paths, canonical_interface, stage_argv = _validate_interface(
        interface, root
    )
    return ValidatedRunnerTemplate(
        history_root=root,
        component_paths=copy.deepcopy(component_paths),
        execution_interface=copy.deepcopy(canonical_interface),
        stage_argv=copy.deepcopy(stage_argv),
    )


def _resolve_history_root(raw: Path) -> Path:
    if not isinstance(raw, Path) or not raw.is_absolute() or _contains_symlink_component(raw):
        raise RunnerTemplateValidationError(
            "history_root_ambiguous", "history root is invalid"
        )
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise RunnerTemplateValidationError(
            "history_root_ambiguous", "history root is unavailable"
        ) from error
    if not root.is_dir() or _git_root_for(root) != root:
        raise RunnerTemplateValidationError(
            "history_root_ambiguous", "history root is not a Git root"
        )
    return root


def _validate_runner_template_privacy(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "runner template path is invalid"
        )
    try:
        resolved = path.resolve(strict=True)
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "runner template is unavailable"
        ) from error
    if not _is_relative_to(resolved, repository):
        return
    relative = resolved.relative_to(repository)
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
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "runner template privacy check failed"
        ) from error
    if completed.returncode != 0:
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "runner template is not private"
        )


def _load_runner_template(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file() or path.stat().st_size > MAX_PRIVATE_INPUT_SIZE:
            raise OSError
        payload = yaml.load(
            path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "runner template is unavailable"
        ) from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "runner template is invalid"
        )
    if (
        set(payload) != RUNNER_TEMPLATE_KEYS
        or payload.get("schema_version") != RUNNER_TEMPLATE_SCHEMA_VERSION
        or not isinstance(payload.get("stage1_scan"), Mapping)
        or not isinstance(payload.get("execution_interface"), Mapping)
    ):
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "runner template contract is invalid"
        )
    return copy.deepcopy(payload)


def _validate_stage1_scan(raw: object, root: Path) -> None:
    if not isinstance(raw, Mapping) or set(raw) != {"name", "argv"}:
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "Stage1 scan step is invalid"
        )
    _validate_argv(raw.get("argv"), root)


def _validate_interface(
    raw: object, root: Path
) -> tuple[dict[str, Path], dict[str, Any], dict[str, tuple[str, ...]]]:
    if not isinstance(raw, Mapping) or set(raw) != EXECUTION_INTERFACE_KEYS:
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "execution interface is invalid"
        )
    controller = raw.get("controller")
    chain = raw.get("execution_chain")
    environment = raw.get("environment")
    if not isinstance(controller, Mapping) or not isinstance(chain, list):
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "execution interface is invalid"
        )
    if not isinstance(environment, Mapping) or not isinstance(
        environment.get("values"), Mapping
    ):
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "execution interface is invalid"
        )
    controller_path = _validate_component_argv(
        controller, COMPONENT_MARKERS["controller"], root
    )
    controller_argv = _validate_argv(controller.get("argv"), root)
    component_paths = {"controller": controller_path}
    if len(chain) != len(EXECUTION_STAGES):
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "execution interface is invalid"
        )
    canonical_chain: list[dict[str, Any]] = []
    stage_argv: dict[str, tuple[str, ...]] = {}
    private_stage_executables: set[Path] = set()
    for expected_stage, entry in zip(EXECUTION_STAGES, chain, strict=True):
        if not isinstance(entry, Mapping) or entry.get("stage") != expected_stage:
            raise RunnerTemplateValidationError(
                "execution_interface_unavailable", "execution interface is invalid"
            )
        role = COMPONENT_STAGE_ROLES.get(expected_stage)
        if role is None:
            argv = _validate_argv(entry.get("argv"), root)
            if expected_stage in {"quantization", "ap"}:
                _validate_private_only_stage_executable(
                    argv[0], private_stage_executables
                )
        else:
            component = _validate_component_argv(
                entry, COMPONENT_MARKERS[role], root
            )
            component_paths[role] = component
            argv = _validate_argv(entry.get("argv"), root)
        canonical_entry = copy.deepcopy(dict(entry))
        canonical_entry["argv"] = argv
        canonical_chain.append(canonical_entry)
        stage_argv[expected_stage] = tuple(argv)
    activation_argv = _validate_argv(environment.get("activation_argv"), root)
    canonical_interface = copy.deepcopy(dict(raw))
    canonical_controller = copy.deepcopy(dict(controller))
    canonical_controller["argv"] = controller_argv
    canonical_interface["controller"] = canonical_controller
    canonical_interface["execution_chain"] = canonical_chain
    canonical_environment = copy.deepcopy(dict(environment))
    canonical_environment["activation_argv"] = activation_argv
    canonical_interface["environment"] = canonical_environment
    return component_paths, canonical_interface, stage_argv


def _validate_component_argv(raw: object, marker: str, root: Path) -> Path:
    if not isinstance(raw, Mapping):
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "execution interface is invalid"
        )
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "execution interface is invalid"
        )
    if Path(argv[0]).name != marker:
        raise RunnerTemplateValidationError(
            "history_root_ambiguous", "template component marker does not match role"
        )
    return _resolve_private_executable(argv[0], root, "history_root_ambiguous")


def _validate_argv(raw: object, root: Path) -> list[str]:
    if not isinstance(raw, list) or not raw or any(
        not isinstance(token, str)
        or not token
        or any(shell_token in token for shell_token in SHELL_TOKENS)
        for token in raw
    ):
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "execution interface argv is invalid"
        )
    executable = _resolve_private_executable(
        raw[0], root, "execution_interface_unavailable"
    )
    return [str(executable), *raw[1:]]


def _validate_private_only_stage_executable(
    executable: str, seen: set[Path]
) -> None:
    path = Path(executable)
    if path.name in COMPONENT_MARKERS.values() or path in seen:
        raise RunnerTemplateValidationError(
            "execution_interface_unavailable", "private-only stage executable is invalid"
        )
    seen.add(path)


def _resolve_private_executable(raw: str, root: Path, category: str) -> Path:
    path = Path(raw)
    if ".." in path.parts:
        raise RunnerTemplateValidationError(category, "template executable is invalid")
    candidate = path if path.is_absolute() else root / path
    if _contains_symlink_component(candidate):
        raise RunnerTemplateValidationError(category, "template executable is unsafe")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RunnerTemplateValidationError(category, "template executable is unavailable") from error
    if (
        not resolved.is_absolute()
        or not _is_relative_to(resolved, root)
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
        or _git_root_for(resolved) != root
    ):
        raise RunnerTemplateValidationError(category, "template executable is invalid")
    return resolved


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
        raise RunnerTemplateValidationError(
            "history_root_ambiguous", "Git root resolution failed"
        ) from error
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) != 1 or not lines[0]:
        raise RunnerTemplateValidationError(
            "history_root_ambiguous", "Git root resolution failed"
        )
    try:
        return Path(lines[0]).resolve(strict=True)
    except OSError as error:
        raise RunnerTemplateValidationError(
            "history_root_ambiguous", "Git root is unavailable"
        ) from error


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
