"""Normalize explicit private P6 history inputs into one local root."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from framework.stage6.p6_external_training_binding_v1 import (
    bind_external_training_contract,
    external_training_binding_to_mapping,
    validate_external_training_binding,
)
from framework.stage6.p6_history_execution_closure_v1 import (
    copy_execution_closure,
    render_normalized_runner_template,
    validate_execution_closure_manifest,
    validate_normalized_runner_closure,
    validate_source_runner_roles,
)
from framework.stage6.p6_history_recipe_normalization_v1 import (
    P6HistoryNormalizationError,
    SOURCE_MAP_KEYS,
    SOURCE_MAP_SCHEMA_VERSION,
    SOURCE_MAP_V2_SCHEMA_VERSION,
    _derivation_invalid,
    _invalid,
    _recipe_from_v2_source_map,
    _validate_recipe,
    _validate_source_group,
)
from framework.stage6.p6_runner_template_validator_v1 import (
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_source_wrapper_profile_v1 import (
    extract_project_python_from_source,
    render_self_contained_source_wrapper,
)


REGISTRY_SCHEMA_VERSION = "stage5_candidate_source_registry_v1"
LEGACY_SCHEMA_VERSION = "p6_h800_coptv2x_local_v2"
INPUT_NAMES = (
    "gold176_rows",
    "gold176_graph_features",
    "capability_profiles",
    "closure",
)
ASSET_LABELS = ("training-data", "model-init", "toolchain")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
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
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "Git root resolution failed"
        ) from error
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) != 1 or not lines[0]:
        _invalid("Git root resolution failed")
    raw_root = Path(lines[0])
    if not raw_root.is_absolute() or _contains_symlink_component(raw_root):
        _invalid("Git root is invalid")
    try:
        root = raw_root.resolve(strict=True)
    except OSError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "Git root is unavailable"
        ) from error
    if not root.is_dir() or not _is_relative_to(path.resolve(strict=False), root):
        _invalid("Git root does not contain the private input")
    return root


def _resolve_existing_root(raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str):
        _invalid(f"{label} is invalid")
    path = Path(raw_path)
    if not path.is_absolute() or _contains_symlink_component(path):
        _invalid(f"{label} is invalid")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", f"{label} is unavailable"
        ) from error
    if not resolved.is_dir():
        _invalid(f"{label} is invalid")
    return resolved


def _resolve_existing_json(raw_path: object, history_root: Path, label: str) -> Path:
    if not isinstance(raw_path, str):
        _invalid(f"{label} is invalid")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path.suffix != ".json"
        or _contains_symlink_component(path)
    ):
        _invalid(f"{label} is invalid")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", f"{label} is unavailable"
        ) from error
    if not resolved.is_file() or not _is_relative_to(resolved, history_root):
        _invalid(f"{label} escapes history root")
    if _git_root_for(resolved) != _git_root_for(history_root):
        _invalid(f"{label} does not share the history Git root")
    return resolved


def _validate_private_source_map(
    source_map: Mapping[str, Any],
    history_root: Path,
    *,
    runner_template_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(source_map, Mapping):
        _invalid("source map contract is invalid")
    schema_version = source_map.get("schema_version")
    if schema_version == SOURCE_MAP_SCHEMA_VERSION:
        if set(source_map) != set(SOURCE_MAP_KEYS):
            _invalid("source map contract is invalid")
    elif schema_version == SOURCE_MAP_V2_SCHEMA_VERSION:
        if not {
            "external_training_binding",
            "execution_code_closure",
        }.issubset(source_map):
            _invalid("source map runtime binding is incomplete")
        if source_map.get("recipe_mode") not in {
            "explicit_dynamic_recipe",
            "procedural_profile",
        }:
            _derivation_invalid("source map recipe mode is invalid")
    else:
        _invalid("source map contract is invalid")
    if (
        not isinstance(history_root, Path)
        or not history_root.is_absolute()
        or _contains_symlink_component(history_root)
    ):
        _invalid("history root is invalid")
    try:
        resolved_history_root = history_root.resolve(strict=True)
    except OSError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "history root is unavailable"
        ) from error
    declared_history_root = _resolve_existing_root(
        source_map.get("history_root"), "history root"
    )
    if declared_history_root != resolved_history_root:
        _invalid("history root is inconsistent")
    git_root = _git_root_for(resolved_history_root)
    if resolved_history_root != git_root:
        _invalid("history root is invalid")
    recipe_v2_source = schema_version == SOURCE_MAP_V2_SCHEMA_VERSION
    assets = source_map.get("asset_paths")
    if not isinstance(assets, Mapping) or set(assets) != set(ASSET_LABELS):
        _invalid("asset mapping is invalid")
    canonical_assets: dict[str, Path] = {}
    if not recipe_v2_source:
        canonical_assets = {
            label: _resolve_existing_root(assets[label], f"{label} asset")
            for label in ASSET_LABELS
        }
        if len(set(canonical_assets.values())) != len(ASSET_LABELS):
            _invalid("asset mapping paths must be unique")
        if any(
            not _is_relative_to(path, resolved_history_root)
            or _git_root_for(path) != git_root
            for path in canonical_assets.values()
        ):
            _invalid("asset mapping escapes history root")
    raw_inputs = source_map.get("input_sources")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != set(INPUT_NAMES):
        _invalid("input source mapping is invalid")
    canonical_inputs = {
        name: _resolve_existing_json(raw_inputs[name], resolved_history_root, name)
        for name in INPUT_NAMES
    }
    if len(set(canonical_inputs.values())) != len(INPUT_NAMES):
        _invalid("input source paths must be unique")
    if schema_version == SOURCE_MAP_SCHEMA_VERSION:
        recipe = _validate_recipe(source_map.get("dynamic_materialization_recipe"))
        derived = False
    else:
        recipe, derived = _recipe_from_v2_source_map(
            source_map, resolved_history_root, runner_template_path
        )
    raw_source_group = copy.deepcopy(source_map.get("source_contract"))
    external_training = None
    execution_closure = None
    source_runner = None
    project_python = None
    if recipe_v2_source:
        if runner_template_path is None:
            _invalid("runner template is required")
        external_training = validate_external_training_binding(
            source_map.get("external_training_binding"),
            code_toolchain_root=resolved_history_root,
            local_output_root=resolved_history_root / ".p6-normalization-output-sentinel",
            reserved_paths=(),
        )
        execution_closure = validate_execution_closure_manifest(
            source_map.get("execution_code_closure"),
            source_history_root=resolved_history_root,
            external_training=external_training,
        )
        source_runner = validate_pre_provision_runner_template(
            runner_template_path,
            resolved_history_root,
            require_exact_history_environment=True,
        )
        validate_source_runner_roles(source_runner, execution_closure)
        source_role = next(
            role for role in execution_closure.roles if role.role == "source_materializer"
        )
        source_root = next(
            root
            for root in execution_closure.roots
            if root.closure_id == source_role.closure_id
        )
        project_python = extract_project_python_from_source(
            source_root.source_root / source_role.entrypoint_relative_path
        )
        if not isinstance(raw_source_group, Mapping) or not isinstance(
            raw_source_group.get("source_contract"), Mapping
        ):
            _invalid("source contract is invalid")
        raw_source_group = copy.deepcopy(dict(raw_source_group))
        raw_source_group["source_contract"] = bind_external_training_contract(
            raw_source_group["source_contract"], external_training
        )
    if derived:
        if not isinstance(raw_source_group, Mapping):
            _derivation_invalid("source contract is invalid")
        raw_contract = raw_source_group.get("source_contract")
        if not isinstance(raw_contract, Mapping):
            _derivation_invalid("source contract is invalid")
        existing_recipe = raw_contract.get("dynamic_materialization_recipe")
        if existing_recipe is not None and existing_recipe != recipe:
            _derivation_invalid("source contract recipe is inconsistent")
        raw_source_group = copy.deepcopy(dict(raw_source_group))
        raw_source_group["source_contract"] = copy.deepcopy(dict(raw_contract))
        raw_source_group["source_contract"][
            "dynamic_materialization_recipe"
        ] = copy.deepcopy(recipe)
    try:
        source_group = _validate_source_group(raw_source_group, recipe)
    except P6HistoryNormalizationError as error:
        if schema_version == SOURCE_MAP_V2_SCHEMA_VERSION:
            raise P6HistoryNormalizationError(
                "history_recipe_derivation_invalid",
                "source contract recipe is inconsistent",
            ) from error
        raise
    canonical = {
        "history_root": resolved_history_root,
        "asset_paths": canonical_assets,
        "input_sources": canonical_inputs,
        "source_contract": source_group,
        "dynamic_materialization_recipe": recipe,
    }
    if recipe_v2_source:
        canonical["external_training"] = external_training
        canonical["execution_closure"] = execution_closure
        canonical["source_runner"] = source_runner
        canonical["project_python"] = project_python
    if derived:
        canonical["derived_recipe"] = copy.deepcopy(recipe)
    return canonical


def _git_check_ignored(repository: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repository)
    except ValueError:
        return True
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(relative)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _validate_private_destination(private_dir: Path) -> Path:
    if (
        not isinstance(private_dir, Path)
        or not private_dir.is_absolute()
        or _contains_symlink_component(private_dir)
    ):
        _invalid("private root destination is invalid")
    try:
        destination = private_dir.resolve(strict=False)
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid",
            "private root destination is unavailable",
        ) from error
    if destination.exists():
        _invalid("private root destination already exists")
    if _is_relative_to(destination, repository) and not _git_check_ignored(
        repository, destination
    ):
        _invalid("private root destination is not ignored")
    parent = destination.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        _invalid("private root destination is unavailable")
    return destination


def _copy_json_inputs(input_sources: Mapping[str, Path], inputs_dir: Path) -> dict[str, Path]:
    inputs_dir.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for name in INPUT_NAMES:
        destination = inputs_dir / f"{name}.json"
        shutil.copyfile(input_sources[name], destination, follow_symlinks=False)
        json.loads(destination.read_text(encoding="utf-8"))
        paths[name] = destination
    return paths


def _copy_private_tree(source: Path, destination: Path) -> None:
    if _contains_symlink_component(source):
        _invalid("private asset source is invalid")
    destination.mkdir(parents=True)
    for raw_path in source.rglob("*"):
        if _contains_symlink_component(raw_path):
            _invalid("private asset source contains a symlink")
        relative = raw_path.relative_to(source)
        target = destination / relative
        if raw_path.is_dir():
            target.mkdir()
        elif raw_path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_path, target, follow_symlinks=False)
            shutil.copymode(raw_path, target, follow_symlinks=False)
        else:
            _invalid("private asset source contains an unsupported entry")


def _initialize_private_git_root(root: Path) -> None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "init", "-q"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "private root initialization failed"
        ) from error
    if completed.returncode != 0:
        _invalid("private root initialization failed")
    try:
        (root / ".git" / "info" / "exclude").write_text(
            "external-training-binding.yaml\n"
            "runner-template.yaml\n"
            "source-wrapper-profile.yaml\n"
            "legacy.local.yaml\n"
            "runs/\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "private root initialization failed"
        ) from error


def _build_registry_v1(source_group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "groups": [copy.deepcopy(dict(source_group))],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _atomic_write_text(path, serialized)


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, yaml.safe_dump(dict(payload), sort_keys=False))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _legacy_locator(
    paths: Mapping[str, Path],
    private_root: Path,
    *,
    expected_recipe_path: Path | None = None,
    recipe_v2: bool = False,
) -> dict[str, Any]:
    python_executable = str(Path(sys.executable).resolve(strict=True))
    locator = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "target": "h800",
        "asset_paths": {
            "training-data": str(private_root / "inputs"),
            "model-init": str(paths["registry"]),
            "toolchain": str(
                private_root / ("execution-closure" if recipe_v2 else "toolchain")
            ),
        },
        "local_input_paths": {name: str(paths[name]) for name in INPUT_NAMES},
        "candidate_source_mode": "framework_stage2_search_space",
        "stage2_search_space_path": str(private_root / "stage1_partition_manifest.json"),
        "source_registry_step": {
            "name": "build_source_registry",
            "argv": [
                python_executable,
                str(REPOSITORY_ROOT / "tools/release/build_p6_history_registry.py"),
                "--binding",
                "{binding}",
                "--pyramid-candidate-plan",
                "{pyramid_candidate_plan}",
                "--source-registry-json",
                "{source_registry_json}",
                "--local-output-root",
                "{local_output_root}",
            ],
        },
        "measurement_step": {
            "name": "measure_batch",
            "argv": [
                python_executable,
                str(REPOSITORY_ROOT / "tools/release/measure_p6_history_batch.py"),
                "--binding",
                "{binding}",
                "--measurement-request",
                "{measurement_request}",
                "--feedback-json",
                "{feedback_json}",
                "--round-output-root",
                "{round_output_root}",
            ],
        },
        "local_output_root": str(private_root / "runs"),
    }
    if expected_recipe_path is not None:
        locator["history_recipe_derivation_path"] = str(expected_recipe_path)
    return locator


def normalize_history_inputs(
    source_map: Mapping[str, Any],
    history_root: Path,
    private_dir: Path,
    *,
    runner_template_path: Path | None = None,
) -> dict[str, Path]:
    """Fail-closed normalization from a private source map to a private root."""
    staged: Path | None = None
    destination = _validate_private_destination(private_dir)
    try:
        canonical = _validate_private_source_map(
            source_map,
            history_root,
            runner_template_path=runner_template_path,
        )
        if "external_training" in canonical:
            validate_external_training_binding(
                external_training_binding_to_mapping(
                    canonical["external_training"]
                ),
                code_toolchain_root=destination,
                local_output_root=(
                    destination.parent / f".{destination.name}.runtime-output"
                ),
                reserved_paths=(),
            )
        staged = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".staging",
            )
        )
        _initialize_private_git_root(staged)
        paths = _copy_json_inputs(canonical["input_sources"], staged / "inputs")
        registry_path = staged / "registry" / "candidate-source-registry.json"
        _atomic_write_json(registry_path, _build_registry_v1(canonical["source_contract"]))
        paths["registry"] = registry_path
        recipe_v2_source = "external_training" in canonical
        if recipe_v2_source:
            copied_roles = copy_execution_closure(
                canonical["execution_closure"], staged_private_root=staged
            )
            external_path = staged / "external-training-binding.yaml"
            _atomic_write_yaml(
                external_path,
                external_training_binding_to_mapping(canonical["external_training"]),
            )
            paths["external_training_binding"] = external_path
            source_role = next(
                role
                for role in canonical["execution_closure"].roles
                if role.role == "source_materializer"
            )
            source_closure_root = next(
                root
                for root in canonical["execution_closure"].roots
                if root.closure_id == source_role.closure_id
            )
            profile = {
                "schema_version": "p6_private_source_wrapper_profile_v2",
                "wrapper_kind": "repo_cwd_exec_v1",
                "destination_relative_path": (
                    "documented-stage5-chain/"
                    "stage5_materialize_round_sources_v1.sh"
                ),
                "implementation_relative_path": copied_roles[
                    "source_materializer"
                ].relative_to(staged).as_posix(),
                "implementation_cwd_relative_path": source_closure_root.destination_relative_root.as_posix(),
                "project_python": str(canonical["project_python"]),
            }
            profile_path = staged / "source-wrapper-profile.yaml"
            _atomic_write_yaml(profile_path, profile)
            render_self_contained_source_wrapper(profile, history_root=staged)
            paths["source_wrapper_profile"] = profile_path
            runner_payload = render_normalized_runner_template(
                canonical["source_runner"],
                normalized_private_root=staged,
                copied_role_paths=copied_roles,
            )
            runner_path = staged / "runner-template.yaml"
            _atomic_write_yaml(runner_path, runner_payload)
            paths["runner_template"] = runner_path
            validate_normalized_runner_closure(
                runner_path,
                normalized_private_root=staged,
                expected_closure=canonical["execution_closure"],
            )
        else:
            _copy_private_tree(
                canonical["asset_paths"]["toolchain"], staged / "toolchain"
            )
        if "derived_recipe" in canonical:
            derivation_path = staged / "derivation" / "recipe.json"
            _atomic_write_json(derivation_path, canonical["derived_recipe"])
            paths["derivation_recipe"] = derivation_path
        public_paths = {
            name: destination / path.relative_to(staged)
            for name, path in paths.items()
        }
        legacy_path = staged / "legacy.local.yaml"
        _atomic_write_yaml(
            legacy_path,
            _legacy_locator(
                public_paths,
                destination,
                expected_recipe_path=public_paths.get("derivation_recipe"),
                recipe_v2=recipe_v2_source,
            ),
        )
        paths["legacy"] = legacy_path
        relative_paths = {
            name: path.relative_to(staged)
            for name, path in paths.items()
        }
        os.replace(staged, destination)
        staged = None
        return {name: destination / relative for name, relative in relative_paths.items()}
    except P6HistoryNormalizationError:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "private history normalization failed"
        ) from error
    finally:
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)
