"""Normalize explicit private P6 history inputs into one local root."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import yaml

from framework.stage6.p6_external_training_binding_v1 import (
    bind_external_training_contract,
    validate_external_training_binding,
)
from framework.stage6.p6_history_execution_closure_v1 import (
    validate_execution_closure_manifest,
    validate_source_runner_roles,
)
from framework.stage6.p6_history_normalization_staging_v1 import (
    copy_private_tree,
    publish_normalized_history,
    stage_base_history,
    stage_recipe_v2_history,
    validate_destination_external,
)
from framework.stage6.p6_history_recipe_normalization_v1 import (
    P6HistoryNormalizationError,
    SOURCE_MAP_KEYS,
    SOURCE_MAP_SCHEMA_VERSION,
    SOURCE_MAP_V2_SCHEMA_VERSION,
    SOURCE_MAP_V3_SCHEMA_VERSION,
    _derivation_invalid,
    _invalid,
    _recipe_from_v2_source_map,
    _validate_recipe,
    _validate_source_group,
)
from framework.stage6.p6_post_source_leaf_binding_v1 import (
    P6PostSourceLeafBindingError,
    validate_post_source_leaf_binding,
)
from framework.stage6.p6_runner_template_validator_v1 import (
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_source_wrapper_profile_v1 import (
    extract_project_python_from_source,
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


def _validate_source_map_schema(source_map: Mapping[str, Any]) -> tuple[str, bool]:
    if not isinstance(source_map, Mapping):
        _invalid("source map contract is invalid")
    schema_version = source_map.get("schema_version")
    if schema_version == SOURCE_MAP_SCHEMA_VERSION:
        if set(source_map) != set(SOURCE_MAP_KEYS):
            _invalid("source map contract is invalid")
    elif schema_version in {SOURCE_MAP_V2_SCHEMA_VERSION, SOURCE_MAP_V3_SCHEMA_VERSION}:
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
        if schema_version == SOURCE_MAP_V3_SCHEMA_VERSION and (
            "post_source_leaf_binding" not in source_map
        ):
            _invalid("post-source leaf binding is missing")
        if schema_version == SOURCE_MAP_V2_SCHEMA_VERSION and (
            "post_source_leaf_binding" in source_map
        ):
            _invalid("post-source leaf binding requires source map v3")
    else:
        _invalid("source map contract is invalid")
    return schema_version, schema_version in {
        SOURCE_MAP_V2_SCHEMA_VERSION,
        SOURCE_MAP_V3_SCHEMA_VERSION,
    }


def _canonical_history_root(
    source_map: Mapping[str, Any], history_root: Path
) -> tuple[Path, Path]:
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
    return resolved_history_root, git_root


def _canonical_assets(
    source_map: Mapping[str, Any],
    resolved_history_root: Path,
    git_root: Path,
    *,
    recipe_v2_source: bool,
) -> dict[str, Path]:
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
    return canonical_assets


def _canonical_inputs(
    source_map: Mapping[str, Any], resolved_history_root: Path
) -> dict[str, Path]:
    raw_inputs = source_map.get("input_sources")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != set(INPUT_NAMES):
        _invalid("input source mapping is invalid")
    canonical_inputs = {
        name: _resolve_existing_json(raw_inputs[name], resolved_history_root, name)
        for name in INPUT_NAMES
    }
    if len(set(canonical_inputs.values())) != len(INPUT_NAMES):
        _invalid("input source paths must be unique")
    return canonical_inputs


def _source_recipe(
    source_map: Mapping[str, Any],
    schema_version: str,
    resolved_history_root: Path,
    runner_template_path: Path | None,
) -> tuple[Mapping[str, Any], bool]:
    if schema_version == SOURCE_MAP_SCHEMA_VERSION:
        return _validate_recipe(source_map.get("dynamic_materialization_recipe")), False
    return _recipe_from_v2_source_map(
        source_map, resolved_history_root, runner_template_path
    )


def _execution_runtime(
    source_map: Mapping[str, Any],
    resolved_history_root: Path,
    runner_template_path: Path | None,
) -> tuple[Any, Any, Any, Path]:
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
    return external_training, execution_closure, source_runner, project_python


def _post_source_binding(
    source_map: Mapping[str, Any], schema_version: str, execution_closure: Any
) -> Any:
    if schema_version != SOURCE_MAP_V3_SCHEMA_VERSION:
        return None
    try:
        return validate_post_source_leaf_binding(
            source_map.get("post_source_leaf_binding"),
            execution_closure=execution_closure,
        )
    except P6PostSourceLeafBindingError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid",
            "post-source leaf binding is invalid",
        ) from error


def _bind_source_group(raw_source_group: object, external_training: Any) -> dict[str, Any]:
    if not isinstance(raw_source_group, Mapping) or not isinstance(
        raw_source_group.get("source_contract"), Mapping
    ):
        _invalid("source contract is invalid")
    bound = copy.deepcopy(dict(raw_source_group))
    bound["source_contract"] = bind_external_training_contract(
        bound["source_contract"], external_training
    )
    return bound


def _with_derived_recipe(
    raw_source_group: object, recipe: Mapping[str, Any], *, derived: bool
) -> object:
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
    return raw_source_group


def _validated_source_group(
    raw_source_group: object,
    recipe: Mapping[str, Any],
    schema_version: str,
) -> Mapping[str, Any]:
    try:
        return _validate_source_group(raw_source_group, recipe)
    except P6HistoryNormalizationError as error:
        if schema_version == SOURCE_MAP_V2_SCHEMA_VERSION:
            raise P6HistoryNormalizationError(
                "history_recipe_derivation_invalid",
                "source contract recipe is inconsistent",
            ) from error
        raise


def _canonical_source_map(
    *,
    resolved_history_root: Path,
    canonical_assets: Mapping[str, Path],
    canonical_inputs: Mapping[str, Path],
    source_group: Mapping[str, Any],
    recipe: Mapping[str, Any],
    runtime: tuple[Any, Any, Any, Path] | None,
    post_source_leaf_binding: Any,
    derived: bool,
) -> dict[str, Any]:
    canonical = {
        "history_root": resolved_history_root,
        "asset_paths": dict(canonical_assets),
        "input_sources": dict(canonical_inputs),
        "source_contract": source_group,
        "dynamic_materialization_recipe": recipe,
    }
    if runtime is not None:
        external_training, execution_closure, source_runner, project_python = runtime
        canonical["external_training"] = external_training
        canonical["execution_closure"] = execution_closure
        canonical["source_runner"] = source_runner
        canonical["project_python"] = project_python
    if post_source_leaf_binding is not None:
        canonical["post_source_leaf_binding"] = post_source_leaf_binding
    if derived:
        canonical["derived_recipe"] = copy.deepcopy(recipe)
    return canonical


def _validate_private_source_map(
    source_map: Mapping[str, Any],
    history_root: Path,
    *,
    runner_template_path: Path | None = None,
) -> dict[str, Any]:
    schema_version, recipe_v2_source = _validate_source_map_schema(source_map)
    resolved_root, git_root = _canonical_history_root(source_map, history_root)
    assets = _canonical_assets(
        source_map, resolved_root, git_root, recipe_v2_source=recipe_v2_source
    )
    inputs = _canonical_inputs(source_map, resolved_root)
    recipe, derived = _source_recipe(
        source_map, schema_version, resolved_root, runner_template_path
    )
    raw_source_group = copy.deepcopy(source_map.get("source_contract"))
    runtime = None
    post_source_leaf_binding = None
    if recipe_v2_source:
        runtime = _execution_runtime(source_map, resolved_root, runner_template_path)
        post_source_leaf_binding = _post_source_binding(
            source_map, schema_version, runtime[1]
        )
        raw_source_group = _bind_source_group(raw_source_group, runtime[0])
    raw_source_group = _with_derived_recipe(
        raw_source_group, recipe, derived=derived
    )
    source_group = _validated_source_group(
        raw_source_group, recipe, schema_version
    )
    return _canonical_source_map(
        resolved_history_root=resolved_root,
        canonical_assets=assets,
        canonical_inputs=inputs,
        source_group=source_group,
        recipe=recipe,
        runtime=runtime,
        post_source_leaf_binding=post_source_leaf_binding,
        derived=derived,
    )


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
            source_map, history_root, runner_template_path=runner_template_path
        )
        validate_destination_external(canonical, destination)
        staged = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".staging",
            )
        )
        paths = stage_base_history(canonical, staged)
        recipe_v2_source = "external_training" in canonical
        if recipe_v2_source:
            paths = stage_recipe_v2_history(canonical, staged, paths)
        else:
            copy_private_tree(
                canonical["asset_paths"]["toolchain"], staged / "toolchain"
            )
        result = publish_normalized_history(
            canonical,
            staged,
            destination,
            paths,
            recipe_v2_source=recipe_v2_source,
        )
        staged = None
        return result
    except P6HistoryNormalizationError:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "private history normalization failed"
        ) from error
    finally:
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)
