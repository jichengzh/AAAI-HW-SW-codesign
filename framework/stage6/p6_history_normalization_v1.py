"""Normalize explicit private P6 history inputs into one local root."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import string
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from framework.stage5.production_search_v1 import validate_source_contract
from framework.stage6.p6_history_recipe_bridge_v1 import (
    P6HistoryRecipeDerivationError,
    derive_dynamic_recipe_from_procedural_source,
)
from framework.stage6.p6_history_recipe_profiles_v1 import (
    CANONICAL_ARTIFACT_ID_TEMPLATE,
    CANONICAL_GROUP_ID_TEMPLATE,
    PROFILE_V1,
    RECIPE_V2,
    SHARED_SOURCE_PATH_KEYS,
)


SOURCE_MAP_SCHEMA_VERSION = "p6_history_normalization_source_v1"
SOURCE_MAP_V2_SCHEMA_VERSION = "p6_history_normalization_source_v2"
REGISTRY_SCHEMA_VERSION = "stage5_candidate_source_registry_v1"
LEGACY_SCHEMA_VERSION = "p6_h800_coptv2x_local_v2"
RECIPE_SCHEMA_VERSION = "p6_history_dynamic_materialization_recipe_v1"
INPUT_NAMES = (
    "gold176_rows",
    "gold176_graph_features",
    "capability_profiles",
    "closure",
)
ASSET_LABELS = ("training-data", "model-init", "toolchain")
STAGE_WIDTH_FIELDS = ("stage1_width", "stage2_width", "stage3_width")
OUTPUT_TEMPLATE_KEYS = (
    "training_path_template",
    "checkpoint_path_template",
    "onnx_path_template",
    "calibration_path_template",
)
SOURCE_MAP_KEYS = frozenset(
    {
        "schema_version",
        "asset_paths",
        "input_sources",
        "source_contract",
        "dynamic_materialization_recipe",
        "history_root",
    }
)
SOURCE_MAP_COMMON_KEYS = frozenset(
    {
        "schema_version",
        "asset_paths",
        "input_sources",
        "source_contract",
        "history_root",
        "recipe_mode",
    }
)
SOURCE_MAP_V2_EXPLICIT_KEYS = SOURCE_MAP_COMMON_KEYS | {
    "dynamic_materialization_recipe"
}
SOURCE_MAP_V2_PROCEDURAL_KEYS = SOURCE_MAP_COMMON_KEYS | {
    "procedural_recipe_profile",
    "procedural_recipe_source",
}
RECIPE_KEYS = frozenset(
    {
        "schema_version",
        "stage_width_fields",
        "group_id_template",
        "artifact_id_template",
        "output_path_templates_by_q_mode",
    }
)
RECIPE_V2_KEYS = frozenset(
    {
        "schema_version",
        "stage_width_fields",
        "group_id_template",
        "artifact_id_template",
        "shared_source_path_templates",
    }
)
PROCEDURAL_ROLES = (
    "controller",
    "source_materializer",
    "performance_plan",
    "finalizer",
)
RECIPE_V2_SAMPLE_WIDTHS = ((16, 32, 64), (16, 33, 65), (17, 33, 65))
FORBIDDEN_CONTEXT_TOKENS = (
    "metric",
    "objective",
    "cache",
    "result",
    "terminal",
    "latency",
    "energy",
    "ap30",
    "ap50",
    "ap70",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class P6HistoryNormalizationError(ValueError):
    """Stable categorized failure for private history normalization."""

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")


def _invalid(detail: str) -> None:
    raise P6HistoryNormalizationError("history_normalization_invalid", detail)


def _derivation_invalid(detail: str) -> None:
    raise P6HistoryNormalizationError("history_recipe_derivation_invalid", detail)


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


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source contract is not canonical JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _validate_no_forbidden_context(value: object) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key != "source_status" and (
                key == "status"
                or key.endswith("_status")
                or any(token in key for token in FORBIDDEN_CONTEXT_TOKENS)
            ):
                _invalid("source contract contains forbidden result context")
            _validate_no_forbidden_context(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _validate_no_forbidden_context(child)


def _validate_format_template(
    raw_template: object, allowed_fields: set[str]
) -> set[str]:
    if not isinstance(raw_template, str) or not raw_template.strip():
        _invalid("dynamic materialization template is invalid")
    observed_fields: set[str] = set()
    try:
        parsed = tuple(string.Formatter().parse(raw_template))
    except ValueError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid",
            "dynamic materialization template is invalid",
        ) from error
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in allowed_fields or format_spec or conversion is not None:
            _invalid("dynamic materialization template field is invalid")
        observed_fields.add(field_name)
    if not observed_fields:
        _invalid("dynamic materialization template has no authoritative fields")
    return observed_fields


def _render_template(template: str, values: Mapping[str, object]) -> str:
    try:
        rendered = template.format_map(values)
    except (KeyError, ValueError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid",
            "dynamic materialization template failed",
        ) from error
    if not rendered or any(character in rendered for character in ("\x00", "\r", "\n")):
        _invalid("dynamic materialization template rendered an invalid value")
    return rendered


def _validate_recipe(raw_recipe: object) -> dict[str, Any]:
    if isinstance(raw_recipe, Mapping) and raw_recipe.get("schema_version") == RECIPE_V2:
        return _validate_recipe_v2(raw_recipe)
    if not isinstance(raw_recipe, Mapping) or set(raw_recipe) != set(RECIPE_KEYS):
        _invalid("dynamic materialization recipe is missing or invalid")
    recipe = copy.deepcopy(dict(raw_recipe))
    if (
        recipe.get("schema_version") != RECIPE_SCHEMA_VERSION
        or recipe.get("stage_width_fields") != list(STAGE_WIDTH_FIELDS)
    ):
        _invalid("dynamic materialization recipe is incompatible")
    for name in ("group_id_template", "artifact_id_template"):
        _validate_format_template(recipe.get(name), set(STAGE_WIDTH_FIELDS))
    raw_outputs = recipe.get("output_path_templates_by_q_mode")
    if not isinstance(raw_outputs, Mapping) or set(raw_outputs) != {"fp16", "int8"}:
        _invalid("dynamic materialization output mapping is invalid")
    allowed_output_fields = {*STAGE_WIDTH_FIELDS, "group_id", "artifact_id", "q_mode"}
    for raw_templates in raw_outputs.values():
        if not isinstance(raw_templates, Mapping) or set(raw_templates) != set(
            OUTPUT_TEMPLATE_KEYS
        ):
            _invalid("dynamic materialization output mapping is incomplete")
        for key in OUTPUT_TEMPLATE_KEYS:
            _validate_format_template(raw_templates.get(key), allowed_output_fields)
    _validate_recipe_outputs_do_not_collide(recipe)
    return recipe


def _validate_recipe_v2(raw_recipe: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw_recipe) != set(RECIPE_V2_KEYS):
        _invalid("dynamic materialization recipe is missing or invalid")
    recipe = copy.deepcopy(dict(raw_recipe))
    if (
        recipe.get("schema_version") != RECIPE_V2
        or recipe.get("stage_width_fields") != list(STAGE_WIDTH_FIELDS)
        or recipe.get("group_id_template") != CANONICAL_GROUP_ID_TEMPLATE
        or recipe.get("artifact_id_template") != CANONICAL_ARTIFACT_ID_TEMPLATE
    ):
        _invalid("dynamic materialization recipe is incompatible")
    for name in ("group_id_template", "artifact_id_template"):
        _validate_format_template(recipe.get(name), set(STAGE_WIDTH_FIELDS))
    raw_templates = recipe.get("shared_source_path_templates")
    if not isinstance(raw_templates, Mapping) or set(raw_templates) != set(
        SHARED_SOURCE_PATH_KEYS
    ):
        _invalid("shared source path mapping is incomplete")
    allowed_fields = {*STAGE_WIDTH_FIELDS, "group_id", "artifact_id"}
    for key in SHARED_SOURCE_PATH_KEYS:
        template = raw_templates.get(key)
        fields = _validate_format_template(template, allowed_fields)
        assert isinstance(template, str)
        path = Path(template)
        if (
            path.is_absolute()
            or PureWindowsPath(template).is_absolute()
            or "\\" in template
            or ".." in path.parts
        ):
            _invalid("shared source path escapes the allowed root")
        if not {"group_id", "artifact_id"}.intersection(fields) and not set(
            STAGE_WIDTH_FIELDS
        ).issubset(fields):
            _invalid("shared source path lacks canonical group identity")
    _validate_recipe_v2_outputs_do_not_collide(recipe, raw_templates)
    return recipe


def _validate_recipe_v2_outputs_do_not_collide(
    recipe: Mapping[str, Any], templates: Mapping[str, Any]
) -> None:
    seen_across_groups: set[Path] = set()
    for widths in RECIPE_V2_SAMPLE_WIDTHS:
        values = dict(zip(STAGE_WIDTH_FIELDS, widths, strict=True))
        group_id = _render_template(recipe["group_id_template"], values)
        artifact_id = _render_template(recipe["artifact_id_template"], values)
        render_values = {**values, "group_id": group_id, "artifact_id": artifact_id}
        rendered = {
            Path(_render_template(templates[key], render_values))
            for key in SHARED_SOURCE_PATH_KEYS
        }
        if len(rendered) != len(SHARED_SOURCE_PATH_KEYS):
            _invalid("shared source paths collide in a group")
        if seen_across_groups.intersection(rendered):
            _invalid("shared source paths collide across groups")
        seen_across_groups.update(rendered)


def _validate_recipe_outputs_do_not_collide(recipe: Mapping[str, Any]) -> None:
    values = {
        "stage1_width": 16,
        "stage2_width": 32,
        "stage3_width": 64,
    }
    group_id = _render_template(recipe["group_id_template"], values)
    artifact_id = _render_template(recipe["artifact_id_template"], values)
    rendered_paths: set[Path] = set()
    for q_mode, templates in recipe["output_path_templates_by_q_mode"].items():
        render_values = {
            **values,
            "group_id": group_id,
            "artifact_id": artifact_id,
            "q_mode": q_mode,
        }
        for key in OUTPUT_TEMPLATE_KEYS:
            relative = Path(_render_template(templates[key], render_values))
            if relative.is_absolute() or ".." in relative.parts:
                _invalid("dynamic materialization output escapes the allowed root")
            if relative in rendered_paths:
                _invalid("dynamic materialization output path collides")
            rendered_paths.add(relative)


def _validate_source_group(raw_group: object, recipe: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_group, Mapping):
        _invalid("source contract is invalid")
    group = copy.deepcopy(dict(raw_group))
    if group.get("materialization_kind") != "local_pyramid_tvm":
        _invalid("source contract is not the Pyramid/TVM source")
    contract = group.get("source_contract")
    if not isinstance(contract, Mapping):
        _invalid("source contract is invalid")
    contract_copy = copy.deepcopy(dict(contract))
    if contract_copy.get("dynamic_materialization_recipe") != recipe:
        _invalid("source contract recipe is inconsistent")
    _validate_no_forbidden_context(contract_copy)
    group["source_contract"] = contract_copy
    group["source_contract_sha256"] = _canonical_sha(contract_copy)
    try:
        validate_source_contract(group)
    except (TypeError, ValueError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source contract is invalid"
        ) from error
    if (
        group.get("model") != "pyramid"
        or group.get("source_status") != "ready"
        or group.get("source_contract", {}).get("model") != "pyramid"
    ):
        _invalid("source contract is incompatible")
    return group


def _validate_procedural_recipe_source(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) not in (
        {"role_refs"},
        {"role_selection"},
    ):
        _derivation_invalid("procedural recipe source is invalid")
    source = copy.deepcopy(dict(raw))
    refs = source.get("role_refs", source.get("role_selection"))
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        if tuple(refs) != PROCEDURAL_ROLES:
            _derivation_invalid("procedural recipe roles are invalid")
    elif isinstance(refs, Mapping):
        if set(refs) != set(PROCEDURAL_ROLES) or any(
            not isinstance(value, (str, Mapping)) for value in refs.values()
        ):
            _derivation_invalid("procedural recipe roles are invalid")
    else:
        _derivation_invalid("procedural recipe roles are invalid")
    return source


def _recipe_from_v2_source_map(
    source_map: Mapping[str, Any],
    history_root: Path,
    runner_template_path: Path | None,
) -> tuple[dict[str, Any], bool]:
    mode = source_map.get("recipe_mode")
    if mode == "explicit_dynamic_recipe":
        if set(source_map) != set(SOURCE_MAP_V2_EXPLICIT_KEYS):
            _derivation_invalid("explicit recipe fields are inconsistent")
        try:
            return _validate_recipe(source_map.get("dynamic_materialization_recipe")), False
        except P6HistoryNormalizationError as error:
            raise P6HistoryNormalizationError(
                "history_recipe_derivation_invalid", "explicit recipe is invalid"
            ) from error
    if mode != "procedural_profile" or set(source_map) != set(
        SOURCE_MAP_V2_PROCEDURAL_KEYS
    ):
        _derivation_invalid("procedural recipe fields are inconsistent")
    if runner_template_path is None:
        _derivation_invalid("runner template is required")
    if source_map.get("procedural_recipe_profile") != PROFILE_V1:
        _derivation_invalid("procedural recipe profile is unknown")
    procedural_source = _validate_procedural_recipe_source(
        source_map.get("procedural_recipe_source")
    )
    bridge_source_map = {
        "procedural_recipe_profile": source_map.get("procedural_recipe_profile"),
        "procedural_recipe_source": procedural_source,
    }
    try:
        recipe = derive_dynamic_recipe_from_procedural_source(
            source_map=bridge_source_map,
            runner_template_path=runner_template_path,
            history_root=history_root,
        )
    except P6HistoryRecipeDerivationError as error:
        raise P6HistoryNormalizationError(
            "history_recipe_derivation_invalid", "procedural recipe derivation failed"
        ) from error
    return recipe, True


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
    assets = source_map.get("asset_paths")
    if not isinstance(assets, Mapping) or set(assets) != set(ASSET_LABELS):
        _invalid("asset mapping is invalid")
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
) -> dict[str, Any]:
    python_executable = str(Path(sys.executable).resolve(strict=True))
    locator = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "target": "h800",
        "asset_paths": {
            "training-data": str(private_root / "inputs"),
            "model-init": str(paths["registry"]),
            "toolchain": str(private_root / "toolchain"),
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
        _copy_private_tree(canonical["asset_paths"]["toolchain"], staged / "toolchain")
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
