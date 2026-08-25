"""Pure recipe and source-map canonicalization for P6 history normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import string
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
SOURCE_MAP_V3_SCHEMA_VERSION = "p6_history_normalization_source_v3"
RECIPE_SCHEMA_VERSION = "p6_history_dynamic_materialization_recipe_v1"
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
        "external_training_binding",
        "execution_code_closure",
    }
)
SOURCE_MAP_V2_EXPLICIT_KEYS = SOURCE_MAP_COMMON_KEYS | {
    "dynamic_materialization_recipe"
}
SOURCE_MAP_V2_PROCEDURAL_KEYS = SOURCE_MAP_COMMON_KEYS | {
    "procedural_recipe_profile",
    "procedural_recipe_source",
}
SOURCE_MAP_V3_EXPLICIT_KEYS = SOURCE_MAP_V2_EXPLICIT_KEYS | {
    "post_source_leaf_binding"
}
SOURCE_MAP_V3_PROCEDURAL_KEYS = SOURCE_MAP_V2_PROCEDURAL_KEYS | {
    "post_source_leaf_binding"
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


class P6HistoryNormalizationError(ValueError):
    """Stable categorized failure for private history normalization."""

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")


class _UniqueSourceMapLoader(yaml.SafeLoader):
    pass


def _construct_unique_source_mapping(
    loader: _UniqueSourceMapLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.YAMLError("duplicate key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSourceMapLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_source_mapping,
)


def load_source_map_document(path: Path) -> dict[str, Any]:
    """Load JSON or YAML through one duplicate-key rejecting safe loader."""
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSourceMapLoader)
    except (OSError, TypeError, UnicodeError, yaml.YAMLError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source map is unavailable"
        ) from error
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        _invalid("source map is invalid")
    return copy.deepcopy(payload)


def _invalid(detail: str) -> None:
    raise P6HistoryNormalizationError("history_normalization_invalid", detail)


def _derivation_invalid(detail: str) -> None:
    raise P6HistoryNormalizationError("history_recipe_derivation_invalid", detail)


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
    mode, schema_version = source_map.get("recipe_mode"), source_map.get("schema_version")
    if mode == "explicit_dynamic_recipe":
        expected_keys = (
            SOURCE_MAP_V3_EXPLICIT_KEYS
            if schema_version == SOURCE_MAP_V3_SCHEMA_VERSION
            else SOURCE_MAP_V2_EXPLICIT_KEYS
        )
        if set(source_map) != set(expected_keys):
            _derivation_invalid("explicit recipe fields are inconsistent")
        try:
            return _validate_recipe(source_map.get("dynamic_materialization_recipe")), False
        except P6HistoryNormalizationError as error:
            raise P6HistoryNormalizationError(
                "history_recipe_derivation_invalid", "explicit recipe is invalid"
            ) from error
    expected_keys = (
        SOURCE_MAP_V3_PROCEDURAL_KEYS
        if schema_version == SOURCE_MAP_V3_SCHEMA_VERSION
        else SOURCE_MAP_V2_PROCEDURAL_KEYS
    )
    if mode != "procedural_profile" or set(source_map) != set(expected_keys):
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
