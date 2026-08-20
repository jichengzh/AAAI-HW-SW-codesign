"""Pure derivation of P6 recipe-v2 from public evidence and a validated runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
import string
from typing import Any

from framework.stage6.p6_history_recipe_profiles_v1 import (
    CANONICAL_ARTIFACT_ID_TEMPLATE,
    CANONICAL_GROUP_ID_TEMPLATE,
    RECIPE_V2,
    SHARED_SOURCE_PATH_KEYS,
    STAGE_WIDTH_FIELDS,
    RecipeProfile,
    get_recipe_profile,
)
from framework.stage6.p6_runner_template_validator_v1 import (
    RunnerTemplateValidationError,
    ValidatedRunnerTemplate,
    validate_pre_provision_runner_template,
)


class P6HistoryRecipeDerivationError(ValueError):
    """Stable public failure raised when procedural recipe derivation is unsafe."""

    def __init__(self, detail: str) -> None:
        self.category = "history_recipe_derivation_invalid"
        self.detail = detail
        super().__init__(f"{self.category}: {detail}")


_FORBIDDEN_SOURCE_MAP_FIELDS = frozenset(
    {
        "argv",
        "capability",
        "capabilities",
        "marker",
        "markers",
        "shared_source_path_templates",
        "shared_source_paths",
        "output_path",
        "output_paths",
        "output_path_templates_by_q_mode",
        "source_paths",
        "path_templates",
        *SHARED_SOURCE_PATH_KEYS,
        "base_checkpoint",
        "checkpoint_root",
        "config_root",
        "dataset_root",
        "gpu",
        "gpu_index",
        "gpu_uuid",
        "training_hyperparameter",
        "training_hyperparameters",
        "training_parameters",
    }
)
_RECIPE_KEYS = frozenset(
    {
        "schema_version",
        "stage_width_fields",
        "group_id_template",
        "artifact_id_template",
        "shared_source_path_templates",
    }
)
_TEMPLATE_FIELDS = frozenset({*STAGE_WIDTH_FIELDS, "group_id", "artifact_id"})
_SAMPLE_WIDTHS = ((16, 32, 64), (17, 33, 65))


def derive_dynamic_recipe_from_procedural_source(
    *,
    source_map: Mapping[str, Any],
    runner_template_path: Path,
    history_root: Path,
) -> dict[str, Any]:
    """Render a recipe-v2 without inspecting or invoking any historical program."""
    try:
        profile = _profile_from_source_map(source_map)
        validated = validate_pre_provision_runner_template(
            runner_template_path, history_root
        )
        _validate_runner_against_profile(validated, profile)
        recipe = _render_recipe(profile)
        _validate_recipe(recipe)
        return recipe
    except P6HistoryRecipeDerivationError:
        raise
    except RunnerTemplateValidationError as error:
        raise P6HistoryRecipeDerivationError(
            "runner template validation failed"
        ) from error
    except (TypeError, ValueError, KeyError) as error:
        raise P6HistoryRecipeDerivationError("procedural profile is invalid") from error


def _profile_from_source_map(source_map: Mapping[str, Any]) -> RecipeProfile:
    if not isinstance(source_map, Mapping):
        raise P6HistoryRecipeDerivationError("source map is invalid")
    _reject_source_map_semantic_claims(source_map)
    profile = get_recipe_profile(source_map.get("procedural_recipe_profile"))
    if profile is None:
        raise P6HistoryRecipeDerivationError("procedural profile is unknown")
    return profile


def _reject_source_map_semantic_claims(value: object) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise P6HistoryRecipeDerivationError("source map is invalid")
            if raw_key.lower() in _FORBIDDEN_SOURCE_MAP_FIELDS:
                raise P6HistoryRecipeDerivationError(
                    "source map contains unsupported semantic evidence"
                )
            _reject_source_map_semantic_claims(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_source_map_semantic_claims(child)


def _validate_runner_against_profile(
    validated: ValidatedRunnerTemplate, profile: RecipeProfile
) -> None:
    if set(validated.component_paths) != set(profile.component_markers):
        raise P6HistoryRecipeDerivationError("runner marker roles differ from profile")
    if any(
        path.name != profile.component_markers[role]
        for role, path in validated.component_paths.items()
    ):
        raise P6HistoryRecipeDerivationError("runner markers differ from profile")
    if tuple(validated.stage_argv) != profile.execution_stages:
        raise P6HistoryRecipeDerivationError("runner stages differ from profile")
    source_stage = validated.stage_argv.get("source_materialization")
    if source_stage is None or not source_stage:
        raise P6HistoryRecipeDerivationError("source stage is unavailable")
    expected_placeholders = profile.source_materializer_placeholders
    observed_placeholders = tuple(
        token for token in source_stage[1:] if "{" in token or "}" in token
    )
    if observed_placeholders != expected_placeholders:
        raise P6HistoryRecipeDerivationError("source stage placeholders differ")


def _render_recipe(profile: RecipeProfile) -> dict[str, Any]:
    return {
        "schema_version": RECIPE_V2,
        "stage_width_fields": list(profile.stage_width_fields),
        "group_id_template": profile.group_id_template,
        "artifact_id_template": profile.artifact_id_template,
        "shared_source_path_templates": dict(profile.shared_source_path_templates),
    }


def _validate_recipe(recipe: Mapping[str, Any]) -> None:
    if set(recipe) != _RECIPE_KEYS:
        raise P6HistoryRecipeDerivationError("recipe fields are invalid")
    if (
        recipe.get("schema_version") != RECIPE_V2
        or recipe.get("stage_width_fields") != list(STAGE_WIDTH_FIELDS)
        or recipe.get("group_id_template") != CANONICAL_GROUP_ID_TEMPLATE
        or recipe.get("artifact_id_template") != CANONICAL_ARTIFACT_ID_TEMPLATE
    ):
        raise P6HistoryRecipeDerivationError("recipe surface is incompatible")
    _validate_template(recipe["group_id_template"], set(STAGE_WIDTH_FIELDS))
    _validate_template(recipe["artifact_id_template"], set(STAGE_WIDTH_FIELDS))
    templates = recipe.get("shared_source_path_templates")
    if not isinstance(templates, Mapping) or set(templates) != set(
        SHARED_SOURCE_PATH_KEYS
    ):
        raise P6HistoryRecipeDerivationError("shared source paths are incomplete")
    for template in templates.values():
        _validate_relative_path_template(template)
    _validate_rendered_paths_do_not_collide(recipe, templates)


def _validate_template(template: object, allowed_fields: set[str]) -> None:
    if not isinstance(template, str) or not template:
        raise P6HistoryRecipeDerivationError("recipe template is invalid")
    try:
        parsed = tuple(string.Formatter().parse(template))
    except ValueError as error:
        raise P6HistoryRecipeDerivationError("recipe template is invalid") from error
    fields = set()
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in allowed_fields or format_spec or conversion is not None:
            raise P6HistoryRecipeDerivationError("recipe template field is invalid")
        fields.add(field_name)
    if not fields:
        raise P6HistoryRecipeDerivationError("recipe template has no identity fields")


def _validate_relative_path_template(template: object) -> None:
    _validate_template(template, set(_TEMPLATE_FIELDS))
    assert isinstance(template, str)
    path = Path(template)
    if (
        path.is_absolute()
        or PureWindowsPath(template).is_absolute()
        or "\\" in template
        or ".." in path.parts
    ):
        raise P6HistoryRecipeDerivationError("shared source path is not relative")


def _validate_rendered_paths_do_not_collide(
    recipe: Mapping[str, Any], templates: Mapping[str, Any]
) -> None:
    seen_across_groups: set[str] = set()
    for widths in _SAMPLE_WIDTHS:
        values = dict(zip(STAGE_WIDTH_FIELDS, widths, strict=True))
        group_id = recipe["group_id_template"].format_map(values)
        artifact_id = recipe["artifact_id_template"].format_map(values)
        rendered_paths = {
            template.format_map({**values, "group_id": group_id, "artifact_id": artifact_id})
            for template in templates.values()
        }
        if len(rendered_paths) != len(SHARED_SOURCE_PATH_KEYS):
            raise P6HistoryRecipeDerivationError("shared source paths collide in a group")
        if seen_across_groups.intersection(rendered_paths):
            raise P6HistoryRecipeDerivationError("shared source paths collide across groups")
        seen_across_groups.update(rendered_paths)
