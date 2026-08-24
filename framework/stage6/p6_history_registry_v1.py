"""Materialize a dynamic P6 Pyramid plan into a Stage5 registry-v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import string
import subprocess
import tempfile
from typing import Any

from framework.stage5.production_search_v1 import (
    validate_source_contract,
)
from framework.stage6.p6_history_binding_v1 import (
    P6HistoryBindingError,
    validate_history_execution_binding,
)
from framework.stage6.p6_history_recipe_profiles_v1 import (
    RECIPE_V2,
    SHARED_SOURCE_PATH_KEYS,
)
from framework.stage6.p6_formal_plan_contract_v1 import (
    formal_base_widths as candidate_plan_formal_base_widths,
    materialize_p6_history_groups,
    recipe_v2_base_stage_widths,
    validate_p6_candidate_plan,
)
from framework.stage6.p6_history_training_contract_v1 import (
    P6HistoryTrainingContractError,
    validate_recipe_v2_training_template,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BINDING_SCHEMA_VERSION = "p6_history_binding_v1"
REGISTRY_SCHEMA_VERSION = "stage5_candidate_source_registry_v2"
RECIPE_V1 = "p6_history_dynamic_materialization_recipe_v1"
ALLOWED_Q_MODES = frozenset({"fp16", "int8"})
STAGE_WIDTH_FIELDS = ("stage1_width", "stage2_width", "stage3_width")
OUTPUT_TEMPLATE_KEYS = (
    "training_path_template",
    "checkpoint_path_template",
    "onnx_path_template",
    "calibration_path_template",
)
RECIPE_V1_KEYS = {
    "schema_version",
    "stage_width_fields",
    "group_id_template",
    "artifact_id_template",
    "output_path_templates_by_q_mode",
}
RECIPE_V2_KEYS = {
    "schema_version",
    "stage_width_fields",
    "group_id_template",
    "artifact_id_template",
    "shared_source_path_templates",
}
EXPECTED_BINDING_TARGET = {
    "model": "pyramid",
    "hardware": "h800",
    "backend": "tvm_auto",
}
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


class P6HistoryRegistryError(ValueError):
    """Stable categorized failure for dynamic history materialization."""

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")


@dataclass(frozen=True)
class ValidatedSourceTemplate:
    """Detached template and its authoritative private history root."""

    private_root: Path
    template: Mapping[str, Any]
    recipe: Mapping[str, Any]


def _invalid(detail: str) -> None:
    raise P6HistoryRegistryError("source_registry_invalid", detail)


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
        raise P6HistoryRegistryError(
            "source_registry_invalid", "source contract is not canonical JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_existing_root(raw_path: object, label: str) -> Path:
    try:
        path = Path(raw_path)  # type: ignore[arg-type]
    except TypeError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", f"{label} is invalid"
        ) from error
    if not path.is_absolute() or path.is_symlink():
        _invalid(f"{label} is invalid")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", f"{label} is unavailable"
        ) from error
    if not resolved.is_dir():
        _invalid(f"{label} is invalid")
    return resolved


def _resolve_registry_output(
    raw_path: str | Path | None, local_output_root: Path
) -> Path:
    path = (
        local_output_root / "source_registry.json"
        if raw_path is None
        else Path(raw_path)
    )
    if not path.is_absolute() or path.is_symlink():
        _invalid("source registry destination is invalid")
    try:
        parent = path.parent.resolve(strict=True)
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "source registry destination is unavailable"
        ) from error
    if (
        not parent.is_dir()
        or not _is_relative_to(resolved, local_output_root)
        or resolved == local_output_root
        or (resolved.exists() and not resolved.is_file())
    ):
        _invalid("source registry destination is invalid")
    return resolved


def _registry_output_is_git_ignored(repository: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repository)
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(relative)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
        )
    except (OSError, ValueError):
        return False
    return completed.returncode == 0


def _require_private_registry_output(path: Path) -> None:
    try:
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "source registry destination is unavailable"
        ) from error
    if _is_relative_to(path, repository) and not _registry_output_is_git_ignored(
        repository, path
    ):
        _invalid("source registry destination is invalid")


def _validate_plan(
    raw_plan: Mapping[str, Any],
) -> dict[tuple[tuple[int, int, int], str], tuple[str, str, str]]:
    try:
        return validate_p6_candidate_plan(raw_plan)
    except ValueError as error:
        _invalid(str(error))


def _validate_formal_base_matches_binding(
    raw_plan: Mapping[str, Any],
    template: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> None:
    formal_base_widths = candidate_plan_formal_base_widths(raw_plan)
    binding_base_widths = recipe_v2_base_stage_widths(
        template, str(recipe.get("schema_version"))
    )
    if (
        formal_base_widths is not None
        and binding_base_widths is not None
        and formal_base_widths != binding_base_widths
    ):
        _invalid("formal axis base width differs from binding base width")


def _validate_template(binding: Mapping[str, Any]) -> ValidatedSourceTemplate:
    private_root, template = _validate_template_binding(binding)
    validated_template = _validate_source_contract_template(template, binding)
    recipe = _validate_recipe(validated_template.get("dynamic_materialization_recipe"))
    if recipe.get("schema_version") == RECIPE_V2:
        validated_template = _validate_recipe_v2_template(
            validated_template, private_root
        )
    return ValidatedSourceTemplate(
        private_root=private_root,
        template=validated_template,
        recipe=recipe,
    )


def _validate_template_binding(
    binding: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping):
        _invalid("history binding must be an object")
    try:
        validate_history_execution_binding(binding)
    except P6HistoryBindingError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "history execution interface is invalid"
        ) from error
    if (
        binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or binding.get("target") != EXPECTED_BINDING_TARGET
        or binding.get("status") != "validated"
    ):
        _invalid("history binding contract is incompatible")
    private_root = _resolve_existing_root(
        binding.get("private_root", ""), "history binding root"
    )
    raw_template = binding.get("source_contract_template")
    if not isinstance(raw_template, Mapping):
        _invalid("source contract template is missing")
    template = copy.deepcopy(dict(raw_template))
    _validate_no_forbidden_context(template)
    _validate_legacy_paths(template, private_root)
    return private_root, template


def _validate_source_contract_template(
    template: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_sha = template.get("source_evidence_sha256")
    if not _is_sha256(evidence_sha):
        _invalid("source evidence SHA256 is invalid")
    template_sha = _canonical_sha(template)
    expected_template_sha = binding.get("source_contract_template_sha256")
    if expected_template_sha is not None and (
        not _is_sha256(expected_template_sha)
        or expected_template_sha != template_sha
    ):
        _invalid("source contract template SHA256 is inconsistent")
    expected_evidence_sha = binding.get("source_evidence_sha256")
    if expected_evidence_sha is not None and expected_evidence_sha != evidence_sha:
        _invalid("source evidence SHA256 is inconsistent")
    template_group = {
        "group_id": template.get("group_id"),
        "model": template.get("model"),
        "width": copy.deepcopy(template.get("width")),
        "source_status": template.get("source_status"),
        "source_evidence_sha256": evidence_sha,
        "source_contract": template,
        "source_contract_sha256": template_sha,
    }
    try:
        validated_template = validate_source_contract(template_group)
    except (TypeError, ValueError) as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "source contract template is invalid"
        ) from error
    if (
        validated_template.get("model") != "pyramid"
        or validated_template.get("source_status") not in {"ready", "materializable"}
    ):
        _invalid("source contract template is incompatible")
    return validated_template


def _validate_recipe_v2_template(
    template: Mapping[str, Any],
    private_root: Path,
) -> dict[str, Any]:
    try:
        return validate_recipe_v2_training_template(
            template,
            private_root=private_root,
        )
    except P6HistoryTrainingContractError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "recipe-v2 training template is invalid"
        ) from error


def _validate_no_forbidden_context(value: object) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key != "source_status" and (
                key == "status"
                or key.endswith("_status")
                or any(token in key for token in FORBIDDEN_CONTEXT_TOKENS)
            ):
                _invalid("source contract template contains forbidden result context")
            _validate_no_forbidden_context(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _validate_no_forbidden_context(child)


def _validate_legacy_paths(
    value: object,
    private_root: Path,
    *,
    path_context: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in {
                "dynamic_materialization_recipe",
                "external_training_binding",
            }:
                continue
            _validate_legacy_paths(
                child,
                private_root,
                path_context=path_context or "path" in key.lower(),
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _validate_legacy_paths(
                child,
                private_root,
                path_context=path_context,
            )
        return
    if not path_context:
        return
    if not isinstance(value, str):
        _invalid("legacy source path is invalid")
    try:
        path = Path(value)
        resolved = path.resolve(strict=False)
    except (OSError, TypeError) as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "legacy source path is invalid"
        ) from error
    if not path.is_absolute() or not _is_relative_to(resolved, private_root):
        _invalid("legacy source path escapes the history root")


def _validate_recipe(raw_recipe: object) -> dict[str, Any]:
    if not isinstance(raw_recipe, Mapping):
        _invalid("dynamic materialization recipe is missing or invalid")
    recipe = copy.deepcopy(dict(raw_recipe))
    schema_version = recipe.get("schema_version")
    expected_keys = RECIPE_V2_KEYS if schema_version == RECIPE_V2 else RECIPE_V1_KEYS
    if (
        set(recipe) != expected_keys
        or schema_version not in {RECIPE_V1, RECIPE_V2}
        or recipe.get("stage_width_fields") != list(STAGE_WIDTH_FIELDS)
    ):
        _invalid("dynamic materialization recipe is incompatible")
    for name in ("group_id_template", "artifact_id_template"):
        _validate_format_template(recipe.get(name), set(STAGE_WIDTH_FIELDS))
    if schema_version == RECIPE_V2:
        raw_shared = recipe.get("shared_source_path_templates")
        if not isinstance(raw_shared, Mapping) or set(raw_shared) != set(
            SHARED_SOURCE_PATH_KEYS
        ):
            _invalid("dynamic materialization shared paths are incomplete")
        allowed_shared_fields = {*STAGE_WIDTH_FIELDS, "group_id", "artifact_id"}
        for key in SHARED_SOURCE_PATH_KEYS:
            _validate_format_template(raw_shared.get(key), allowed_shared_fields)
        return recipe
    raw_outputs = recipe.get("output_path_templates_by_q_mode")
    if (
        not isinstance(raw_outputs, Mapping)
        or not raw_outputs
        or not set(raw_outputs) <= ALLOWED_Q_MODES
    ):
        _invalid("dynamic materialization output mapping is invalid")
    allowed_output_fields = {
        *STAGE_WIDTH_FIELDS,
        "group_id",
        "artifact_id",
        "q_mode",
    }
    for q_mode, raw_templates in raw_outputs.items():
        if (
            not isinstance(q_mode, str)
            or not isinstance(raw_templates, Mapping)
            or set(raw_templates) != set(OUTPUT_TEMPLATE_KEYS)
        ):
            _invalid("dynamic materialization output mapping is incomplete")
        for key in OUTPUT_TEMPLATE_KEYS:
            _validate_format_template(raw_templates.get(key), allowed_output_fields)
    return recipe


def _validate_format_template(raw_template: object, allowed_fields: set[str]) -> None:
    if not isinstance(raw_template, str) or not raw_template.strip():
        _invalid("dynamic materialization template is invalid")
    observed_fields: set[str] = set()
    try:
        parsed = tuple(string.Formatter().parse(raw_template))
    except ValueError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "dynamic materialization template is invalid"
        ) from error
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if (
            field_name not in allowed_fields
            or format_spec
            or conversion is not None
        ):
            _invalid("dynamic materialization template field is invalid")
        observed_fields.add(field_name)
    if not observed_fields:
        _invalid("dynamic materialization template has no authoritative fields")


def _materialize_groups(
    plan_mapping: Mapping[tuple[tuple[int, int, int], str], tuple[str, str, str]],
    template: Mapping[str, Any],
    recipe: Mapping[str, Any],
    private_root: Path,
    local_output_root: Path,
    registry_output_path: Path,
) -> list[dict[str, Any]]:
    try:
        return materialize_p6_history_groups(
            plan_mapping,
            template,
            recipe,
            private_root,
            local_output_root,
            registry_output_path,
        )
    except ValueError as error:
        _invalid(str(error))


def _registry_identity_map(
    registry: Mapping[str, Any],
) -> dict[tuple[tuple[int, int, int], str], tuple[str, str, str]]:
    entries: list[
        tuple[tuple[tuple[int, int, int], str], tuple[str, str, str]]
    ] = []
    raw_groups = registry.get("groups")
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        _invalid("derived registry groups are invalid")
    for group in raw_groups:
        if not isinstance(group, Mapping):
            _invalid("derived registry group is invalid")
        width = group.get("width")
        q_modes = group.get("available_q_modes")
        provenance = group.get("source_point_ids_by_q_mode")
        if (
            not isinstance(width, list)
            or len(width) != 3
            or not isinstance(q_modes, list)
            or not isinstance(provenance, Mapping)
        ):
            _invalid("derived registry identity is invalid")
        width_identity = (int(width[0]), int(width[1]), int(width[2]))
        for q_mode in q_modes:
            source_point_ids = provenance.get(q_mode)
            if not isinstance(source_point_ids, list) or len(source_point_ids) != 3:
                _invalid("derived registry provenance is invalid")
            entries.append(
                (
                    (width_identity, str(q_mode)),
                    (
                        str(source_point_ids[0]),
                        str(source_point_ids[1]),
                        str(source_point_ids[2]),
                    ),
                )
            )
    mapping = dict(entries)
    if len(mapping) != len(entries):
        _invalid("derived registry identity collides")
    return mapping


def _atomic_write_registry(path: Path, registry: Mapping[str, Any]) -> None:
    try:
        serialized = (
            json.dumps(
                registry,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "derived registry is not canonical JSON"
        ) from error
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "source registry persistence failed"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def materialize_history_registry(
    plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    local_registry_root: str | Path,
    registry_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, fully materialize, and atomically persist one dynamic registry."""
    try:
        local_output_root = _resolve_existing_root(local_registry_root, "local output root")
        output_path = _resolve_registry_output(registry_output_path, local_output_root)
        _require_private_registry_output(output_path)
        plan_mapping = _validate_plan(plan)
        validated = _validate_template(binding)
        _validate_formal_base_matches_binding(
            plan,
            validated.template,
            validated.recipe,
        )
        groups = _materialize_groups(
            plan_mapping,
            validated.template,
            validated.recipe,
            validated.private_root,
            local_output_root,
            output_path,
        )
        registry = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "groups": groups,
        }
        if _registry_identity_map(registry) != plan_mapping:
            _invalid("derived registry identities do not match the candidate plan")
        _atomic_write_registry(output_path, registry)
        return copy.deepcopy(registry)
    except P6HistoryRegistryError:
        raise
    except (OSError, OverflowError, TypeError, ValueError) as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "source registry input is invalid"
        ) from error
