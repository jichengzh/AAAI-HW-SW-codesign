"""Materialize a dynamic P6 Pyramid plan into a Stage5 registry-v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
import string
import tempfile
from typing import Any

from framework.stage5.genome_contract_v1 import (
    canonical_group_id,
    validate_structure_identity,
)
from framework.stage5.production_search_v1 import (
    source_group_q_modes,
    validate_source_contract,
)
from framework.stage6.p6_history_binding_v1 import (
    P6HistoryBindingError,
    validate_history_execution_binding,
)


PLAN_SCHEMA_VERSION = "p6_pyramid_candidate_plan_v2"
BINDING_SCHEMA_VERSION = "p6_history_binding_v1"
REGISTRY_SCHEMA_VERSION = "stage5_candidate_source_registry_v2"
RECIPE_SCHEMA_VERSION = "p6_history_dynamic_materialization_recipe_v1"
ALLOWED_Q_MODES = frozenset({"fp16", "int8"})
STAGE_WIDTH_FIELDS = ("stage1_width", "stage2_width", "stage3_width")
OUTPUT_TEMPLATE_KEYS = (
    "training_path_template",
    "checkpoint_path_template",
    "onnx_path_template",
    "calibration_path_template",
)
RECIPE_KEYS = {
    "schema_version",
    "stage_width_fields",
    "group_id_template",
    "artifact_id_template",
    "output_path_templates_by_q_mode",
}
EXPECTED_BINDING_TARGET = {
    "model": "pyramid",
    "hardware": "h800",
    "backend": "tvm_auto",
}
EXPECTED_PLAN_FIELDS = {
    "schema_version": PLAN_SCHEMA_VERSION,
    "source_schema": "stage2_search_space_v1",
    "target_model": "pyramid",
    "execution_backend": "tvm_auto",
    "candidate_source_mode": "framework_stage2_search_space",
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


def _validate_plan(
    raw_plan: Mapping[str, Any],
) -> dict[tuple[tuple[int, int, int], str], tuple[str, str, str]]:
    if not isinstance(raw_plan, Mapping):
        _invalid("candidate plan must be an object")
    if any(raw_plan.get(key) != value for key, value in EXPECTED_PLAN_FIELDS.items()):
        _invalid("candidate plan contract is incompatible")
    hardware_target = raw_plan.get("hardware_target")
    if not isinstance(hardware_target, str) or not (
        hardware_target == "h800" or hardware_target.startswith("h800_")
    ):
        _invalid("candidate plan hardware is incompatible")
    candidates = raw_plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        _invalid("candidate plan candidates are invalid")

    entries: list[
        tuple[tuple[tuple[int, int, int], str], tuple[str, str, str]]
    ] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            _invalid("candidate plan candidate is invalid")
        width = candidate.get("width")
        q_mode = candidate.get("q_mode")
        source_point_ids = candidate.get("source_point_ids")
        if (
            not isinstance(width, list)
            or len(width) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in width
            )
            or q_mode not in ALLOWED_Q_MODES
            or not isinstance(source_point_ids, list)
            or len(source_point_ids) != 3
            or any(
                not isinstance(source_point_id, str) or not source_point_id.strip()
                for source_point_id in source_point_ids
            )
        ):
            _invalid("candidate plan identity is invalid")
        width_identity = (width[0], width[1], width[2])
        provenance = (
            source_point_ids[0],
            source_point_ids[1],
            source_point_ids[2],
        )
        entries.append(((width_identity, str(q_mode)), provenance))

    mapping = dict(entries)
    if len(mapping) != len(entries):
        _invalid("candidate plan identity is duplicated")
    ordered_entries = sorted(
        entries,
        key=lambda item: (item[0][0], item[0][1], item[1]),
    )
    if entries != ordered_entries:
        _invalid("candidate plan is not canonical")
    candidate_count = raw_plan.get("candidate_count")
    structure_count = raw_plan.get("structure_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != len(entries)
        or isinstance(structure_count, bool)
        or not isinstance(structure_count, int)
        or structure_count != len({identity[0] for identity in mapping})
    ):
        _invalid("candidate plan counts are inconsistent")
    return mapping


def _validate_template(binding: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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
    recipe = _validate_recipe(validated_template.get("dynamic_materialization_recipe"))
    return validated_template, recipe


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
            if key == "dynamic_materialization_recipe":
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
    if not isinstance(raw_recipe, Mapping) or set(raw_recipe) != RECIPE_KEYS:
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


def _render_template(template: str, values: Mapping[str, object]) -> str:
    try:
        rendered = template.format_map(values)
    except (KeyError, ValueError) as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "dynamic materialization template failed"
        ) from error
    if not rendered or any(character in rendered for character in ("\x00", "\r", "\n")):
        _invalid("dynamic materialization template rendered an invalid value")
    return rendered


def _render_output_path(
    template: str, values: Mapping[str, object], local_output_root: Path
) -> Path:
    rendered = _render_template(template, values)
    relative = Path(rendered)
    if relative.is_absolute() or ".." in relative.parts:
        _invalid("dynamic materialization output escapes the allowed root")
    try:
        resolved = (local_output_root / relative).resolve(strict=False)
    except OSError as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "dynamic materialization output is unavailable"
        ) from error
    if (
        resolved == local_output_root
        or not _is_relative_to(resolved, local_output_root)
        or resolved.is_symlink()
        or (resolved.exists() and resolved.is_dir())
    ):
        _invalid("dynamic materialization output escapes the allowed root")
    return resolved


def _materialize_groups(
    plan_mapping: Mapping[
        tuple[tuple[int, int, int], str], tuple[str, str, str]
    ],
    template: Mapping[str, Any],
    recipe: Mapping[str, Any],
    local_output_root: Path,
    registry_output_path: Path,
) -> list[dict[str, Any]]:
    by_width: dict[
        tuple[int, int, int], dict[str, tuple[str, str, str]]
    ] = {}
    for (width, q_mode), source_point_ids in plan_mapping.items():
        by_width.setdefault(width, {})[q_mode] = source_point_ids

    observed_artifacts: set[str] = set()
    observed_outputs = {registry_output_path}
    groups: list[dict[str, Any]] = []
    output_templates_by_q_mode = recipe["output_path_templates_by_q_mode"]
    for width, provenance_by_q_mode in sorted(by_width.items()):
        width_values = dict(zip(STAGE_WIDTH_FIELDS, width, strict=True))
        expected_group_id = canonical_group_id("pyramid", width)
        group_id = _render_template(recipe["group_id_template"], width_values)
        artifact_id = _render_template(recipe["artifact_id_template"], width_values)
        if group_id != expected_group_id or artifact_id in observed_artifacts:
            _invalid("dynamic materialization identity collides or drifts")
        observed_artifacts.add(artifact_id)

        available_q_modes = sorted(provenance_by_q_mode)
        outputs_by_q_mode: dict[str, dict[str, str]] = {}
        for q_mode in available_q_modes:
            raw_output_templates = output_templates_by_q_mode.get(q_mode)
            if not isinstance(raw_output_templates, Mapping):
                _invalid("dynamic materialization output mapping is incomplete")
            render_values = {
                **width_values,
                "group_id": group_id,
                "artifact_id": artifact_id,
                "q_mode": q_mode,
            }
            rendered_outputs: dict[str, str] = {}
            for template_key in OUTPUT_TEMPLATE_KEYS:
                raw_path_template = raw_output_templates.get(template_key)
                if not isinstance(raw_path_template, str):
                    _invalid("dynamic materialization output mapping is incomplete")
                resolved_output = _render_output_path(
                    raw_path_template, render_values, local_output_root
                )
                if resolved_output in observed_outputs:
                    _invalid("dynamic materialization output path collides")
                observed_outputs.add(resolved_output)
                rendered_outputs[template_key.removesuffix("_template")] = str(
                    resolved_output
                )
            outputs_by_q_mode[q_mode] = rendered_outputs

        contract = copy.deepcopy(dict(template))
        contract.pop("dynamic_materialization_recipe", None)
        contract.update(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": list(width),
                "artifact_id": artifact_id,
                "stage_widths": width_values,
                "materialization_outputs_by_q_mode": outputs_by_q_mode,
            }
        )
        evidence_sha = contract["source_evidence_sha256"]
        group = {
            "group_id": group_id,
            "model": "pyramid",
            "width": list(width),
            "source_status": contract["source_status"],
            "source_evidence_sha256": evidence_sha,
            "source_contract": contract,
            "source_contract_sha256": _canonical_sha(contract),
            "materialization_kind": "local_pyramid_tvm",
            "available_q_modes": available_q_modes,
            "source_point_ids_by_q_mode": {
                q_mode: list(provenance_by_q_mode[q_mode])
                for q_mode in available_q_modes
            },
            "graph_features": {
                "group_id": group_id,
                "model": "pyramid",
                "width": list(width),
                **width_values,
            },
        }
        try:
            validate_structure_identity(group)
            validate_source_contract(group)
            source_group_q_modes(REGISTRY_SCHEMA_VERSION, group)
        except (TypeError, ValueError) as error:
            raise P6HistoryRegistryError(
                "source_registry_invalid", "derived source contract is invalid"
            ) from error
        groups.append(group)
    return groups


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
        local_output_root = _resolve_existing_root(
            local_registry_root, "local output root"
        )
        output_path = _resolve_registry_output(registry_output_path, local_output_root)
        plan_mapping = _validate_plan(plan)
        template, recipe = _validate_template(binding)
        groups = _materialize_groups(
            plan_mapping,
            template,
            recipe,
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
