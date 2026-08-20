"""Public, versioned P6 procedural recipe profiles.

Profiles deliberately describe only public interface evidence.  Private source
contracts provide concrete checkpoints, inputs, and training configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


RECIPE_V2 = "p6_history_dynamic_materialization_recipe_v2"
PROFILE_V1 = "p6_stage5_pyramid_h800_tvm_profile_v1"
STAGE_WIDTH_FIELDS = ("stage1_width", "stage2_width", "stage3_width")
SHARED_SOURCE_PATH_KEYS = (
    "checkpoint_path",
    "checkpoint_dir",
    "config_path",
    "training_done_marker",
    "onnx_path",
    "onnx_report_path",
    "calibration_root",
    "calibration_npz",
    "calibration_summary",
    "trt_calibration_dir",
    "source_done_marker",
)
CANONICAL_GROUP_ID_TEMPLATE = "pyramid|{stage1_width}x{stage2_width}x{stage3_width}"
CANONICAL_ARTIFACT_ID_TEMPLATE = "pyramid-{stage1_width}-{stage2_width}-{stage3_width}"


@dataclass(frozen=True)
class RecipeProfile:
    """Immutable public evidence used to derive one recipe-v2 shape."""

    profile_id: str
    target: Mapping[str, str]
    component_markers: Mapping[str, str]
    execution_stages: tuple[str, ...]
    stage_width_fields: tuple[str, ...]
    q_modes: tuple[str, ...]
    group_id_template: str
    artifact_id_template: str
    shared_source_path_templates: Mapping[str, str]
    source_materializer_placeholders: tuple[str, ...]
    source_materializer_invocation_flags: tuple[str, ...]


def _frozen_mapping(raw: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(raw))


def recipe_profile_from_mapping(raw: Mapping[str, Any]) -> RecipeProfile:
    """Construct an immutable profile from a public-profile serialization."""
    if not isinstance(raw, Mapping):
        raise ValueError("profile must be a mapping")
    required_keys = {
        "profile_id",
        "target",
        "component_markers",
        "execution_stages",
        "stage_width_fields",
        "q_modes",
        "group_id_template",
        "artifact_id_template",
        "shared_source_path_templates",
        "source_materializer_placeholders",
        "source_materializer_invocation_flags",
    }
    if set(raw) != required_keys:
        raise ValueError("profile fields are invalid")
    target = raw["target"]
    markers = raw["component_markers"]
    templates = raw["shared_source_path_templates"]
    sequences = (
        raw["execution_stages"],
        raw["stage_width_fields"],
        raw["q_modes"],
        raw["source_materializer_placeholders"],
        raw["source_materializer_invocation_flags"],
    )
    if (
        not isinstance(raw["profile_id"], str)
        or not isinstance(raw["group_id_template"], str)
        or not isinstance(raw["artifact_id_template"], str)
        or not all(isinstance(value, Mapping) for value in (target, markers, templates))
        or not all(isinstance(value, list) for value in sequences)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for mapping in (target, markers, templates)
            for key, value in mapping.items()
        )
        or any(
            not all(isinstance(item, str) for item in sequence)
            for sequence in sequences
        )
    ):
        raise ValueError("profile values are invalid")
    return RecipeProfile(
        profile_id=raw["profile_id"],
        target=_frozen_mapping(target),
        component_markers=_frozen_mapping(markers),
        execution_stages=tuple(raw["execution_stages"]),
        stage_width_fields=tuple(raw["stage_width_fields"]),
        q_modes=tuple(raw["q_modes"]),
        group_id_template=raw["group_id_template"],
        artifact_id_template=raw["artifact_id_template"],
        shared_source_path_templates=_frozen_mapping(templates),
        source_materializer_placeholders=tuple(raw["source_materializer_placeholders"]),
        source_materializer_invocation_flags=tuple(
            raw["source_materializer_invocation_flags"]
        ),
    )


def serialize_recipe_profile(profile: RecipeProfile) -> dict[str, Any]:
    """Return a detached serialization suitable for inspection or persistence."""
    return {
        "profile_id": profile.profile_id,
        "target": dict(profile.target),
        "component_markers": dict(profile.component_markers),
        "execution_stages": list(profile.execution_stages),
        "stage_width_fields": list(profile.stage_width_fields),
        "q_modes": list(profile.q_modes),
        "group_id_template": profile.group_id_template,
        "artifact_id_template": profile.artifact_id_template,
        "shared_source_path_templates": dict(profile.shared_source_path_templates),
        "source_materializer_placeholders": list(
            profile.source_materializer_placeholders
        ),
        "source_materializer_invocation_flags": list(
            profile.source_materializer_invocation_flags
        ),
    }


_PROFILE_V1 = recipe_profile_from_mapping(
    {
        "profile_id": PROFILE_V1,
        "target": {"model": "pyramid", "hardware": "h800", "backend": "tvm_auto"},
        "component_markers": {
            "controller": "stage5_task_round_controller_v3.sh",
            "source_materializer": "stage5_materialize_round_sources_v1.sh",
            "performance_plan": "stage5_build_performance_plan_v2.py",
            "finalizer": "stage5_finalize_feedback_v2.py",
        },
        "execution_stages": [
            "source_materialization",
            "quantization",
            "performance",
            "ap",
            "finalization",
        ],
        "stage_width_fields": list(STAGE_WIDTH_FIELDS),
        "q_modes": ["fp16", "int8"],
        "group_id_template": CANONICAL_GROUP_ID_TEMPLATE,
        "artifact_id_template": CANONICAL_ARTIFACT_ID_TEMPLATE,
        "shared_source_path_templates": {
            "checkpoint_path": "materialized/{artifact_id}/checkpoint/model.ckpt",
            "checkpoint_dir": "materialized/{artifact_id}/checkpoint",
            "config_path": "materialized/{artifact_id}/config/source-config.json",
            "training_done_marker": "materialized/{artifact_id}/markers/training.done",
            "onnx_path": "materialized/{artifact_id}/onnx/model.onnx",
            "onnx_report_path": "materialized/{artifact_id}/onnx/report.json",
            "calibration_root": "materialized/{artifact_id}/calibration",
            "calibration_npz": "materialized/{artifact_id}/calibration/cache.npz",
            "calibration_summary": "materialized/{artifact_id}/calibration/summary.json",
            "trt_calibration_dir": "materialized/{artifact_id}/trt-calibration",
            "source_done_marker": "materialized/{artifact_id}/markers/source.done",
        },
        "source_materializer_placeholders": [
            "{measurement_request}",
            "{round_output_root}",
        ],
        "source_materializer_invocation_flags": [
            "--request",
            "--model",
            "--group-id",
            "--gpu",
        ],
    }
)
RECIPE_PROFILES: Mapping[str, RecipeProfile] = MappingProxyType({PROFILE_V1: _PROFILE_V1})


def get_recipe_profile(profile_id: object) -> RecipeProfile | None:
    """Return the immutable public profile selected by an ignored source map."""
    if not isinstance(profile_id, str):
        return None
    return RECIPE_PROFILES.get(profile_id)
