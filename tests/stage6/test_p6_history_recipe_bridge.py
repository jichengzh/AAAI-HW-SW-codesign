"""Behavioral coverage for the P6 procedural recipe-v2 renderer."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import traceback
from typing import Any, Callable, Mapping

import pytest
import yaml

from framework.stage6 import p6_history_recipe_bridge_v1 as bridge
from framework.stage6.p6_history_recipe_profiles_v1 import (
    PROFILE_V1,
    RECIPE_V2,
    SHARED_SOURCE_PATH_KEYS,
    get_recipe_profile,
    recipe_profile_from_mapping,
    serialize_recipe_profile,
)


MARKERS = {
    "controller": "stage5_task_round_controller_v3.sh",
    "source_materializer": "stage5_materialize_round_sources_v1.sh",
    "performance_plan": "stage5_build_performance_plan_v2.py",
    "finalizer": "stage5_finalize_feedback_v2.py",
}
STAGES = (
    "source_materialization",
    "quantization",
    "performance",
    "ap",
    "finalization",
)


def _write_yaml(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic executable\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _runner_template() -> dict[str, Any]:
    return {
        "schema_version": "p6_history_runner_template_v1",
        "stage1_scan": {
            "name": "build_stage1_partition",
            "argv": [
                "private-runner/bin/scan-private",
                "{stage1_partition_manifest}",
                "{local_output_root}",
            ],
        },
        "execution_interface": {
            "schema_version": "p6_history_runner_interface_v1",
            "controller": {
                "argv": [f"documented-stage5-chain/{MARKERS['controller']}"]
            },
            "execution_chain": [
                {
                    "stage": "source_materialization",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['source_materializer']}",
                        "{measurement_request}",
                        "{round_output_root}",
                    ],
                },
                {
                    "stage": "quantization",
                    "argv": [
                        "private-runner/bin/quantize-private",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                },
                {
                    "stage": "performance",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['performance_plan']}",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                },
                {
                    "stage": "ap",
                    "argv": [
                        "private-runner/bin/measure-ap-private",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                },
                {
                    "stage": "finalization",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['finalizer']}",
                        "{measurement_request}",
                        "{task_state}",
                        "{actual_feedback}",
                        "{actual_receipt}",
                        "{finalization_barrier}",
                        "{round_output_root}",
                    ],
                },
            ],
            "environment": {
                "values": {},
                "activation_argv": ["private-runner/bin/activate-private"],
            },
            "output_layout": {},
            "actual_feedback": {},
        },
    }


def _write_valid_template(tmp_path: Path) -> tuple[Path, Path]:
    history_root = tmp_path / "private-history"
    history_root.mkdir()
    subprocess.run(["git", "init", "-q", str(history_root)], check=True)
    for marker in MARKERS.values():
        _write_executable(history_root / "documented-stage5-chain" / marker)
    for name in (
        "scan-private",
        "quantize-private",
        "measure-ap-private",
        "activate-private",
    ):
        _write_executable(history_root / "private-runner" / "bin" / name)
    return (
        _write_yaml(tmp_path / "private-inputs" / "runner-template.yaml", _runner_template()),
        history_root,
    )


def _source_map(**extra: object) -> dict[str, Any]:
    return {"procedural_recipe_profile": PROFILE_V1, **extra}


def _derive(tmp_path: Path, source_map: Mapping[str, Any] | None = None) -> dict[str, Any]:
    template, history_root = _write_valid_template(tmp_path)
    return bridge.derive_dynamic_recipe_from_procedural_source(
        source_map=_source_map() if source_map is None else source_map,
        runner_template_path=template,
        history_root=history_root,
    )


def _invalid_profile(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    profile = get_recipe_profile(PROFILE_V1)
    assert profile is not None
    serialized = serialize_recipe_profile(profile)
    mutate(serialized)
    monkeypatch.setattr(
        bridge,
        "get_recipe_profile",
        lambda _profile_id: recipe_profile_from_mapping(serialized),
    )


def test_known_profile_derives_shared_recipe_without_runner_execution(tmp_path: Path) -> None:
    """Catches returning a q-mode-specific recipe or invoking a history program."""
    recipe = _derive(tmp_path)

    assert recipe == {
        "schema_version": RECIPE_V2,
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
        "group_id_template": "pyramid|{stage1_width}x{stage2_width}x{stage3_width}",
        "artifact_id_template": "pyramid-{stage1_width}-{stage2_width}-{stage3_width}",
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
    }
    assert "q_mode" not in json.dumps(recipe, sort_keys=True)


def test_unknown_profile_is_a_stable_derivation_failure(tmp_path: Path) -> None:
    """Catches accepting a source map that selects an undocumented profile."""
    with pytest.raises(bridge.P6HistoryRecipeDerivationError) as captured:
        _derive(tmp_path, _source_map(procedural_recipe_profile="unknown-profile"))

    assert captured.value.category == "history_recipe_derivation_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["component_markers"].pop("finalizer"),
        lambda payload: payload["execution_stages"].pop(),
        lambda payload: payload["shared_source_path_templates"].pop("onnx_path"),
        lambda payload: payload["shared_source_path_templates"].update({"extra": "x/{artifact_id}"}),
        lambda payload: payload["shared_source_path_templates"].update({"onnx_path": "/absolute"}),
        lambda payload: payload["shared_source_path_templates"].update({"onnx_path": "../escape"}),
        lambda payload: payload["shared_source_path_templates"].update({"onnx_path": "x/{q_mode}/model.onnx"}),
        lambda payload: payload["shared_source_path_templates"].update({"onnx_path": payload["shared_source_path_templates"]["checkpoint_path"]}),
        lambda payload: payload["shared_source_path_templates"].update({"onnx_path": "constant/onnx.onnx"}),
        lambda payload: payload["shared_source_path_templates"].update({"onnx_path": "same/{stage1_width}/model.onnx"}),
        lambda payload: payload.update({"q_modes": ["int8", "fp16"]}),
        lambda payload: payload.update({"source_materializer_invocation_flags": ["--model", "--request", "--group-id", "--gpu"]}),
    ],
)
def test_profile_recipe_drift_is_a_stable_derivation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    """Catches profile drift in evidence, template shape, or group-safe paths."""
    _invalid_profile(monkeypatch, mutation)

    with pytest.raises(bridge.P6HistoryRecipeDerivationError) as captured:
        _derive(tmp_path)

    assert captured.value.category == "history_recipe_derivation_invalid"


def test_runner_validation_error_does_not_expose_private_traceback_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches preserving a private validator error as an exception cause."""
    sentinel = "/synthetic-private-root/secret-runner-template.yaml"

    def fail_validation(_template: Path, _root: Path) -> None:
        raise bridge.RunnerTemplateValidationError(
            "execution_interface_unavailable", sentinel
        )

    monkeypatch.setattr(bridge, "validate_pre_provision_runner_template", fail_validation)

    with pytest.raises(bridge.P6HistoryRecipeDerivationError) as captured:
        _derive(tmp_path)

    rendered = "".join(
        traceback.format_exception(
            captured.type, captured.value, captured.tb, chain=True
        )
    )
    assert captured.value.category == "history_recipe_derivation_invalid"
    assert captured.value.detail == "runner template validation failed"
    assert sentinel not in rendered


@pytest.mark.parametrize("role", tuple(MARKERS))
def test_runner_marker_drift_is_a_stable_derivation_failure(
    tmp_path: Path, role: str
) -> None:
    """Catches allowing a runner marker to diverge from the public evidence."""
    template, history_root = _write_valid_template(tmp_path)
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    if role == "controller":
        argv = payload["execution_interface"]["controller"]["argv"]
    else:
        stage = next(
            entry
            for entry in payload["execution_interface"]["execution_chain"]
            if entry["stage"]
            == {
                "source_materializer": "source_materialization",
                "performance_plan": "performance",
                "finalizer": "finalization",
            }[role]
        )
        argv = stage["argv"]
    argv[0] = "private-runner/bin/drifted-component"
    _write_yaml(template, payload)

    with pytest.raises(bridge.P6HistoryRecipeDerivationError) as captured:
        bridge.derive_dynamic_recipe_from_procedural_source(
            source_map=_source_map(),
            runner_template_path=template,
            history_root=history_root,
        )

    assert captured.value.category == "history_recipe_derivation_invalid"


@pytest.mark.parametrize(
    "source_argv",
    [
        ["{round_output_root}", "{measurement_request}"],
        ["{measurement_request}", "{round_output_root}", "{measurement_request}"],
        ["{measurement_request}", "{round_output_root}", "{unexpected}"],
    ],
)
def test_source_stage_placeholder_surface_must_match_exactly(
    tmp_path: Path, source_argv: list[str]
) -> None:
    """Catches accepting reordered, repeated, or extra source-stage placeholders."""
    template, history_root = _write_valid_template(tmp_path)
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    source_stage = payload["execution_interface"]["execution_chain"][0]
    source_stage["argv"][1:] = source_argv
    _write_yaml(template, payload)

    with pytest.raises(bridge.P6HistoryRecipeDerivationError) as captured:
        bridge.derive_dynamic_recipe_from_procedural_source(
            source_map=_source_map(),
            runner_template_path=template,
            history_root=history_root,
        )

    assert captured.value.category == "history_recipe_derivation_invalid"


@pytest.mark.parametrize(
    "self_report",
    [
        {"capability": "training"},
        {"argv": ["made-up"]},
        {"marker": "made-up"},
        {"output_paths": {"checkpoint_path": "made-up"}},
        {"checkpoint_path": "made-up"},
        {"base_checkpoint": "made-up"},
        {"training_hyperparameter": "made-up"},
        {"gpu": "made-up"},
    ],
)
def test_source_map_cannot_self_report_semantic_evidence(
    tmp_path: Path, self_report: Mapping[str, Any]
) -> None:
    """Catches trusting semantic evidence supplied by the ignored source map."""
    with pytest.raises(bridge.P6HistoryRecipeDerivationError) as captured:
        _derive(tmp_path, _source_map(**self_report))

    assert captured.value.category == "history_recipe_derivation_invalid"


def test_public_profile_contains_only_documented_public_surface() -> None:
    """Catches publishing invented component roles or private static configuration."""
    profile = get_recipe_profile(PROFILE_V1)
    assert profile is not None
    serialized = json.dumps(serialize_recipe_profile(profile), sort_keys=True)

    assert set(profile.shared_source_path_templates) == set(SHARED_SOURCE_PATH_KEYS)
    for forbidden in (
        "training_component",
        "checkpoint_export",
        "onnx_export",
        "int8_calibration",
        "private-gpu",
        "base_checkpoint",
        "training_hyperparameter",
    ):
        assert forbidden not in serialized
