from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import pytest
import yaml

import framework.stage6.p6_history_normalization_staging_v1 as normalization_staging
from framework.stage6.p6_history_normalization_v1 import (
    P6HistoryNormalizationError,
    normalize_history_inputs,
)
from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_history_execution_closure_v1 import EXECUTION_CLOSURE_ROLES
from framework.stage6.p6_history_recipe_profiles_v1 import PROFILE_V1
from tests.p6_source_wrapper_support import source_bridge_output_paths, source_bridge_request
from tests.stage6.test_p6_history_recipe_bridge import (
    MARKERS,
    _runner_template,
)
from tests.stage6.test_coptv2x_h800_search import (
    _write_runner_template as _write_full_runner_template,
)


INPUT_NAMES = (
    "gold176_rows",
    "gold176_graph_features",
    "capability_profiles",
    "closure",
)


def _sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return str(path)


def _recipe() -> dict[str, Any]:
    templates = {
        q_mode: {
            name: (
                f"materialized/{{group_id}}/{{q_mode}}/{name.removesuffix('_path_template')}.json"
            )
            for name in (
                "training_path_template",
                "checkpoint_path_template",
                "onnx_path_template",
                "calibration_path_template",
            )
        }
        for q_mode in ("fp16", "int8")
    }
    return {
        "schema_version": "p6_history_dynamic_materialization_recipe_v1",
        "stage_width_fields": ["stage1_width", "stage2_width", "stage3_width"],
        "group_id_template": "pyramid|{stage1_width}x{stage2_width}x{stage3_width}",
        "artifact_id_template": "pyramid-{stage1_width}-{stage2_width}-{stage3_width}",
        "output_path_templates_by_q_mode": templates,
    }


def _recipe_v2() -> dict[str, Any]:
    return {
        "schema_version": "p6_history_dynamic_materialization_recipe_v2",
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


def _tree_sha(root: Path) -> str:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append({"kind": "directory", "path": relative})
        else:
            payload = path.read_bytes()
            entries.append(
                {
                    "kind": "regular_file",
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
    return hashlib.sha256(
        json.dumps(entries, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _runtime_external_and_closure(
    source_map: dict[str, Any], tmp_path: Path
) -> tuple[dict[str, Any], Path]:
    payload = copy.deepcopy(source_map)
    history_root = Path(payload["history_root"])
    for marker in MARKERS.values():
        _write_executable(history_root / "documented-stage5-chain" / marker)
    for name in ("scan-private", "quantize-private", "measure-ap-private", "activate-private"):
        _write_executable(history_root / "private-runner" / "bin" / name)
    closure_root = history_root / "execution-source"
    role_names = {
        "stage1_scan": "scan",
        "controller": "stage5_task_round_controller_v3.sh",
        "source_materializer": "stage5_materialize_round_sources_v1.sh",
        "quantization": "quantize",
        "performance": "stage5_build_performance_plan_v2.py",
        "ap": "ap",
        "finalization": "stage5_finalize_feedback_v2.py",
        "activation": "activate",
    }
    role_files = {role: closure_root / name for role, name in role_names.items()}
    project_python = tmp_path / "project-env/bin/python3.9"
    project_python.parent.mkdir(parents=True, exist_ok=True)
    project_python.write_text(
        "#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n", encoding="utf-8"
    )
    project_python.chmod(0o700)
    launcher = project_python.parent / "python"
    if not launcher.exists():
        launcher.symlink_to(project_python.name)
    for role, path in role_files.items():
        body = (
            f"""#!/bin/sh
PY=${{PY:-{project_python}}}
exec "$PY" - "$@" <<'PYCODE'
import json
from pathlib import Path
import sys

request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
row = request["rows"][0]
contract = row["source_contract"]
config = Path(contract["config_path"])
config.parent.mkdir(parents=True, exist_ok=True)
config.write_text("fixture:config_path\\n", encoding="utf-8")
marker = Path(contract["source_done_marker"])
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text("done\\n", encoding="utf-8")
marker.with_name("source_evidence.json").write_text(json.dumps({{
    "schema_version": "stage5_source_materialization_evidence_v1",
    "group_id": row["group_id"],
    "source_plan_sha256": row["source_evidence_sha256"],
    "status": "ready",
}}, sort_keys=True), encoding="utf-8")
Path("normalized-wrapper-executed.txt").write_text("copied", encoding="utf-8")
PYCODE
"""
            if role == "source_materializer"
            else "#!/bin/sh\nexit 0\n"
        )
        _write_executable(path)
        path.write_text(body, encoding="utf-8")
        path.chmod(0o700)
    (closure_root / "sibling.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    operator = tmp_path / "operator-assets"
    dataset = operator / "dataset"
    stable = operator / "stable"
    dataset.mkdir(parents=True)
    stable.mkdir()
    checkpoint = stable / "base.ckpt"
    config = stable / "pyramid.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text(
        "model:\n  args:\n    fusion_backbone:\n      num_filters: [3, 5, 7]\n",
        encoding="utf-8",
    )
    payload["external_training_binding"] = {
        "schema_version": "p6_external_training_binding_v1",
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
        "dataset_root": str(dataset),
        "base_checkpoint_path": str(checkpoint),
        "base_checkpoint_sha256": None,
        "pyramid_config_path": str(config),
        "pyramid_config_sha256": None,
        "training_parameters": {
            "training_mode": "finetune",
            "epochs": 1,
            "target_epoch": 9,
            "seed": 0,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "batch_size": 1,
            "dataset_split": "train",
            "checkpoint_selection": "best",
            "freeze_policy": "partial",
            "groups": 3,
            "width_per_group": 5,
            "base_stage_widths": [3, 5, 7],
        },
    }
    payload["execution_code_closure"] = {
        "schema_version": "p6_execution_code_closure_v1",
        "roots": [
            {
                "closure_id": "history",
                "source_root": str(closure_root),
                "destination_relative_root": "execution-closure/history",
                "sha256": _tree_sha(closure_root),
            }
        ],
        "roles": {
            role: {"closure_id": "history", "entrypoint_relative_path": name}
            for role, name in role_names.items()
        },
    }
    contract = payload["source_contract"]["source_contract"]
    for key in (
        "training_required",
        "training_source_kind",
        "dataset_root",
        "base_checkpoint_path",
        "base_checkpoint_sha256",
        "pyramid_config_path",
        "pyramid_config_sha256",
        "training_parameters",
    ):
        contract.pop(key, None)
    contract["stage_widths"] = [16, 32, 64]
    minimal_runner = _runner_template()
    runner = tmp_path / "private-inputs" / "runner-template.yaml"
    runner.parent.mkdir(parents=True, exist_ok=True)
    _write_full_runner_template(runner)
    runner_payload = yaml.safe_load(runner.read_text(encoding="utf-8"))
    runner_payload["stage1_scan"] = minimal_runner["stage1_scan"]
    target_interface = runner_payload["execution_interface"]
    source_interface = minimal_runner["execution_interface"]
    target_interface["controller"] = source_interface["controller"]
    for target, source in zip(
        target_interface["execution_chain"],
        source_interface["execution_chain"],
        strict=True,
    ):
        target["argv"] = source["argv"]
    target_interface["environment"]["activation_argv"] = source_interface["environment"][
        "activation_argv"
    ]
    runner_payload["stage1_scan"]["argv"][0] = str(role_files["stage1_scan"])
    target_interface["controller"]["argv"][0] = str(role_files["controller"])
    stage_roles = {
        "source_materialization": (
            history_root / "documented-stage5-chain" / MARKERS["source_materializer"]
        ),
        "quantization": role_files["quantization"],
        "performance": role_files["performance"],
        "ap": role_files["ap"],
        "finalization": role_files["finalization"],
    }
    for entry in target_interface["execution_chain"]:
        entry["argv"][0] = str(stage_roles[entry["stage"]])
    target_interface["environment"]["activation_argv"][0] = str(role_files["activation"])
    runner_payload["execution_interface"]["environment"]["values"] = {
        "CUDA_VISIBLE_DEVICES": {"kind": "literal", "value": "17,19,23"},
        "P6_HISTORY_RUN_MODE": {"kind": "literal", "value": "bound"},
        "P6_HISTORY_PRIVATE_ROOT": {"kind": "private_path", "value": "."},
        "P6_HISTORY_TASK_STATE": {"kind": "placeholder", "value": "{task_state}"},
        "P6_HISTORY_ROUND_OUTPUT_ROOT": {"kind": "placeholder", "value": "{round_output_root}"},
    }
    assert set(runner_payload["execution_interface"]["environment"]["values"]) == set(
        EXPECTED_HISTORY_ENV_KEYS
    )
    runner.write_text(yaml.safe_dump(runner_payload, sort_keys=False), encoding="utf-8")
    return payload, runner


def _as_v2_explicit(
    source_map: dict[str, Any], recipe: Mapping[str, Any], tmp_path: Path
) -> tuple[dict[str, Any], Path]:
    payload = copy.deepcopy(source_map)
    payload["schema_version"] = "p6_history_normalization_source_v2"
    payload["recipe_mode"] = "explicit_dynamic_recipe"
    payload["dynamic_materialization_recipe"] = copy.deepcopy(dict(recipe))
    payload["source_contract"]["source_contract"]["dynamic_materialization_recipe"] = copy.deepcopy(
        dict(recipe)
    )
    return _runtime_external_and_closure(payload, tmp_path)


def _write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic executable\n", encoding="utf-8")
    path.chmod(0o700)


def _as_v2_procedural(source_map: dict[str, Any], tmp_path: Path) -> tuple[dict[str, Any], Path]:
    payload = copy.deepcopy(source_map)
    payload["schema_version"] = "p6_history_normalization_source_v2"
    payload["recipe_mode"] = "procedural_profile"
    payload["procedural_recipe_profile"] = PROFILE_V1
    payload["procedural_recipe_source"] = {
        "role_refs": [
            "controller",
            "source_materializer",
            "performance_plan",
            "finalizer",
        ]
    }
    payload.pop("dynamic_materialization_recipe")
    payload["source_contract"]["source_contract"].pop("dynamic_materialization_recipe")
    return _runtime_external_and_closure(payload, tmp_path)


def _source_group() -> dict[str, Any]:
    evidence_sha = hashlib.sha256(b"synthetic-p6-history-source").hexdigest()
    contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": "pyramid|16x32x64",
        "model": "pyramid",
        "width": [16, 32, 64],
        "artifact_id": "synthetic-pyramid-template",
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "synthetic_fixture",
        "dynamic_materialization_recipe": _recipe(),
    }
    return {
        "group_id": contract["group_id"],
        "model": contract["model"],
        "width": contract["width"],
        "source_status": contract["source_status"],
        "source_evidence_sha256": evidence_sha,
        "source_contract": contract,
        "source_contract_sha256": _sha(contract),
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "synthetic_fixture",
    }


def valid_private_source_map(tmp_path: Path) -> dict[str, Any]:
    subprocess_safe_git_root = tmp_path / "history-root" / "source-git"
    history_root = subprocess_safe_git_root
    subprocess_safe_git_root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(subprocess_safe_git_root)], check=True)
    (subprocess_safe_git_root / "registry").mkdir()
    (subprocess_safe_git_root / "documented-stage5-chain").mkdir()
    input_paths = {
        name: _write_json(
            subprocess_safe_git_root / "inputs" / f"{name}.source.json",
            {"schema_version": f"synthetic_{name}_v1", "name": name},
        )
        for name in INPUT_NAMES
    }
    return {
        "schema_version": "p6_history_normalization_source_v1",
        "history_root": str(history_root),
        "asset_paths": {
            "training-data": str(subprocess_safe_git_root / "inputs"),
            "model-init": str(subprocess_safe_git_root / "registry"),
            "toolchain": str(subprocess_safe_git_root / "documented-stage5-chain"),
        },
        "input_sources": input_paths,
        "source_contract": _source_group(),
        "dynamic_materialization_recipe": _recipe(),
    }


def _history_root(source_map: Mapping[str, Any]) -> Path:
    value = source_map["history_root"]
    assert isinstance(value, str)
    return Path(value)


def test_normalizer_copies_four_json_inputs_and_writes_one_registry(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    history_root = _history_root(source_map)
    private_dir = tmp_path / "private-normalized"

    paths = normalize_history_inputs(source_map, history_root, private_dir)

    assert set(paths) == {
        "gold176_rows",
        "gold176_graph_features",
        "capability_profiles",
        "closure",
        "registry",
        "legacy",
    }
    for name in INPUT_NAMES:
        assert paths[name] == private_dir / "inputs" / f"{name}.json"
        assert json.loads(paths[name].read_text(encoding="utf-8"))["name"] == name
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    assert registry["schema_version"] == "stage5_candidate_source_registry_v1"
    assert len(registry["groups"]) == 1
    legacy = yaml.safe_load(paths["legacy"].read_text(encoding="utf-8"))
    assert legacy["schema_version"] == "p6_h800_coptv2x_local_v2"
    assert legacy["local_input_paths"] == {name: str(paths[name]) for name in INPUT_NAMES}
    assert legacy["asset_paths"] == {
        "training-data": str(private_dir / "inputs"),
        "model-init": str(paths["registry"]),
        "toolchain": str(private_dir / "toolchain"),
    }


def test_normalizer_preserves_virtual_environment_python_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normalized legacy locator must keep the active virtual environment."""
    python_target = tmp_path / "runtime/python3.10"
    python_target.parent.mkdir(parents=True)
    python_target.write_text("synthetic executable\n", encoding="utf-8")
    python_target.chmod(0o700)
    python_alias = tmp_path / "venv/bin/python"
    python_alias.parent.mkdir(parents=True)
    python_alias.symlink_to(python_target)
    assert python_alias != python_alias.resolve()
    monkeypatch.setattr(normalization_staging.sys, "executable", str(python_alias))
    source_map = valid_private_source_map(tmp_path)

    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        tmp_path / "private-normalized",
    )

    legacy = yaml.safe_load(paths["legacy"].read_text(encoding="utf-8"))
    assert legacy["source_registry_step"]["argv"][0] == str(python_alias)
    assert legacy["measurement_step"]["argv"][0] == str(python_alias)


def _outside_root(source_map: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    escaped = tmp_path / "outside.json"
    escaped.write_text(json.dumps({"name": "escaped"}), encoding="utf-8")
    source_map["input_sources"]["closure"] = str(escaped)
    return source_map


def _symlink_input(source_map: dict[str, Any], _tmp_path: Path) -> dict[str, Any]:
    original = Path(source_map["input_sources"]["closure"])
    link = original.with_name("closure-link.json")
    link.symlink_to(original)
    source_map["input_sources"]["closure"] = str(link)
    return source_map


def _wrong_recipe(source_map: dict[str, Any], _tmp_path: Path) -> dict[str, Any]:
    source_map["dynamic_materialization_recipe"]["schema_version"] = "wrong_recipe"
    return source_map


def _colliding_template(source_map: dict[str, Any], _tmp_path: Path) -> dict[str, Any]:
    templates = source_map["dynamic_materialization_recipe"]["output_path_templates_by_q_mode"][
        "fp16"
    ]
    templates["training_path_template"] = "materialized/collision.json"
    templates["checkpoint_path_template"] = "materialized/collision.json"
    return source_map


@pytest.mark.parametrize(
    "mutator",
    [_outside_root, _symlink_input, _wrong_recipe, _colliding_template],
)
def test_normalizer_fails_without_writing_partial_private_root(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any], Path], dict[str, Any]],
) -> None:
    source_map = mutator(copy.deepcopy(valid_private_source_map(tmp_path)), tmp_path)
    history_root = _history_root(source_map)
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError, match=r"^history_normalization_invalid:"):
        normalize_history_inputs(source_map, history_root, private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_forbidden_result_context_without_partial_root(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    source_map["source_contract"]["source_contract"]["metrics"] = {"latency_ms": 1.0}
    source_map["source_contract"]["source_contract_sha256"] = _sha(
        source_map["source_contract"]["source_contract"]
    )
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError, match=r"^history_normalization_invalid:"):
        normalize_history_inputs(source_map, _history_root(source_map), private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_fake_git_directory_without_partial_root(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    history_root = tmp_path / "fake-history-root"
    history_root.mkdir()
    (history_root / ".git").mkdir()
    (history_root / "registry").mkdir()
    (history_root / "documented-stage5-chain").mkdir()
    input_paths = {
        name: _write_json(
            history_root / "inputs" / f"{name}.source.json",
            {"schema_version": f"synthetic_{name}_v1", "name": name},
        )
        for name in INPUT_NAMES
    }
    source_map["history_root"] = str(history_root)
    source_map["asset_paths"] = {
        "training-data": str(history_root / "inputs"),
        "model-init": str(history_root / "registry"),
        "toolchain": str(history_root / "documented-stage5-chain"),
    }
    source_map["input_sources"] = input_paths
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError, match=r"^history_normalization_invalid:"):
        normalize_history_inputs(source_map, history_root, private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_duplicate_input_sources_without_partial_root(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    source_map["input_sources"]["closure"] = source_map["input_sources"]["gold176_rows"]
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError, match=r"^history_normalization_invalid:"):
        normalize_history_inputs(source_map, _history_root(source_map), private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_nonunique_asset_paths_without_partial_root(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    source_map["asset_paths"]["model-init"] = source_map["asset_paths"]["training-data"]
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError, match=r"^history_normalization_invalid:"):
        normalize_history_inputs(source_map, _history_root(source_map), private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_asset_from_nested_git_root_without_copying(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    history_root = _history_root(source_map)
    nested_toolchain = history_root / "documented-stage5-chain" / "nested"
    nested_toolchain.mkdir()
    subprocess.run(["git", "init", "-q", str(nested_toolchain)], check=True)
    _write_json(nested_toolchain / "marker.json", {"name": "nested"})
    source_map["asset_paths"]["toolchain"] = str(nested_toolchain)
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError, match=r"^history_normalization_invalid:"):
        normalize_history_inputs(source_map, history_root, private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_symlink_history_root_argument_without_partial_root(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    history_root = _history_root(source_map)
    symlink_root = tmp_path / "history-root-link"
    symlink_root.symlink_to(history_root, target_is_directory=True)
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError, match=r"^history_normalization_invalid:"):
        normalize_history_inputs(source_map, symlink_root, private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_history_root_below_git_top_level_without_partial_root(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    repository = tmp_path / "history-repository"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    history_root = repository / "subdir-history-root"
    (history_root / "registry").mkdir(parents=True)
    (history_root / "documented-stage5-chain").mkdir()
    input_paths = {
        name: _write_json(
            history_root / "inputs" / f"{name}.source.json",
            {"schema_version": f"synthetic_{name}_v1", "name": name},
        )
        for name in INPUT_NAMES
    }
    source_map["history_root"] = str(history_root)
    source_map["asset_paths"] = {
        "training-data": str(history_root / "inputs"),
        "model-init": str(history_root / "registry"),
        "toolchain": str(history_root / "documented-stage5-chain"),
    }
    source_map["input_sources"] = input_paths
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError, match=r"^history_normalization_invalid:"):
        normalize_history_inputs(source_map, history_root, private_dir)

    assert not private_dir.exists()


def test_v1_explicit_recipe_preserves_return_shape_without_runner_template(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    private_dir = tmp_path / "private-normalized"

    paths = normalize_history_inputs(source_map, _history_root(source_map), private_dir)

    assert set(paths) == {
        "gold176_rows",
        "gold176_graph_features",
        "capability_profiles",
        "closure",
        "registry",
        "legacy",
    }
    assert "derivation_recipe" not in paths


@pytest.mark.parametrize("recipe", [_recipe(), _recipe_v2()])
def test_v2_explicit_mode_accepts_supported_recipe_versions(
    tmp_path: Path, recipe: Mapping[str, Any]
) -> None:
    source_map, runner = _as_v2_explicit(valid_private_source_map(tmp_path), recipe, tmp_path)
    private_dir = tmp_path / "private-normalized"

    paths = normalize_history_inputs(
        source_map, _history_root(source_map), private_dir, runner_template_path=runner
    )

    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    assert registry["groups"][0]["source_contract"]["dynamic_materialization_recipe"] == recipe
    assert "derivation_recipe" not in paths
    assert {"runner_template", "source_wrapper_profile", "external_training_binding"}.issubset(
        paths
    )


def test_v2_procedural_mode_derives_recipe_and_records_canonical_value(
    tmp_path: Path,
) -> None:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    private_dir = tmp_path / "private-normalized"

    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        private_dir,
        runner_template_path=runner,
    )

    derived = json.loads(paths["derivation_recipe"].read_text(encoding="utf-8"))
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    assert derived == _recipe_v2()
    assert registry["groups"][0]["source_contract"]["dynamic_materialization_recipe"] == derived
    legacy = yaml.safe_load(paths["legacy"].read_text(encoding="utf-8"))
    assert legacy["history_recipe_derivation_path"] == str(paths["derivation_recipe"])


def test_v2_normalizer_keeps_training_external_and_writes_all_role_runner(
    tmp_path: Path,
) -> None:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    private_dir = tmp_path / "private-normalized"

    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        private_dir,
        runner_template_path=runner,
    )

    external = yaml.safe_load(paths["external_training_binding"].read_text())
    declared = source_map["external_training_binding"]
    checkpoint = Path(declared["base_checkpoint_path"])
    config = Path(declared["pyramid_config_path"])
    assert external["dataset_root"] == declared["dataset_root"]
    assert external["base_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert external["pyramid_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    runner_payload = yaml.safe_load(paths["runner_template"].read_text())
    interface = runner_payload["execution_interface"]
    assert len(interface["execution_chain"]) == 5
    assert len(EXECUTION_CLOSURE_ROLES) == 8
    copied_files = tuple(path for path in private_dir.rglob("*") if path.is_file())
    external_stats = {(path.stat().st_dev, path.stat().st_ino) for path in (checkpoint, config)}
    assert not external_stats.intersection(
        (path.stat().st_dev, path.stat().st_ino) for path in copied_files
    )
    assert checkpoint.read_bytes() not in {path.read_bytes() for path in copied_files}
    assert config.read_bytes() not in {path.read_bytes() for path in copied_files}
    profile = yaml.safe_load(paths["source_wrapper_profile"].read_text())
    destination = private_dir / profile["destination_relative_path"]
    implementation = private_dir / profile["implementation_relative_path"]
    assert destination.name == implementation.name == MARKERS["source_materializer"]
    assert destination.resolve() != implementation.resolve()
    environment = {
        "CUDA_VISIBLE_DEVICES": "17,19,23",
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(private_dir),
        "P6_HISTORY_TASK_STATE": str(private_dir / "task-state.json"),
        "P6_HISTORY_ROUND_OUTPUT_ROOT": str(private_dir / "round-00"),
    }
    round_root = Path(environment["P6_HISTORY_ROUND_OUTPUT_ROOT"])
    round_root.mkdir()
    request = round_root / "measurement-request.json"
    output_paths = source_bridge_output_paths(round_root / "materialized/pyramid-16-32-64")
    request.write_text(
        json.dumps(source_bridge_request(external, source_contract_fields=output_paths)),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            str(destination),
            "--request",
            str(request),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|16x32x64",
            "--gpu",
            "17",
        ],
        cwd=round_root,
        env=environment,
        shell=False,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert set(environment) == set(EXPECTED_HISTORY_ENV_KEYS)
    assert (implementation.parent / "normalized-wrapper-executed.txt").read_text() == "copied"


def test_source_map_v3_requires_post_source_leaf_binding(tmp_path: Path) -> None:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    source_map["schema_version"] = "p6_history_normalization_source_v3"
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError) as captured:
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            private_dir,
            runner_template_path=runner,
        )

    assert captured.value.category == "history_normalization_invalid"
    assert not private_dir.exists()


@pytest.mark.parametrize("missing_key", ("external_training_binding", "execution_code_closure"))
def test_v2_normalizer_requires_runtime_keys_before_destination_creation(
    tmp_path: Path, missing_key: str
) -> None:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    source_map.pop(missing_key)
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError) as captured:
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            private_dir,
            runner_template_path=runner,
        )
    assert captured.value.category == "history_normalization_invalid"
    assert not private_dir.exists()


def test_v2_normalizer_rejects_destination_overlapping_external_dataset(
    tmp_path: Path,
) -> None:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    private_dir = (
        Path(source_map["external_training_binding"]["dataset_root"]) / "normalized-private"
    )

    with pytest.raises(P6HistoryNormalizationError):
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            private_dir,
            runner_template_path=runner,
        )
    assert not private_dir.exists()


@pytest.mark.parametrize("mutation", ("source_argv_tail", "activation_role"))
def test_v2_normalizer_rejects_non_self_contained_role_binding(
    tmp_path: Path, mutation: str
) -> None:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    if mutation == "source_argv_tail":
        runner_payload = yaml.safe_load(runner.read_text(encoding="utf-8"))
        runner_payload["stage1_scan"]["argv"].append(runner_payload["stage1_scan"]["argv"][0])
        runner.write_text(yaml.safe_dump(runner_payload), encoding="utf-8")
    else:
        source_map["execution_code_closure"]["roles"]["activation"]["entrypoint_relative_path"] = (
            "quantize"
        )
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError):
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            private_dir,
            runner_template_path=runner,
        )
    assert not private_dir.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "explicit_with_procedural",
        "procedural_with_explicit",
        "procedural_missing_runner",
        "procedural_self_report",
    ],
)
def test_v2_recipe_modes_fail_closed_without_partial_root(tmp_path: Path, mutation: str) -> None:
    source_map = valid_private_source_map(tmp_path)
    runner: Path | None = None
    if mutation == "explicit_with_procedural":
        source_map, runner = _as_v2_explicit(source_map, _recipe(), tmp_path)
        source_map["procedural_recipe_profile"] = PROFILE_V1
        source_map["procedural_recipe_source"] = {"role_refs": []}
    else:
        source_map, runner = _as_v2_procedural(source_map, tmp_path)
        if mutation == "procedural_with_explicit":
            source_map["dynamic_materialization_recipe"] = _recipe_v2()
        elif mutation == "procedural_missing_runner":
            runner = None
        elif mutation == "procedural_self_report":
            source_map["procedural_recipe_source"]["capability"] = "training"
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(
        P6HistoryNormalizationError,
        match=r"^history_recipe_derivation_invalid:",
    ):
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            private_dir,
            runner_template_path=runner,
        )

    assert not private_dir.exists()
