from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import pytest
import yaml

from framework.stage6.p6_history_normalization_v1 import (
    P6HistoryNormalizationError,
    normalize_history_inputs,
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
                "materialized/{group_id}/{q_mode}/"
                f"{name.removesuffix('_path_template')}.json"
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
    templates = source_map["dynamic_materialization_recipe"][
        "output_path_templates_by_q_mode"
    ]["fp16"]
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

    with pytest.raises(
        P6HistoryNormalizationError, match=r"^history_normalization_invalid:"
    ):
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

    with pytest.raises(
        P6HistoryNormalizationError, match=r"^history_normalization_invalid:"
    ):
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

    with pytest.raises(
        P6HistoryNormalizationError, match=r"^history_normalization_invalid:"
    ):
        normalize_history_inputs(source_map, history_root, private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_duplicate_input_sources_without_partial_root(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    source_map["input_sources"]["closure"] = source_map["input_sources"]["gold176_rows"]
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(
        P6HistoryNormalizationError, match=r"^history_normalization_invalid:"
    ):
        normalize_history_inputs(source_map, _history_root(source_map), private_dir)

    assert not private_dir.exists()


def test_normalizer_rejects_nonunique_asset_paths_without_partial_root(
    tmp_path: Path,
) -> None:
    source_map = valid_private_source_map(tmp_path)
    source_map["asset_paths"]["model-init"] = source_map["asset_paths"]["training-data"]
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(
        P6HistoryNormalizationError, match=r"^history_normalization_invalid:"
    ):
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

    with pytest.raises(
        P6HistoryNormalizationError, match=r"^history_normalization_invalid:"
    ):
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

    with pytest.raises(
        P6HistoryNormalizationError, match=r"^history_normalization_invalid:"
    ):
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

    with pytest.raises(
        P6HistoryNormalizationError, match=r"^history_normalization_invalid:"
    ):
        normalize_history_inputs(source_map, history_root, private_dir)

    assert not private_dir.exists()
