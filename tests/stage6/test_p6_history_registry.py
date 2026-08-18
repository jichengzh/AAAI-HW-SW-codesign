from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from framework.stage5.production_search_v1 import validate_source_contract
from framework.stage6.p6_history_registry_v1 import (
    P6HistoryRegistryError,
    materialize_history_registry,
)


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan(q_modes: tuple[str, ...], *, duplicate: bool = False) -> dict[str, Any]:
    widths = ([17, 31, 63], [23, 47, 95])
    candidates = [
        {
            "width": list(width),
            "q_mode": q_mode,
            "source_point_ids": [
                f"stage{stage}-{q_mode}-w{value}"
                for stage, value in enumerate(width, start=1)
            ],
        }
        for width in widths
        for q_mode in q_modes
    ]
    if duplicate:
        candidates.append(copy.deepcopy(candidates[0]))
    candidates.sort(
        key=lambda row: (
            tuple(row["width"]),
            row["q_mode"],
            tuple(row["source_point_ids"]),
        )
    )
    return {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "source_schema": "stage2_search_space_v1",
        "target_model": "pyramid",
        "hardware_target": "h800",
        "execution_backend": "tvm_auto",
        "candidate_source_mode": "framework_stage2_search_space",
        "structure_count": len({tuple(row["width"]) for row in candidates}),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _recipe() -> dict[str, Any]:
    output_templates = {
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
        "group_id_template": (
            "pyramid|{stage1_width}x{stage2_width}x{stage3_width}"
        ),
        "artifact_id_template": (
            "pyramid-{stage1_width}-{stage2_width}-{stage3_width}"
        ),
        "output_path_templates_by_q_mode": output_templates,
    }


def _binding(tmp_path: Path) -> dict[str, Any]:
    private_root = tmp_path / "synthetic-history"
    private_root.mkdir()
    evidence_sha = hashlib.sha256(b"synthetic-history-evidence").hexdigest()
    return {
        "schema_version": "p6_history_binding_v1",
        "target": {
            "model": "pyramid",
            "hardware": "h800",
            "backend": "tvm_auto",
        },
        "private_root": str(private_root),
        "source_contract_template": {
            "schema_version": "stage5_source_contract_v1",
            "group_id": "pyramid|16x32x64",
            "model": "pyramid",
            "width": [16, 32, 64],
            "artifact_id": "synthetic-legacy-template",
            "source_status": "ready",
            "source_evidence_sha256": evidence_sha,
            "materialization_scope": "synthetic_fixture",
            "dynamic_materialization_recipe": _recipe(),
        },
        "status": "validated",
    }


def _plan_identity_map(
    plan: Mapping[str, Any],
) -> dict[tuple[tuple[int, ...], str], tuple[str, ...]]:
    return {
        (tuple(row["width"]), row["q_mode"]): tuple(row["source_point_ids"])
        for row in plan["candidates"]
    }


def _registry_identity_map(
    registry: Mapping[str, Any],
) -> dict[tuple[tuple[int, ...], str], tuple[str, ...]]:
    return {
        (tuple(group["width"]), q_mode): tuple(
            group["source_point_ids_by_q_mode"][q_mode]
        )
        for group in registry["groups"]
        for q_mode in group["available_q_modes"]
    }


@pytest.mark.parametrize("q_modes", [("fp16",), ("int8",), ("fp16", "int8")])
def test_registry_materializes_every_dynamic_identity_without_pruning(
    tmp_path: Path, q_modes: tuple[str, ...]
) -> None:
    plan = _plan(q_modes)
    binding = _binding(tmp_path)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    original_plan = copy.deepcopy(plan)
    original_binding = copy.deepcopy(binding)

    registry = materialize_history_registry(plan, binding, local_output_root)

    assert registry["schema_version"] == "stage5_candidate_source_registry_v2"
    assert _registry_identity_map(registry) == _plan_identity_map(plan)
    assert (
        sum(len(group["available_q_modes"]) for group in registry["groups"])
        == plan["candidate_count"]
    )
    assert all("source_contract" in group for group in registry["groups"])
    assert all(validate_source_contract(group) for group in registry["groups"])
    assert json.loads((local_output_root / "source_registry.json").read_text()) == registry
    assert plan == original_plan
    assert binding == original_binding


def test_registry_binds_all_authoritative_output_paths_beneath_local_root(
    tmp_path: Path,
) -> None:
    plan = _plan(("fp16", "int8"))
    binding = _binding(tmp_path)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()

    registry = materialize_history_registry(plan, binding, local_output_root)

    rendered_paths: set[str] = set()
    for group in registry["groups"]:
        contract = group["source_contract"]
        assert "dynamic_materialization_recipe" not in contract
        assert contract["stage_widths"] == {
            "stage1_width": group["width"][0],
            "stage2_width": group["width"][1],
            "stage3_width": group["width"][2],
        }
        outputs_by_q_mode = contract["materialization_outputs_by_q_mode"]
        assert set(outputs_by_q_mode) == set(group["available_q_modes"])
        for outputs in outputs_by_q_mode.values():
            assert set(outputs) == {
                "training_path",
                "checkpoint_path",
                "onnx_path",
                "calibration_path",
            }
            for value in outputs.values():
                resolved = Path(value)
                assert resolved.is_relative_to(local_output_root.resolve())
                assert value not in rendered_paths
                rendered_paths.add(value)


def test_registry_output_contains_no_search_result_or_terminal_leakage(
    tmp_path: Path,
) -> None:
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    registry = materialize_history_registry(
        _plan(("fp16", "int8")), _binding(tmp_path), local_output_root
    )

    forbidden = {"metrics", "objectives", "status", "cache", "result", "terminal"}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            assert not (set(value) & forbidden)
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(registry)


def _drop_recipe(binding: dict[str, Any]) -> None:
    binding["source_contract_template"].pop("dynamic_materialization_recipe")


def _drop_one_calibration_mapping(binding: dict[str, Any]) -> None:
    recipe = binding["source_contract_template"]["dynamic_materialization_recipe"]
    recipe["output_path_templates_by_q_mode"]["int8"].pop(
        "calibration_path_template"
    )


def _invalidate_contract(binding: dict[str, Any]) -> None:
    binding["source_contract_template"]["materialization_scope"] = ""


def _escape_output_root(binding: dict[str, Any]) -> None:
    recipe = binding["source_contract_template"]["dynamic_materialization_recipe"]
    recipe["output_path_templates_by_q_mode"]["fp16"][
        "training_path_template"
    ] = "../escaped-training.json"


def _collide_output_paths(binding: dict[str, Any]) -> None:
    recipe = binding["source_contract_template"]["dynamic_materialization_recipe"]
    recipe["output_path_templates_by_q_mode"]["fp16"] = {
        "training_path_template": "materialized/collision.json",
        "checkpoint_path_template": "materialized/collision.json",
        "onnx_path_template": "materialized/collision.json",
        "calibration_path_template": "materialized/collision.json",
    }


def _add_forbidden_result_context(binding: dict[str, Any]) -> None:
    binding["source_contract_template"]["metrics"] = {"latency_ms": 1.0}


def _add_legacy_path_escape(binding: dict[str, Any]) -> None:
    binding["source_contract_template"]["checkpoint_path"] = str(
        Path(binding["private_root"]).parent / "outside-checkpoint.pt"
    )


def _add_template_sha_mismatch(binding: dict[str, Any]) -> None:
    binding["source_contract_template_sha256"] = "0" * 64


def _invalidate_private_root(binding: dict[str, Any]) -> None:
    binding["private_root"] = None


@pytest.mark.parametrize(
    "mutate_binding",
    [
        _drop_recipe,
        _drop_one_calibration_mapping,
        _invalidate_contract,
        _escape_output_root,
        _collide_output_paths,
        _add_forbidden_result_context,
        _add_legacy_path_escape,
        _add_template_sha_mismatch,
        _invalidate_private_root,
    ],
)
def test_registry_fails_closed_before_write_for_incomplete_or_unsafe_recipe(
    tmp_path: Path, mutate_binding: Callable[[dict[str, Any]], None]
) -> None:
    plan = _plan(("fp16", "int8"))
    binding = _binding(tmp_path)
    mutate_binding(binding)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()
    registry_path = local_output_root / "source_registry.json"

    with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
        materialize_history_registry(plan, binding, local_output_root)

    assert not registry_path.exists()
    assert plan["candidate_count"] == 4


def test_registry_rejects_duplicate_plan_identity_before_write(tmp_path: Path) -> None:
    plan = _plan(("fp16",), duplicate=True)
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()

    with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
        materialize_history_registry(plan, _binding(tmp_path), local_output_root)

    assert not (local_output_root / "source_registry.json").exists()
    assert len(plan["candidates"]) == 3


def test_registry_rejects_source_evidence_sha_inconsistency_before_write(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)
    binding["source_contract_template"]["source_evidence_sha256"] = "0" * 63
    local_output_root = tmp_path / "private-output"
    local_output_root.mkdir()

    with pytest.raises(P6HistoryRegistryError, match=r"^source_registry_invalid:"):
        materialize_history_registry(
            _plan(("fp16",)), binding, local_output_root
        )

    assert not (local_output_root / "source_registry.json").exists()
