"""Contract tests for the public, in-memory Stage4 selection completion."""

from __future__ import annotations

import copy
import builtins
import math
import socket
import subprocess
from collections.abc import Mapping
from typing import Any

import pytest

from framework.stage4.selection_completion_v1 import build_selection_completion_report


ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)


def _rows(group_count: int = 8) -> list[dict[str, object]]:
    return [
        {
            "manifest_job_id": f"g{group_index}|{dispatch_key}|{q_mode}",
            "group_id": f"g{group_index}",
            "dispatch_key": dispatch_key,
            "q_mode": q_mode,
            "latency_ms": 1.0 + arm_index,
            "energy_j": 2.0 + arm_index,
            "ap70": 0.5 + arm_index / 100,
        }
        for group_index in range(group_count)
        for arm_index, (dispatch_key, q_mode) in enumerate(ARMS)
    ]


def _selection(row: Mapping[str, object]) -> dict[str, object]:
    return {
        field: row[field]
        for field in ("manifest_job_id", "group_id", "dispatch_key", "q_mode")
    }


def _manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    group_ids = sorted({str(row["group_id"]) for row in rows})
    selected_by_group = {
        group_id: next(
            row
            for row in rows
            if row["group_id"] == group_id
            and row["dispatch_key"] == "tvm_auto"
            and row["q_mode"] == "int8"
        )
        for group_id in group_ids
    }
    midpoint = len(group_ids) // 2
    folds = []
    for fold_index, fold_groups in enumerate((group_ids[:midpoint], group_ids[midpoint:])):
        folds.append(
            {
                "fold_id": f"f{fold_index}",
                "candidate_group_ids": fold_groups,
                "selected_rows": [_selection(selected_by_group[group_id]) for group_id in fold_groups],
            }
        )
    return {
        "schema_version": "stage4_selection_completion_manifest_v1",
        "required_arm_product": [list(arm) for arm in ARMS],
        "candidate_row_ids": sorted(str(row["manifest_job_id"]) for row in rows),
        "selection_policy": "predicted_frontier_diversity",
        "folds": folds,
        "replay_policies": {
            "random": {
                "groups_to_95pct_oracle_HV_median": len(group_ids),
                "success_rate": 1.0,
            },
            "predicted_frontier_diversity": {
                "groups_to_95pct_oracle_HV_median": len(group_ids) - 2,
                "success_rate": 1.0,
            },
        },
    }


def _build(
    *,
    rows: list[dict[str, object]] | None = None,
    manifest: dict[str, object] | None = None,
) -> Mapping[str, Any]:
    input_rows = rows or _rows()
    return build_selection_completion_report(input_rows, manifest or _manifest(input_rows))


def test_builds_an_immutable_anonymous_closure_compatible_report() -> None:
    rows = _rows()
    manifest = _manifest(rows)
    rows_before = copy.deepcopy(rows)
    manifest_before = copy.deepcopy(manifest)

    report = _build(rows=rows, manifest=manifest)

    assert set(report) == {"schema_version", "summary", "selection", "replay", "binding"}
    assert report["schema_version"] == "stage4_selection_completion_v1"
    assert report["summary"] == {
        "evaluation_row_count": 32,
        "evaluation_group_count": 8,
        "retain_ranker": False,
        "selected_uncertainty_method": "lgbm_quantile",
    }
    assert report["selection"] == {
        "policy": "predicted_frontier_diversity",
        "selected_group_count": 8,
        "selected_arm_counts": {"tvm_auto|int8": 8},
    }
    assert report["replay"] == {
        "policies": {
            "random": {
                "groups_to_95pct_oracle_HV_median": 8,
                "success_rate": 1.0,
            },
            "predicted_frontier_diversity": {
                "groups_to_95pct_oracle_HV_median": 6,
                "success_rate": 1.0,
            },
        }
    }
    assert report["binding"] == {
        "manifest_schema_version": "stage4_selection_completion_manifest_v1",
        "fold_count": 2,
        "required_arm_count": 4,
    }
    assert rows == rows_before
    assert manifest == manifest_before
    assert "g0" not in repr(report)
    assert "manifest_job_id" not in repr(report)
    with pytest.raises(TypeError):
        report["summary"]["evaluation_row_count"] = 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows, _manifest: rows.pop(),
        lambda rows, _manifest: rows.__setitem__(
            1, {**rows[1], "manifest_job_id": rows[0]["manifest_job_id"]}
        ),
        lambda rows, _manifest: rows.__setitem__(
            0, {**rows[0], "latency_ms": math.inf}
        ),
        lambda rows, _manifest: rows.__setitem__(
            0, {**rows[0], "energy_j": 0.0}
        ),
        lambda rows, _manifest: rows.__setitem__(
            0, {**rows[0], "ap70": 1.1}
        ),
        lambda rows, _manifest: rows.__setitem__(
            0, {**rows[0], "terminal_status": "private"}
        ),
        lambda rows, _manifest: rows.__setitem__(
            0, {**rows[0], "root": "private-root"}
        ),
        lambda rows, _manifest: rows.__setitem__(
            0, {**rows[0], "dispatch_key": ["tvm_auto"]}
        ),
    ],
)
def test_fails_closed_for_incomplete_duplicate_nonfinite_or_leaking_candidates(
    mutation: Any,
) -> None:
    rows = _rows()
    manifest = _manifest(rows)
    mutation(rows, manifest)

    with pytest.raises(ValueError, match="invalid selection input"):
        _build(rows=rows, manifest=manifest)


def test_rejects_non_anonymous_ids_and_does_not_echo_them() -> None:
    private_marker = "private-uri-root-cache-checkpoint"
    rows = _rows()
    for row in rows:
        if row["group_id"] == "g0":
            row["group_id"] = private_marker
            row["manifest_job_id"] = (
                f"{private_marker}|{row['dispatch_key']}|{row['q_mode']}"
            )
    manifest = _manifest(rows)

    with pytest.raises(ValueError, match="invalid selection input") as exc_info:
        _build(rows=rows, manifest=manifest)

    assert private_marker not in str(exc_info.value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.__setitem__("schema_version", "private-schema"),
        lambda manifest: manifest.__setitem__(
            "required_arm_product", manifest["required_arm_product"][:-1]
        ),
        lambda manifest: manifest.__setitem__("selection_policy", "private-policy"),
        lambda manifest: manifest.__setitem__(
            "candidate_row_ids", manifest["candidate_row_ids"][:-1]
        ),
        lambda manifest: manifest["folds"][1].__setitem__(
            "candidate_group_ids", ["g0", "g7"]
        ),
        lambda manifest: manifest["folds"][0].__setitem__(
            "selected_rows",
            [
                *manifest["folds"][0]["selected_rows"],
                copy.deepcopy(manifest["folds"][0]["selected_rows"][0]),
            ],
        ),
        lambda manifest: manifest["replay_policies"]["random"].__setitem__(
            "success_rate", math.nan
        ),
        lambda manifest: manifest["replay_policies"]["random"].__setitem__(
            "groups_to_95pct_oracle_HV_median", 0
        ),
    ],
)
def test_fails_closed_for_manifest_schema_fold_selection_and_policy_drift(
    mutation: Any,
) -> None:
    rows = _rows()
    manifest = _manifest(rows)
    mutation(manifest)

    with pytest.raises(ValueError, match="invalid selection input"):
        _build(rows=rows, manifest=manifest)


def test_rejects_a_selected_row_outside_its_candidate_four_arm_set() -> None:
    rows = _rows()
    manifest = _manifest(rows)
    manifest["folds"][0]["selected_rows"][0] = {
        "manifest_job_id": "g0|trt_engine|fp16",
        "group_id": "g0",
        "dispatch_key": "trt_engine",
        "q_mode": "int8",
    }

    with pytest.raises(ValueError, match="invalid selection input"):
        _build(rows=rows, manifest=manifest)


def test_selection_output_has_the_closure_audit_summary_and_policy_shape() -> None:
    report = _build()

    assert report["summary"]["retain_ranker"] is False
    assert report["summary"]["selected_uncertainty_method"] == "lgbm_quantile"
    assert set(report["replay"]["policies"]) == {
        "random",
        "predicted_frontier_diversity",
    }
    assert all(
        set(policy_report)
        == {"groups_to_95pct_oracle_HV_median", "success_rate"}
        for policy_report in report["replay"]["policies"].values()
    )


def test_normal_completion_has_no_file_network_or_process_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_side_effect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("selection completion must stay in memory")

    monkeypatch.setattr(builtins, "open", fail_side_effect)
    monkeypatch.setattr(socket, "socket", fail_side_effect)
    monkeypatch.setattr(subprocess, "Popen", fail_side_effect)

    report = _build()

    assert report["summary"]["evaluation_group_count"] == 8
