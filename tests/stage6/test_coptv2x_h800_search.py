from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6.coptv2x_h800_search_v2 import (
    LocalExecutionStep,
    LocalP6CoptV2XConfig,
    P6CoptV2XContractError,
    P6CoptV2XExecutionError,
    PublicP6CoptV2XContract,
    load_local_config,
    load_public_contract,
    run_p6_coptv2x_search,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_yaml(path: Path, content: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def _public_contract(**overrides: Any) -> dict[str, Any]:
    contract = {
        "schema_version": "p6_h800_coptv2x_search_contract_v2",
        "search_id": "p6-pyramid-h800-tvm",
        "target": "h800",
        "target_model": "pyramid",
        "execution_backend": "tvm_auto",
        "seed": 73,
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
        "configuration_label": "p6-pyramid-h800-tvm",
        "candidate_space_label": "coptv2x-pyramid-width-grid-v1",
        "metric_names": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }
    return {**contract, **overrides}


def _local_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    output_root = tmp_path / "private-output"
    payload = {
        "schema_version": "p6_h800_coptv2x_local_v2",
        "target": "h800",
        "asset_paths": {
            "training-data": str(tmp_path / "training-data"),
            "model-init": str(tmp_path / "model-init"),
            "toolchain": str(tmp_path / "toolchain"),
        },
        "local_input_paths": {
            "gold176_rows": str(tmp_path / "gold176_rows.json"),
            "gold176_graph_features": str(tmp_path / "gold176_graph_features.json"),
            "capability_profiles": str(tmp_path / "capability_profiles.json"),
            "closure": str(tmp_path / "closure.json"),
        },
        "source_registry_step": {
            "name": "build_source_registry",
            "argv": [
                "python",
                "local_build_registry.py",
                "{local_output_root}",
                "{source_registry_json}",
            ],
        },
        "measurement_step": {
            "name": "measure_batch",
            "argv": [
                "python",
                "local_measure.py",
                "{measurement_request}",
                "{feedback_json}",
                "{round_output_root}",
            ],
        },
        "local_output_root": str(output_root),
    }
    return {**payload, **overrides}


def _profile() -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id="h800-tvm-auto",
        hardware_target="h800",
        compiler_fingerprint="a" * 64,
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )


def _non_target_profile() -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id="h800-trt-engine",
        hardware_target="h800",
        compiler_fingerprint="b" * 64,
        dispatch_key="trt_engine",
        features={"int8_propagation": 1.0, "qdq_fold": 1.0},
    )


def _graph(group_id: str, width: list[int]) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "conv_count": 27,
        "conv_macs": float(width[0] * width[1] * width[2]),
        "group_conv_count": 3,
    }


def _gold176(
    *, include_non_target_backend: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for index in range(176):
        width = [16 + (index % 7) * 8, 32 + (index % 8) * 8, 64 + (index % 9) * 8]
        group_id = f"gold-{index:03d}"
        q_mode = "int8" if index % 2 else "fp16"
        non_target = include_non_target_backend and index >= 88
        dispatch_key = "trt_engine" if non_target else "tvm_auto"
        profile_id = "h800-trt-engine" if non_target else "h800-tvm-auto"
        graphs.append(_graph(group_id, width))
        rows.append(
            {
                "manifest_job_id": f"{group_id}|q={q_mode}|profile={profile_id}",
                "row_id": f"{group_id}|q={q_mode}|profile={profile_id}",
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "dispatch_key": dispatch_key,
                "capability_profile_id": profile_id,
                "q_mode": q_mode,
                "latency_ms": 2.0 + index * 0.01,
                "energy_j": 0.5 + index * 0.005,
                "ap30": 0.90,
                "ap50": 0.80,
                "ap70": 0.70 - index * 0.0001,
                "terminal_status": "measured_success_gold",
                "training_source": "initial_coldstart",
            }
        )
    return rows, graphs


def test_initial_coldstart_keeps_true_failures_as_evidence_outside_value_fit() -> None:
    """The reviewed Gold176 ledger has 174 value rows and two real failures."""
    rows, graphs = _gold176(include_non_target_backend=True)
    for index, status in ((174, "feasibility_failure"), (175, "numerical_feasibility_failure")):
        rows[index] = {
            key: value
            for key, value in rows[index].items()
            if key not in {"latency_ms", "energy_j", "ap30", "ap50", "ap70"}
        }
        rows[index]["terminal_status"] = status

    bundle = execution.fit_initial_coldstart_bundle(
        rows, graphs, [_profile(), _non_target_profile()], seed=73
    )

    assert bundle.manifest["input_row_count"] == 176
    assert bundle.manifest["value_training_row_count"] == 174


def _source_group(group_id: str, width: list[int]) -> dict[str, Any]:
    evidence_sha = hashlib.sha256(f"source:{group_id}".encode()).hexdigest()
    source_contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": group_id,
        "model": "pyramid",
        "width": width,
        "artifact_id": f"fixture-{group_id}",
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "synthetic_fixture",
    }
    contract_sha = hashlib.sha256(
        json.dumps(
            source_contract,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": width,
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "source_contract": source_contract,
        "source_contract_sha256": contract_sha,
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
        "graph_features": _graph(group_id, width),
    }


def _closure() -> dict[str, Any]:
    return {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": 176},
        "frozen_holdout": {"groups": []},
    }


def _write_source_registry(path: Path, *, count: int) -> None:
    widths = [
        [16 + (index // 49) * 8, 32 + (index // 7 % 7) * 8, 64 + (index % 7) * 8]
        for index in range(count)
    ]
    groups = [
        _source_group(f"pyramid|{'x'.join(map(str, width))}", width) for width in widths
    ]
    path.write_text(
        json.dumps({"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}),
        encoding="utf-8",
    )


def _write_feedback_from_request(request_path: Path, feedback_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "p6_h800_coptv2x_feedback_v2",
        "measurement_request_sha256": request["measurement_request_sha256"],
        "rows": [
            {
                "row_id": row["row_id"],
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for row in request["rows"]
        ],
    }
    feedback_path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_request() -> dict[str, Any]:
    return {
        "measurement_request_sha256": "request-identity",
        "rows": [{"row_id": f"row-{index}"} for index in range(4)],
    }


def _minimal_feedback() -> dict[str, Any]:
    return {
        "schema_version": "p6_h800_coptv2x_feedback_v2",
        "measurement_request_sha256": "request-identity",
        "rows": [
            {
                "row_id": f"row-{index}",
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for index in range(4)
        ],
    }


def _minimal_task() -> execution.SearchTask:
    return execution.SearchTask(
        task_id="minimal-task",
        target_model="pyramid",
        hardware_id="h800",
        capability_profile=_profile(),
    )


def _loaded_local_config(
    tmp_path: Path,
    *,
    source_group_count: int = 343,
    include_non_target_backend: bool = True,
) -> LocalP6CoptV2XConfig:
    del source_group_count
    gold_rows, gold_graphs = _gold176(
        include_non_target_backend=include_non_target_backend
    )
    payloads = {
        "gold176_rows": gold_rows,
        "gold176_graph_features": gold_graphs,
        "capability_profiles": [
            _profile(),
            *([_non_target_profile()] if include_non_target_backend else []),
        ],
        "closure": _closure(),
    }
    for name, payload in payloads.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    return load_local_config(_write_yaml(tmp_path / "local.yaml", _local_config(tmp_path)), contract)


def test_run_p6_rejects_incomplete_coldstart_profile_context(tmp_path: Path) -> None:
    """Gold176 fitting must receive every capability profile represented in its ledger."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    profiles_path = local.local_input_paths["capability_profiles"]
    profiles_path.write_text(json.dumps([_profile()]), encoding="utf-8")

    with pytest.raises(P6CoptV2XContractError, match="coldstart capability"):
        run_p6_coptv2x_search(
            contract,
            local,
            "abc123",
            lambda argv, cwd: (_ for _ in ()).throw(
                AssertionError(f"source step must not run: {argv} {cwd}")
            ),
        )


def test_run_p6_builds_registry_refits_gold176_and_runs_four_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path, include_non_target_backend=True)
    requests: list[dict[str, Any]] = []
    initial_fit_input_counts: list[int] = []
    online_fit_input_counts: list[int] = []
    real_initial_fit = execution.fit_initial_coldstart_bundle
    real_online_fit = execution.fit_online_bundle

    def recording_initial_fit(
        rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any
    ) -> Any:
        initial_fit_input_counts.append(len(rows))
        return real_initial_fit(rows, *args, **kwargs)

    def recording_online_fit(
        rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any
    ) -> Any:
        online_fit_input_counts.append(len(rows))
        return real_online_fit(rows, *args, **kwargs)

    monkeypatch.setattr(execution, "fit_initial_coldstart_bundle", recording_initial_fit)
    monkeypatch.setattr(execution, "fit_online_bundle", recording_online_fit)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            source_registry_path = Path(argv[3])
            _write_source_registry(source_registry_path, count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        requests.append(request)
        feedback = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": [
                {
                    "row_id": row["row_id"],
                    "terminal_status": "measured_success_gold",
                    "latency_ms": 3.0 + len(requests),
                    "energy_j": 0.7 + len(requests) / 10,
                    "ap30": 0.91,
                    "ap50": 0.82,
                    "ap70": 0.73,
                }
                for row in request["rows"]
            ],
        }
        Path(argv[3]).write_text(json.dumps(feedback), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert state.completed_rounds == 4
    assert state.measured_candidate_count == 16
    assert len(requests) == 4
    assert [request["round_index"] for request in requests] == [0, 1, 2, 3]
    assert all(
        request["required_metrics"] == ["latency_ms", "energy_j", "ap30", "ap50", "ap70"]
        for request in requests
    )
    selected = [row["row_id"] for request in requests for row in request["rows"]]
    assert len(selected) == len(set(selected)) == 16
    assert {row["q_mode"] for request in requests for row in request["rows"]} <= {
        "fp16",
        "int8",
    }
    assert {row["dispatch_key"] for request in requests for row in request["rows"]} == {
        "tvm_auto"
    }
    assert initial_fit_input_counts == [176]
    assert online_fit_input_counts == [180, 184, 188]
    assert state.local_state_path == local.local_output_root / "state.json"
    stored_state = json.loads(state.local_state_path.read_text(encoding="utf-8"))
    assert stored_state == {
        "schema_version": "p6_h800_coptv2x_local_state_v2",
        "status": "completed",
        "code_revision": "abc123",
        "completed_rounds": 4,
        "measured_candidate_count": 16,
        "failure_code": None,
    }
    serialized_state = json.dumps(stored_state, sort_keys=True)
    assert str(tmp_path) not in serialized_state
    assert not any(row_id in serialized_state for row_id in selected)


@pytest.mark.parametrize(
    "terminal_status", ["feasibility_failure", "numerical_feasibility_failure"]
)
def test_run_p6_accepts_true_candidate_failures_as_budget_consuming_feedback(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    """Catches treating valid candidate failures as an invalid or free batch."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path, source_group_count=343)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        feedback = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": [
                {
                    "row_id": row["row_id"],
                    "terminal_status": terminal_status,
                    "failure_reason": "synthetic_feasibility",
                }
                for row in request["rows"]
            ],
        }
        Path(argv[3]).write_text(json.dumps(feedback), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert state.measured_candidate_count == 16


def test_run_p6_excludes_failure_only_graph_features_from_online_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches feasibility-only graph fields changing the online model schema."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    real_select = execution.select_task_batch
    real_online_fit = execution.fit_online_bundle
    online_bundles: list[Any] = []
    online_fit_rows: list[list[dict[str, Any]]] = []
    selection_call_count = 0

    def select_failure_candidate(
        predicted_rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal selection_call_count
        selection = real_select(predicted_rows, *args, **kwargs)
        failure_candidate = next(
            (
                row
                for row in predicted_rows
                if "failure_only_feature" in row["graph_features"]
            ),
            None,
        )
        selected_by_id: dict[str, Mapping[str, Any]] = {}
        preferred = [failure_candidate] if selection_call_count == 0 and failure_candidate else []
        for row in [*preferred, *selection["selected_rows"], *predicted_rows]:
            if "failure_only_feature" in row["graph_features"] and row not in preferred:
                continue
            selected_by_id.setdefault(str(row["row_id"]), row)
        selected = list(selected_by_id.values())[:4]
        selection_call_count += 1
        return {
            **selection,
            "selected_row_count": len(selected),
            "selected_row_ids": [row["row_id"] for row in selected],
            "selected_rows": selected,
        }

    def record_online_fit(
        rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any
    ) -> Any:
        online_fit_rows.append([dict(row) for row in rows])
        bundle = real_online_fit(rows, *args, **kwargs)
        online_bundles.append(bundle)
        return bundle

    monkeypatch.setattr(execution, "select_task_batch", select_failure_candidate)
    monkeypatch.setattr(execution, "fit_online_bundle", record_online_fit)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            registry_path = Path(argv[3])
            _write_source_registry(registry_path, count=343)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["groups"][0]["graph_features"] = {
                **registry["groups"][0]["graph_features"],
                "failure_only_feature": 1.0,
            }
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        rows = []
        for row in request["rows"]:
            if "failure_only_feature" in row["graph_features"]:
                rows.append(
                    {
                        "row_id": row["row_id"],
                        "terminal_status": "feasibility_failure",
                        "failure_reason": "synthetic_feasibility",
                    }
                )
            else:
                rows.append(
                    {
                        "row_id": row["row_id"],
                        "terminal_status": "measured_success_gold",
                        "latency_ms": 3.0,
                        "energy_j": 0.8,
                        "ap30": 0.91,
                        "ap50": 0.82,
                        "ap70": 0.73,
                    }
                )
        Path(argv[3]).write_text(
            json.dumps(
                {
                    "schema_version": "p6_h800_coptv2x_feedback_v2",
                    "measurement_request_sha256": request["measurement_request_sha256"],
                    "rows": rows,
                }
            ),
            encoding="utf-8",
        )
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert state.measured_candidate_count == 16
    assert [bundle.manifest["input_row_count"] for bundle in online_bundles] == [179, 183, 187]
    assert [bundle.manifest["value_training_row_count"] for bundle in online_bundles] == [179, 183, 187]
    assert all("graph:failure_only_feature" not in bundle.feature_names for bundle in online_bundles)
    assert all(
        "failure_only_feature" not in row.get("graph_features", {})
        for rows in online_fit_rows
        for row in rows
    )


def test_release_feedback_rows_detaches_nested_request_identity_context() -> None:
    """Catches later request mutation changing released training rows."""
    request = _minimal_request()
    request["rows"][0].update(
        {
            "graph_features": {"group_id": "row-0", "stable_feature": 1.0},
            "source_contract": {"group_id": "row-0", "artifact_id": "fixture-row-0"},
        }
    )

    released = execution._release_feedback_rows(
        _minimal_feedback(), request, task=_minimal_task()
    )
    request["rows"][0]["graph_features"]["stable_feature"] = 9.0
    request["rows"][0]["source_contract"]["artifact_id"] = "mutated"

    assert released[0]["graph_features"] == {"group_id": "row-0", "stable_feature": 1.0}
    assert released[0]["source_contract"] == {
        "group_id": "row-0",
        "artifact_id": "fixture-row-0",
    }


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_row", "feedback_candidate_mismatch"),
        ("extra_row", "feedback_candidate_mismatch"),
        ("wrong_request_sha", "feedback_request_mismatch"),
        ("wrong_row_id", "feedback_candidate_mismatch"),
        ("missing_metric", "feedback_metrics_invalid"),
        ("nan_metric", "feedback_metrics_invalid"),
        ("public_runner_failure", "feedback_terminal_status_invalid"),
    ],
)
def test_run_p6_quarantines_invalid_feedback_batches(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    """Catches partial release or budget advancement from malformed feedback."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path, source_group_count=343)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        rows = [
            {
                "row_id": row["row_id"],
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for row in request["rows"]
        ]
        if mutation == "missing_row":
            rows.pop()
        elif mutation == "extra_row":
            rows.append(dict(rows[0]))
        elif mutation == "wrong_row_id":
            rows[0]["row_id"] = "wrong"
        elif mutation == "missing_metric":
            del rows[0]["ap50"]
        elif mutation == "nan_metric":
            rows[0]["latency_ms"] = float("nan")
        elif mutation == "public_runner_failure":
            rows[0] = {"row_id": rows[0]["row_id"], "terminal_status": "public_runner_failure"}
        payload = {
            "schema_version": "p6_h800_coptv2x_feedback_v2",
            "measurement_request_sha256": (
                "wrong" if mutation == "wrong_request_sha" else request["measurement_request_sha256"]
            ),
            "rows": rows,
        }
        Path(argv[3]).write_text(json.dumps(payload), encoding="utf-8")
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "failed"
    assert state.failure_code == expected_code
    assert state.completed_rounds == 0
    assert state.measured_candidate_count == 0
    failure = json.loads((local.local_output_root / "round-00" / "failure.json").read_text())
    assert failure == {
        "schema_version": "p6_h800_coptv2x_failure_v2",
        "failure_code": expected_code,
        "completed_rounds": 0,
    }


def test_run_p6_rejects_incomplete_source_space(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] != "local_build_registry.py":
            raise AssertionError("measurement must not start for an incomplete space")
        _write_source_registry(Path(argv[3]), count=342)
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == "source_registry_invalid"


@pytest.mark.parametrize(
    ("registry_payload", "expected_code"),
    [(None, "source_registry_missing"), ([], "source_registry_invalid")],
)
def test_run_p6_reports_stable_source_registry_failures(
    tmp_path: Path, registry_payload: object | None, expected_code: str
) -> None:
    """Catches source output errors escaping with unstable local exceptions."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] != "local_build_registry.py":
            raise AssertionError("measurement must not start after source failure")
        if registry_payload is not None:
            Path(argv[3]).write_text(json.dumps(registry_payload), encoding="utf-8")
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert captured.value.failure_code == expected_code


def test_run_p6_reports_invalid_local_input_with_a_stable_code(tmp_path: Path) -> None:
    """Catches unreadable controller inputs being mislabeled as contract failures."""
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    local.local_input_paths["gold176_rows"].unlink()

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(
            contract,
            local,
            "abc123",
            lambda argv, cwd: (_ for _ in ()).throw(AssertionError((argv, cwd))),
        )

    assert captured.value.failure_code == "local_input_invalid"


def test_run_p6_never_selects_a_gold176_source_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    overlap_group = "pyramid|16x32x64"
    gold_rows_path = local.local_input_paths["gold176_rows"]
    gold_graphs_path = local.local_input_paths["gold176_graph_features"]
    gold_rows = json.loads(gold_rows_path.read_text(encoding="utf-8"))
    gold_graphs = json.loads(gold_graphs_path.read_text(encoding="utf-8"))
    gold_rows[0] = {**gold_rows[0], "group_id": overlap_group, "width": [16, 32, 64]}
    gold_graphs[0] = {**gold_graphs[0], "group_id": overlap_group, "width": [16, 32, 64]}
    gold_rows_path.write_text(json.dumps(gold_rows), encoding="utf-8")
    gold_graphs_path.write_text(json.dumps(gold_graphs), encoding="utf-8")
    selected_groups: list[str] = []
    real_select = execution.select_task_batch

    def select_without_gold(predicted_rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any):
        assert all(str(row["group_id"]) != overlap_group for row in predicted_rows)
        return real_select(predicted_rows, *args, **kwargs)

    monkeypatch.setattr(execution, "select_task_batch", select_without_gold)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        request_path = Path(argv[2])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        selected_groups.extend(str(row["group_id"]) for row in request["rows"])
        _write_feedback_from_request(request_path, Path(argv[3]))
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert overlap_group not in selected_groups


@pytest.mark.parametrize(
    "mutation",
    [
        lambda closure: closure.update(stage4_closed=False),
        lambda closure: closure.update(stage5_search_ready=False),
        lambda closure: closure["training_source_rows"].update(initial_coldstart=175),
        lambda closure: closure["canonical_value_heads"].update(ap70="legacy-head"),
        lambda closure: closure.update(uncertainty_policy="legacy-policy"),
        lambda closure: closure.update(selected_acquisition_policy="legacy-policy"),
    ],
)
def test_run_p6_rejects_unclosed_or_drifted_stage4_contract(
    tmp_path: Path, mutation: Any
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    closure_path = local.local_input_paths["closure"]
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    mutation(closure)
    closure_path.write_text(json.dumps(closure), encoding="utf-8")

    with pytest.raises(P6CoptV2XContractError, match="closure"):
        run_p6_coptv2x_search(
            contract,
            local,
            "abc123",
            lambda argv, cwd: (_ for _ in ()).throw(AssertionError((argv, cwd))),
        )


def test_run_p6_does_not_reuse_feedback_from_an_earlier_run(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)

    def first_runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        else:
            _write_feedback_from_request(Path(argv[2]), Path(argv[3]))
        return 0

    assert run_p6_coptv2x_search(contract, local, "abc123", first_runner).status == "completed"

    def stale_runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        return 0

    state = run_p6_coptv2x_search(contract, local, "abc123", stale_runner)

    assert state.status == "failed"
    assert state.failure_code == "feedback_missing"
    assert state.completed_rounds == 0
    assert state.measured_candidate_count == 0


@pytest.mark.parametrize(
    "metric,value",
    [
        ("latency_ms", 0.0),
        ("energy_j", -0.1),
        ("ap30", -0.01),
        ("ap50", 1.01),
        ("ap70", float("nan")),
    ],
)
def test_release_feedback_rejects_nonphysical_metrics(metric: str, value: float) -> None:
    feedback = _minimal_feedback()
    feedback["rows"][0][metric] = value

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        execution._release_feedback_rows(feedback, _minimal_request(), task=_minimal_task())

    assert captured.value.failure_code == "feedback_metrics_invalid"


def test_run_step_redacts_runner_exceptions() -> None:
    step = LocalExecutionStep("measure_batch", ("python", "/private/adapter.py"))

    def raising_runner(argv: tuple[str, ...], cwd: Path) -> int:
        raise RuntimeError(f"secret failure: {argv!r} from {cwd}")

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        execution._run_step(
            step,
            {},
            cwd=Path("/private/output"),
            runner=raising_runner,
        )

    assert str(captured.value) == "local command failed"
    assert "/private" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "/private" not in repr(captured.value.__context__)


def test_run_p6_rejects_symlinked_round_output_without_touching_victim(
    tmp_path: Path,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    local.local_output_root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    victim_feedback = victim / "feedback.json"
    victim_feedback.write_text("retain", encoding="utf-8")
    (local.local_output_root / "round-00").symlink_to(victim, target_is_directory=True)
    measurement_started = False

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_started
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        else:
            measurement_started = True
        return 0

    with pytest.raises(P6CoptV2XExecutionError, match="output"):
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert measurement_started is False
    assert victim_feedback.read_text(encoding="utf-8") == "retain"
    assert not (victim / "measurement_request.json").exists()


def test_run_p6_rejects_symlinked_source_registry_before_adapter(
    tmp_path: Path,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    local.local_output_root.mkdir()
    victim = tmp_path / "victim-registry.json"
    victim.write_text("retain", encoding="utf-8")
    (local.local_output_root / "source_registry.json").symlink_to(victim)
    adapter_started = False

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal adapter_started
        del cwd
        adapter_started = True
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        return 1

    with pytest.raises(P6CoptV2XExecutionError, match="output|command"):
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert adapter_started is False
    assert victim.read_text(encoding="utf-8") == "retain"


def test_run_p6_rejects_symlinked_state_without_touching_victim(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    local.local_output_root.mkdir()
    victim = tmp_path / "victim-state.json"
    victim.write_text("retain", encoding="utf-8")
    (local.local_output_root / "state.json").symlink_to(victim)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        else:
            _write_feedback_from_request(Path(argv[2]), Path(argv[3]))
        return 0

    with pytest.raises(P6CoptV2XExecutionError, match="output"):
        run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert victim.read_text(encoding="utf-8") == "retain"


def test_run_p6_revalidates_feedback_leaf_after_adapter(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    local = _loaded_local_config(tmp_path)
    captured_request: dict[str, Any] | None = None

    def capture_runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal captured_request
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        captured_request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        return 1

    failed = run_p6_coptv2x_search(contract, local, "abc123", capture_runner)
    assert failed.status == "failed"
    assert failed.failure_code == "command_failed"
    assert captured_request is not None
    victim = tmp_path / "external-feedback.json"
    victim_payload = {
        "schema_version": "p6_h800_coptv2x_feedback_v2",
        "measurement_request_sha256": captured_request["measurement_request_sha256"],
        "rows": [
            {
                "row_id": row["row_id"],
                "terminal_status": "measured_success_gold",
                "latency_ms": 3.0,
                "energy_j": 0.8,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for row in captured_request["rows"]
        ],
    }
    victim_text = json.dumps(victim_payload, sort_keys=True)
    victim.write_text(victim_text, encoding="utf-8")
    measurement_calls = 0

    def symlink_runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal measurement_calls
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
            return 0
        measurement_calls += 1
        if measurement_calls == 1:
            Path(argv[3]).symlink_to(victim)
            return 0
        return 1

    with pytest.raises(P6CoptV2XExecutionError, match="output") as captured:
        run_p6_coptv2x_search(contract, local, "abc123", symlink_runner)

    assert captured.value.failure_code == "unsafe_output"
    assert measurement_calls == 1
    assert victim.read_text(encoding="utf-8") == victim_text


def test_release_feedback_rejects_duplicate_or_extra_rows() -> None:
    feedback = _minimal_feedback()
    feedback["rows"].append(dict(feedback["rows"][0]))

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        execution._release_feedback_rows(feedback, _minimal_request(), task=_minimal_task())

    assert captured.value.failure_code == "feedback_candidate_mismatch"


def test_load_public_contract_requires_fixed_pyramid_h800_tvm_budget(tmp_path: Path) -> None:
    invalid_contracts = [
        _public_contract(target="orin"),
        _public_contract(target_model="codriving"),
        _public_contract(execution_backend="unsupported_backend"),
        _public_contract(sample_budget=15),
        _public_contract(batch_size=2),
        _public_contract(round_count=5),
    ]

    for index, payload in enumerate(invalid_contracts):
        with pytest.raises(P6CoptV2XContractError):
            load_public_contract(_write_yaml(tmp_path / f"contract-{index}.yaml", payload))


@pytest.mark.parametrize(
    "forbidden_key",
    ["path", "argv", "host", "candidate_id", "raw_logs", "checkpoint", "sha256", "hash"],
)
def test_load_public_contract_rejects_private_execution_keys(
    tmp_path: Path, forbidden_key: str
) -> None:
    payload = _public_contract()
    payload[forbidden_key] = "private-value"

    with pytest.raises(P6CoptV2XContractError, match="forbidden"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_id", "/private/run"),
        ("configuration_label", "h800-host-01"),
        ("configuration_label", "hostname-01"),
        ("configuration_label", "raw-logs-2026"),
        ("configuration_label", "candidate-42"),
        ("configuration_label", "candidate-id-42"),
        ("target_model", "python train.py"),
    ],
)
def test_load_public_contract_rejects_private_or_command_style_public_values(
    tmp_path: Path, field: str, value: str
) -> None:
    with pytest.raises(P6CoptV2XContractError, match="restricted"):
        load_public_contract(
            _write_yaml(tmp_path / "contract.yaml", _public_contract(**{field: value}))
        )


@pytest.mark.parametrize(
    "asset",
    [
        {"label": "checkpoints", "version": "v1", "license_status": "cleared"},
        {"label": "training-data", "version": "sha256:abc123", "license_status": "cleared"},
        {"label": "training-data", "version": "hash:abc123", "license_status": "cleared"},
        {"label": "training-data", "version": "a" * 64, "license_status": "cleared"},
    ],
)
def test_load_public_contract_rejects_restricted_asset_values(
    tmp_path: Path, asset: dict[str, str]
) -> None:
    with pytest.raises(P6CoptV2XContractError, match="restricted"):
        load_public_contract(
            _write_yaml(tmp_path / "contract.yaml", _public_contract(assets=[asset]))
        )


def test_load_public_contract_accepts_the_public_example() -> None:
    contract = load_public_contract(
        REPOSITORY_ROOT / "configs/execution/p6_h800_search.example.yaml"
    )

    assert isinstance(contract, PublicP6CoptV2XContract)
    assert contract.target == "h800"
    assert contract.target_model == "pyramid"
    assert contract.execution_backend == "tvm_auto"
    assert (contract.round_count, contract.batch_size, contract.sample_budget) == (4, 4, 16)
    assert contract.metric_names == ("latency_ms", "energy_j", "ap30", "ap50", "ap70")


@pytest.mark.parametrize("field", ["legacy_round_limit", "legacy_report", "legacy_output"])
def test_load_public_contract_rejects_unknown_round_and_output_keys(
    tmp_path: Path, field: str
) -> None:
    payload = _public_contract()
    payload[field] = "legacy"

    with pytest.raises(P6CoptV2XContractError, match="unknown|forbidden"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", payload))


def test_load_local_config_requires_source_and_measurement_steps(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()

    loaded = load_local_config(_write_yaml(tmp_path / "local.yaml", _local_config(tmp_path)), contract)

    assert isinstance(loaded, LocalP6CoptV2XConfig)
    assert loaded.source_registry_step.name == "build_source_registry"
    assert loaded.measurement_step.name == "measure_batch"
    assert loaded.local_output_root == tmp_path / "private-output"


def test_load_local_config_rejects_relative_asset_and_output_paths(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    relative_root_payload = _local_config(tmp_path, local_output_root="relative")
    relative_asset_payload = _local_config(tmp_path)
    relative_asset_payload["asset_paths"]["training-data"] = "relative"

    with pytest.raises(P6CoptV2XContractError, match="absolute path"):
        load_local_config(_write_yaml(tmp_path / "relative-root.yaml", relative_root_payload), contract)
    with pytest.raises(P6CoptV2XContractError, match="absolute path"):
        load_local_config(_write_yaml(tmp_path / "relative-asset.yaml", relative_asset_payload), contract)


@pytest.mark.parametrize(
    "key",
    ["candidate_registry", "measurements", "result_path_template", "steps", "result_step"],
)
def test_load_local_config_rejects_legacy_stage5_input_and_result_keys(
    tmp_path: Path, key: str
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    payload = _local_config(tmp_path)
    payload[key] = "legacy"

    with pytest.raises(P6CoptV2XContractError, match="unknown|forbidden"):
        load_local_config(_write_yaml(tmp_path / "local.yaml", payload), contract)


def test_load_local_config_rejects_shell_or_unknown_template_tokens(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    shell_payload = _local_config(
        tmp_path,
        measurement_step={
            "name": "measure_batch",
            "argv": ["bash", "-c", "private", "{measurement_request}"],
        },
    )
    token_payload = _local_config(
        tmp_path,
        measurement_step={
            "name": "measure_batch",
            "argv": ["python", "local_measure.py", "{candidate_id}"],
        },
    )

    with pytest.raises(P6CoptV2XContractError, match="shell executable"):
        load_local_config(_write_yaml(tmp_path / "shell.yaml", shell_payload), contract)
    with pytest.raises(P6CoptV2XContractError, match="template"):
        load_local_config(_write_yaml(tmp_path / "token.yaml", token_payload), contract)
