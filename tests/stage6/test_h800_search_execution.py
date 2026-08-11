from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage6 import h800_search_execution_v1 as execution
from framework.stage6.h800_search_execution_v1 import (
    H800SearchContractError,
    H800SearchSummary,
    load_local_config,
    load_public_contract,
    run_h800_search,
)


def _write_yaml(path: Path, content: Mapping[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(dict(content), sort_keys=False), encoding="utf-8")
    return path


def _public_contract(**overrides: Any) -> dict[str, Any]:
    contract = {
        "schema_version": "p6_h800_search_contract_v1",
        "search_id": "p6-h800-example",
        "target": "h800",
        "target_model": "pyramid",
        "seed": 73,
        "max_rounds": 2,
        "batch_size": 2,
        "configuration_label": "p6-h800-baseline",
        "metric_names": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "candidate_space_label": "pyramid-v1",
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }
    return {**contract, **overrides}


def _local_config(*, asset_label: str = "training-data", **overrides: Any) -> dict[str, Any]:
    config = {
        "schema_version": "p6_h800_search_local_v1",
        "target": "h800",
        "asset_paths": {
            asset_label: "/private/p6/training-data",
            "model-init": "/private/p6/model-init",
            "toolchain": "/private/p6/toolchain",
        },
        "stage5_input_paths": {"feedback": "/private/stage5/feedback.json"},
        "steps": [
            {"name": "train", "argv": ["python", "train.py", "{candidate_request}"]},
            {
                "name": "evaluate",
                "argv": ["python", "evaluate.py", "{candidate_request}", "{result_json}"],
            },
        ],
        "result_step": "evaluate",
        "result_path_template": "/private/p6/results/final.json",
        "local_output_root": "/private/p6/output",
    }
    return {**config, **overrides}


def test_load_public_contract_requires_h800_and_public_fields(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "contract.yaml", _public_contract(target="orin"))

    with pytest.raises(H800SearchContractError, match="target must be h800"):
        load_public_contract(path)


def test_load_public_contract_rejects_an_invalid_schema_version(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "contract.yaml",
        _public_contract(schema_version="p6_h800_search_contract_v0"),
    )

    with pytest.raises(H800SearchContractError, match="schema_version"):
        load_public_contract(path)


@pytest.mark.parametrize(
    "metric_names",
    [
        ["latency_ms", "energy_j", "ap30", "ap50"],
        ["latency_ms", "energy_j", "ap30", "ap50", "ap70", "throughput"],
        ["latency_ms", "energy_j", "ap30", "ap50", "ap70", "latency_ms"],
    ],
)
def test_load_public_contract_requires_the_exact_runtime_metric_set(
    tmp_path: Path, metric_names: list[str]
) -> None:
    path = _write_yaml(
        tmp_path / "contract.yaml", _public_contract(metric_names=metric_names)
    )

    with pytest.raises(H800SearchContractError, match="metric_names"):
        load_public_contract(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_id", None),
        ("configuration_label", None),
        ("target_model", None),
        ("seed", 0),
        ("max_rounds", 0),
        ("batch_size", 0),
        ("metric_names", []),
        ("assets", []),
    ],
)
def test_load_public_contract_requires_complete_public_fields(
    tmp_path: Path, field: str, value: Any
) -> None:
    contract = _public_contract()
    if value is None:
        del contract[field]
    else:
        contract[field] = value

    with pytest.raises(H800SearchContractError):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", contract))


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("max_rounds", "2"),
        ("batch_size", -1),
        ("metric_names", ["ap", "ap"]),
        (
            "assets",
            [
                {"label": "training-data", "version": "v1", "license_status": "cleared"},
                {"label": "training-data", "version": "v2", "license_status": "cleared"},
            ],
        ),
        (
            "assets",
            [{"label": "", "version": "v1", "license_status": "cleared"}],
        ),
    ],
)
def test_load_public_contract_rejects_malformed_declared_values(
    tmp_path: Path, field: str, value: Any
) -> None:
    with pytest.raises(H800SearchContractError):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract(**{field: value})))


@pytest.mark.parametrize(
    "forbidden_key",
    ["path", "command", "host", "candidate_id", "sha256", "raw_logs", "checkpoint"],
)
def test_load_public_contract_rejects_private_execution_details(
    tmp_path: Path, forbidden_key: str
) -> None:
    contract = _public_contract()
    contract[forbidden_key] = "private-value"

    with pytest.raises(H800SearchContractError, match="forbidden"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", contract))


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_id", "/private/run"),
        ("configuration_label", "h800-host-01"),
        ("configuration_label", "raw-logs-2026"),
        ("configuration_label", "candidate-42"),
        ("target_model", "python train.py"),
    ],
)
def test_load_public_contract_rejects_restricted_information_in_allowed_strings(
    tmp_path: Path, field: str, value: str
) -> None:
    with pytest.raises(H800SearchContractError, match="restricted"):
        load_public_contract(
            _write_yaml(tmp_path / "contract.yaml", _public_contract(**{field: value}))
        )


@pytest.mark.parametrize(
    "value",
    ["run python train.py", "train.py --out result.json", "echo secret"],
)
def test_load_public_contract_rejects_embedded_command_style_strings(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(H800SearchContractError, match="restricted"):
        load_public_contract(
            _write_yaml(tmp_path / "contract.yaml", _public_contract(configuration_label=value))
        )


@pytest.mark.parametrize("value", ["candidate-alpha", "candidate-id-alpha"])
def test_load_public_contract_rejects_non_numeric_candidate_identifiers(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(H800SearchContractError, match="restricted"):
        load_public_contract(
            _write_yaml(tmp_path / "contract.yaml", _public_contract(configuration_label=value))
        )


@pytest.mark.parametrize("value", [f"build-{'a' * 64}", f"v1-{'b' * 40}"])
def test_load_public_contract_rejects_embedded_raw_digests(tmp_path: Path, value: str) -> None:
    with pytest.raises(H800SearchContractError, match="restricted"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract(target_model=value)))


@pytest.mark.parametrize(
    "asset",
    [
        {"label": "checkpoint", "version": "v1", "license_status": "cleared"},
        {"label": "training-data", "version": "sha256:abc123", "license_status": "cleared"},
        {
            "label": "training-data",
            "version": "a" * 64,
            "license_status": "cleared",
        },
    ],
)
def test_load_public_contract_rejects_restricted_information_in_asset_values(
    tmp_path: Path, asset: dict[str, str]
) -> None:
    with pytest.raises(H800SearchContractError, match="restricted"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract(assets=[asset])))


def test_load_public_contract_rejects_unknown_and_non_mapping_roots(tmp_path: Path) -> None:
    with pytest.raises(H800SearchContractError, match="unknown"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract(extra="value")))

    path = tmp_path / "list.yaml"
    path.write_text("- not-a-contract\n", encoding="utf-8")
    with pytest.raises(H800SearchContractError, match="mapping"):
        load_public_contract(path)


def test_load_local_config_rejects_asset_label_mismatch(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    path = _write_yaml(tmp_path / "local.yaml", _local_config(asset_label="unregistered"))

    with pytest.raises(H800SearchContractError, match="asset labels"):
        load_local_config(path, contract)


def test_load_local_config_rejects_shell_string_step(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    path = _write_yaml(
        tmp_path / "local.yaml",
        _local_config(steps=[{"name": "evaluate", "argv": "python train.py"}]),
    )

    with pytest.raises(H800SearchContractError, match="argv"):
        load_local_config(path, contract)


@pytest.mark.parametrize(
    "executable",
    ["bash", "/bin/sh", "C:\\Windows\\System32\\cmd.exe", "PowerShell", "pwsh"],
)
def test_load_local_config_rejects_shell_executables(
    tmp_path: Path, executable: str
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    path = _write_yaml(
        tmp_path / "local.yaml",
        _local_config(
            steps=[
                {
                    "name": "evaluate",
                    "argv": [executable, "-c", "private-command", "{result_json}"],
                }
            ]
        ),
    )

    with pytest.raises(H800SearchContractError, match="shell executable"):
        load_local_config(path, contract)


@pytest.mark.parametrize(
    "argv",
    [
        ["/usr/bin/env", "bash", "-c", "private-command", "{result_json}"],
        ["ENV", "sh", "-c", "private-command", "{result_json}"],
        [
            "C:\\Windows\\System32\\env.exe",
            "PowerShell",
            "-Command",
            "private-command",
            "{result_json}",
        ],
    ],
)
def test_load_local_config_rejects_env_command_wrappers(
    tmp_path: Path, argv: list[str]
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    path = _write_yaml(
        tmp_path / "local.yaml",
        _local_config(steps=[{"name": "evaluate", "argv": argv}]),
    )

    with pytest.raises(H800SearchContractError, match="command wrapper"):
        load_local_config(path, contract)


def test_load_local_config_allows_direct_non_shell_argv(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    path = _write_yaml(
        tmp_path / "local.yaml",
        _local_config(
            steps=[
                {
                    "name": "evaluate",
                    "argv": ["/usr/bin/python3", "evaluate.py", "{result_json}"],
                }
            ]
        ),
    )

    loaded = load_local_config(path, contract)

    assert loaded.steps[0].argv[0] == "/usr/bin/python3"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "p6_h800_search_local_v0"),
        ("target", "a100"),
        ("asset_paths", {"training-data": "relative/data"}),
        ("stage5_input_paths", {"feedback": "relative/feedback.json"}),
        ("result_path_template", "relative/result.json"),
        ("local_output_root", "relative/output"),
        ("steps", []),
        ("result_step", "train"),
    ],
)
def test_load_local_config_rejects_contract_and_path_violations(
    tmp_path: Path, field: str, value: Any
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))

    with pytest.raises(H800SearchContractError):
        load_local_config(
            _write_yaml(tmp_path / "local.yaml", _local_config(**{field: value})), contract
        )


def test_load_local_config_rejects_duplicate_steps_unknown_placeholders_and_unknown_keys(
    tmp_path: Path,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    malformed_configs = (
        _local_config(
            steps=[
                {"name": "evaluate", "argv": ["python", "evaluate.py"]},
                {"name": "evaluate", "argv": ["python", "evaluate.py", "{result_json}"]},
            ],
            result_step="evaluate",
        ),
        _local_config(
            steps=[
                {"name": "evaluate", "argv": ["python", "evaluate.py", "--out={result_json}"]}
            ]
        ),
        _local_config(extra="value"),
    )

    for index, config in enumerate(malformed_configs):
        with pytest.raises(H800SearchContractError):
            load_local_config(_write_yaml(tmp_path / f"local-{index}.yaml", config), contract)


def test_load_local_config_is_pure_and_returns_immutable_values(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    output_root = tmp_path / "not-created"
    config = _local_config(
        asset_paths={
            "training-data": "/unavailable/training-data",
            "model-init": "/unavailable/model-init",
            "toolchain": "/unavailable/toolchain",
        },
        stage5_input_paths={"feedback": "/unavailable/feedback.json"},
        result_path_template="/unavailable/final.json",
        local_output_root=str(output_root),
    )

    loaded = load_local_config(_write_yaml(tmp_path / "local.yaml", config), contract)

    assert not output_root.exists()
    assert loaded.asset_paths["training-data"] == Path("/unavailable/training-data")
    assert loaded.steps[-1].argv[-1] == "{result_json}"
    with pytest.raises(TypeError):
        loaded.asset_paths["new"] = Path("/unavailable/new")  # type: ignore[index]


def _profiles() -> list[dict[str, Any]]:
    return [
        build_capability_profile(
            capability_profile_id=f"h800-{dispatch}",
            hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(dispatch.encode()).hexdigest(),
            dispatch_key=dispatch,
            features={"int8_propagation": propagation, "qdq_fold": propagation / 2},
        )
        for dispatch, propagation in (("tvm_auto", 0.0), ("trt_engine", 1.0))
    ]


def _graph(group_id: str, model: str, width: list[int]) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "model": model,
        "width": list(width),
        "conv_count": 24 if model == "codriving" else 27,
        "conv_macs": float(math.prod(width)),
        "group_conv_count": 0 if model == "codriving" else 3,
    }


def _source_lineage(group_id: str, model: str, width: list[int]) -> dict[str, Any]:
    evidence_sha = hashlib.sha256(f"p6-test:{group_id}".encode()).hexdigest()
    contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": group_id,
        "model": model,
        "width": list(width),
        "artifact_id": f"public-{model}-{width[0]}",
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "external_public_test_fixture",
    }
    contract_sha = hashlib.sha256(
        json.dumps(
            contract, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "source_contract": contract,
        "source_contract_sha256": contract_sha,
    }


def _stage5_inputs() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    widths = ([16, 32, 64], [24, 48, 96], [32, 64, 128])
    profiles = _profiles()
    for model_index, model in enumerate(("pyramid", "codriving")):
        for width_index, width in enumerate(widths):
            group_id = f"{model}|{'x'.join(map(str, width))}"
            graphs.append(_graph(group_id, model, list(width)))
            lineage = _source_lineage(group_id, model, list(width))
            for profile in profiles:
                for q_mode in ("fp16", "int8"):
                    latency = 1.0 + model_index + width_index * 0.2
                    ap70 = 0.76 - width_index * 0.03 - (0.01 if q_mode == "int8" else 0)
                    rows.append(
                        {
                            "manifest_job_id": (
                                f"{group_id}|q={q_mode}|profile="
                                f"{profile['capability_profile_id']}"
                            ),
                            "group_id": group_id,
                            "model": model,
                            "width": list(width),
                            "dispatch_key": profile["dispatch_key"],
                            "capability_profile_id": profile["capability_profile_id"],
                            "q_mode": q_mode,
                            "source_contract_sha256": lineage["source_contract_sha256"],
                            "latency_ms": latency,
                            "energy_j": latency * 0.25,
                            "ap30": min(1.0, ap70 + 0.2),
                            "ap50": min(1.0, ap70 + 0.1),
                            "ap70": ap70,
                            "terminal_status": "measured_success_gold",
                            "training_source": (
                                "initial_coldstart" if width_index < 2 else "online_feedback"
                            ),
                        }
                    )

    candidate_groups = []
    for width in ([40, 80, 160], [48, 96, 192]):
        group_id = f"pyramid|{'x'.join(map(str, width))}"
        candidate_groups.append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": list(width),
                **_source_lineage(group_id, "pyramid", list(width)),
                "graph_features": _graph(group_id, "pyramid", list(width)),
            }
        )
    closure = {
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
        "training_source_rows": {"initial_coldstart": 16, "online_feedback": 8},
        "frozen_holdout": {"groups": []},
    }
    return {
        "measurements": rows,
        "candidate_registry": {
            "schema_version": "stage5_candidate_source_registry_v1",
            "groups": candidate_groups,
        },
        "graph_features": graphs,
        "capability_profiles": profiles,
        "closure": closure,
    }


def _loaded_contract(tmp_path: Path, **overrides: Any):
    return load_public_contract(
        _write_yaml(
            tmp_path / "contract.yaml",
            _public_contract(
                metric_names=["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
                **overrides,
            ),
        )
    )


def _loaded_local_config(
    tmp_path: Path,
    *,
    output_root: Path | None = None,
    steps: list[dict[str, Any]] | None = None,
):
    inputs = _stage5_inputs()
    input_paths: dict[str, str] = {}
    for name, payload in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        input_paths[name] = str(path)
    asset_paths: dict[str, str] = {}
    for label in ("training-data", "model-init", "toolchain"):
        path = tmp_path / label
        path.mkdir()
        asset_paths[label] = str(path)
    config = _local_config(
        asset_paths=asset_paths,
        stage5_input_paths=input_paths,
        steps=steps
        or [
            {
                "name": "evaluate",
                "argv": ["fake-evaluate", "{candidate_request}", "{result_json}"],
            }
        ],
        result_step="evaluate",
        result_path_template=str(tmp_path / "result.json"),
        local_output_root=str(output_root or tmp_path / "output"),
    )
    contract = _loaded_contract(tmp_path)
    return load_local_config(_write_yaml(tmp_path / "local.yaml", config), contract)


def _successful_result(request: Mapping[str, Any]) -> dict[str, Any]:
    feedback_count = int(request["feedback_count"])
    return {
        "candidate_id": request["candidate_id"],
        "measurements": [
            {
                "candidate_id": row["row_id"],
                "latency_ms": 2.0 + feedback_count / 100,
                "energy_j": 0.5 + feedback_count / 400,
                "ap30": 0.90,
                "ap50": 0.80,
                "ap70": 0.70,
            }
            for row in request["rows"]
        ],
    }


def _write_successful_measurement(seen_requests: list[dict[str, Any]]):
    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        seen_requests.append(request)
        Path(argv[2]).write_text(json.dumps(_successful_result(request)), encoding="utf-8")
        return 0

    return runner


def test_run_h800_search_feeds_each_round_back_before_next_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_requests: list[dict[str, Any]] = []
    measured_row_counts: list[int] = []
    online_feedback_counts: list[int] = []
    real_fit = execution.fit_production_bundle

    def recording_fit(rows: list[Mapping[str, Any]], *args: Any, **kwargs: Any):
        measured_row_counts.append(len(rows))
        online_feedback_counts.append(
            sum(row.get("training_source") == "online_feedback" for row in rows)
        )
        return real_fit(rows, *args, **kwargs)

    monkeypatch.setattr(execution, "fit_production_bundle", recording_fit)

    summary = run_h800_search(
        contract=_loaded_contract(tmp_path, max_rounds=2, batch_size=1),
        local=_loaded_local_config(tmp_path),
        code_revision="abc123",
        command_runner=_write_successful_measurement(seen_requests),
    )

    assert isinstance(summary, H800SearchSummary)
    assert summary.status == "completed"
    assert summary.completed_rounds == 2
    assert summary.successful_candidate_count == 2
    assert len(seen_requests) == 2
    assert seen_requests[1]["feedback_count"] > seen_requests[0]["feedback_count"]
    assert measured_row_counts == [24, 28]
    assert online_feedback_counts == [8, 12]
    assert summary.aggregate_metrics["latency_ms"]["count"] == 8.0
    assert summary.failure_code is None


def test_run_h800_search_rejects_non_h800_capability_profiles_before_selection(
    tmp_path: Path,
) -> None:
    local = _loaded_local_config(tmp_path)
    non_h800_profiles = [
        build_capability_profile(
            capability_profile_id=profile["capability_profile_id"],
            hardware_target="rtx4090",
            compiler_fingerprint=profile["compiler_fingerprint"],
            dispatch_key=profile["dispatch_key"],
            features=profile["features"],
        )
        for profile in _profiles()
    ]
    local.stage5_input_paths["capability_profiles"].write_text(
        json.dumps(non_h800_profiles), encoding="utf-8"
    )
    calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal calls
        del argv, cwd
        calls += 1
        return 0

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=1, batch_size=1),
        local,
        "abc123",
        runner,
    )

    assert summary.status == "failed"
    assert summary.failure_code == "capability_target_invalid"
    assert summary.completed_rounds == 0
    assert calls == 0


def test_run_h800_search_stops_after_a_nonzero_command(tmp_path: Path) -> None:
    calls = 0
    local = _loaded_local_config(tmp_path)

    def failing_runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal calls
        del argv, cwd
        calls += 1
        return 9

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=2, batch_size=1),
        local,
        "abc123",
        failing_runner,
    )

    assert summary.status == "failed"
    assert summary.failure_code == "command_failed"
    assert summary.completed_rounds == 0
    assert summary.successful_candidate_count == 0
    assert calls == 1
    failure_record = json.loads(
        (local.local_output_root / "failure.json").read_text(encoding="utf-8")
    )
    assert failure_record["failure_code"] == "command_failed"
    assert isinstance(failure_record["detail"], str)


def test_run_h800_search_requires_the_result_step_to_produce_the_result(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    local = _loaded_local_config(
        tmp_path,
        steps=[
            {"name": "train", "argv": ["fake-train", "{candidate_request}", "{result_json}"]},
            {
                "name": "evaluate",
                "argv": ["fake-evaluate", "{candidate_request}", "{result_json}"],
            },
        ],
    )

    def stale_result_runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        calls.append(argv[0])
        if argv[0] == "fake-train":
            request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
            Path(argv[2]).write_text(json.dumps(_successful_result(request)), encoding="utf-8")
        return 0

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=1, batch_size=1),
        local,
        "abc123",
        stale_result_runner,
    )

    assert summary.status == "failed"
    assert summary.failure_code == "result_missing"
    assert calls == ["fake-train", "fake-evaluate"]


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("missing", "result_missing"),
        ("invalid_json", "result_invalid_json"),
        ("candidate_mismatch", "result_candidate_mismatch"),
        ("missing_metric", "result_metrics_invalid"),
        ("nan", "result_metrics_invalid"),
        ("infinity", "result_metrics_invalid"),
    ],
)
def test_run_h800_search_rejects_invalid_results_and_stops(
    tmp_path: Path, case: str, expected_code: str
) -> None:
    calls = 0

    def malformed_runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal calls
        del cwd
        calls += 1
        request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        result_path = Path(argv[2])
        if case == "missing":
            return 0
        if case == "invalid_json":
            result_path.write_text("not-json", encoding="utf-8")
            return 0
        result = _successful_result(request)
        if case == "candidate_mismatch":
            result["candidate_id"] = "wrong-candidate"
        elif case == "missing_metric":
            del result["measurements"][0]["ap50"]
        elif case == "nan":
            result["measurements"][0]["latency_ms"] = float("nan")
        elif case == "infinity":
            result["measurements"][0]["energy_j"] = float("inf")
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=2, batch_size=1),
        _loaded_local_config(tmp_path),
        "abc123",
        malformed_runner,
    )

    assert summary.status == "failed"
    assert summary.failure_code == expected_code
    assert summary.completed_rounds == 0
    assert summary.successful_candidate_count == 0
    assert calls == 1


def test_run_h800_search_fails_when_stage5_selection_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal calls
        del argv, cwd
        calls += 1
        return 0

    monkeypatch.setattr(
        execution,
        "select_predicted_frontier_diversity",
        lambda *args, **kwargs: {"selected_group_ids": [], "groups": []},
    )

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=2, batch_size=1),
        _loaded_local_config(tmp_path),
        "abc123",
        runner,
    )

    assert summary.status == "failed"
    assert summary.failure_code == "stage5_selection_failed"
    assert summary.completed_rounds == 0
    assert summary.successful_candidate_count == 0
    assert calls == 0


@pytest.mark.parametrize("mutation", ["missing_arm", "duplicate_arm"])
def test_run_h800_search_rejects_incomplete_or_duplicate_selected_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    real_select = execution.select_predicted_frontier_diversity

    def invalid_selection(*args: Any, **kwargs: Any) -> dict[str, Any]:
        selection = real_select(*args, **kwargs)
        rows = selection["groups"][0]["rows"]
        if mutation == "missing_arm":
            rows.pop()
        else:
            rows[1] = copy.deepcopy(rows[0])
        return selection

    monkeypatch.setattr(execution, "select_predicted_frontier_diversity", invalid_selection)
    calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal calls
        del argv, cwd
        calls += 1
        return 0

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=1, batch_size=1),
        _loaded_local_config(tmp_path),
        "abc123",
        runner,
    )

    assert summary.status == "failed"
    assert summary.failure_code == "stage5_selection_failed"
    assert summary.completed_rounds == 0
    assert summary.successful_candidate_count == 0
    assert calls == 0


def test_run_h800_search_converts_unexpected_stage5_errors_to_a_stable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_selection(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise RuntimeError("private Stage5 detail")

    monkeypatch.setattr(execution, "select_predicted_frontier_diversity", fail_selection)

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=2, batch_size=1),
        _loaded_local_config(tmp_path),
        "abc123",
        lambda argv, cwd: 0,
    )

    assert summary.status == "failed"
    assert summary.failure_code == "stage5_selection_failed"
    assert summary.completed_rounds == 0


@pytest.mark.parametrize("missing", ["asset", "stage5_input"])
def test_run_h800_search_rejects_missing_local_inputs_before_selection(
    tmp_path: Path, missing: str
) -> None:
    local = _loaded_local_config(tmp_path)
    if missing == "asset":
        local.asset_paths["training-data"].rmdir()
    else:
        local.stage5_input_paths["closure"].unlink()
    calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal calls
        del argv, cwd
        calls += 1
        return 0

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=2, batch_size=1),
        local,
        "abc123",
        runner,
    )

    assert summary.status == "failed"
    assert summary.failure_code == "local_input_invalid"
    assert summary.completed_rounds == 0
    assert calls == 0


def test_run_h800_search_fails_closed_when_local_record_cannot_be_written(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output-is-a-file"
    output_root.write_text("occupied", encoding="utf-8")
    calls = 0

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal calls
        del argv, cwd
        calls += 1
        return 0

    summary = run_h800_search(
        _loaded_contract(tmp_path, max_rounds=2, batch_size=1),
        _loaded_local_config(tmp_path, output_root=output_root),
        "abc123",
        runner,
    )

    assert summary.status == "failed"
    assert summary.failure_code == "local_record_write_failed"
    assert summary.completed_rounds == 0
    assert summary.successful_candidate_count == 0
    assert calls == 0
