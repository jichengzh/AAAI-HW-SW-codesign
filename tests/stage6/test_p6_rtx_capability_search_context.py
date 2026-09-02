from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6.coptv2x_h800_search_v2 import (
    P6CoptV2XContractError,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_capability_context_v1 import (
    build_rtx_capability_context,
    canonical_probe_code_sha256,
    capability_context_to_mapping,
    compiler_fingerprint,
)
from framework.stage6.p6_capability_runtime_authority_v1 import (
    P6CapabilityRuntimeAuthorityError,
)
from framework.stage6.p6_history_normalization_v1 import normalize_history_inputs
from tools.release import probe_p6_rtx_capability as probe_cli
from tests.stage6.test_coptv2x_h800_search import (
    _closure,
    _gold176,
    _local_v3_config,
    _non_target_profile,
    _profile,
    _public_v3_contract,
    _write_profile_search_inputs,
    _write_feedback_from_request,
    _write_source_registry,
    _write_yaml,
)
from tests.stage6.test_p6_capability_context import (
    historical_profiles,
    historical_source_bytes,
    probe_evidence,
)
from tests.stage6.test_p6_history_normalization import _history_root
from tests.stage6.test_p6_post_source_adapter_profile import v5_private_source_map


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _historical_source_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        execution,
        "historical_capability_source_sha256",
        lambda root: hashlib.sha256(historical_source_bytes()).hexdigest(),
    )

    def rebuild(**kwargs: Any) -> SimpleNamespace:
        context = json.loads(
            kwargs["capability_context_path"].read_text(encoding="utf-8")
        )
        evidence = context["measurement_evidence"]
        return SimpleNamespace(
            runtime_identity=evidence["runtime_identity"],
            probe_records={
                "neutral": evidence["neutral_records"],
                "pruning": evidence["pruning_records"],
            },
        )

    monkeypatch.setattr(execution, "probe_normalized_capability_authority", rebuild)


def _write_rtx_context_source(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    source_map, runner = v5_private_source_map(tmp_path)
    source_map["hardware_profile"] = "rtx4090"
    support = next(
        item
        for item in source_map["execution_code_closure"]["roots"]
        if item["closure_id"] == "tvm-support"
    )
    runtime_site = tmp_path / "approved-runtime" / "site-packages"
    runtime_site.mkdir(parents=True)
    nvlibs = tmp_path / "approved-runtime" / "nvlibs.path"
    nvlibs.write_text("/runtime/lib\n", encoding="utf-8")
    runner_payload = yaml.safe_load(runner.read_text(encoding="utf-8"))
    runner_payload["execution_interface"]["environment"]["values"][
        "CUDA_VISIBLE_DEVICES"
    ]["value"] = "0,1,2,3"
    runner_payload["execution_interface"]["environment"]["values"].update(
        {
            "P6_TVM_PYTHON": {
                "kind": "external_executable",
                "value": "/usr/bin/python3.10",
            },
            "P6_TVM_SITE": {
                "kind": "external_directory",
                "value": str(runtime_site),
            },
            "P6_TVM_NVLIBS_FILE": {
                "kind": "external_file",
                "value": str(nvlibs),
            },
            "P6_TVM_SUPPORT_ROOT": {
                "kind": "private_path",
                "value": support["source_root"],
            },
            "P6_TVM_SUPPORT_ROOT_SHA256": {
                "kind": "literal",
                "value": support["sha256"],
            },
        }
    )
    runner.write_text(yaml.safe_dump(runner_payload, sort_keys=False), encoding="utf-8")
    evidence = probe_evidence()
    evidence["probe_code_sha256"] = canonical_probe_code_sha256(REPOSITORY_ROOT)
    evidence["runtime_identity"]["support_root_sha256"] = support["sha256"]
    runtime = dict(evidence["runtime_identity"])
    runtime.pop("compiler_fingerprint")
    evidence["runtime_identity"]["compiler_fingerprint"] = compiler_fingerprint(runtime)
    source_bytes = historical_source_bytes()
    profiles = historical_profiles()
    context = capability_context_to_mapping(
        build_rtx_capability_context(
            historical_source_bytes=source_bytes,
            evidence=evidence,
            profile=execution.load_hardware_execution_profile("rtx4090"),
            repository_root=REPOSITORY_ROOT,
            expected_probe_code_sha256=canonical_probe_code_sha256(REPOSITORY_ROOT),
            expected_support_root_sha256=support["sha256"],
            expected_historical_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            trusted_runtime_identity=evidence["runtime_identity"],
            trusted_probe_records={
                "neutral": evidence["neutral_records"],
                "pruning": evidence["pruning_records"],
            },
        )
    )
    source_path = _history_root(source_map) / "rtx-capability-context.json"
    source_path.write_text(json.dumps(context), encoding="utf-8")
    source_map["input_sources"]["capability_profiles"] = str(source_path)
    rows, graphs = _gold176(include_non_target_backend=True)
    for name, payload in (
        ("gold176_rows", rows),
        ("gold176_graph_features", graphs),
        ("closure", _closure()),
    ):
        Path(source_map["input_sources"][name]).write_text(json.dumps(payload), encoding="utf-8")
    normalized = tmp_path / "normalized"
    normalize_history_inputs(
        source_map,
        _history_root(source_map),
        normalized,
        runner_template_path=runner,
    )
    return normalized, profiles


def _loaded_pair(tmp_path: Path):
    normalized, profiles = _write_rtx_context_source(tmp_path)
    contract = load_public_contract(
        _write_yaml(tmp_path / "public.yaml", _public_v3_contract("rtx4090"))
    )
    local_payload = _local_v3_config(
        tmp_path,
        "rtx4090",
        local_input_paths={
            name: str(normalized / "inputs" / f"{name}.json")
            for name in (
                "gold176_rows",
                "gold176_graph_features",
                "capability_profiles",
                "closure",
            )
        },
    )
    local = load_local_config(_write_yaml(tmp_path / "local.yaml", local_payload), contract)
    return contract, local, profiles


def test_rtx_loader_preserves_gold_profiles_and_adds_one_measured_active_context(
    tmp_path: Path,
) -> None:
    contract, local, historical = _loaded_pair(tmp_path)

    _, _, model_profiles, active = execution._load_search_inputs(local, contract)

    assert model_profiles[:2] == historical
    assert model_profiles[2] == active
    assert active["hardware_target"] == "rtx4090"
    assert active["dispatch_key"] == "tvm_auto"


def test_rtx_round_zero_fits_gold_only_and_prediction_uses_active_context(
    tmp_path: Path,
) -> None:
    contract, local, historical = _loaded_pair(tmp_path)
    frozen, _, model_profiles, active = execution._load_search_inputs(local, contract)
    task = execution._build_search_task(contract, active)

    initial_fit, initial_prediction = execution._round_capability_profiles(
        model_profiles, frozen_gold=frozen, task=task, round_index=0
    )
    online_fit, online_prediction = execution._round_capability_profiles(
        model_profiles, frozen_gold=frozen, task=task, round_index=1
    )

    assert initial_fit == historical
    assert initial_prediction == [active]
    assert online_fit == [*historical, active]
    assert online_prediction == [active]
    assert active["capability_profile_id"] not in {row["capability_profile_id"] for row in frozen}


def test_rtx_loader_rejects_legacy_relabelled_profiles(tmp_path: Path) -> None:
    rows, graphs = _gold176(include_non_target_backend=True, hardware_target="rtx4090")
    _write_profile_search_inputs(
        tmp_path,
        rows=rows,
        graphs=graphs,
        profiles=[_profile("rtx4090"), _non_target_profile("rtx4090")],
    )
    contract = load_public_contract(
        _write_yaml(tmp_path / "public.yaml", _public_v3_contract("rtx4090"))
    )
    local = load_local_config(
        _write_yaml(tmp_path / "local.yaml", _local_v3_config(tmp_path, "rtx4090")),
        contract,
    )
    with pytest.raises(P6CoptV2XContractError, match="capability context"):
        execution._load_search_inputs(local, contract)


def test_rtx_loader_fails_closed_without_external_runtime_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, local, _ = _loaded_pair(tmp_path)
    monkeypatch.setattr(
        execution,
        "probe_normalized_capability_authority",
        lambda **kwargs: (_ for _ in ()).throw(P6CapabilityRuntimeAuthorityError()),
    )

    with pytest.raises(P6CoptV2XContractError, match="capability context"):
        execution._load_search_inputs(local, contract)


def test_probe_cli_requires_exact_normalized_five_key_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normalized, _ = _write_rtx_context_source(tmp_path)
    runner_path = normalized / "runner-template.yaml"
    profile_path = normalized / "post-source-adapter-profile.yaml"
    runner = yaml.safe_load(runner_path.read_text(encoding="utf-8"))
    values = runner["execution_interface"]["environment"]["values"]
    runtime = {
        key: values[key]["value"]
        for key in (
            "P6_TVM_PYTHON",
            "P6_TVM_SITE",
            "P6_TVM_NVLIBS_FILE",
            "P6_TVM_SUPPORT_ROOT",
            "P6_TVM_SUPPORT_ROOT_SHA256",
        )
    }
    for key, value in runtime.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(probe_cli.sys, "executable", runtime["P6_TVM_PYTHON"])

    _, validated = probe_cli._runtime_values(normalized, profile_path, runner_path)
    assert validated == runtime

    for missing in runtime:
        monkeypatch.delenv(missing)
        with pytest.raises(ValueError, match="runtime"):
            probe_cli._runtime_values(normalized, profile_path, runner_path)
        monkeypatch.setenv(missing, runtime[missing])


def test_rtx_controller_fit_predict_and_measurement_context_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, local, historical = _loaded_pair(tmp_path)
    initial_profiles: list[list[str]] = []
    online_profiles: list[list[str]] = []
    prediction_profiles: list[list[str]] = []
    requests: list[dict[str, Any]] = []
    real_initial = execution.fit_initial_coldstart_bundle
    real_online = execution.fit_online_bundle
    real_predict = execution.predict_candidate_rows

    def initial(rows, graphs, profiles, **kwargs):
        initial_profiles.append([item["hardware_target"] for item in profiles])
        return real_initial(rows, graphs, profiles, **kwargs)

    def online(rows, graphs, profiles, **kwargs):
        online_profiles.append([item["hardware_target"] for item in profiles])
        return real_online(rows, graphs, profiles, **kwargs)

    def predict(bundle, rows, profiles):
        prediction_profiles.append([item["hardware_target"] for item in profiles])
        return real_predict(bundle, rows, profiles)

    monkeypatch.setattr(execution, "fit_initial_coldstart_bundle", initial)
    monkeypatch.setattr(execution, "fit_online_bundle", online)
    monkeypatch.setattr(execution, "predict_candidate_rows", predict)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        if argv[1] == "local_build_registry.py":
            _write_source_registry(Path(argv[3]), count=343)
        else:
            request = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            requests.append(request)
            _write_feedback_from_request(Path(argv[2]), Path(argv[3]))
        return 0

    state = execution.run_p6_coptv2x_search(contract, local, "abc123", runner)

    assert state.status == "completed"
    assert initial_profiles == [["h800", "h800"]]
    assert online_profiles == [["h800", "h800", "rtx4090"]] * 3
    assert prediction_profiles == [["rtx4090"]] * 4
    assert all(
        row["hardware_id"] == "rtx4090"
        and row["capability_profile_id"] == "rtx4090-tvm-auto-probe-v1"
        for request in requests
        for row in request["rows"]
    )
    assert len({row["row_id"] for request in requests for row in request["rows"]}) == 16
    assert [item["capability_digest"] for item in historical] == [
        item["capability_digest"]
        for item in json.loads(
            (local.local_input_paths["capability_profiles"]).read_text(encoding="utf-8")
        )["historical_profiles"]
    ]
