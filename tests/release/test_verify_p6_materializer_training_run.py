"""Exact-evidence acceptance coverage for P6 completion verification."""

from __future__ import annotations

import json
import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Iterator
from unittest import mock

import pytest
import yaml

from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6.coptv2x_h800_search_v2 import P6CoptV2XExecutionError
from framework.stage6.p6_public_report_v1 import (
    P6HardwareSpecificReportProvenance,
    validate_hardware_specific_report_provenance,
)
from framework.stage6.p6_capability_context_v1 import compiler_fingerprint
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PROFILE_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION_V2,
    PROFILE_SCHEMA_VERSION_V3,
    PROFILE_SCHEMA_VERSION_V4,
    load_post_source_adapter_profile,
)
from framework.stage6.p6_history_measurement_v1 import (
    resolve_validated_history_round_paths,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (
    RUN_CONTEXT_RELATIVE_PATH,
    canonical_json_sha256,
)
from tools.release.verify_p6_materializer_training_run import (
    P6MaterializerCompletionReport,
    main,
    verify_materializer_training_run,
)
import tools.release.verify_p6_materializer_training_run as verification
from tests.release.test_run_p6_h800_search import (
    _fixture_runtime_authority,
    _history_cli_fixture,
    _install_measured_rtx_context,
    _run_cli,
)
from tools.release import run_p6_h800_search as search_cli
from tests.stage6.test_coptv2x_h800_search import (
    _local_v3_config,
    _public_v3_contract,
    _write_yaml,
)
from tests.stage6.p6_performance_native_fixture import (
    quant_contract_path,
    sha256_file,
)
from tests.stage6.test_p6_performance_round_adapter import _PerformanceRunner
from tests.stage6.test_p6_capability_context import historical_source_bytes
from tests.stage6.test_p6_rtx_capability_search_context import (
    _write_rtx_context_source,
)


@pytest.fixture(scope="module")
def _completed_template(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    workspace = tmp_path_factory.mktemp("p6-completion")
    live = workspace / "live"
    live.mkdir()
    paths = _history_cli_fixture(live)
    result = _run_cli(paths, env=paths["env"])
    assert result.returncode == 0, result.stderr
    _write_native_completion_leaves(paths)
    snapshot = workspace / "snapshot"
    shutil.copytree(live, snapshot, symlinks=True)
    yield {"live": live, "snapshot": snapshot, "paths": paths}


@pytest.fixture(scope="module")
def _completed_rtx_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Any]]:
    workspace = tmp_path_factory.mktemp("p6-rtx-completion")
    live = workspace / "live"
    live.mkdir()
    paths = _history_cli_fixture(live, hardware_profile="rtx4090")
    _install_measured_rtx_context(workspace, paths)
    with (
        mock.patch.object(
            execution,
            "historical_capability_source_sha256",
            lambda root: hashlib.sha256(historical_source_bytes()).hexdigest(),
        ),
        mock.patch.object(
            execution,
            "probe_normalized_capability_authority",
            _fixture_runtime_authority,
        ),
        mock.patch.dict("os.environ", paths["env"], clear=True),
    ):
        result = search_cli.main(
            [
                "--contract",
                str(paths["contract"]),
                "--local-config",
                str(paths["local"]),
                "--code-revision",
                "test-revision",
            ]
        )
    assert result == 0
    _write_native_completion_leaves(paths)
    snapshot = workspace / "snapshot"
    shutil.copytree(live, snapshot, symlinks=True)
    yield {"live": live, "snapshot": snapshot, "paths": paths}


@pytest.fixture
def completed_run(_completed_template: dict[str, Any]) -> dict[str, Any]:
    live = _completed_template["live"]
    shutil.rmtree(live)
    shutil.copytree(_completed_template["snapshot"], live, symlinks=True)
    return dict(_completed_template["paths"])


@pytest.fixture
def completed_rtx_run(
    _completed_rtx_template: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    live = _completed_rtx_template["live"]
    shutil.rmtree(live)
    shutil.copytree(_completed_rtx_template["snapshot"], live, symlinks=True)
    monkeypatch.setattr(
        execution,
        "historical_capability_source_sha256",
        lambda root: hashlib.sha256(historical_source_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        execution,
        "probe_normalized_capability_authority",
        _fixture_runtime_authority,
    )
    return dict(_completed_rtx_template["paths"])


def _verify_kwargs(paths: dict[str, Any]) -> dict[str, Path]:
    return {
        "public_contract_path": paths["contract"],
        "local_config_path": paths["local"],
        "private_binding_path": paths["binding"],
    }


def _apply_completion_mutation(paths: dict[str, Any], mutation: str) -> None:
    output_root = paths["output_root"]
    context_path = output_root / RUN_CONTEXT_RELATIVE_PATH
    binding = _read_json(paths["binding"])
    private_root = Path(binding["private_root"])
    interface = binding["execution_interface"]

    if mutation == "binding_hardware_profile_drift":
        indices = binding["gpu_policy"]["indices"][:3]
        binding["target"]["hardware"] = "h800"
        binding["gpu_policy"] = {
            "indices": indices,
            "uuid_by_index": {
                str(index): binding["gpu_policy"]["uuid_by_index"][str(index)]
                for index in indices
            },
            "hardware_profile": "h800",
        }
        binding["execution_interface"]["environment"]["values"][
            "CUDA_VISIBLE_DEVICES"
        ]["value"] = ",".join(str(index) for index in indices)
        _write_json(paths["binding"], binding)
        return
    if mutation == "post_source_hardware_profile_drift":
        profile_path = private_root / "post-source-adapter-profile.yaml"
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        profile.update(
            {
                "hardware_profile": "h800",
                "target": {
                    "model": "pyramid",
                    "hardware": "h800",
                    "backend": "tvm_auto",
                },
            }
        )
        profile_path.write_text(
            yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
        )
        return
    if mutation == "candidate_plan_hardware_profile_drift":
        plan_path = output_root / "pyramid_candidate_plan.json"
        plan = _read_json(plan_path)
        plan["hardware_target"] = "h800"
        _write_json(plan_path, plan)
        context = _read_json(context_path)
        context["candidate_plan_sha256"] = canonical_json_sha256(plan)
        context["run_context_sha256"] = canonical_json_sha256(
            {
                name: value
                for name, value in context.items()
                if name != "run_context_sha256"
            }
        )
        _write_json(context_path, context)
        return
    if mutation == "post_source_profile_symlink":
        (private_root / "post-source-adapter-profile.yaml").symlink_to(
            private_root / "missing-post-source-profile.yaml"
        )
        return

    if mutation.startswith("performance_"):
        _mutate_performance_evidence(paths, private_root, interface, mutation)
        return

    if mutation == "context_missing":
        context_path.unlink()
        return
    if mutation == "context_hash_drift":
        context = _read_json(context_path)
        context["code_revision"] = "PRIVATE-COMPLETION-TOKEN"
        _write_json(context_path, context)
        return
    if mutation == "decoy_context":
        context_path.rename(context_path.with_name("plausible-run-context.json"))
        return

    receipt_paths = sorted(
        (output_root / ".p6-materializer-training-bridge-v1/group-receipts").glob(
            "*.json"
        )
    )
    receipt_path = next(
        path
        for path in receipt_paths
        if _read_json(path)["producer_round_index"] == 0
    )
    receipt = _read_json(receipt_path)
    if mutation == "group_receipt_missing":
        receipt_path.unlink()
        return
    if mutation == "decoy_receipt":
        receipt_path.rename(receipt_path.with_name("plausible-group-receipt.json"))
        return
    if mutation in {
        "producer_request_hash_drift",
        "producer_is_later_round",
        "producer_row_missing",
        "producer_row_hash_drift",
        "group_receipt_cross_run",
    }:
        key, value = {
            "producer_request_hash_drift": (
                "producer_measurement_request_sha256",
                "0" * 64,
            ),
            "producer_is_later_round": ("producer_round_index", 3),
            "producer_row_missing": (
                "producer_row_id",
                "PRIVATE-COMPLETION-TOKEN-missing-row",
            ),
            "producer_row_hash_drift": ("producer_row_sha256", "1" * 64),
            "group_receipt_cross_run": ("run_nonce", "2" * 64),
        }[mutation]
        receipt[key] = value
        receipt["receipt_sha256"] = canonical_json_sha256(
            {name: item for name, item in receipt.items() if name != "receipt_sha256"}
        )
        _write_json(receipt_path, receipt)
        return
    if mutation == "producer_request_missing":
        producer_round = receipt["producer_round_index"]
        producer_paths = _round_paths(paths, interface, private_root, producer_round)
        producer_paths["measurement_request"].unlink()
        return

    round_paths = _round_paths(paths, interface, private_root, 0)
    public_request_path = output_root / "round-00/measurement_request.json"
    private_request_path = round_paths["measurement_request"]
    if mutation == "public_private_request_mismatch":
        private_request = _read_json(private_request_path)
        private_request["PRIVATE-COMPLETION-TOKEN"] = True
        _write_json(private_request_path, private_request)
        return
    if mutation in {"duplicate_selected_row_id", "duplicate_measurement_identity"}:
        replacement = _read_json(output_root / "round-00/measurement_request.json")[
            "rows"
        ][0]
        _replace_round_row(paths, interface, private_root, 1, replacement)
        return
    if mutation == "gold176_overlap":
        local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
        gold_path = Path(local["local_input_paths"]["gold176_rows"])
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        selected = _read_json(public_request_path)["rows"][0]
        gold[0].update(
            {
                "manifest_job_id": selected["manifest_job_id"],
                "row_id": selected["row_id"],
                "group_id": selected["group_id"],
                "model": selected["model"],
                "width": selected["width"],
                "dispatch_key": selected["dispatch_key"],
                "capability_profile_id": selected["capability_profile_id"],
                "q_mode": selected["q_mode"],
            }
        )
        _write_json(gold_path, gold)
        return
    if mutation == "group_receipt_artifact_tamper":
        row = _read_json(public_request_path)["rows"][0]
        contract = row["source_contract"]
        source_paths = contract.get("shared_source_paths", contract)
        artifact = Path(source_paths["checkpoint_path"])
        artifact.write_text("PRIVATE-COMPLETION-TOKEN\n", encoding="utf-8")
        return
    if mutation == "group_receipt_marker_tamper":
        row = _read_json(public_request_path)["rows"][0]
        contract = row["source_contract"]
        source_paths = contract.get("shared_source_paths", contract)
        marker = Path(source_paths["source_done_marker"])
        marker.write_text("PRIVATE-COMPLETION-TOKEN\n", encoding="utf-8")
        return

    missing_paths = {
        "task_state_missing": round_paths["task_state"],
        "actual_result_missing": round_paths["actual_feedback"],
        "actual_completion_receipt_missing": round_paths["actual_receipt"],
        "finalization_barrier_missing": round_paths["finalization_barrier"],
    }
    if mutation in missing_paths:
        missing_paths[mutation].unlink()
        return
    if mutation == "actual_completion_receipt_mismatch":
        actual_receipt = _read_json(round_paths["actual_receipt"])
        actual_receipt["measurement_request_sha256"] = "3" * 64
        _write_json(round_paths["actual_receipt"], actual_receipt)
        return

    result_path = round_paths["actual_feedback"]
    actual_result = _read_json(result_path)
    row = actual_result["rows"][0]
    if mutation == "terminal_failure_status":
        row["terminal_status"] = "PRIVATE-COMPLETION-TOKEN"
    elif mutation == "metric_missing":
        row.pop("ap70")
    elif mutation == "metric_extra":
        row["private_metric"] = "PRIVATE-COMPLETION-TOKEN"
    elif mutation == "metric_boolean":
        row["latency_ms"] = True
    elif mutation == "metric_nan":
        row["ap30"] = float("nan")
    elif mutation == "metric_infinite":
        row["ap50"] = float("inf")
    elif mutation == "latency_nonpositive":
        row["latency_ms"] = 0
    elif mutation == "energy_nonpositive":
        row["energy_j"] = -1
    elif mutation == "ap_out_of_bounds":
        row["ap70"] = 1.01
    else:  # pragma: no cover - test helper contract
        raise AssertionError(mutation)
    _write_json(result_path, actual_result)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _mutate_performance_evidence(
    paths: dict[str, Any],
    private_root: Path,
    interface: dict[str, Any],
    mutation: str,
) -> None:
    round_root = _round_paths(paths, interface, private_root, 0)["round_root"]
    performance_root = round_root / "performance"
    if mutation == "performance_manifest_missing":
        (performance_root / "performance_manifest.json").unlink()
        return
    jobs_path = performance_root / "performance_jobs.jsonl"
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines() if line]
    states_path = performance_root / "performance_state.jsonl"
    states = [json.loads(line) for line in states_path.read_text().splitlines() if line]
    if mutation == "performance_result_tampered":
        Path(states[0]["result_json"]).write_text("{}\n", encoding="utf-8")
        return
    if mutation in {
        "performance_missing_observed_runtime",
        "performance_wrong_runtime_arch",
    }:
        result_state = next(row for row in states if row["job_id"] == jobs[0]["job_id"])
        result_path = Path(result_state["result_json"])
        result = _read_json(result_path)
        if mutation == "performance_missing_observed_runtime":
            result.pop("observed_runtime_evidence")
        else:
            result["observed_runtime_evidence"]["target_arch"] = "sm90"
        _write_json(result_path, result)
        result_state["result_sha256"] = sha256_file(result_path)
        states_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in states),
            encoding="utf-8",
        )
        return
    field, value = {
        "performance_wrong_profile": ("hardware_profile", "h800"),
        "performance_wrong_arch": ("tvm_arch", "sm90"),
        "performance_wrong_cache": ("tvm_cache_namespace", "h800-sm90"),
        "performance_wrong_candidate": ("candidate_id", "wrong-candidate"),
        "performance_wrong_q_mode": ("q_mode", "int8"),
        "performance_wrong_configuration": ("configuration_digest", "0" * 64),
        "performance_wrong_checkpoint": ("checkpoint_digest", "1" * 64),
        "performance_wrong_code": ("code_digest", "2" * 64),
        "performance_wrong_toolchain": ("toolchain_id", "tensorrt"),
        "performance_tensorrt_marker": ("engine_path", "/private/model.engine"),
    }[mutation]
    jobs[0]["expected_tvm_measurement_manifest"][field] = value
    result_state = next(row for row in states if row["job_id"] == jobs[0]["job_id"])
    result_path = Path(result_state["result_json"])
    result = _read_json(result_path)
    result["tvm_measurement_manifest"][field] = value
    _write_json(result_path, result)
    result_state["result_sha256"] = sha256_file(result_path)
    jobs_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    states_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in states),
        encoding="utf-8",
    )


def _write_native_completion_leaves(paths: dict[str, Any]) -> None:
    binding = _read_json(paths["binding"])
    private_root = Path(binding["private_root"])
    interface = binding["execution_interface"]
    for round_index in range(4):
        round_paths = _round_paths(paths, interface, private_root, round_index)
        round_root = round_paths["round_root"]
        final_root = round_root / "final"
        promotion_root = round_root / "actual_feedback"
        final_root.mkdir(parents=True, exist_ok=True)
        promotion_root.mkdir(parents=True, exist_ok=True)
        _write_json(final_root / "stage5_feedback_v2_final.json", [{"native": True}] * 4)
        _write_json(
            final_root / "stage5_feedback_v2_audit.json",
            {"schema_version": "stage5_feedback_batch_v2"},
        )
        _write_json(
            final_root / "atomic_batch_audit.json",
            {"schema_version": "stage5_atomic_batch_audit_v2"},
        )
        _write_json(promotion_root / "stage5_feedback_v3_actual.json", [{"native": True}] * 4)
        _write_json(
            promotion_root / "actual_feedback_batch_audit_v3.json",
            {"schema_version": "stage5_actual_feedback_batch_audit_v3"},
        )
        if binding["target"]["hardware"] == "rtx4090":
            _write_native_performance_leaves(
                paths, binding, private_root, round_index, round_paths
            )


def _write_native_performance_leaves(
    paths: dict[str, Any],
    binding: dict[str, Any],
    private_root: Path,
    round_index: int,
    round_paths: dict[str, Path],
) -> None:
    request_path = paths["output_root"] / f"round-{round_index:02d}/measurement_request.json"
    request = _read_json(request_path)
    round_root = round_paths["round_root"]
    _write_native_quant_contracts(round_root, request)
    profile = load_post_source_adapter_profile(
        private_root / "post-source-adapter-profile.yaml", private_root=private_root
    )
    indices = tuple(binding["gpu_policy"]["indices"])
    runner = _PerformanceRunner(
        state_statuses=("success",) * 4,
        tvm_manifest_profile=profile,
        gpu_indices=indices,
    )
    performance_root = round_root / "performance"
    runner._run_planner(
        (
            "planner",
            "--request-json",
            str(request_path),
            "--output-dir",
            str(performance_root),
            "--quant-contract-root",
            str(round_root / "quant_contracts"),
        )
    )
    runner._run_executor(
        (
            "executor",
            "--jobs-jsonl",
            str(performance_root / "performance_jobs.jsonl"),
            "--state-jsonl",
            str(performance_root / "performance_state.jsonl"),
        )
    )


def _write_native_quant_contracts(round_root: Path, request: dict[str, Any]) -> None:
    for row in request["rows"]:
        if row["q_mode"] != "int8":
            continue
        source = row["source_contract"]
        quant_path = quant_contract_path(round_root, row)
        quant_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            quant_path,
            {
                "schema": "stage3_tvm_int8_quant_contract_v3",
                "onnx_path": source["onnx_path"],
                "onnx_sha256": sha256_file(Path(source["onnx_path"])),
                "calibration_npz": source["calibration_npz"],
                "calibration_npz_sha256": sha256_file(Path(source["calibration_npz"])),
                "calibration_summary": source["calibration_summary"],
                "calibration_summary_sha256": sha256_file(Path(source["calibration_summary"])),
                "params": {"spatial_features": {"scale": 0.5}},
            },
        )


def _round_paths(
    paths: dict[str, Any],
    interface: dict[str, Any],
    private_root: Path,
    round_index: int,
) -> dict[str, Path]:
    return resolve_validated_history_round_paths(
        interface,
        private_root,
        paths["output_root"] / f"round-{round_index:02d}",
        round_index,
    )


def _replace_round_row(
    paths: dict[str, Any],
    interface: dict[str, Any],
    private_root: Path,
    round_index: int,
    replacement: dict[str, Any],
) -> None:
    public_path = (
        paths["output_root"]
        / f"round-{round_index:02d}"
        / "measurement_request.json"
    )
    private_paths = _round_paths(paths, interface, private_root, round_index)
    request = _read_json(public_path)
    request["rows"][0] = replacement
    request["row_sha256"] = {
        row["row_id"]: canonical_json_sha256(row) for row in request["rows"]
    }
    request["measurement_request_sha256"] = canonical_json_sha256(
        {
            name: value
            for name, value in request.items()
            if name != "measurement_request_sha256"
        }
    )
    _write_json(public_path, request)
    _write_json(private_paths["measurement_request"], request)
    row_evidence = {
        row["row_id"]: row["source_evidence_sha256"] for row in request["rows"]
    }
    state_rows = [
        {
            "row_id": row["row_id"],
            "row_sha256": request["row_sha256"][row["row_id"]],
            "source_evidence_sha256": row["source_evidence_sha256"],
            "terminal_status": "measured_success_gold",
        }
        for row in request["rows"]
    ]
    result_rows = [
        {
            **row,
            "latency_ms": 2.0,
            "energy_j": 0.5,
            "ap30": 0.91,
            "ap50": 0.82,
            "ap70": 0.73,
        }
        for row in state_rows
    ]
    validation = {
        "measurement_request_sha256": request["measurement_request_sha256"],
        "row_sha256": request["row_sha256"],
        "source_evidence_sha256": row_evidence,
    }
    _write_json(private_paths["task_state"], {"stage": "finalization", "rows": state_rows})
    _write_json(
        private_paths["actual_feedback"],
        {
            "measurement_request_sha256": request["measurement_request_sha256"],
            "rows": result_rows,
        },
    )
    _write_json(private_paths["actual_receipt"], validation)
    _write_json(private_paths["finalization_barrier"], validation)


def test_completion_report_publishes_only_profile_provenance_and_counts() -> None:
    assert tuple(P6MaterializerCompletionReport.__dataclass_fields__) == (
        "schema_version",
        "status",
        "hardware_profile",
        "provenance",
        "completed_rounds",
        "selected_rows",
        "gold176_remeasured_rows",
    )


def test_gold_identity_accepts_manifest_job_id_without_synthetic_row_id() -> None:
    assert verification._identity(
        {
            "manifest_job_id": "pyramid|16x16x16|q=fp16|profile=h800",
            "group_id": "pyramid|16x16x16",
        }
    ) == "pyramid|16x16x16|q=fp16|profile=h800"


def test_v3_rtx_verification_context_forwards_contract_to_search_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normalized, _ = _write_rtx_context_source(tmp_path / "context-source")
    contract_path = _write_yaml(
        tmp_path / "contract-v3.yaml", _public_v3_contract("rtx4090")
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
    local_path = _write_yaml(
        tmp_path / "local-v3.yaml", local_payload
    )
    binding_root = tmp_path / "binding-fixture"
    binding_root.mkdir()
    binding_path = _history_cli_fixture(binding_root)["binding"]
    output_root = tmp_path / "private-output"
    output_root.mkdir()
    _write_json(output_root / "pyramid_candidate_plan.json", {})
    expected_context = object()
    observed_indices: list[tuple[int, ...] | None] = []

    def load_context(**_: Any) -> object:
        return expected_context

    monkeypatch.setattr(verification, "load_fresh_run_context", load_context)
    monkeypatch.setattr(
        "framework.stage6.coptv2x_h800_search_v2.historical_capability_source_sha256",
        lambda root: hashlib.sha256(historical_source_bytes()).hexdigest(),
    )

    def rebuild(**kwargs: Any) -> SimpleNamespace:
        observed_indices.append(kwargs["expected_gpu_indices"])
        raw = _read_json(kwargs["capability_context_path"])
        evidence = raw["measurement_evidence"]
        return SimpleNamespace(
            runtime_identity=evidence["runtime_identity"],
            probe_records={
                "neutral": evidence["neutral_records"],
                "pruning": evidence["pruning_records"],
            },
        )

    monkeypatch.setattr(
        "framework.stage6.coptv2x_h800_search_v2.probe_normalized_capability_authority",
        rebuild,
    )
    monkeypatch.setattr(
        verification,
        "validate_history_execution_binding",
        lambda binding, profile: binding["execution_interface"],
    )
    monkeypatch.setattr(
        verification, "_require_post_source_profile", lambda *_: None
    )
    monkeypatch.setattr(verification, "validate_p6_candidate_plan", lambda *_a, **_k: {})

    contract, local, _, _, frozen_gold, context, *_ = (
        verification._load_verification_context(contract_path, local_path, binding_path)
    )

    assert contract.hardware_profile is local.hardware_profile
    assert local.hardware_profile.target_hardware_id == "rtx4090"
    assert len(frozen_gold) == 176
    assert context is expected_context
    binding = _read_json(binding_path)
    assert observed_indices == [tuple(binding["gpu_policy"]["indices"])]


def test_completion_accepts_four_round_current_run_with_shared_receipts(
    completed_run: dict[str, Any],
) -> None:
    report = verify_materializer_training_run(**_verify_kwargs(completed_run))
    receipt_root = (
        completed_run["output_root"]
        / ".p6-materializer-training-bridge-v1"
        / "group-receipts"
    )
    selected_rows = [
        row
        for round_index in range(4)
        for row in json.loads(
            (
                completed_run["output_root"]
                / f"round-{round_index:02d}"
                / "measurement_request.json"
            ).read_text(encoding="utf-8")
        )["rows"]
    ]

    assert report.status == "completed"
    assert report.hardware_profile == "h800"
    binding = _read_json(completed_run["binding"])
    training = binding["source_contract_template"]["external_training_binding"]
    assert isinstance(report.provenance, P6HardwareSpecificReportProvenance)
    assert report.provenance.schema_version == "p6_hardware_specific_report_provenance_v3"
    assert report.provenance.hardware_model_family == "h800"
    assert report.provenance.gpu_count == len(binding["gpu_policy"]["indices"])
    assert report.provenance.code_revision == "test-revision"
    assert report.provenance.ap_provenance == {
        "data_split": "trainval_coptv2x",
        "checkpoint_initial_state": training["base_checkpoint_sha256"],
        "training_config_digest": training["pyramid_config_sha256"],
        "seed": 20260821,
        "metric_protocol": "coptv2x-ap30-ap50-ap70-v1",
        "dataset_snapshot_digest": None,
        "evaluation_snapshot_digest": None,
        "cross_hardware_comparison_status": (
            "unavailable_unverified_dataset_identity"
        ),
    }
    assert report.provenance.runtime_observation_status == "unavailable_legacy_profile"
    assert report.provenance.observed_runtime_digest is None
    with pytest.raises(FrozenInstanceError):
        report.provenance.target = "rtx4090"  # type: ignore[misc]
    assert report.completed_rounds == 4
    assert report.selected_rows == 16
    assert report.gold176_remeasured_rows == 0
    assert len({row["row_id"] for row in selected_rows}) == 16
    assert len(list(receipt_root.glob("*.json"))) < 16


def test_completed_rtx_verification_replays_recorded_runtime_after_system_python_upgrade(
    completed_rtx_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def upgraded_runtime_authority(**kwargs: Any) -> SimpleNamespace:
        raw = _read_json(kwargs["capability_context_path"])
        evidence = raw["measurement_evidence"]
        runtime = {
            **evidence["runtime_identity"],
            "python_executable_sha256": hashlib.sha256(
                b"upgraded-system-python"
            ).hexdigest(),
        }
        runtime["compiler_fingerprint"] = compiler_fingerprint(
            {
                key: value
                for key, value in runtime.items()
                if key != "compiler_fingerprint"
            }
        )
        return SimpleNamespace(
            runtime_identity=runtime,
            probe_records={
                "neutral": evidence["neutral_records"],
                "pruning": evidence["pruning_records"],
            },
        )

    monkeypatch.setattr(
        execution,
        "probe_normalized_capability_authority",
        upgraded_runtime_authority,
    )

    report = verify_materializer_training_run(**_verify_kwargs(completed_rtx_run))

    assert report.status == "completed"


def test_completed_fake_rtx_hardware_profile_tree_reports_public_profile_id(
    completed_rtx_run: dict[str, Any],
) -> None:
    report = verify_materializer_training_run(**_verify_kwargs(completed_rtx_run))

    assert report.hardware_profile == "rtx4090"
    assert report.provenance.comparison_scope == "hardware_specific"
    assert report.provenance.hardware_profile == "rtx4090"
    assert report.provenance.target == "rtx4090"
    assert report.provenance.execution_backend == "tvm_auto"
    assert report.provenance.tvm_arch == "sm89"
    assert report.provenance.latency_energy_hardware_profile == "rtx4090"
    assert report.provenance.pareto_hardware_profile == "rtx4090"
    binding = _read_json(completed_rtx_run["binding"])
    training = binding["source_contract_template"]["external_training_binding"]
    assert report.provenance.target_model == "pyramid"
    assert report.provenance.hardware_model_family == "rtx4090"
    assert report.provenance.gpu_count == 4
    assert report.provenance.tvm_cache_namespace == "rtx4090-sm89"
    assert report.provenance.code_revision == "test-revision"
    assert report.provenance.ap_provenance == {
        "data_split": "trainval_coptv2x",
        "checkpoint_initial_state": training["base_checkpoint_sha256"],
        "training_config_digest": training["pyramid_config_sha256"],
        "seed": 20260821,
        "metric_protocol": "coptv2x-ap30-ap50-ap70-v1",
        "dataset_snapshot_digest": None,
        "evaluation_snapshot_digest": None,
        "cross_hardware_comparison_status": (
            "unavailable_unverified_dataset_identity"
        ),
    }
    assert report.provenance.runtime_observation_status == "observed"
    assert report.provenance.observed_runtime_digest is not None
    assert report.completed_rounds == 4
    assert report.selected_rows == 16
    assert report.gold176_remeasured_rows == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "performance_manifest_missing",
        "performance_result_tampered",
        "performance_wrong_profile",
        "performance_wrong_arch",
        "performance_wrong_cache",
        "performance_wrong_candidate",
        "performance_wrong_q_mode",
        "performance_wrong_configuration",
        "performance_wrong_checkpoint",
        "performance_wrong_code",
        "performance_wrong_toolchain",
        "performance_tensorrt_marker",
        "performance_missing_observed_runtime",
        "performance_wrong_runtime_arch",
    ],
)
def test_completed_rtx_rejects_unbound_native_performance_evidence(
    completed_rtx_run: dict[str, Any], mutation: str
) -> None:
    _apply_completion_mutation(completed_rtx_run, mutation)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        verify_materializer_training_run(**_verify_kwargs(completed_rtx_run))

    assert captured.value.failure_code == "history_execution_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "binding_hardware_profile_drift",
        "post_source_hardware_profile_drift",
        "candidate_plan_hardware_profile_drift",
    ],
)
def test_completed_rtx_rejects_wrong_hardware_profile_evidence_before_counts(
    completed_rtx_run: dict[str, Any], mutation: str
) -> None:
    state = _read_json(completed_rtx_run["output_root"] / "state.json")
    assert (state["completed_rounds"], state["measured_candidate_count"]) == (4, 16)
    _apply_completion_mutation(completed_rtx_run, mutation)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        verify_materializer_training_run(**_verify_kwargs(completed_rtx_run))

    assert captured.value.failure_code == "history_execution_invalid"


@pytest.mark.parametrize(
    "relative_path",
    [
        "final/stage5_feedback_v2_final.json",
        "final/stage5_feedback_v2_audit.json",
        "final/atomic_batch_audit.json",
        "actual_feedback/stage5_feedback_v3_actual.json",
        "actual_feedback/actual_feedback_batch_audit_v3.json",
    ],
)
def test_completion_requires_native_finalization_and_promotion_leaves(
    completed_run: dict[str, Any], relative_path: str
) -> None:
    binding = _read_json(completed_run["binding"])
    round_paths = _round_paths(
        completed_run,
        binding["execution_interface"],
        Path(binding["private_root"]),
        0,
    )
    (round_paths["round_root"] / relative_path).unlink()

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        verify_materializer_training_run(**_verify_kwargs(completed_run))

    assert captured.value.failure_code == "history_execution_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "context_missing",
        "context_hash_drift",
        "public_private_request_mismatch",
        "producer_request_missing",
        "producer_request_hash_drift",
        "producer_is_later_round",
        "producer_row_missing",
        "producer_row_hash_drift",
        "group_receipt_missing",
        "group_receipt_cross_run",
        "group_receipt_artifact_tamper",
        "group_receipt_marker_tamper",
        "task_state_missing",
        "actual_result_missing",
        "actual_completion_receipt_missing",
        "actual_completion_receipt_mismatch",
        "finalization_barrier_missing",
        "duplicate_selected_row_id",
        "duplicate_measurement_identity",
        "gold176_overlap",
        "terminal_failure_status",
        "metric_missing",
        "metric_extra",
        "metric_boolean",
        "metric_nan",
        "metric_infinite",
        "latency_nonpositive",
        "energy_nonpositive",
        "ap_out_of_bounds",
        "decoy_context",
        "decoy_receipt",
        "post_source_profile_symlink",
    ],
)
def test_completion_rejects_invalid_exact_evidence(
    completed_run: dict[str, Any], mutation: str
) -> None:
    _apply_completion_mutation(completed_run, mutation)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        verify_materializer_training_run(**_verify_kwargs(completed_run))

    assert captured.value.failure_code == "history_execution_invalid"
    assert "PRIVATE-COMPLETION-TOKEN" not in str(captured.value)


def test_completion_cli_emits_only_allowlisted_fields(
    completed_run: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "--contract",
            str(completed_run["contract"]),
            "--local-config",
            str(completed_run["local"]),
            "--binding",
            str(completed_run["binding"]),
        ]
    )
    output = capsys.readouterr()

    assert result == 0
    assert output.err == ""
    assert set(json.loads(output.out)) == set(
        P6MaterializerCompletionReport.__dataclass_fields__
    )
    payload = json.loads(output.out)
    assert payload["hardware_profile"] == "h800"
    provenance = payload["provenance"]
    assert provenance["schema_version"] == "p6_hardware_specific_report_provenance_v3"
    assert provenance["hardware_profile"] == "h800"
    assert provenance["target_model"] == "pyramid"
    assert provenance["code_revision"] == "test-revision"
    assert provenance["ap_provenance"]["seed"] == 20260821
    assert provenance["runtime_observation_status"] == "unavailable_legacy_profile"
    assert provenance["observed_runtime_digest"] is None
    assert provenance["ap_provenance"]["cross_hardware_comparison_status"] == (
        "unavailable_unverified_dataset_identity"
    )
    assert all("/" not in str(value) for value in provenance.values())


@pytest.mark.parametrize(
    ("schema_version", "required"),
    [
        (PROFILE_SCHEMA_VERSION, False),
        (PROFILE_SCHEMA_VERSION_V2, False),
        (PROFILE_SCHEMA_VERSION_V3, False),
        (PROFILE_SCHEMA_VERSION_V4, True),
    ],
)
def test_native_performance_attestation_gate_is_formal_v4_only(
    tmp_path: Path, schema_version: str, required: bool
) -> None:
    paths = _history_cli_fixture(tmp_path, hardware_profile="rtx4090")
    private_root = Path(_read_json(paths["binding"])["private_root"])
    profile = load_post_source_adapter_profile(
        private_root / "post-source-adapter-profile.yaml",
        private_root=private_root,
    )
    profile = replace(profile, schema_version=schema_version)

    assert verification._requires_native_performance_attestation(profile) is required


def test_completion_cli_rejects_profile_drift_in_constructed_provenance(
    completed_run: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def validate_with_profile_drift(raw: dict[str, object]) -> Any:
        drifted = {**raw, "target": "rtx4090"}
        return validate_hardware_specific_report_provenance(drifted)

    monkeypatch.setattr(
        verification,
        "validate_hardware_specific_report_provenance",
        validate_with_profile_drift,
        raising=False,
    )

    result = main(
        [
            "--contract",
            str(completed_run["contract"]),
            "--local-config",
            str(completed_run["local"]),
            "--binding",
            str(completed_run["binding"]),
        ]
    )
    output = capsys.readouterr()

    assert result == 1
    assert output.out == ""
    assert output.err == "verification_failed\n"


def test_completion_cli_redacts_private_failures(
    completed_run: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    _apply_completion_mutation(completed_run, "context_hash_drift")
    private_binding = (
        completed_run["binding"].parent
        / "PRIVATE-COMPLETION-TOKEN-binding.json"
    )
    shutil.copy2(completed_run["binding"], private_binding)
    result = main(
        [
            "--contract",
            str(completed_run["contract"]),
            "--local-config",
            str(completed_run["local"]),
            "--binding",
            str(private_binding),
        ]
    )
    output = capsys.readouterr()

    assert result == 1
    assert output.out == ""
    assert output.err == "verification_failed\n"
    assert "PRIVATE-COMPLETION-TOKEN" not in output.out + output.err
