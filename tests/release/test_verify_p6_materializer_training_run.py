"""Exact-evidence acceptance coverage for P6 completion verification."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Iterator

import pytest
import yaml

from framework.stage6.coptv2x_h800_search_v2 import P6CoptV2XExecutionError
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
from tests.release.test_run_p6_h800_search import _history_cli_fixture, _run_cli


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


@pytest.fixture
def completed_run(_completed_template: dict[str, Any]) -> dict[str, Any]:
    live = _completed_template["live"]
    shutil.rmtree(live)
    shutil.copytree(_completed_template["snapshot"], live, symlinks=True)
    return dict(_completed_template["paths"])


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


def test_completion_report_does_not_publish_receipt_mapping() -> None:
    assert tuple(P6MaterializerCompletionReport.__dataclass_fields__) == (
        "schema_version",
        "status",
        "completed_rounds",
        "selected_rows",
        "gold176_remeasured_rows",
    )


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
    assert report.completed_rounds == 4
    assert report.selected_rows == 16
    assert report.gold176_remeasured_rows == 0
    assert len({row["row_id"] for row in selected_rows}) == 16
    assert len(list(receipt_root.glob("*.json"))) < 16


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
