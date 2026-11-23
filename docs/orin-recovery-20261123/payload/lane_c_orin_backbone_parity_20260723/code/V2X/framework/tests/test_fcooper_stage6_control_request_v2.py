from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.fcooper_execute_measurement_row_v2 import validate_control_request
from scripts.fcooper_stage6_control_request_v2 import (
    CONTROL_PHASES,
    aggregate_feedback,
    build_control_requests,
    canonical_sha256,
    validate_candidate_plan_against_five_arm_plan,
    select_complete_arm_winner,
    repeat_spec,
    validate_original_contract,
)
from scripts.stage6_prepare_fcooper_five_arm_v2 import fixed_original_candidate


ROOT = Path(__file__).resolve().parents[2]


def candidate(index: int, *, width: list[int] | None = None, q_mode: str = "int8") -> dict:
    row_id = f"candidate-{index:02d}"
    selected_width = width or [32, 64, 128, 64, 128]
    return {
        "row_id": row_id,
        "manifest_job_id": row_id,
        "task_id": "S5-FCO-TRT-V2",
        "task_sha256": "a" * 64,
        "model": "fcooper",
        "hardware_id": "h800",
        "group_id": f"fcooper|{row_id}",
        "width": selected_width,
        "width_schema": [
            "backbone.s0",
            "backbone.s1",
            "backbone.s2",
            "neck.deblock",
            "neck.output",
        ],
        "q_mode": q_mode,
        "source_evidence_sha256": f"{index + 1:064x}",
    }


def successful_feedback(row: dict, *, latency: float = 1.0) -> dict:
    feedback = {
        **row,
        "terminal_status": "measured_success_gold",
        "training_source": "online_feedback",
        "latency_ms": latency,
        "energy_j": 0.2,
        "ap70": 0.61,
        "checkpoint_sha256": "b" * 64,
        "recovery_training_report_sha256": "c" * 64,
    }
    feedback["actual_feedback_row_sha256"] = canonical_sha256(feedback)
    return feedback


def test_control_requests_are_content_addressed_and_chunked_4_plus_1() -> None:
    requests = build_control_requests(
        [candidate(index) for index in range(5)],
        arm_id="compression_only",
        phase="measure",
        builder_optimization_level=0,
    )

    assert [request["batch_size"] for request in requests] == [4, 1]
    assert {request["task_id"] for request in requests} == {
        "S6-FCO-TRT-COMPRESSION-ONLY-MEASURE-V2"
    }
    assert all(request["arm_id"] == "compression_only" for request in requests)
    assert all(request["phase"] == "measure" for request in requests)
    assert all(
        request["measurement_request_sha256"]
        == canonical_sha256(
            {
                key: value
                for key, value in request.items()
                if key != "measurement_request_sha256"
            }
        )
        for request in requests
    )
    for request in requests:
        for row_index, row in enumerate(request["rows"]):
            assert validate_control_request(request, row_index=row_index) == row


def test_control_requests_reject_duplicate_rows_builder_drift_and_pilot() -> None:
    row = candidate(0)
    with pytest.raises(ValueError, match="unique"):
        build_control_requests(
            [row, row],
            arm_id="compression_only",
            phase="measure",
            builder_optimization_level=0,
        )
    with pytest.raises(ValueError, match="builder"):
        build_control_requests(
            [row],
            arm_id="schedule_only",
            phase="measure",
            builder_optimization_level=0,
        )
    pilot = {
        **row,
        "source_contract": {
            "path": "/tmp/fcooper_workpackage_a_20260723/checkpoint.pth"
        },
    }
    with pytest.raises(ValueError, match="pilot"):
        build_control_requests(
            [pilot],
            arm_id="compression_only",
            phase="measure",
            builder_optimization_level=0,
        )


def test_candidate_plan_must_match_frozen_five_arm_selection() -> None:
    rows = [candidate(index) for index in range(16)]
    plan = {
        "schema_version": "stage6_fcooper_five_arm_plan_v2",
        "task_id": "S5-FCO-TRT-V2",
        "arms": {
            "compression_only": {
                "selected_row_ids": [row["row_id"] for row in rows],
                "builder_optimization_level": 0,
            }
        },
    }
    candidate_plan = {
        "schema_version": "stage6_fcooper_arm_candidate_plan_v2",
        "task_id": "S5-FCO-TRT-V2",
        "arm_id": "compression_only",
        "phase": "measure",
        "builder_optimization_level": 0,
        "row_count": 16,
        "rows": rows,
    }

    validated = validate_candidate_plan_against_five_arm_plan(
        candidate_plan,
        plan,
        arm_id="compression_only",
        phase="measure",
        builder_optimization_level=0,
    )

    assert [row["row_id"] for row in validated] == [
        row["row_id"] for row in rows
    ]
    injected = {**candidate_plan, "rows": [*rows[:-1], candidate(99)]}
    with pytest.raises(ValueError, match="five-arm plan"):
        validate_candidate_plan_against_five_arm_plan(
            injected,
            plan,
            arm_id="compression_only",
            phase="measure",
            builder_optimization_level=0,
        )


def test_every_control_phase_has_an_explicit_task_id() -> None:
    assert CONTROL_PHASES == {
        ("schedule_only", "measure"): (
            "S6-FCO-TRT-SCHEDULE-ONLY-MEASURE-V2",
            5,
        ),
        ("compression_only", "measure"): (
            "S6-FCO-TRT-COMPRESSION-ONLY-MEASURE-V2",
            0,
        ),
        ("compress_then_tune", "screen"): (
            "S6-FCO-TRT-COMPRESS-THEN-TUNE-SCREEN-V2",
            0,
        ),
        ("compress_then_tune", "tuned_remeasurement"): (
            "S6-FCO-TRT-COMPRESS-THEN-TUNE-TUNED-V2",
            5,
        ),
    }


def test_feedback_aggregation_requires_every_content_bound_row(tmp_path: Path) -> None:
    requests = build_control_requests(
        [candidate(0), candidate(1)],
        arm_id="compression_only",
        phase="measure",
        builder_optimization_level=0,
    )
    request = requests[0]
    artifact_root = tmp_path / "artifacts"
    first = request["rows"][0]
    row_tag = hashlib.sha256(
        f"{request['task_id']}:{first['row_id']}".encode()
    ).hexdigest()[:16]
    feedback_path = artifact_root / "execution" / row_tag / "feedback_row.json"
    feedback_path.parent.mkdir(parents=True)
    feedback_path.write_text(
        json.dumps(
            successful_feedback(
                {
                    **first,
                    "measurement_request_row_sha256": request["row_sha256"][
                        first["row_id"]
                    ],
                }
            )
        )
    )

    with pytest.raises(ValueError, match="complete"):
        aggregate_feedback(requests, artifact_root=artifact_root, expected_count=2)

    second = request["rows"][1]
    row_tag = hashlib.sha256(
        f"{request['task_id']}:{second['row_id']}".encode()
    ).hexdigest()[:16]
    feedback_path = artifact_root / "execution" / row_tag / "feedback_row.json"
    feedback_path.parent.mkdir(parents=True)
    feedback_path.write_text(
        json.dumps(
            successful_feedback(
                {
                    **second,
                    "measurement_request_row_sha256": request["row_sha256"][
                        second["row_id"]
                    ],
                },
                latency=0.9,
            )
        )
    )

    rows, audit = aggregate_feedback(
        requests, artifact_root=artifact_root, expected_count=2
    )

    assert [row["row_id"] for row in rows] == ["candidate-00", "candidate-01"]
    assert audit["passed"] is True
    assert audit["row_count"] == 2
    assert len(audit["feedback_sha256"]) == 2


def test_winner_selection_waits_for_complete_arm_feedback() -> None:
    rows = [successful_feedback(candidate(index), latency=2.0 - index / 10) for index in range(4)]
    with pytest.raises(ValueError, match="complete"):
        select_complete_arm_winner(
            rows[:3],
            expected_count=4,
            ap70_ref=0.63,
            max_ap_drop=0.10,
        )

    selected = select_complete_arm_winner(
        rows,
        expected_count=4,
        ap70_ref=0.63,
        max_ap_drop=0.10,
    )

    assert selected["row_id"] == "candidate-03"
    assert selected["selection_status"] == "selected_feasible"


def test_fixed_original_candidate_is_scanner_derived_fp32() -> None:
    original = candidate(
        0,
        width=[64, 128, 256, 128, 256],
        q_mode="fp32",
    )

    selected = fixed_original_candidate([candidate(1), original])

    assert selected["row_id"] == original["row_id"]
    with pytest.raises(ValueError, match="scanner-derived original"):
        fixed_original_candidate([candidate(1)])


def test_original_contract_preflight_binds_native_source_and_ap(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    ap_report = tmp_path / "ap.json"
    config.write_text("model: fcooper\n")
    checkpoint.write_bytes(b"checkpoint")
    ap_report.write_text('{"status": "success_full"}\n')
    contract = {
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "ap_reference_report_path": str(ap_report),
        "ap_reference_report_sha256": hashlib.sha256(
            ap_report.read_bytes()
        ).hexdigest(),
    }

    audit = validate_original_contract(
        contract,
        source_config=config,
        source_checkpoint=checkpoint,
    )

    assert audit["passed"] is True
    drifted = {**contract, "ap_reference_report_sha256": "0" * 64}
    with pytest.raises(ValueError, match="AP reference"):
        validate_original_contract(
            drifted,
            source_config=config,
            source_checkpoint=checkpoint,
        )


def test_supervisor_uses_control_mode_four_lanes_and_dynamic_gpu7_repeats() -> None:
    source = (
        ROOT / "scripts/fcooper_stage6_five_arm_supervisor_v2.sh"
    ).read_text(encoding="utf-8")

    assert "--request-kind" in source
    assert "stage6-control" in source
    assert "MAX_PARALLEL=4" in source
    assert "REPEAT_GPU=7" in source
    assert "same_gpu7_repeat_" in source
    assert "feedback_history_final_t16.json" in source
    assert "fcooper_workpackage_a_20260723" in source
    assert "fcooper_execute_measurement_row_v1.py" not in source
    assert "stage5_measurement_request_v2" not in source
    assert "fcooper_trt_ap_bridge_v1.py" in source
    assert "stage6_finalize_fcooper_table1_v2.py" in source


def test_repeat_spec_binds_config_checkpoint_and_onnx(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    onnx = tmp_path / "model.onnx"
    for path, content in (
        (config, b"config"),
        (checkpoint, b"checkpoint"),
        (onnx, b"onnx"),
    ):
        path.write_bytes(content)
    evidence = {
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_path": str(onnx),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
    }
    evidence_path = tmp_path / "source_evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    selection = {
        "row_id": "winner",
        "terminal_status": "measured_success_gold",
        "width": [64, 128, 256, 128, 256],
        "q_mode": "fp16",
        "materialized_source_evidence_path": str(evidence_path),
        "materialized_source_evidence_sha256": hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "graph_features": {"onnx_sha256": evidence["onnx_sha256"]},
    }

    spec = repeat_spec(selection)

    assert spec["config_path"] == str(config)
    assert spec["checkpoint_path"] == str(checkpoint)


def test_repeat_spec_rejects_pruned_source_without_recovery_contract(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    onnx = tmp_path / "model.onnx"
    for path in (config, checkpoint, onnx):
        path.write_bytes(path.name.encode())
    evidence = {
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_path": str(onnx),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
    }
    evidence_path = tmp_path / "source_evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    selection = {
        "row_id": "pruned",
        "terminal_status": "measured_success_gold",
        "width": [32, 64, 128, 64, 128],
        "q_mode": "fp16",
        "materialized_source_evidence_path": str(evidence_path),
        "materialized_source_evidence_sha256": hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "graph_features": {"onnx_sha256": evidence["onnx_sha256"]},
    }

    with pytest.raises((KeyError, ValueError), match="recovery"):
        repeat_spec(selection)
