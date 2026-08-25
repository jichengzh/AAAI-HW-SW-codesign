from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any

import pytest

from framework.stage6.p6_finalization_round_adapter_v1 import (
    P6FinalizationRoundAdapterError,
    run_finalization_round,
)
from framework.stage6.p6_history_feedback_validation_v1 import translate_history_feedback
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
)
from tests.stage6.test_p6_quantization_round_adapter import (
    _read_json,
    _write_json,
    _write_round_request,
    _write_task_state,
)


class _Result:
    returncode = 0


class _FinalizationRunner:
    def __init__(
        self,
        *,
        finalize_returncode: int = 0,
        promote_returncode: int = 0,
        statuses: Sequence[str] = (
            "measured_success_gold",
            "measured_success_gold",
            "feasibility_failure",
            "numerical_feasibility_failure",
        ),
        promoted_mutation: Any | None = None,
    ) -> None:
        self.calls: list[Mapping[str, Any]] = []
        self._finalize_returncode = finalize_returncode
        self._promote_returncode = promote_returncode
        self._statuses = tuple(statuses)
        self._promoted_mutation = promoted_mutation

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        stage = "finalize" if "--manifest-json" in argv else "promote"
        self.calls.append(
            MappingProxyType(
                {"stage": stage, "argv": tuple(argv), "cwd": cwd, "env": dict(env), "shell": shell}
            )
        )
        result = _Result()
        result.returncode = self._finalize_returncode if stage == "finalize" else self._promote_returncode
        if result.returncode == 0:
            if stage == "finalize":
                self._write_finalized(argv)
            else:
                self._write_promoted(argv)
        return result

    def _write_finalized(self, argv: Sequence[str]) -> None:
        request = _read_json(Path(argv[argv.index("--measurement-request-json") + 1]))
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            _native_feedback_row(row, request, status)
            for row, status in zip(request["rows"], self._statuses, strict=True)
        ]
        _write_json(output_dir / "stage5_feedback_v2_final.json", rows)
        (output_dir / "stage5_feedback_v2_final.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        _write_json(
            output_dir / "stage5_feedback_v2_audit.json",
            {"schema_version": "stage5_feedback_batch_v2", "groups": [], "summary": _summary(rows)},
        )
        _write_json(
            output_dir / "atomic_batch_audit.json",
            {
                "schema_version": "stage5_atomic_batch_audit_v2",
                "feedback_released": True,
                "batch_quarantined": False,
                "budget_consumed": 4,
                "released_feedback_rows": rows,
            },
        )

    def _write_promoted(self, argv: Sequence[str]) -> None:
        feedback = _read_json(Path(argv[argv.index("--feedback-json") + 1]))
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        rows = [_promoted_row(row) for row in feedback]
        if self._promoted_mutation is not None:
            self._promoted_mutation(rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "stage5_feedback_v3_actual.json", rows)
        _write_json(
            output_dir / "actual_feedback_batch_audit_v3.json",
            {
                "schema_version": "stage5_actual_feedback_batch_audit_v3",
                "promoted_row_count": 4,
                "silent_surrogate_fallback_count": 0,
                "rows": [
                    {
                        "manifest_job_id": row["manifest_job_id"],
                        "candidate_graph_features_sha256": row["candidate_graph_features_sha256"],
                        "materialized_graph_features_sha256": row["materialized_graph_features_sha256"],
                        "actual_feedback_row_sha256": row["actual_feedback_row_sha256"],
                    }
                    for row in rows
                ],
            },
        )


class _LegacyRequestRunner(_FinalizationRunner):
    """Model the native finalizer's stage5_feedback_row_v2 identity contract."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        if "--manifest-json" in argv:
            request = _read_json(
                Path(argv[argv.index("--measurement-request-json") + 1])
            )
            manifest = _read_json(Path(argv[argv.index("--manifest-json") + 1]))
            if (
                any(
                    row.get("schema_version") != "stage5_feedback_row_v2"
                    for row in request["rows"]
                )
                or manifest.get("source_request_sha256")
                != request.get("measurement_request_sha256")
            ):
                result = _Result()
                result.returncode = 1
                return result
        return super().run(argv, cwd=cwd, env=env, shell=shell)


def test_finalization_invokes_native_leaves_and_projects_existing_completion_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    runner = _FinalizationRunner()

    run_finalization_round(*fixture["args"], runner)

    assert [call["stage"] for call in runner.calls] == ["finalize", "promote"]
    _assert_native_argv(fixture, runner.calls)
    assert all(call["shell"] is False for call in runner.calls)
    assert runner.calls[0]["cwd"] == fixture["profile"].leaves[0].implementation_cwd
    assert runner.calls[1]["cwd"] == fixture["profile"].leaves[1].implementation_cwd
    result = _read_json(fixture["actual_feedback"])
    assert set(result) == {"measurement_request_sha256", "rows"}
    assert [row["terminal_status"] for row in result["rows"]] == [
        "measured_success_gold",
        "measured_success_gold",
        "feasibility_failure",
        "numerical_feasibility_failure",
    ]
    assert set(result["rows"][0]) == {
        "row_id", "row_sha256", "source_evidence_sha256", "terminal_status",
        "latency_ms", "energy_j", "ap30", "ap50", "ap70",
    }
    assert set(result["rows"][2]) == {
        "row_id", "row_sha256", "source_evidence_sha256", "terminal_status", "failure_reason"
    }
    assert result["rows"][2]["failure_reason"] == "unspecified"
    expected_mapping = _completion_mapping(fixture["request"])
    assert _read_json(fixture["actual_receipt"]) == expected_mapping
    assert _read_json(fixture["barrier"]) == expected_mapping
    assert _read_json(fixture["task_state"])["stage"] == "finalization"
    translated = translate_history_feedback(
        fixture["request"], _interface(), fixture["paths"], read_json=_read_json
    )
    assert translated["rows"][0]["latency_ms"] == 2.0


def test_finalization_uses_private_legacy_request_view_without_canonical_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch the native schema overwrite rejecting canonical candidate rows."""
    fixture = _fixture(tmp_path, monkeypatch)
    canonical_request = fixture["round_root"] / "measurement-request.json"
    canonical_manifest = fixture["round_root"] / "performance/performance_manifest.json"
    request_bytes = canonical_request.read_bytes()
    manifest_bytes = canonical_manifest.read_bytes()
    runner = _LegacyRequestRunner(statuses=("measured_success_gold",) * 4)

    run_finalization_round(*fixture["args"], runner)

    finalize_argv = runner.calls[0]["argv"]
    legacy_request = Path(
        finalize_argv[finalize_argv.index("--measurement-request-json") + 1]
    )
    legacy_manifest = Path(finalize_argv[finalize_argv.index("--manifest-json") + 1])
    promote_argv = runner.calls[1]["argv"]
    assert Path(
        promote_argv[promote_argv.index("--measurement-request-json") + 1]
    ) == legacy_request
    assert legacy_request != canonical_request
    assert legacy_manifest != canonical_manifest
    assert canonical_request.read_bytes() == request_bytes
    assert canonical_manifest.read_bytes() == manifest_bytes
    assert not legacy_request.exists()
    assert not legacy_manifest.exists()
    assert _read_json(fixture["actual_feedback"])["measurement_request_sha256"] == (
        fixture["request"]["measurement_request_sha256"]
    )
    assert _read_json(fixture["actual_feedback"])["rows"][0]["row_sha256"] == (
        fixture["request"]["row_sha256"][fixture["request"]["rows"][0]["row_id"]]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].__setitem__("row_id", "wrong"),
        lambda rows: rows[0].__setitem__("measurement_request_row_sha256", "0" * 64),
        lambda rows: rows[0].__setitem__("source_evidence_sha256", "0" * 64),
        lambda rows: rows[0].__setitem__("latency_ms", float("nan")),
        lambda rows: rows[0].__setitem__("ap70", 1.1),
    ],
)
def test_finalization_rejects_promoted_identity_or_metric_drift_without_p6_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    runner = _FinalizationRunner(promoted_mutation=mutation)

    with pytest.raises(P6FinalizationRoundAdapterError):
        run_finalization_round(*fixture["args"], runner)

    _assert_unpublished(fixture)


@pytest.mark.parametrize("failed_stage", ["finalize", "promote"])
def test_native_leaf_failure_keeps_completion_unpublished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    runner = _FinalizationRunner(
        finalize_returncode=1 if failed_stage == "finalize" else 0,
        promote_returncode=1 if failed_stage == "promote" else 0,
    )

    with pytest.raises(P6FinalizationRoundAdapterError):
        run_finalization_round(*fixture["args"], runner)

    assert [call["stage"] for call in runner.calls] == (
        ["finalize"] if failed_stage == "finalize" else ["finalize", "promote"]
    )
    _assert_unpublished(fixture)


@pytest.mark.parametrize("failure_call", [1, 2, 3, 4])
def test_publication_barrier_is_last_and_writer_failure_restores_nonfinal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    writes: list[Path] = []

    def writer(path: Path, payload: Mapping[str, Any]) -> None:
        writes.append(path)
        if len(writes) == failure_call:
            raise OSError("injected writer failure")
        _write_json(path, payload)

    with pytest.raises(P6FinalizationRoundAdapterError):
        run_finalization_round(*fixture["args"], _FinalizationRunner(), writer=writer)

    assert writes[:failure_call] == [
        fixture["actual_feedback"], fixture["actual_receipt"], fixture["task_state"], fixture["barrier"]
    ][:failure_call]
    assert not fixture["barrier"].exists()
    assert _read_json(fixture["task_state"])["stage"] == "ap"


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    profile = _profile(tmp_path)
    round_root = tmp_path / "round"
    request = _write_round_request(round_root, ("fp16", "int8", "fp16", "int8"))
    task_state = _write_ap_state(round_root, request)
    _write_native_inputs(round_root)
    _set_runtime_env(monkeypatch, tmp_path)
    actual_feedback = round_root / "actual-feedback.json"
    actual_receipt = round_root / "receipt.json"
    barrier = round_root / "barrier.json"
    return {
        "profile": profile,
        "request": request,
        "task_state": task_state,
        "actual_feedback": actual_feedback,
        "actual_receipt": actual_receipt,
        "barrier": barrier,
        "round_root": round_root,
        "args": (
            profile, round_root / "measurement-request.json", task_state, actual_feedback,
            actual_receipt, barrier, round_root,
        ),
        "paths": {
            "task_state": task_state,
            "actual_feedback": actual_feedback,
            "actual_receipt": actual_receipt,
            "finalization_barrier": barrier,
        },
    }


def _profile(tmp_path: Path) -> ValidatedPostSourceAdapterProfile:
    private = tmp_path / "private"
    leaves = []
    for index, name in enumerate(("feedback_finalize", "feedback_promote")):
        cwd = private / name
        cwd.mkdir(parents=True)
        script = cwd / f"{name}.py"
        script.write_text("# fake native leaf\n", encoding="utf-8")
        script.chmod(0o700)
        leaves.append(PostSourceLeaf(name, script, cwd, str(index) * 64))
    return ValidatedPostSourceAdapterProfile(
        "p6_post_source_adapter_profile_v1", private, Path(sys.executable), (), tuple(leaves)
    )


def _write_ap_state(round_root: Path, request: Mapping[str, Any]) -> Path:
    path = _write_task_state(round_root, request)
    state = _read_json(path)
    state["stage"] = "ap"
    _write_json(path, state)
    return path


def _write_native_inputs(round_root: Path) -> None:
    files = (
        round_root / "performance/performance_manifest.json",
        round_root / "performance/performance_state.jsonl",
        round_root / "ap/ap_plan.jsonl",
        round_root / "ap/ap_state.jsonl",
    )
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def _native_feedback_row(
    request_row: Mapping[str, Any], request: Mapping[str, Any], status: str
) -> dict[str, Any]:
    row = {
        **dict(request_row),
        "schema_version": "stage5_feedback_row_v2",
        "measurement_request_row_sha256": request["row_sha256"][request_row["row_id"]],
        "terminal_status": status,
        "latency_ms": 2.0,
        "energy_j": 0.5,
        "ap30": 0.91,
        "ap50": 0.82,
        "ap70": 0.73,
        "failure_reason": None,
    }
    if status != "measured_success_gold":
        row.update(
            latency_ms=None, energy_j=None, ap30=None, ap50=None, ap70=None,
            failure_reason="private/path:reason",
        )
    return row


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "measured": sum(row["terminal_status"] == "measured_success_gold" for row in rows),
        "failure": sum(row["terminal_status"] != "measured_success_gold" for row in rows),
        "pending": 0,
        "total": 4,
    }


def _promoted_row(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(row["graph_features"])
    actual = {**candidate, "graph_feature_provenance": "materialized_onnx_extracted_v1"}
    promoted = {
        **dict(row),
        "candidate_graph_features": candidate,
        "candidate_graph_features_sha256": _canonical_sha(candidate),
        "graph_features": actual,
        "materialized_graph_features_sha256": _canonical_sha(actual),
        "historical_feedback_row_sha256": _canonical_sha(row),
        "feedback_feature_contract": "actual_feedback_v3",
        "graph_feature_promotion_schema": "stage5_actual_feedback_promotion_v3",
    }
    return {**promoted, "actual_feedback_row_sha256": _canonical_sha(promoted)}


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _completion_mapping(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "measurement_request_sha256": request["measurement_request_sha256"],
        "row_sha256": dict(request["row_sha256"]),
        "source_evidence_sha256": {
            row["row_id"]: row["source_evidence_sha256"] for row in request["rows"]
        },
    }


def _assert_native_argv(
    fixture: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]
) -> None:
    profile = fixture["profile"]
    root = fixture["round_root"]
    finalize = calls[0]["argv"]
    promote = calls[1]["argv"]
    legacy_manifest = Path(finalize[3])
    legacy_request = Path(finalize[5])
    assert legacy_manifest.parent == legacy_request.parent
    assert legacy_manifest.parent.parent == root
    assert legacy_manifest.parent.name.startswith(".legacy-finalization-")
    assert finalize == (
        str(profile.project_python), str(profile.leaves[0].implementation),
        "--manifest-json", str(legacy_manifest),
        "--measurement-request-json", str(legacy_request),
        "--ap-plan-jsonl", str(root / "ap/ap_plan.jsonl"),
        "--performance-state-jsonl", str(root / "performance/performance_state.jsonl"),
        "--ap-state-jsonl", str(root / "ap/ap_state.jsonl"),
        "--output-dir", str(root / "final"),
    )
    assert promote == (
        str(profile.project_python), str(profile.leaves[1].implementation),
        "--measurement-request-json", str(legacy_request),
        "--feedback-json", str(root / "final/stage5_feedback_v2_final.json"),
        "--output-dir", str(root / "actual_feedback"),
    )


def _interface() -> dict[str, Any]:
    row = {
        "rows_key": "rows", "row_id_key": "row_id", "row_hash_key": "row_sha256",
        "source_evidence_key": "source_evidence_sha256", "status_key": "terminal_status",
    }
    mapping = {
        "request_sha256_key": "measurement_request_sha256", "row_hashes_key": "row_sha256",
        "source_evidence_key": "source_evidence_sha256",
    }
    return {
        "output_layout": {"task_state": {**row, "stage_key": "stage", "stage_order": ["ap", "finalization"]}},
        "actual_feedback": {
            "result": {**row, "metric_keys": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"]},
            "receipt": mapping, "finalization_barrier": mapping,
        },
    }


def _assert_unpublished(fixture: Mapping[str, Any]) -> None:
    assert not fixture["actual_feedback"].exists()
    assert not fixture["actual_receipt"].exists()
    assert not fixture["barrier"].exists()
    assert _read_json(fixture["task_state"])["stage"] == "ap"


def _set_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(("2", "5", "7")))
    monkeypatch.setenv("P6_HISTORY_RUN_MODE", "bound")
    monkeypatch.setenv("P6_HISTORY_PRIVATE_ROOT", str(tmp_path / "private"))
    monkeypatch.setenv("P6_HISTORY_TASK_STATE", str(tmp_path / "round/state/task-state.json"))
    monkeypatch.setenv("P6_HISTORY_ROUND_OUTPUT_ROOT", str(tmp_path / "round"))
