from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from framework.stage6 import p6_history_measurement_v1 as measurement
from framework.stage6 import p6_source_reuse_evidence_v1 as evidence
from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_history_measurement_v1 import (
    P6HistoryMeasurementError,
    run_history_measurement_batch,
)
from framework.stage6.p6_history_source_materialization_v1 import (
    project_source_materialization_request,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (
    P6SourceReuseEvidenceError,
    SOURCE_ARTIFACT_KEYS,
    SOURCE_MARKER_KEYS,
    canonical_json_sha256,
    create_fresh_run_context,
    plan_source_reuse_paths,
    receipt_path_for_group,
)
from tests.stage6 import test_p6_history_measurement as fixtures
from tests.stage6 import test_p6_source_reuse_evidence_receipts as reuse_fixtures


DIRECTORY_KEYS = frozenset(
    {"checkpoint_dir", "calibration_root", "trt_calibration_dir"}
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _rehash(request: dict[str, Any]) -> None:
    request["row_sha256"] = {
        row["row_id"]: canonical_json_sha256(row) for row in request["rows"]
    }
    request["measurement_request_sha256"] = canonical_json_sha256(
        {key: value for key, value in request.items() if key != "measurement_request_sha256"}
    )


def _make_mixed_q_group(request: dict[str, Any]) -> None:
    first = request["rows"][0]
    second = request["rows"][1]
    preserved_q = second["q_mode"]
    duplicate = copy.deepcopy(first)
    duplicate["q_mode"] = preserved_q
    duplicate["strategy_id"] = f"q={preserved_q}"
    duplicate["genome"][-1] = preserved_q
    duplicate["row_id"] = first["row_id"].replace("q=fp16", f"q={preserved_q}")
    duplicate["manifest_job_id"] = duplicate["row_id"]
    request["rows"][1] = duplicate
    _rehash(request)


def _flip_q_modes(request: dict[str, Any]) -> None:
    for row in request["rows"]:
        previous = row["q_mode"]
        replacement = "int8" if previous == "fp16" else "fp16"
        row["q_mode"] = replacement
        row["strategy_id"] = f"q={replacement}"
        row["genome"][-1] = replacement
        row["row_id"] = row["row_id"].replace(f"q={previous}", f"q={replacement}")
        row["manifest_job_id"] = row["row_id"]
    _rehash(request)


def _create_context(root: Path, request: Mapping[str, Any]) -> None:
    unique_groups: dict[str, Mapping[str, Any]] = {}
    for row in request["rows"]:
        unique_groups.setdefault(row["group_id"], row["source_contract"])
    plan = {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "candidates": [
            {"group_id": row["group_id"], "q_mode": row["q_mode"]}
            for row in request["rows"]
        ],
    }
    registry = {
        "schema_version": "stage5_candidate_source_registry_v2",
        "groups": [
            {"group_id": group_id, "source_contract": contract}
            for group_id, contract in sorted(unique_groups.items())
        ],
    }
    _write_json(root / "pyramid_candidate_plan.json", plan)
    _write_json(root / "source_registry.json", registry)
    create_fresh_run_context(
        local_output_root=root,
        task_contract={
            "task_id": request["task_id"],
            "task_sha256": request["task_sha256"],
        },
        code_revision="0123456789abcdef",
        candidate_plan=plan,
        source_registry=registry,
    )


class BundleRunner(fixtures.FakeRunner):
    def __init__(
        self,
        request: Mapping[str, Any],
        *,
        bundle_mutation: str | None = None,
        fail_call: int | None = None,
    ) -> None:
        super().__init__(request, fail_call=fail_call)
        self.bundle_mutation = bundle_mutation

    def _write_source_markers(self, argv: Sequence[str]) -> None:
        row = next(item for item in self.request["rows"] if item["group_id"] == argv[6])
        contract = row["source_contract"]
        for key in SOURCE_ARTIFACT_KEYS:
            if self.bundle_mutation == f"missing_{key}":
                continue
            path = Path(contract[key])
            if key in DIRECTORY_KEYS:
                path.mkdir(parents=True)
                (path / "payload.bin").write_bytes(key.encode("utf-8"))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(key.encode("utf-8"))
        for key in SOURCE_MARKER_KEYS:
            if self.bundle_mutation == f"missing_{key}":
                continue
            path = Path(contract[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(key.encode("utf-8"))
        request_mtime = Path(argv[2]).stat().st_mtime_ns
        training = Path(contract["training_done_marker"])
        source = Path(contract["source_done_marker"])
        if training.exists() and source.exists():
            os.utime(training, ns=(request_mtime + 1, request_mtime + 1))
            os.utime(source, ns=(request_mtime + 2, request_mtime + 2))
        if self.bundle_mutation == "reversed_markers":
            os.utime(training, ns=(request_mtime + 3, request_mtime + 3))
        if self.bundle_mutation == "wrapper_receipt":
            receipt = receipt_path_for_group(
                plan_source_reuse_paths(Path(contract["checkpoint_path"]).parents[2]),
                row["group_id"],
            )
            receipt.write_bytes(b'{"wrapper":"forbidden"}\n')


def _runtime(
    tmp_path: Path,
    *,
    mixed_q: bool = False,
    create_context: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    private_root = tmp_path / "private"
    private_root.mkdir()
    controller_round = private_root / "controller-round-0"
    controller_round.mkdir()
    request = dict(
        project_source_materialization_request(
            fixtures._unprojected_recipe_v2_request(private_root)
        ).request
    )
    if mixed_q:
        _make_mixed_q_group(request)
    binding = fixtures._binding(private_root)
    if create_context:
        _create_context(private_root, request)
    return request, binding, controller_round


def _source_groups(runner: BundleRunner) -> tuple[str, ...]:
    return tuple(
        call.argv[6]
        for call in runner.calls
        if Path(call.argv[0]).name == "stage5_materialize_round_sources_v1.sh"
    )


def _downstream(runner: BundleRunner) -> tuple[str, ...]:
    names = {
        "quantize",
        "stage5_build_performance_plan_v2.py",
        "measure-ap",
        "stage5_finalize_feedback_v2.py",
    }
    return tuple(
        Path(call.argv[0]).name
        for call in runner.calls
        if Path(call.argv[0]).name in names
    )


@pytest.mark.parametrize(
    "mutation", ("checkpoint_drift", "config_drift", "missing", "symlink", "overlap", "inode_alias")
)
def test_measurement_revalidates_external_binding_before_gpu(
    tmp_path: Path, mutation: str
) -> None:
    request, binding, public_round = _runtime(tmp_path)
    first_contract = request["rows"][0]["source_contract"]
    external = first_contract["external_training_binding"]
    checkpoint = Path(external["base_checkpoint_path"])
    config = Path(external["pyramid_config_path"])
    if mutation == "checkpoint_drift":
        checkpoint.write_bytes(b"drift-after-provision")
    elif mutation == "config_drift":
        config.write_bytes(b"config-drift-after-provision")
    elif mutation == "missing":
        checkpoint.unlink()
    elif mutation == "symlink":
        checkpoint.unlink()
        checkpoint.symlink_to(config)
    elif mutation == "overlap":
        overlapping = Path(first_contract["checkpoint_path"])
        overlapping.parent.mkdir(parents=True, exist_ok=True)
        overlapping.write_bytes(b"overlapping-checkpoint")
        digest = hashlib.sha256(overlapping.read_bytes()).hexdigest()
        for row in request["rows"]:
            row_external = row["source_contract"]["external_training_binding"]
            row_external["base_checkpoint_path"] = str(overlapping)
            row_external["base_checkpoint_sha256"] = digest
            row["source_contract_sha256"] = canonical_json_sha256(
                row["source_contract"]
            )
        _rehash(request)
    else:
        alias = Path(first_contract["checkpoint_path"])
        alias.parent.mkdir(parents=True, exist_ok=True)
        os.link(checkpoint, alias)
    probe = fixtures.FakeProbe()
    runner = BundleRunner(request)

    with pytest.raises(P6HistoryMeasurementError, match="^history_execution_invalid$"):
        run_history_measurement_batch(request, binding, public_round, runner, probe)

    assert probe.calls == []
    assert runner.calls == []


def _install_deletion_after_validation(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
) -> dict[str, Any]:
    original_lstat = Path.lstat
    original_stat = Path.stat
    state: dict[str, Any] = {"validated": 0, "fired": False}

    def delete() -> None:
        if not state["fired"]:
            state["fired"] = True
            target.unlink()

    def racing_lstat(path: Path):
        if path == target and state["validated"]:
            delete()
        result = original_lstat(path)
        if path == target:
            state["validated"] += 1
        return result

    def racing_stat(path: Path, *args: Any, **kwargs: Any):
        if path == target and state["validated"]:
            delete()
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", racing_lstat)
    monkeypatch.setattr(Path, "stat", racing_stat)
    return state


def _install_late_regular_mutation(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    mutation: str,
) -> dict[str, Any]:
    original_lstat = Path.lstat
    state: dict[str, Any] = {"validated": 0, "fired": False}

    def racing_lstat(path: Path):
        info = original_lstat(path)
        if path != target:
            return info
        state["validated"] += 1
        if state["validated"] != 3:
            return info
        state["fired"] = True
        payload = b"X" * max(info.st_size, 1)
        if mutation == "inode_replacement":
            target.unlink()
        target.write_bytes(payload)
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", racing_lstat)
    return state


def _direct_race_target(fixture: Any, target_key: str) -> Path:
    if target_key == "measurement_request":
        return fixture.private_root / "private-runs/0/measurement-request.json"
    return Path(fixture.request["rows"][0]["source_contract"][target_key])


def test_first_use_publishes_receipts_after_sorted_source_calls(tmp_path: Path) -> None:
    request, binding, public_round = _runtime(tmp_path)
    runner = BundleRunner(request)

    feedback = run_history_measurement_batch(
        request, binding, public_round, runner, fixtures.FakeProbe()
    )

    group_ids = tuple(sorted({row["group_id"] for row in request["rows"]}))
    assert _source_groups(runner) == group_ids
    source_calls = [
        call
        for call in runner.calls
        if Path(call.argv[0]).name == "stage5_materialize_round_sources_v1.sh"
    ]
    assert all(call.cwd.name == "0" for call in source_calls)
    assert all(set(call.env) == set(EXPECTED_HISTORY_ENV_KEYS) for call in source_calls)
    assert all("PYTHONPATH" not in call.env and call.shell is False for call in source_calls)
    paths = plan_source_reuse_paths(public_round.parent)
    assert all(receipt_path_for_group(paths, group_id).is_file() for group_id in group_ids)
    assert len(feedback["rows"]) == 4
    assert _downstream(runner) == (
        "quantize",
        "stage5_build_performance_plan_v2.py",
        "measure-ap",
        "stage5_finalize_feedback_v2.py",
    )


def test_same_round_fp16_int8_share_one_source_call_but_keep_all_rows(
    tmp_path: Path,
) -> None:
    request, binding, public_round = _runtime(tmp_path, mixed_q=True)
    runner = BundleRunner(request)

    feedback = run_history_measurement_batch(
        request, binding, public_round, runner, fixtures.FakeProbe()
    )

    group_ids = tuple(sorted({row["group_id"] for row in request["rows"]}))
    assert _source_groups(runner) == group_ids
    assert len(_source_groups(runner)) == 3
    assert len(feedback["rows"]) == 4


def test_later_round_ready_groups_skip_source_and_still_run_downstream(
    tmp_path: Path,
) -> None:
    request, binding, round_zero = _runtime(tmp_path)
    first = BundleRunner(request)
    run_history_measurement_batch(request, binding, round_zero, first, fixtures.FakeProbe())
    later = copy.deepcopy(request)
    later["round_index"] = 2
    _flip_q_modes(later)
    round_two = round_zero.parent / "controller-round-2"
    round_two.mkdir()
    second = BundleRunner(later)

    feedback = run_history_measurement_batch(
        later, binding, round_two, second, fixtures.FakeProbe()
    )

    assert _source_groups(second) == ()
    assert len(feedback["rows"]) == 4
    assert len(_downstream(second)) == 4


def test_unprojected_request_is_written_and_receipted_as_one_projected_truth(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    public_round = private_root / "controller-round-0"
    public_round.mkdir()
    original = fixtures._unprojected_recipe_v2_request(private_root)
    snapshot = copy.deepcopy(original)
    projected = dict(project_source_materialization_request(original).request)
    binding = fixtures._binding(private_root)
    _create_context(private_root, projected)
    runner = BundleRunner(projected)

    feedback = run_history_measurement_batch(
        original, binding, public_round, runner, fixtures.FakeProbe()
    )

    written = json.loads(
        (private_root / "private-runs/0/measurement-request.json").read_text(
            encoding="utf-8"
        )
    )
    assert original == snapshot
    assert written == projected
    assert feedback["measurement_request_sha256"] == projected[
        "measurement_request_sha256"
    ]


def test_already_projected_request_remains_immutable_and_idempotent(tmp_path: Path) -> None:
    request, binding, public_round = _runtime(tmp_path)
    snapshot = copy.deepcopy(request)

    feedback = run_history_measurement_batch(
        request, binding, public_round, BundleRunner(request), fixtures.FakeProbe()
    )

    assert request == snapshot
    assert feedback["measurement_request_sha256"] == request[
        "measurement_request_sha256"
    ]


def test_missing_context_fails_before_gpu_or_process(tmp_path: Path) -> None:
    request, binding, public_round = _runtime(tmp_path, create_context=False)
    runner = BundleRunner(request)
    probe = fixtures.FakeProbe()

    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(request, binding, public_round, runner, probe)

    assert str(exc_info.value) == "history_execution_invalid"
    assert probe.calls == []
    assert runner.calls == []


def test_preexisting_marker_fails_before_gpu_and_source_launch(tmp_path: Path) -> None:
    request, binding, public_round = _runtime(tmp_path)
    marker = Path(request["rows"][0]["source_contract"]["training_done_marker"])
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"stale")
    runner = BundleRunner(request)
    probe = fixtures.FakeProbe()

    with pytest.raises(P6HistoryMeasurementError):
        run_history_measurement_batch(request, binding, public_round, runner, probe)

    assert probe.calls == []
    assert runner.calls == []


@pytest.mark.parametrize("mutation", ("partial", "stale", "mismatch"))
def test_invalid_reuse_state_fails_before_gpu_or_process(
    tmp_path: Path, mutation: str
) -> None:
    request, binding, first_round = _runtime(tmp_path)
    run_history_measurement_batch(
        request, binding, first_round, BundleRunner(request), fixtures.FakeProbe()
    )
    first_row = request["rows"][0]
    if mutation == "partial":
        Path(first_row["source_contract"]["checkpoint_path"]).unlink()
    elif mutation == "mismatch":
        Path(first_row["source_contract"]["checkpoint_path"]).write_bytes(b"tampered")
    else:
        receipt = receipt_path_for_group(
            plan_source_reuse_paths(first_round.parent), first_row["group_id"]
        )
        raw = json.loads(receipt.read_text(encoding="utf-8"))
        raw["run_context_sha256"] = "1" * 64
        raw["receipt_sha256"] = canonical_json_sha256(
            {key: value for key, value in raw.items() if key != "receipt_sha256"}
        )
        _write_json(receipt, raw)
    later = copy.deepcopy(request)
    later["round_index"] = 1
    _rehash(later)
    later_round = first_round.parent / "controller-round-1"
    later_round.mkdir()
    runner = BundleRunner(later)
    probe = fixtures.FakeProbe()

    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(later, binding, later_round, runner, probe)

    assert str(exc_info.value) == "history_execution_invalid"
    assert probe.calls == []
    assert runner.calls == []


def test_missing_private_producer_round_is_redacted_before_reuse_process(
    tmp_path: Path,
) -> None:
    token_root = tmp_path / "PRIVATE-TOKEN-P6-REUSE"
    token_root.mkdir()
    request, binding, first_round = _runtime(token_root)
    run_history_measurement_batch(
        request, binding, first_round, BundleRunner(request), fixtures.FakeProbe()
    )
    private_producer_round = first_round.parent / "private-runs/0"
    relocated = first_round.parent / "PRIVATE-TOKEN-P6-REUSE-producer-round"
    private_producer_round.rename(relocated)
    later = copy.deepcopy(request)
    later["round_index"] = 1
    _rehash(later)
    later_round = first_round.parent / "controller-round-1"
    later_round.mkdir()
    runner = BundleRunner(later)
    probe = fixtures.FakeProbe()

    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(later, binding, later_round, runner, probe)

    assert str(exc_info.value) == "history_execution_invalid"
    assert "PRIVATE-TOKEN-P6-REUSE" not in str(exc_info.value)
    assert str(token_root) not in str(exc_info.value)
    assert runner.calls == []
    assert probe.calls == []


@pytest.mark.parametrize(
    "target_key",
    (*SOURCE_MARKER_KEYS, "measurement_request"),
)
def test_first_use_mtime_race_is_redacted_before_receipt_or_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_key: str,
) -> None:
    token_root = tmp_path / "PRIVATE-TOKEN-P6-MTIME-RACE"
    token_root.mkdir()
    request, binding, public_round = _runtime(token_root)
    group_id = min(row["group_id"] for row in request["rows"])
    row = next(row for row in request["rows"] if row["group_id"] == group_id)
    target = (
        public_round.parent / "private-runs/0/measurement-request.json"
        if target_key == "measurement_request"
        else Path(row["source_contract"][target_key])
    )
    state = _install_deletion_after_validation(monkeypatch, target)
    runner = BundleRunner(request)

    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(
            request, binding, public_round, runner, fixtures.FakeProbe()
        )

    assert state["fired"] is True
    assert str(exc_info.value) == "history_execution_invalid"
    assert "PRIVATE-TOKEN-P6-MTIME-RACE" not in str(exc_info.value)
    assert str(token_root) not in str(exc_info.value)
    paths = plan_source_reuse_paths(public_round.parent)
    assert not receipt_path_for_group(paths, group_id).exists()
    assert _downstream(runner) == ()


@pytest.mark.parametrize(
    ("target_key", "mutation"),
    tuple(
        (target_key, mutation)
        for target_key in (*SOURCE_MARKER_KEYS, "measurement_request")
        for mutation in ("inode_replacement", "in_place")
    ),
)
def test_late_regular_identity_race_rolls_back_direct_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_key: str,
    mutation: str,
) -> None:
    fixture = reuse_fixtures._fresh_fixture(tmp_path / "PRIVATE-TOKEN-P6-LATE-RACE")
    reuse_fixtures._write_complete_bundle(fixture)
    state = _install_late_regular_mutation(
        monkeypatch, _direct_race_target(fixture, target_key), mutation
    )

    with pytest.raises(P6SourceReuseEvidenceError) as exc_info:
        reuse_fixtures._publish(fixture)

    assert state["fired"] is True
    assert str(exc_info.value) == "history_execution_invalid"
    assert exc_info.value.private_category == "p6_source_reuse_mismatch"
    assert "PRIVATE-TOKEN-P6-LATE-RACE" not in str(exc_info.value)
    assert not reuse_fixtures._receipt_path(fixture).exists()


@pytest.mark.parametrize("target_key", (*SOURCE_MARKER_KEYS, "measurement_request"))
def test_late_regular_identity_race_stops_measurement_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_key: str,
) -> None:
    token_root = tmp_path / "PRIVATE-TOKEN-P6-LATE-RACE"
    token_root.mkdir()
    request, binding, public_round = _runtime(token_root)
    group_id = min(row["group_id"] for row in request["rows"])
    row = next(row for row in request["rows"] if row["group_id"] == group_id)
    target = (
        public_round.parent / "private-runs/0/measurement-request.json"
        if target_key == "measurement_request"
        else Path(row["source_contract"][target_key])
    )
    state = _install_late_regular_mutation(monkeypatch, target, "inode_replacement")
    runner = BundleRunner(request)

    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(request, binding, public_round, runner, fixtures.FakeProbe())

    assert state["fired"] is True
    assert str(exc_info.value) == "history_execution_invalid"
    assert "PRIVATE-TOKEN-P6-LATE-RACE" not in str(exc_info.value)
    paths = plan_source_reuse_paths(public_round.parent)
    assert not receipt_path_for_group(paths, group_id).exists()
    assert _downstream(runner) == ()


@pytest.mark.parametrize("mutation", ("source_bytes", "receipt_bytes", "unrelated_receipt"))
def test_post_link_mutation_never_returns_stale_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = reuse_fixtures._fresh_fixture(tmp_path / "PRIVATE-TOKEN-P6-POST-LINK")
    reuse_fixtures._write_complete_bundle(fixture)
    receipt_path = reuse_fixtures._receipt_path(fixture)
    source = _direct_race_target(fixture, "source_done_marker")
    unrelated = b"PRIVATE-TOKEN-P6-UNRELATED\n"
    original_link = evidence.os.link

    def mutating_link(src: Path, dst: Path, **kwargs: Any) -> None:
        original_link(src, dst, **kwargs)
        if Path(dst) != receipt_path:
            return
        if mutation == "source_bytes":
            source.write_bytes(b"X" * source.stat().st_size)
        elif mutation == "receipt_bytes":
            receipt_path.write_bytes(b"X" * receipt_path.stat().st_size)
        else:
            receipt_path.unlink()
            receipt_path.write_bytes(unrelated)

    monkeypatch.setattr(evidence.os, "link", mutating_link)

    with pytest.raises(P6SourceReuseEvidenceError) as exc_info:
        reuse_fixtures._publish(fixture)

    assert str(exc_info.value) == "history_execution_invalid"
    assert exc_info.value.private_category == "p6_source_reuse_mismatch"
    assert "PRIVATE-TOKEN-P6-POST-LINK" not in str(exc_info.value)
    if mutation == "unrelated_receipt":
        assert receipt_path.read_bytes() == unrelated
    else:
        assert not receipt_path.exists()


def test_tamper_between_publication_and_downstream_gate_stops_all_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, binding, public_round = _runtime(tmp_path)
    runner = BundleRunner(request)
    original_gate = measurement.require_selected_groups_ready_current_run

    def _tampering_gate(*args: Any, **kwargs: Any):
        contract = request["rows"][0]["source_contract"]
        Path(contract["checkpoint_path"]).write_bytes(b"tampered-between-gates")
        return original_gate(*args, **kwargs)

    monkeypatch.setattr(
        measurement, "require_selected_groups_ready_current_run", _tampering_gate
    )

    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(
            request, binding, public_round, runner, fixtures.FakeProbe()
        )

    assert str(exc_info.value) == "history_execution_invalid"
    assert _downstream(runner) == ()


@pytest.mark.parametrize(
    "mutation",
    ("wrapper_receipt", "missing_checkpoint_path", "missing_training_done_marker", "reversed_markers"),
)
def test_invalid_first_use_bundle_stops_every_downstream_stage(
    tmp_path: Path, mutation: str
) -> None:
    request, binding, public_round = _runtime(tmp_path)
    runner = BundleRunner(request, bundle_mutation=mutation)

    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(request, binding, public_round, runner, fixtures.FakeProbe())

    assert str(exc_info.value) == "history_execution_invalid"
    assert _downstream(runner) == ()


def test_source_failure_is_redacted_and_stops_downstream(tmp_path: Path) -> None:
    request, binding, public_round = _runtime(tmp_path)
    runner = BundleRunner(request, fail_call=2)

    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(request, binding, public_round, runner, fixtures.FakeProbe())

    assert str(exc_info.value) in {"history_execution_invalid", "history_execution_failed"}
    assert "private" not in str(exc_info.value).lower()
    assert _downstream(runner) == ()


def test_private_runner_token_never_reaches_public_error(tmp_path: Path) -> None:
    request, binding, public_round = _runtime(tmp_path)

    class PrivateTokenRunner(BundleRunner):
        def run(self, argv: Sequence[str], **kwargs: Any):
            if Path(argv[0]).name == "stage5_materialize_round_sources_v1.sh":
                raise RuntimeError("PRIVATE-TOKEN-P6-REUSE")
            return super().run(argv, **kwargs)

    runner = PrivateTokenRunner(request)
    with pytest.raises(P6HistoryMeasurementError) as exc_info:
        run_history_measurement_batch(
            request, binding, public_round, runner, fixtures.FakeProbe()
        )

    assert str(exc_info.value) == "history_execution_invalid"
    assert "PRIVATE-TOKEN-P6-REUSE" not in str(exc_info.value)
    assert _downstream(runner) == ()
