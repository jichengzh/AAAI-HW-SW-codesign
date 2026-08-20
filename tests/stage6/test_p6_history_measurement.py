from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from framework.stage6.p6_history_binding_v1 import GpuRecord, public_binding_projection
from framework.stage6.p6_history_measurement_v1 import (
    P6HistoryMeasurementError,
    run_history_measurement_batch,
)
from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
from framework.stage6.p6_history_source_materialization_v1 import (
    project_source_materialization_request,
)


METRICS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
SYNTHETIC_GPU_INDICES = (101, 103, 107)
STAGES = (
    "source_materialization",
    "quantization",
    "performance",
    "ap",
    "finalization",
)


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _executable(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic private executable\n", encoding="utf-8")
    path.chmod(0o700)
    return str(path)


def _binding(
    private_root: Path, *, gpu_indices: tuple[int, int, int] = SYNTHETIC_GPU_INDICES
) -> dict[str, Any]:
    component_root = private_root / "components"
    components = {
        "controller": _executable(component_root / "stage5_task_round_controller_v3.sh"),
        "source_materializer": _executable(
            component_root / "stage5_materialize_round_sources_v1.sh"
        ),
        "performance_plan": _executable(component_root / "stage5_build_performance_plan_v2.py"),
        "finalizer": _executable(component_root / "stage5_finalize_feedback_v2.py"),
    }
    quantize = _executable(component_root / "quantize")
    ap = _executable(component_root / "measure-ap")
    activate = _executable(component_root / "activate")
    row_fields = {
        "rows_key": "rows",
        "row_id_key": "row_id",
        "row_hash_key": "row_sha256",
        "source_evidence_key": "source_evidence_sha256",
        "status_key": "terminal_status",
    }
    terminal = [
        "measured_success_gold",
        "feasibility_failure",
        "numerical_feasibility_failure",
    ]
    validation = {
        "format": "json",
        "request_sha256_key": "measurement_request_sha256",
        "row_hashes_key": "row_sha256",
        "source_evidence_key": "source_evidence_sha256",
    }
    interface = {
        "schema_version": "p6_history_runner_interface_v1",
        "controller": {"argv": [components["controller"]]},
        "execution_chain": [
            {
                "stage": "source_materialization",
                "argv": [
                    components["source_materializer"],
                    "{measurement_request}",
                    "{round_output_root}",
                ],
                "required_placeholders": [
                    "{measurement_request}",
                    "{round_output_root}",
                ],
            },
            {
                "stage": "quantization",
                "argv": [quantize, "{task_state}", "{round_output_root}"],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "performance",
                "argv": [
                    components["performance_plan"],
                    "{task_state}",
                    "{round_output_root}",
                ],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "ap",
                "argv": [ap, "{task_state}", "{round_output_root}"],
                "required_placeholders": ["{task_state}", "{round_output_root}"],
            },
            {
                "stage": "finalization",
                "argv": [
                    components["finalizer"],
                    "{measurement_request}",
                    "{task_state}",
                    "{actual_feedback}",
                    "{actual_receipt}",
                    "{finalization_barrier}",
                    "{round_output_root}",
                ],
                "required_placeholders": [
                    "{measurement_request}",
                    "{task_state}",
                    "{actual_feedback}",
                    "{actual_receipt}",
                    "{finalization_barrier}",
                    "{round_output_root}",
                ],
            },
        ],
        "environment": {
            "values": {
                "CUDA_VISIBLE_DEVICES": {
                    "kind": "literal",
                    "value": ",".join(str(index) for index in gpu_indices),
                },
                "P6_HISTORY_RUN_MODE": {"kind": "literal", "value": "bound"},
                "P6_HISTORY_PRIVATE_ROOT": {
                    "kind": "private_path",
                    "value": str(private_root),
                },
                "P6_HISTORY_TASK_STATE": {
                    "kind": "placeholder",
                    "value": "{task_state}",
                },
                "P6_HISTORY_ROUND_OUTPUT_ROOT": {
                    "kind": "placeholder",
                    "value": "{round_output_root}",
                },
            },
            "activation_argv": [activate, "private-bound"],
        },
        "output_layout": {
            "round_root_template": "private-runs/{round_id}",
            "task_state": {
                "path_template": "private-runs/{round_id}/state/task-state.json",
                "format": "json",
                **row_fields,
                "stage_key": "stage",
                "row_count": 4,
                "allowed_terminal_statuses": terminal,
                "stage_order": list(STAGES),
            },
        },
        "actual_feedback": {
            "result": {
                "path_template": "private-runs/{round_id}/actual-feedback.json",
                "format": "json",
                **row_fields,
                "row_count": 4,
                "allowed_terminal_statuses": terminal,
                "metric_keys": list(METRICS),
            },
            "receipt": {
                "path_template": "private-runs/{round_id}/receipt.json",
                **validation,
            },
            "finalization_barrier": {
                "path_template": "private-runs/{round_id}/barrier.json",
                **validation,
            },
        },
    }
    return {
        "schema_version": "p6_history_binding_v1",
        "target": {"model": "pyramid", "hardware": "h800", "backend": "tvm_auto"},
        "private_root": str(private_root),
        "component_paths": components,
        "execution_interface": interface,
        "gpu_policy": {
            "indices": list(gpu_indices),
            "uuid_by_index": {
                str(index): f"GPU-synthetic-{index}" for index in gpu_indices
            },
            "model": "h800",
            "maximum_occupancy": 0.05,
        },
        "status": "validated",
    }


def _request() -> dict[str, Any]:
    task_sha = hashlib.sha256(b"task").hexdigest()
    rows: list[dict[str, Any]] = []
    for index in range(4):
        width = [16 + index, 32 + index, 64 + index]
        group_id = f"pyramid|{width[0]}x{width[1]}x{width[2]}"
        evidence = hashlib.sha256(f"evidence-{index}".encode()).hexdigest()
        source_contract = {
            "schema_version": "stage5_source_contract_v1",
            "group_id": group_id,
            "model": "pyramid",
            "width": width,
            "artifact_id": f"artifact-{index}",
            "source_status": "ready",
            "source_evidence_sha256": evidence,
            "materialization_scope": "synthetic_fixture",
        }
        q_mode = "fp16" if index % 2 == 0 else "int8"
        row_id = f"{group_id}|q={q_mode}|profile=h800-tvm-auto"
        rows.append(
            {
                "schema_version": "stage5_candidate_row_v2",
                "task_id": "P6-H800-PYRAMID",
                "task_sha256": task_sha,
                "row_id": row_id,
                "manifest_job_id": row_id,
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "width_schema": ["w0", "w1", "w2"],
                "structure_widths": {
                    "w0": width[0],
                    "w1": width[1],
                    "w2": width[2],
                },
                "genome": [*width, q_mode],
                "strategy_id": f"q={q_mode}",
                "q_mode": q_mode,
                "hardware_id": "h800",
                "capability_profile_id": "h800-tvm-auto",
                "capability_digest": hashlib.sha256(b"profile").hexdigest(),
                "dispatch_key": "tvm_auto",
                "source_status": "ready",
                "materialization_kind": "local_pyramid_tvm",
                "source_evidence_kind": "local_synthetic",
                "source_contract": source_contract,
                "source_contract_sha256": _sha(source_contract),
                "source_evidence_sha256": evidence,
                "graph_features": {
                    "group_id": group_id,
                    "model": "pyramid",
                    "width": width,
                },
            }
        )
    body = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "P6-H800-PYRAMID",
        "task_sha256": task_sha,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": list(METRICS),
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**body, "measurement_request_sha256": _sha(body)}


def _unprojected_recipe_v2_request() -> dict[str, Any]:
    request = _request()
    for row in request["rows"]:
        group_slug = "-".join(map(str, row["width"]))
        contract = row["source_contract"]
        contract["artifact_id"] = f"pyramid-{group_slug}"
        contract.update(
            {
                "stage_widths": {
                    "stage1_width": row["width"][0],
                    "stage2_width": row["width"][1],
                    "stage3_width": row["width"][2],
                },
                "base_checkpoint_path": "/private/synthetic/base/model.ckpt",
                "dataset_root": "/private/synthetic/dataset",
                "training_parameters": {"epochs": 7, "optimizer": "synthetic"},
                "shared_source_paths": {
                    key: f"/private/synthetic/materialized/{group_slug}/{key}"
                    for key in SHARED_SOURCE_PATH_KEYS
                },
            }
        )
        row["source_contract_sha256"] = _sha(contract)
    request["row_sha256"] = {
        row["row_id"]: _sha(row) for row in request["rows"]
    }
    _rehash_request(request)
    return request


def _records(
    *,
    indices: tuple[int, int, int] = SYNTHETIC_GPU_INDICES,
    model: str = "NVIDIA H800 80GB HBM3",
    occupancy: float = 0.0,
    drift_index: int | None = None,
) -> tuple[GpuRecord, ...]:
    return tuple(
        GpuRecord(
            index=index,
            uuid=(
                f"GPU-drift-{index}"
                if index == drift_index
                else f"GPU-synthetic-{index}"
            ),
            model_name=model,
            occupancy=occupancy,
        )
        for index in indices
    )


class FakeProbe:
    def __init__(self, *snapshots: tuple[GpuRecord, ...]) -> None:
        self.snapshots = list(snapshots or (_records(), _records()))
        self.calls: list[tuple[int, ...]] = []

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        self.calls.append(indices)
        return self.snapshots.pop(0)


@dataclass(frozen=True)
class Call:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    shell: bool


@dataclass(frozen=True)
class Result:
    returncode: int = 0


class FakeRunner:
    def __init__(
        self,
        request: Mapping[str, Any],
        *,
        mutation: str | None = None,
        fail_call: int | None = None,
        raise_call: int | None = None,
    ) -> None:
        self.request = copy.deepcopy(dict(request))
        self.mutation = mutation
        self.fail_call = fail_call
        self.raise_call = raise_call
        self.calls: list[Call] = []
        self.initial_state: dict[str, Any] | None = None

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> Result:
        call = Call(tuple(argv), cwd, dict(env), shell)
        self.calls.append(call)
        call_index = len(self.calls)
        if call_index == self.raise_call:
            raise RuntimeError("PRIVATE runner detail")
        if call_index == self.fail_call:
            return Result(23)
        if Path(argv[0]).name == "stage5_materialize_round_sources_v1.sh":
            self.initial_state = json.loads(
                cwd.joinpath("state/task-state.json").read_text(encoding="utf-8")
            )
        if Path(argv[0]).name == "stage5_finalize_feedback_v2.py":
            self._finalize(argv)
        return Result()

    def _finalize(self, argv: Sequence[str]) -> None:
        request_path, state_path, result_path, receipt_path, barrier_path = map(Path, argv[1:6])
        disk_request = json.loads(request_path.read_text(encoding="utf-8"))
        assert disk_request == self.request
        row_hashes = dict(self.request["row_sha256"])
        evidence = {row["row_id"]: row["source_evidence_sha256"] for row in self.request["rows"]}
        state_rows = [
            {
                "row_id": row["row_id"],
                "row_sha256": row_hashes[row["row_id"]],
                "source_evidence_sha256": row["source_evidence_sha256"],
                "terminal_status": "measured_success_gold",
            }
            for row in self.request["rows"]
        ]
        result_rows = [
            {
                **state_row,
                "latency_ms": 2.0 + index,
                "energy_j": 0.5 + index / 10,
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.73,
            }
            for index, state_row in enumerate(state_rows)
        ]
        state: dict[str, Any] = {"stage": "finalization", "rows": state_rows}
        result: dict[str, Any] = {
            "measurement_request_sha256": self.request["measurement_request_sha256"],
            "rows": result_rows,
        }
        validation: dict[str, Any] = {
            "measurement_request_sha256": self.request["measurement_request_sha256"],
            "row_sha256": row_hashes,
            "source_evidence_sha256": evidence,
        }
        receipt = copy.deepcopy(validation)
        barrier = copy.deepcopy(validation)
        self._mutate(state, result, receipt, barrier)
        for path, payload in (
            (state_path, state),
            (result_path, result),
            (receipt_path, receipt),
            (barrier_path, barrier),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        if self.mutation == "symlink_state_parent":
            stale_parent = state_path.parent.with_name("stale-valid-state")
            state_path.parent.rename(stale_parent)
            state_path.parent.symlink_to(stale_parent, target_is_directory=True)

    def _mutate(
        self,
        state: dict[str, Any],
        result: dict[str, Any],
        receipt: dict[str, Any],
        barrier: dict[str, Any],
    ) -> None:
        mutation = self.mutation
        if mutation == "missing_result_row":
            result["rows"].pop()
        elif mutation == "duplicate_result_row":
            result["rows"][-1] = copy.deepcopy(result["rows"][0])
        elif mutation == "extra_result_row":
            result["rows"].append({**result["rows"][0], "row_id": "extra"})
        elif mutation == "bad_metric":
            result["rows"][0]["latency_ms"] = math.inf
        elif mutation == "bad_ap":
            result["rows"][0]["ap70"] = 1.1
        elif mutation == "bad_status":
            result["rows"][0]["terminal_status"] = "public_runner_failure"
        elif mutation == "unsafe_reason":
            row = result["rows"][0]
            for metric in METRICS:
                row.pop(metric)
            row["terminal_status"] = "feasibility_failure"
            row["failure_reason"] = "/private/path leaked SECRET=abc"
            state["rows"][0]["terminal_status"] = "feasibility_failure"
        elif mutation == "missing_reason":
            row = result["rows"][0]
            for metric in METRICS:
                row.pop(metric)
            row["terminal_status"] = "feasibility_failure"
            state["rows"][0]["terminal_status"] = "feasibility_failure"
        elif mutation == "bad_result_request":
            result["measurement_request_sha256"] = "0" * 64
        elif mutation == "bad_result_hash":
            result["rows"][0]["row_sha256"] = "0" * 64
        elif mutation == "bad_result_evidence":
            result["rows"][0]["source_evidence_sha256"] = "0" * 64
        elif mutation == "bad_receipt":
            receipt["row_sha256"].pop(next(iter(receipt["row_sha256"])))
        elif mutation == "legacy_receipt":
            receipt["rows"] = receipt.pop("row_sha256")
        elif mutation == "bad_barrier":
            barrier["source_evidence_sha256"][next(iter(barrier["source_evidence_sha256"]))] = (
                "0" * 64
            )
        elif mutation == "bad_state":
            state["stage"] = "ap"
        elif mutation == "missing_state_row":
            state["rows"].pop()
        elif mutation == "state_status_mismatch":
            state["rows"][0]["terminal_status"] = "feasibility_failure"
        elif mutation == "missing_barrier":
            barrier.clear()


def _run(
    tmp_path: Path,
    *,
    request: Mapping[str, Any] | None = None,
    runner: FakeRunner | None = None,
    probe: FakeProbe | None = None,
) -> tuple[dict[str, Any], FakeRunner, FakeProbe, Path]:
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_output_root = private_root / "controller-round"
    round_output_root.mkdir()
    actual_request = copy.deepcopy(dict(request or _request()))
    actual_runner = runner or FakeRunner(actual_request)
    actual_probe = probe or FakeProbe()
    feedback = run_history_measurement_batch(
        actual_request,
        _binding(private_root),
        round_output_root,
        actual_runner,
        actual_probe,
    )
    return feedback, actual_runner, actual_probe, round_output_root


def _rehash_request(request: dict[str, Any]) -> None:
    body = {key: value for key, value in request.items() if key != "measurement_request_sha256"}
    request["measurement_request_sha256"] = _sha(body)


def test_valid_route_executes_activation_and_five_stages_then_returns_four_rows(
    tmp_path: Path,
) -> None:
    """Catches stage omission/reordering, shell execution, ambient env, or partial feedback."""
    feedback, runner, probe, _ = _run(tmp_path)

    assert feedback["schema_version"] == "p6_h800_coptv2x_feedback_v2"
    assert feedback["measurement_request_sha256"] == _request()["measurement_request_sha256"]
    assert len(feedback["rows"]) == 4
    assert {row["row_id"] for row in feedback["rows"]} == {
        row["row_id"] for row in _request()["rows"]
    }
    assert all(set(row) == {"row_id", "terminal_status", *METRICS} for row in feedback["rows"])
    assert [Path(call.argv[0]).name for call in runner.calls] == [
        "activate",
        "stage5_materialize_round_sources_v1.sh",
        "stage5_materialize_round_sources_v1.sh",
        "stage5_materialize_round_sources_v1.sh",
        "stage5_materialize_round_sources_v1.sh",
        "quantize",
        "stage5_build_performance_plan_v2.py",
        "measure-ap",
        "stage5_finalize_feedback_v2.py",
    ]
    source_calls = [
        call
        for call in runner.calls
        if Path(call.argv[0]).name == "stage5_materialize_round_sources_v1.sh"
    ]
    assert [call.argv[1::2] for call in source_calls] == [
        (
            "--request",
            "--model",
            "--group-id",
            "--gpu",
        )
    ] * 4
    expected_request_path = str(source_calls[0].cwd / "measurement-request.json")
    assert [call.argv[2] for call in source_calls] == [expected_request_path] * 4
    assert [call.argv[4] for call in source_calls] == ["pyramid"] * 4
    assert [call.argv[6] for call in source_calls] == sorted(
        row["group_id"] for row in _request()["rows"]
    )
    assert [call.argv[8] for call in source_calls] == ["101", "103", "107", "101"]
    assert all(call.shell is False for call in runner.calls)
    assert all(Path(call.argv[0]).name not in {"sh", "bash", "shell"} for call in runner.calls)
    assert all(
        set(call.env)
        == {
            "CUDA_VISIBLE_DEVICES",
            "P6_HISTORY_RUN_MODE",
            "P6_HISTORY_PRIVATE_ROOT",
            "P6_HISTORY_TASK_STATE",
            "P6_HISTORY_ROUND_OUTPUT_ROOT",
        }
        for call in runner.calls
    )
    assert probe.calls == [SYNTHETIC_GPU_INDICES, SYNTHETIC_GPU_INDICES]
    assert runner.initial_state is not None
    assert runner.initial_state["stage"] == "initialized"
    assert len(runner.initial_state["rows"]) == 4


def test_measurement_projects_recipe_v2_before_write_invocation_and_feedback(
    tmp_path: Path,
) -> None:
    """Catches original/projected request hashes becoming two measurement truths."""
    original = _unprojected_recipe_v2_request()
    original_snapshot = copy.deepcopy(original)
    projected = dict(project_source_materialization_request(original).request)
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    runner = FakeRunner(projected)

    feedback = run_history_measurement_batch(
        original, _binding(private_root), round_root, runner, FakeProbe()
    )

    request_path = private_root / "private-runs/0/measurement-request.json"
    assert json.loads(request_path.read_text(encoding="utf-8")) == projected
    assert original == original_snapshot
    assert feedback["measurement_request_sha256"] == projected["measurement_request_sha256"]
    source_calls = [
        call
        for call in runner.calls
        if Path(call.argv[0]).name == "stage5_materialize_round_sources_v1.sh"
    ]
    assert [call.argv[2] for call in source_calls] == [str(request_path)] * 4
    assert all(
        "shared_source_paths" not in row["source_contract"]
        for row in projected["rows"]
    )


def test_measurement_projection_is_idempotent_for_already_projected_request(
    tmp_path: Path,
) -> None:
    """Catches a second projection changing canonical row or request hashes."""
    projected = dict(
        project_source_materialization_request(_unprojected_recipe_v2_request()).request
    )
    snapshot = copy.deepcopy(projected)
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()

    feedback = run_history_measurement_batch(
        projected,
        _binding(private_root),
        round_root,
        FakeRunner(projected),
        FakeProbe(),
    )

    assert projected == snapshot
    assert feedback["measurement_request_sha256"] == projected["measurement_request_sha256"]
    assert json.loads(
        (private_root / "private-runs/0/measurement-request.json").read_text(
            encoding="utf-8"
        )
    ) == projected


def test_measurement_revalidates_the_binding_private_policy_before_and_after_execution(
    tmp_path: Path,
) -> None:
    """Catches a runtime probe reverting to a fixed device policy."""
    policy_indices = (211, 223, 227)
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    request = _request()
    runner = FakeRunner(request)
    probe = FakeProbe(
        _records(indices=policy_indices), _records(indices=policy_indices)
    )

    run_history_measurement_batch(
        request,
        _binding(private_root, gpu_indices=policy_indices),
        round_root,
        runner,
        probe,
    )

    assert probe.calls == [policy_indices, policy_indices]


def test_runtime_rejects_policy_that_differs_from_execution_interface_before_probe(
    tmp_path: Path,
) -> None:
    """Catches probe and runner dispatch under different private GPU selections."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    request = _request()
    runner = FakeRunner(request)
    probe = FakeProbe()
    binding = _binding(private_root)
    mismatched_indices = (211, 223, 227)
    binding["gpu_policy"] = {
        "indices": list(mismatched_indices),
        "uuid_by_index": {
            str(index): f"GPU-synthetic-{index}" for index in mismatched_indices
        },
        "model": "h800",
        "maximum_occupancy": 0.05,
    }

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(request, binding, round_root, runner, probe)

    assert raised.value.category == "history_execution_invalid"
    assert probe.calls == []
    assert runner.calls == []


def test_runtime_rejects_binding_uuid_collision_after_whitespace_normalization(
    tmp_path: Path,
) -> None:
    """Catches UUID uniqueness checks performed before canonical whitespace stripping."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    request = _request()
    runner = FakeRunner(request)
    probe = FakeProbe()
    binding = _binding(private_root)
    first, second, _ = SYNTHETIC_GPU_INDICES
    binding["gpu_policy"]["uuid_by_index"][str(first)] = " GPU-collision "
    binding["gpu_policy"]["uuid_by_index"][str(second)] = "GPU-collision"

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(request, binding, round_root, runner, probe)

    assert raised.value.category == "history_execution_invalid"
    assert probe.calls == []
    assert runner.calls == []


def test_public_binding_projection_does_not_expose_private_gpu_policy(tmp_path: Path) -> None:
    """Catches a public binding surface copying private GPU selection or UUIDs."""
    private_binding = _binding(tmp_path)

    projection = public_binding_projection(private_binding)
    serialized_projection = json.dumps(projection, sort_keys=True)

    assert "gpu_policy" not in projection
    assert "GPU-synthetic-101" not in serialized_projection
    assert "101" not in serialized_projection


@pytest.mark.parametrize("indices", ([101, 101, 107], [107, 103, 101], [101, 103]))
def test_runtime_rejects_noncanonical_binding_gpu_policy_indices(
    tmp_path: Path,
    indices: list[int],
) -> None:
    """Catches malformed binding policy indices before probing or execution."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    request = _request()
    binding = _binding(private_root)
    binding["gpu_policy"]["indices"] = indices
    runner = FakeRunner(request)
    probe = FakeProbe()

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(request, binding, round_root, runner, probe)

    assert raised.value.category == "history_execution_invalid"
    assert runner.calls == []
    assert probe.calls == []


def test_allowed_failure_is_returned_with_public_safe_reason(tmp_path: Path) -> None:
    """Catches raw private failure detail escaping into P6 feedback."""
    request = _request()
    feedback, _, _, _ = _run(
        tmp_path, request=request, runner=FakeRunner(request, mutation="unsafe_reason")
    )

    failed = feedback["rows"][0]
    assert failed == {
        "row_id": request["rows"][0]["row_id"],
        "terminal_status": "feasibility_failure",
        "failure_reason": "unspecified",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_result_row",
        "duplicate_result_row",
        "extra_result_row",
        "bad_metric",
        "bad_ap",
        "bad_status",
        "missing_reason",
        "bad_result_request",
        "bad_result_hash",
        "bad_result_evidence",
        "bad_receipt",
        "legacy_receipt",
        "bad_barrier",
        "bad_state",
        "missing_state_row",
        "state_status_mismatch",
        "symlink_state_parent",
        "missing_barrier",
    ],
)
def test_invalid_history_bridge_fails_closed(tmp_path: Path, mutation: str) -> None:
    """Catches incomplete, legacy, mismatched, or non-atomic history completion."""
    request = _request()
    runner = FakeRunner(request, mutation=mutation)
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(
            request, _binding(private_root), round_root, runner, FakeProbe()
        )

    assert raised.value.category == "history_execution_invalid"
    assert "PRIVATE" not in str(raised.value)
    assert not (round_root / "feedback.json").exists()


def _bad_request(case: str) -> dict[str, Any]:
    request = _request()
    if case == "schema":
        request["schema_version"] = "legacy_request"
    elif case == "request_hash":
        request["measurement_request_sha256"] = "0" * 64
    elif case == "row_hash":
        request["row_sha256"][request["rows"][0]["row_id"]] = "0" * 64
        _rehash_request(request)
    elif case == "duplicate_row":
        request["rows"][-1] = copy.deepcopy(request["rows"][0])
        _rehash_request(request)
    elif case == "identity":
        request["rows"][0]["dispatch_key"] = "trt_engine"
        request["row_sha256"][request["rows"][0]["row_id"]] = _sha(request["rows"][0])
        _rehash_request(request)
    elif case == "q_mode":
        request["rows"][0]["q_mode"] = "mixed"
        request["rows"][0]["genome"][-1] = "mixed"
        request["rows"][0]["strategy_id"] = "q=mixed"
        request["row_sha256"][request["rows"][0]["row_id"]] = _sha(request["rows"][0])
        _rehash_request(request)
    elif case == "source_evidence":
        request["rows"][0]["source_evidence_sha256"] = ""
        request["row_sha256"][request["rows"][0]["row_id"]] = _sha(request["rows"][0])
        _rehash_request(request)
    elif case == "budget":
        request["sample_budget"] = 20
        _rehash_request(request)
    elif case == "round":
        request["round_index"] = -1
        _rehash_request(request)
    elif case == "extra_key":
        request["private_override"] = "forbidden"
        _rehash_request(request)
    return request


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "request_hash",
        "row_hash",
        "duplicate_row",
        "identity",
        "q_mode",
        "source_evidence",
        "budget",
        "round",
        "extra_key",
    ],
)
def test_malformed_request_stops_before_gpu_or_process(tmp_path: Path, case: str) -> None:
    """Catches reinterpretation of malformed or non-P6 request variants."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    request = _bad_request(case)
    runner = FakeRunner(request)
    probe = FakeProbe()

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(request, _binding(private_root), round_root, runner, probe)

    assert raised.value.category == "history_request_invalid"
    assert runner.calls == []
    assert probe.calls == []


@pytest.mark.parametrize(
    "probe",
    [
        FakeProbe(_records(model="NVIDIA H100"), _records()),
        FakeProbe(_records(occupancy=0.2), _records()),
        FakeProbe(_records(drift_index=107), _records()),
        FakeProbe(_records(), _records(model="NVIDIA H100")),
        FakeProbe(_records(), _records(occupancy=0.2)),
        FakeProbe(_records(), _records(drift_index=107)),
    ],
)
def test_gpu_admission_and_pre_post_drift_fail_closed(tmp_path: Path, probe: FakeProbe) -> None:
    """Catches use of non-H800, occupied, or UUID-drifted GPUs before feedback."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    request = _request()
    runner = FakeRunner(request)

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(request, _binding(private_root), round_root, runner, probe)

    assert raised.value.category == "history_gpu_admission_failed"
    assert not (round_root / "feedback.json").exists()


@pytest.mark.parametrize("mode", ["nonzero", "exception"])
def test_runner_failure_never_becomes_success(tmp_path: Path, mode: str) -> None:
    """Catches nonzero or raised runner failures being mistaken for finalization."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    request = _request()
    runner = FakeRunner(
        request,
        fail_call=6 if mode == "nonzero" else None,
        raise_call=6 if mode == "exception" else None,
    )

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(
            request, _binding(private_root), round_root, runner, FakeProbe()
        )

    assert raised.value.category == "history_execution_failed"
    assert "PRIVATE" not in str(raised.value)


@pytest.mark.parametrize("mode", ["nonzero", "exception"])
def test_source_runner_failure_stops_before_later_wrappers(
    tmp_path: Path, mode: str
) -> None:
    """Catches source failure continuing into quantization or leaking private details."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    request = _request()
    runner = FakeRunner(
        request,
        fail_call=3 if mode == "nonzero" else None,
        raise_call=3 if mode == "exception" else None,
    )

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(
            request, _binding(private_root), round_root, runner, FakeProbe()
        )

    assert raised.value.category == "history_execution_invalid"
    assert "PRIVATE" not in str(raised.value)
    assert all(
        Path(call.argv[0]).name
        not in {
            "quantize",
            "stage5_build_performance_plan_v2.py",
            "measure-ap",
            "stage5_finalize_feedback_v2.py",
        }
        for call in runner.calls
    )


@pytest.mark.parametrize("case", ["relative", "missing", "symlink"])
def test_unsafe_or_symlinked_controller_round_root_fails_before_process(
    tmp_path: Path, case: str
) -> None:
    """Catches use of an unvalidated controller-local feedback boundary."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    external_root = tmp_path / "external-controller-root"
    if case == "relative":
        round_root = Path("relative-controller-root-must-not-exist")
    elif case == "missing":
        round_root = external_root
    else:
        actual_root = tmp_path / "actual-controller-root"
        actual_root.mkdir()
        external_root.symlink_to(actual_root, target_is_directory=True)
        round_root = external_root
    request = _request()
    runner = FakeRunner(request)
    probe = FakeProbe()

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(request, _binding(private_root), round_root, runner, probe)

    assert raised.value.category == "history_execution_invalid"
    assert runner.calls == []
    assert probe.calls == []


def test_template_resolution_rejects_preexisting_lexical_symlink_parent(
    tmp_path: Path,
) -> None:
    """Catches resolve erasing an in-boundary symlink from a declared template."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    unexpected_target = private_root / "unexpected-target"
    unexpected_target.mkdir()
    (private_root / "link-parent").symlink_to(
        unexpected_target, target_is_directory=True
    )
    binding = _binding(private_root)
    binding["execution_interface"]["output_layout"]["task_state"][
        "path_template"
    ] = "link-parent/{round_id}/task-state.json"
    request = _request()
    runner = FakeRunner(request)
    probe = FakeProbe()

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(request, binding, round_root, runner, probe)

    assert raised.value.category == "history_execution_invalid"
    assert runner.calls == []
    assert probe.calls == []
    assert not (unexpected_target / "0" / "task-state.json").exists()


def test_binding_validation_precedes_gpu_and_process(tmp_path: Path) -> None:
    """Catches adapters consuming mutable/unvalidated interface data directly."""
    private_root = tmp_path / "private"
    private_root.mkdir()
    round_root = private_root / "controller-round"
    round_root.mkdir()
    binding = _binding(private_root)
    binding["execution_interface"]["execution_chain"][0]["stage"] = "tampered"
    request = _request()
    runner = FakeRunner(request)
    probe = FakeProbe()

    with pytest.raises(P6HistoryMeasurementError) as raised:
        run_history_measurement_batch(request, binding, round_root, runner, probe)

    assert raised.value.category == "history_execution_invalid"
    assert runner.calls == []
    assert probe.calls == []
