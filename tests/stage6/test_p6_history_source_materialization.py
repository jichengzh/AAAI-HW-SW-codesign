from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from framework.stage6.p6_history_source_materialization_v1 import (
    P6HistorySourceMaterializationError,
    build_source_invocations,
    run_source_invocations,
)


def _gpu_policy() -> dict[str, object]:
    indices = (101, 103, 107)
    return {
        "indices": list(indices),
        "uuid_by_index": {
            str(index): f"GPU-synthetic-{index}" for index in indices
        },
        "model": "h800",
        "maximum_occupancy": 0.05,
    }


def _executable(path: Path) -> Path:
    path.write_text("synthetic executable\n", encoding="utf-8")
    path.chmod(0o700)
    return path


@dataclass(frozen=True)
class _Result:
    returncode: int = 0


class _RecordingRunner:
    def __init__(self, *, returncode: int = 0, raises: bool = False) -> None:
        self.returncode = returncode
        self.raises = raises
        self.calls: list[
            tuple[tuple[str, ...], Path, Mapping[str, str], bool]
        ] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> _Result:
        self.calls.append((tuple(argv), cwd, dict(env), shell))
        if self.raises:
            raise RuntimeError("PRIVATE path and runner diagnostics")
        return _Result(self.returncode)


def test_source_invocations_dedupe_canonical_group_order_and_round_robin_policy(
    tmp_path: Path,
) -> None:
    """Catches first-seen ordering, ambient GPUs, duplicate calls, or positional argv."""
    materializer = _executable(tmp_path / "source-materializer")
    projected_request = tmp_path / "projected-request.json"
    groups = (
        "pyramid|29x53x101",
        "pyramid|17x31x63",
        "pyramid|23x47x95",
        "pyramid|17x31x63",
    )

    invocations = build_source_invocations(
        projected_request,
        groups,
        source_materializer=materializer,
        validated_gpu_policy=_gpu_policy(),
    )

    assert invocations == (
        (
            str(materializer),
            "--request",
            str(projected_request),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|17x31x63",
            "--gpu",
            "101",
        ),
        (
            str(materializer),
            "--request",
            str(projected_request),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|23x47x95",
            "--gpu",
            "103",
        ),
        (
            str(materializer),
            "--request",
            str(projected_request),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|29x53x101",
            "--gpu",
            "107",
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_gpu",
        "nonpolicy_uuid",
        "duplicate_uuid",
        "malformed_index",
        "wrong_model",
        "wrong_occupancy_gate",
    ],
)
def test_source_invocations_reject_malformed_or_nonpolicy_gpu_policy(
    tmp_path: Path, mutation: str
) -> None:
    """Catches invocation planning from anything except the validated binding policy."""
    policy = _gpu_policy()
    if mutation == "duplicate_gpu":
        policy["indices"] = [101, 101, 107]
    elif mutation == "nonpolicy_uuid":
        policy["uuid_by_index"] = {
            "101": "GPU-synthetic-101",
            "103": "GPU-synthetic-103",
            "109": "GPU-synthetic-109",
        }
    elif mutation == "duplicate_uuid":
        policy["uuid_by_index"]["103"] = " GPU-synthetic-101 "
    elif mutation == "malformed_index":
        policy["indices"] = [101, True, 107]
    elif mutation == "wrong_model":
        policy["model"] = "h100"
    else:
        policy["maximum_occupancy"] = 0.5

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        build_source_invocations(
            tmp_path / "request.json",
            ("pyramid|17x31x63",),
            source_materializer=_executable(tmp_path / "source-materializer"),
            validated_gpu_policy=policy,
        )

    assert captured.value.category == "history_execution_invalid"
    assert str(tmp_path) not in str(captured.value)


def test_source_invocations_preserve_validated_gpu_policy_existing_order(
    tmp_path: Path,
) -> None:
    """Catches sorting or guessing GPU order inside the source adapter."""
    policy = _gpu_policy()
    policy["indices"] = [107, 103, 101]
    policy["uuid_by_index"] = {
        str(index): f"GPU-synthetic-{index}" for index in policy["indices"]
    }

    invocations = build_source_invocations(
        tmp_path / "request.json",
        (
            "pyramid|29x53x101",
            "pyramid|17x31x63",
            "pyramid|23x47x95",
        ),
        source_materializer=_executable(tmp_path / "source-materializer"),
        validated_gpu_policy=policy,
    )

    assert [argv[6] for argv in invocations] == [
        "pyramid|17x31x63",
        "pyramid|23x47x95",
        "pyramid|29x53x101",
    ]
    assert [argv[8] for argv in invocations] == ["107", "103", "101"]


def test_source_runner_validates_complete_plan_before_direct_argv_execution(
    tmp_path: Path,
) -> None:
    """Catches shell execution, mutable env reuse, or a duplicate partial invocation."""
    materializer = _executable(tmp_path / "source-materializer")
    request_path = tmp_path / "request.json"
    invocations = build_source_invocations(
        request_path,
        ("pyramid|17x31x63", "pyramid|23x47x95"),
        source_materializer=materializer,
        validated_gpu_policy=_gpu_policy(),
    )
    duplicate_plan = (*invocations, invocations[0])
    runner = _RecordingRunner()

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        run_source_invocations(
            duplicate_plan,
            runner=runner,
            cwd=tmp_path,
            env={"SYNTHETIC_MODE": "offline"},
        )

    assert captured.value.category == "history_execution_invalid"
    assert runner.calls == []

    environment = {"SYNTHETIC_MODE": "offline"}
    run_source_invocations(
        invocations,
        runner=runner,
        cwd=tmp_path,
        env=environment,
    )
    environment["SYNTHETIC_MODE"] = "mutated"
    assert [call[0] for call in runner.calls] == list(invocations)
    assert all(call[1] == tmp_path for call in runner.calls)
    assert all(call[2] == {"SYNTHETIC_MODE": "offline"} for call in runner.calls)
    assert all(call[3] is False for call in runner.calls)


@pytest.mark.parametrize("mode", ["nonzero", "exception"])
def test_source_runner_failure_is_stable_and_redacted(
    tmp_path: Path, mode: str
) -> None:
    """Catches source failures leaking private runner or path details."""
    invocations = build_source_invocations(
        tmp_path / "request.json",
        ("pyramid|17x31x63",),
        source_materializer=_executable(tmp_path / "source-materializer"),
        validated_gpu_policy=_gpu_policy(),
    )
    runner = _RecordingRunner(returncode=17, raises=mode == "exception")

    with pytest.raises(P6HistorySourceMaterializationError) as captured:
        run_source_invocations(
            invocations,
            runner=runner,
            cwd=tmp_path,
            env={"PRIVATE_PATH": str(tmp_path)},
        )

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
    assert str(tmp_path) not in str(captured.value)
