"""Fresh-run guards for the P6 controller."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pytest

from framework.stage5.single_target_search_v2 import validate_search_task
from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6 import p6_source_reuse_evidence_v1 as reuse_evidence
from framework.stage6.coptv2x_h800_search_v2 import (
    P6CoptV2XContractError,
    P6CoptV2XExecutionError,
    _prepare_round_output,
    load_public_contract,
    run_p6_coptv2x_search,
)
from framework.stage6.p6_source_reuse_evidence_v1 import (
    canonical_json_sha256,
    load_fresh_run_context,
    plan_source_reuse_paths,
    receipt_path_for_group,
)
from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
from tests.stage6.test_coptv2x_h800_search import (
    _FullChainCalls,
    _complete_framework_stage2_search_space,
    _framework_local_config_with_stage1_step,
    _real_stage1_partition_manifest,
    _write_yaml,
)


def _canonical_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _mutating_registry_runner(
    calls: _FullChainCalls, mutation: str
) -> Any:
    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        result = calls.runner(argv, cwd)
        if argv[0] != "fake-registry":
            return result
        registry_path = Path(argv[2])
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        first = registry["groups"][0]
        contract = first["source_contract"]
        if mutation == "mixed_recipe":
            for key in (
                "dynamic_materialization_recipe",
                "shared_source_paths",
                "training_required",
                "training_source_kind",
                "base_checkpoint_path",
                "dataset_root",
                "pyramid_config_path",
                "training_parameters",
            ):
                contract.pop(key, None)
            first["source_contract_sha256"] = _canonical_sha(contract)
        elif mutation == "incomplete_recipe":
            contract["shared_source_paths"].pop("calibration_npz")
            first["source_contract_sha256"] = _canonical_sha(contract)
        elif mutation == "plan_registry_drift":
            q_mode = first["available_q_modes"][0]
            first["source_point_ids_by_q_mode"][q_mode][0] = "drifted-source-point"
        else:  # pragma: no cover - test helper contract
            raise AssertionError(mutation)
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        return result

    return runner


def _context_task_contract(
    contract: execution.PublicP6CoptV2XContract,
    local: execution.LocalP6CoptV2XConfig,
) -> dict[str, Any]:
    _, _, _, profile = execution._load_search_inputs(local, contract)
    return validate_search_task(execution._build_search_task(contract, profile))


def _rtx_framework_local_config(tmp_path: Path) -> execution.LocalP6CoptV2XConfig:
    return _framework_local_config_with_stage1_step(
        tmp_path,
        contract_overrides={
            "schema_version": "p6_coptv2x_search_contract_v3",
            "hardware_profile": "rtx4090",
            "target": "rtx4090",
        },
        local_overrides={
            "schema_version": "p6_coptv2x_local_v3",
            "hardware_profile": "rtx4090",
            "target": "rtx4090",
        },
    )


def test_prepare_round_output_rejects_an_existing_round_root(tmp_path: Path) -> None:
    """A second controller invocation must not reuse a round's diagnostics."""
    (tmp_path / "round-00").mkdir()

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        _prepare_round_output(tmp_path, 0)

    assert captured.value.failure_code == "unsafe_output"


def test_controller_creates_context_after_registry_exactly_once_before_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _framework_local_config_with_stage1_step(tmp_path)
    contract = load_public_contract(tmp_path / "contract.yaml")
    calls = _FullChainCalls()
    events: list[str] = []
    real_create_context = execution.create_fresh_run_context

    def record_context(**kwargs: Any) -> Any:
        events.append("context")
        return real_create_context(**kwargs)

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        events.append(argv[0])
        return calls.runner(argv, cwd)

    monkeypatch.setattr(execution, "create_fresh_run_context", record_context)
    state = run_p6_coptv2x_search(
        contract, local, "rev-fresh-context", runner
    )

    assert state.status == "completed"
    assert events.count("context") == 1
    assert events.index("fake-stage1") < events.index("fake-registry")
    assert events.index("fake-registry") < events.index("context")
    assert events.index("context") < events.index("fake-measure")
    task_contract = _context_task_contract(contract, local)
    context = load_fresh_run_context(
        local_output_root=local.local_output_root,
        expected_task_id=task_contract["task_id"],
        expected_task_sha256=task_contract["task_sha256"],
    )
    plan = json.loads(
        (local.local_output_root / "pyramid_candidate_plan.json").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads(
        (local.local_output_root / "source_registry.json").read_text(encoding="utf-8")
    )
    assert context.code_revision == "rev-fresh-context"
    assert context.candidate_plan_sha256 == canonical_json_sha256(plan)
    assert context.source_registry_sha256 == canonical_json_sha256(registry)


def test_hardware_profile_stage2_target_mismatch_fails_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _rtx_framework_local_config(tmp_path)
    monkeypatch.setattr(
        execution,
        "load_stage2_search_space",
        lambda path: _complete_framework_stage2_search_space(),
    )
    runner_calls = 0

    def forbidden_runner(argv: tuple[str, ...], cwd: Path) -> int:
        nonlocal runner_calls
        runner_calls += 1
        raise AssertionError((argv, cwd))

    with pytest.raises(P6CoptV2XContractError, match="hardware profile"):
        execution._build_source_registry(local, forbidden_runner)

    assert runner_calls == 0


def test_hardware_profile_full_controller_rejects_stage2_target_mismatch_before_downstream_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _rtx_framework_local_config(tmp_path)
    contract = load_public_contract(tmp_path / "contract.yaml")
    monkeypatch.setattr(
        execution,
        "load_stage2_search_space",
        lambda path: _complete_framework_stage2_search_space(),
    )
    runner_calls: list[str] = []

    def runner(argv: tuple[str, ...], cwd: Path) -> int:
        del cwd
        runner_calls.append(argv[0])
        if argv[0] != "fake-stage1":
            raise AssertionError(argv)
        h800_manifest = _real_stage1_partition_manifest()
        manifest = {
            **h800_manifest,
            "hw_capability": {
                **h800_manifest["hw_capability"],
                "name": "rtx4090",
            },
        }
        _write_yaml(Path(argv[1]), manifest)
        return 0

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "rev-stage2-target", runner)

    assert captured.value.failure_code == "stage1_scan_invalid"
    assert runner_calls == ["fake-stage1"]


@pytest.mark.parametrize(
    "mutation", ["mixed_recipe", "incomplete_recipe", "plan_registry_drift"]
)
def test_controller_rejects_invalid_registry_before_context_or_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    local = _framework_local_config_with_stage1_step(tmp_path)
    contract = load_public_contract(tmp_path / "contract.yaml")
    calls = _FullChainCalls()
    context_calls = 0
    real_create_context = execution.create_fresh_run_context

    def record_context(**kwargs: Any) -> Any:
        nonlocal context_calls
        context_calls += 1
        return real_create_context(**kwargs)

    monkeypatch.setattr(execution, "create_fresh_run_context", record_context)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(
            contract,
            local,
            "rev-fresh-context",
            _mutating_registry_runner(calls, mutation),
        )

    assert captured.value.failure_code == "source_registry_invalid"
    assert context_calls == 0
    assert calls.measurement_count == 0


def _precreate_controller_destination(
    local: execution.LocalP6CoptV2XConfig, mutation: str
) -> None:
    output_root = local.local_output_root
    output_root.mkdir(parents=True)
    plan = execution.build_pyramid_candidate_plan(
        _complete_framework_stage2_search_space()
    )
    width = plan["candidates"][0]["width"]
    group_id = f"pyramid|{'x'.join(map(str, width))}"
    paths = {
        "stage1_manifest": local.stage2_search_space_path,
        "candidate_plan": output_root / "pyramid_candidate_plan.json",
        "source_registry": output_root / "source_registry.json",
        "state": output_root / "state.json",
        "public_round_root": output_root / "round-00",
        "metadata_namespace": plan_source_reuse_paths(output_root).metadata_root,
        "run_context": plan_source_reuse_paths(output_root).run_context,
        "receipt_namespace": plan_source_reuse_paths(output_root).receipt_root,
        "group_receipt": receipt_path_for_group(
            plan_source_reuse_paths(output_root), group_id
        ),
        "materialized_root": output_root / "materialized",
    }
    if mutation in SHARED_SOURCE_PATH_KEYS:
        slug = "-".join(map(str, width))
        path = output_root / "materialized" / slug / mutation
    else:
        path = paths[mutation]
    assert path is not None
    directories = {
        "public_round_root",
        "metadata_namespace",
        "receipt_namespace",
        "materialized_root",
        "checkpoint_dir",
        "calibration_root",
        "trt_calibration_dir",
    }
    if mutation in directories:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale controller evidence\n", encoding="utf-8")


@pytest.mark.parametrize(
    "mutation",
    [
        "stage1_manifest",
        "candidate_plan",
        "source_registry",
        "state",
        "public_round_root",
        "metadata_namespace",
        "run_context",
        "receipt_namespace",
        "group_receipt",
        "materialized_root",
        *SHARED_SOURCE_PATH_KEYS,
    ],
)
def test_controller_freshness_matrix_rejects_without_nonce_or_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    local = _framework_local_config_with_stage1_step(tmp_path)
    contract = load_public_contract(tmp_path / "contract.yaml")
    _precreate_controller_destination(local, mutation)
    calls = _FullChainCalls()
    nonce_calls = 0
    real_token_bytes = reuse_evidence.secrets.token_bytes

    def record_token_bytes(length: int) -> bytes:
        nonlocal nonce_calls
        nonce_calls += 1
        return real_token_bytes(length)

    monkeypatch.setattr(reuse_evidence.secrets, "token_bytes", record_token_bytes)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(contract, local, "rev-fresh-context", calls.runner)

    assert captured.value.failure_code in {
        "unsafe_output",
        "unsafe_destination",
        "history_execution_invalid",
    }
    assert nonce_calls == 0
    assert calls.measurement_count == 0


def test_controller_rejects_malformed_revision_before_context_or_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _framework_local_config_with_stage1_step(tmp_path)
    contract = load_public_contract(tmp_path / "contract.yaml")
    calls = _FullChainCalls()
    context_calls = 0

    def forbidden_context(**kwargs: Any) -> Any:
        nonlocal context_calls
        context_calls += 1
        raise AssertionError(kwargs)

    monkeypatch.setattr(execution, "create_fresh_run_context", forbidden_context)

    with pytest.raises(P6CoptV2XContractError):
        run_p6_coptv2x_search(contract, local, "", calls.runner)

    assert calls.names == []
    assert context_calls == 0


def test_failed_root_cannot_relaunch_in_place(tmp_path: Path) -> None:
    local = _framework_local_config_with_stage1_step(tmp_path)
    contract = load_public_contract(tmp_path / "contract.yaml")
    calls = _FullChainCalls()

    def fail_round_one(argv: tuple[str, ...], cwd: Path) -> int:
        if argv[0] == "fake-measure" and calls.measurement_count == 1:
            calls.names = [*calls.names, argv[0]]
            calls.measurement_count += 1
            return 1
        return calls.runner(argv, cwd)

    first = run_p6_coptv2x_search(
        contract, local, "rev-fresh-context", fail_round_one
    )
    assert first.status == "failed"
    assert first.completed_rounds == 1
    calls_before_relaunch = tuple(calls.names)

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        run_p6_coptv2x_search(
            contract, local, "rev-fresh-context", fail_round_one
        )

    assert captured.value.failure_code in {
        "history_execution_invalid",
        "unsafe_output",
        "unsafe_destination",
    }
    assert tuple(calls.names) == calls_before_relaunch
