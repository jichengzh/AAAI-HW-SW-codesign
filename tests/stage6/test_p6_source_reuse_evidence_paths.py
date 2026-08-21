from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

import pytest

from framework.stage6.p6_source_reuse_evidence_v1 import (
    GROUP_RECEIPT_RELATIVE_ROOT,
    RUN_CONTEXT_RELATIVE_PATH,
    RUN_METADATA_RELATIVE_ROOT,
    SOURCE_OUTPUT_RELATIVE_ROOT,
    P6SourceReuseEvidenceError,
    canonical_json_sha256,
    create_fresh_run_context,
    load_fresh_run_context,
    plan_source_reuse_paths,
    receipt_path_for_group,
    resolve_existing_source_reuse_paths,
)


GROUP_ID = "pyramid|16x32x64"
TASK_CONTRACT = {
    "task_id": "P6-H800-PYRAMID",
    "task_sha256": hashlib.sha256(b"task").hexdigest(),
}


def _canonical_write(path: Path, value: Any) -> None:
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


def _candidate_plan() -> dict[str, Any]:
    return {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "candidates": [
            {
                "candidate_id": "pyramid-16-32-64-fp16",
                "group_id": GROUP_ID,
                "q_mode": "fp16",
                "width": [16, 32, 64],
            }
        ],
    }


def _shared_paths(local_output_root: Path) -> dict[str, str]:
    group_root = local_output_root / SOURCE_OUTPUT_RELATIVE_ROOT / "pyramid-16-32-64"
    return {
        "checkpoint_path": str(group_root / "checkpoint.pt"),
        "checkpoint_dir": str(group_root / "checkpoint"),
        "config_path": str(group_root / "config.json"),
        "training_done_marker": str(group_root / "training.done"),
        "onnx_path": str(group_root / "model.onnx"),
        "onnx_report_path": str(group_root / "onnx-report.json"),
        "calibration_root": str(group_root / "calibration"),
        "calibration_npz": str(group_root / "calibration.npz"),
        "calibration_summary": str(group_root / "calibration-summary.json"),
        "trt_calibration_dir": str(group_root / "trt-calibration"),
        "source_done_marker": str(group_root / "source.done"),
    }


def _source_registry(local_output_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "stage5_candidate_source_registry_v2",
        "groups": [
            {
                "group_id": GROUP_ID,
                "source_contract": _shared_paths(local_output_root),
            }
        ],
    }


def _fresh_inputs(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    local_output_root = tmp_path / "local-output"
    local_output_root.mkdir()
    plan = _candidate_plan()
    registry = _source_registry(local_output_root)
    _canonical_write(local_output_root / "pyramid_candidate_plan.json", plan)
    _canonical_write(local_output_root / "source_registry.json", registry)
    return local_output_root, plan, registry


def _create(
    local_output_root: Path,
    plan: dict[str, Any],
    registry: dict[str, Any],
):
    return create_fresh_run_context(
        local_output_root=local_output_root,
        task_contract=TASK_CONTRACT,
        candidate_plan=plan,
        source_registry=registry,
        code_revision="0123456789abcdef",
    )


def test_paths_are_exact_and_receipts_are_group_hashed(tmp_path: Path) -> None:
    local_output_root, _, _ = _fresh_inputs(tmp_path)

    planned = plan_source_reuse_paths(local_output_root)

    assert planned.local_output_root == local_output_root
    assert planned.metadata_root == local_output_root / RUN_METADATA_RELATIVE_ROOT
    assert planned.run_context == local_output_root / RUN_CONTEXT_RELATIVE_PATH
    assert planned.receipt_root == local_output_root / GROUP_RECEIPT_RELATIVE_ROOT
    expected_name = hashlib.sha256(
        b"p6-group-receipt-v1\0" + GROUP_ID.encode("utf-8")
    ).hexdigest()
    assert receipt_path_for_group(planned, GROUP_ID) == (
        planned.receipt_root / f"{expected_name}.json"
    )


def test_context_creation_is_create_only_canonical_and_reloadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_output_root, plan, registry = _fresh_inputs(tmp_path)
    calls: list[int] = []

    def _token_bytes(size: int) -> bytes:
        calls.append(size)
        return bytes(range(size))

    monkeypatch.setattr(
        "framework.stage6.p6_source_reuse_evidence_v1.secrets.token_bytes",
        _token_bytes,
    )

    context = _create(local_output_root, plan, registry)
    paths = resolve_existing_source_reuse_paths(local_output_root)
    loaded = load_fresh_run_context(
        local_output_root=local_output_root,
        expected_task_id=TASK_CONTRACT["task_id"],
        expected_task_sha256=TASK_CONTRACT["task_sha256"],
    )

    assert calls == [32]
    assert loaded == context
    assert context.run_nonce == bytes(range(32)).hex()
    assert context.local_output_root_sha256 == hashlib.sha256(
        b"p6-local-output-root-v1\0" + str(local_output_root).encode("utf-8")
    ).hexdigest()
    assert context.candidate_plan_sha256 == canonical_json_sha256(plan)
    assert context.source_registry_sha256 == canonical_json_sha256(registry)
    assert stat.S_IMODE(paths.metadata_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.receipt_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.run_context.stat().st_mode) == 0o600
    serialized = json.loads(paths.run_context.read_text(encoding="utf-8"))
    assert serialized["run_context_sha256"] == context.run_context_sha256
    without_hash = {key: value for key, value in serialized.items() if key != "run_context_sha256"}
    assert context.run_context_sha256 == canonical_json_sha256(without_hash)


def test_existing_paths_require_context_and_receipt_namespace(tmp_path: Path) -> None:
    local_output_root, _, _ = _fresh_inputs(tmp_path)

    with pytest.raises(P6SourceReuseEvidenceError) as exc_info:
        resolve_existing_source_reuse_paths(local_output_root)

    assert str(exc_info.value) == "history_execution_invalid"
    assert exc_info.value.private_category == "p6_source_reuse_partial"


@pytest.mark.parametrize(
    ("public_category", "private_category"),
    (
        ("history_execution_invalid", "p6_source_reuse_partial"),
        ("history_execution_invalid", "p6_source_reuse_stale"),
        ("history_execution_invalid", "p6_source_reuse_mismatch"),
        ("unsafe_destination", None),
    ),
)
def test_evidence_error_exposes_exact_public_and_private_categories(
    public_category: str,
    private_category: str | None,
) -> None:
    error = P6SourceReuseEvidenceError(
        public_category=public_category,
        private_category=private_category,
    )

    assert str(error) == public_category
    assert error.public_category == public_category
    assert error.private_category == private_category


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    (
        ("schema_version", True),
        ("run_nonce", 1),
        ("task_id", True),
        ("task_sha256", 1),
        ("code_revision", 1),
        ("code_revision", True),
        ("code_revision", ""),
        ("local_output_root_sha256", 1),
        ("candidate_plan_schema_version", False),
        ("candidate_plan_sha256", 1),
        ("source_registry_schema_version", 1),
        ("source_registry_sha256", False),
        ("created_before_round_index", False),
        ("created_before_round_index", True),
        ("created_before_round_index", 1),
        ("created_before_round_index", 0.0),
    ),
)
def test_context_loader_rejects_rehashed_malformed_exact_field_types(
    tmp_path: Path,
    field: str,
    malformed_value: Any,
) -> None:
    local_output_root, plan, registry = _fresh_inputs(tmp_path)
    _create(local_output_root, plan, registry)
    paths = resolve_existing_source_reuse_paths(local_output_root)
    raw = json.loads(paths.run_context.read_text(encoding="utf-8"))
    raw[field] = malformed_value
    raw["run_context_sha256"] = canonical_json_sha256(
        {key: value for key, value in raw.items() if key != "run_context_sha256"}
    )
    _canonical_write(paths.run_context, raw)

    with pytest.raises(P6SourceReuseEvidenceError):
        load_fresh_run_context(
            local_output_root=local_output_root,
            expected_task_id=TASK_CONTRACT["task_id"],
            expected_task_sha256=TASK_CONTRACT["task_sha256"],
        )


@pytest.mark.parametrize(
    "malformed_hash",
    (True, 1, "", "A" * 64, "0" * 63),
)
def test_context_loader_rejects_malformed_context_hash_type_or_spelling(
    tmp_path: Path,
    malformed_hash: Any,
) -> None:
    local_output_root, plan, registry = _fresh_inputs(tmp_path)
    _create(local_output_root, plan, registry)
    paths = resolve_existing_source_reuse_paths(local_output_root)
    raw = json.loads(paths.run_context.read_text(encoding="utf-8"))
    raw["run_context_sha256"] = malformed_hash
    _canonical_write(paths.run_context, raw)

    with pytest.raises(P6SourceReuseEvidenceError):
        load_fresh_run_context(
            local_output_root=local_output_root,
            expected_task_id=TASK_CONTRACT["task_id"],
            expected_task_sha256=TASK_CONTRACT["task_sha256"],
        )


Mutation = Callable[[Path, dict[str, Any], dict[str, Any]], None]


def _mutate_relative_root(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan, registry
    os.chdir(root.parent)


def _mutate_noncanonical_root(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del root, plan, registry


def _mutate_missing_root(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan, registry
    for child in tuple(root.iterdir()):
        child.unlink()
    root.rmdir()


def _mutate_root_symlink_component(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan, registry
    target = root.parent / "real-parent"
    target.mkdir()
    moved = target / root.name
    root.rename(moved)
    root.parent.joinpath("root-link").symlink_to(target, target_is_directory=True)


def _mutate_root_not_directory(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan, registry
    for child in tuple(root.iterdir()):
        child.unlink()
    root.rmdir()
    root.write_text("not a directory", encoding="utf-8")


def _mutate_metadata_symlink(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan, registry
    target = root / "metadata-target"
    target.mkdir()
    (root / RUN_METADATA_RELATIVE_ROOT).symlink_to(target, target_is_directory=True)


def _mutate_context_preexists(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan, registry
    _canonical_write(root / RUN_CONTEXT_RELATIVE_PATH, {"unexpected": True})


def _mutate_receipt_root_preexists(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan, registry
    (root / GROUP_RECEIPT_RELATIVE_ROOT).mkdir(parents=True)


def _mutate_materialized_root_preexists(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan, registry
    (root / SOURCE_OUTPUT_RELATIVE_ROOT).mkdir()


def _mutate_declared_output_preexists(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan
    marker = Path(registry["groups"][0]["source_contract"]["training_done_marker"])
    marker.parent.mkdir(parents=True)
    marker.write_text("old", encoding="utf-8")


def _mutate_group_receipt_preexists(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del plan
    paths = plan_source_reuse_paths(root)
    paths.receipt_root.mkdir(parents=True)
    receipt_path_for_group(paths, registry["groups"][0]["group_id"]).write_text(
        "{}", encoding="utf-8"
    )


def _mutate_plan_file_mismatch(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del registry
    plan["candidates"][0]["q_mode"] = "int8"


def _mutate_registry_file_mismatch(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del root, plan
    registry["groups"][0]["group_id"] = "pyramid|32x64x128"


def _mutate_nothing(
    root: Path, plan: dict[str, Any], registry: dict[str, Any]
) -> None:
    del root, plan, registry


@pytest.mark.parametrize(
    ("mutation_name", "mutate"),
    (
        ("relative_root", _mutate_relative_root),
        ("noncanonical_root", _mutate_noncanonical_root),
        ("missing_root", _mutate_missing_root),
        ("root_symlink_component", _mutate_root_symlink_component),
        ("root_not_directory", _mutate_root_not_directory),
        ("metadata_symlink", _mutate_metadata_symlink),
        ("context_preexists", _mutate_context_preexists),
        ("receipt_root_preexists", _mutate_receipt_root_preexists),
        ("materialized_root_preexists", _mutate_materialized_root_preexists),
        ("one_declared_output_preexists", _mutate_declared_output_preexists),
        ("one_group_receipt_preexists", _mutate_group_receipt_preexists),
        ("plan_file_mismatch", _mutate_plan_file_mismatch),
        ("registry_file_mismatch", _mutate_registry_file_mismatch),
        ("duplicate_json_key", _mutate_nothing),
        ("unknown_context_key", _mutate_nothing),
        ("noncanonical_context_hash", _mutate_nothing),
        ("second_context_create", _mutate_nothing),
    ),
)
def test_context_boundary_mutations_fail_closed_without_private_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_name: str,
    mutate: Mutation,
) -> None:
    local_output_root, plan, registry = _fresh_inputs(tmp_path)
    original_cwd = Path.cwd()
    mutate(local_output_root, plan, registry)
    try:
        if mutation_name == "relative_root":
            def action() -> Any:
                return plan_source_reuse_paths(Path("local-output"))
        elif mutation_name == "noncanonical_root":
            def action() -> Any:
                return plan_source_reuse_paths(
                    local_output_root / ".." / local_output_root.name
                )
        elif mutation_name == "root_symlink_component":
            def action() -> Any:
                return plan_source_reuse_paths(
                    local_output_root.parent / "root-link" / local_output_root.name
                )
        elif mutation_name in {
            "duplicate_json_key",
            "unknown_context_key",
            "noncanonical_context_hash",
            "second_context_create",
        }:
            context = _create(local_output_root, plan, registry)
            paths = resolve_existing_source_reuse_paths(local_output_root)
            raw = json.loads(paths.run_context.read_text(encoding="utf-8"))
            if mutation_name == "duplicate_json_key":
                text = paths.run_context.read_text(encoding="utf-8").rstrip()
                paths.run_context.write_text(
                    text[:-1] + ',"task_id":"duplicate"}\n', encoding="utf-8"
                )
            elif mutation_name == "unknown_context_key":
                raw["unknown"] = True
                _canonical_write(paths.run_context, raw)
            elif mutation_name == "noncanonical_context_hash":
                raw["run_context_sha256"] = context.run_context_sha256.upper()
                _canonical_write(paths.run_context, raw)
            if mutation_name == "second_context_create":
                def action() -> Any:
                    return _create(local_output_root, plan, registry)
            else:
                def action() -> Any:
                    return load_fresh_run_context(
                    local_output_root=local_output_root,
                    expected_task_id=TASK_CONTRACT["task_id"],
                    expected_task_sha256=TASK_CONTRACT["task_sha256"],
                )
        else:
            def action() -> Any:
                return _create(local_output_root, plan, registry)

        with pytest.raises(P6SourceReuseEvidenceError) as exc_info:
            action()
    finally:
        monkeypatch.chdir(original_cwd)

    unsafe_root_mutations = {
        "relative_root", "noncanonical_root", "missing_root",
        "root_symlink_component", "root_not_directory",
    }
    expected_public = (
        "unsafe_destination"
        if mutation_name in unsafe_root_mutations
        else "history_execution_invalid"
    )
    assert str(exc_info.value) == expected_public
    assert exc_info.value.public_category == expected_public
    if expected_public == "unsafe_destination":
        assert exc_info.value.private_category is None
    assert str(local_output_root) not in str(exc_info.value)
