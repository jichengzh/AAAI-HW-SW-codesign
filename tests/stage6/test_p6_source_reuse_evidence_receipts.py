from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from framework.stage6.p6_source_reuse_evidence_v1 import (
    P6GroupSourceReceipt,
    P6SourceReuseEvidenceError,
    SOURCE_ARTIFACT_KEYS,
    SOURCE_MARKER_KEYS,
    canonical_json_sha256,
    classify_selected_group_sources,
    create_fresh_run_context,
    first_use_group_ids,
    plan_source_reuse_paths,
    receipt_path_for_group,
    require_selected_groups_ready_current_run,
    validate_and_publish_group_receipt,
)


GROUP_ID = "pyramid|16x32x64"
TASK_ID = "P6-H800-PYRAMID"
TASK_SHA = hashlib.sha256(b"task").hexdigest()
DIRECTORY_KEYS = ("checkpoint_dir", "calibration_root", "trt_calibration_dir")


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


def _contract(root: Path) -> dict[str, Any]:
    group_root = root / "materialized" / "pyramid-16-32-64"
    evidence = hashlib.sha256(b"source-evidence").hexdigest()
    return {
        "schema_version": "stage5_source_contract_v1",
        "group_id": GROUP_ID,
        "model": "pyramid",
        "width": [16, 32, 64],
        "artifact_id": "pyramid-16-32-64",
        "source_status": "materializable",
        "source_evidence_sha256": evidence,
        "materialization_scope": "p6_history_selected_candidate",
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


def _request(root: Path, *, round_index: int = 0) -> dict[str, Any]:
    contract = _contract(root)
    row_id = f"{GROUP_ID}|q=fp16|profile=h800-tvm-auto"
    row = {
        "row_id": row_id,
        "group_id": GROUP_ID,
        "q_mode": "fp16",
        "source_contract": contract,
        "source_contract_sha256": canonical_json_sha256(contract),
        "source_evidence_sha256": contract["source_evidence_sha256"],
    }
    body = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": TASK_ID,
        "task_sha256": TASK_SHA,
        "round_index": round_index,
        "row_sha256": {row_id: canonical_json_sha256(row)},
        "rows": [row],
    }
    return {**body, "measurement_request_sha256": canonical_json_sha256(body)}


@dataclass(frozen=True)
class Fixture:
    root: Path
    private_root: Path
    request: dict[str, Any]
    run_context: Any
    interface: dict[str, Any]

    @property
    def classification_kwargs(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "run_context": self.run_context,
            "local_output_root": self.root,
            "interface": self.interface,
            "private_root": self.private_root,
        }

    @property
    def receipt_kwargs(self) -> dict[str, Any]:
        return {
            "run_context": self.run_context,
            "local_output_root": self.root,
            "interface": self.interface,
            "private_root": self.private_root,
        }


def _fresh_fixture(tmp_path: Path, *, round_index: int = 0) -> Fixture:
    root = tmp_path / "local-output"
    private_root = tmp_path / "private"
    root.mkdir(parents=True)
    private_root.mkdir(parents=True)
    request = _request(root, round_index=round_index)
    plan = {
        "schema_version": "p6_pyramid_candidate_plan_v2",
        "candidates": [{"group_id": GROUP_ID, "q_mode": "fp16"}],
    }
    registry = {
        "schema_version": "stage5_candidate_source_registry_v2",
        "groups": [
            {
                "group_id": GROUP_ID,
                "source_contract": request["rows"][0]["source_contract"],
            }
        ],
    }
    _write_json(root / "pyramid_candidate_plan.json", plan)
    _write_json(root / "source_registry.json", registry)
    context = create_fresh_run_context(
        local_output_root=root,
        task_contract={"task_id": TASK_ID, "task_sha256": TASK_SHA},
        code_revision="0123456789abcdef",
        candidate_plan=plan,
        source_registry=registry,
    )
    interface = {"output_layout": {"round_root_template": "private-runs/{round_id}"}}
    request_path = (
        private_root / f"private-runs/{round_index}/measurement-request.json"
    )
    _write_json(request_path, request)
    os.utime(request_path, ns=(1_000_000_000, 1_000_000_000))
    return Fixture(root, private_root, request, context, interface)


def _write_complete_bundle(fixture: Fixture, *, reverse_tree_order: bool = False) -> None:
    contract = fixture.request["rows"][0]["source_contract"]
    for key in SOURCE_ARTIFACT_KEYS:
        path = Path(contract[key])
        if key in DIRECTORY_KEYS:
            path.mkdir(parents=True)
            relative_files = ("z.bin", "nested/a.bin")
            if reverse_tree_order:
                relative_files = tuple(reversed(relative_files))
            for relative in relative_files:
                leaf = path / relative
                leaf.parent.mkdir(parents=True, exist_ok=True)
                with leaf.open("xb") as handle:
                    handle.write(relative.encode("utf-8"))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(key.encode("utf-8"))
    training = Path(contract["training_done_marker"])
    source = Path(contract["source_done_marker"])
    with training.open("xb") as handle:
        handle.write(b"training-complete")
    with source.open("xb") as handle:
        handle.write(b"source-complete")
    os.utime(training, ns=(2_000_000_000, 2_000_000_000))
    os.utime(source, ns=(3_000_000_000, 3_000_000_000))


def _publish(fixture: Fixture) -> P6GroupSourceReceipt:
    return validate_and_publish_group_receipt(
        fixture.request,
        group_id=GROUP_ID,
        **fixture.receipt_kwargs,
    )


def _receipt_path(fixture: Fixture) -> Path:
    return receipt_path_for_group(plan_source_reuse_paths(fixture.root), GROUP_ID)


def _rewrite_receipt(fixture: Fixture, mutation: Any) -> None:
    path = _receipt_path(fixture)
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutation(raw)
    if raw.get("receipt_sha256") != "malformed":
        raw["receipt_sha256"] = canonical_json_sha256(
            {key: value for key, value in raw.items() if key != "receipt_sha256"}
        )
    _write_json(path, raw)


def _rehash_request(request: dict[str, Any]) -> None:
    request["row_sha256"] = {
        row["row_id"]: canonical_json_sha256(row) for row in request["rows"]
    }
    request["measurement_request_sha256"] = canonical_json_sha256(
        {key: value for key, value in request.items() if key != "measurement_request_sha256"}
    )


def test_first_use_publishes_adapter_receipt_and_reclassifies_ready(
    tmp_path: Path,
) -> None:
    fixture = _fresh_fixture(tmp_path)
    before = classify_selected_group_sources(**fixture.classification_kwargs)
    assert [(item.group_id, item.state) for item in before] == [(GROUP_ID, "UNSEEN")]

    _write_complete_bundle(fixture)
    receipt = _publish(fixture)

    assert receipt.producer_row_id == min(row["row_id"] for row in fixture.request["rows"])
    assert (
        receipt.producer_measurement_request_sha256
        == fixture.request["measurement_request_sha256"]
    )
    assert first_use_group_ids(
        classify_selected_group_sources(**fixture.classification_kwargs)
    ) == ()
    assert require_selected_groups_ready_current_run(
        **fixture.classification_kwargs
    ) == (receipt,)


def test_directory_tree_digest_is_creation_order_independent_and_exact(
    tmp_path: Path,
) -> None:
    first = _fresh_fixture(tmp_path / "first")
    second = _fresh_fixture(tmp_path / "second")
    _write_complete_bundle(first)
    _write_complete_bundle(second, reverse_tree_order=True)

    first_receipt = _publish(first)
    second_receipt = _publish(second)
    first_digest = dict(first_receipt.artifact_digests)["checkpoint_dir"]
    second_digest = dict(second_receipt.artifact_digests)["checkpoint_dir"]
    entries = [
        {"kind": "directory", "path": "nested"},
        {
            "kind": "regular_file",
            "path": "nested/a.bin",
            "sha256": hashlib.sha256(b"nested/a.bin").hexdigest(),
            "size": len(b"nested/a.bin"),
        },
        {
            "kind": "regular_file",
            "path": "z.bin",
            "sha256": hashlib.sha256(b"z.bin").hexdigest(),
            "size": len(b"z.bin"),
        },
    ]
    assert first_digest == second_digest
    assert first_digest.kind == "directory_tree"
    assert first_digest.sha256 == canonical_json_sha256(entries)


@pytest.mark.parametrize(
    ("mutation", "expected_state"),
    (
        ("bare_training_and_source_markers", "INVALID_PARTIAL"),
        ("one_marker_only", "INVALID_PARTIAL"),
        ("artifact_without_receipt", "INVALID_PARTIAL"),
        ("receipt_only", "INVALID_PARTIAL"),
        ("receipt_missing_one_artifact", "INVALID_PARTIAL"),
        ("cross_run_receipt_copy", "INVALID_STALE"),
        ("different_run_nonce", "INVALID_STALE"),
        ("different_task", "INVALID_STALE"),
        ("different_revision", "INVALID_STALE"),
        ("different_root_fingerprint", "INVALID_STALE"),
        ("different_plan_hash", "INVALID_STALE"),
        ("different_registry_hash", "INVALID_STALE"),
        ("unknown_receipt_key", "INVALID_MISMATCH"),
        ("malformed_receipt_hash", "INVALID_MISMATCH"),
        ("wrong_group_key", "INVALID_MISMATCH"),
        ("artifact_byte_tamper", "INVALID_MISMATCH"),
        ("marker_byte_tamper", "INVALID_MISMATCH"),
        ("producer_request_tamper", "INVALID_MISMATCH"),
        ("producer_row_hash_tamper", "INVALID_MISMATCH"),
        ("consumer_contract_drift", "INVALID_MISMATCH"),
        ("consumer_evidence_drift", "INVALID_MISMATCH"),
        ("artifact_symlink", "INVALID_MISMATCH"),
        ("artifact_hard_link", "INVALID_MISMATCH"),
        ("directory_contains_symlink", "INVALID_MISMATCH"),
        ("directory_contains_hard_link", "INVALID_MISMATCH"),
        ("directory_contains_fifo", "INVALID_MISMATCH"),
        ("wrong_leaf_type", "INVALID_MISMATCH"),
        ("lexical_path_escape", "INVALID_MISMATCH"),
        ("resolved_parent_escape", "INVALID_MISMATCH"),
    ),
)
def test_group_classification_is_exact_and_fail_closed(
    tmp_path: Path, mutation: str, expected_state: str
) -> None:
    fixture = _fresh_fixture(tmp_path)
    contract = fixture.request["rows"][0]["source_contract"]
    receipt = _receipt_path(fixture)
    if mutation == "bare_training_and_source_markers":
        for key in SOURCE_MARKER_KEYS:
            path = Path(contract[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"marker")
    elif mutation == "one_marker_only":
        path = Path(contract["training_done_marker"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"marker")
    elif mutation == "artifact_without_receipt":
        path = Path(contract["checkpoint_path"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"artifact")
    elif mutation == "receipt_only":
        receipt.write_text("{}", encoding="utf-8")
    else:
        _write_complete_bundle(fixture)
        _publish(fixture)
        if mutation == "receipt_missing_one_artifact":
            Path(contract["checkpoint_path"]).unlink()
        elif mutation in {
            "cross_run_receipt_copy",
            "different_revision",
            "different_root_fingerprint",
            "different_plan_hash",
            "different_registry_hash",
        }:
            _rewrite_receipt(
                fixture, lambda raw: raw.__setitem__("run_context_sha256", "1" * 64)
            )
        elif mutation == "different_run_nonce":
            _rewrite_receipt(fixture, lambda raw: raw.__setitem__("run_nonce", "1" * 64))
        elif mutation == "different_task":
            _rewrite_receipt(fixture, lambda raw: raw.__setitem__("task_sha256", "1" * 64))
        elif mutation == "unknown_receipt_key":
            _rewrite_receipt(fixture, lambda raw: raw.__setitem__("unknown", True))
        elif mutation == "malformed_receipt_hash":
            _rewrite_receipt(
                fixture, lambda raw: raw.__setitem__("receipt_sha256", "malformed")
            )
        elif mutation == "wrong_group_key":
            _rewrite_receipt(
                fixture, lambda raw: raw.__setitem__("group_key_sha256", "1" * 64)
            )
        elif mutation == "artifact_byte_tamper":
            Path(contract["checkpoint_path"]).write_bytes(b"tampered")
        elif mutation == "marker_byte_tamper":
            Path(contract["source_done_marker"]).write_bytes(b"tampered")
        elif mutation == "producer_request_tamper":
            producer = fixture.private_root / "private-runs/0/measurement-request.json"
            producer.write_text("{}\n", encoding="utf-8")
        elif mutation == "producer_row_hash_tamper":
            _rewrite_receipt(
                fixture, lambda raw: raw.__setitem__("producer_row_sha256", "1" * 64)
            )
        elif mutation in {"consumer_contract_drift", "consumer_evidence_drift"}:
            row = fixture.request["rows"][0]
            key = (
                "source_contract_sha256"
                if mutation == "consumer_contract_drift"
                else "source_evidence_sha256"
            )
            row[key] = "1" * 64
            _rehash_request(fixture.request)
        elif mutation == "artifact_symlink":
            path = Path(contract["checkpoint_path"])
            path.unlink()
            path.symlink_to(Path(contract["config_path"]))
        elif mutation == "artifact_hard_link":
            path = Path(contract["checkpoint_path"])
            sibling = path.with_name("linked-checkpoint")
            os.link(path, sibling)
        elif mutation.startswith("directory_contains_"):
            directory = Path(contract["checkpoint_dir"])
            if mutation == "directory_contains_symlink":
                (directory / "unsafe").symlink_to(Path(contract["config_path"]))
            elif mutation == "directory_contains_hard_link":
                os.link(directory / "z.bin", directory / "hard-link")
            else:
                os.mkfifo(directory / "fifo")
        elif mutation == "wrong_leaf_type":
            path = Path(contract["checkpoint_path"])
            path.unlink()
            path.mkdir()
        elif mutation == "lexical_path_escape":
            fixture.request["rows"][0]["source_contract"]["checkpoint_path"] = str(
                fixture.root / "materialized/../outside"
            )
            fixture.request["rows"][0]["source_contract_sha256"] = canonical_json_sha256(
                fixture.request["rows"][0]["source_contract"]
            )
            _rehash_request(fixture.request)
        elif mutation == "resolved_parent_escape":
            parent = fixture.root / "materialized" / "escape-parent"
            parent.symlink_to(fixture.private_root, target_is_directory=True)
            fixture.request["rows"][0]["source_contract"]["checkpoint_path"] = str(
                parent / "outside"
            )
            fixture.request["rows"][0]["source_contract_sha256"] = canonical_json_sha256(
                fixture.request["rows"][0]["source_contract"]
            )
            _rehash_request(fixture.request)

    decision = classify_selected_group_sources(**fixture.classification_kwargs)[0]
    assert decision.state == expected_state
    if expected_state.startswith("INVALID"):
        with pytest.raises(P6SourceReuseEvidenceError) as exc_info:
            first_use_group_ids((decision,))
        assert str(exc_info.value) == "history_execution_invalid"
        assert expected_state not in str(exc_info.value)


def test_publication_refuses_and_preserves_wrapper_created_receipt(tmp_path: Path) -> None:
    fixture = _fresh_fixture(tmp_path)
    _write_complete_bundle(fixture)
    path = _receipt_path(fixture)
    wrapper_bytes = b'{"wrapper":"must-not-publish"}\n'
    path.write_bytes(wrapper_bytes)

    with pytest.raises(P6SourceReuseEvidenceError) as exc_info:
        _publish(fixture)

    assert str(exc_info.value) == "history_execution_invalid"
    assert path.read_bytes() == wrapper_bytes
