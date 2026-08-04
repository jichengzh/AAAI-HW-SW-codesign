"""Black-box tests for the private P3 disposition ledger builder."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "tools/release/build_p3_disposition_ledger.py"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_key(path: Path, value: bytes = b"k" * 32) -> None:
    path.write_bytes(value)
    path.chmod(0o600)


def _inventory(private_marker: str) -> dict[str, object]:
    return {
        "format": 1,
        "tool": "private-source-inventory/1.1.0",
        "candidates": [
            {
                "path": f"private/{private_marker}/method.py",
                "origin": "tracked",
                "classification": "migrate_code",
                "bytes": 17,
            },
            {
                "path": f"private/{private_marker}/fixture.json",
                "origin": "tracked",
                "classification": "fixture_candidate",
                "bytes": 3,
            },
            {
                "path": f"private/{private_marker}/parameters.yaml",
                "origin": "untracked",
                "classification": "migrate_config",
                "bytes": 9,
            },
        ],
    }


def _decisions(private_marker: str) -> dict[str, object]:
    return {
        "format": 1,
        "decisions": [
            {
                "path": f"private/{private_marker}/method.py",
                "origin": "tracked",
                "disposition": "migrated_public",
                "reason": "pure_memory_contract",
                "batch": "P3-1",
                "evidence": "stage6_contract_tests",
                "next_phase": None,
            },
            {
                "path": f"private/{private_marker}/fixture.json",
                "origin": "tracked",
                "disposition": "external_contract_p4",
                "reason": "requires_licensed_input",
                "batch": "P3-2",
                "evidence": "input_boundary_review",
                "next_phase": "P4",
            },
            {
                "path": f"private/{private_marker}/parameters.yaml",
                "origin": "untracked",
                "disposition": "blocked_license_or_permission",
                "reason": "third_party_permission_pending",
                "batch": "P3-3",
                "evidence": "license_review",
                "next_phase": "P7",
            },
        ],
    }


def _run(inventory: Path, decisions: Path, key: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--inventory",
            str(inventory),
            "--decisions",
            str(decisions),
            "--hmac-key",
            str(key),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_builder_emits_hmac_ids_and_only_public_disposition_fields(tmp_path: Path) -> None:
    private_marker = "private-lab-marker"
    inventory = tmp_path / "inventory.json"
    decisions = tmp_path / "decisions.json"
    key = tmp_path / "ledger.key"
    output = tmp_path / "public-ledger.json"
    _write_json(inventory, _inventory(private_marker))
    _write_json(decisions, _decisions(private_marker))
    key_value = b"H" * 32
    _write_key(key, key_value)

    result = _run(inventory, decisions, key, output)

    assert result.returncode == 0, result.stderr
    ledger = json.loads(output.read_text(encoding="utf-8"))
    assert set(ledger) == {"counts", "entries", "format", "inventory_sha256", "tool"}
    assert ledger["counts"] == {
        "by_classification": [
            {"classification": "fixture_candidate", "count": 1},
            {"classification": "migrate_code", "count": 1},
            {"classification": "migrate_config", "count": 1},
        ],
        "by_classification_and_disposition": [
            {
                "classification": "fixture_candidate",
                "count": 1,
                "disposition": "external_contract_p4",
            },
            {
                "classification": "migrate_code",
                "count": 1,
                "disposition": "migrated_public",
            },
            {
                "classification": "migrate_config",
                "count": 1,
                "disposition": "blocked_license_or_permission",
            },
        ],
        "by_disposition": [
            {"count": 1, "disposition": "blocked_license_or_permission"},
            {"count": 1, "disposition": "external_contract_p4"},
            {"count": 1, "disposition": "migrated_public"},
        ],
        "total_candidates": 3,
    }
    assert {
        "candidate_id",
        "classification",
        "disposition",
        "reason",
        "batch",
        "evidence",
        "next_phase",
    } == set(ledger["entries"][0])
    expected_id = hmac.new(
        key_value,
        b"\0".join(
            (
                b"p3-disposition-ledger-v1",
                b"tracked",
                f"private/{private_marker}/method.py".encode(),
            )
        ),
        hashlib.sha256,
    ).hexdigest()
    assert expected_id in {entry["candidate_id"] for entry in ledger["entries"]}
    rendered = output.read_text(encoding="utf-8")
    assert private_marker not in rendered
    assert "private/" not in rendered
    assert key_value.decode("ascii") not in rendered


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("decisions", payload["decisions"][:-1]),
        lambda payload: payload["decisions"].append(payload["decisions"][0].copy()),
        lambda payload: payload["decisions"][0].__setitem__("disposition", "pending"),
        lambda payload: payload["decisions"][0].__setitem__("reason", "contains/a/path"),
    ],
    ids=("missing", "duplicate", "pending", "unsafe-reason"),
)
def test_builder_fails_closed_for_incomplete_or_unsafe_decisions(
    tmp_path: Path, mutate
) -> None:
    private_marker = "never-echo-this-private-marker"
    inventory = tmp_path / "inventory.json"
    decisions = tmp_path / "decisions.json"
    key = tmp_path / "ledger.key"
    output = tmp_path / "public-ledger.json"
    decision_payload = _decisions(private_marker)
    mutate(decision_payload)
    _write_json(inventory, _inventory(private_marker))
    _write_json(decisions, decision_payload)
    _write_key(key)

    result = _run(inventory, decisions, key, output)

    assert result.returncode == 2
    assert not output.exists()
    assert private_marker not in result.stderr
    assert "private/" not in result.stderr


@pytest.mark.parametrize("unsafe_target", ("inventory", "decisions", "key", "output"))
def test_builder_rejects_symbolic_links_without_echoing_private_paths(
    tmp_path: Path, unsafe_target: str
) -> None:
    private_marker = "private-symlink-marker"
    inventory = tmp_path / "inventory.json"
    decisions = tmp_path / "decisions.json"
    key = tmp_path / "ledger.key"
    output = tmp_path / "public-ledger.json"
    _write_json(inventory, _inventory(private_marker))
    _write_json(decisions, _decisions(private_marker))
    _write_key(key)

    targets = {"inventory": inventory, "decisions": decisions, "key": key, "output": output}
    unsafe = targets[unsafe_target]
    backing = tmp_path / f"{unsafe_target}-backing"
    if unsafe_target == "key":
        _write_key(backing)
    else:
        _write_json(backing, {"private": private_marker})
    if unsafe.exists() or unsafe.is_symlink():
        unsafe.unlink()
    unsafe.symlink_to(backing)

    result = _run(inventory, decisions, key, output)

    assert result.returncode == 2
    assert private_marker not in result.stderr
    assert str(backing) not in result.stderr
    if unsafe_target == "output":
        assert json.loads(backing.read_text(encoding="utf-8")) == {"private": private_marker}


def test_builder_requires_a_private_non_group_readable_hmac_key(tmp_path: Path) -> None:
    private_marker = "private-key-marker"
    inventory = tmp_path / "inventory.json"
    decisions = tmp_path / "decisions.json"
    key = tmp_path / "ledger.key"
    output = tmp_path / "public-ledger.json"
    _write_json(inventory, _inventory(private_marker))
    _write_json(decisions, _decisions(private_marker))
    _write_key(key)
    key.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)

    result = _run(inventory, decisions, key, output)

    assert result.returncode == 2
    assert not output.exists()
    assert private_marker not in result.stderr


def test_builder_rejects_hard_linked_input_and_does_not_create_output(tmp_path: Path) -> None:
    private_marker = "private-hard-link-marker"
    inventory = tmp_path / "inventory.json"
    inventory_backing = tmp_path / "inventory-backing.json"
    decisions = tmp_path / "decisions.json"
    key = tmp_path / "ledger.key"
    output = tmp_path / "public-ledger.json"
    _write_json(inventory_backing, _inventory(private_marker))
    os.link(inventory_backing, inventory)
    _write_json(decisions, _decisions(private_marker))
    _write_key(key)

    result = _run(inventory, decisions, key, output)

    assert result.returncode == 2
    assert not output.exists()
    assert private_marker not in result.stderr


def test_builder_rejects_a_symbolic_link_in_an_input_parent(tmp_path: Path) -> None:
    private_marker = "private-parent-link-marker"
    private_directory = tmp_path / "private-directory"
    private_directory.mkdir()
    inventory_backing = private_directory / "inventory.json"
    inventory_parent = tmp_path / "inventory-parent"
    decisions = tmp_path / "decisions.json"
    key = tmp_path / "ledger.key"
    output = tmp_path / "public-ledger.json"
    _write_json(inventory_backing, _inventory(private_marker))
    inventory_parent.symlink_to(private_directory, target_is_directory=True)
    _write_json(decisions, _decisions(private_marker))
    _write_key(key)

    result = _run(inventory_parent / "inventory.json", decisions, key, output)

    assert result.returncode == 2
    assert not output.exists()
    assert private_marker not in result.stderr


def test_builder_rejects_a_symbolic_link_in_the_output_parent(tmp_path: Path) -> None:
    private_marker = "private-output-parent-marker"
    inventory = tmp_path / "inventory.json"
    decisions = tmp_path / "decisions.json"
    key = tmp_path / "ledger.key"
    output_backing = tmp_path / "output-backing"
    output_parent = tmp_path / "output-parent"
    _write_json(inventory, _inventory(private_marker))
    _write_json(decisions, _decisions(private_marker))
    _write_key(key)
    output_backing.mkdir()
    output_parent.symlink_to(output_backing, target_is_directory=True)

    result = _run(inventory, decisions, key, output_parent / "public-ledger.json")

    assert result.returncode == 2
    assert not (output_backing / "public-ledger.json").exists()
    assert private_marker not in result.stderr
