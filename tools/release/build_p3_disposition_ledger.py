#!/usr/bin/env python3
"""Build a public, path-free P3 disposition ledger from controlled local inputs.

The inventory and decision files are deliberately local-only because they contain
private relative paths.  The generated ledger contains only HMAC-SHA256 candidate
identifiers, P2 classifications, disposition codes, and constrained audit metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


FORMAT_VERSION = 1
TOOL_VERSION = "1.0.0"
MAX_JSON_BYTES = 10 * 1024 * 1024
MIN_HMAC_KEY_BYTES = 32
MAX_HMAC_KEY_BYTES = 4096
CANDIDATE_CLASSES = frozenset(
    {"fixture_candidate", "migrate_code", "migrate_config", "migrate_document"}
)
ORIGINS = frozenset({"tracked", "untracked"})
DISPOSITIONS = frozenset(
    {
        "migrated_public",
        "rewritten_public",
        "external_contract_p4",
        "environment_contract_p5",
        "execution_contract_p6",
        "duplicate_or_superseded",
        "excluded_nonessential",
        "blocked_license_or_permission",
    }
)
NEXT_PHASE_BY_DISPOSITION: Mapping[str, str | None] = {
    "migrated_public": None,
    "rewritten_public": None,
    "external_contract_p4": "P4",
    "environment_contract_p5": "P5",
    "execution_contract_p6": "P6",
    "duplicate_or_superseded": None,
    "excluded_nonessential": None,
    "blocked_license_or_permission": "P7",
}
SAFE_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,79}")
BATCH = re.compile(r"P3(?:-[1-9][0-9]*)?")


class LedgerSafetyError(Exception):
    """A safe-to-display P3 ledger construction failure."""


@dataclass(frozen=True)
class Candidate:
    """Private candidate metadata retained only while building one ledger."""

    path: str
    origin: str
    classification: str


@dataclass(frozen=True)
class Decision:
    """Validated local disposition metadata safe to include in the public ledger."""

    disposition: str
    reason: str
    batch: str
    evidence: str
    next_phase: str | None


def _absolute_lexical_path(raw_path: Path) -> Path:
    """Return an absolute lexical path without resolving or accepting symlinks."""
    try:
        expanded = raw_path.expanduser()
    except RuntimeError as error:
        raise LedgerSafetyError("unable to access local input") from error
    return Path(os.path.abspath(os.fspath(expanded)))


def _open_directory(path: Path) -> int:
    """Open an absolute directory component-by-component without following links."""
    if not path.is_absolute() or path == path.parent:
        raise LedgerSafetyError("unsafe local path")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path.anchor, flags)
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise LedgerSafetyError("unsafe local path") from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _stable_metadata(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_file(path: Path, purpose: str, max_bytes: int, private: bool = False) -> bytes:
    """Read one pinned regular file without allowing links or unbounded input."""
    absolute = _absolute_lexical_path(path)
    parent_descriptor: int | None = None
    file_descriptor: int | None = None
    try:
        parent_descriptor = _open_directory(absolute.parent)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
        before = os.fstat(file_descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise LedgerSafetyError(f"unsafe {purpose}")
        if private and mode & 0o077:
            raise LedgerSafetyError("unsafe hmac key")
        if before.st_size > max_bytes:
            raise LedgerSafetyError(f"unsafe {purpose}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(file_descriptor, 1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise LedgerSafetyError(f"unsafe {purpose}")
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
    except LedgerSafetyError:
        raise
    except OSError as error:
        raise LedgerSafetyError(f"unable to read {purpose}") from error
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    if _stable_metadata(before) != _stable_metadata(after):
        raise LedgerSafetyError(f"unsafe {purpose}")
    return b"".join(chunks)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _load_json(path: Path, purpose: str) -> tuple[dict[str, object], bytes]:
    payload = _read_regular_file(path, purpose, MAX_JSON_BYTES)
    try:
        parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise LedgerSafetyError(f"invalid {purpose}") from error
    if not isinstance(parsed, dict):
        raise LedgerSafetyError(f"invalid {purpose}")
    return parsed, payload


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise LedgerSafetyError("invalid local inventory")
    if value.startswith("/") or any(ord(character) < 32 for character in value):
        raise LedgerSafetyError("invalid local inventory")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise LedgerSafetyError("invalid local inventory")
    return value


def _candidate_key(path: str, origin: str) -> tuple[str, str]:
    return origin, path


def _parse_inventory(inventory: Mapping[str, object]) -> dict[tuple[str, str], Candidate]:
    if inventory.get("format") != FORMAT_VERSION:
        raise LedgerSafetyError("invalid local inventory")
    records = inventory.get("candidates")
    if not isinstance(records, list):
        raise LedgerSafetyError("invalid local inventory")
    candidates: dict[tuple[str, str], Candidate] = {}
    for record in records:
        if not isinstance(record, dict):
            raise LedgerSafetyError("invalid local inventory")
        path = _safe_relative_path(record.get("path"))
        origin = record.get("origin")
        classification = record.get("classification")
        byte_count = record.get("bytes")
        if (
            not isinstance(origin, str)
            or origin not in ORIGINS
            or not isinstance(classification, str)
            or classification not in CANDIDATE_CLASSES
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise LedgerSafetyError("invalid local inventory")
        key = _candidate_key(path, origin)
        if key in candidates:
            raise LedgerSafetyError("invalid local inventory")
        candidates[key] = Candidate(path=path, origin=origin, classification=classification)
    if not candidates:
        raise LedgerSafetyError("invalid local inventory")
    return candidates


def _safe_token(value: object) -> str:
    if not isinstance(value, str) or not SAFE_TOKEN.fullmatch(value):
        raise LedgerSafetyError("invalid local decisions")
    return value


def _parse_decisions(decision_file: Mapping[str, object]) -> dict[tuple[str, str], Decision]:
    if set(decision_file) != {"format", "decisions"} or decision_file.get("format") != FORMAT_VERSION:
        raise LedgerSafetyError("invalid local decisions")
    records = decision_file.get("decisions")
    if not isinstance(records, list):
        raise LedgerSafetyError("invalid local decisions")
    decisions: dict[tuple[str, str], Decision] = {}
    required_fields = {
        "path",
        "origin",
        "disposition",
        "reason",
        "batch",
        "evidence",
        "next_phase",
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != required_fields:
            raise LedgerSafetyError("invalid local decisions")
        path = _safe_relative_path(record.get("path"))
        origin = record.get("origin")
        disposition = record.get("disposition")
        if not isinstance(origin, str) or origin not in ORIGINS:
            raise LedgerSafetyError("invalid local decisions")
        if not isinstance(disposition, str) or disposition not in DISPOSITIONS:
            raise LedgerSafetyError("invalid local decisions")
        batch = record.get("batch")
        if not isinstance(batch, str) or not BATCH.fullmatch(batch):
            raise LedgerSafetyError("invalid local decisions")
        next_phase = record.get("next_phase")
        if next_phase != NEXT_PHASE_BY_DISPOSITION[disposition]:
            raise LedgerSafetyError("invalid local decisions")
        key = _candidate_key(path, origin)
        if key in decisions:
            raise LedgerSafetyError("invalid local decisions")
        decisions[key] = Decision(
            disposition=disposition,
            reason=_safe_token(record.get("reason")),
            batch=batch,
            evidence=_safe_token(record.get("evidence")),
            next_phase=next_phase,
        )
    return decisions


def _candidate_id(key: bytes, candidate: Candidate) -> str:
    message = b"\0".join(
        (
            b"p3-disposition-ledger-v1",
            candidate.origin.encode("utf-8"),
            candidate.path.encode("utf-8"),
        )
    )
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def build_ledger(inventory_path: Path, decisions_path: Path, hmac_key_path: Path) -> dict[str, object]:
    """Return a public P3 ledger after validating all local-only inputs."""
    inventory_file, inventory_payload = _load_json(inventory_path, "local inventory")
    decision_file, _ = _load_json(decisions_path, "local decisions")
    key = _read_regular_file(hmac_key_path, "hmac key", MAX_HMAC_KEY_BYTES, private=True)
    if len(key) < MIN_HMAC_KEY_BYTES:
        raise LedgerSafetyError("unsafe hmac key")
    candidates = _parse_inventory(inventory_file)
    decisions = _parse_decisions(decision_file)
    if set(candidates) != set(decisions):
        raise LedgerSafetyError("local decisions do not cover every candidate")

    entries: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    combined_counts: Counter[tuple[str, str]] = Counter()
    for candidate in candidates.values():
        decision = decisions[_candidate_key(candidate.path, candidate.origin)]
        class_counts[candidate.classification] += 1
        disposition_counts[decision.disposition] += 1
        combined_counts[(candidate.classification, decision.disposition)] += 1
        entries.append(
            {
                "candidate_id": _candidate_id(key, candidate),
                "classification": candidate.classification,
                "disposition": decision.disposition,
                "reason": decision.reason,
                "batch": decision.batch,
                "evidence": decision.evidence,
                "next_phase": decision.next_phase,
            }
        )
    return {
        "format": FORMAT_VERSION,
        "tool": f"p3-disposition-ledger/{TOOL_VERSION}",
        "inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
        "counts": {
            "total_candidates": len(entries),
            "by_classification": [
                {"classification": classification, "count": count}
                for classification, count in sorted(class_counts.items())
            ],
            "by_disposition": [
                {"disposition": disposition, "count": count}
                for disposition, count in sorted(disposition_counts.items())
            ],
            "by_classification_and_disposition": [
                {"classification": classification, "disposition": disposition, "count": count}
                for (classification, disposition), count in sorted(combined_counts.items())
            ],
        },
        "entries": sorted(entries, key=lambda entry: str(entry["candidate_id"])),
    }


def _open_safe_output_parent(output_path: Path) -> tuple[Path, int]:
    absolute = _absolute_lexical_path(output_path)
    parent_descriptor = _open_directory(absolute.parent)
    try:
        metadata = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise LedgerSafetyError("unsafe output directory")
        try:
            os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return absolute, parent_descriptor
        raise LedgerSafetyError("output path must not already exist")
    except Exception:
        os.close(parent_descriptor)
        raise


def _write_output(output_path: Path, ledger: Mapping[str, object]) -> None:
    payload = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    absolute, parent_descriptor = _open_safe_output_parent(output_path)
    temporary_name = f".{absolute.name}.{secrets.token_hex(16)}.tmp"
    temporary_created = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644, dir_fd=parent_descriptor, follow_symlinks=False)
        os.link(
            temporary_name,
            absolute.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.fsync(parent_descriptor)
    except FileExistsError as error:
        raise LedgerSafetyError("output path must not already exist") from error
    except OSError as error:
        raise LedgerSafetyError("unable to write public ledger") from error
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--hmac-key", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        ledger = build_ledger(arguments.inventory, arguments.decisions, arguments.hmac_key)
        _write_output(arguments.output, ledger)
    except LedgerSafetyError as error:
        print(f"p3 disposition ledger failed: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print("p3 disposition ledger failed: unable to complete ledger", file=sys.stderr)
        return 2
    print("p3 disposition ledger created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
