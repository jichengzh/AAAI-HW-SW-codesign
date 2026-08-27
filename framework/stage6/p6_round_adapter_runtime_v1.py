"""Shared minimal runtime for P6 post-source round adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

from framework.stage6.p6_gpu_policy_v1 import parse_gpu_indices_csv
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    ValidatedPostSourceAdapterProfile,
)


MAX_ROUND_JSON_BYTES = 16 * 1024 * 1024


class P6RoundAdapterRuntimeError(ValueError):
    """Stable path-free runtime failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


class RunnerResult(Protocol):
    returncode: int


class LeafRunner(Protocol):
    """Injected direct-argv process boundary for historical leaves."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> RunnerResult: ...


@dataclass(frozen=True)
class RoundContext:
    profile: ValidatedPostSourceAdapterProfile
    request: Mapping[str, Any]
    task_state: Mapping[str, Any]
    task_state_path: Path
    round_root: Path
    gpu_indices: tuple[str, ...]


def load_round_context(
    profile: ValidatedPostSourceAdapterProfile,
    task_state: Path,
    round_root: Path,
    *,
    prior_stage: str,
) -> RoundContext:
    """Read and validate the canonical request/state identity for one round."""
    try:
        if not isinstance(profile, ValidatedPostSourceAdapterProfile):
            _invalid()
        root = _existing_directory(round_root)
        state_path = _beneath_existing_file(Path(task_state), root)
        request = _read_mapping(root / "measurement-request.json")
        state = _read_mapping(state_path)
        _validate_request_identity(request)
        _validate_state_identity(state, request, prior_stage=prior_stage)
        return RoundContext(
            profile=profile,
            request=request,
            task_state=state,
            task_state_path=state_path,
            round_root=root,
            gpu_indices=_ordered_gpu_indices(),
        )
    except P6RoundAdapterRuntimeError:
        raise
    except (OSError, TypeError, ValueError, KeyError):
        raise P6RoundAdapterRuntimeError() from None


def advance_task_state(context: RoundContext, next_stage: str) -> None:
    """Atomically replace only the task-state stage string."""
    try:
        if not isinstance(context, RoundContext) or not _valid_stage(next_stage):
            _invalid()
        current = _read_mapping(context.task_state_path)
        if current != context.task_state:
            _invalid()
        updated = {"stage": next_stage, "rows": current["rows"]}
        _atomic_write_json(context.task_state_path, updated)
    except P6RoundAdapterRuntimeError:
        raise
    except (OSError, TypeError, ValueError, KeyError):
        raise P6RoundAdapterRuntimeError() from None


def _validate_request_identity(request: Mapping[str, Any]) -> None:
    if (
        request.get("schema_version") != "stage5_measurement_request_v2"
        or request.get("batch_size") != 4
        or not isinstance(request.get("rows"), list)
        or len(request["rows"]) != 4
        or not isinstance(request.get("row_sha256"), Mapping)
    ):
        _invalid()
    rows = request["rows"]
    row_ids = [_row_identity(row) for row in rows]
    if len(set(row_ids)) != 4 or set(request["row_sha256"]) != set(row_ids):
        _invalid()
    if any(request["row_sha256"][row_id] != _canonical_sha(row) for row, row_id in zip(rows, row_ids, strict=True)):
        _invalid()
    body = {key: value for key, value in request.items() if key != "measurement_request_sha256"}
    if request.get("measurement_request_sha256") != _canonical_sha(body):
        _invalid()


def _row_identity(row: object) -> str:
    if not isinstance(row, Mapping):
        _invalid()
    row_id = row.get("row_id")
    width = row.get("width")
    q_mode = row.get("q_mode")
    contract = row.get("source_contract")
    if (
        not isinstance(row_id, str)
        or not row_id
        or q_mode not in {"fp16", "int8"}
        or not isinstance(width, list)
        or len(width) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in width)
        or not isinstance(contract, Mapping)
    ):
        _invalid()
    return row_id


def _validate_state_identity(
    state: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    prior_stage: str,
) -> None:
    if (
        set(state) != {"stage", "rows"}
        or state.get("stage") != prior_stage
        or not isinstance(state.get("rows"), list)
    ):
        _invalid()
    if len(state["rows"]) != 4:
        _invalid()
    request_rows = request["rows"]
    row_hashes = request["row_sha256"]
    for state_row, request_row in zip(state["rows"], request_rows, strict=True):
        if not isinstance(state_row, Mapping):
            _invalid()
        row_id = request_row["row_id"]
        source_evidence = state_row.get("source_evidence_sha256")
        if (
            state_row.get("row_id") != row_id
            or state_row.get("row_sha256") != row_hashes[row_id]
            or not _is_sha(source_evidence)
            or source_evidence != request_row.get("source_evidence_sha256")
            or state_row.get("terminal_status") != "pending"
        ):
            _invalid()


def _ordered_gpu_indices() -> tuple[str, ...]:
    try:
        indices = parse_gpu_indices_csv(os.environ.get("CUDA_VISIBLE_DEVICES"))
    except ValueError:
        _invalid()
    return tuple(str(index) for index in indices)


def _existing_directory(path: Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        _invalid()
    return resolved


def _beneath_existing_file(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file() or not _beneath(resolved, root):
        _invalid()
    return resolved


def _read_mapping(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ROUND_JSON_BYTES:
        _invalid()
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, Mapping):
        _invalid()
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate key")
        payload[key] = value
    return payload


def _valid_stage(stage: object) -> bool:
    return isinstance(stage, str) and stage in {"quantization", "performance", "ap", "finalization"}


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _beneath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _invalid() -> None:
    raise P6RoundAdapterRuntimeError()
