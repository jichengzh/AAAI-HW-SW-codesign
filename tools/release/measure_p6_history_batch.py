#!/usr/bin/env python3
"""Execute one private P6 history batch and atomically persist safe feedback."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.stage6.p6_history_binding_v1 import GpuRecord  # noqa: E402
from framework.stage6.p6_history_measurement_v1 import (  # noqa: E402
    P6HistoryMeasurementError,
    run_history_measurement_batch,
)


NVIDIA_SMI_ARGV = (
    "nvidia-smi",
    "--id=5,6,7",
    "--query-gpu=index,uuid,name,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
MAX_PRIVATE_JSON_BYTES = 16 * 1024 * 1024


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


class SubprocessRunner:
    """Production direct-argv runner with no ambient environment inheritance."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        shell: bool,
    ) -> subprocess.CompletedProcess[str]:
        if shell is not False:
            raise ValueError("shell execution denied")
        return subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env),
            shell=False,
            check=False,
            text=True,
            capture_output=True,
        )


class NvidiaSmiGpuProbe:
    """Probe exactly GPUs 5, 6, and 7 through the fixed nvidia-smi argv."""

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        if indices != (5, 6, 7):
            raise ValueError("fixed GPU indices required")
        completed = subprocess.run(
            NVIDIA_SMI_ARGV,
            env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C"},
            shell=False,
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
        if completed.returncode != 0:
            raise RuntimeError("GPU query failed")
        records = _parse_gpu_records(completed.stdout)
        by_index = {record.index: record for record in records}
        return tuple(by_index[index] for index in indices if index in by_index)


def _parse_gpu_records(output: str) -> tuple[GpuRecord, ...]:
    records: list[GpuRecord] = []
    seen: set[int] = set()
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 5:
            raise ValueError("invalid GPU query row")
        index_text, uuid, model, used_text, total_text = fields
        index = int(index_text)
        if index in seen:
            raise ValueError("duplicate GPU index")
        seen.add(index)
        used = float(used_text)
        total = float(total_text)
        if total <= 0.0:
            raise ValueError("invalid GPU memory total")
        records.append(GpuRecord(index, uuid, model, used / total))
    return tuple(records)


def _absolute_path(raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--binding", required=True, type=_absolute_path)
    parser.add_argument("--measurement-request", required=True, type=_absolute_path)
    parser.add_argument("--feedback-json", required=True, type=_absolute_path)
    parser.add_argument("--round-output-root", required=True, type=_absolute_path)
    return parser.parse_args(argv)


def _load_private_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise P6HistoryMeasurementError("history_execution_invalid")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > MAX_PRIVATE_JSON_BYTES:
            raise OSError
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise P6HistoryMeasurementError("history_execution_invalid") from None
    if not isinstance(payload, Mapping):
        raise P6HistoryMeasurementError("history_execution_invalid")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate key")
        payload[key] = value
    return payload


def _prepare_destination(round_root: Path, feedback_path: Path) -> Path:
    try:
        root = round_root.absolute()
        if root.is_symlink():
            raise OSError
        resolved_root = root.resolve(strict=True)
        parent = _verified_destination_parent(root, feedback_path.parent.absolute())
    except (OSError, ValueError):
        raise P6HistoryMeasurementError("unsafe_destination") from None
    if not resolved_root.is_dir() or feedback_path.is_symlink() or feedback_path.exists():
        raise P6HistoryMeasurementError("unsafe_destination")
    destination = parent / feedback_path.name
    if not _beneath(destination, resolved_root):
        raise P6HistoryMeasurementError("unsafe_destination")
    repository = REPOSITORY_ROOT.resolve(strict=True)
    if _beneath(destination, repository) and not _git_ignored(repository, destination):
        raise P6HistoryMeasurementError("unsafe_destination")
    return destination


def _verified_destination_parent(root: Path, parent: Path) -> Path:
    relative = parent.relative_to(root)
    current = root
    mode = current.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise OSError
    for component in relative.parts:
        current /= component
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise OSError
    return current.resolve(strict=True)


def _git_ignored(repository: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repository)
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(relative)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
        )
    except (OSError, ValueError):
        return False
    return completed.returncode == 0


def _write_feedback_atomic(
    round_root: Path, feedback_path: Path, feedback: Mapping[str, Any]
) -> None:
    path = _prepare_destination(round_root, feedback_path)
    encoded = (
        json.dumps(
            feedback,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise OSError
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _beneath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed adapter and expose stable categories only."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        _prepare_destination(args.round_output_root, args.feedback_json)
        binding = _load_private_json(args.binding)
        request = _load_private_json(args.measurement_request)
        feedback = run_history_measurement_batch(
            request,
            binding,
            args.round_output_root,
            SubprocessRunner(),
            NvidiaSmiGpuProbe(),
        )
        _write_feedback_atomic(args.round_output_root, args.feedback_json, feedback)
    except P6HistoryMeasurementError as error:
        sys.stderr.write(f"{error.category}\n")
        return 1
    except (OSError, TypeError, ValueError):
        sys.stderr.write("history_execution_invalid\n")
        return 1
    sys.stdout.write("measurement_feedback_written\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
