"""Security contracts for scanner-owned immutable checkpoint snapshots."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import stat

import pytest

import framework.stage1.formal_checkpoint_snapshot as snapshot_module
from framework.stage1.formal_checkpoint_snapshot import (
    scanner_owned_checkpoint_snapshot,
)


class _Adapter:
    def __init__(self, checkpoint: Path, trusted_root: Path) -> None:
        self.ckpt_path = str(checkpoint)
        self.formal_checkpoint_root = trusted_root


class _LeakingResolverAdapter(_Adapter):
    def resolved_checkpoint_path(self) -> Path:
        raise ValueError("formal checkpoint resolver failed at /private/secret/checkpoint.pth")


def test_snapshot_is_owner_only_sealed_reusable_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root = tmp_path / "trusted"
    snapshot_root = tmp_path / "snapshots"
    trusted_root.mkdir()
    snapshot_root.mkdir()
    checkpoint = trusted_root / "checkpoint.pth"
    checkpoint.write_bytes(b"immutable-checkpoint-bytes")
    monkeypatch.setattr(snapshot_module.tempfile, "tempdir", str(snapshot_root))

    with scanner_owned_checkpoint_snapshot(_Adapter(checkpoint, trusted_root)) as authority:
        loader_path = authority.loader_path
        assert list(snapshot_root.iterdir()) == []
        assert stat.S_IMODE(loader_path.stat().st_mode) == 0o400
        assert loader_path.read_bytes() == checkpoint.read_bytes()
        assert loader_path.read_bytes() == checkpoint.read_bytes()
        assert authority.digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with pytest.raises(OSError):
            os.open(loader_path, os.O_WRONLY)
        with pytest.raises(OSError):
            os.open(loader_path, os.O_RDWR)
        with pytest.raises(TypeError, match="construction is private"):
            replace(authority, _seal=object())

    assert list(snapshot_root.iterdir()) == []
    assert not loader_path.exists()
    with pytest.raises(ValueError, match="snapshot is unavailable"):
        _ = authority.loader_path


def test_snapshot_wraps_resolver_error_without_private_path_leak(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    checkpoint = trusted_root / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError) as caught:
        with scanner_owned_checkpoint_snapshot(_LeakingResolverAdapter(checkpoint, trusted_root)):
            pytest.fail("resolver failure must not yield an authority")

    assert str(caught.value) == "formal checkpoint authority is invalid"
    assert "/private/" not in str(caught.value)
    assert caught.value.__cause__ is not None


def test_snapshot_rejects_same_size_write_with_restored_mtime(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    checkpoint = trusted_root / "checkpoint.pth"
    expected = b"ORIGINAL-CONTENT"
    replacement = b"MODIFIED-CONTENT"
    assert len(expected) == len(replacement)
    checkpoint.write_bytes(expected)

    with scanner_owned_checkpoint_snapshot(_Adapter(checkpoint, trusted_root)) as authority:
        loader_path = authority.loader_path
        original_stat = loader_path.stat()
        try:
            writable_fd = os.open(loader_path, os.O_RDWR)
        except OSError:
            writable_fd = -1
        if writable_fd >= 0:
            try:
                os.write(writable_fd, replacement)
            finally:
                os.close(writable_fd)
            os.utime(
                loader_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )

        assert writable_fd == -1
        assert loader_path.read_bytes() == expected
        assert hashlib.sha256(loader_path.read_bytes()).hexdigest() == authority.digest


def test_snapshot_rejects_trusted_root_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = tmp_path / "authority-container"
    trusted_root = container / "trusted"
    trusted_root.mkdir(parents=True)
    checkpoint = trusted_root / "checkpoint.pth"
    checkpoint.write_bytes(b"ORIGINAL")

    alternate_parent = tmp_path / "alternate-container"
    alternate_root = alternate_parent / "trusted"
    alternate_root.mkdir(parents=True)
    (alternate_root / checkpoint.name).write_bytes(b"REDIRECT")
    parked_container = tmp_path / "parked-container"
    original_open_directory = snapshot_module._open_directory

    def swapping_open_directory(path: str | Path, *, dir_fd: int | None = None) -> int:
        should_swap = Path(path) == trusted_root or (
            Path(path) == Path(container.name) and dir_fd is not None
        )
        if not should_swap:
            return original_open_directory(path, dir_fd=dir_fd)
        container.rename(parked_container)
        container.symlink_to(alternate_parent, target_is_directory=True)
        try:
            return original_open_directory(path, dir_fd=dir_fd)
        finally:
            container.unlink()
            parked_container.rename(container)

    monkeypatch.setattr(snapshot_module, "_open_directory", swapping_open_directory)

    with pytest.raises(ValueError, match="checkpoint authority is invalid"):
        with scanner_owned_checkpoint_snapshot(_Adapter(checkpoint, trusted_root)):
            pytest.fail("ancestor replacement must not yield an authority")


@pytest.mark.parametrize("change_kind", ["replace", "in_place"])
def test_snapshot_rejects_source_change_during_initial_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_kind: str,
) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    checkpoint = trusted_root / "checkpoint.pth"
    replacement = trusted_root / "replacement.pth"
    checkpoint.write_bytes(b"original-checkpoint")
    replacement.write_bytes(b"replacement-content")
    original_copy = snapshot_module._copy_fd_to_snapshot

    def changing_copy(source_fd: int, snapshot_fd: int) -> str:
        digest = original_copy(source_fd, snapshot_fd)
        if change_kind == "replace":
            replacement.replace(checkpoint)
        else:
            os.utime(checkpoint, None)
            checkpoint.write_bytes(replacement.read_bytes())
        return digest

    monkeypatch.setattr(snapshot_module, "_copy_fd_to_snapshot", changing_copy)

    with pytest.raises(ValueError, match="checkpoint authority is invalid"):
        with scanner_owned_checkpoint_snapshot(_Adapter(checkpoint, trusted_root)):
            pytest.fail("changed checkpoint must not yield an authority")
