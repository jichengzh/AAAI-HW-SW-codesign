"""Immutable scanner-owned checkpoint authority for formal Stage1 scans."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterator


_SEAL = object()
_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CheckpointFileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class FormalCheckpointSnapshot:
    """Sealed authority whose loader path names one held snapshot fd."""

    digest: str
    source_identity: CheckpointFileIdentity
    snapshot_identity: CheckpointFileIdentity
    _snapshot_fd: int = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise TypeError("formal checkpoint snapshot construction is private")

    @property
    def loader_path(self) -> Path:
        try:
            identity = _identity(os.fstat(self._snapshot_fd))
            access_mode = fcntl.fcntl(self._snapshot_fd, fcntl.F_GETFL)
        except OSError as error:
            raise ValueError("formal checkpoint snapshot is unavailable") from error
        if identity != self.snapshot_identity or access_mode & os.O_ACCMODE != os.O_RDONLY:
            raise ValueError("formal checkpoint snapshot is unavailable")
        return Path(f"/proc/self/fd/{self._snapshot_fd}")


@dataclass(frozen=True)
class _SnapshotResources:
    authority: FormalCheckpointSnapshot
    source_fd: int
    source_path: Path
    trusted_root: Path
    temporary_path: Path | None


@contextmanager
def scanner_owned_checkpoint_snapshot(
    adapter: Any,
) -> Iterator[FormalCheckpointSnapshot]:
    """Yield one immutable checkpoint copy and reject source drift."""
    resources = _create_snapshot(adapter)
    try:
        yield resources.authority
    finally:
        try:
            _verify_snapshot_authority(resources.authority)
            _verify_source_authority(resources)
        finally:
            _cleanup_snapshot(resources)


def _create_snapshot(adapter: Any) -> _SnapshotResources:
    source_fd = -1
    snapshot_fd = -1
    writable_fd = -1
    temporary_path: Path | None = None
    try:
        source_path = _adapter_checkpoint_path(adapter)
        trusted_root = _adapter_checkpoint_root(adapter)
        source_fd = _open_checkpoint(source_path, trusted_root)
        source_identity = _identity(os.fstat(source_fd))
        writable_fd, raw_path = tempfile.mkstemp(prefix=".stage1-checkpoint-", suffix=".pth")
        temporary_path = Path(raw_path)
        os.fchmod(writable_fd, 0o600)
        digest = _copy_fd_to_snapshot(source_fd, writable_fd)
        os.fsync(writable_fd)
        os.lseek(writable_fd, 0, os.SEEK_SET)
        if _hash_fd(writable_fd) != digest:
            raise ValueError("formal checkpoint authority changed")
        snapshot_identity = _identity(os.fstat(writable_fd))
        if snapshot_identity.size != source_identity.size:
            raise ValueError("formal checkpoint authority changed")
        os.fchmod(writable_fd, 0o400)
        os.fsync(writable_fd)
        os.close(writable_fd)
        writable_fd = -1
        snapshot_fd = _open_readonly_snapshot(temporary_path, snapshot_identity, digest)
        temporary_path.unlink()
        temporary_path = None
        authority = FormalCheckpointSnapshot(
            digest=digest,
            source_identity=source_identity,
            snapshot_identity=snapshot_identity,
            _snapshot_fd=snapshot_fd,
            _seal=_SEAL,
        )
        resources = _SnapshotResources(
            authority, source_fd, source_path, trusted_root, temporary_path
        )
        _verify_snapshot_authority(authority)
        _verify_source_authority(resources)
        return resources
    except Exception as error:
        try:
            _close_unlink(source_fd, snapshot_fd, writable_fd, temporary_path)
        except Exception as cleanup_error:
            raise ValueError("formal checkpoint authority is invalid") from cleanup_error
        raise ValueError("formal checkpoint authority is invalid") from error


def _open_readonly_snapshot(path: Path, expected: CheckpointFileIdentity, digest: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    snapshot_fd = os.open(path, flags)
    try:
        if _identity(os.fstat(snapshot_fd)) != expected:
            raise ValueError("formal checkpoint authority changed")
        os.lseek(snapshot_fd, 0, os.SEEK_SET)
        if _hash_fd(snapshot_fd) != digest:
            raise ValueError("formal checkpoint authority changed")
        os.lseek(snapshot_fd, 0, os.SEEK_SET)
        return snapshot_fd
    except Exception:
        os.close(snapshot_fd)
        raise


def _verify_snapshot_authority(authority: FormalCheckpointSnapshot) -> None:
    try:
        os.lseek(authority._snapshot_fd, 0, os.SEEK_SET)
        digest = _hash_fd(authority._snapshot_fd)
        identity = _identity(os.fstat(authority._snapshot_fd))
        access_mode = fcntl.fcntl(authority._snapshot_fd, fcntl.F_GETFL)
    except OSError as error:
        raise ValueError("formal checkpoint authority changed") from error
    if (
        digest != authority.digest
        or identity != authority.snapshot_identity
        or access_mode & os.O_ACCMODE != os.O_RDONLY
    ):
        raise ValueError("formal checkpoint authority changed")


def _verify_source_authority(resources: _SnapshotResources) -> None:
    expected = resources.authority.source_identity
    try:
        os.lseek(resources.source_fd, 0, os.SEEK_SET)
        digest = _hash_fd(resources.source_fd)
        held_identity = _identity(os.fstat(resources.source_fd))
        current_fd = _open_checkpoint(resources.source_path, resources.trusted_root)
        try:
            current_identity = _identity(os.fstat(current_fd))
        finally:
            os.close(current_fd)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("formal checkpoint authority changed") from error
    if (
        digest != resources.authority.digest
        or held_identity != expected
        or current_identity != expected
    ):
        raise ValueError("formal checkpoint authority changed")


def _adapter_checkpoint_path(adapter: Any) -> Path:
    resolver = getattr(adapter, "resolved_checkpoint_path", None)
    value = resolver() if callable(resolver) else getattr(adapter, "ckpt_path", None)
    if not isinstance(value, (str, Path)):
        raise ValueError("formal checkpoint authority is invalid")
    return Path(value)


def _adapter_checkpoint_root(adapter: Any) -> Path:
    root = getattr(adapter, "formal_checkpoint_root", None)
    if not isinstance(root, Path):
        raise ValueError("formal checkpoint authority is invalid")
    return root


def _open_checkpoint(path: Path, trusted_root: Path) -> int:
    relative = _validated_relative_path(path, trusted_root)
    directory_fd = _open_absolute_directory(trusted_root)
    try:
        for component in relative.parts[:-1]:
            next_fd = _open_directory(component, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        checkpoint_fd = os.open(relative.name, flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    if not stat.S_ISREG(os.fstat(checkpoint_fd).st_mode):
        os.close(checkpoint_fd)
        raise ValueError("formal checkpoint authority is invalid")
    return checkpoint_fd


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("formal checkpoint authority is invalid")
    directory_fd = _open_directory(path.anchor)
    try:
        for component in path.parts[1:]:
            next_fd = _open_directory(component, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _open_directory(path: str | Path, *, dir_fd: int | None = None) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    return os.open(path, flags, dir_fd=dir_fd)


def _validated_relative_path(path: Path, trusted_root: Path) -> Path:
    try:
        if (
            not path.is_absolute()
            or not trusted_root.is_absolute()
            or _contains_symlink_component(path)
            or _contains_symlink_component(trusted_root)
        ):
            raise ValueError
        if any(part in {".", ".."} for part in (*path.parts, *trusted_root.parts)):
            raise ValueError
        root_parts = trusted_root.parts
        if path.parts[: len(root_parts)] != root_parts:
            raise ValueError
        relative_parts = path.parts[len(root_parts) :]
        if not relative_parts:
            raise ValueError
        return Path(*relative_parts)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("formal checkpoint authority is invalid") from error


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink() for component in (path, *path.parents)
    )


def _copy_fd_to_snapshot(source_fd: int, snapshot_fd: int) -> str:
    os.lseek(source_fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(source_fd, _CHUNK_BYTES):
        digest.update(chunk)
        _write_all(snapshot_fd, chunk)
    return digest.hexdigest()


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    while chunk := os.read(fd, _CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("snapshot write failed")
        view = view[written:]


def _identity(value: os.stat_result) -> CheckpointFileIdentity:
    return CheckpointFileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
    )


def _cleanup_snapshot(resources: _SnapshotResources) -> None:
    _close_unlink(
        resources.source_fd,
        resources.authority._snapshot_fd,
        -1,
        resources.temporary_path,
    )


def _close_unlink(
    source_fd: int,
    snapshot_fd: int,
    writable_fd: int,
    temporary_path: Path | None,
) -> None:
    for fd in (source_fd, snapshot_fd, writable_fd):
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    if temporary_path is not None:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as error:
            raise ValueError("formal checkpoint snapshot cleanup failed") from error


__all__ = ["FormalCheckpointSnapshot", "scanner_owned_checkpoint_snapshot"]
