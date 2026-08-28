"""Immutable scanner-owned model configuration authority."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from framework.stage1.formal_checkpoint_snapshot import (
    FormalCheckpointSnapshot,
    scanner_owned_checkpoint_snapshot,
)


_SEAL = object()


@dataclass(frozen=True)
class FormalConfigSnapshot:
    """Typed view of a held, read-only configuration snapshot."""

    digest: str
    _authority: FormalCheckpointSnapshot = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise TypeError("formal config snapshot construction is private")

    @property
    def loader_path(self) -> Path:
        return self._authority.loader_path


@dataclass(frozen=True)
class _ConfigArtifactAdapter:
    ckpt_path: Path
    formal_checkpoint_root: Path


class _ConfigSnapshotContext(AbstractContextManager[FormalConfigSnapshot]):
    def __init__(self, adapter: Any) -> None:
        self._manager = scanner_owned_checkpoint_snapshot(_config_adapter(adapter))

    def __enter__(self) -> FormalConfigSnapshot:
        try:
            authority = self._manager.__enter__()
            return FormalConfigSnapshot(authority.digest, authority, _SEAL)
        except Exception as error:
            raise ValueError("formal config authority is invalid") from error

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            return self._manager.__exit__(exc_type, exc_value, traceback)
        except Exception as error:
            raise ValueError("formal config authority is invalid") from error


def scanner_owned_config_snapshot(adapter: Any) -> _ConfigSnapshotContext:
    """Create one secure config snapshot using the declared trusted root."""

    return _ConfigSnapshotContext(adapter)


def _config_adapter(adapter: Any) -> _ConfigArtifactAdapter:
    try:
        path = Path(adapter.config_path)
        root = adapter.formal_config_root
        if not isinstance(root, Path):
            raise TypeError
        return _ConfigArtifactAdapter(path, root)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("formal config authority is invalid") from error


__all__ = ["FormalConfigSnapshot", "scanner_owned_config_snapshot"]
