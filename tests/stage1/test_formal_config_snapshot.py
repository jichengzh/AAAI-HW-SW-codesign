"""Scanner-owned authority tests for model-native configuration bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from framework.stage1.formal_config_snapshot import scanner_owned_config_snapshot


class _Adapter:
    def __init__(self, path: Path, root: Path) -> None:
        self.config_path = str(path)
        self.formal_config_root = root


def test_config_snapshot_binds_digest_and_native_loader_to_identical_bytes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    raw = b"model:\n  width: 12\n"
    config.write_bytes(raw)

    with scanner_owned_config_snapshot(_Adapter(config, tmp_path)) as authority:
        assert authority.loader_path.read_bytes() == raw
        assert authority.digest == hashlib.sha256(raw).hexdigest()


def test_config_snapshot_rejects_source_drift_after_native_load(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  width: 12\n", encoding="utf-8")

    with pytest.raises(ValueError, match="^formal config authority is invalid$"):
        with scanner_owned_config_snapshot(_Adapter(config, tmp_path)) as authority:
            assert authority.loader_path.read_text(encoding="utf-8").endswith("12\n")
            config.write_text("model:\n  width: 16\n", encoding="utf-8")
