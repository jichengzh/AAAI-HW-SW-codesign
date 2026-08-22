"""Fresh-run guards for the P6 controller."""

from __future__ import annotations

from pathlib import Path

import pytest

from framework.stage6.coptv2x_h800_search_v2 import (
    P6CoptV2XExecutionError,
    _prepare_round_output,
)


def test_prepare_round_output_rejects_an_existing_round_root(tmp_path: Path) -> None:
    """A second controller invocation must not reuse a round's diagnostics."""
    (tmp_path / "round-00").mkdir()

    with pytest.raises(P6CoptV2XExecutionError) as captured:
        _prepare_round_output(tmp_path, 0)

    assert captured.value.failure_code == "unsafe_output"
