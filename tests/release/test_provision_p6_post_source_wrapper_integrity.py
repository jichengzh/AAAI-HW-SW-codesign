from __future__ import annotations

from pathlib import Path

from framework.stage6.p6_post_source_wrapper_template_v1 import WRAPPER_RELATIVE_PATHS
from tests.release.test_provision_p6_full_chain_local_config import (
    _run_cli,
    _v3_valid_args,
)


def test_cli_rejects_tampered_post_source_wrapper_without_rewriting_it(
    tmp_path: Path,
) -> None:
    args, normalized = _v3_valid_args(tmp_path)
    wrapper = (
        tmp_path
        / "v3-private-normalized"
        / WRAPPER_RELATIVE_PATHS["quantization"]
    )
    tampered = b"#!/bin/sh\nexit 91\n"
    wrapper.write_bytes(tampered)
    wrapper.chmod(0o700)

    result = _run_cli(
        tmp_path,
        *args,
        "--post-source-adapter-profile",
        str(normalized["post_source_adapter_profile"]),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_execution_invalid\n"
    assert wrapper.read_bytes() == tampered
    assert not (tmp_path / "v3-private-output" / "binding.json").exists()
    assert not (tmp_path / "v3-private-output" / "p6.local.yaml").exists()
