#!/usr/bin/env python3
"""Build the private P6 Stage1 partition manifest with the real scanner."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage6.p6_stage1_bridge_v1 import (  # noqa: E402
    P6Stage1BridgeError,
    build_p6_stage1_partition_manifest,
    run_real_stage1_scan,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--hardware", required=True, type=_absolute_path)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stage1-repo-root", required=True, type=_absolute_path)
    parser.add_argument("--heal-root", required=True, type=_absolute_path)
    parser.add_argument("--heal-checkpoint-root", required=True, type=_absolute_path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Return stable status strings without leaking private paths."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        build_p6_stage1_partition_manifest(
            args.output,
            args.hardware,
            args.device,
            {
                "STAGE1_REPO_ROOT": str(args.stage1_repo_root),
                "HEAL_ROOT": str(args.heal_root),
                "HEAL_CKPT_ROOT": str(args.heal_checkpoint_root),
            },
            run_real_stage1_scan,
        )
    except P6Stage1BridgeError:
        sys.stderr.write("stage1_scan_invalid\n")
        return 1
    except Exception:  # noqa: BLE001 - public CLI exposes only stable category.
        sys.stderr.write("stage1_scan_invalid\n")
        return 1

    sys.stdout.write("stage1_manifest_written\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
