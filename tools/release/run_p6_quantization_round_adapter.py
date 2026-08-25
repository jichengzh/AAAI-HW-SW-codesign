#!/usr/bin/env python3
"""Run one P6 post-source quantization round from a generated wrapper."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage6.p6_post_source_adapter_profile_v1 import (  # noqa: E402
    P6PostSourceAdapterProfileError,
    load_post_source_adapter_profile,
)
from framework.stage6.p6_quantization_round_adapter_v1 import (  # noqa: E402
    P6QuantizationRoundAdapterError,
    run_quantization_round,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


class SubprocessLeafRunner:
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


def _absolute_path(raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--profile", required=True, type=_absolute_path)
    parser.add_argument("task_state", type=_absolute_path)
    parser.add_argument("round_output_root", type=_absolute_path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        private_root = args.profile.parent.resolve(strict=True)
        profile = load_post_source_adapter_profile(args.profile, private_root=private_root)
        run_quantization_round(
            profile,
            args.task_state,
            args.round_output_root,
            SubprocessLeafRunner(),
        )
    except (P6PostSourceAdapterProfileError, P6QuantizationRoundAdapterError):
        sys.stderr.write("history_execution_invalid\n")
        return 1
    except (OSError, TypeError, ValueError):
        sys.stderr.write("history_execution_invalid\n")
        return 1
    sys.stdout.write("quantization_complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
