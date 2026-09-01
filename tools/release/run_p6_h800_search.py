"""Run the shared local P6.1 Pyramid/TVM hardware-profile search."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage6.coptv2x_h800_search_v2 import (  # noqa: E402
    P6CoptV2XContractError,
    P6CoptV2XExecutionError,
    load_local_config,
    load_public_contract,
    run_p6_coptv2x_search,
    validate_code_revision,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


def _revision_label(value: str) -> str:
    try:
        return validate_code_revision(value)
    except P6CoptV2XContractError as error:
        raise argparse.ArgumentTypeError("invalid code revision") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--local-config", required=True, type=Path)
    parser.add_argument("--code-revision", required=True, type=_revision_label)
    return parser.parse_args(argv)


def _run_command(argv: tuple[str, ...], cwd: Path) -> int:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        shell=False,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Load a legacy H800 or selected v3 profile and return stable exit codes."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        contract = load_public_contract(args.contract)
    except P6CoptV2XContractError:
        sys.stderr.write("contract_error\n")
        return 2

    try:
        local = load_local_config(args.local_config, contract)
    except P6CoptV2XContractError:
        sys.stderr.write("contract_error\n")
        return 2

    try:
        state = run_p6_coptv2x_search(contract, local, args.code_revision, _run_command)
    except P6CoptV2XContractError:
        sys.stderr.write("contract_error\n")
        return 2
    except P6CoptV2XExecutionError as error:
        if error.failure_code == "unsafe_output":
            sys.stderr.write("contract_error\n")
            return 2
        sys.stderr.write("execution_failed\n")
        return 1
    except OSError:
        sys.stderr.write("execution_failed\n")
        return 1
    if state.status != "completed":
        sys.stderr.write("execution_failed\n")
        return 1
    sys.stdout.write("completed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
