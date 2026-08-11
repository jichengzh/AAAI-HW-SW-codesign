"""Run the explicit local P6 H800 search and publish a redacted summary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.stage6.h800_search_execution_v1 import (  # noqa: E402
    H800SearchContractError,
    load_local_config,
    load_public_contract,
    run_h800_search,
    validate_code_revision,
    write_public_summary,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


def _revision_label(value: str) -> str:
    try:
        return validate_code_revision(value)
    except H800SearchContractError as error:
        raise argparse.ArgumentTypeError("invalid code revision") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--local-config", required=True, type=Path)
    parser.add_argument("--public-summary", required=True, type=Path)
    parser.add_argument("--code-revision", required=True, type=_revision_label)
    return parser.parse_args(argv)


def _paths_conflict(public_summary: Path, local_output_root: Path) -> bool:
    public_path = public_summary.resolve(strict=False)
    local_path = local_output_root.resolve(strict=False)
    return (
        public_path == local_path
        or local_path in public_path.parents
        or public_path in local_path.parents
    )


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
    """Return stable exit codes for completed, failed, and invalid searches."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        contract = load_public_contract(args.contract)
    except H800SearchContractError:
        sys.stderr.write("contract_error\n")
        return 2

    try:
        local = load_local_config(args.local_config, contract)
    except H800SearchContractError:
        sys.stderr.write("contract_error\n")
        return 2
    try:
        outputs_conflict = _paths_conflict(args.public_summary, local.local_output_root)
    except (OSError, RuntimeError):
        outputs_conflict = True
    if outputs_conflict:
        sys.stderr.write("contract_error\n")
        return 2

    try:
        summary = run_h800_search(
            contract=contract,
            local=local,
            code_revision=args.code_revision,
            command_runner=_run_command,
            public_summary_path=args.public_summary,
        )
    except H800SearchContractError:
        sys.stderr.write("contract_error\n")
        return 2
    try:
        write_public_summary(args.public_summary, summary)
    except OSError:
        sys.stderr.write("summary_write_error\n")
        return 2
    if summary.status == "failed":
        sys.stderr.write("execution_failed\n")
        return 1
    sys.stdout.write("completed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
