"""Validate explicit environment inputs without inspecting the local host."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.environment_contract import (  # noqa: E402
    EnvironmentContractError,
    ValidationResult,
    load_environment_contract,
    load_environment_observation,
    validate_environment,
)


REPORT_SCHEMA = "environment_contract_report_v1"
_WRITE_FAILURE_MESSAGE = "Unable to write report.\n"
_ARGUMENT_FAILURE_MESSAGE = "Invalid command input.\n"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, _ARGUMENT_FAILURE_MESSAGE)


def render_report(result: ValidationResult) -> dict[str, object]:
    """Render a stable report without exposing observations or filesystem details."""
    return {
        "schema": REPORT_SCHEMA,
        "target": result.target,
        "passed": result.passed,
        "failures": [
            {"code": failure.code, "field": failure.field} for failure in result.failures
        ],
    }


def render_invalid_report() -> dict[str, object]:
    """Render the fixed report used for every invalid explicit input."""
    return {
        "schema": REPORT_SCHEMA,
        "target": None,
        "passed": False,
        "failures": [{"code": "input.invalid", "field": "input"}],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _write_json(output: Path, payload: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_invalid_report(output: Path) -> bool:
    try:
        _write_json(output, render_invalid_report())
    except OSError:
        sys.stderr.write(_WRITE_FAILURE_MESSAGE)
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Validate supplied files and return stable, redacted exit codes."""
    try:
        args = _parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        contract = load_environment_contract(args.contract, repository_root=REPOSITORY_ROOT)
        observation = load_environment_observation(args.observation)
        report = render_report(validate_environment(contract, observation))
    except (EnvironmentContractError, OSError, json.JSONDecodeError, yaml.YAMLError):
        if not _write_invalid_report(args.output):
            return 2
        return 2

    try:
        _write_json(args.output, report)
    except OSError:
        sys.stderr.write(_WRITE_FAILURE_MESSAGE)
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
