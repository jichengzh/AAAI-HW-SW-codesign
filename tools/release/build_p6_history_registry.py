#!/usr/bin/env python3
"""Build one private Stage5 registry-v2 from a dynamic P6 candidate plan."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.stage6.p6_history_registry_v1 import (  # noqa: E402
    P6HistoryRegistryError,
    materialize_history_registry,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "argument_error\n")


def _absolute_path(raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _ArgumentParser(allow_abbrev=False)
    parser.add_argument("--binding", required=True, type=_absolute_path)
    parser.add_argument(
        "--pyramid-candidate-plan", required=True, type=_absolute_path
    )
    parser.add_argument("--source-registry-json", required=True, type=_absolute_path)
    parser.add_argument("--local-output-root", required=True, type=_absolute_path)
    return parser.parse_args(argv)


def _load_private_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise P6HistoryRegistryError(
            "source_registry_invalid", "private JSON input is invalid"
        )
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > 16 * 1024 * 1024:
            raise OSError
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P6HistoryRegistryError(
            "source_registry_invalid", "private JSON input is unavailable"
        ) from error
    if not isinstance(payload, Mapping):
        raise P6HistoryRegistryError(
            "source_registry_invalid", "private JSON input is invalid"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed registry adapter and expose stable categories only."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        binding = _load_private_json(args.binding)
        plan = _load_private_json(args.pyramid_candidate_plan)
        materialize_history_registry(
            plan,
            binding,
            args.local_output_root,
            args.source_registry_json,
        )
    except P6HistoryRegistryError as error:
        sys.stderr.write(f"{error.category}\n")
        return 1
    except (OSError, TypeError, ValueError):
        sys.stderr.write("source_registry_invalid\n")
        return 1

    sys.stdout.write("source_registry_written\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
