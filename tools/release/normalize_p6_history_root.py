"""Normalize explicit private P6 history sources into a private local root."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from framework.stage6.p6_history_normalization_v1 import (  # noqa: E402
    P6HistoryNormalizationError,
    normalize_history_inputs,
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
    parser = _ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--source-map", required=True, type=_absolute_path)
    parser.add_argument("--history-root", required=True, type=_absolute_path)
    parser.add_argument("--private-dir", required=True, type=_absolute_path)
    return parser.parse_args(argv)


def _load_source_map(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source map path is invalid"
        )
    try:
        resolved = path.resolve(strict=True)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source map is unavailable"
        ) from error
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source map is invalid"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run normalization without echoing private path or contract values."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        normalize_history_inputs(
            _load_source_map(args.source_map),
            args.history_root,
            args.private_dir,
        )
    except P6HistoryNormalizationError:
        sys.stderr.write("history_normalization_invalid\n")
        return 1
    except Exception:
        sys.stderr.write("history_normalization_invalid\n")
        return 1

    sys.stdout.write("p6_history_root_normalized\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
