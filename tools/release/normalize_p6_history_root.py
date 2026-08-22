"""Normalize explicit private P6 history sources into a private local root."""

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

from framework.stage6.p6_history_normalization_v1 import (  # noqa: E402
    P6HistoryNormalizationError,
    normalize_history_inputs,
)
from framework.stage6.p6_history_recipe_normalization_v1 import (  # noqa: E402
    load_source_map_document,
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
    parser.add_argument("--runner-template", required=False, type=_absolute_path)
    return parser.parse_args(argv)


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_check_ignored(repository: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repository)
    except ValueError:
        return True
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(relative)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _validate_private_source_map_path(path: Path) -> Path:
    if _contains_symlink_component(path):
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source map path is invalid"
        )
    try:
        resolved = path.resolve(strict=True)
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source map is unavailable"
        ) from error
    if not resolved.is_file():
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source map path is invalid"
        )
    if _is_relative_to(resolved, repository) and not _git_check_ignored(
        repository, resolved
    ):
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "source map path is not private"
        )
    return resolved


def _load_source_map(path: Path) -> dict[str, object]:
    try:
        resolved = _validate_private_source_map_path(path)
        payload = load_source_map_document(resolved)
    except (OSError, UnicodeError, P6HistoryNormalizationError) as error:
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
            runner_template_path=args.runner_template,
        )
    except P6HistoryNormalizationError as error:
        sys.stderr.write(f"{error.category}\n")
        return 1
    except Exception:
        sys.stderr.write("history_normalization_invalid\n")
        return 1

    sys.stdout.write("p6_history_root_normalized\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
