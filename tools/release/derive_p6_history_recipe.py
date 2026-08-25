"""Derive one private P6 recipe from an ignored procedural source map."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage6.p6_history_recipe_bridge_v1 import (  # noqa: E402
    P6HistoryRecipeDerivationError,
    derive_dynamic_recipe_from_procedural_source,
)
from framework.stage6.p6_history_recipe_normalization_v1 import (  # noqa: E402
    P6HistoryNormalizationError,
    load_source_map_document,
)


SOURCE_MAP_V2 = "p6_history_normalization_source_v2"
SOURCE_MAP_V3 = "p6_history_normalization_source_v3"
MAX_PRIVATE_INPUT_SIZE = 4 * 1024 * 1024
PROCEDURAL_ROLES = (
    "controller",
    "source_materializer",
    "performance_plan",
    "finalizer",
)
PROCEDURAL_SOURCE_MAP_KEYS = frozenset(
    {
        "schema_version",
        "history_root",
        "asset_paths",
        "input_sources",
        "source_contract",
        "recipe_mode",
        "procedural_recipe_profile",
        "procedural_recipe_source",
        "external_training_binding",
        "execution_code_closure",
    }
)
PROCEDURAL_SOURCE_MAP_V3_KEYS = PROCEDURAL_SOURCE_MAP_KEYS | {
    "post_source_leaf_binding"
}


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
    parser.add_argument("--runner-template", required=True, type=_absolute_path)
    parser.add_argument("--recipe-json", required=True, type=_absolute_path)
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
            shell=False,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _private_existing_file(path: Path) -> Path:
    if _contains_symlink_component(path):
        raise P6HistoryRecipeDerivationError("private input path is invalid")
    try:
        resolved = path.resolve(strict=True)
        repository = REPOSITORY_ROOT.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > MAX_PRIVATE_INPUT_SIZE:
            raise OSError
    except OSError as error:
        raise P6HistoryRecipeDerivationError(
            "private input is unavailable"
        ) from error
    if _is_relative_to(resolved, repository) and not _git_check_ignored(
        repository, resolved
    ):
        raise P6HistoryRecipeDerivationError("private input is not ignored")
    return resolved


def _load_source_map(path: Path) -> dict[str, Any]:
    try:
        resolved = _private_existing_file(path)
        payload = load_source_map_document(resolved)
    except (OSError, UnicodeError, ValueError, P6HistoryNormalizationError) as error:
        raise P6HistoryRecipeDerivationError("source map is unavailable") from error
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    expected_keys = (
        PROCEDURAL_SOURCE_MAP_V3_KEYS
        if schema_version == SOURCE_MAP_V3
        else PROCEDURAL_SOURCE_MAP_KEYS
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != set(expected_keys)
        or schema_version not in {SOURCE_MAP_V2, SOURCE_MAP_V3}
        or payload.get("recipe_mode") != "procedural_profile"
        or not isinstance(payload.get("procedural_recipe_profile"), str)
        or not isinstance(payload.get("procedural_recipe_source"), Mapping)
    ):
        raise P6HistoryRecipeDerivationError("source map is invalid")
    _validate_procedural_recipe_source(payload["procedural_recipe_source"])
    return payload


def _validate_procedural_recipe_source(raw: object) -> None:
    if not isinstance(raw, Mapping) or set(raw) not in (
        {"role_refs"},
        {"role_selection"},
    ):
        raise P6HistoryRecipeDerivationError("procedural recipe source is invalid")
    refs = raw.get("role_refs", raw.get("role_selection"))
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        if tuple(refs) != PROCEDURAL_ROLES:
            raise P6HistoryRecipeDerivationError("procedural recipe roles are invalid")
        return
    if isinstance(refs, Mapping) and set(refs) == set(PROCEDURAL_ROLES) and all(
        isinstance(value, (str, Mapping)) for value in refs.values()
    ):
        return
    raise P6HistoryRecipeDerivationError("procedural recipe roles are invalid")


def _declared_history_root(source_map: Mapping[str, Any]) -> Path:
    raw = source_map.get("history_root")
    path = Path(raw) if isinstance(raw, str) else None
    if (
        path is None
        or not path.is_absolute()
        or _contains_symlink_component(path)
    ):
        raise P6HistoryRecipeDerivationError("history root is invalid")
    return path


def _private_output_path(
    path: Path, *, protected_inputs: tuple[Path, Path]
) -> Path:
    if _contains_symlink_component(path):
        raise P6HistoryRecipeDerivationError("recipe output path is invalid")
    try:
        parent = path.parent.resolve(strict=True)
        destination = (parent / path.name).resolve(strict=False)
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise P6HistoryRecipeDerivationError(
            "recipe output is unavailable"
        ) from error
    if not parent.is_dir() or destination in protected_inputs:
        raise P6HistoryRecipeDerivationError("recipe output path is invalid")
    if destination.exists() and not destination.is_file():
        raise P6HistoryRecipeDerivationError("recipe output path is invalid")
    if _is_relative_to(destination, repository) and not _git_check_ignored(
        repository, destination
    ):
        raise P6HistoryRecipeDerivationError("recipe output is not ignored")
    return destination


def _atomic_write_recipe(path: Path, recipe: Mapping[str, Any]) -> None:
    try:
        serialized = (
            json.dumps(
                recipe,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise P6HistoryRecipeDerivationError("recipe is not canonical JSON") from error
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise P6HistoryRecipeDerivationError("recipe output could not be written") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Derive and atomically write a recipe without echoing private values."""
    try:
        args = _parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    try:
        source_map_path = _private_existing_file(args.source_map)
        runner_template_path = _private_existing_file(args.runner_template)
        recipe_path = _private_output_path(
            args.recipe_json,
            protected_inputs=(source_map_path, runner_template_path),
        )
        source_map = _load_source_map(source_map_path)
        recipe = derive_dynamic_recipe_from_procedural_source(
            source_map={
                "procedural_recipe_profile": source_map[
                    "procedural_recipe_profile"
                ],
                "procedural_recipe_source": source_map[
                    "procedural_recipe_source"
                ],
            },
            runner_template_path=runner_template_path,
            history_root=_declared_history_root(source_map),
        )
        _atomic_write_recipe(recipe_path, recipe)
    except P6HistoryRecipeDerivationError:
        sys.stderr.write("history_recipe_derivation_invalid\n")
        return 1
    except Exception:
        sys.stderr.write("history_recipe_derivation_invalid\n")
        return 1

    sys.stdout.write("p6_history_recipe_derived\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
