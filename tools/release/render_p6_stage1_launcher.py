"""Render the private self-locating P6 real-Stage1 launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
from typing import Sequence


class P6Stage1LauncherRenderError(ValueError):
    """Stable path-free renderer failure."""


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )


def _existing_directory(path: Path) -> Path:
    if not path.is_absolute() or _contains_symlink_component(path):
        raise P6Stage1LauncherRenderError("stage1 launcher input is invalid")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P6Stage1LauncherRenderError(
            "stage1 launcher input is unavailable"
        ) from error
    if not resolved.is_dir():
        raise P6Stage1LauncherRenderError("stage1 launcher input is invalid")
    return resolved


def _existing_executable(path: Path) -> Path:
    if not path.is_absolute() or _contains_symlink_component(path):
        raise P6Stage1LauncherRenderError("stage1 launcher input is invalid")
    try:
        resolved = path.resolve(strict=True)
        info = resolved.lstat()
    except OSError as error:
        raise P6Stage1LauncherRenderError(
            "stage1 launcher input is unavailable"
        ) from error
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or not info.st_mode & stat.S_IXUSR
    ):
        raise P6Stage1LauncherRenderError("stage1 launcher input is invalid")
    return resolved


def _existing_regular_file(path: Path) -> Path:
    if not path.is_absolute() or _contains_symlink_component(path):
        raise P6Stage1LauncherRenderError("stage1 launcher input is invalid")
    try:
        resolved = path.resolve(strict=True)
        info = resolved.lstat()
    except OSError as error:
        raise P6Stage1LauncherRenderError(
            "stage1 launcher input is unavailable"
        ) from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise P6Stage1LauncherRenderError("stage1 launcher input is invalid")
    return resolved


def _planned_output(path: Path) -> Path:
    if not path.is_absolute() or _contains_symlink_component(path):
        raise P6Stage1LauncherRenderError("stage1 launcher output is invalid")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise P6Stage1LauncherRenderError(
            "stage1 launcher output is unavailable"
        ) from error
    destination = parent / path.name
    if not parent.is_dir() or destination.exists() or destination.is_symlink():
        raise P6Stage1LauncherRenderError("stage1 launcher output is invalid")
    return destination


_CONTAINMENT_CHECK = """PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE_ROOT:$OVERLAY_ROOT" "$TOOLING_PYTHON" - "$OUTPUT_MANIFEST" "$LOCAL_OUTPUT_ROOT" <<'PY_CONTAINMENT'
from pathlib import Path
import sys

output = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve(strict=True)
if not root.is_dir() or output.exists() or output.is_symlink():
    raise SystemExit(65)
canonical_output = output.parent.resolve(strict=True) / output.name
try:
    canonical_output.relative_to(root)
except ValueError:
    raise SystemExit(65) from None
if canonical_output == root:
    raise SystemExit(65)
PY_CONTAINMENT"""


_MAPPER_AUDIT = """PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE_ROOT:$OVERLAY_ROOT" "$TOOLING_PYTHON" - "$MAPPER" <<'PY_AUDIT'
import ast
from pathlib import Path
import sys

tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "run_real_stage1_scan"
    for node in ast.walk(tree)
):
    raise SystemExit(71)
PY_AUDIT"""


_SCHEMA_CHECK = """PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE_ROOT:$OVERLAY_ROOT" "$TOOLING_PYTHON" - "$OUTPUT_MANIFEST" <<'PY_SCHEMA'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema") != "stage1_partition_manifest_v1":
    raise SystemExit(72)
PY_SCHEMA
"""


def _launcher_runtime_header(
    *,
    tooling_python: Path,
    heal_root: Path,
    heal_checkpoint_root: Path,
    scenario_path: Path,
) -> str:
    return f"""#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then exit 64; fi
OUTPUT_MANIFEST=$1
LOCAL_OUTPUT_ROOT=$2
BIN_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
EXECUTION_ROOT=$(CDPATH= cd -- "$BIN_DIR/../.." && pwd -P)
CODE_ROOT=$EXECUTION_ROOT/public-code
OVERLAY_ROOT=$EXECUTION_ROOT/dependency-overlay
MAPPER=$EXECUTION_ROOT/private-runner/bin/stage1-map-real-private.py
HARDWARE=$CODE_ROOT/configs/hardware/h800.yaml
TOOLING_PYTHON={shlex.quote(str(tooling_python))}
HEAL_ROOT={shlex.quote(str(heal_root))}
HEAL_CHECKPOINT_ROOT={shlex.quote(str(heal_checkpoint_root))}
SCENARIO={shlex.quote(str(scenario_path))}
case "$OUTPUT_MANIFEST" in "$LOCAL_OUTPUT_ROOT"/*) ;; *) exit 65 ;; esac
[ -d "$LOCAL_OUTPUT_ROOT" ] || exit 66
[ ! -e "$OUTPUT_MANIFEST" ] || exit 67
[ -f "$MAPPER" ] && [ -x "$MAPPER" ] || exit 68
[ -f "$HARDWARE" ] || exit 69
[ -d "$HEAL_ROOT" ] && [ -d "$HEAL_CHECKPOINT_ROOT" ] || exit 70"""


def _mapper_invocation() -> str:
    return """cd "$CODE_ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE_ROOT:$OVERLAY_ROOT" "$TOOLING_PYTHON" "$MAPPER" --hardware "$HARDWARE" --output "$OUTPUT_MANIFEST" --device cuda:0 --scenario "$SCENARIO" --stage1-repo-root "$CODE_ROOT" --heal-root "$HEAL_ROOT" --heal-checkpoint-root "$HEAL_CHECKPOINT_ROOT\""""


def _launcher_text(
    *,
    tooling_python: Path,
    heal_root: Path,
    heal_checkpoint_root: Path,
    scenario_path: Path,
) -> str:
    return "\n".join(
        (
            _launcher_runtime_header(
                tooling_python=tooling_python,
                heal_root=heal_root,
                heal_checkpoint_root=heal_checkpoint_root,
                scenario_path=scenario_path,
            ),
            _CONTAINMENT_CHECK,
            _MAPPER_AUDIT,
            _mapper_invocation(),
            _SCHEMA_CHECK,
        )
    )


def _unlink_if_matching_descriptor(*, destination: Path, descriptor: int) -> None:
    try:
        destination_info = destination.lstat()
        descriptor_info = os.fstat(descriptor)
    except OSError:
        return
    if (
        stat.S_ISREG(destination_info.st_mode)
        and (destination_info.st_dev, destination_info.st_ino)
        == (descriptor_info.st_dev, descriptor_info.st_ino)
    ):
        destination.unlink(missing_ok=True)


def _publish_no_overwrite(*, temporary: Path, destination: Path) -> int:
    temporary_descriptor = -1
    destination_descriptor = -1
    linked = False
    returned = False
    try:
        temporary_descriptor = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW)
        temporary_info = os.fstat(temporary_descriptor)
        os.link(temporary, destination)
        linked = True
        destination_descriptor = os.open(destination, os.O_RDONLY | os.O_NOFOLLOW)
        destination_info = os.fstat(destination_descriptor)
        if (destination_info.st_dev, destination_info.st_ino) != (
            temporary_info.st_dev,
            temporary_info.st_ino,
        ):
            raise P6Stage1LauncherRenderError(
                "stage1 launcher output could not be written"
            )
        temporary.unlink()
        returned = True
        return destination_descriptor
    except (OSError, P6Stage1LauncherRenderError) as error:
        if linked:
            _unlink_if_matching_descriptor(
                destination=destination,
                descriptor=temporary_descriptor,
            )
        if isinstance(error, P6Stage1LauncherRenderError):
            raise
        raise P6Stage1LauncherRenderError(
            "stage1 launcher output could not be written"
        ) from error
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if destination_descriptor >= 0 and not returned:
            os.close(destination_descriptor)


def _rollback_published_file(*, destination: Path, descriptor: int) -> None:
    _unlink_if_matching_descriptor(destination=destination, descriptor=descriptor)


def _validated_render_inputs(
    *,
    output_path: Path,
    tooling_python: Path,
    heal_root: Path,
    heal_checkpoint_root: Path,
    scenario_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    destination = _planned_output(output_path)
    python = _existing_executable(tooling_python)
    heal = _existing_directory(heal_root)
    checkpoints = _existing_directory(heal_checkpoint_root)
    execution_root = destination.parent.parent.parent
    _existing_executable(
        execution_root / "private-runner" / "bin" / "stage1-map-real-private.py"
    )
    _existing_directory(execution_root / "public-code")
    _existing_directory(execution_root / "dependency-overlay")
    _existing_regular_file(
        execution_root / "public-code" / "configs" / "hardware" / "h800.yaml"
    )
    closure_scenario = _existing_regular_file(
        execution_root
        / "public-code"
        / "configs"
        / "stage1"
        / "p6_h800_formal_scan.yaml"
    )
    scenario = _existing_regular_file(scenario_path)
    if scenario != closure_scenario:
        raise P6Stage1LauncherRenderError("stage1 launcher input is invalid")
    return destination, python, heal, checkpoints, scenario


def _write_launcher_temporary(
    *,
    destination: Path,
    tooling_python: Path,
    heal_root: Path,
    heal_checkpoint_root: Path,
    scenario_path: Path,
) -> Path:
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(
                _launcher_text(
                    tooling_python=tooling_python,
                    heal_root=heal_root,
                    heal_checkpoint_root=heal_checkpoint_root,
                    scenario_path=scenario_path,
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o700)
        result = temporary
        temporary = None
        return result
    except OSError as error:
        raise P6Stage1LauncherRenderError(
            "stage1 launcher output could not be written"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_launcher(*, temporary: Path, destination: Path) -> Path:
    descriptor = -1
    completed = False
    try:
        descriptor = _publish_no_overwrite(
            temporary=temporary,
            destination=destination,
        )
        rendered = _existing_executable(destination)
        completed = True
        return rendered
    finally:
        if not completed and descriptor >= 0:
            _rollback_published_file(destination=destination, descriptor=descriptor)
        if descriptor >= 0:
            os.close(descriptor)


def render_stage1_launcher(
    *,
    output_path: Path,
    tooling_python: Path,
    heal_root: Path,
    heal_checkpoint_root: Path,
    scenario_path: Path,
) -> Path:
    """Atomically render a launcher bound to validated external runtime roots."""
    destination, python, heal, checkpoints, scenario = _validated_render_inputs(
        output_path=output_path,
        tooling_python=tooling_python,
        heal_root=heal_root,
        heal_checkpoint_root=heal_checkpoint_root,
        scenario_path=scenario_path,
    )
    temporary = _write_launcher_temporary(
        destination=destination,
        tooling_python=python,
        heal_root=heal,
        heal_checkpoint_root=checkpoints,
        scenario_path=scenario,
    )
    try:
        return _publish_launcher(temporary=temporary, destination=destination)
    finally:
        temporary.unlink(missing_ok=True)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--tooling-python", required=True, type=_absolute_path)
    parser.add_argument("--heal-root", required=True, type=_absolute_path)
    parser.add_argument("--heal-checkpoint-root", required=True, type=_absolute_path)
    parser.add_argument("--scenario", required=True, type=_absolute_path)
    try:
        args = parser.parse_args(argv)
        render_stage1_launcher(
            output_path=args.output,
            tooling_python=args.tooling_python,
            heal_root=args.heal_root,
            heal_checkpoint_root=args.heal_checkpoint_root,
            scenario_path=args.scenario,
        )
    except (P6Stage1LauncherRenderError, SystemExit):
        sys.stderr.write("stage1_launcher_render_invalid\n")
        return 1
    sys.stdout.write("p6_stage1_launcher_rendered\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
