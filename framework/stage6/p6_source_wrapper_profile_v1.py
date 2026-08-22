"""Render and validate a deterministic private P6 source marker wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Literal

import yaml

from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_runner_template_validator_v1 import (
    ValidatedRunnerTemplate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILE_SCHEMA_VERSION = "p6_private_source_wrapper_profile_v1"
WRAPPER_KIND = "repo_cwd_exec_v1"
SOURCE_MARKER_BASENAME = "stage5_materialize_round_sources_v1.sh"
PROFILE_KEYS = frozenset(
    {
        "schema_version",
        "wrapper_kind",
        "destination_relative_path",
        "implementation_relative_path",
        "implementation_cwd_relative_path",
    }
)
EXPECTED_EXECUTION_STAGES = (
    "source_materialization",
    "quantization",
    "performance",
    "ap",
    "finalization",
)
SOURCE_TEMPLATE_ARGV_TAIL = ("{measurement_request}", "{round_output_root}")
SOURCE_RUNTIME_ARGV_SHAPE = (
    "--request",
    "<absolute-private-request-json>",
    "--model",
    "pyramid",
    "--group-id",
    "<canonical-pyramid-group-id>",
    "--gpu",
    "<validated-binding-gpu-index>",
)
MAX_PROFILE_SIZE = 1024 * 1024
SAFE_RELATIVE_PATH = re.compile(r"\A[A-Za-z0-9._/-]+\Z")
_WRAPPER_TEMPLATE = '''#!/usr/bin/python3
"""Generated P6 historical source-materializer compatibility wrapper."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "P6_HISTORY_RUN_MODE",
    "P6_HISTORY_PRIVATE_ROOT",
    "P6_HISTORY_TASK_STATE",
    "P6_HISTORY_ROUND_OUTPUT_ROOT",
)
BINDING_TRAINING_KEYS = (
    "training_required",
    "training_source_kind",
    "dataset_root",
    "base_checkpoint_path",
    "base_checkpoint_sha256",
    "pyramid_config_path",
    "pyramid_config_sha256",
    "training_parameters",
)
LEGACY_TRAINING_KEYS = (
    *BINDING_TRAINING_KEYS,
    "base_checkpoint_dir",
    "training_epoches",
    "groups",
    "width_per_group",
)
BINDING_KEYS = ("schema_version", *BINDING_TRAINING_KEYS)
PARAMETER_KEYS = tuple("""training_mode epochs target_epoch seed optimizer learning_rate
batch_size dataset_split checkpoint_selection freeze_policy groups width_per_group""".split())
REQUEST_KEYS = tuple("""schema_version task_id task_sha256 round_index batch_size sample_budget
required_metrics atomic_feedback real_h800_measurement_required row_sha256 rows
measurement_request_sha256""".split())
ROW_KEYS = tuple("""schema_version task_id task_sha256 row_id manifest_job_id group_id model width
width_schema structure_widths genome strategy_id q_mode hardware_id capability_profile_id
capability_digest dispatch_key source_status materialization_kind source_evidence_kind
source_contract source_contract_sha256 source_evidence_sha256 graph_features""".split())
OUTPUT_PATH_KEYS = tuple("""checkpoint_path checkpoint_dir config_path training_done_marker
onnx_path onnx_report_path calibration_root calibration_npz calibration_summary
trt_calibration_dir source_done_marker""".split())
CANONICAL_MATERIALIZATION_KIND = "local_pyramid_tvm"
LEGACY_MATERIALIZATION_KIND = "pyramid_prepare_train_export"
IMPLEMENTATION_RELATIVE = __IMPLEMENTATION_RELATIVE__
IMPLEMENTATION_CWD_RELATIVE = __IMPLEMENTATION_CWD_RELATIVE__


class CompatibilityError(ValueError):
    pass


def _canonical_sha(payload):
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CompatibilityError
        result[key] = value
    return result


def _invalid_constant(_value):
    raise CompatibilityError


def _valid_parameters(raw):
    if not isinstance(raw, dict) or set(raw) != set(PARAMETER_KEYS):
        return False
    strings = (
        "training_mode", "optimizer", "dataset_split",
        "checkpoint_selection", "freeze_policy",
    )
    return (
        all(isinstance(raw[key], str) and raw[key].strip() for key in strings)
        and isinstance(raw["epochs"], int) and not isinstance(raw["epochs"], bool)
        and raw["epochs"] > 0
        and isinstance(raw["target_epoch"], int)
        and not isinstance(raw["target_epoch"], bool)
        and raw["target_epoch"] > 0
        and isinstance(raw["seed"], int) and not isinstance(raw["seed"], bool)
        and raw["seed"] >= 0
        and isinstance(raw["learning_rate"], (int, float))
        and not isinstance(raw["learning_rate"], bool)
        and math.isfinite(raw["learning_rate"])
        and raw["learning_rate"] > 0
        and isinstance(raw["batch_size"], int)
        and not isinstance(raw["batch_size"], bool)
        and raw["batch_size"] > 0
        and isinstance(raw["groups"], int)
        and not isinstance(raw["groups"], bool)
        and raw["groups"] > 0
        and isinstance(raw["width_per_group"], int)
        and not isinstance(raw["width_per_group"], bool)
        and raw["width_per_group"] > 0
    )


def _validated_binding(raw):
    if not isinstance(raw, dict) or set(raw) != set(BINDING_KEYS):
        raise CompatibilityError
    digests = (raw["base_checkpoint_sha256"], raw["pyramid_config_sha256"])
    paths = (raw["dataset_root"], raw["base_checkpoint_path"], raw["pyramid_config_path"])
    if (
        raw["schema_version"] != "p6_external_training_binding_v1"
        or raw["training_required"] is not True
        or raw["training_source_kind"] != "selected_candidate_finetune"
        or any(not isinstance(value, str) or not value for value in paths)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        )
        or not _valid_parameters(raw["training_parameters"])
    ):
        raise CompatibilityError
    parameters = raw["training_parameters"]
    return {
        **{key: copy.deepcopy(raw[key]) for key in BINDING_TRAINING_KEYS},
        "base_checkpoint_dir": str(Path(raw["base_checkpoint_path"]).parent),
        "training_epoches": parameters["target_epoch"],
        "groups": parameters["groups"],
        "width_per_group": parameters["width_per_group"],
    }


def _canonical_output_path(raw):
    if not isinstance(raw, str) or not raw:
        raise CompatibilityError
    path = Path(raw)
    if not path.is_absolute() or str(path) != raw or ".." in path.parts:
        raise CompatibilityError
    return path

def _config_paths(row):
    contract = row["source_contract"]
    paths = tuple(_canonical_output_path(contract.get(key)) for key in OUTPUT_PATH_KEYS)
    if len(set(paths)) != len(OUTPUT_PATH_KEYS):
        raise CompatibilityError
    checkpoint = paths[OUTPUT_PATH_KEYS.index("checkpoint_dir")]
    canonical = paths[OUTPUT_PATH_KEYS.index("config_path")]
    legacy = checkpoint / "config.yaml"
    if (
        not canonical.is_relative_to(checkpoint.parent)
        or legacy != canonical and legacy in paths
    ):
        raise CompatibilityError
    return legacy, canonical

def _legacy_row(row):
    contract = row.get("source_contract") if isinstance(row, dict) else None
    width = row.get("width") if isinstance(row, dict) else None
    if (
        not isinstance(contract, dict)
        or not isinstance(row.get("row_id"), str)
        or not row["row_id"]
        or row.get("model") != "pyramid"
        or row.get("materialization_kind") != CANONICAL_MATERIALIZATION_KIND
        or not isinstance(width, list)
        or not width
        or any(type(item) is not int for item in width)
        or set(contract).intersection(LEGACY_TRAINING_KEYS)
    ):
        raise CompatibilityError
    legacy_config, _ = _config_paths(row)
    legacy_contract = {
        **{key: value for key, value in contract.items() if key != "external_training_binding"},
        **_validated_binding(contract.get("external_training_binding")),
        "config_path": str(legacy_config),
    }
    evidence = {
        "kind": LEGACY_MATERIALIZATION_KIND,
        "width": copy.deepcopy(width),
        "contract": legacy_contract,
    }
    return {
        **copy.deepcopy(row),
        "materialization_kind": LEGACY_MATERIALIZATION_KIND,
        "source_contract": legacy_contract,
        "source_contract_sha256": _canonical_sha(legacy_contract),
        "source_evidence_sha256": _canonical_sha(evidence),
    }

def _validated_request_rows(request):
    if not isinstance(request, dict) or set(request) != set(REQUEST_KEYS):
        raise CompatibilityError
    rows = request.get("rows")
    row_hashes = request.get("row_sha256")
    if (
        request.get("schema_version") != "stage5_measurement_request_v2"
        or not isinstance(rows, list)
        or not rows
        or not isinstance(row_hashes, dict)
        or any(not isinstance(row, dict) or set(row) != set(ROW_KEYS) for row in rows)
    ):
        raise CompatibilityError
    row_ids = tuple(row.get("row_id") for row in rows)
    if (
        any(not isinstance(row_id, str) or not row_id for row_id in row_ids)
        or len(set(row_ids)) != len(row_ids)
        or set(row_hashes) != set(row_ids)
        or any(
            row["source_contract_sha256"] != _canonical_sha(row["source_contract"])
            or row_hashes[row_id] != _canonical_sha(row)
            for row, row_id in zip(rows, row_ids, strict=True)
        )
    ):
        raise CompatibilityError
    body = {
        key: value for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    if request.get("measurement_request_sha256") != _canonical_sha(body):
        raise CompatibilityError
    return rows

def _legacy_view(request, group_id):
    rows = _validated_request_rows(request)
    bindings = tuple(
        row.get("source_contract", {}).get("external_training_binding")
        if isinstance(row, dict) and isinstance(row.get("source_contract"), dict)
        else None
        for row in rows
    )
    if any(binding != bindings[0] for binding in bindings[1:]):
        raise CompatibilityError
    projected_rows = [_legacy_row(row) for row in rows]
    selected = frozenset(
        _config_paths(row) for row in rows if row.get("group_id") == group_id
    )
    if len(selected) != 1:
        raise CompatibilityError
    projected = {
        **copy.deepcopy(request),
        "rows": projected_rows,
        "row_sha256": {
            row["row_id"]: _canonical_sha(row) for row in projected_rows
        },
    }
    body = {
        key: value for key, value in projected.items()
        if key != "measurement_request_sha256"
    }
    return {**projected, "measurement_request_sha256": _canonical_sha(body)}, next(iter(selected))

def _runtime_paths(argv):
    if (
        len(argv) != 9
        or tuple(argv[1::2]) != ("--request", "--model", "--group-id", "--gpu")
        or argv[4] != "pyramid"
        or not argv[6]
        or not argv[8].isdigit()
    ):
        raise CompatibilityError
    root = Path(os.environ["P6_HISTORY_PRIVATE_ROOT"]).resolve(strict=True)
    round_root = Path(os.environ["P6_HISTORY_ROUND_OUTPUT_ROOT"]).resolve(strict=True)
    request = Path(argv[2]).resolve(strict=True)
    implementation = (root / IMPLEMENTATION_RELATIVE).resolve(strict=True)
    implementation_cwd = (root / IMPLEMENTATION_CWD_RELATIVE).resolve(strict=True)
    if (
        not root.is_dir()
        or not round_root.is_dir()
        or not request.is_file()
        or not request.is_relative_to(round_root)
        or not implementation.is_file()
        or not os.access(implementation, os.X_OK)
        or not implementation_cwd.is_dir()
        or not implementation.is_relative_to(implementation_cwd)
        or not Path(__file__).resolve().is_relative_to(root)
    ):
        raise CompatibilityError
    return root, round_root, request, implementation, implementation_cwd

def _write_legacy_request(round_root, request):
    descriptor, spelling = tempfile.mkstemp(
        dir=round_root, prefix=".p6-legacy-source-request-", suffix=".json"
    )
    path = Path(spelling)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(request, handle, ensure_ascii=True, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path

def _publish_config(source, destination):
    content = source.read_bytes() if source.is_file() else b""
    if not content:
        raise CompatibilityError
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise CompatibilityError
        return
    descriptor, spelling = tempfile.mkstemp(dir=destination.parent, prefix=".p6-config-")
    temporary = Path(spelling)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != content:
                raise CompatibilityError
    finally:
        temporary.unlink(missing_ok=True)

def _preflight_config(source, destination):
    if source == destination and destination.exists():
        raise CompatibilityError


def main(argv):
    temporary = None
    try:
        _, round_root, request, implementation, implementation_cwd = _runtime_paths(argv)
        loaded = json.loads(
            request.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_mapping,
            parse_constant=_invalid_constant,
        )
        legacy, config_paths = _legacy_view(loaded, argv[6])
        _preflight_config(*config_paths)
        temporary = _write_legacy_request(round_root, legacy)
        child_argv = [str(implementation), "--request", str(temporary), *argv[3:]]
        child_env = {
            **{key: os.environ[key] for key in ENVIRONMENT_KEYS},
            "PYTHONPATH": str(implementation_cwd),
        }
        completed = subprocess.run(child_argv, cwd=implementation_cwd, env=child_env,
                                   shell=False, check=False)
        if completed.returncode == 0:
            _publish_config(*config_paths)
        return completed.returncode
    except (CompatibilityError, KeyError, OSError, OverflowError, TypeError, ValueError):
        print("history_execution_invalid", file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''


class P6SourceWrapperProfileError(ValueError):
    """Stable, path-free source-wrapper validation failure."""

    category = "history_execution_invalid"

    def __init__(self) -> None:
        super().__init__(self.category)


@dataclass(frozen=True)
class ValidatedSourceWrapper:
    """Validated private source marker and its fixed runtime argv contract."""

    executable: Path
    argv_shape: tuple[str, ...]
    marker_basename: str


@dataclass(frozen=True)
class SourceWrapperRenderPlan:
    """Immutable deterministic deployment plan derived from a private profile."""

    destination: Path
    wrapper_kind: Literal["repo_cwd_exec_v1"]
    implementation: Path
    implementation_cwd: Path
    expected_bytes: bytes
    expected_sha256: str


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    keys: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            _invalid()
        keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_source_wrapper_profile(path: Path) -> dict[str, Any]:
    """Load one absolute, non-symlinked private profile with duplicate-key checks."""
    if not isinstance(path, Path) or not path.is_absolute() or _contains_symlink_component(path):
        _invalid()
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > MAX_PROFILE_SIZE:
            _invalid()
        _validate_profile_privacy(resolved)
        payload = yaml.load(resolved.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except P6SourceWrapperProfileError:
        raise
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise P6SourceWrapperProfileError() from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        _invalid()
    return dict(payload)


def _validate_profile_privacy(path: Path) -> None:
    try:
        repository = REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if not _is_relative_to(path, repository):
        return
    relative = path.relative_to(repository)
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(relative)],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise P6SourceWrapperProfileError() from error
    if completed.returncode != 0:
        _invalid()


def expected_wrapper_bytes_from_profile(
    source_wrapper_profile: Path,
    *,
    history_root: Path,
) -> bytes:
    """Recompute the exact wrapper bytes from an ignored private profile."""
    profile = load_source_wrapper_profile(source_wrapper_profile)
    return _build_render_plan(profile, history_root=history_root).expected_bytes


def render_self_contained_source_wrapper(
    profile: Mapping[str, Any],
    *,
    history_root: Path,
) -> ValidatedSourceWrapper:
    """Create an absent marker or accept an exact existing generated marker."""
    plan = _build_render_plan(profile, history_root=history_root)
    _write_wrapper_script(plan)
    return _validate_rendered_wrapper(plan)


def validate_self_contained_source_wrapper(
    validated_template: ValidatedRunnerTemplate,
    *,
    source_wrapper_profile: Path,
) -> ValidatedSourceWrapper:
    """Bind a validated source stage to the exact bytes specified by its profile."""
    if not isinstance(validated_template, ValidatedRunnerTemplate):
        _invalid()
    plan = _build_render_plan(
        load_source_wrapper_profile(source_wrapper_profile),
        history_root=validated_template.history_root,
    )
    if tuple(validated_template.stage_argv) != EXPECTED_EXECUTION_STAGES:
        _invalid()
    source_argv = tuple(validated_template.stage_argv["source_materialization"])
    if (
        len(source_argv) != 3
        or tuple(source_argv[1:]) != SOURCE_TEMPLATE_ARGV_TAIL
        or Path(source_argv[0]) != plan.destination
    ):
        _invalid()
    source_component = validated_template.component_paths.get("source_materializer")
    if source_component != plan.destination:
        _invalid()
    environment = validated_template.execution_interface.get("environment")
    values = environment.get("values") if isinstance(environment, Mapping) else None
    if not isinstance(values, Mapping) or set(values) != set(EXPECTED_HISTORY_ENV_KEYS):
        _invalid()
    return _validate_rendered_wrapper(plan)


def _build_render_plan(
    profile: Mapping[str, Any],
    *,
    history_root: Path,
) -> SourceWrapperRenderPlan:
    root = _resolve_history_root(history_root)
    if not isinstance(profile, Mapping) or set(profile) != PROFILE_KEYS:
        _invalid()
    if (
        profile.get("schema_version") != PROFILE_SCHEMA_VERSION
        or profile.get("wrapper_kind") != WRAPPER_KIND
    ):
        _invalid()
    destination_relative = _validated_relative_path(profile.get("destination_relative_path"))
    implementation_relative = _validated_relative_path(profile.get("implementation_relative_path"))
    implementation_cwd_relative = _validated_relative_path(
        profile.get("implementation_cwd_relative_path")
    )
    if destination_relative.name != SOURCE_MARKER_BASENAME:
        _invalid()
    destination = _resolve_destination(root, destination_relative)
    implementation = _resolve_existing_private_path(
        root, implementation_relative, require_directory=False
    )
    implementation_cwd = _resolve_existing_private_path(
        root, implementation_cwd_relative, require_directory=True
    )
    if implementation == destination or not _is_relative_to(implementation, implementation_cwd):
        _invalid()
    expected_bytes = _wrapper_bytes(
        implementation_relative,
        implementation_cwd_relative,
    )
    return SourceWrapperRenderPlan(
        destination=destination,
        wrapper_kind=WRAPPER_KIND,
        implementation=implementation,
        implementation_cwd=implementation_cwd,
        expected_bytes=expected_bytes,
        expected_sha256=hashlib.sha256(expected_bytes).hexdigest(),
    )


def _validated_relative_path(raw: object) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or not SAFE_RELATIVE_PATH.fullmatch(raw)
        or "//" in raw
        or "\\" in raw
    ):
        _invalid()
    path = Path(raw)
    if path.is_absolute() or path == Path(".") or ".." in path.parts or ".git" in path.parts:
        _invalid()
    return path


def _resolve_history_root(raw: Path) -> Path:
    if not isinstance(raw, Path) or not raw.is_absolute() or _contains_symlink_component(raw):
        _invalid()
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if not root.is_dir() or _git_root_for(root) != root:
        _invalid()
    return root


def _resolve_existing_private_path(
    root: Path,
    relative: Path,
    *,
    require_directory: bool,
) -> Path:
    candidate = root / relative
    if _contains_symlink_component(candidate):
        _invalid()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if not _is_relative_to(resolved, root) or _git_root_for(resolved) != root:
        _invalid()
    if require_directory:
        if not resolved.is_dir():
            _invalid()
    elif not resolved.is_file() or not os.access(resolved, os.X_OK):
        _invalid()
    return resolved


def _resolve_destination(root: Path, relative: Path) -> Path:
    candidate = root / relative
    if _contains_symlink_component(candidate):
        _invalid()
    try:
        destination = candidate.resolve(strict=False)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if not _is_relative_to(destination, root) or destination == root:
        _invalid()
    if candidate.exists() and (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve(strict=True) != destination
    ):
        _invalid()
    return destination


def _wrapper_bytes(implementation: Path, implementation_cwd: Path) -> bytes:
    body = _WRAPPER_TEMPLATE.replace(
        "__IMPLEMENTATION_RELATIVE__", repr(implementation.as_posix())
    ).replace("__IMPLEMENTATION_CWD_RELATIVE__", repr(implementation_cwd.as_posix()))
    return body.encode("utf-8")


def _write_wrapper_script(plan: SourceWrapperRenderPlan) -> None:
    destination = plan.destination
    if destination.exists() or destination.is_symlink():
        _validate_rendered_wrapper(plan)
        destination.chmod(0o700)
        return
    try:
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    if _contains_symlink_component(destination):
        _invalid()
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o700,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(plan.expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        destination.chmod(0o700)
    except FileExistsError:
        _validate_rendered_wrapper(plan)
    except OSError as error:
        if created:
            destination.unlink(missing_ok=True)
        raise P6SourceWrapperProfileError() from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_rendered_wrapper(
    plan: SourceWrapperRenderPlan,
) -> ValidatedSourceWrapper:
    destination = plan.destination
    if (
        destination.is_symlink()
        or not destination.is_file()
        or not os.access(destination, os.X_OK)
        or destination.name != SOURCE_MARKER_BASENAME
    ):
        _invalid()
    try:
        actual_bytes = destination.read_bytes()
    except OSError as error:
        raise P6SourceWrapperProfileError() from error
    actual_sha256 = hashlib.sha256(actual_bytes).hexdigest()
    if actual_sha256 != plan.expected_sha256 or actual_bytes != plan.expected_bytes:
        _invalid()
    return ValidatedSourceWrapper(
        executable=destination,
        argv_shape=SOURCE_RUNTIME_ARGV_SHAPE,
        marker_basename=SOURCE_MARKER_BASENAME,
    )


def _git_root_for(path: Path) -> Path:
    working_directory = path if path.is_dir() else path.parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(working_directory), "rev-parse", "--show-toplevel"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise P6SourceWrapperProfileError() from error
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) != 1 or not lines[0]:
        _invalid()
    try:
        return Path(lines[0]).resolve(strict=True)
    except OSError as error:
        raise P6SourceWrapperProfileError() from error


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink() for component in (path, *path.parents)
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _invalid() -> None:
    raise P6SourceWrapperProfileError()
