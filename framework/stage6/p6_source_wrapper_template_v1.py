"""Generated P6 source-wrapper script template."""

from __future__ import annotations

from pathlib import Path


WRAPPER_TEMPLATE = '''#!/usr/bin/python3
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


def render_wrapper_template(
    *,
    implementation: Path,
    implementation_cwd: Path,
    project_python: Path | None,
) -> str:
    """Render byte-compatible v1 or the explicit v2 runtime variant."""
    body = WRAPPER_TEMPLATE.replace(
        "__IMPLEMENTATION_RELATIVE__", repr(implementation.as_posix())
    ).replace("__IMPLEMENTATION_CWD_RELATIVE__", repr(implementation_cwd.as_posix()))
    if project_python is None:
        return body
    project_literal = str(project_python)
    body = _replace_once(
        body,
        f"IMPLEMENTATION_CWD_RELATIVE = {implementation_cwd.as_posix()!r}\n",
        (
            f"IMPLEMENTATION_CWD_RELATIVE = {implementation_cwd.as_posix()!r}\n"
            f"PROJECT_PYTHON = {project_literal!r}\n"
        ),
    )
    body = _replace_once(
        body,
        "_config_paths(row) for row in rows if row.get(\"group_id\") == group_id",
        (
            "(*_config_paths(row), _training_marker_path(row)) "
            "for row in rows if row.get(\"group_id\") == group_id"
        ),
    )
    body = _replace_once(body, "def _runtime_paths(argv):\n", _V2_RUNTIME_HELPERS)
    body = _replace_once(
        body,
        "        legacy, config_paths = _legacy_view(loaded, argv[6])\n"
        "        _preflight_config(*config_paths)\n",
        (
            "        legacy, selected_outputs = _legacy_view(loaded, argv[6])\n"
            "        legacy_config, canonical_config, marker = selected_outputs\n"
            "        _preflight_config(legacy_config, canonical_config)\n"
            "        _preflight_marker(marker)\n"
            "        marker.parent.mkdir(parents=True, exist_ok=True)\n"
        ),
    )
    body = _replace_once(
        body,
        '            "PYTHONPATH": str(implementation_cwd),\n',
        (
            '            "PYTHONPATH": str(implementation_cwd),\n'
            '            "PY": PROJECT_PYTHON,\n'
            '            "PATH": f"{Path(PROJECT_PYTHON).parent}:/usr/bin:/bin",\n'
        ),
    )
    return _replace_once(
        body,
        (
            "        if completed.returncode == 0:\n"
            "            _publish_config(*config_paths)\n"
        ),
        (
            "        if completed.returncode == 0:\n"
            "            _publish_config(legacy_config, canonical_config)\n"
            "        else:\n"
            "            _cleanup_failed_outputs(\n"
            "                marker, legacy_config, canonical_config\n"
            "            )\n"
        ),
    )


def _replace_once(body: str, old: str, new: str) -> str:
    if body.count(old) != 1:
        raise ValueError("source wrapper template drift")
    return body.replace(old, new)


_V2_RUNTIME_HELPERS = '''def _training_marker_path(row):
    contract = row["source_contract"]
    return _canonical_output_path(contract.get("training_done_marker"))

def _preflight_marker(marker):
    if marker.exists():
        raise CompatibilityError

def _cleanup_failed_outputs(marker, legacy_config, canonical_config):
    marker.unlink(missing_ok=True)
    if legacy_config == canonical_config:
        canonical_config.unlink(missing_ok=True)

def _validate_project_python_runtime():
    project_python = Path(PROJECT_PYTHON)
    python_launcher = project_python.parent / "python"
    try:
        resolved = project_python.resolve(strict=True)
        launcher_resolved = python_launcher.resolve(strict=True)
    except OSError:
        raise CompatibilityError
    if (
        resolved != project_python
        or not project_python.is_file()
        or not os.access(project_python, os.X_OK)
        or not python_launcher.is_file()
        or not os.access(python_launcher, os.X_OK)
        or launcher_resolved != project_python
    ):
        raise CompatibilityError

def _runtime_paths(argv):
    _validate_project_python_runtime()
'''
