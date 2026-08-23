"""Execution-level coverage for the P6 historical source compatibility view."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_source_wrapper_profile_v1 import (
    render_self_contained_source_wrapper,
)
from tests.p6_source_wrapper_support import (
    source_bridge_output_paths,
    source_bridge_request,
)


SOURCE_MARKER = "stage5_materialize_round_sources_v1.sh"
LEGACY_TRAINING_KEYS = (
    "training_required",
    "training_source_kind",
    "dataset_root",
    "base_checkpoint_path",
    "base_checkpoint_dir",
    "base_checkpoint_sha256",
    "pyramid_config_path",
    "pyramid_config_sha256",
    "training_parameters",
    "training_epoches",
    "groups",
    "width_per_group",
)
OUTPUT_PATH_KEYS = (
    "checkpoint_path",
    "checkpoint_dir",
    "config_path",
    "training_done_marker",
    "onnx_path",
    "onnx_report_path",
    "calibration_root",
    "calibration_npz",
    "calibration_summary",
    "trt_calibration_dir",
    "source_done_marker",
)
DIRECTORY_OUTPUT_KEYS = frozenset(
    {"checkpoint_dir", "calibration_root", "trt_calibration_dir"}
)
GENERATED_CONFIG = b"fixture:config_path\n"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _private_git_root(tmp_path: Path) -> Path:
    root = tmp_path / "private-history"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def _profile() -> dict[str, str]:
    return {
        "schema_version": "p6_private_source_wrapper_profile_v1",
        "wrapper_kind": "repo_cwd_exec_v1",
        "destination_relative_path": f"documented-stage5-chain/{SOURCE_MARKER}",
        "implementation_relative_path": (
            "private-relocated-history-repo/bin/"
            "stage5_materialize_round_sources_v1.original.sh"
        ),
        "implementation_cwd_relative_path": "private-relocated-history-repo",
    }


def _external_binding(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "p6_external_training_binding_v1",
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
        "dataset_root": str(tmp_path / "operator-assets" / "dataset"),
        "base_checkpoint_path": str(tmp_path / "operator-assets" / "base.ckpt"),
        "base_checkpoint_sha256": "a" * 64,
        "pyramid_config_path": str(tmp_path / "operator-assets" / "pyramid.yaml"),
        "pyramid_config_sha256": "b" * 64,
        "training_parameters": {
            "training_mode": "finetune_selected_width",
            "epochs": 8,
            "target_epoch": 31,
            "seed": 0,
            "optimizer": "adam",
            "learning_rate": 0.002,
            "batch_size": 2,
            "dataset_split": "trainval_coptv2x",
            "checkpoint_selection": "best_ap70",
            "freeze_policy": "pyramid_backbone_partial",
            "groups": 3,
            "width_per_group": 5,
            "base_stage_widths": [3, 5, 7],
        },
    }


def _canonical_request(
    tmp_path: Path, *, config_at_checkpoint: bool = False
) -> dict[str, Any]:
    artifact = tmp_path / "private-output" / "materialized" / "pyramid-16-32-64"
    return source_bridge_request(
        _external_binding(tmp_path),
        source_contract_fields=source_bridge_output_paths(
            artifact, config_at_checkpoint=config_at_checkpoint
        ),
    )


def _write_relocated_materializer(
    history_root: Path, *, returncode: int = 0
) -> Path:
    repository = history_root / "private-relocated-history-repo"
    module = repository / "framework" / "stage5" / "measurement_plan_v2.py"
    module.parent.mkdir(parents=True)
    (repository / "framework" / "__init__.py").write_text("", encoding="utf-8")
    (module.parent / "__init__.py").write_text("", encoding="utf-8")
    module.write_text("IMPORT_SENTINEL = 'closure-local'\n", encoding="utf-8")
    implementation = (
        repository / "bin" / "stage5_materialize_round_sources_v1.original.sh"
    )
    implementation.parent.mkdir()
    implementation.write_text(
        f"#!{sys.executable}\n"
        "import hashlib\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "from framework.stage5.measurement_plan_v2 import IMPORT_SENTINEL\n"
        "\n"
        "def sha(payload):\n"
        "    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, "
        "separators=(',', ':')).encode('utf-8')\n"
        "    return hashlib.sha256(encoded).hexdigest()\n"
        "\n"
        "request_path = Path(sys.argv[sys.argv.index('--request') + 1])\n"
        "request = json.loads(request_path.read_text(encoding='utf-8'))\n"
        f"legacy_keys = {LEGACY_TRAINING_KEYS!r}\n"
        f"output_keys = {OUTPUT_PATH_KEYS!r}\n"
        f"directory_keys = {tuple(sorted(DIRECTORY_OUTPUT_KEYS))!r}\n"
        "for row in request['rows']:\n"
        "    contract = row['source_contract']\n"
        "    assert set(contract) == set(legacy_keys) | set(output_keys)\n"
        "    assert 'external_training_binding' not in contract\n"
        "    assert row['model'] == 'pyramid'\n"
        "    assert isinstance(row['width'], list) and row['width']\n"
        "    assert all(type(item) is int for item in row['width'])\n"
        "    assert row['materialization_kind'] == 'pyramid_prepare_train_export'\n"
        "    assert row['source_contract_sha256'] == sha(contract)\n"
        "    evidence = {\n"
        "        'kind': 'pyramid_prepare_train_export',\n"
        "        'width': row['width'],\n"
        "        'contract': contract,\n"
        "    }\n"
        "    assert row['source_evidence_sha256'] == sha(evidence)\n"
        "    assert request['row_sha256'][row['row_id']] == sha(row)\n"
        "    expected_config = Path(contract['checkpoint_dir']) / 'config.yaml'\n"
        "    assert Path(contract['config_path']) == expected_config\n"
        "    for key in output_keys:\n"
        "        target = Path(contract[key])\n"
        "        if key in directory_keys:\n"
        "            target.mkdir(parents=True, exist_ok=True)\n"
        "        else:\n"
        "            target.parent.mkdir(parents=True, exist_ok=True)\n"
        "            target.write_bytes(b'fixture:' + key.encode() + b'\\n')\n"
        "    assert expected_config.is_file() and expected_config.stat().st_size > 0\n"
        "body = {key: value for key, value in request.items() "
        "if key != 'measurement_request_sha256'}\n"
        "assert request['measurement_request_sha256'] == sha(body)\n"
        "assert IMPORT_SENTINEL == 'closure-local'\n"
        "round_root = Path(os.environ['P6_HISTORY_ROUND_OUTPUT_ROOT'])\n"
        "observed = {\n"
        "    'request_path': str(request_path),\n"
        "    'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "    'training': {key: request['rows'][0]['source_contract'][key] "
        "for key in legacy_keys},\n"
        "    'contract': request['rows'][0]['source_contract'],\n"
        "    'config_path': request['rows'][0]['source_contract']['config_path'],\n"
        "    'materialization_kind': request['rows'][0]['materialization_kind'],\n"
        "    'source_evidence_sha256': request['rows'][0]['source_evidence_sha256'],\n"
        "}\n"
        "(round_root / 'implementation-observed.json').write_text(\n"
        "    json.dumps(observed, sort_keys=True), encoding='utf-8'\n"
        ")\n"
        f"raise SystemExit({returncode})\n",
        encoding="utf-8",
    )
    implementation.chmod(0o700)
    return implementation


def _runtime_env(history_root: Path, round_root: Path) -> dict[str, str]:
    task_state = round_root / "state" / "task-state.json"
    task_state.parent.mkdir()
    task_state.write_text("{}", encoding="utf-8")
    return {
        "CUDA_VISIBLE_DEVICES": "101,103,107",
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(history_root),
        "P6_HISTORY_TASK_STATE": str(task_state),
        "P6_HISTORY_ROUND_OUTPUT_ROOT": str(round_root),
    }


def _rehash_request(request: dict[str, Any]) -> None:
    for row in request["rows"]:
        row["source_contract_sha256"] = _sha(row["source_contract"])
    request["row_sha256"] = {
        row["row_id"]: _sha(row) for row in request["rows"]
    }
    body = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request["measurement_request_sha256"] = _sha(body)


def _run_wrapper(
    wrapper: Path,
    request_path: Path,
    round_root: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(wrapper),
            "--request",
            str(request_path),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|16x32x64",
            "--gpu",
            "101",
        ],
        cwd=round_root,
        env=env,
        shell=False,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("config_at_checkpoint", (False, True))
def test_wrapper_projects_nested_training_for_relocated_implementation(
    tmp_path: Path,
    config_at_checkpoint: bool,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(
        tmp_path, config_at_checkpoint=config_at_checkpoint
    )
    request_path = round_root / "measurement-request.json"
    original_bytes = (json.dumps(canonical, sort_keys=True) + "\n").encode("utf-8")
    request_path.write_bytes(original_bytes)
    env = _runtime_env(history_root, round_root)

    completed = _run_wrapper(wrapper, request_path, round_root, env)

    assert completed.returncode == 0, completed.stderr
    assert set(env) == set(EXPECTED_HISTORY_ENV_KEYS)
    assert request_path.read_bytes() == original_bytes
    observed = json.loads(
        (round_root / "implementation-observed.json").read_text(encoding="utf-8")
    )
    assert observed["request_path"] != str(request_path)
    assert Path(observed["request_path"]).parent == round_root
    assert not Path(observed["request_path"]).exists()
    assert observed["pythonpath"] == str(
        history_root / "private-relocated-history-repo"
    )
    binding = canonical["rows"][0]["source_contract"]["external_training_binding"]
    assert observed["training"] == {
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
        "dataset_root": str(tmp_path / "operator-assets" / "dataset"),
        "base_checkpoint_path": str(tmp_path / "operator-assets" / "base.ckpt"),
        "base_checkpoint_dir": str(tmp_path / "operator-assets"),
        "base_checkpoint_sha256": "a" * 64,
        "pyramid_config_path": str(tmp_path / "operator-assets" / "pyramid.yaml"),
        "pyramid_config_sha256": "b" * 64,
        "training_parameters": binding["training_parameters"],
        "training_epoches": 31,
        "groups": 3,
        "width_per_group": 5,
    }
    legacy_contract = observed["contract"]
    assert observed["materialization_kind"] == "pyramid_prepare_train_export"
    assert observed["source_evidence_sha256"] == _sha(
        {
            "kind": "pyramid_prepare_train_export",
            "width": [16, 32, 64],
            "contract": legacy_contract,
        }
    )
    canonical_contract = canonical["rows"][0]["source_contract"]
    expected_legacy_config = Path(canonical_contract["checkpoint_dir"]) / "config.yaml"
    assert observed["config_path"] == str(expected_legacy_config)
    for key in OUTPUT_PATH_KEYS:
        output = Path(canonical_contract[key])
        if key in DIRECTORY_OUTPUT_KEYS:
            assert output.is_dir()
        else:
            assert output.is_file() and output.stat().st_size > 0
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))
    assert canonical == _canonical_request(
        tmp_path, config_at_checkpoint=config_at_checkpoint
    )


def test_wrapper_accepts_same_group_mixed_q_rows_with_one_config_pair(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    int8 = copy.deepcopy(canonical["rows"][0])
    int8.update(
        row_id="row-1",
        manifest_job_id="row-1",
        genome=[16, 32, 64, "int8"],
        strategy_id="q=int8",
        q_mode="int8",
    )
    canonical["rows"].append(int8)
    canonical["batch_size"] = 2
    _rehash_request(canonical)
    request_path = round_root / "measurement-request.json"
    original_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
    request_path.write_bytes(original_bytes)

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 0, completed.stderr
    assert request_path.read_bytes() == original_bytes
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


@pytest.mark.parametrize(
    ("key", "value"),
    (("onnx_path", None), ("calibration_npz", "relative/cache.npz")),
)
def test_wrapper_rejects_incomplete_or_noncanonical_output_paths_before_child(
    tmp_path: Path,
    key: str,
    value: str | None,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    contract = canonical["rows"][0]["source_contract"]
    if value is None:
        contract.pop(key)
    else:
        contract[key] = value
    _rehash_request(canonical)
    request_path = round_root / "measurement-request.json"
    request_path.write_text(json.dumps(canonical), encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert completed.stderr == "history_execution_invalid\n"
    assert not (round_root / "implementation-observed.json").exists()
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


def test_wrapper_rejects_legacy_config_collision_with_another_output_before_child(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    contract = canonical["rows"][0]["source_contract"]
    contract["onnx_path"] = str(Path(contract["checkpoint_dir"]) / "config.yaml")
    _rehash_request(canonical)
    request_path = round_root / "measurement-request.json"
    request_path.write_text(json.dumps(canonical), encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert completed.stderr == "history_execution_invalid\n"
    assert not (round_root / "implementation-observed.json").exists()
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


def _mutate_missing_binding(request: dict[str, Any]) -> None:
    request["rows"][0]["source_contract"].pop("external_training_binding")


def _mutate_non_mapping_binding(request: dict[str, Any]) -> None:
    request["rows"][0]["source_contract"]["external_training_binding"] = []


def _mutate_incomplete_binding(request: dict[str, Any]) -> None:
    request["rows"][0]["source_contract"]["external_training_binding"].pop(
        "training_parameters"
    )


def _mutate_invalid_parameters(request: dict[str, Any]) -> None:
    binding = request["rows"][0]["source_contract"]["external_training_binding"]
    binding["training_parameters"]["epochs"] = True


def _mutate_flat_collision(request: dict[str, Any]) -> None:
    request["rows"][0]["source_contract"]["training_required"] = True


def _mutate_unsupported_model(request: dict[str, Any]) -> None:
    request["rows"][0]["model"] = "unsupported-model"


def _mutate_unsupported_kind(request: dict[str, Any]) -> None:
    request["rows"][0]["materialization_kind"] = "unsupported-kind"


def _mutate_invalid_width_type(request: dict[str, Any]) -> None:
    request["rows"][0]["width"] = [16, True, 64]


def _mutate_stale_contract_hash(request: dict[str, Any]) -> None:
    request["rows"][0]["source_contract_sha256"] = "0" * 64


def _mutate_stale_row_hash(request: dict[str, Any]) -> None:
    request["row_sha256"]["row-0"] = "0" * 64


def _mutate_stale_request_hash(request: dict[str, Any]) -> None:
    request["measurement_request_sha256"] = "0" * 64


def _mutate_missing_row_hashes(request: dict[str, Any]) -> None:
    request.pop("row_sha256")


def _mutate_duplicate_row_id(request: dict[str, Any]) -> None:
    duplicate = copy.deepcopy(request["rows"][0])
    request["rows"].append(duplicate)
    request["row_sha256"] = {"row-0": _sha(duplicate)}
    body = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request["measurement_request_sha256"] = _sha(body)


def _mutate_missing_row_identity(request: dict[str, Any]) -> None:
    request["rows"][0].pop("source_contract_sha256")


@pytest.mark.parametrize(
    "mutate",
    (
        _mutate_missing_binding,
        _mutate_non_mapping_binding,
        _mutate_incomplete_binding,
        _mutate_invalid_parameters,
        _mutate_flat_collision,
    ),
)
def test_wrapper_rejects_invalid_nested_training_before_implementation(
    tmp_path: Path,
    mutate: Any,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    request = _canonical_request(tmp_path)
    mutate(request)
    _rehash_request(request)
    request_path = round_root / "measurement-request.json"
    original_bytes = json.dumps(request, sort_keys=True).encode("utf-8")
    request_path.write_bytes(original_bytes)

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert completed.stderr == "history_execution_invalid\n"
    assert not (round_root / "implementation-observed.json").exists()
    assert request_path.read_bytes() == original_bytes
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


@pytest.mark.parametrize(
    "mutate",
    (
        _mutate_unsupported_model,
        _mutate_unsupported_kind,
        _mutate_invalid_width_type,
    ),
)
def test_wrapper_rejects_unsupported_pyramid_row_before_implementation(
    tmp_path: Path,
    mutate: Any,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    request = _canonical_request(tmp_path)
    mutate(request)
    _rehash_request(request)
    request_path = round_root / "measurement-request.json"
    original_bytes = json.dumps(request, sort_keys=True).encode("utf-8")
    request_path.write_bytes(original_bytes)

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert completed.stderr == "history_execution_invalid\n"
    assert not (round_root / "implementation-observed.json").exists()
    assert request_path.read_bytes() == original_bytes
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


@pytest.mark.parametrize(
    "mutate",
    (
        _mutate_stale_contract_hash,
        _mutate_stale_row_hash,
        _mutate_stale_request_hash,
        _mutate_missing_row_hashes,
        _mutate_duplicate_row_id,
        _mutate_missing_row_identity,
    ),
)
def test_wrapper_rejects_stale_canonical_identity_before_implementation(
    tmp_path: Path,
    mutate: Any,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    request = _canonical_request(tmp_path)
    mutate(request)
    request_path = round_root / "measurement-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert completed.stderr == "history_execution_invalid\n"
    assert not (round_root / "implementation-observed.json").exists()
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


@pytest.mark.parametrize("nonfinite", (float("inf"), float("nan")))
def test_wrapper_rejects_nonfinite_training_number_before_implementation(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    request = _canonical_request(tmp_path)
    training = request["rows"][0]["source_contract"]["external_training_binding"]
    training["training_parameters"]["learning_rate"] = nonfinite
    _rehash_request(request)
    request_path = round_root / "measurement-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert not (round_root / "implementation-observed.json").exists()
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


def test_wrapper_rejects_overflowing_json_number_before_implementation(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    request_path = round_root / "measurement-request.json"
    valid = json.dumps(_canonical_request(tmp_path))
    overflowing = valid.replace('"learning_rate": 0.002', '"learning_rate": 1e10000')
    assert overflowing != valid
    request_path.write_text(overflowing, encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert not (round_root / "implementation-observed.json").exists()
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


@pytest.mark.parametrize(
    ("existing", "expected_returncode"),
    ((GENERATED_CONFIG, 0), (b"operator-conflict\n", 2)),
)
def test_wrapper_preserves_matching_config_and_rejects_conflict(
    tmp_path: Path,
    existing: bytes,
    expected_returncode: int,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    canonical_config = Path(canonical["rows"][0]["source_contract"]["config_path"])
    canonical_config.parent.mkdir(parents=True)
    canonical_config.write_bytes(existing)
    request_path = round_root / "measurement-request.json"
    original_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
    request_path.write_bytes(original_bytes)

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == expected_returncode
    assert canonical_config.read_bytes() == existing
    assert (round_root / "implementation-observed.json").is_file()
    assert request_path.read_bytes() == original_bytes
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


def test_wrapper_rejects_preexisting_same_path_config_before_child(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path, config_at_checkpoint=True)
    config = Path(canonical["rows"][0]["source_contract"]["config_path"])
    config.parent.mkdir(parents=True)
    config.write_bytes(b"operator-conflict\n")
    request_path = round_root / "measurement-request.json"
    request_path.write_text(json.dumps(canonical), encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert config.read_bytes() == b"operator-conflict\n"
    assert not (round_root / "implementation-observed.json").exists()
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


def test_wrapper_rejects_config_path_outside_checkpoint_artifact_before_child(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    canonical["rows"][0]["source_contract"]["config_path"] = str(
        tmp_path / "outside-artifact" / "source-config.json"
    )
    _rehash_request(canonical)
    request_path = round_root / "measurement-request.json"
    original_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
    request_path.write_bytes(original_bytes)

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert completed.stderr == "history_execution_invalid\n"
    assert not (round_root / "implementation-observed.json").exists()
    assert request_path.read_bytes() == original_bytes
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


def test_wrapper_propagates_implementation_failure_and_cleans_legacy_view(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root, returncode=23)
    wrapper = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    ).executable
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    request_path = round_root / "measurement-request.json"
    original_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
    request_path.write_bytes(original_bytes)

    completed = _run_wrapper(
        wrapper,
        request_path,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 23
    observed = json.loads(
        (round_root / "implementation-observed.json").read_text(encoding="utf-8")
    )
    assert not Path(observed["request_path"]).exists()
    contract = canonical["rows"][0]["source_contract"]
    assert (Path(contract["checkpoint_dir"]) / "config.yaml").is_file()
    assert not Path(contract["config_path"]).exists()
    assert request_path.read_bytes() == original_bytes
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))
