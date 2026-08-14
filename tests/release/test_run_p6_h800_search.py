from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from framework.stage2.canonical_search_v3 import build_capability_profile

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide POSIX limits.
    resource = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/release/run_p6_h800_search.py"


def _public_contract() -> dict[str, Any]:
    return {
        "schema_version": "p6_h800_coptv2x_search_contract_v2",
        "search_id": "p6-pyramid-h800-tvm",
        "target": "h800",
        "target_model": "pyramid",
        "execution_backend": "tvm_auto",
        "seed": 73,
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
        "configuration_label": "p6-pyramid-h800-tvm",
        "candidate_space_label": "coptv2x-pyramid-width-grid-v1",
        "metric_names": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }


def _profile() -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id="h800-tvm-auto",
        hardware_target="h800",
        compiler_fingerprint="a" * 64,
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )


def _graph(group_id: str, width: list[int]) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": list(width),
        "conv_count": 27,
        "conv_macs": float(width[0] * width[1] * width[2]),
        "group_conv_count": 3,
    }


def _gold176() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for index in range(176):
        width = [16 + (index % 7) * 8, 32 + (index % 8) * 8, 64 + (index % 9) * 8]
        group_id = f"gold-{index:03d}"
        q_mode = "int8" if index % 2 else "fp16"
        graphs.append(_graph(group_id, width))
        rows.append(
            {
                "manifest_job_id": f"{group_id}|q={q_mode}|profile=h800-tvm-auto",
                "row_id": f"{group_id}|q={q_mode}|profile=h800-tvm-auto",
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "dispatch_key": "tvm_auto",
                "capability_profile_id": "h800-tvm-auto",
                "q_mode": q_mode,
                "latency_ms": 2.0 + index * 0.01,
                "energy_j": 0.5 + index * 0.005,
                "ap30": 0.90,
                "ap50": 0.80,
                "ap70": 0.70 - index * 0.0001,
                "terminal_status": "measured_success_gold",
                "training_source": "initial_coldstart",
            }
        )
    return rows, graphs


def _closure() -> dict[str, Any]:
    return {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": 176},
        "frozen_holdout": {"groups": []},
    }


def _source_group(group_id: str, width: list[int]) -> dict[str, Any]:
    evidence_sha = hashlib.sha256(f"source:{group_id}".encode()).hexdigest()
    source_contract = {
        "schema_version": "stage5_source_contract_v1",
        "group_id": group_id,
        "model": "pyramid",
        "width": width,
        "artifact_id": f"fixture-{group_id}",
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "materialization_scope": "synthetic_fixture",
    }
    contract_sha = hashlib.sha256(
        json.dumps(
            source_contract,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "group_id": group_id,
        "model": "pyramid",
        "width": width,
        "source_status": "ready",
        "source_evidence_sha256": evidence_sha,
        "source_contract": source_contract,
        "source_contract_sha256": contract_sha,
        "materialization_kind": "local_pyramid_tvm",
        "source_evidence_kind": "local_synthetic",
        "graph_features": _graph(group_id, width),
    }


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    return path


def _write_source_template(path: Path) -> Path:
    groups = [
        _source_group(
            f"pyramid|{first}x{second}x{third}",
            [first, second, third],
        )
        for first in range(16, 72, 8)
        for second in range(32, 88, 8)
        for third in range(64, 120, 8)
    ]
    assert len(groups) == 343
    path.write_text(
        json.dumps(
            {
                "schema_version": "stage5_candidate_source_registry_v1",
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fake_source_registry_adapter(path: Path) -> Path:
    path.write_text(
        """from __future__ import annotations
from pathlib import Path
import shutil
import sys

output_root, template_path, output_path = map(Path, sys.argv[1:])
del output_root
shutil.copyfile(template_path, output_path)
""",
        encoding="utf-8",
    )
    return path


def _write_noop_source_registry_adapter(path: Path) -> Path:
    path.write_text(
        """from __future__ import annotations
import sys

del sys.argv
""",
        encoding="utf-8",
    )
    return path


def _write_fake_measurement_adapter(path: Path) -> Path:
    path.write_text(
        """from __future__ import annotations
import json
from pathlib import Path
import sys

request_path, feedback_path, round_root, call_log, mode = sys.argv[1:]
del round_root
request = json.loads(Path(request_path).read_text(encoding="utf-8"))
with Path(call_log).open("a", encoding="utf-8") as handle:
    handle.write("measure\\n")
if mode == "fail":
    sys.stderr.write("PRIVATE_ADAPTER_STDERR\\n")
    raise SystemExit(9)
feedback = {
    "schema_version": "p6_h800_coptv2x_feedback_v2",
    "measurement_request_sha256": request["measurement_request_sha256"],
    "rows": [
        {
            "row_id": row["row_id"],
            "terminal_status": "measured_success_gold",
            "latency_ms": 2.0,
            "energy_j": 0.5,
            "ap30": 0.9,
            "ap50": 0.8,
            "ap70": 0.7,
        }
        for row in request["rows"]
    ],
}
Path(feedback_path).write_text(json.dumps(feedback), encoding="utf-8")
""",
        encoding="utf-8",
    )
    return path


def _cli_fixture(tmp_path: Path, *, mode: str = "success") -> dict[str, Path]:
    contract_path = _write_yaml(tmp_path / "contract.yaml", _public_contract())
    gold_rows, gold_graphs = _gold176()
    input_payloads = {
        "gold176_rows": gold_rows,
        "gold176_graph_features": gold_graphs,
        "capability_profiles": [_profile()],
        "closure": _closure(),
    }
    input_paths: dict[str, str] = {}
    for name, payload in input_payloads.items():
        input_path = tmp_path / f"{name}.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        input_paths[name] = str(input_path)

    asset_paths: dict[str, str] = {}
    for label in ("training-data", "model-init", "toolchain"):
        asset_path = tmp_path / label
        asset_path.mkdir()
        asset_paths[label] = str(asset_path)

    source_template = _write_source_template(tmp_path / "source_template.json")
    source_adapter = _write_fake_source_registry_adapter(tmp_path / "fake_registry.py")
    measurement_adapter = _write_fake_measurement_adapter(tmp_path / "fake_measurement.py")
    call_log = tmp_path / "PRIVATE_CALL_LOG.txt"
    output_root = tmp_path / "PRIVATE_LOCAL_OUTPUT"
    local_path = _write_yaml(
        tmp_path / "local.yaml",
        {
            "schema_version": "p6_h800_coptv2x_local_v2",
            "target": "h800",
            "asset_paths": asset_paths,
            "local_input_paths": input_paths,
            "source_registry_step": {
                "name": "build_source_registry",
                "argv": [
                    sys.executable,
                    str(source_adapter),
                    "{local_output_root}",
                    str(source_template),
                    "{source_registry_json}",
                ],
            },
            "measurement_step": {
                "name": "measure_batch",
                "argv": [
                    sys.executable,
                    str(measurement_adapter),
                    "{measurement_request}",
                    "{feedback_json}",
                    "{round_output_root}",
                    str(call_log),
                    mode,
                ],
            },
            "local_output_root": str(output_root),
        },
    )
    return {
        "contract": contract_path,
        "local": local_path,
        "call_log": call_log,
        "output_root": output_root,
    }


def _run_cli(
    paths: Mapping[str, Path],
    *extra: str,
    env: Mapping[str, str] | None = None,
    preexec_fn: object | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--local-config",
            str(paths["local"]),
            "--code-revision",
            "test-revision",
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
        preexec_fn=preexec_fn,
    )


def test_cli_runs_v2_loop_without_public_summary_and_keeps_outputs_local(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = _run_cli(paths)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "completed\n"
    assert "public-summary" not in result.stdout + result.stderr
    state = json.loads((paths["output_root"] / "state.json").read_text(encoding="utf-8"))
    serialized_state = json.dumps(state, sort_keys=True)
    assert state["status"] == "completed"
    assert state["completed_rounds"] == 4
    assert state["measured_candidate_count"] == 16
    assert str(tmp_path) not in serialized_state
    assert "candidate-" not in serialized_state
    assert paths["call_log"].read_text(encoding="utf-8").splitlines() == [
        "measure",
        "measure",
        "measure",
        "measure",
    ]


def test_cli_rejects_legacy_public_summary_argument(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = _run_cli(paths, "--public-summary", str(tmp_path / "summary.json"))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not paths["call_log"].exists()


def test_cli_requires_explicit_local_config_without_starting_an_adapter(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--code-revision",
            "test-revision",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not paths["call_log"].exists()


def test_cli_fails_closed_for_an_invalid_local_contract(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local["measurement_step"]["argv"] = ["bash", "-c", "private"]
    _write_yaml(paths["local"], local)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()


def test_cli_reports_an_unsafe_local_output_boundary_as_a_contract_error(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)
    protected_directory = tmp_path / "protected-output"
    protected_directory.mkdir()
    paths["output_root"].symlink_to(protected_directory, target_is_directory=True)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()


@pytest.mark.parametrize("document", ["contract", "local"])
def test_cli_reports_invalid_utf8_as_a_stable_contract_error(
    tmp_path: Path,
    document: str,
) -> None:
    paths = _cli_fixture(tmp_path)
    paths[document].write_bytes(b"\xff\xfe")

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert "Traceback" not in result.stderr
    assert not paths["call_log"].exists()


def test_cli_reports_a_local_measurement_failure_without_adapter_details(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path, mode="fail")

    result = _run_cli(paths)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "execution_failed\n"
    assert paths["call_log"].read_text(encoding="utf-8").splitlines() == ["measure"]
    state_text = (paths["output_root"] / "state.json").read_text(encoding="utf-8")
    state = json.loads(state_text)
    assert state == {
        "code_revision": "test-revision",
        "completed_rounds": 0,
        "failure_code": "command_failed",
        "measured_candidate_count": 0,
        "schema_version": "p6_h800_coptv2x_local_state_v2",
        "status": "failed",
    }
    for sentinel in ("PRIVATE_ADAPTER_STDERR", str(tmp_path), "candidate-", "checkpoint"):
        assert sentinel.lower() not in (result.stdout + result.stderr + state_text).lower()


@pytest.mark.skipif(
    resource is None or not hasattr(resource, "RLIMIT_FSIZE"),
    reason="requires a POSIX process file-size limit",
)
def test_cli_normalizes_a_local_record_write_error(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)
    paths["output_root"].mkdir()
    _write_source_template(paths["output_root"] / "source_registry.json")
    noop_source = _write_noop_source_registry_adapter(tmp_path / "noop_source.py")
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local["source_registry_step"]["argv"] = [
        sys.executable,
        str(noop_source),
        "{local_output_root}",
        "{source_registry_json}",
    ]
    _write_yaml(paths["local"], local)
    assert resource is not None
    _, hard_limit = resource.getrlimit(resource.RLIMIT_FSIZE)

    def prevent_controller_record_writes() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, hard_limit))

    result = _run_cli(
        paths,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("COV_CORE_", "COVERAGE_PROCESS_"))
            },
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONWARNINGS": "ignore",
        },
        preexec_fn=prevent_controller_record_writes,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "execution_failed\n"
    request_path = paths["output_root"] / "round-00" / "measurement_request.json"
    assert request_path.is_file()
    assert request_path.stat().st_size == 0
    assert not paths["call_log"].exists()
    for sentinel in (str(tmp_path), str(REPOSITORY_ROOT), "argv", "Traceback"):
        assert sentinel.lower() not in (result.stdout + result.stderr).lower()


def test_cli_rejects_an_empty_code_revision_without_starting_an_adapter(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--local-config",
            str(paths["local"]),
            "--code-revision",
            "",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not paths["call_log"].exists()
