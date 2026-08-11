from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from tests.stage6.test_h800_search_execution import _public_contract, _stage5_inputs


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/release/run_p6_h800_search.py"
PUBLIC_SUMMARY_KEYS = {
    "schema",
    "target",
    "code_revision",
    "seed",
    "configuration_label",
    "assets",
    "status",
    "planned_rounds",
    "completed_rounds",
    "successful_candidate_count",
    "aggregate_metrics",
    "failure_code",
}


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    return path


def _write_fake_trainer(path: Path) -> Path:
    path.write_text(
        """from __future__ import annotations
import json
from pathlib import Path
import sys

request_path, result_path, call_log, mode = sys.argv[1:]
request = json.loads(Path(request_path).read_text(encoding=\"utf-8\"))
with Path(call_log).open(\"a\", encoding=\"utf-8\") as handle:
    handle.write(request[\"candidate_id\"] + \"\\n\")
if mode == \"fail\":
    sys.stderr.write(\"PRIVATE_TRAINER_STDERR\\n\")
    raise SystemExit(9)
result = {
    \"candidate_id\": request[\"candidate_id\"],
    \"measurements\": [
        {
            \"candidate_id\": row[\"row_id\"],
            \"latency_ms\": 2.0,
            \"energy_j\": 0.5,
            \"ap30\": 0.9,
            \"ap50\": 0.8,
            \"ap70\": 0.7,
        }
        for row in request[\"rows\"]
    ],
}
Path(result_path).write_text(json.dumps(result), encoding=\"utf-8\")
""",
        encoding="utf-8",
    )
    return path


def _cli_fixture(tmp_path: Path, *, mode: str = "success") -> dict[str, Path]:
    contract_path = _write_yaml(
        tmp_path / "contract.yaml",
        _public_contract(max_rounds=1, batch_size=1),
    )
    input_paths: dict[str, str] = {}
    for name, payload in _stage5_inputs().items():
        input_path = tmp_path / f"{name}.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        input_paths[name] = str(input_path)

    asset_paths: dict[str, str] = {}
    for label in ("training-data", "model-init", "toolchain"):
        asset_path = tmp_path / label
        asset_path.mkdir()
        asset_paths[label] = str(asset_path)

    trainer_path = _write_fake_trainer(tmp_path / "fake_trainer.py")
    call_log = tmp_path / "PRIVATE_CALL_LOG.txt"
    output_root = tmp_path / "PRIVATE_LOCAL_OUTPUT"
    result_path = output_root / "PRIVATE_CHECKPOINT_RESULT.json"
    local_path = _write_yaml(
        tmp_path / "local.yaml",
        {
            "schema_version": "p6_h800_search_local_v1",
            "target": "h800",
            "asset_paths": asset_paths,
            "stage5_input_paths": input_paths,
            "steps": [
                {
                    "name": "evaluate",
                    "argv": [
                        sys.executable,
                        str(trainer_path),
                        "{candidate_request}",
                        "{result_json}",
                        str(call_log),
                        mode,
                    ],
                }
            ],
            "result_step": "evaluate",
            "result_path_template": str(result_path),
            "local_output_root": str(output_root),
        },
    )
    return {
        "contract": contract_path,
        "local": local_path,
        "summary": tmp_path / "public-summary.json",
        "call_log": call_log,
        "output_root": output_root,
        "result": result_path,
        "trainer": trainer_path,
    }


def _run_cli(paths: Mapping[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--local-config",
            str(paths["local"]),
            "--public-summary",
            str(paths["summary"]),
            "--code-revision",
            "test-revision",
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_completes_with_fake_trainer_and_publishes_redacted_summary(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)

    result = _run_cli(paths)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "completed\n"
    public = json.loads(paths["summary"].read_text(encoding="utf-8"))
    serialized = json.dumps(public, sort_keys=True)
    candidate_id = paths["call_log"].read_text(encoding="utf-8").strip()
    assert set(public) == PUBLIC_SUMMARY_KEYS
    assert public["status"] == "completed"
    assert public["code_revision"] == "test-revision"
    assert public["completed_rounds"] == 1
    assert set(public["aggregate_metrics"]["latency_ms"]) == {
        "count",
        "min",
        "max",
        "mean",
    }
    for sentinel in (
        str(tmp_path),
        str(paths["trainer"]),
        str(paths["result"]),
        candidate_id,
        "PRIVATE_CHECKPOINT_RESULT.json",
        "PRIVATE_CALL_LOG",
    ):
        assert sentinel not in serialized


def test_cli_requires_explicit_local_config_without_starting_a_candidate(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--public-summary",
            str(paths["summary"]),
            "--code-revision",
            "test-revision",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr == "argument_error\n"
    assert not paths["call_log"].exists()
    assert not paths["summary"].exists()


def test_cli_rejects_invalid_local_config_without_publishing(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local["local_output_root"] = "relative/private-output"
    _write_yaml(paths["local"], local)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()
    assert not paths["summary"].exists()


@pytest.mark.parametrize(
    "case",
    ["asset_label_mismatch", "duplicate_step", "shell_argv", "private_summary"],
)
def test_cli_fails_closed_for_invalid_local_yaml(
    tmp_path: Path,
    case: str,
) -> None:
    paths = _cli_fixture(tmp_path)
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    if case in {"asset_label_mismatch", "private_summary"}:
        del local["asset_paths"]["toolchain"]
    elif case == "duplicate_step":
        local["steps"].append(dict(local["steps"][0]))
    else:
        local["steps"][0]["argv"] = ["bash", "-c", "PRIVATE_COMMAND", "{result_json}"]
    if case == "private_summary":
        paths["summary"] = paths["output_root"] / "public-summary.json"
    _write_yaml(paths["local"], local)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()
    assert not paths["summary"].exists()


def test_cli_rejects_result_path_equal_to_public_summary_before_execution(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)
    local = yaml.safe_load(paths["local"].read_text(encoding="utf-8"))
    local["result_path_template"] = str(paths["summary"])
    _write_yaml(paths["local"], local)

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()
    assert not paths["summary"].exists()


@pytest.mark.parametrize("document", ["contract", "local"])
def test_cli_reports_invalid_utf8_as_stable_contract_error(
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
    assert not paths["summary"].exists()


def test_cli_stops_after_failed_subcommand_and_redacts_process_details(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path, mode="fail")

    result = _run_cli(paths)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "execution_failed\n"
    assert len(paths["call_log"].read_text(encoding="utf-8").splitlines()) == 1
    public_text = paths["summary"].read_text(encoding="utf-8")
    public = json.loads(public_text)
    assert public["status"] == "failed"
    assert public["failure_code"] == "command_failed"
    for sentinel in (
        "PRIVATE_TRAINER_STDERR",
        str(tmp_path),
        str(paths["trainer"]),
        "candidate-",
        "checkpoint",
        "sha256",
    ):
        assert sentinel.lower() not in public_text.lower()
        assert sentinel.lower() not in (result.stdout + result.stderr).lower()


def test_cli_rejects_local_output_override_and_empty_code_revision(tmp_path: Path) -> None:
    paths = _cli_fixture(tmp_path)

    override = _run_cli(paths, "--local-output-root", str(tmp_path / "override"))
    empty_revision = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--contract",
            str(paths["contract"]),
            "--local-config",
            str(paths["local"]),
            "--public-summary",
            str(paths["summary"]),
            "--code-revision",
            "",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert override.returncode == 2
    assert empty_revision.returncode == 2
    assert not paths["call_log"].exists()


def test_cli_rejects_public_summary_inside_private_output_without_publishing(
    tmp_path: Path,
) -> None:
    paths = _cli_fixture(tmp_path)
    paths["summary"] = paths["output_root"] / "public-summary.json"

    result = _run_cli(paths)

    assert result.returncode == 2
    assert result.stderr == "contract_error\n"
    assert not paths["call_log"].exists()
    assert not paths["summary"].exists()
