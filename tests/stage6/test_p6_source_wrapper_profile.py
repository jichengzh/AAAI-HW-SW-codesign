"""Behavioral tests for the private P6 source-wrapper deployment profile."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_runner_template_validator_v1 import (
    ValidatedRunnerTemplate,
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_source_wrapper_profile_v1 import (
    P6SourceWrapperProfileError,
    render_self_contained_source_wrapper,
    validate_self_contained_source_wrapper,
)
from tests.p6_source_wrapper_support import (
    source_bridge_output_paths,
    source_bridge_request,
)
from tests.stage6.test_p6_runner_template_validator import _write_valid_template


SOURCE_MARKER = "stage5_materialize_round_sources_v1.sh"
SOURCE_STAGE_TAIL = ("{measurement_request}", "{round_output_root}")


def _canonical_request(output_root: Path) -> dict[str, Any]:
    binding = {
        "schema_version": "p6_external_training_binding_v1",
        "training_required": True,
        "training_source_kind": "selected_candidate_finetune",
        "dataset_root": "/operator/dataset",
        "base_checkpoint_path": "/operator/base.ckpt",
        "base_checkpoint_sha256": "a" * 64,
        "pyramid_config_path": "/operator/pyramid.yaml",
        "pyramid_config_sha256": "b" * 64,
        "training_parameters": {
            "training_mode": "finetune_selected_width",
            "epochs": 8,
            "target_epoch": 9,
            "seed": 0,
            "optimizer": "adam",
            "learning_rate": 0.002,
            "batch_size": 2,
            "dataset_split": "trainval_coptv2x",
            "checkpoint_selection": "best_ap70",
            "freeze_policy": "pyramid_backbone_partial",
            "groups": 3,
            "width_per_group": 5,
        },
    }
    return source_bridge_request(
        binding,
        source_contract_fields=source_bridge_output_paths(
            output_root / "materialized" / "pyramid-16-32-64"
        ),
    )


def _wrapper_profile(project_python: Path | None = None) -> dict[str, str]:
    profile = {
        "schema_version": "p6_private_source_wrapper_profile_v1",
        "wrapper_kind": "repo_cwd_exec_v1",
        "destination_relative_path": f"documented-stage5-chain/{SOURCE_MARKER}",
        "implementation_relative_path": (
            "private-relocated-history-repo/bin/stage5_materialize_round_sources_v1.original.sh"
        ),
        "implementation_cwd_relative_path": "private-relocated-history-repo",
    }
    if project_python is None:
        return profile
    return {
        **profile,
        "schema_version": "p6_private_source_wrapper_profile_v2",
        "project_python": str(project_python),
    }


def _write_profile(path: Path, profile: dict[str, str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(profile or _wrapper_profile(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_relocated_materializer(history_root: Path) -> Path:
    repository = history_root / "private-relocated-history-repo"
    implementation = repository / "bin" / "stage5_materialize_round_sources_v1.original.sh"
    implementation.parent.mkdir(parents=True, exist_ok=True)
    (repository / "history_contract_validator.py").write_text("VALID = 'ok'\n", encoding="utf-8")
    implementation.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "from pathlib import Path\n"
        "import sys\n"
        "import history_contract_validator as validator\n"
        "Path('observed-cwd.txt').write_text(str(Path.cwd()), encoding='utf-8')\n"
        "Path('sibling-import-ok.txt').write_text(validator.VALID, encoding='utf-8')\n"
        "request = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))\n"
        "contract = request['rows'][0]['source_contract']\n"
        "config = Path(contract['checkpoint_dir']) / 'config.yaml'\n"
        "assert Path(contract['config_path']) == config\n"
        "config.parent.mkdir(parents=True, exist_ok=True)\n"
        "config.write_text('fixture:config_path\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    implementation.chmod(0o700)
    return implementation


def _write_project_python(tmp_path: Path) -> Path:
    executable = tmp_path / "project-env/bin/python3.9"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n", encoding="utf-8")
    executable.chmod(0o700)
    (executable.parent / "python").symlink_to(executable.name)
    return executable


def _write_valid_template_with_generated_wrapper(
    tmp_path: Path,
    *,
    legacy_v1: bool = False,
) -> tuple[Path, Path, Path, ValidatedRunnerTemplate]:
    template, history_root = _write_valid_template(tmp_path)
    marker = history_root / "documented-stage5-chain" / SOURCE_MARKER
    marker.unlink()
    _write_relocated_materializer(history_root)
    project_python = None if legacy_v1 else _write_project_python(tmp_path)
    payload = _wrapper_profile(project_python)
    profile = _write_profile(
        tmp_path / "ignored-inputs" / "source-wrapper-profile.yaml", payload
    )
    render_self_contained_source_wrapper(payload, history_root=history_root)
    validated = validate_pre_provision_runner_template(template, history_root)
    return template, history_root, profile, validated


def _with_source_executable(
    validated: ValidatedRunnerTemplate, executable: Path
) -> ValidatedRunnerTemplate:
    stage_argv = dict(validated.stage_argv)
    stage_argv["source_materialization"] = (
        str(executable),
        *SOURCE_STAGE_TAIL,
    )
    return replace(validated, stage_argv=stage_argv)


def test_source_wrapper_profile_accepts_marker_stage_under_private_root(
    tmp_path: Path,
) -> None:
    _, history_root, profile, validated = _write_valid_template_with_generated_wrapper(tmp_path)

    wrapper = validate_self_contained_source_wrapper(
        validated,
        source_wrapper_profile=profile,
    )

    assert wrapper.executable == (history_root / "documented-stage5-chain" / SOURCE_MARKER)
    assert wrapper.argv_shape == (
        "--request",
        "<absolute-private-request-json>",
        "--model",
        "pyramid",
        "--group-id",
        "<canonical-pyramid-group-id>",
        "--gpu",
        "<validated-binding-gpu-index>",
    )
    assert wrapper.marker_basename == SOURCE_MARKER


def test_training_runtime_rejects_legacy_v1_source_wrapper_profile(
    tmp_path: Path,
) -> None:
    _, _, profile, validated = _write_valid_template_with_generated_wrapper(
        tmp_path, legacy_v1=True
    )

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        validate_self_contained_source_wrapper(
            validated,
            source_wrapper_profile=profile,
        )

    assert captured.value.category == "history_execution_invalid"


def test_training_runtime_rejects_project_python_launcher_drift(
    tmp_path: Path,
) -> None:
    _, _, profile, validated = _write_valid_template_with_generated_wrapper(tmp_path)
    payload = yaml.safe_load(profile.read_text(encoding="utf-8"))
    project_python = Path(payload["project_python"])
    alternate = project_python.parent / "alternate-python"
    alternate.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    alternate.chmod(0o700)
    launcher = project_python.parent / "python"
    launcher.unlink()
    launcher.symlink_to(alternate.name)

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        validate_self_contained_source_wrapper(
            validated,
            source_wrapper_profile=profile,
        )

    assert captured.value.category == "history_execution_invalid"


def test_source_wrapper_execs_relocated_implementation_from_private_cwd_with_exact_env(
    tmp_path: Path,
) -> None:
    _, history_root, _, validated = _write_valid_template_with_generated_wrapper(tmp_path)
    round_root = history_root / "private-runs" / "0"
    round_root.mkdir(parents=True)
    request = round_root / "measurement-request.json"
    request.write_text(
        json.dumps(_canonical_request(history_root / "private-output")),
        encoding="utf-8",
    )
    task_state = round_root / "state" / "task-state.json"
    task_state.parent.mkdir()
    task_state.write_text("{}", encoding="utf-8")
    env = {
        "CUDA_VISIBLE_DEVICES": "101,103,107",
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(history_root),
        "P6_HISTORY_TASK_STATE": str(task_state),
        "P6_HISTORY_ROUND_OUTPUT_ROOT": str(round_root),
    }

    completed = subprocess.run(
        [
            validated.stage_argv["source_materialization"][0],
            "--request",
            str(request),
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

    assert completed.returncode == 0, completed.stderr
    assert set(env) == set(EXPECTED_HISTORY_ENV_KEYS)
    relocated_repo = history_root / "private-relocated-history-repo"
    assert (relocated_repo / "observed-cwd.txt").read_text(encoding="utf-8").strip() == str(
        relocated_repo
    )
    assert (relocated_repo / "sibling-import-ok.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.parametrize(
    "mutation",
    [
        "source_not_first_stage",
        "wrong_basename",
        "outside_root",
        "symlink",
        "not_executable",
        "digest_drift",
        "environment_key_drift",
    ],
)
def test_source_wrapper_profile_rejects_non_self_contained_binding_shape(
    tmp_path: Path, mutation: str
) -> None:
    _, history_root, profile, validated = _write_valid_template_with_generated_wrapper(tmp_path)
    marker = history_root / "documented-stage5-chain" / SOURCE_MARKER
    mutated = validated
    if mutation == "source_not_first_stage":
        stage_argv = {
            "quantization": validated.stage_argv["quantization"],
            "source_materialization": validated.stage_argv["source_materialization"],
            **{
                name: argv
                for name, argv in validated.stage_argv.items()
                if name not in {"quantization", "source_materialization"}
            },
        }
        mutated = replace(validated, stage_argv=stage_argv)
    elif mutation == "wrong_basename":
        wrong = history_root / "documented-stage5-chain" / "wrong-source.sh"
        wrong.write_bytes(marker.read_bytes())
        wrong.chmod(0o700)
        mutated = _with_source_executable(validated, wrong)
    elif mutation == "outside_root":
        outside = tmp_path / SOURCE_MARKER
        outside.write_bytes(marker.read_bytes())
        outside.chmod(0o700)
        mutated = _with_source_executable(validated, outside)
    elif mutation == "symlink":
        target = history_root / "private-relocated-history-repo" / "linked-wrapper"
        target.symlink_to(marker)
        mutated = _with_source_executable(validated, target)
    elif mutation == "not_executable":
        marker.chmod(0o600)
    elif mutation == "digest_drift":
        marker.write_bytes(marker.read_bytes() + b"# drift\n")
    else:
        interface: dict[str, Any] = {
            **validated.execution_interface,
            "environment": {
                **validated.execution_interface["environment"],  # type: ignore[dict-item]
                "values": {
                    **validated.execution_interface["environment"]["values"],  # type: ignore[index]
                    "PYTHONPATH": {"kind": "literal", "value": "private"},
                },
            },
        }
        mutated = replace(validated, execution_interface=interface)

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        validate_self_contained_source_wrapper(
            mutated,
            source_wrapper_profile=profile,
        )

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
