"""Behavioral coverage for the shared pre-provision runner-template validator."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from framework.stage6 import p6_runner_template_validator_v1 as validator


MARKERS = {
    "controller": "stage5_task_round_controller_v3.sh",
    "source_materializer": "stage5_materialize_round_sources_v1.sh",
    "performance_plan": "stage5_build_performance_plan_v2.py",
    "finalizer": "stage5_finalize_feedback_v2.py",
}
STAGES = (
    "source_materialization",
    "quantization",
    "performance",
    "ap",
    "finalization",
)


def _write_yaml(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_executable(path: Path, *, executable: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic executable\n", encoding="utf-8")
    path.chmod(0o700 if executable else 0o600)
    return path


def _runner_template() -> dict[str, Any]:
    return {
        "schema_version": "p6_history_runner_template_v1",
        "stage1_scan": {
            "name": "build_stage1_partition",
            "argv": [
                "private-runner/bin/scan-private",
                "{stage1_partition_manifest}",
                "{local_output_root}",
            ],
        },
        "execution_interface": {
            "schema_version": "p6_history_runner_interface_v1",
            "controller": {
                "argv": [f"documented-stage5-chain/{MARKERS['controller']}"]
            },
            "execution_chain": [
                {
                    "stage": "source_materialization",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['source_materializer']}",
                        "{measurement_request}",
                        "{round_output_root}",
                    ],
                },
                {
                    "stage": "quantization",
                    "argv": [
                        "private-runner/bin/quantize-private",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                },
                {
                    "stage": "performance",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['performance_plan']}",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                },
                {
                    "stage": "ap",
                    "argv": [
                        "private-runner/bin/measure-ap-private",
                        "{task_state}",
                        "{round_output_root}",
                    ],
                },
                {
                    "stage": "finalization",
                    "argv": [
                        f"documented-stage5-chain/{MARKERS['finalizer']}",
                        "{measurement_request}",
                        "{task_state}",
                        "{actual_feedback}",
                        "{actual_receipt}",
                        "{finalization_barrier}",
                        "{round_output_root}",
                    ],
                },
            ],
            "environment": {"values": {}, "activation_argv": ["private-runner/bin/activate-private"]},
            "output_layout": {},
            "actual_feedback": {},
        },
    }


def _write_valid_template(tmp_path: Path) -> tuple[Path, Path]:
    history_root = tmp_path / "private-history"
    history_root.mkdir()
    subprocess.run(["git", "init", "-q", str(history_root)], check=True)
    for marker in MARKERS.values():
        _write_executable(history_root / "documented-stage5-chain" / marker)
    for name in (
        "scan-private",
        "quantize-private",
        "measure-ap-private",
        "activate-private",
    ):
        _write_executable(history_root / "private-runner" / "bin" / name)
    return (
        _write_yaml(tmp_path / "private-inputs" / "runner-template.yaml", _runner_template()),
        history_root,
    )


def _template_payload(template: Path) -> dict[str, Any]:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_validator_returns_private_copy_for_exact_template(
    tmp_path: Path,
) -> None:
    """Catches accepting a template without preserving its five-stage contract."""
    template, history_root = _write_valid_template(tmp_path)

    validated = validator.validate_pre_provision_runner_template(template, history_root)

    assert validated.history_root == history_root
    assert tuple(validated.stage_argv) == STAGES
    assert validated.component_paths == {
        role: history_root / "documented-stage5-chain" / marker
        for role, marker in MARKERS.items()
    }
    validated.execution_interface["controller"]["argv"][0] = "changed-only-in-return"  # type: ignore[index]
    assert _template_payload(template)["execution_interface"]["controller"]["argv"] == [
        f"documented-stage5-chain/{MARKERS['controller']}"
    ]


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_validator_rejects_noncanonical_execution_stage_set(
    tmp_path: Path, mutation: str
) -> None:
    """Catches a missing, extra, or duplicate execution stage being accepted."""
    template, history_root = _write_valid_template(tmp_path)
    payload = _template_payload(template)
    chain = payload["execution_interface"]["execution_chain"]
    if mutation == "missing":
        chain.pop()
    elif mutation == "extra":
        chain.append({"stage": "unexpected", "argv": ["private-runner/bin/quantize-private"]})
    else:
        chain[-1]["stage"] = "ap"
    _write_yaml(template, payload)

    with pytest.raises(validator.RunnerTemplateValidationError) as captured:
        validator.validate_pre_provision_runner_template(template, history_root)

    assert captured.value.category == "execution_interface_unavailable"


def test_validator_rejects_documented_component_marker_drift(tmp_path: Path) -> None:
    """Catches assigning a documented role to a different marker basename."""
    template, history_root = _write_valid_template(tmp_path)
    payload = _template_payload(template)
    payload["execution_interface"]["execution_chain"][2]["argv"][0] = (
        f"documented-stage5-chain/{MARKERS['finalizer']}"
    )
    _write_yaml(template, payload)

    with pytest.raises(validator.RunnerTemplateValidationError) as captured:
        validator.validate_pre_provision_runner_template(template, history_root)

    assert captured.value.category == "history_root_ambiguous"


def test_validator_rejects_shell_token_in_component_argv(tmp_path: Path) -> None:
    """Catches dropping an unsafe token while canonicalizing a component argv."""
    template, history_root = _write_valid_template(tmp_path)
    payload = _template_payload(template)
    payload["execution_interface"]["controller"]["argv"].append("unsafe;token")
    _write_yaml(template, payload)

    with pytest.raises(validator.RunnerTemplateValidationError) as captured:
        validator.validate_pre_provision_runner_template(template, history_root)

    assert captured.value.category == "execution_interface_unavailable"


@pytest.mark.parametrize("mutation", ["public_marker", "shared_private_executable"])
def test_validator_rejects_nonunique_or_documented_private_only_stages(
    tmp_path: Path, mutation: str
) -> None:
    """Catches quantization/AP reusing a public marker or each other's executable."""
    template, history_root = _write_valid_template(tmp_path)
    payload = _template_payload(template)
    chain = payload["execution_interface"]["execution_chain"]
    if mutation == "public_marker":
        chain[1]["argv"][0] = (
            f"documented-stage5-chain/{MARKERS['source_materializer']}"
        )
    else:
        chain[3]["argv"][0] = chain[1]["argv"][0]
    _write_yaml(template, payload)

    with pytest.raises(validator.RunnerTemplateValidationError) as captured:
        validator.validate_pre_provision_runner_template(template, history_root)

    assert captured.value.category == "execution_interface_unavailable"


@pytest.mark.parametrize("mutation", ["outside_root", "symlink", "non_executable"])
def test_validator_rejects_unsafe_stage_executable_paths(
    tmp_path: Path, mutation: str
) -> None:
    """Catches an escaped, linked, or non-executable stage argv0."""
    template, history_root = _write_valid_template(tmp_path)
    payload = _template_payload(template)
    if mutation == "outside_root":
        outside = _write_executable(tmp_path / "outside" / MARKERS["controller"])
        payload["execution_interface"]["controller"]["argv"][0] = str(outside)
    elif mutation == "symlink":
        source = history_root / "private-runner" / "bin" / "quantize-private"
        link = history_root / "private-runner" / "bin" / "quantize-link"
        link.symlink_to(source)
        payload["execution_interface"]["execution_chain"][1]["argv"][0] = str(link)
    else:
        _write_executable(
            history_root / "private-runner" / "bin" / "measure-ap-private",
            executable=False,
        )
    _write_yaml(template, payload)

    with pytest.raises(validator.RunnerTemplateValidationError):
        validator.validate_pre_provision_runner_template(template, history_root)


def test_validator_rejects_trackable_repository_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches accepting a runner template that Git would permit in the repository."""
    external_template, history_root = _write_valid_template(tmp_path)
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    template = _write_yaml(
        repository / "local" / "runner-template.yaml", _template_payload(external_template)
    )
    monkeypatch.setattr(validator, "REPOSITORY_ROOT", repository)

    with pytest.raises(validator.RunnerTemplateValidationError) as captured:
        validator.validate_pre_provision_runner_template(template, history_root)

    assert captured.value.category == "execution_interface_unavailable"
