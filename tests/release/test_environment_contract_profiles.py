"""Public profile fixtures remain offline, synthetic environment contracts."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from framework.capability_schema import HardwareCapability


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "release" / "validate_environment_contract.py"
PROFILES = ("cpu_reference", "rtx4090", "h800")
FORBIDDEN_KEYS = {
    "host",
    "hostname",
    "ssh",
    "uuid",
    "serial",
    "sha256",
    "measurement",
    "provenance",
}
FORBIDDEN_TEXT = ("ssh", "nvidia-smi", "http://", "https://")
PRIVATE_PATH_PATTERN = re.compile(
    r"(?:/" + "home" + r"/|/" + "users" + r"/|[a-z]:[\\/])", re.IGNORECASE
)
PRIVATE_HOME_PATH = "/" + "home" + "/operator"
PRIVATE_USERS_PATH = "/" + "Users" + "/operator"
PRIVATE_WINDOWS_PATH = "C:" + "\\" + "Users" + "\\" + "operator"


@pytest.mark.parametrize(
    "polluted_document",
    [
        {"host": "redacted"},
        {"hostname": "redacted"},
        {"ssh": "redacted"},
        {"uuid": "redacted"},
        {"serial": "redacted"},
        {"sha256": "redacted"},
        {"measurement": "redacted"},
        {"provenance": "redacted"},
        {"runtime": {"note": "prefix " + PRIVATE_HOME_PATH}},
        {"runtime": {"note": "prefix " + PRIVATE_USERS_PATH}},
        {"runtime": {"note": PRIVATE_WINDOWS_PATH}},
        {"runtime": {"note": "https://private.example"}},
        {"runtime": {"note": "nvidia-smi"}},
    ],
)
def test_public_redaction_guard_rejects_bound_private_markers(
    polluted_document: dict[str, object],
) -> None:
    with pytest.raises(AssertionError):
        _assert_public_json_is_redacted(polluted_document)


def _run_cli(contract: Path, observation: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--contract",
            str(contract),
            "--observation",
            str(observation),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _assert_public_json_is_redacted(value: object) -> None:
    if isinstance(value, dict):
        assert not (set(value) & FORBIDDEN_KEYS)
        assert value.get("paper_evidence") is not True
        for item in value.values():
            _assert_public_json_is_redacted(item)
    elif isinstance(value, list):
        for item in value:
            _assert_public_json_is_redacted(item)
    elif isinstance(value, str):
        assert not value.startswith("/")
        assert PRIVATE_PATH_PATTERN.search(value) is None
        assert not any(token in value.lower() for token in FORBIDDEN_TEXT)


@pytest.mark.parametrize(("target", "invalid_name", "code"), [
    ("cpu_reference", "cpu_reference-invalid-gpu.json", "runtime.gpu_count.unexpected"),
    ("rtx4090", "rtx4090-invalid-cuda.json", "runtime.cuda.out_of_range"),
    ("h800", "h800-invalid-driver.json", "runtime.driver.out_of_range"),
])
def test_public_profiles_have_valid_and_invalid_offline_observations(
    target: str, invalid_name: str, code: str, tmp_path: Path
) -> None:
    contract = ROOT / "configs/environment" / f"{target}.yaml"
    valid = ROOT / "data/demo/environment_observations" / f"{target}-valid.json"
    invalid = ROOT / "data/demo/environment_observations" / invalid_name

    assert _run_cli(contract, valid, tmp_path / "valid.json").returncode == 0
    assert _run_cli(contract, invalid, tmp_path / "invalid.json").returncode == 1
    assert code in _read_json(tmp_path / "invalid.json")["failures"][0]["code"]


@pytest.mark.parametrize(("target", "ip_name"), [
    ("cpu_reference", "cpu"),
    ("rtx4090", "gpu"),
    ("h800", "gpu"),
])
def test_public_hardware_profiles_are_parseable(target: str, ip_name: str) -> None:
    capability = HardwareCapability.from_yaml(ROOT / "configs/hardware" / f"{target}.yaml")

    assert set(capability.ips) == {ip_name}


@pytest.mark.parametrize(
    ("target", "expected_arch", "expected_precisions", "requires_gpu"),
    [
        ("cpu_reference", "generic-cpu", [], False),
        ("rtx4090", "Ada sm89", ["FP32", "TF32", "FP16", "INT8"], True),
        ("h800", "Hopper sm90", ["FP32", "TF32", "BF16", "FP16", "INT8", "FP8"], True),
    ],
)
def test_public_profiles_declare_the_published_capabilities_and_constraints(
    target: str, expected_arch: str, expected_precisions: list[str], requires_gpu: bool
) -> None:
    hardware_path = ROOT / "configs/hardware" / f"{target}.yaml"
    contract_path = ROOT / "configs/environment" / f"{target}.yaml"
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    if target == "cpu_reference":
        assert hardware["name"] == "CPU reference"
    assert HardwareCapability.from_yaml(hardware_path).arch == expected_arch
    assert hardware["ips"]["cpu" if not requires_gpu else "gpu"]["precisions"] == expected_precisions
    capability = HardwareCapability.from_yaml(hardware_path)
    assert capability.features.tensor_core is requires_gpu
    assert contract["schema"] == "environment_contract_v1"
    assert contract["target"] == target
    assert contract["hardware_capability"] == f"configs/hardware/{target}.yaml"
    assert contract["requires_gpu"] is requires_gpu
    assert contract["runtime"]["framework_name"] == "pytorch"
    assert contract["runtime"]["framework_version"] == ">=2.0"
    assert contract["runtime"]["python"] == ">=3.10,<3.12"
    assert contract["runtime"]["cuda"] == (">=12.0,<13" if requires_gpu else None)
    assert contract["runtime"]["driver"] == (">=525" if requires_gpu else None)


@pytest.mark.parametrize("target", PROFILES)
def test_public_observations_are_synthetic_and_redacted(target: str) -> None:
    observations = sorted((ROOT / "data/demo/environment_observations").glob(f"{target}-*.json"))

    assert len(observations) == 2
    for observation_path in observations:
        observation = _read_json(observation_path)
        assert observation["schema"] == "environment_observation_v1"
        assert observation["target"] == target
        _assert_public_json_is_redacted(observation)


@pytest.mark.parametrize("target", PROFILES)
def test_public_observation_fixtures_use_only_the_declared_runtime_deltas(target: str) -> None:
    observations_root = ROOT / "data/demo/environment_observations"
    valid = _read_json(observations_root / f"{target}-valid.json")
    invalid = _read_json(
        observations_root
        / {
            "cpu_reference": "cpu_reference-invalid-gpu.json",
            "rtx4090": "rtx4090-invalid-cuda.json",
            "h800": "h800-invalid-driver.json",
        }[target]
    )

    assert set(valid) == set(invalid)
    assert set(valid["runtime"]) == set(invalid["runtime"])
    assert valid["runtime"] == {
        "python": "3.11.9",
        "cuda": "12.4" if target != "cpu_reference" else None,
        "driver": "550.54" if target != "cpu_reference" else None,
        "framework_name": "pytorch",
        "framework_version": "2.4.0",
    }
    assert valid["gpu_count"] == (1 if target != "cpu_reference" else 0)
    deltas = {
        key: (valid[key], invalid[key])
        for key in valid
        if valid[key] != invalid[key]
    }
    expected_deltas = {
        "cpu_reference": {"gpu_count": (0, 1)},
        "rtx4090": {"runtime": (valid["runtime"], {**valid["runtime"], "cuda": "11.8"})},
        "h800": {"runtime": (valid["runtime"], {**valid["runtime"], "driver": "524.0"})},
    }
    assert deltas == expected_deltas[target]
