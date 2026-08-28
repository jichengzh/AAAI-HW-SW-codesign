from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from framework.stage1_bridge import load_stage2_search_space
from tests.stage6.pyramid_formal_space_support import (
    scanner_owned_pyramid_stage1_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/release/build_p6_stage1_manifest.py"


def _valid_stage1_manifest(*, hardware_name: str = "h800") -> dict[str, Any]:
    manifest = scanner_owned_pyramid_stage1_manifest()
    manifest["hw_capability"]["name"] = hardware_name
    manifest["formal_scan"] = {"status": "derived"}
    return manifest


def _write_fake_stage1_repo(path: Path, *, manifest: dict[str, Any]) -> Path:
    stage1 = path / "framework" / "stage1"
    stage1.mkdir(parents=True)
    (path / "framework" / "__init__.py").write_text("", encoding="utf-8")
    (stage1 / "__init__.py").write_text("", encoding="utf-8")
    (stage1 / "adapters.py").write_text(
        """
from __future__ import annotations

from pathlib import Path

ADAPTERS_MARKER = "supplied-stage1-adapters"


class Adapter:
    name = "pyramid_lidar"
    formal_scenario_root = Path(__file__).resolve().parents[2] / "configs" / "stage1"


class ScanScenario:
    def __init__(self, **values):
        self.values = values


def get_adapter(name: str) -> Adapter:
    if name != "pyramid_lidar":
        raise KeyError(name)
    return Adapter()
""",
        encoding="utf-8",
    )
    (stage1 / "hardware_scan.py").write_text(
        """
from __future__ import annotations

HARDWARE_MARKER = "supplied-stage1-hardware"


class HwCapability:
    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def from_yaml(cls, path):
        text = path.read_text(encoding="utf-8").casefold()
        if "h800" not in text:
            return cls("a100")
        return cls("h800")
""",
        encoding="utf-8",
    )
    (stage1 / "formal_scan_evidence.py").write_text(
        """
from __future__ import annotations

import yaml

from framework.stage1.adapters import ScanScenario


def load_scan_scenario(path, *, trusted_root):
    path.resolve(strict=True).relative_to(trusted_root.resolve(strict=True))
    return ScanScenario(**yaml.safe_load(path.read_text(encoding="utf-8")))


def validate_scenario_hardware(scenario, hardware):
    del scenario, hardware
""",
        encoding="utf-8",
    )
    (stage1 / "graph_scan.py").write_text(
        f"""
from __future__ import annotations

import json
import os
from pathlib import Path

from framework.stage1.adapters import ADAPTERS_MARKER
from framework.stage1.hardware_scan import HARDWARE_MARKER


def scan(
    adapter, hw, device: str = "cpu", scenario=None,
    profile_latency_mode: str = "auto",
) -> dict:
    if profile_latency_mode != "off":
        raise RuntimeError("formal release scan enabled latency profiling")
    call_log = Path(os.environ["P6_STAGE1_TEST_CALL_LOG"])
    prior = call_log.read_text(encoding="utf-8") if call_log.exists() else ""
    call_log.write_text(
        prior + "CALL:" + "|".join([
            adapter.name,
            hw.name,
            device,
            scenario.__class__.__name__,
            ADAPTERS_MARKER,
            HARDWARE_MARKER,
            profile_latency_mode,
            os.environ["HEAL_ROOT"],
            os.environ["HEAL_CKPT_ROOT"],
        ]),
        encoding="utf-8",
    )
    return json.loads({json.dumps(manifest)!r})
""",
        encoding="utf-8",
    )
    return path


def _write_hardware(path: Path) -> Path:
    path.write_text("basic:\n  name: NVIDIA H800\n", encoding="utf-8")
    return path


def _write_scenario(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """hardware_precisions: [FP16, INT8]
backend_precisions: [FP16, INT8]
compression_modes: [fp16, int8]
graph_quant_unit_policy:
  pyramid_backbone: [FP16, INT8]
alignment:
  default_round_to: 8
""",
        encoding="utf-8",
    )
    return path


def _run_cli(
    tmp_path: Path,
    *extra: str,
    stage1_repo_root: Path | None = None,
    output_path: Path | None = None,
    hardware_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    root = stage1_repo_root or _write_fake_stage1_repo(
        tmp_path / "stage1-repo",
        manifest=_valid_stage1_manifest(),
    )
    hardware = hardware_path or _write_hardware(tmp_path / "h800.yaml")
    scenario = _write_scenario(root / "configs" / "stage1" / "scenario.yaml")
    output = output_path or (tmp_path / "manifest.json")
    heal_root = tmp_path / "heal"
    heal_checkpoint_root = tmp_path / "heal-checkpoints"
    heal_root.mkdir(exist_ok=True)
    heal_checkpoint_root.mkdir(exist_ok=True)
    call_log = tmp_path / "call.log"
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--hardware",
            str(hardware),
            "--output",
            str(output),
            "--device",
            "cuda:0",
            "--scenario",
            str(scenario),
            "--stage1-repo-root",
            str(root),
            "--heal-root",
            str(heal_root),
            "--heal-checkpoint-root",
            str(heal_checkpoint_root),
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        env={"P6_STAGE1_TEST_CALL_LOG": str(call_log)},
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_builds_stage1_manifest_with_private_paths_hidden(tmp_path: Path) -> None:
    result = _run_cli(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "stage1_manifest_written\n"
    assert result.stderr == ""
    output_path = tmp_path / "manifest.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == _valid_stage1_manifest()
    assert (tmp_path / "call.log").read_text(encoding="utf-8") == (
        "CALL:pyramid_lidar|h800|cuda:0|ScanScenario|supplied-stage1-adapters|"
        f"supplied-stage1-hardware|off|{tmp_path / 'heal'}|"
        f"{tmp_path / 'heal-checkpoints'}"
    )
    assert str(tmp_path) not in result.stdout + result.stderr
    assert load_stage2_search_space(output_path)["axis_schema"]["free_axes"]


def test_cli_requires_explicit_scenario_without_running_scanner(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--scenario", "")

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not (tmp_path / "call.log").exists()


def test_cli_rejects_relative_path_without_scanning(tmp_path: Path) -> None:
    result = _run_cli(
        tmp_path,
        output_path=Path("relative-manifest.json"),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"
    assert not (tmp_path / "call.log").exists()
    assert not (REPOSITORY_ROOT / "relative-manifest.json").exists()


def test_cli_rejects_invalid_scanner_result_without_output(tmp_path: Path) -> None:
    stage1_repo = _write_fake_stage1_repo(
        tmp_path / "stage1-repo",
        manifest={**_valid_stage1_manifest(), "scan_status": "partial"},
    )

    result = _run_cli(tmp_path, stage1_repo_root=stage1_repo)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "stage1_scan_invalid\n"
    assert not (tmp_path / "manifest.json").exists()
