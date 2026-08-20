from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "tools/release/build_p6_stage1_manifest.py"


def _valid_stage1_manifest(*, hardware_name: str = "h800") -> dict[str, Any]:
    return {
        "schema": "stage1_partition_manifest_v1",
        "stage": "stage1_partition",
        "model": "pyramid_lidar",
        "scan_status": "ok",
        "hw_capability": {"name": hardware_name},
        "view_b1_search_groups": [
            {
                "search_group_id": "pyramid_group.s0",
                "bucket": "pyramid_backbone",
                "widths": [224],
                "round_to": 32,
                "int8_buildable_align": 128,
                "max_rate": 0.875,
                "grouped_conv": True,
                "criterion_pool": ["L1"],
                "member_b1_groups": ["pyramid_group.s0"],
            }
        ],
        "view_b2_quant_units": [
            {
                "unit": "pyramid_backbone",
                "quantizable": True,
                "legal_bits": ["FP16", "INT8"],
                "member_groups": ["pyramid_group.s0"],
            }
        ],
        "view_d_routing_segments": {"segments": [{"device": "gpu", "n_nodes": 1}]},
    }


def _write_fake_stage1_repo(path: Path, *, manifest: dict[str, Any]) -> Path:
    stage1 = path / "framework" / "stage1"
    stage1.mkdir(parents=True)
    (path / "framework" / "__init__.py").write_text("", encoding="utf-8")
    (stage1 / "__init__.py").write_text("", encoding="utf-8")
    (stage1 / "adapters.py").write_text(
        """
from __future__ import annotations

ADAPTERS_MARKER = "supplied-stage1-adapters"


class Adapter:
    name = "pyramid_lidar"


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
    (stage1 / "graph_scan.py").write_text(
        f"""
from __future__ import annotations

import json
import os
from pathlib import Path

from framework.stage1.adapters import ADAPTERS_MARKER
from framework.stage1.hardware_scan import HARDWARE_MARKER


def scan(adapter, hw, device: str = "cpu") -> dict:
    call_log = Path(os.environ["P6_STAGE1_TEST_CALL_LOG"])
    call_log.write_text(
        "|".join([
            adapter.name,
            hw.name,
            device,
            ADAPTERS_MARKER,
            HARDWARE_MARKER,
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
        "pyramid_lidar|h800|cuda:0|supplied-stage1-adapters|"
        f"supplied-stage1-hardware|{tmp_path / 'heal'}|"
        f"{tmp_path / 'heal-checkpoints'}"
    )
    assert str(tmp_path) not in result.stdout + result.stderr


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
