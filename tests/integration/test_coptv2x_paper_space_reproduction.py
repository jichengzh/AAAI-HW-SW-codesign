"""Public CPU-only reproduction gate for the three CoptV2X paper spaces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from framework.reproduction.coptv2x_paper_space_v1 import (
    build_stage1_scanner_manifest,
    derive_paper_axis_bundle,
    build_paper_space_reproduction,
    load_paper_scanner_evidence,
)
from framework.stage6.pyramid_search_space_adapter_v1 import (
    build_pyramid_candidate_plan,
)
from framework.stage2.formal_software_space_v1 import build_formal_software_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

PAPER_ORACLES = [
    (
        "pyramid",
        343,
        686,
        [
            [16, 24, 32, 40, 48, 56, 64],
            [32, 48, 64, 80, 96, 112, 128],
            [64, 96, 128, 160, 192, 224, 256],
        ],
    ),
    (
        "codriving",
        343,
        686,
        [
            [16, 24, 32, 40, 48, 56, 64],
            [32, 48, 64, 80, 96, 112, 128],
            [64, 96, 128, 160, 192, 224, 256],
        ],
    ),
    (
        "fcooper",
        1792,
        3584,
        [
            [32, 64],
            [32, 64, 96, 128],
            [32, 64, 96, 128, 160, 192, 224, 256],
            [32, 64, 96, 128],
            [64, 96, 128, 160, 192, 224, 256],
        ],
    ),
]


@pytest.mark.parametrize(
    ("model", "expected_structures", "expected_candidates", "expected_widths"),
    PAPER_ORACLES,
)
def test_public_reproduction_pipeline_derives_paper_space_from_scanner_evidence(
    model: str,
    expected_structures: int,
    expected_candidates: int,
    expected_widths: list[list[int]],
) -> None:
    result = build_paper_space_reproduction(model)
    plan = result.mutable_formal_plan()

    assert [axis["legal_widths"] for axis in plan["axis_schema"]["free_axes"]] == (
        expected_widths
    )
    assert plan["structure_count"] == expected_structures
    assert plan["candidate_count"] == expected_candidates
    assert plan["q_modes"] == ["fp16", "int8"]
    assert result.scanner_manifest["schema"] == "stage1_scanner_contract_v1"
    assert result.search_space["formal_candidate_policy"]["source"] == (
        "scanner_structural_axes"
    )
    assert not any("expected" in key for key in result.evidence)


def test_public_reproduction_pipeline_aligns_pyramid_p6_thin_adapter_identity() -> None:
    result = build_paper_space_reproduction("pyramid")
    formal_plan = result.mutable_formal_plan()
    p6_plan = build_pyramid_candidate_plan(result.mutable_search_space())

    assert p6_plan["structure_count"] == formal_plan["structure_count"] == 343
    assert p6_plan["candidate_count"] == formal_plan["candidate_count"] == 686
    assert p6_plan["q_modes"] == formal_plan["q_modes"] == ["fp16", "int8"]
    assert [row["formal_candidate_id"] for row in p6_plan["candidates"]] == [
        row["candidate_id"] for row in formal_plan["candidates"]
    ]
    assert p6_plan["candidates"][0]["source_point_ids"] == [
        "backbone.s0:w16",
        "backbone.s1:w32",
        "backbone.s2:w64",
    ]
    assert p6_plan["candidates"][-1]["width"] == [64, 128, 256]


def test_cli_prints_exact_deterministic_all_model_summary() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/reproduce_paper_search_space.py"),
            "--model",
            "all",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "pyramid: structures=343 candidates=686 q_modes=fp16,int8\n"
        "codriving: structures=343 candidates=686 q_modes=fp16,int8\n"
        "fcooper: structures=1792 candidates=3584 q_modes=fp16,int8\n"
    )
    assert result.stderr == ""


def test_cli_rejects_invalid_model_without_running_fallback() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/reproduce_paper_search_space.py"),
            "--model",
            "v2xvit",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert result.stdout == ""


def test_cli_prioritizes_repository_framework_over_ambient_pythonpath(
    tmp_path: Path,
) -> None:
    ambient_root = tmp_path / "ambient"
    ambient_framework = ambient_root / "framework"
    ambient_framework.mkdir(parents=True)
    marker = tmp_path / "ambient-framework-imported"
    (ambient_framework / "__init__.py").write_text(
        """from pathlib import Path
import os

Path(os.environ["PAPER_SPACE_AMBIENT_IMPORT_MARKER"]).write_text("imported", encoding="utf-8")
raise RuntimeError("ambient framework imported")
""",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PAPER_SPACE_AMBIENT_IMPORT_MARKER": str(marker),
        "PYTHONPATH": os.pathsep.join((str(ambient_root), str(REPOSITORY_ROOT))),
    }

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/reproduce_paper_search_space.py"),
            "--model",
            "pyramid",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "pyramid: structures=343 candidates=686 q_modes=fp16,int8\n"
    assert not marker.exists()


def test_explicit_empty_evidence_does_not_fallback_to_fixture() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        derive_paper_axis_bundle("pyramid", {})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.pop("scan_scenario"), "missing required fields"),
        (lambda data: data.update({"loaded_config": []}), "loaded_config"),
        (lambda data: data.update({"expected_candidates": 686}), "unknown fields"),
        (lambda data: data.update({"model": "codriving"}), "does not match"),
    ],
)
def test_paper_scanner_evidence_boundary_fails_closed(
    mutate: object, message: str
) -> None:
    evidence = load_paper_scanner_evidence("pyramid")
    mutate(evidence)

    with pytest.raises(ValueError, match=message):
        derive_paper_axis_bundle("pyramid", evidence)


def test_caller_mutation_after_call_does_not_affect_derived_manifest() -> None:
    evidence = load_paper_scanner_evidence("pyramid")
    manifest = build_stage1_scanner_manifest("pyramid", evidence)

    evidence["scan_scenario"]["hardware_precisions"].clear()
    evidence["checkpoint_evidence"]["module_widths"].clear()

    assert manifest["hw_capability"]["ips"]["gpu"]["precisions"] == ["FP16", "INT8"]
    assert manifest["scanner_structural_axes"]


def test_public_reproduction_artifacts_are_directly_consumable_fresh_copies() -> None:
    result = build_paper_space_reproduction("pyramid")

    formal_plan = build_formal_software_plan(result.search_space)
    p6_plan = build_pyramid_candidate_plan(result.search_space)

    assert formal_plan["candidate_count"] == result.formal_plan["candidate_count"]
    assert p6_plan["candidate_count"] == result.formal_plan["candidate_count"]

    result.search_space["model"] = "polluted"
    result.search_space["axis_schema"]["free_axes"][0]["legal_widths"].clear()
    result.formal_plan["q_modes"].append("polluted")
    result.evidence["loaded_config"]["model"]["args"].clear()
    result.scanner_manifest["scanner_structural_axes"].clear()

    assert result.search_space["model"] == "pyramid_lidar"
    assert result.search_space["axis_schema"]["free_axes"][0]["legal_widths"] == [
        16, 24, 32, 40, 48, 56, 64
    ]
    assert result.formal_plan["q_modes"] == ["fp16", "int8"]
    assert "args" in result.evidence["loaded_config"]["model"]
    assert result.scanner_manifest["scanner_structural_axes"]

    copied = result.mutable_search_space()
    copied["model"] = "local-copy"
    assert result.search_space["model"] == "pyramid_lidar"
    assert copied["model"] == "local-copy"
