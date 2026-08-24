"""End-to-end contract for the explicit, deterministic Stage2 demo CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.stage2.formal_software_test_support import with_scanner_owned_contract


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREPARE = REPOSITORY_ROOT / "scripts" / "prepare_stage2_demo_data.py"
OPTIMIZE = REPOSITORY_ROOT / "scripts" / "stage2_optimize_model.py"
B4 = ["-m", "framework.run_b4_ablation"]
PQS = ["-m", "framework.run_pqs_ablation"]
CODRIVING = ["-m", "framework.run_pqs_codriving"]
PHASE_B4 = REPOSITORY_ROOT / "scripts" / "phase2" / "b4_integrate.py"
PHASE_B5 = REPOSITORY_ROOT / "scripts" / "phase2" / "b5_verify_convergence.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )


def _upgrade_to_test_only_canonical_formal_scanner_fixture(
    demo_root: Path,
) -> None:
    """Upgrade legacy diagnostic demo output only inside this test's temp root.

    ``prepare_stage2_demo_data.py`` remains a legacy diagnostic demo generator,
    not a paper-space reproduction entry point and not a scanner-axis producer.
    """
    manifest_path = demo_root / "framework/partitions/pyramid_lidar_partition.yaml"
    classification_path = demo_root / "results/model_classifier.json"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    migrated = with_scanner_owned_contract(manifest, "pyramid")
    manifest_path.write_text(
        yaml.safe_dump(migrated, sort_keys=False), encoding="utf-8"
    )
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    record = next(
        row for row in classification["models"] if row["model"] == "pyramid_lidar"
    )
    record["manifest_digest"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    classification_path.write_text(
        json.dumps(classification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_demo_requires_an_explicit_safe_output_root(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _run(str(PREPARE))

    unsafe = subprocess.run(
        [sys.executable, str(PREPARE), "--output-root", str(REPOSITORY_ROOT)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert unsafe.returncode != 0
    assert "repository root" in unsafe.stderr

    non_demo = tmp_path / "not-demo"
    non_demo.mkdir()
    (non_demo / "user-file.txt").write_text("preserve me", encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, str(PREPARE), "--output-root", str(non_demo)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "existing non-demo directory" in rejected.stderr
    assert (non_demo / "user-file.txt").read_text(encoding="utf-8") == "preserve me"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    escaped = subprocess.run(
        [sys.executable, str(PREPARE), "--output-root", str(symlink_parent / "demo")],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert escaped.returncode != 0
    assert "symlink" in escaped.stderr


def test_demo_pipeline_is_schema_valid_and_byte_stable_across_two_runs(tmp_path: Path) -> None:
    demo_root = tmp_path / "demo"
    out_json = tmp_path / "out" / "stage2.json"
    command = [
        str(OPTIMIZE),
        "--manifest",
        str(demo_root / "framework/partitions/pyramid_lidar_partition.yaml"),
        "--classification",
        str(demo_root / "results/model_classifier.json"),
        "--out-json",
        str(out_json),
    ]

    _run(str(PREPARE), "--output-root", str(demo_root))
    manifest_path = demo_root / "framework/partitions/pyramid_lidar_partition.yaml"
    classification_path = demo_root / "results/model_classifier.json"
    legacy_manifest = manifest_path.read_bytes()
    legacy_classification = classification_path.read_bytes()
    _upgrade_to_test_only_canonical_formal_scanner_fixture(demo_root)
    _run(*command)
    first_bytes = out_json.read_bytes()

    manifest_path.write_bytes(legacy_manifest)
    classification_path.write_bytes(legacy_classification)
    _run(str(PREPARE), "--output-root", str(demo_root))
    _upgrade_to_test_only_canonical_formal_scanner_fixture(demo_root)
    _run(*command)
    second_bytes = out_json.read_bytes()
    output = json.loads(second_bytes)

    assert output["schema"] == "stage2_output_v1"
    assert output["search_space_summary"]["n_software_candidates"] == 1
    assert output["optimization_status"]["allowed"] is True
    assert output["optimization_status"]["mode"] == "joint"
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(second_bytes).hexdigest()


@pytest.mark.parametrize("command", [B4, PQS, CODRIVING, [str(PHASE_B4)], [str(PHASE_B5)]])
def test_published_stage2_clis_require_an_explicit_output_root(command: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, *command], cwd=REPOSITORY_ROOT, capture_output=True, text=True
    )

    assert result.returncode != 0
    assert "output-root" in result.stderr


def test_search_clis_write_only_to_explicit_temp_roots(tmp_path: Path) -> None:
    demo_root = tmp_path / "demo"
    _run(str(PREPARE), "--output-root", str(demo_root))
    common = ["--seeds", "1", "--budget", "4", "--pop", "2", "--quiet"]
    commands = [
        (
            B4
            + [
                *common,
                "--starts",
                "1",
                "--input-root",
                str(demo_root / "results"),
                "--output-root",
                str(tmp_path / "b4"),
            ],
            tmp_path / "b4" / "results" / "b4_ablation_results.json",
        ),
        (
            PQS
            + [
                *common,
                "--manifest",
                str(demo_root / "framework/partitions/pyramid_lidar_partition.yaml"),
                "--input-root",
                str(demo_root / "results"),
                "--output-root",
                str(tmp_path / "pqs"),
            ],
            tmp_path / "pqs" / "results" / "pqs_ablation_results.json",
        ),
        (
            CODRIVING
            + [
                *common,
                "--manifest",
                str(demo_root / "framework/partitions/codriving_partition.yaml"),
                "--input-root",
                str(demo_root / "results"),
                "--output-root",
                str(tmp_path / "codriving"),
            ],
            tmp_path / "codriving" / "results" / "coupling_map" / "C0c_codriving_pqs.json",
        ),
    ]

    for command, expected_output in commands:
        _run(*command)
        assert expected_output.is_file()

    _run(
        str(PHASE_B5),
        "--input-root",
        str(demo_root / "results"),
        "--output-root",
        str(tmp_path / "b5"),
    )
    assert (tmp_path / "b5" / "results" / "b5_convergence_verification.json").is_file()

    (demo_root / "results" / "lut_results_grid.csv").write_text(
        "label,default_us,tuned_us\n", encoding="utf-8"
    )
    (demo_root / "results" / "ap70_depgraph_expansion.json").write_text(
        json.dumps({"results": []}), encoding="utf-8"
    )
    _run(
        str(PHASE_B4),
        "--input-root",
        str(demo_root / "results"),
        "--output-root",
        str(tmp_path / "phase-b4"),
    )
    assert (tmp_path / "phase-b4" / "results" / "latency_lut_pyramid.json").is_file()


def test_b4_integration_import_has_no_output_side_effect(tmp_path: Path) -> None:
    output_root = tmp_path / "must-stay-empty"
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.phase2.b4_integrate"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not output_root.exists()
