"""Behavioral and reproducibility tests for the public Stage4 selector."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from framework.stage4.cost_model_selection_v1 import (
    DEFAULT_CANDIDATES,
    derive_seed,
    encode_rows,
    grouped_folds,
    run_nested_selection,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "scripts" / "reproduce" / "cost_model_selection.py"


def _fixture_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": f"measurement-{index}",
            "group_id": f"model-width-{index}",
            "capability_profile_id": f"device-{index % 2}",
            "width": [16 + index * 8, 32 + index * 4, 64 + index * 2],
            "q_mode": "int8" if index % 2 else "fp32",
            "model": "codriving" if index % 2 else "fcooper",
            "latency_ms": 10.0 + index * 1.7,
            "energy_j": 2.0 + index * 0.3,
            "ap70": 0.45 + index * 0.01,
        }
        for index in range(5)
    ]


def _graph_features() -> list[dict[str, Any]]:
    return [
        {"group_id": f"model-width-{index}", "node_count": 100 + index, "edge_count": 200 + index}
        for index in range(5)
    ]


def _capability_profiles() -> list[dict[str, Any]]:
    return [
        {
            "capability_profile_id": f"device-{index}",
            "features": {"compute_tflops": 100.0 + index, "memory_gb": 24.0 + index},
        }
        for index in range(2)
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )


def test_grouped_folds_are_5_by_3_without_group_leakage_and_use_documented_seeds() -> None:
    """Catches split-count, group-boundary, or seed-derivation regressions."""
    rows = _fixture_rows()

    outer = grouped_folds(rows, n_splits=5, seed=20260716)

    assert len(outer) == 5
    for outer_index, fold in enumerate(outer):
        assert set(fold.train_groups).isdisjoint(fold.test_groups)
        inner_rows = [rows[index] for index in fold.train_indices]
        inner = grouped_folds(
            inner_rows,
            n_splits=3,
            seed=derive_seed(20260716, outer_index=outer_index, purpose="inner"),
        )
        assert len(inner) == 3
        assert all(set(part.train_groups).isdisjoint(part.test_groups) for part in inner)

    assert derive_seed(20260716, outer_index=2, candidate_index=1, inner_index=0) == 20260926
    assert derive_seed(20260716, outer_index=4, purpose="outer_fit") == 20261720


def test_encoding_excludes_targets_and_run_selects_predictors_deterministically() -> None:
    """Catches target leakage or unstable selection under an unchanged fixed seed."""
    rows = _fixture_rows()
    graph_features = _graph_features()
    profiles = _capability_profiles()

    encoded = encode_rows(rows, graph_features, profiles)
    assert not any("latency" in name or "energy" in name or "ap70" in name for name in encoded.feature_names)

    candidates = (
        "extra_trees_raw",
        "lgbm_l1_raw",
        "extra_trees_residual",
        "lgbm_l1_residual",
    )
    first = run_nested_selection(rows, graph_features, profiles, candidates=candidates)
    second = run_nested_selection(rows, graph_features, profiles, candidates=candidates)

    assert first == second
    assert all(len(target["outer_folds"]) == 5 for target in first["targets"].values())
    assert all(
        fold["selected_candidate"] in candidates
        for target in first["targets"].values()
        for fold in target["outer_folds"]
    )


def test_feature_encoding_accepts_unlabeled_candidates_with_identical_schema_and_order() -> None:
    """Catches shared feature encoding accidentally requiring measurement targets."""
    labeled = _fixture_rows()
    unlabeled = [
        {
            key: value
            for key, value in row.items()
            if key not in {"latency_ms", "energy_j", "ap70"}
        }
        for row in labeled
    ]

    labeled_encoding = encode_rows(labeled, _graph_features(), _capability_profiles())
    candidate_encoding = encode_rows(
        unlabeled, _graph_features(), _capability_profiles()
    )

    assert candidate_encoding.feature_names == labeled_encoding.feature_names
    assert candidate_encoding.matrix.tolist() == labeled_encoding.matrix.tolist()


@pytest.mark.parametrize(
    "target_value",
    [None, float("nan"), float("inf"), -float("inf")],
)
def test_training_entry_rejects_missing_or_nonfinite_targets(target_value: Any) -> None:
    """Catches moving target validation out of the actual training boundary."""
    rows = _fixture_rows()
    if target_value is None:
        rows[0].pop("latency_ms")
    else:
        rows[0]["latency_ms"] = target_value

    with pytest.raises(ValueError, match="latency_ms.*finite"):
        run_nested_selection(
            rows,
            _graph_features(),
            _capability_profiles(),
            candidates=("extra_trees_raw",),
        )


def test_selection_rejects_fewer_than_five_groups() -> None:
    """Catches silent reduction of the preregistered five-fold outer protocol."""
    with pytest.raises(ValueError, match="number of groups"):
        run_nested_selection(
            _fixture_rows()[:4],
            _graph_features()[:4],
            _capability_profiles(),
        )


def test_core_rejects_missing_duplicate_or_leaky_capability_context() -> None:
    """Catches missing/ambiguous profiles and capability target leakage."""
    rows = _fixture_rows()
    profiles = _capability_profiles()

    with pytest.raises(ValueError, match="missing feature context"):
        encode_rows(rows, _graph_features(), profiles[:1])
    with pytest.raises(ValueError, match="duplicate capability_profile_id"):
        encode_rows(rows, _graph_features(), [*profiles, profiles[0]])
    leaky = [*profiles]
    leaky[0] = {**leaky[0], "features": {"latency_ms": 1.0}}
    with pytest.raises(ValueError, match="label-like fields"):
        encode_rows(rows, _graph_features(), leaky)
    graph_leak = _graph_features()
    graph_leak[0]["observed_latency_ms"] = 1.0
    with pytest.raises(ValueError, match="label-like fields"):
        encode_rows(rows, graph_leak, profiles)


def test_core_validation_errors_do_not_disclose_measurement_or_context_identities() -> None:
    """Catches public validation errors that expose caller-supplied identities."""
    rows = _fixture_rows()
    rows[0].update(
        {
            "row_id": "secret-job-123",
            "group_id": "secret-group",
            "capability_profile_id": "secret-profile",
        }
    )
    rows[1]["row_id"] = "secret-job-123"
    with pytest.raises(ValueError) as duplicate_error:
        encode_rows(rows, _graph_features(), _capability_profiles())

    missing_rows = _fixture_rows()
    for row in missing_rows:
        row["group_id"] = "secret-group"
        row["capability_profile_id"] = "secret-profile"
    with pytest.raises(ValueError) as missing_error:
        encode_rows(missing_rows, [], [])

    duplicate_profiles = _capability_profiles()
    duplicate_profiles[0]["capability_profile_id"] = "secret-profile"
    duplicate_profiles.append(dict(duplicate_profiles[0]))
    with pytest.raises(ValueError) as profile_error:
        encode_rows(_fixture_rows(), _graph_features(), duplicate_profiles)

    for error in (duplicate_error, missing_error, profile_error):
        rendered = str(error.value)
        assert "secret-job-123" not in rendered
        assert "secret-group" not in rendered
        assert "secret-profile" not in rendered


@pytest.mark.parametrize(
    ("width", "expected_error"),
    [
        ([float("nan")], "finite numbers"),
        ([float("inf")], "finite numbers"),
        (["3"], "finite numbers"),
        ([], "one to five"),
        ([1, 2, 3, 4, 5, 6], "one to five"),
    ],
)
def test_core_rejects_invalid_width_before_feature_encoding(
    width: list[Any], expected_error: str
) -> None:
    """Catches invalid width axes before they can enter the feature matrix."""
    rows = _fixture_rows()
    rows[0]["width"] = width

    with pytest.raises(ValueError, match=expected_error):
        encode_rows(rows, _graph_features(), _capability_profiles())


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda rows: rows.__setitem__(0, {key: value for key, value in rows[0].items() if key != "row_id"}), "requires manifest_job_id or row_id"),
        (lambda rows: rows.__setitem__(0, ["not", "an", "object"]), "must be a JSON object"),
        (lambda rows: rows[0].__setitem__("energy_j", float("nan")), "must be finite"),
        (lambda rows: rows[0].__setitem__("width", [float("nan")]), "finite numbers"),
    ],
)
def test_cli_rejects_missing_identity_non_object_and_nonfinite_labels(
    tmp_path: Path, mutate: Any, expected_error: str
) -> None:
    """Catches malformed measurement records before any model fit can start."""
    measurements = tmp_path / "measurements.jsonl"
    graph_features = tmp_path / "graph-features.jsonl"
    profiles = tmp_path / "capability-profiles.jsonl"
    rows = _fixture_rows()
    mutate(rows)
    measurements.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_jsonl(graph_features, _graph_features())
    _write_jsonl(profiles, _capability_profiles())

    result = subprocess.run(
        [
            sys.executable, str(CLI),
            "--measurements", str(measurements),
            "--graph-features", str(graph_features),
            "--capability-profiles", str(profiles),
            "--measurements-provenance", "measured-v1",
            "--graph-features-provenance", "scanner-v1",
            "--capability-profiles-provenance", "hardware-profile-v1",
            "--output-root", str(tmp_path / "outputs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_cli_stderr_does_not_disclose_secret_input_identities(tmp_path: Path) -> None:
    """Catches CLI forwarding of row, group, or profile identities in public errors."""
    measurements = tmp_path / "measurements.jsonl"
    graph_features = tmp_path / "graph-features.jsonl"
    profiles = tmp_path / "capability-profiles.jsonl"
    rows = _fixture_rows()
    rows[0].update(
        {
            "row_id": "secret-job-123",
            "group_id": "secret-group",
            "capability_profile_id": "secret-profile",
        }
    )
    rows[1]["row_id"] = "secret-job-123"
    _write_jsonl(measurements, rows)
    _write_jsonl(graph_features, _graph_features())
    _write_jsonl(profiles, _capability_profiles())

    result = subprocess.run(
        [
            sys.executable, str(CLI),
            "--measurements", str(measurements),
            "--graph-features", str(graph_features),
            "--capability-profiles", str(profiles),
            "--measurements-provenance", "measured-v1",
            "--graph-features-provenance", "scanner-v1",
            "--capability-profiles-provenance", "hardware-profile-v1",
            "--output-root", str(tmp_path / "outputs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "duplicate measurement identity" in result.stderr
    assert "secret-job-123" not in result.stderr
    assert "secret-group" not in result.stderr
    assert "secret-profile" not in result.stderr


def test_cli_reads_three_jsonl_inputs_writes_stable_artifacts_and_records_provenance(
    tmp_path: Path,
) -> None:
    """Catches missing third input, non-deterministic artifacts, or incomplete manifest hashes."""
    measurements = tmp_path / "measurements.jsonl"
    graph_features = tmp_path / "graph-features.jsonl"
    profiles = tmp_path / "capability-profiles.jsonl"
    _write_jsonl(measurements, _fixture_rows())
    _write_jsonl(graph_features, _graph_features())
    _write_jsonl(profiles, _capability_profiles())
    output_root = tmp_path / "outputs"
    arguments = (
        "--measurements", str(measurements),
        "--graph-features", str(graph_features),
        "--capability-profiles", str(profiles),
        "--measurements-provenance", "measured-v1",
        "--graph-features-provenance", "scanner-v1",
        "--capability-profiles-provenance", "hardware-profile-v1",
        "--output-root", str(output_root),
    )

    _run_cli(*arguments)
    first = {name: (output_root / name).read_bytes() for name in ("report.json", "folds.csv", "manifest.json")}
    _run_cli(*arguments)
    second = {name: (output_root / name).read_bytes() for name in first}

    assert first == second
    manifest = json.loads(second["manifest.json"])
    report = json.loads(second["report.json"])
    assert manifest["inputs"]["capability_profiles"]["provenance"] == "hardware-profile-v1"
    assert manifest["inputs"]["capability_profiles"]["sha256"] == hashlib.sha256(profiles.read_bytes()).hexdigest()
    assert manifest["outputs"]["report.json"] == hashlib.sha256(second["report.json"]).hexdigest()
    assert manifest["outputs"]["folds.csv"] == hashlib.sha256(second["folds.csv"]).hexdigest()
    assert manifest["outputs"]["manifest.json"]["sha256_scope"] == "manifest_body_before_self_digest"
    assert set(report["candidate_specs"]) == set(DEFAULT_CANDIDATES)
    assert str(tmp_path) not in second["manifest.json"].decode("utf-8")
    with (output_root / "folds.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 15


def test_cli_rejects_invalid_jsonl_identity_labels_and_profile_references(tmp_path: Path) -> None:
    """Catches permissive JSONL ingestion that would corrupt a reported experiment."""
    measurements = tmp_path / "measurements.jsonl"
    graph_features = tmp_path / "graph-features.jsonl"
    profiles = tmp_path / "capability-profiles.jsonl"
    bad_rows = _fixture_rows()
    bad_rows[1]["row_id"] = bad_rows[0]["row_id"]
    bad_rows[2]["latency_ms"] = float("inf")
    _write_jsonl(measurements, bad_rows)
    _write_jsonl(graph_features, _graph_features())
    _write_jsonl(profiles, _capability_profiles())

    result = subprocess.run(
        [
            sys.executable, str(CLI),
            "--measurements", str(measurements),
            "--graph-features", str(graph_features),
            "--capability-profiles", str(profiles),
            "--measurements-provenance", "measured-v1",
            "--graph-features-provenance", "scanner-v1",
            "--capability-profiles-provenance", "hardware-profile-v1",
            "--output-root", str(tmp_path / "outputs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "duplicate measurement identity" in result.stderr


def test_cli_rejects_repository_root_and_symlinked_output_escape(tmp_path: Path) -> None:
    """Catches output roots that could write outside the caller's explicit directory."""
    measurements = tmp_path / "measurements.jsonl"
    graph_features = tmp_path / "graph-features.jsonl"
    profiles = tmp_path / "capability-profiles.jsonl"
    _write_jsonl(measurements, _fixture_rows())
    _write_jsonl(graph_features, _graph_features())
    _write_jsonl(profiles, _capability_profiles())
    common = [
        "--measurements", str(measurements),
        "--graph-features", str(graph_features),
        "--capability-profiles", str(profiles),
        "--measurements-provenance", "measured-v1",
        "--graph-features-provenance", "scanner-v1",
        "--capability-profiles-provenance", "hardware-profile-v1",
        "--output-root",
    ]
    repository_root = subprocess.run(
        [sys.executable, str(CLI), *common, str(REPOSITORY_ROOT)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    escaped = subprocess.run(
        [sys.executable, str(CLI), *common, str(symlink_parent / "outputs")],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    assert repository_root.returncode != 0
    assert "repository root" in repository_root.stderr
    assert escaped.returncode != 0
    assert "symlink" in escaped.stderr
