from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from framework.stage6 import paper_table_v1 as paper_table
from framework.stage6.paper_table_v1 import (  # type: ignore[import-not-found]
    DEFAULT_ARMS,
    PublicInputError,
    build_backend_tables,
    select_representative_points,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "scripts" / "reproduce" / "stage6_table.py"


def _stage6_cli_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage6_table_test_module", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells() -> list[dict[str, str]]:
    return [
        {"model": "model-a", "backend": "backend-x", "method": "method-1", "status": "complete"},
        {"model": "model-a", "backend": "backend-x", "method": "method-2", "status": "complete"},
    ]


def _row(
    evidence_id: str,
    *,
    model: str = "model-a",
    backend: str = "backend-x",
    method: str = "method-1",
    AP70: float = 80.0,
    latency_ms: float = 10.0,
    energy_j: float = 4.0,
    terminal_status: str = "measured_success_gold",
    independent_validation_passed: bool = True,
    evidence_sha_verified: bool = True,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "model": model,
        "backend": backend,
        "method": method,
        "AP70": AP70,
        "latency_ms": latency_ms,
        "energy_j": energy_j,
        "terminal_status": terminal_status,
        "independent_validation_passed": independent_validation_passed,
        "evidence_sha_verified": evidence_sha_verified,
    }


def _baseline(*, model: str = "model-a", backend: str = "backend-x", AP70: float = 80.0) -> dict[str, Any]:
    return _row(
        "baseline",
        model=model,
        backend=backend,
        method="original_default",
        AP70=AP70,
        latency_ms=20.0,
        energy_j=8.0,
    )


def _only_selected(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    report = select_representative_points(measurements, [_cells()[0]])
    assert report["cells"][0]["status"] == "selected"
    return report["cells"][0]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("delta", [0.099, 0.10])
def test_selects_lowest_latency_feasible_point_at_or_below_ap70_boundary(delta: float) -> None:
    """Catches treating the frozen inclusive AP70 feasibility boundary as exclusive."""
    selected = _only_selected(
        [_baseline(), _row("boundary", AP70=80.0 - delta, latency_ms=5.0)]
    )

    assert selected["representative"]["evidence_id"] == "boundary"


def test_rejects_point_outside_ap70_feasibility_boundary() -> None:
    """Catches selecting a faster point with AP70 loss greater than 0.10."""
    selected = select_representative_points(
        [_baseline(), _row("too-low", AP70=79.899, latency_ms=1.0)], [_cells()[0]]
    )["cells"][0]

    assert selected["status"] == "no_feasible_point"
    assert selected["representative"]["evidence_id"] == "too-low"
    assert selected["ap_constraint_violated"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminal_status", "measured_success_demo"),
        ("terminal_status", "measurement_failed"),
        ("independent_validation_passed", False),
        ("evidence_sha_verified", False),
    ],
)
def test_rejects_unclosed_evidence_before_selection(field: str, value: Any) -> None:
    """Catches a non-gold, failed, or SHA-unverified record entering the paper table."""
    rejected = _row("rejected", latency_ms=1.0)
    rejected[field] = value
    selected = _only_selected([_baseline(), _row("trusted", latency_ms=10.0), rejected])

    assert selected["representative"]["evidence_id"] == "trusted"


def test_uses_energy_tie_break_only_inside_one_percent_latency_window() -> None:
    """Catches using energy outside the frozen one-percent latency window."""
    selected = _only_selected(
        [
            _baseline(),
            _row("fast-high-energy", latency_ms=10.0, energy_j=10.0),
            _row("window-low-energy", latency_ms=10.1, energy_j=1.0),
            _row("outside-window", latency_ms=10.101, energy_j=0.01),
        ]
    )

    assert selected["representative"]["evidence_id"] == "window-low-energy"


def test_uses_stable_identity_tie_break_after_equal_frozen_keys() -> None:
    """Catches output changing with input order when frozen numeric selector keys tie."""
    rows = [
        _baseline(),
        _row("z-evidence", latency_ms=10.0, energy_j=1.0),
        _row("a-evidence", latency_ms=10.0, energy_j=1.0),
    ]

    first = _only_selected(rows)
    second = _only_selected([rows[0], *reversed(rows[1:])])

    assert first["representative"]["evidence_id"] == "a-evidence"
    assert second["representative"]["evidence_id"] == "a-evidence"


def test_never_falls_back_between_model_backend_or_method_cells() -> None:
    """Catches borrowing a trusted point from another requested cell."""
    report = select_representative_points(
        [_baseline(), _row("other-method", method="method-2", latency_ms=1.0)], _cells()
    )
    by_method = {cell["cell"]["method"]: cell for cell in report["cells"]}

    assert by_method["method-1"]["status"] == "no_trusted_point"
    assert by_method["method-1"]["representative"] is None
    assert by_method["method-2"]["representative"]["evidence_id"] == "other-method"


def test_reports_explicit_failure_cell_when_no_trusted_point_exists() -> None:
    """Catches silently omitting paper-table cells that lack trusted evidence."""
    report = select_representative_points(
        [_baseline(), _row("demo-only", terminal_status="measured_success_demo")], [_cells()[0]]
    )

    failure = report["cells"][0]
    assert failure["cell"] == _cells()[0]
    assert failure["status"] == "no_trusted_point"
    assert failure["outcome"] == "no_point_satisfies_ap_floor"
    assert failure["representative"] is None
    assert failure["ap_constraint_violated"] is False


@pytest.mark.parametrize("field", ["AP70", "latency_ms", "energy_j"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_measurement_values(field: str, value: float) -> None:
    """Catches NaN or infinity entering feasibility and tie-break comparisons."""
    invalid = _row("invalid")
    invalid[field] = value

    with pytest.raises(PublicInputError, match="finite"):
        select_representative_points([_baseline(), invalid], [_cells()[0]])


@pytest.mark.parametrize(
    ("field", "value"),
    [("AP70", -0.01), ("AP70", 100.01), ("latency_ms", 0.0), ("latency_ms", -1.0), ("energy_j", -0.01)],
)
def test_rejects_finite_metrics_outside_public_physical_ranges(field: str, value: float) -> None:
    """Catches finite but physically invalid values escaping public validation and crashing selection."""
    invalid = _row("invalid")
    invalid[field] = value

    with pytest.raises(PublicInputError, match="physical ranges"):
        select_representative_points([_baseline(), invalid], [_cells()[0]])


def test_rejects_duplicate_evidence_ids_and_missing_required_fields() -> None:
    """Catches ambiguous evidence bindings and incomplete evidence rows."""
    duplicate = [_baseline(), _row("same"), _row("same", latency_ms=9.0)]
    missing = _row("missing")
    del missing["backend"]

    with pytest.raises(PublicInputError, match="duplicate evidence_id"):
        select_representative_points(duplicate, [_cells()[0]])
    with pytest.raises(PublicInputError, match="missing required fields"):
        select_representative_points([_baseline(), missing], [_cells()[0]])


def test_selection_does_not_mutate_caller_inputs() -> None:
    """Catches in-place sorting or annotation of caller-owned measurement and cell records."""
    measurements = [_baseline(), _row("trusted"), _row("later", latency_ms=11.0)]
    cells = [_cells()[0]]
    original_measurements = copy.deepcopy(measurements)
    original_cells = copy.deepcopy(cells)

    select_representative_points(measurements, cells)

    assert measurements == original_measurements
    assert cells == original_cells


def test_public_errors_do_not_echo_raw_path_or_evidence_identity(tmp_path: Path) -> None:
    """Catches public validation errors disclosing caller path or private evidence identity."""
    secret = tmp_path / "private-identity.json"
    secret.write_text("{}", encoding="utf-8")
    bad = _row("secret-evidence-id")
    del bad["method"]

    with pytest.raises(PublicInputError) as error:
        select_representative_points([_baseline(), bad], [_cells()[0]])

    message = str(error.value)
    assert str(secret) not in message
    assert "secret-evidence-id" not in message


def test_frozen_backend_tables_preserve_statuses_ranks_and_closure_flags() -> None:
    """Catches drift in the migrated frozen table rows, ranks, or closure state."""
    complete_point = _row("complete", latency_ms=10.0, energy_j=2.0)
    failed_point = _row("failed", AP70=79.0, latency_ms=9.0, energy_j=1.0)
    arms = {
        DEFAULT_ARMS[0]: {"status": "complete", "independent_validation_complete": True, "points": [complete_point], "outer_genomes": 4, "failure_count": 1},
        DEFAULT_ARMS[1]: {"status": "complete_failure", "failure_evidence_sha_verified": True, "points": [failed_point], "failure_reason": "budget"},
        DEFAULT_ARMS[2]: {"status": "running", "points": []},
    }

    report = build_backend_tables(
        backend="backend-x", baseline=_baseline(), arms=arms, deltas=[0.10]
    )

    rows = report["tables"]["delta_0.10"]
    by_method = {row["method"]: row for row in rows}
    assert by_method[DEFAULT_ARMS[0]]["selection_status"] == "selected"
    assert by_method[DEFAULT_ARMS[1]]["selection_status"] == "feasibility_failure"
    assert by_method[DEFAULT_ARMS[1]]["ap_constraint_violated"] is None
    assert by_method[DEFAULT_ARMS[2]]["selection_status"] == "no_feasible_point"
    assert report["missing_arms"] == list(DEFAULT_ARMS[3:])
    assert report["nonterminal_arms"] == [DEFAULT_ARMS[2]]
    assert report["paper_ready"] is False
    assert by_method[DEFAULT_ARMS[0]]["latency_rank"] == 1

    with pytest.raises(ValueError, match="AP delta"):
        build_backend_tables(backend="backend-x", baseline=_baseline(), arms={}, deltas=[0])


def test_cli_run_uses_real_adapter_and_records_failure_cells(tmp_path: Path) -> None:
    """Catches CLI bypassing the real selector or omitting a no-trusted-point cell."""
    measurements = tmp_path / "measurements.json"
    evidence = tmp_path / "evidence.json"
    cells = tmp_path / "cells.json"
    output_root = tmp_path / "artifacts"
    _write_json(measurements, [_baseline(), _row("demo", terminal_status="measured_success_demo")])
    _write_json(evidence, [{"evidence_id": "baseline", "sha256_status": "verified"}, {"evidence_id": "demo", "sha256_status": "verified"}])
    _write_json(cells, [_cells()[0]])

    stage6_table = _stage6_cli_module()
    args = stage6_table._parse_args(
        [
            "--measurements", str(measurements),
            "--evidence", str(evidence),
            "--cells", str(cells),
            "--measurements-provenance", "measurements-v1",
            "--evidence-provenance", "evidence-v1",
            "--cells-provenance", "cells-v1",
            "--output-root", str(output_root),
        ]
    )
    report = stage6_table.run(args)

    assert report["cells"][0]["status"] == "no_trusted_point"
    assert (output_root / "paper_table.md").read_text(encoding="utf-8").count("|") >= 16


def test_cli_public_helpers_reject_unsafe_paths_invalid_provenance_and_bad_json(tmp_path: Path) -> None:
    """Catches a public CLI helper accepting unsafe paths, path-like provenance, or invalid JSON."""
    stage6_table = _stage6_cli_module()
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json}", encoding="utf-8")

    with pytest.raises(PublicInputError, match="filesystem or repository root"):
        stage6_table._safe_path("/", label="measurements", require_file=True)
    with pytest.raises(PublicInputError, match="non-path provenance"):
        stage6_table._provenance("private/path", label="measurements-provenance")
    with pytest.raises(PublicInputError, match="valid JSON"):
        stage6_table._load_records(bad_json, label="measurements")


def test_cli_main_converts_public_input_error_to_argument_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Catches CLI traceback disclosure instead of sanitized argparse failure output."""
    stage6_table = _stage6_cli_module()
    measurements = tmp_path / "measurements.json"
    evidence = tmp_path / "evidence.json"
    cells = tmp_path / "cells.json"
    _write_json(measurements, [_baseline(), _row("trusted")])
    _write_json(evidence, [{"evidence_id": "baseline", "sha256_status": "verified"}])
    _write_json(cells, [_cells()[0]])

    with pytest.raises(SystemExit, match="2"):
        stage6_table.main(
            [
                "--measurements", str(measurements),
                "--evidence", str(evidence),
                "--cells", str(cells),
                "--measurements-provenance", "measurements-v1",
                "--evidence-provenance", "evidence-v1",
                "--cells-provenance", "cells-v1",
                "--output-root", str(tmp_path / "out"),
            ]
        )

    stderr = capsys.readouterr().err
    assert "measurements reference undeclared evidence" in stderr
    assert str(measurements) not in stderr


def test_cli_main_sanitizes_invalid_physical_measurement_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a frozen-selection traceback exposing paths for a negative latency row."""
    stage6_table = _stage6_cli_module()
    measurements = tmp_path / "private-measurements.json"
    evidence = tmp_path / "evidence.json"
    cells = tmp_path / "cells.json"
    _write_json(measurements, [_baseline(), _row("private-evidence", latency_ms=-1.0)])
    _write_json(evidence, [{"evidence_id": "baseline", "sha256_status": "verified"}, {"evidence_id": "private-evidence", "sha256_status": "verified"}])
    _write_json(cells, [_cells()[0]])

    with pytest.raises(SystemExit, match="2"):
        stage6_table.main(
            [
                "--measurements", str(measurements),
                "--evidence", str(evidence),
                "--cells", str(cells),
                "--measurements-provenance", "measurements-v1",
                "--evidence-provenance", "evidence-v1",
                "--cells-provenance", "cells-v1",
                "--output-root", str(tmp_path / "out"),
            ]
        )

    stderr = capsys.readouterr().err
    assert "physical ranges" in stderr
    assert str(measurements) not in stderr
    assert "private-evidence" not in stderr


def test_cli_writes_idempotent_artifacts_and_binds_output_rows_to_evidence(tmp_path: Path) -> None:
    """Catches missing reproducibility artifacts or an unbound selected paper-table row."""
    measurements = tmp_path / "measurements.json"
    evidence = tmp_path / "evidence.jsonl"
    cells = tmp_path / "cells.json"
    output_root = tmp_path / "artifacts"
    trusted = _row("trusted", latency_ms=10.0)
    trusted.update(
        {
            "private_abs_path": "/private/release/measurement.json",
            "internal_note": "do-not-publish",
            "arbitrary_extra": {"unpublished": "identity"},
        }
    )
    _write_json(measurements, [_baseline(), trusted])
    evidence.write_text(json.dumps({"evidence_id": "baseline", "sha256_status": "verified"}) + "\n" + json.dumps({"evidence_id": "trusted", "sha256_status": "verified"}) + "\n", encoding="utf-8")
    _write_json(cells, [_cells()[0]])
    arguments = (
        "--measurements", str(measurements),
        "--evidence", str(evidence),
        "--cells", str(cells),
        "--measurements-provenance", "measurement-release-v1",
        "--evidence-provenance", "evidence-release-v1",
        "--cells-provenance", "cell-spec-v1",
        "--output-root", str(output_root),
    )

    first = _run_cli(*arguments)
    assert first.returncode == 0, first.stderr
    artifacts = ("paper_table.raw.json", "paper_table.csv", "paper_table.md", "manifest.json")
    first_bytes = {name: (output_root / name).read_bytes() for name in artifacts}
    assert _run_cli(*arguments).returncode == 0
    assert first_bytes == {name: (output_root / name).read_bytes() for name in artifacts}
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_rows"][0]["evidence_id"] == "trusted"
    assert manifest["inputs"]["measurements"]["provenance"] == "measurement-release-v1"
    assert manifest["outputs"]["paper_table.csv"]["sha256"] == hashlib.sha256(
        first_bytes["paper_table.csv"]
    ).hexdigest()
    manifest_body = dict(manifest)
    body_sha256 = manifest_body.pop("manifest_body_sha256")
    assert body_sha256 == hashlib.sha256(
        (json.dumps(manifest_body, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()
    for data in first_bytes.values():
        rendered = data.decode("utf-8")
        assert "/private/release/measurement.json" not in rendered
        assert "do-not-publish" not in rendered
        assert "unpublished" not in rendered
    assert "trusted" in first_bytes["paper_table.raw.json"].decode("utf-8")


def test_frozen_source_verification_rejects_unknown_or_drifted_sha() -> None:
    """Catches a CLI accepting an unrecognized frozen source identity or changed core contract."""
    with pytest.raises(PublicInputError, match="frozen source verification failed"):
        paper_table.verify_frozen_source("0" * 64)
    with pytest.raises(PublicInputError, match="frozen source verification failed"):
        paper_table.verify_frozen_source(
            paper_table.FROZEN_SOURCE_SHA256, contract_sha256="0" * 64
        )


def test_cli_fails_closed_when_frozen_source_identity_drifts(tmp_path: Path) -> None:
    """Catches the release CLI writing artifacts after its controlled frozen SHA changes."""
    stage6_table = _stage6_cli_module()
    measurements = tmp_path / "measurements.json"
    evidence = tmp_path / "evidence.json"
    cells = tmp_path / "cells.json"
    output_root = tmp_path / "artifacts"
    _write_json(measurements, [_baseline(), _row("trusted")])
    _write_json(evidence, [{"evidence_id": "baseline", "sha256_status": "verified"}, {"evidence_id": "trusted", "sha256_status": "verified"}])
    _write_json(cells, [_cells()[0]])
    stage6_table.FROZEN_SOURCE_SHA256 = "0" * 64
    args = stage6_table._parse_args(
        [
            "--measurements", str(measurements),
            "--evidence", str(evidence),
            "--cells", str(cells),
            "--measurements-provenance", "measurements-v1",
            "--evidence-provenance", "evidence-v1",
            "--cells-provenance", "cells-v1",
            "--output-root", str(output_root),
        ]
    )

    with pytest.raises(PublicInputError, match="frozen source verification failed"):
        stage6_table.run(args)

    assert not output_root.exists()


def test_cli_refuses_different_existing_output_and_sanitizes_errors(tmp_path: Path) -> None:
    """Catches destructive reruns and path/evidence leakage from public CLI errors."""
    secret_measurements = tmp_path / "secret-measurements.json"
    evidence = tmp_path / "evidence.json"
    cells = tmp_path / "cells.json"
    output_root = tmp_path / "artifacts"
    _write_json(secret_measurements, [_baseline(), _row("private-evidence-id")])
    _write_json(evidence, [{"evidence_id": "baseline", "sha256_status": "verified"}, {"evidence_id": "private-evidence-id", "sha256_status": "verified"}])
    _write_json(cells, [_cells()[0]])
    output_root.mkdir()
    (output_root / "paper_table.raw.json").write_text("different", encoding="utf-8")

    result = _run_cli(
        "--measurements", str(secret_measurements),
        "--evidence", str(evidence),
        "--cells", str(cells),
        "--measurements-provenance", "measurements-v1",
        "--evidence-provenance", "evidence-v1",
        "--cells-provenance", "cells-v1",
        "--output-root", str(output_root),
    )

    assert result.returncode != 0
    assert "refusing to overwrite different output: paper_table.raw.json" in result.stderr
    assert str(secret_measurements) not in result.stderr
    assert "private-evidence-id" not in result.stderr


def test_cli_produces_identical_artifacts_in_two_independent_output_roots(tmp_path: Path) -> None:
    """Catches artifact bytes depending on the caller's output directory."""
    measurements = tmp_path / "measurements.json"
    evidence = tmp_path / "evidence.json"
    cells = tmp_path / "cells.json"
    _write_json(measurements, [_baseline(), _row("trusted")])
    _write_json(evidence, [{"evidence_id": "baseline", "sha256_status": "verified"}, {"evidence_id": "trusted", "sha256_status": "verified"}])
    _write_json(cells, [_cells()[0]])
    fixed = (
        "--measurements", str(measurements),
        "--evidence", str(evidence),
        "--cells", str(cells),
        "--measurements-provenance", "measurements-v1",
        "--evidence-provenance", "evidence-v1",
        "--cells-provenance", "cells-v1",
    )
    left, right = tmp_path / "left", tmp_path / "right"

    assert _run_cli(*fixed, "--output-root", str(left)).returncode == 0
    assert _run_cli(*fixed, "--output-root", str(right)).returncode == 0
    assert {
        name: (left / name).read_bytes()
        for name in ("paper_table.raw.json", "paper_table.csv", "paper_table.md", "manifest.json")
    } == {
        name: (right / name).read_bytes()
        for name in ("paper_table.raw.json", "paper_table.csv", "paper_table.md", "manifest.json")
    }


def test_cli_rejects_symlink_or_repository_root_inputs(tmp_path: Path) -> None:
    """Catches path traversal through a symlink and dangerous repository-root output."""
    real = tmp_path / "real.json"
    link = tmp_path / "link.json"
    _write_json(real, [_baseline(), _row("trusted")])
    link.symlink_to(real)
    evidence = tmp_path / "evidence.json"
    cells = tmp_path / "cells.json"
    _write_json(evidence, [{"evidence_id": "baseline", "sha256_status": "verified"}, {"evidence_id": "trusted", "sha256_status": "verified"}])
    _write_json(cells, [_cells()[0]])

    result = _run_cli(
        "--measurements", str(link),
        "--evidence", str(evidence),
        "--cells", str(cells),
        "--measurements-provenance", "measurements-v1",
        "--evidence-provenance", "evidence-v1",
        "--cells-provenance", "cells-v1",
        "--output-root", str(REPOSITORY_ROOT),
    )

    assert result.returncode != 0
    assert "must not traverse a symlink or parent escape" in result.stderr
