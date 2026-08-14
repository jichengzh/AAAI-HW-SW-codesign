"""Behavioral and CLI tests for selection-only single-target Stage5 search."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5 import single_target_search_v2 as single

from .test_production_search import (
    closure,
    profiles,
    registry,
    source_lineage,
    training_data,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "scripts" / "reproduce" / "stage5_selection.py"


def _load_stage5_cli() -> Any:
    spec = importlib.util.spec_from_file_location("stage5_selection_cli", CLI)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load Stage5 selection CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage5_cli = _load_stage5_cli()


def task() -> single.SearchTask:
    return single.SearchTask(
        task_id="S5-PYR-TVM",
        target_model="pyramid",
        hardware_id="h800",
        capability_profile=profiles()[0],
        sample_budget=16,
        batch_size=4,
        round_count=4,
    )


def fcooper_registry() -> dict[str, Any]:
    schema = [
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    ]
    groups = []
    for width in ([64, 128, 256, 128, 256], [32, 64, 128, 64, 128]):
        structure = dict(zip(schema, width))
        group_id = "fcooper|" + "|".join(
            f"{name}={structure[name]}" for name in schema
        )
        groups.append(
            {
                "group_id": group_id,
                "model": "fcooper",
                "width": list(width),
                "width_schema": schema,
                "structure_widths": structure,
                **source_lineage(group_id, "fcooper", list(width)),
                "graph_features": {
                    "group_id": group_id,
                    "model": "fcooper",
                    "width": list(width),
                    "conv_count": 31,
                    "conv_macs": float(width[0] * width[1] * width[2]),
                },
            }
        )
    return {"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}


def candidate_predictions(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(candidate_rows):
        result.append(
            {
                **copy.deepcopy(row),
                "predictions": {
                    "latency_ms": 1.0 + index,
                    "energy_j": 0.2 + index / 10,
                    "ap70": 0.8 - index / 100,
                },
                "prediction_intervals": {
                    key: {"lower": value - 0.1, "median": value, "upper": value + 0.1}
                    for key, value in {
                        "latency_ms": 1.0 + index,
                        "energy_j": 0.2 + index / 10,
                        "ap70": 0.8 - index / 100,
                    }.items()
                },
            }
        )
    return result


def test_search_task_and_measurement_request_have_exact_deterministic_identity() -> None:
    """Catches task drift, row reordering, duplicate genomes, or request hash drift."""
    search_task = task()
    contract = single.validate_search_task(search_task)
    manifest = single.build_task_candidate_manifest(
        registry(), task=search_task, measured_row_ids=set()
    )
    selected = single.select_task_batch(
        candidate_predictions(manifest["rows"]),
        measured_rows=[],
        measured_graph_features=[],
        task=search_task,
    )
    before = copy.deepcopy(selected["selected_rows"])

    first = single.build_measurement_request(
        task=search_task, selected_rows=selected["selected_rows"], round_index=0
    )
    second = single.build_measurement_request(
        task=search_task, selected_rows=selected["selected_rows"], round_index=0
    )

    assert first == second
    assert first["task_sha256"] == contract["task_sha256"]
    assert first["measurement_request_sha256"]
    assert [row["row_id"] for row in first["rows"]] == selected["selected_row_ids"]
    assert len(first["row_sha256"]) == len(set(first["row_sha256"])) == 4
    assert selected["selected_rows"] == before

    mutated = copy.deepcopy(selected["selected_rows"])
    mutated[0]["task_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="task"):
        single.build_measurement_request(
            task=search_task, selected_rows=mutated, round_index=0
        )


def test_production_request_row_hash_binds_task_hardware_round_and_policy() -> None:
    """Catches replay of a candidate row under a different execution request."""
    search_task = task()
    task_contract = single.validate_search_task(search_task)
    manifest = single.build_task_candidate_manifest(
        registry(), task=search_task, measured_row_ids=set()
    )
    selection = {
        "selected_group_ids": [manifest["rows"][0]["group_id"]],
        "selected_rows": manifest["rows"][:4],
    }

    baseline = stage5_cli._build_production_request(
        task=search_task,
        task_contract=task_contract,
        selection=selection,
        round_index=0,
    )
    task_changed = replace(search_task, task_id="S5-PYR-TVM-OTHER")
    alternate_profile = build_capability_profile(
        capability_profile_id="h100-tvm",
        hardware_target="h100",
        compiler_fingerprint=hashlib.sha256(b"h100-tvm").hexdigest(),
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )
    hardware_changed = replace(
        search_task, hardware_id="h100", capability_profile=alternate_profile
    )
    variants = [
        stage5_cli._build_production_request(
            task=task_changed,
            task_contract=single.validate_search_task(task_changed),
            selection=selection,
            round_index=0,
        ),
        stage5_cli._build_production_request(
            task=hardware_changed,
            task_contract=single.validate_search_task(hardware_changed),
            selection=selection,
            round_index=0,
        ),
        stage5_cli._build_production_request(
            task=search_task,
            task_contract=task_contract,
            selection=selection,
            round_index=1,
        ),
    ]

    for row in baseline["rows"]:
        assert row["task_id"] == search_task.task_id
        assert row["task_sha256"] == task_contract["task_sha256"]
        assert row["hardware_id"] == search_task.hardware_id
        assert row["round_index"] == 0
        assert row["selection_policy"] == "predicted_frontier_diversity"
        canonical = json.dumps(
            row, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
        assert baseline["row_sha256"][row["row_id"]] == hashlib.sha256(
            canonical
        ).hexdigest()
    for variant in variants:
        assert all(
            variant["row_sha256"][row_id] != digest
            for row_id, digest in baseline["row_sha256"].items()
        )


def test_search_task_rejects_capability_or_budget_drift() -> None:
    """Catches dispatch/profile mismatch and changes to the frozen B=4,T=16 budget."""
    profile = build_capability_profile(
        capability_profile_id="h800-trt",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"trt").hexdigest(),
        dispatch_key="trt_engine",
        features={"int8_propagation": 1.0},
    )
    bad_budget = single.SearchTask(
        "S5-PYR-TRT", "pyramid", "h800", profile, sample_budget=20, batch_size=5
    )

    with pytest.raises(ValueError, match="B=4, T=16"):
        single.validate_search_task(bad_budget)


def test_single_target_selection_is_ordered_unique_and_rejects_nonfinite_values() -> None:
    """Catches unstable selected IDs, duplicates, and NaN/Inf objective ranking."""
    manifest = single.build_task_candidate_manifest(
        registry(), task=task(), measured_row_ids=set()
    )
    candidates = candidate_predictions(manifest["rows"])

    first = single.select_task_batch(
        candidates, measured_rows=[], measured_graph_features=[], task=task()
    )
    second = single.select_task_batch(
        list(reversed(candidates)),
        measured_rows=[],
        measured_graph_features=[],
        task=task(),
    )

    assert first["policy"] == "predicted_frontier_diversity"
    assert first["selected_row_ids"] == second["selected_row_ids"]
    assert len(first["selected_row_ids"]) == len(set(first["selected_row_ids"])) == 4

    malformed = copy.deepcopy(candidates)
    malformed[0]["prediction_intervals"]["latency_ms"]["upper"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        single.select_task_batch(
            malformed, measured_rows=[], measured_graph_features=[], task=task()
        )


@pytest.mark.parametrize(
    "graph_payload",
    [
        {},
        {"conv_count": 1},
        {"group_id": "different", "conv_count": 1},
        {"group_id": "pyramid|40x80x160", "observed_latency_ms": 1.0},
        {"group_id": "pyramid|40x80x160", "cache_status": 1.0},
        {"group_id": "pyramid|40x80x160", "terminal_status": 1.0},
        {"group_id": "pyramid|40x80x160", "conv_count": float("nan")},
    ],
)
def test_single_target_selection_rejects_invalid_candidate_graph_payload(
    graph_payload: dict[str, Any],
) -> None:
    """Catches absent identity, leakage, and malformed candidate graph context."""
    manifest = single.build_task_candidate_manifest(
        registry(), task=task(), measured_row_ids=set()
    )
    candidates = candidate_predictions(manifest["rows"])
    candidates[0]["graph_features"] = graph_payload

    with pytest.raises(ValueError, match="graph"):
        single.select_task_batch(
            candidates, measured_rows=[], measured_graph_features=[], task=task()
        )


@pytest.mark.parametrize(
    "measured_graph_features",
    [
        ["not-an-object"],
        [{"conv_count": 1}],
        [{"group_id": "measured", "model": "pyramid", "width": [1, 2, 3]}],
        [{"group_id": "measured", "observed_latency_ms": 1.0}],
        [{"group_id": "measured", "cache_status": 1.0}],
        [{"group_id": "measured", "terminal_status": 1.0}],
        [{"group_id": "measured", "conv_count": float("nan")}],
    ],
)
def test_single_target_selection_validates_direct_measured_graph_features(
    measured_graph_features: list[Any],
) -> None:
    """Catches ignored malformed or label-derived measured graph records."""
    manifest = single.build_task_candidate_manifest(
        registry(), task=task(), measured_row_ids=set()
    )

    with pytest.raises(ValueError, match="graph"):
        single.select_task_batch(
            candidate_predictions(manifest["rows"]),
            measured_rows=[],
            measured_graph_features=measured_graph_features,
            task=task(),
        )


@pytest.mark.parametrize(
    "graph_payload",
    [
        {},
        {"conv_count": 1},
        {"group_id": "different", "conv_count": 1},
        {"group_id": "pyramid|16x32x64", "observed_latency_ms": 1.0},
        {"group_id": "pyramid|16x32x64", "cache_status": 1.0},
        {"group_id": "pyramid|16x32x64", "terminal_status": 1.0},
        {"group_id": "pyramid|16x32x64", "conv_count": float("nan")},
    ],
)
def test_single_target_selection_rejects_invalid_nested_measured_row_graph(
    graph_payload: dict[str, Any],
) -> None:
    """Catches measured-row graph leakage before maximin vector construction."""
    manifest = single.build_task_candidate_manifest(
        registry(), task=task(), measured_row_ids=set()
    )
    measured_row = {
        "group_id": "pyramid|16x32x64",
        "model": "pyramid",
        "width": [16, 32, 64],
        "q_mode": "fp16",
        "graph_features": graph_payload,
    }

    with pytest.raises(ValueError, match="graph"):
        single.select_task_batch(
            candidate_predictions(manifest["rows"]),
            measured_rows=[measured_row],
            measured_graph_features=[],
            task=task(),
        )


def test_fcooper_task_and_candidate_manifest_preserve_five_axis_genome() -> None:
    """Catches regression from scanner-derived F-Cooper identity to a triplet."""
    profile = profiles()[1]
    search_task = single.SearchTask(
        "S5-FCO-TRT", "fcooper", "h800", profile
    )

    contract = single.validate_search_task(search_task)
    manifest = single.build_task_candidate_manifest(
        fcooper_registry(), task=search_task, measured_row_ids=set()
    )

    assert contract["genome_schema"] == [
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
        "q_mode",
    ]
    assert all(len(row["genome"]) == 6 for row in manifest["rows"])


def test_online_single_target_refit_preserves_real_heads_seed_and_calibration() -> None:
    """Catches single-target refits diverging from frozen production heads."""
    rows, graphs = training_data()

    bundle = single.fit_online_bundle(rows, graphs, profiles(), seed=37)

    assert bundle.manifest["seed"] == 37
    assert bundle.manifest["canonical_value_heads"] == {
        "latency_ms": "extra_trees_log",
        "energy_j": "extra_trees_log",
        "ap70": "lgbm_huber_residual",
    }
    assert bundle.manifest["uncertainty_policy"] == (
        "lgbm_quantile_plus_group_conformal"
    )
    assert bundle.value_heads["latency_ms"].random_state == 37
    assert bundle.value_heads["ap70"].random_state == 37
    assert set(bundle.interval_heads) == {"latency_ms", "energy_j", "ap70"}


def test_initial_single_target_fit_accepts_only_gold176_coldstart_rows() -> None:
    """Catches restoring legacy four-arm fitting or admitting online rows at round zero."""
    profile = build_capability_profile(
        capability_profile_id="h800-tvm-auto",
        hardware_target="h800",
        compiler_fingerprint="a" * 64,
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.0, "qdq_fold": 0.0},
    )
    rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for index in range(176):
        width = [16 + index % 7 * 8, 32 + index % 8 * 8, 64 + index % 9 * 8]
        group_id = f"gold-{index:03d}"
        q_mode = "int8" if index % 2 else "fp16"
        row_id = f"{group_id}|q={q_mode}|profile=h800-tvm-auto"
        rows.append(
            {
                "manifest_job_id": row_id,
                "row_id": row_id,
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "dispatch_key": "tvm_auto",
                "capability_profile_id": "h800-tvm-auto",
                "q_mode": q_mode,
                "latency_ms": 2.0 + index * 0.01,
                "energy_j": 0.5 + index * 0.005,
                "ap30": 0.90,
                "ap50": 0.80,
                "ap70": 0.70 - index * 0.0001,
                "terminal_status": "measured_success_gold",
                "training_source": "initial_coldstart",
            }
        )
        graphs.append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "conv_count": 27,
                "conv_macs": float(width[0] * width[1] * width[2]),
                "group_conv_count": 3,
            }
        )

    bundle = single.fit_initial_coldstart_bundle(rows, graphs, [profile], seed=41)

    assert bundle.manifest["schema_version"] == "stage5_initial_coldstart_model_bundle_v2"
    assert bundle.manifest["training_view_policy"] == "initial_coldstart_only"
    assert bundle.manifest["input_row_count"] == 176
    assert bundle.manifest["value_training_row_count"] == 176
    assert bundle.manifest["seed"] == 41
    assert bundle.value_heads["latency_ms"].random_state == 41

    contaminated = copy.deepcopy(rows)
    contaminated[-1]["training_source"] = "online_feedback"
    with pytest.raises(ValueError, match="initial_coldstart"):
        single.fit_initial_coldstart_bundle(contaminated, graphs, [profile], seed=41)

    for metric, invalid_value in (
        ("latency_ms", 0.0),
        ("energy_j", -0.1),
        ("ap30", float("nan")),
        ("ap50", -0.01),
        ("ap70", 1.01),
    ):
        invalid = copy.deepcopy(rows)
        invalid[-1][metric] = invalid_value
        with pytest.raises(ValueError, match="Gold176"):
            single.fit_initial_coldstart_bundle(invalid, graphs, [profile], seed=41)


def test_coldstart_and_feedback_views_are_detached_and_task_bound() -> None:
    """Catches mutation, wrong coldstart cardinality, or cross-task feedback drift."""
    coldstart = [
        {"manifest_job_id": f"gold-{index}", "training_source": "initial_coldstart"}
        for index in range(176)
    ]
    frozen = single.freeze_initial_coldstart(coldstart)
    search_task = task()
    contract = single.validate_search_task(search_task)
    feedback = [
        {
            "manifest_job_id": f"feedback-{index}",
            "task_id": search_task.task_id,
            "model": search_task.target_model,
            "hardware_id": search_task.hardware_id,
            "capability_profile_id": contract["capability_profile_id"],
            "dispatch_key": contract["dispatch_key"],
            "task_sha256": contract["task_sha256"],
            "training_source": "online_feedback",
            "terminal_status": "measured_success_gold",
        }
        for index in range(4)
    ]

    audit = single.validate_task_feedback_history(
        feedback, task=search_task, completed_rounds=1
    )

    assert frozen == coldstart
    assert frozen[0] is not coldstart[0]
    assert audit["feedback_rows"] == 4
    drifted = copy.deepcopy(feedback)
    drifted[0]["model"] = "codriving"
    with pytest.raises(ValueError, match="model drift"):
        single.validate_task_feedback_history(
            drifted, task=search_task, completed_rounds=1
        )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def cli_fixture(tmp_path: Path) -> tuple[list[str], Path]:
    rows, graphs = training_data()
    inputs = {
        "measurements": tmp_path / "measurements.jsonl",
        "graph_features": tmp_path / "graph-features.jsonl",
        "capability_profiles": tmp_path / "capability-profiles.json",
        "stage4_closure": tmp_path / "stage4-closure.json",
        "candidate_registry": tmp_path / "candidate-registry.json",
        "search_task": tmp_path / "search-task.json",
    }
    write_jsonl(inputs["measurements"], rows)
    write_jsonl(inputs["graph_features"], graphs)
    write_json(inputs["capability_profiles"], profiles())
    write_json(inputs["stage4_closure"], closure())
    write_json(inputs["candidate_registry"], registry())
    profile = profiles()[0]
    write_json(
        inputs["search_task"],
        {
            "task_id": "S5-PYR-TVM",
            "target_model": "pyramid",
            "hardware_id": "h800",
            "capability_profile_id": profile["capability_profile_id"],
            "sample_budget": 16,
            "batch_size": 4,
            "round_count": 4,
            "round_index": 0,
        },
    )
    output_root = tmp_path / "outputs"
    arguments: list[str] = []
    for name, path in inputs.items():
        option = "--" + name.replace("_", "-")
        arguments.extend([option, str(path), f"{option}-provenance", f"public-{name}-v1"])
    arguments.extend(["--seed", "91", "--output-root", str(output_root)])
    return arguments, output_root


def run_cli(arguments: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def test_selection_only_cli_is_byte_stable_and_records_complete_lineage(
    tmp_path: Path,
) -> None:
    """Catches hidden lineage, nondeterministic IDs, path leakage, or execution creep."""
    arguments, output_root = cli_fixture(tmp_path)

    assert stage5_cli.main(arguments) == 0
    first = {path.name: path.read_bytes() for path in output_root.iterdir()}
    second_root = tmp_path / "outputs-second-run"
    second_arguments = list(arguments)
    second_arguments[second_arguments.index("--output-root") + 1] = str(second_root)
    assert stage5_cli.main(second_arguments) == 0
    second = {path.name: path.read_bytes() for path in second_root.iterdir()}

    assert first == second
    assert set(second) == {
        "selected_ids.json",
        "measurement_request.json",
        "manifest.json",
    }
    manifest = json.loads(second["manifest.json"])
    request = json.loads(second["measurement_request.json"])
    selected = json.loads(second["selected_ids.json"])
    assert set(manifest["inputs"]) == {
        "measurements",
        "graph_features",
        "capability_profiles",
        "stage4_closure",
        "candidate_registry",
        "search_task",
    }
    assert set(manifest["expected_module_sha256"]) == {
        "canonical_search_v3.py",
        "genome_contract_v1.py",
        "production_search_v1.py",
        "single_target_search_v2.py",
    }
    assert set(manifest["current_module_sha256"]) == {
        "canonical_search_v3.py",
        "genome_contract_v1.py",
        "production_search_v1.py",
        "single_target_search_v2.py",
    }
    assert manifest["expected_module_sha256"] == manifest["current_module_sha256"]
    assert set(manifest["migration_lineage_module_sha256"]) == set(
        manifest["expected_module_sha256"]
    )
    assert manifest["selection_interface"] == (
        "framework.stage5.production_search_v1.select_predicted_frontier_diversity"
    )
    assert manifest["selection_policy"] == "predicted_frontier_diversity"
    assert manifest["task_sha256"] == request["task_sha256"]
    assert manifest["measurement_request_sha256"] == request[
        "measurement_request_sha256"
    ]
    assert selected["ordered_selected_ids"] == request["selected_group_ids"]
    assert selected["ordered_selected_row_ids"] == [
        row["row_id"] for row in request["rows"]
    ]
    assert request["hardware_execution"] == "external"
    assert request["execution_performed"] is False
    source_by_group = {
        group["group_id"]: group
        for group in json.loads(
            Path(arguments[arguments.index("--candidate-registry") + 1]).read_text(
                encoding="utf-8"
            )
        )["groups"]
    }
    for row in request["rows"]:
        source = source_by_group[row["group_id"]]
        assert row["source_status"] == source["source_status"]
        assert row["source_contract"] == source["source_contract"]
        assert row["source_evidence_sha256"] == source["source_evidence_sha256"]
    assert str(tmp_path) not in second["manifest.json"].decode()
    assert not {
        "timestamp",
        "hostname",
        "username",
        "absolute_path",
        "tvm_result",
        "tensorrt_result",
        "ap_result",
        "latency_result",
        "energy_result",
    } & set(manifest)


def test_selection_only_cli_fails_closed_when_source_identity_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches publication when a frozen Stage5 module no longer matches its SHA256."""
    arguments, output_root = cli_fixture(tmp_path)
    expected = dict(getattr(stage5_cli, "EXPECTED_MODULE_SHA256", {}))
    expected["canonical_search_v3.py"] = "0" * 64
    monkeypatch.setattr(stage5_cli, "EXPECTED_MODULE_SHA256", expected, raising=False)

    assert stage5_cli.main(arguments) == 2
    assert "source identity verification failed" in capsys.readouterr().err
    assert not (output_root / "manifest.json").exists()


def test_selection_only_cli_fails_closed_on_missing_lineage_and_safe_output_rules(
    tmp_path: Path,
) -> None:
    """Catches implicit lineage defaults or writes outside caller-owned output."""
    arguments, _ = cli_fixture(tmp_path)
    missing_provenance = arguments.copy()
    index = missing_provenance.index("--stage4-closure-provenance")
    del missing_provenance[index : index + 2]

    missing = run_cli(missing_provenance)
    assert missing.returncode != 0
    assert "stage4-closure-provenance" in missing.stderr

    repository_root_args = arguments[:-1] + [str(REPOSITORY_ROOT)]
    root_result = run_cli(repository_root_args)
    assert root_result.returncode != 0
    assert "repository root" in root_result.stderr

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(real_parent, target_is_directory=True)
    symlink_args = arguments[:-1] + [str(symlink / "output")]
    symlink_result = run_cli(symlink_args)
    assert symlink_result.returncode != 0
    assert "symlink" in symlink_result.stderr


def test_selection_only_cli_rejects_invalid_seed_without_echoing_raw_input(
    tmp_path: Path,
) -> None:
    """Catches argparse disclosure of caller-supplied sensitive seed text."""
    arguments, _ = cli_fixture(tmp_path)
    arguments[arguments.index("--seed") + 1] = "secret-seed-path"

    result = run_cli(arguments)

    assert result.returncode != 0
    assert "seed must be" in result.stderr
    assert "secret-seed-path" not in result.stderr


def test_selection_only_cli_rejects_candidate_labels_cache_status_and_sensitive_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches pre-selection access to labels/cache/status and sensitive error echoes."""
    arguments, _ = cli_fixture(tmp_path)
    registry_path = Path(arguments[arguments.index("--candidate-registry") + 1])
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["groups"][0]["latency_ms"] = 0.01
    payload["groups"][0]["cache_status"] = "secret-cache-hit"
    write_json(registry_path, payload)

    result = stage5_cli.main(arguments)
    captured = capsys.readouterr()

    assert result != 0
    assert "forbidden candidate field" in captured.err
    assert "secret-cache-hit" not in captured.err
    assert str(registry_path) not in captured.err


def test_selection_only_cli_fails_closed_when_candidate_source_identity_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches source identity rewriting or selection without verified lineage."""
    arguments, _ = cli_fixture(tmp_path)
    registry_path = Path(arguments[arguments.index("--candidate-registry") + 1])
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["groups"][0].pop("source_evidence_sha256")
    write_json(registry_path, payload)

    result = stage5_cli.main(arguments)
    captured = capsys.readouterr()

    assert result != 0
    assert "selection contract validation failed" in captured.err


@pytest.mark.parametrize(
    "mutation",
    [
        lambda group: group.update(source_contract="secret-contract-string"),
        lambda group: group["source_contract"].pop("schema_version"),
        lambda group: group["source_contract"].update(group_id="secret-drift"),
        lambda group: group.update(source_contract_sha256="0" * 64),
        lambda group: group["source_contract"].update(
            source_evidence_sha256=hashlib.sha256(b"secret-evidence-drift").hexdigest()
        ),
    ],
)
def test_selection_only_cli_fails_closed_on_source_contract_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation,
) -> None:
    """Catches CLI acceptance or disclosure of malformed source contracts."""
    arguments, _ = cli_fixture(tmp_path)
    registry_path = Path(arguments[arguments.index("--candidate-registry") + 1])
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    mutation(payload["groups"][0])
    write_json(registry_path, payload)

    result = stage5_cli.main(arguments)
    captured = capsys.readouterr()

    assert result != 0
    assert captured.err == "error: selection contract validation failed\n"
    assert "secret" not in captured.err


def test_selection_only_cli_accepts_capability_profiles_jsonl(
    tmp_path: Path,
) -> None:
    """Catches advertised JSONL profile ingestion silently accepting only arrays."""
    arguments, _ = cli_fixture(tmp_path)
    profiles_path = Path(arguments[arguments.index("--capability-profiles") + 1])
    write_jsonl(profiles_path, profiles())
    output_root = tmp_path / "jsonl-profile-output"
    arguments[arguments.index("--output-root") + 1] = str(output_root)

    assert stage5_cli.main(arguments) == 0
    assert (output_root / "manifest.json").is_file()


def test_selection_only_cli_refuses_different_content_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches silent replacement of already-published selection artifacts."""
    arguments, output_root = cli_fixture(tmp_path)
    run_cli(arguments, check=True)
    (output_root / "selected_ids.json").write_text('{"different":true}\n', encoding="utf-8")

    result = stage5_cli.main(arguments)
    captured = capsys.readouterr()

    assert result != 0
    assert "overwrite" in captured.err
