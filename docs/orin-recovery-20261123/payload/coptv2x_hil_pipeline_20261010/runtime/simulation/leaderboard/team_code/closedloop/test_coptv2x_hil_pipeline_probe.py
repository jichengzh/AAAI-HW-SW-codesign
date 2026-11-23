"""CPU-only checks for the paired scheduling probe; not Orin evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import pytest


ROOT = Path(__file__).resolve().parents[4]


def load_probe():
    path = ROOT / "scripts/coptv2x_hil_pipeline_probe.py"
    assert path.is_file(), "paired pipeline probe is not implemented"
    spec = importlib.util.spec_from_file_location("pipeline_probe", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_rows():
    return [{"sequence_no": i, "status": "completed", "stages_ns": {"infer": v},
             "timestamps_ns": {"submitted": i * 10, "deliver_end": end}}
            for i, (v, end) in enumerate(((100, 100), (10, 130), (20, 150)))]


def test_actual_source_frame_gaps_determine_schedule_not_sequence_number():
    probe = load_probe()
    samples = [{"source_frame": n} for n in (1, 2, 4, 6, 12)]
    assert probe.offered_schedule_ns(samples, "source", 50) == [0, 50000000, 150000000, 250000000, 550000000]
    assert probe.offered_schedule_ns(samples, "saturated", 50) == [0] * 5


@pytest.mark.parametrize("frames,step", [([1, 1], 50), ([2, 1], 50), ([1], 0), ([True], 50)])
def test_invalid_source_schedule_is_rejected(frames, step):
    with pytest.raises(ValueError):
        load_probe().offered_schedule_ns([{"source_frame": n} for n in frames], "source", step)


def test_summary_reports_intervals_not_latencies_and_excludes_first_from_warm():
    summary = load_probe().summarize_window(sample_rows(), 0, 150, 3, 2)
    assert summary["completed"] == 3
    assert summary["failed"] == 0
    assert summary["elapsed_ns"] == 150
    assert summary["completion_intervals_ns"] == [30, 20]
    assert summary["warm_stages_ns"]["infer"]["mean_ns"] == 15
    assert summary["warm_sample_count"] == 2
    assert summary["inflight_peak"] == 2
    assert summary["throughput_per_second"] == 20000000


def test_single_and_failed_windows_have_no_nan_or_invented_intervals():
    probe = load_probe()
    single = probe.summarize_window(sample_rows()[:1], 0, 100, 1, 1)
    assert single["warm_stages_ns"] == {}
    assert single["completion_intervals_ns"] == []
    assert single["warm_sample_count"] == 0
    failed = probe.summarize_window([{"status": "failed", "error": "bad"}], 0, 100, 3, 1)
    assert (failed["submitted"], failed["completed"], failed["failed"], failed["unsubmitted"]) == (1, 0, 1, 2)
    assert failed["throughput_per_second"] == 0
    json.dumps(failed, allow_nan=False)


def test_out_of_order_completion_is_not_silently_sorted():
    rows = sample_rows()
    rows[1]["timestamps_ns"]["deliver_end"] = 90
    with pytest.raises(ValueError, match="order"):
        load_probe().summarize_window(rows, 0, 150, 3, 2)


def test_increasing_completion_times_do_not_hide_wrong_sequence_order():
    rows = sample_rows()
    rows[1]["sequence_no"], rows[2]["sequence_no"] = 2, 1
    with pytest.raises(ValueError, match="order"):
        load_probe().summarize_window(rows, 0, 150, 3, 2)


def test_timestamp_union_does_not_sum_overlapping_intervals():
    probe = load_probe()
    assert probe.interval_union_ns([(10, 30), (20, 50), (60, 70)]) == 50
    assert probe.interval_union_ns([]) == 0
    with pytest.raises(ValueError):
        probe.interval_union_ns([(30, 10)])


def test_pair_rejects_wrong_source_artifact_shape_and_independent_hashes():
    probe = load_probe()
    row = {"arm_id": "compression_only", "sample_id": "s1", "sequence_no": 0,
           "run_id": "run", "route_id": "route", "scenario_id": "scenario",
           "source_frame": 12, "source_sim_time_ns": 600000000,
           "input_provenance": {"x": {"sha256": "a" * 64}},
           "artifact_identity": {"engine_sha256": "b" * 64}, "model": {"id": "frozen"},
           "output_contract": {"res0": {"shape": [1, 2], "dtype": "float32"}},
           "payload_semantics": "rsu", "decoded_feature_sha256": "c" * 64,
           "wire_payload_sha256": "d" * 64}
    assert probe.compare_pair(row, copy.deepcopy(row))["matched"] is True
    for name in row:
        wrong = dict(row, **{name: "wrong"})
        with pytest.raises(ValueError, match=name):
            probe.compare_pair(row, wrong)


def test_parser_and_direct_script_import_are_cwd_independent(tmp_path):
    probe = load_probe()
    args = probe.build_parser().parse_args(["--data-root", "/data", "--manifest", "/m",
        "--artifact-root", "/a", "--native-repo-root", "/n"])
    assert (args.count, args.depth, args.mode, args.cadence, args.warmup_count, args.sim_step_ms) == (20, 2, "both", "saturated", 1, 50)
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/coptv2x_hil_pipeline_probe.py"), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "--wire-dir" in completed.stdout


def test_serial_window_runs_same_split_work_without_receiver_or_prints(capsys):
    probe = load_probe()
    delivered = []
    rows, summary = probe.run_window([1, 2], [0, 0], "serial", lambda x: x + 1,
        lambda x: x * 2, lambda x: x - 1, delivered.append, time.monotonic_ns)
    assert delivered == [3, 5]
    assert summary["completed"] == 2
    assert summary["inflight_peak"] == 1
    assert capsys.readouterr().out == ""
    for row in rows:
        stamps = row["timestamps_ns"]
        assert list(stamps.values()) == sorted(stamps.values())
        assert row["stages_ns"]["source_to_wire_ready"] == stamps["deliver_end"] - stamps["prepare_start"]
        assert row["stages_ns"]["source_to_encoded_result"] == stamps["finish_end"] - stamps["prepare_start"]
        assert row["stages_ns"]["infer_queue"] == 0


def test_serial_failure_counts_attempted_and_unsubmitted_jobs():
    probe = load_probe()

    def broken(item):
        raise RuntimeError("infer failed")

    rows, summary = probe.run_window([1, 2, 3], [0, 0, 0], "serial", lambda x: x,
        broken, lambda x: x, lambda x: None, time.monotonic_ns)
    assert (summary["submitted"], summary["failed"], summary["unsubmitted"]) == (1, 1, 2)
    assert "infer failed" in rows[0]["error"]
    assert "infer_end" in rows[0]["timestamps_ns"]
    assert "finish_start" not in rows[0]["timestamps_ns"]
    assert rows[0]["stages_ns"]["infer"] >= 0


def test_runtime_selection_rejects_contaminated_dependency_tree(tmp_path, monkeypatch):
    from types import SimpleNamespace
    probe = load_probe()
    selected = tmp_path / "simulation/leaderboard/team_code/closedloop"
    selected.mkdir(parents=True)
    (selected / "ditl_trt_backend.py").touch()
    monkeypatch.setitem(sys.modules, "ditl_trt_backend", SimpleNamespace(__file__="/different/tree/ditl_trt_backend.py"))
    with pytest.raises(ValueError, match="fresh process"):
        probe.load_runtime(tmp_path)


def test_runtime_guard_ignores_unrelated_script_modules(monkeypatch):
    from types import SimpleNamespace
    probe = load_probe()
    monkeypatch.setitem(sys.modules, "coptv2x_hil_orin_worker", SimpleNamespace(__file__=str(ROOT / "scripts/coptv2x_hil_orin_worker.py")))
    assert probe.load_runtime(ROOT).closedloop == str(ROOT / "simulation/leaderboard/team_code/closedloop")


def test_verified_prefix_rejects_repeated_identity_and_preserves_input_manifest():
    probe = load_probe()
    manifest = {"samples": [{"sample_id": "s1", "source_frame": 1, "inputs": {"x": {"path": "1.npy"}}},
                            {"sample_id": "s2", "source_frame": 4, "inputs": {"x": {"path": "2.npy"}}}]}
    prefix = probe.verified_prefix(manifest, 2)
    prefix["samples"][0]["inputs"]["x"]["path"] = "changed"
    assert manifest["samples"][0]["inputs"]["x"]["path"] == "1.npy"
    manifest["samples"][1]["sample_id"] = "s1"
    with pytest.raises(ValueError, match="distinct"):
        probe.verified_prefix(manifest, 2)


@pytest.mark.parametrize("mode", ["serial", "both"])
def test_probe_real_source_codec_validator_after_complete_window(tmp_path, monkeypatch, capsys, mode):
    import hashlib
    probe = load_probe()
    runtime = probe.load_runtime(ROOT, need_executor=mode != "serial")
    from test_coptv2x_hil_backend import ArtifactTree, RuntimeFactory
    tree = ArtifactTree()
    try:
        registry = runtime.artifacts.load_hil_artifact_registry(tree.config_dir)
        tensor = registry["compression_only"].input_contract[0]
        samples = []
        for index, source_frame in enumerate((1, 4)):
            path = tmp_path / ("s%d.npy" % index)
            np.save(path, np.full(tensor.shape, index, dtype=tensor.dtype), allow_pickle=False)
            samples.append({"sample_id": "s%d" % index, "source_frame": source_frame,
                "inputs": {tensor.name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "shape": list(tensor.shape), "dtype": tensor.dtype}}})
        manifest = {"schema_version": "coptv2x.local-input-manifest.v1", "route_id": "route",
                    "scenario_id": "scenario", "batch_order": ["ego", "rsu"], "samples": samples}
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        factory = RuntimeFactory()
        monkeypatch.setattr(runtime.backend, "_default_runtime_factory", factory)
        actual_decode = runtime.protocol.decode_wire_frame
        decoded_after_inferences = []

        def decode(*args, **kwargs):
            decoded_after_inferences.append(backend_runtime[-1].inference_count)
            return actual_decode(*args, **kwargs)

        backend_runtime = []
        def recording_factory(*args, **kwargs):
            engine = factory(*args, **kwargs)
            backend_runtime.append(engine)
            return engine

        monkeypatch.setattr(runtime.backend, "_default_runtime_factory", recording_factory)
        monkeypatch.setattr(runtime.protocol, "decode_wire_frame", decode)
        actual_backend_init = runtime.backend.CoptV2xHilBackend.__init__
        actual_pipeline_init = runtime.pipeline.CoptV2xHilPipeline.__init__
        backend_options, pipeline_options = [], []

        def record_backend(instance, *args, **kwargs):
            backend_options.append({key: kwargs.get(key) for key in ("rsu_only", "trust_readonly_inputs", "checksum_dedup")})
            actual_backend_init(instance, *args, **kwargs)

        def record_pipeline(instance, *args, **kwargs):
            pipeline_options.append(kwargs.get("checksum_dedup"))
            actual_pipeline_init(instance, *args, **kwargs)

        monkeypatch.setattr(runtime.backend.CoptV2xHilBackend, "__init__", record_backend)
        monkeypatch.setattr(runtime.pipeline.CoptV2xHilPipeline, "__init__", record_pipeline)
        args = probe.build_parser().parse_args(["--data-root", str(tmp_path), "--manifest", str(path),
            "--artifact-root", str(tree.artifact_root), "--native-repo-root", str(tmp_path),
            "--config-dir", str(tree.config_dir), "--mode", mode, "--arm", "compression_only",
            "--count", "2", "--wire-dir", str(tmp_path / "wire")])
        assert probe.run(args) == 0
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert events[0]["event"] == "start"
        assert events[-1]["event"] == "complete"
        for event in (events[0], events[-1]):
            assert event["distinct_input_content_count"] == 2
            assert event["acceptance_scope"] == "paired_five_arm_static_replay_per_cadence"
            assert event["acceptance_success"] is False
            assert event["success_scope"] == "execution_only_not_experimental_acceptance"
        assert events[-1]["execution_success"] is True
        assert decoded_after_inferences == [3, 3] * (2 if mode == "both" else 1), "one zero warmup plus both source inferences before any receiver work"
        rows = [row for row in events if row["event"] == "sample"]
        summary = next(row for row in events if row["event"] == "summary")
        assert summary["completed"] == 2
        assert summary["warm_sample_count"] == 1
        assert [row["source_sim_time_ns"] for row in rows] == [50000000, 200000000] * (2 if mode == "both" else 1)
        summaries = {row["mode"]: row for row in events if row["event"] == "summary"}
        assert events[0]["checksum_dedup"] is True
        assert all(row["checksum_dedup"] is True for row in summaries.values())
        assert backend_options == [{"rsu_only": True, "trust_readonly_inputs": True, "checksum_dedup": True}] * len(summaries)
        assert pipeline_options == [True] * len(summaries)
        assert all(row["verification_start_ns"] >= summaries[row["mode"]]["window_end_ns"] for row in rows)
        assert all(row["is_hil"] is False and row["is_carla"] is False for row in rows)
        assert all(row["verified"] is True for row in rows)
        assert rows[0]["source_frame"] == 1
        assert rows[0]["output_contract"]["res0"]["shape"][0] == 1
        assert len(list((tmp_path / "wire").glob("*.wire"))) == (4 if mode == "both" else 2)
        for wire in (tmp_path / "wire").glob("*.wire"):
            header, _, _ = runtime.protocol.decode_wire_bytes(wire.read_bytes())
            assert header["tensor_bundle"]["sha256_scope"] == "codriving_feature_payload.v1"
        if mode == "both":
            assert events[-1]["pairs_compared"] == 2
            assert events[-1]["pair_hash_validation"] == "PASS"
    finally:
        tree.close()


def test_pipeline_window_uses_real_executor_and_preserves_split_values():
    probe = load_probe()
    runtime = probe.load_runtime(ROOT)
    delivered = []

    def slow_infer(value):
        time.sleep(.002)
        return value * 2

    rows, summary = probe.run_window(list(range(6)), [0] * 6, "pipeline", lambda x: x + 1,
        slow_infer, lambda x: x - 1, delivered.append, runtime.clock.monotonic_raw_ns,
        runtime.executor.BoundedHilPipeline)
    assert delivered == [1, 3, 5, 7, 9, 11]
    assert summary["completed"] == 6
    assert summary["executor_snapshot"]["clock_domain"] == "monotonic_raw"
    assert summary["inflight_peak"] == 2
    assert [row["sequence_no"] for row in rows] == list(range(6))
    assert all(row["stages_ns"]["source_to_wire_ready"] >= row["stages_ns"]["source_to_encoded_result"] for row in rows)


def test_source_offering_schedule_reports_actual_submit_lateness():
    probe = load_probe()
    rows, summary = probe.run_window([1, 2, 3], [0, 1000000, 4000000], "serial", lambda x: x,
        lambda x: x, lambda x: x, lambda x: None, time.monotonic_ns)
    assert summary["elapsed_ns"] >= 4000000
    for row in rows:
        assert row["actual_submit_ns"] >= row["offered_ns"]
        assert row["submit_lateness_ns"] == row["actual_submit_ns"] - row["offered_ns"]
        assert row["admission_lateness_ns"] >= row["submit_lateness_ns"]


def test_pipeline_failure_accounts_for_all_accepted_jobs_and_stops_admission():
    probe = load_probe()
    runtime = probe.load_runtime(ROOT)

    def broken(value):
        raise RuntimeError("bad engine")

    rows, summary = probe.run_window(list(range(8)), [0] * 8, "pipeline", lambda x: x,
        broken, lambda x: x, lambda x: None, runtime.clock.monotonic_raw_ns, runtime.executor.BoundedHilPipeline)
    assert summary["completed"] == 0
    assert summary["failed"] == summary["executor_snapshot"]["failed"]
    assert summary["submitted"] + summary["unsubmitted"] == 8
    assert summary["inflight_peak"] <= 2
    assert "bad engine" in summary["admission_error"]
    assert all(row["status"] == "failed" for row in rows)


def test_gpu_idle_uses_union_and_rejects_wrong_clock_domain():
    probe = load_probe()
    rows = [{"edge_timestamps_ns": {"h2d_start": 10, "h2d_end": 20,
        "inference_start": 20, "inference_end": 50, "d2h_start": 50, "d2h_end": 60}}]
    summary = probe._gpu_timeline_summary(rows, 0, 100)
    assert summary["gpu_stage_timestamp_union_ns"] == 50
    assert summary["forward_timestamp_union_ns"] == 30
    assert summary["gpu_stage_idle_inferred_ns"] == 50
    with pytest.raises(ValueError, match="RAW"):
        probe._gpu_timeline_summary(rows, 30, 100)


def test_fatal_setup_error_emits_complete_without_inventing_counts(tmp_path, capsys):
    probe = load_probe()
    assert probe.main(["--data-root", str(tmp_path), "--manifest", str(tmp_path / "missing.json"),
        "--artifact-root", str(tmp_path), "--native-repo-root", str(tmp_path), "--mode", "serial"]) == 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-2]["event"] == "error"
    assert events[-1]["event"] == "complete"
    assert events[-1]["success"] is False
    assert events[-1]["completed"] is None
    assert events[-1]["counts_unknown_due_to_fatal_error"] is True


def acceptance_samples(distinct=True):
    return [{"sample_id": "s%d" % index, "source_frame": index + 1,
             "inputs": {"x": {"path": "%d.npy" % index,
                                "sha256": "%064x" % (index if distinct else 0),
                                "shape": [2, 1, 2, 2], "dtype": "float32"}}}
            for index in range(20)]


def test_twenty_ids_with_repeated_input_content_do_not_satisfy_acceptance():
    probe = load_probe()
    result = probe.experimental_acceptance(acceptance_samples(False), probe.HIL_ARM_IDS,
                                          ("serial", "pipeline"), True, 100)
    assert result["distinct_input_content_count"] == 1
    assert result["execution_success"] is True
    assert result["acceptance_eligible"] is False
    assert result["acceptance_success"] is False


def test_twenty_distinct_input_contents_require_all_arms_modes_and_verified_pairs():
    probe = load_probe()
    samples = acceptance_samples()
    result = probe.experimental_acceptance(samples, probe.HIL_ARM_IDS, ("serial", "pipeline"), True, 100)
    assert result["distinct_input_content_count"] == 20
    assert result["acceptance_eligible"] is True
    assert result["acceptance_success"] is True
    for arms, modes, execution, pairs in [(probe.HIL_ARM_IDS[:1], ("serial", "pipeline"), True, 20),
        (probe.HIL_ARM_IDS, ("serial",), True, 0), (probe.HIL_ARM_IDS, ("serial", "pipeline"), False, 100),
        (probe.HIL_ARM_IDS, ("serial", "pipeline"), True, 99)]:
        assert probe.experimental_acceptance(samples, arms, modes, execution, pairs)["acceptance_success"] is False


def test_content_identity_normalizes_contract_and_ignores_paths_and_input_order():
    probe = load_probe()
    samples = acceptance_samples()[:2]
    samples[1]["inputs"] = copy.deepcopy(samples[0]["inputs"])
    samples[1]["inputs"]["x"].update(path="different.npy", dtype="<f4")
    result = probe.experimental_acceptance(samples, probe.HIL_ARM_IDS, ("serial", "pipeline"), True, 10)
    assert result["distinct_input_content_count"] == 1
    assert result["acceptance_success"] is False
