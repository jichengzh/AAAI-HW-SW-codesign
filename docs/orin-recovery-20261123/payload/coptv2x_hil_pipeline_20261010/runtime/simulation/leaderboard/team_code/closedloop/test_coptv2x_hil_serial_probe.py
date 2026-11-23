"""CPU tests for the serial static-replay diagnostic (not HIL evidence)."""

import copy
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]


def load_probe():
    path = ROOT / "scripts" / "coptv2x_hil_serial_probe.py"
    assert path.is_file(), "serial replay probe has not been implemented"
    spec = importlib.util.spec_from_file_location("serial_probe", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest():
    return {
        "schema_version": "coptv2x.local-input-manifest.v1",
        "route_id": "route", "scenario_id": "scenario", "batch_order": ["ego", "rsu"],
        "samples": [
            {"sample_id": "s%d" % index, "source_frame": index,
             "inputs": {"spatial_features": {"path": "s%d.npy" % index,
                        "sha256": "a" * 64, "shape": [2, 64, 256, 512], "dtype": "float32"}}}
            for index in range(3)
        ],
    }


def test_prefix_keeps_schema_and_order_without_sharing_mutable_nested_data():
    probe = load_probe()
    original = manifest()
    saved = copy.deepcopy(original)
    prefix = probe.prefix_manifest(original, 2)
    assert list(sample["sample_id"] for sample in prefix["samples"]) == ["s0", "s1"]
    assert set(prefix) == set(original)
    prefix["samples"][0]["inputs"]["spatial_features"]["shape"][0] = 99
    assert original == saved


@pytest.mark.parametrize("count", [0, -1, True, 4])
def test_prefix_rejects_invalid_or_unavailable_counts(count):
    with pytest.raises(ValueError):
        load_probe().prefix_manifest(manifest(), count)


def test_warm_statistics_exclude_first_sample_and_use_linear_p95():
    probe = load_probe()
    rows = [{"stages_ns": {"local_total_actual": value, "encode": value // 2}}
            for value in (10000, 10, 20, 30)]
    summary = probe.summarize_samples(rows)
    assert summary["cold_sample"]["stages_ns"]["local_total_actual"] == 10000
    assert summary["warm_sample_count"] == 3
    assert summary["warm_stages_ns"]["local_total_actual"] == {
        "count": 3, "mean_ns": 20.0, "p50_ns": 20.0, "p95_ns": 29.0,
        "min_ns": 10, "max_ns": 30,
    }


def test_single_sample_does_not_promote_cold_measurement_to_warm():
    result = load_probe().summarize_samples([{"stages_ns": {"encode": 999}}])
    assert result["warm_sample_count"] == 0
    assert result["warm_stages_ns"] == {}


def outputs(batch):
    return {"res%d" % index: np.arange(batch * 4, dtype=np.float32).reshape(batch, 1, 2, 2)
            for index in range(3)}


def test_projection_hash_uses_actual_rsu_slice_not_full_batch_or_header_hash():
    probe = load_probe()
    baseline = outputs(2)
    optimized = {name: array[1:2].copy() for name, array in baseline.items()}
    expected = probe.hash_feature_payload(optimized)
    assert probe.rsu_projection_hash(baseline, "baseline") == expected
    assert probe.rsu_projection_hash(optimized, "optimized") == expected
    assert probe.hash_feature_payload(baseline) != expected
    optimized["res1"][0, 0, 0, 0] += 1
    assert probe.rsu_projection_hash(optimized, "optimized") != expected


def test_projection_rejects_wrong_batch_shape():
    with pytest.raises(ValueError):
        load_probe().rsu_projection_hash(outputs(1), "baseline")


def test_pair_comparison_fails_closed_on_tensor_or_provenance_mismatch():
    probe = load_probe()
    base = {"arm_id": "original_default", "sample_id": "s0", "sequence_no": 0,
            "source_frame": 0, "input_provenance": {"spatial_features": {"sha256": "a" * 64}},
            "artifact_identity": {"widths": [1, 2, 3]}, "rsu_projection_sha256": "b" * 64}
    assert probe.compare_pair(base, dict(base))["matched"] is True
    for key in ("rsu_projection_sha256", "sample_id", "input_provenance", "artifact_identity"):
        changed = dict(base, **{key: "different"})
        with pytest.raises(ValueError):
            probe.compare_pair(base, changed)


def test_cli_bounds_and_no_codec_or_remote_transport_options():
    parser = load_probe().build_parser()
    args = parser.parse_args(["--data-root", "/data", "--manifest", "/manifest.json",
                              "--artifact-root", "/artifacts", "--native-repo-root", "/native"])
    assert args.count == 20
    assert args.variant == "both"
    assert args.arm == "all"
    assert Path(args.runtime_root) == ROOT
    with pytest.raises(SystemExit):
        parser.parse_args(["--count", "0"])


def test_runtime_root_selects_explicit_module_tree_and_legacy_baseline_keywords(tmp_path):
    probe = load_probe()
    with pytest.raises(ValueError, match="closedloop"):
        probe.load_runtime(tmp_path)
    assert probe.variant_backend_kwargs("baseline") == {}
    assert probe.variant_backend_kwargs("optimized") == {"rsu_only": True, "trust_readonly_inputs": True}
    assert probe.variant_source_kwargs("baseline") == {"memory_map": True, "lazy_load": True}
    assert probe.variant_source_kwargs("optimized") == {
        "preload_verified": True, "max_preloaded_tensor_bytes": 2147483648,
    }


def test_checksum_dedup_and_per_arm_wire_exports_are_explicit_options(tmp_path):
    parser = load_probe().build_parser()
    required = ["--data-root", "/data", "--manifest", "/manifest.json",
                "--artifact-root", "/artifacts", "--native-repo-root", "/native"]
    assert parser.parse_args(required).checksum_dedup is False
    args = parser.parse_args(required + ["--checksum-dedup", "--arm-wire-dir", str(tmp_path)])
    assert args.checksum_dedup is True
    assert Path(args.arm_wire_dir) == tmp_path


@pytest.mark.parametrize("checksum_dedup", [False, True])
def test_serial_run_uses_real_source_pipeline_decode_and_compares_actual_outputs(tmp_path, monkeypatch, capsys, checksum_dedup):
    """Only CUDA runtime is doubled; source, artifact checks and codec are real."""
    probe = load_probe()
    runtime = probe.load_runtime(ROOT)
    # Reuse the existing artifact fixture and external-GPU boundary double.
    from test_coptv2x_hil_backend import ArtifactTree, RuntimeFactory

    tree = ArtifactTree()
    try:
        registry = runtime.artifacts.load_hil_artifact_registry(tree.config_dir)
        spec = registry["compression_only"]
        tensor = spec.input_contract[0]
        array = np.zeros(tensor.shape, dtype=tensor.dtype)
        input_path = tmp_path / "s0.npy"
        np.save(input_path, array, allow_pickle=False)
        del array
        document = manifest()
        document["samples"] = [{"sample_id": "s0", "source_frame": 0,
            "inputs": {tensor.name: {"path": "s0.npy", "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                                     "shape": list(tensor.shape), "dtype": tensor.dtype}}}]
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
        monkeypatch.setattr(runtime.backend, "_default_runtime_factory", RuntimeFactory())
        actual_validator = runtime.backend.validate_feature_payload_contract

        def legacy_validator(artifact_identity, outputs, config_dir):
            return actual_validator(artifact_identity, outputs, config_dir=config_dir)

        def versioned_validator(artifact_identity, outputs, config_dir, **kwargs):
            if next(iter(outputs.values())).shape[0] == 2:
                # Real legacy stand-in has no payload_semantics keyword.
                return legacy_validator(artifact_identity, outputs, config_dir=config_dir, **kwargs)
            return actual_validator(artifact_identity, outputs, config_dir=config_dir, **kwargs)

        monkeypatch.setattr(runtime.backend, "validate_feature_payload_contract", versioned_validator)
        arguments = [
            "--data-root", str(tmp_path), "--manifest", str(manifest_path),
            "--artifact-root", str(tree.artifact_root), "--native-repo-root", str(tmp_path),
            "--config-dir", str(tree.config_dir), "--arm", "compression_only", "--count", "1",
            "--last-wire", str(tmp_path / "last.wire"),
            "--arm-wire-dir", str(tmp_path / "arm_wires"),
        ]
        if checksum_dedup:
            arguments.append("--checksum-dedup")
        args = probe.build_parser().parse_args(arguments)
        assert probe.run(args) == 0
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        samples = [row for row in rows if row["event"] == "sample"]
        assert len(samples) == 2
        baseline, optimized = samples
        assert baseline["output_contract"]["res0"]["shape"][0] == 2
        assert optimized["output_contract"]["res0"]["shape"][0] == 1
        assert baseline["payload_bytes"] == optimized["payload_bytes"] * 2
        assert optimized["output_contract"]["res0"]["readonly"] is True
        assert baseline["rsu_projection_sha256"] == optimized["decoded_feature_sha256"]
        for sample in samples:
            stamps = sample["timestamps_ns"]
            assert sample["stages_ns"]["local_total_actual"] == stamps["validation_actual_end"] - stamps["source_read_start"]
            assert sample["stages_ns"]["compress"] == 0
            assert sample["remote_input_tensor_bytes"] == 0
            assert sample["is_hil"] is False
        assert [row["matched"] for row in rows if row["event"] == "pair_comparison"] == [True]
        assert rows[-1]["pair_hash_validation"] == "PASS"
        wire_header, wire_arrays, wire_bytes = runtime.protocol.decode_wire_bytes((tmp_path / "last.wire").read_bytes())
        assert wire_bytes > optimized["payload_bytes"]
        assert wire_header["payload_sha256"] == optimized["decoded_feature_sha256"]
        assert probe.hash_feature_payload(wire_arrays) == optimized["decoded_feature_sha256"]
        for variant in ("baseline", "optimized"):
            exported = tmp_path / "arm_wires" / ("compression_only_%s.wire" % variant)
            exported_header, _, _ = runtime.protocol.decode_wire_bytes(exported.read_bytes())
            assert exported_header["payload_sha256"] == samples[variant == "optimized"]["decoded_feature_sha256"]
        if checksum_dedup:
            assert optimized["verified_checksum_reused"] is True
            assert baseline["verified_checksum_reused"] is False
            assert optimized["tensor_bundle_sha256_scope"] == "codriving_feature_payload.v1"
    finally:
        tree.close()
