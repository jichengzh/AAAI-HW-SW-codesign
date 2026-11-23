"""Small CPU fixtures exercise the actual receiver, not a substitute decoder."""

import importlib.util
import io
import json
from pathlib import Path
import sys
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_coptv2x_hil_client import _trigger
from test_coptv2x_hil_receiver_memory import result_wire, rsu_contract
from test_coptv2x_hil_serial_clock import result, RSU_SEMANTICS
from coptv2x_hil_result_buffer import HilResultEnvelope
from ditl_edge_protocol import FRAME_PREFIX, MAGIC, encode_tensor_bundle, prepare_envelope_header

ROOT = Path(__file__).resolve().parents[4]


def load_probe():
    path = ROOT / "scripts/coptv2x_hil_receiver_probe.py"
    assert path.is_file(), "receiver stage probe has not been implemented"
    spec = importlib.util.spec_from_file_location("receiver_probe", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_receiver_repetitions_are_accepted_with_immutable_shared_storage(tmp_path):
    probe = load_probe()
    wire_path = tmp_path / "joint.wire"
    wire_path.write_bytes(result_wire(result(_trigger(7), semantics=RSU_SEMANTICS)))
    rows = probe.probe_wires([wire_path], config_dir="fixture", count=3,
                            contract_validator=rsu_contract)
    samples = [row for row in rows if row["event"] == "receiver_sample"]
    assert len(samples) == 3
    assert [row["temperature"] for row in samples] == ["first_timed_sample", "warm", "warm"]
    assert all(row["accepted"] for row in samples)
    assert all(row["trigger_identity"]["sequence_no"] == 7 for row in samples)
    assert all(row["measurement_scope"] == "fixed_wire_cpu_receiver_diagnostic" for row in rows)
    assert all(row["is_hil"] is False for row in rows)
    for row in samples:
        assert row["canonical_hash_validated"] is True
        assert row["readonly_results"] is True
        assert row["payload_semantics"] == RSU_SEMANTICS
        assert row["tensor_bytes"] == 40
        assert set(row["stages_ns"]) == {"decode", "admission", "exact_lookup", "prefusion", "receiver_elapsed"}
        assert row["stages_ns"]["receiver_elapsed"] == sum(
            row["stages_ns"][name] for name in ("decode", "admission", "exact_lookup", "prefusion"))
        for proof in row["memory_sharing"].values():
            assert proof == {"decode_readonly": True, "admitted_readonly": True,
                             "prefusion_readonly": True, "decode_immutable_bytes_owner": True,
                             "decode_admission_shared": True, "admission_prefusion_shared": True}
    summary = rows[-1]
    assert summary["event"] == "receiver_summary"
    assert summary["accepted_count"] == 3
    assert summary["warm_sample_count"] == 2
    assert summary["cold_sample"]["repeat_index"] == 0
    assert summary["host"]["hostname"]
    assert summary["host"]["numpy_version"] == np.__version__


def test_summary_separates_cold_and_uses_linear_percentiles():
    summary = load_probe().summarize_samples([
        {"stages_ns": {"decode": value}} for value in (9999, 10, 20, 30)])
    assert summary["cold_sample"]["stages_ns"]["decode"] == 9999
    assert summary["warm_stages_ns"]["decode"]["p95_ns"] == 29.0
    assert summary["warm_sample_count"] == 3
    assert load_probe().summarize_samples([{"stages_ns": {"decode": 999}}])["warm_stages_ns"] == {}


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_rejects_invalid_repeat_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        load_probe().probe_wires([], config_dir="fixture", count=count)


def test_canonical_hash_mismatch_is_rejected_by_real_admission(tmp_path):
    probe = load_probe()
    envelope = result(_trigger(0), semantics=RSU_SEMANTICS)
    changed = HilResultEnvelope(replace(envelope.feature_result, payload_sha256="0" * 64), ("rsu",))
    path = tmp_path / "bad.wire"
    path.write_bytes(result_wire(changed))
    with pytest.raises(ValueError, match="payload_hash_mismatch"):
        probe.probe_wires([path], config_dir="fixture", count=1, contract_validator=rsu_contract)


def test_nonfinite_payload_is_rejected_without_excluding_finite_checks(tmp_path):
    probe = load_probe()
    envelope = result(_trigger(0), semantics=RSU_SEMANTICS)
    outputs = dict(envelope.feature_result.outputs)
    outputs["res0"] = np.full_like(outputs["res0"], np.nan)
    path = tmp_path / "nan.wire"
    path.write_bytes(result_wire(HilResultEnvelope(replace(envelope.feature_result, outputs=outputs), ("rsu",))))
    with pytest.raises(ValueError, match="payload_contract_invalid"):
        probe.probe_wires([path], config_dir="fixture", count=1, contract_validator=rsu_contract)


def test_cli_repeated_wires_jsonl_and_default_twenty_repetitions(tmp_path):
    probe = load_probe()
    path = tmp_path / "joint.wire"
    path.write_bytes(result_wire(result(_trigger(0), semantics=RSU_SEMANTICS)))
    output = io.StringIO()
    with mock.patch.object(probe.backend, "create_feature_payload_validator", return_value=rsu_contract):
        assert probe.main(["--wire", str(path), "--wire", str(path), "--config-dir", "fixture"], output=output) == 0
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    summaries = [row for row in rows if row["event"] == "receiver_summary"]
    assert len(summaries) == 2
    assert all(row["accepted_count"] == 20 and row["warm_sample_count"] == 19 for row in summaries)
    assert all(row["checksum_dedup"] is False for row in summaries)


def test_checksum_reuse_optin_keeps_real_canonical_and_provenance_checks(tmp_path):
    probe = load_probe()
    envelope = result(_trigger(2), semantics=RSU_SEMANTICS)
    legacy_wire = result_wire(envelope)
    _, header_size, _ = FRAME_PREFIX.unpack(legacy_wire[:FRAME_PREFIX.size])
    header = json.loads(legacy_wire[FRAME_PREFIX.size:FRAME_PREFIX.size + header_size])
    metadata, payload = encode_tensor_bundle(dict(envelope.feature_result.outputs), feature_checksum=True)
    header_bytes = prepare_envelope_header(header, metadata)
    path = tmp_path / "canonical.wire"
    path.write_bytes(FRAME_PREFIX.pack(MAGIC, len(header_bytes), len(payload)) + header_bytes + payload)
    rows = probe.probe_wires([path], config_dir="fixture", count=2, checksum_dedup=True,
                            contract_validator=rsu_contract, independent_comparison_hash=True)
    sample = rows[1]
    assert sample["protocol_checksum_scope"] == "codriving_feature_payload.v1"
    assert sample["independent_comparison_sha256_outside_window"] == header["payload_sha256"]
    assert sample["canonical_hash_validated"] is True
    assert all(proof["decode_admission_shared"] for proof in sample["memory_sharing"].values())


def test_legacy_batch_two_measures_real_copy_and_rsu_projection(tmp_path):
    from test_coptv2x_hil_client import _small_contract_validator
    probe = load_probe()
    path = tmp_path / "batch2.wire"
    path.write_bytes(result_wire(result(_trigger(1))))
    rows = probe.probe_wires([path], config_dir="fixture", count=1,
                            contract_validator=_small_contract_validator)
    sample = rows[1]
    assert sample["tensor_bytes"] == 80
    assert sample["memory_sharing"]["res0"]["decode_admission_shared"] is False
    assert sample["memory_sharing"]["res0"]["admission_prefusion_shared"] is False


def test_wire_identity_conflict_cannot_be_normalized_away(tmp_path):
    probe = load_probe()
    wire = result_wire(result(_trigger(0), semantics=RSU_SEMANTICS))
    _, header_size, _ = FRAME_PREFIX.unpack(wire[:FRAME_PREFIX.size])
    start = FRAME_PREFIX.size
    header = json.loads(wire[start:start + header_size])
    header["sample_id"] = "incorrect-top-level-identity"
    payload = wire[start + header_size:]
    header_bytes = json.dumps(header).encode("utf-8")
    path = tmp_path / "conflicting.wire"
    path.write_bytes(FRAME_PREFIX.pack(MAGIC, len(header_bytes), len(payload)) + header_bytes + payload)
    with pytest.raises(ValueError, match="identity conflicts"):
        probe.probe_wires([path], config_dir="fixture", count=1, contract_validator=rsu_contract)
