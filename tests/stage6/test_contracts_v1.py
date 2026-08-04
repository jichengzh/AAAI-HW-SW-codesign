from __future__ import annotations

from framework.stage6.contracts_v1 import build_stage6_manifest, validate_stage6_manifest


def test_frozen_six_arm_manifest_passes() -> None:
    manifest = build_stage6_manifest()

    audit = validate_stage6_manifest(manifest)

    assert audit["passed"]
    assert audit["arm_count"] == 6
    assert manifest["common_contract"]["genome"] == ["w0", "w1", "w2", "q_mode"]
    assert manifest["common_contract"]["theoretical_width_grid_size"] == 343
    assert manifest["common_contract"]["registered_width_count"] == 60
    assert manifest["common_contract"]["effective_candidate_pool_size"] == 100
    assert manifest["arms"][1]["q_modes"] == ["fp16", "int8"]
    assert manifest["arms"][3]["outer_budget"] == {"screen": 12, "locked": 4}
    assert manifest["arms"][5]["joint_evidence"]["tvm"]["availability"] == "external"
    assert "results/" not in str(manifest)


def test_hardware_blind_arm_rejects_backend_labels() -> None:
    manifest = build_stage6_manifest()
    arm = manifest["arms"][1]
    arm["acquisition_features"] = [*arm["acquisition_features"], "latency_ms"]

    audit = validate_stage6_manifest(manifest)

    assert not audit["passed"]
    assert "compression_only_backend_label_leakage" in audit["failures"]


def test_reverse_serial_rejects_retune_and_fallback() -> None:
    manifest = build_stage6_manifest()
    arm = manifest["arms"][4]
    arm["compressed_shape_retune_allowed"] = True

    audit = validate_stage6_manifest(manifest)

    assert not audit["passed"]
    assert "tune_then_compress_contract_violation" in audit["failures"]


def test_manifest_rejects_theoretical_grid_as_effective_pool() -> None:
    manifest = build_stage6_manifest()
    manifest["common_contract"]["effective_candidate_pool_size"] = 686

    audit = validate_stage6_manifest(manifest)

    assert not audit["passed"]
    assert "effective_candidate_pool_contract_mismatch" in audit["failures"]


def test_manifest_rejects_unstructured_external_evidence_reference() -> None:
    manifest = build_stage6_manifest()
    manifest["arms"][5]["joint_evidence"]["tvm"] = "external"

    audit = validate_stage6_manifest(manifest)

    assert not audit["passed"]
    assert "joint_evidence_binding_missing" in audit["failures"]


def test_manifest_rejects_a_non_mapping_evidence_container() -> None:
    manifest = build_stage6_manifest()
    manifest["arms"][5]["joint_evidence"] = ["tvm", "trt"]

    audit = validate_stage6_manifest(manifest)

    assert not audit["passed"]
    assert "joint_evidence_binding_missing" in audit["failures"]


def test_manifest_rejects_malformed_nested_containers_with_public_failures() -> None:
    manifest = build_stage6_manifest()
    manifest["arms"][1]["acquisition_features"] = [{"private": "value"}]

    audit = validate_stage6_manifest(manifest)

    assert not audit["passed"]
    assert "compression_only_backend_label_leakage" in audit["failures"]

    manifest = build_stage6_manifest()
    manifest["arms"][5]["joint_evidence"] = [["tvm"], ["trt"]]

    audit = validate_stage6_manifest(manifest)

    assert not audit["passed"]
    assert "joint_evidence_binding_missing" in audit["failures"]


def test_manifest_rejects_non_mapping_arm_entries() -> None:
    manifest = build_stage6_manifest()
    manifest["arms"] = ["not-an-arm"]

    audit = validate_stage6_manifest(manifest)

    assert not audit["passed"]
    assert "six_arm_identity_or_order_mismatch" in audit["failures"]


def test_manifest_rejects_a_non_mapping_root_with_public_failure() -> None:
    audit = validate_stage6_manifest(["not-a-manifest"])

    assert not audit["passed"]
    assert "stage6_manifest_must_be_an_object" in audit["failures"]
