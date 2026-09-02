from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from framework.stage2.canonical_search_v3 import (
    build_capability_profile,
    compute_capability_digest,
)
from framework.stage6.hardware_execution_profile_v1 import (
    load_hardware_execution_profile,
)
from framework.stage6.p6_capability_artifacts_v1 import (
    build_probe_artifact_blobs,
    validate_probe_artifact_blobs,
)
from framework.stage6.p6_capability_context_v1 import (
    P6CapabilityContextError,
    build_rtx_capability_context,
    capability_context_to_mapping,
    compiler_fingerprint,
    validate_historical_capability_source,
    validate_rtx_capability_context,
)
from framework.stage6.p6_capability_probe_specs_v1 import (
    FEATURE_NAMES,
    NEUTRAL_PROBE_IDS,
    PRUNING_PROBE_IDS,
    aggregate_capability_features,
)


RECORD_FIELDS = {
    "schema_version",
    "probe_id",
    "q_mode",
    "onnx_sha256",
    "build_success",
    "observation_status",
    "compiler_ir_sha256",
    "int8_propagated_ops",
    "precision_eligible_ops",
    "qdq_folded_pairs",
    "qdq_pairs",
    "reformat_ops",
    "total_ops",
    "fused_ops",
    "fusible_ops",
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_profile(profile_id: str, dispatch: str) -> dict[str, Any]:
    return build_capability_profile(
        capability_profile_id=profile_id,
        hardware_target="h800",
        compiler_fingerprint=_sha(f"compiler:{dispatch}"),
        dispatch_key=dispatch,
        features={name: float(index + 1) / 100.0 for index, name in enumerate(FEATURE_NAMES)},
    )


def historical_profiles() -> list[dict[str, Any]]:
    return [
        _historical_profile("h800-tvm-auto", "tvm_auto"),
        _historical_profile("h800-trt-engine", "trt_engine"),
    ]


def historical_source_bytes() -> bytes:
    return json.dumps(
        historical_profiles(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _record(probe_id: str, q_mode: str, *, offset: int) -> dict[str, Any]:
    int8 = q_mode == "int8"
    return {
        "schema_version": "p6_compiler_capability_probe_record_v1",
        "probe_id": probe_id,
        "q_mode": q_mode,
        "onnx_sha256": _sha(f"onnx:{probe_id}:{q_mode}"),
        "build_success": True,
        "observation_status": "observed_build_success",
        "compiler_ir_sha256": _sha(f"ir:{probe_id}:{q_mode}"),
        "int8_propagated_ops": 1 if int8 else None,
        "precision_eligible_ops": 2 if int8 else None,
        "qdq_folded_pairs": offset % 2 if int8 else None,
        "qdq_pairs": 1 if int8 else None,
        "reformat_ops": offset % 3,
        "total_ops": 4,
        "fused_ops": offset % 2,
        "fusible_ops": 2,
    }


def _records(probe_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        _record(probe_id, q_mode, offset=index)
        for index, probe_id in enumerate(probe_ids)
        for q_mode in ("fp16", "int8")
    ]


def runtime_identity() -> dict[str, Any]:
    identity = {
        "schema_version": "p6_tvm_compiler_runtime_identity_v1",
        "tvm_version": "0.20.dev0",
        "python_version": "3.10.16",
        "python_executable_sha256": _sha("python"),
        "target": "cuda -arch=sm_89",
        "tvm_arch": "sm89",
        "cuda_compute_version": "8.9",
        "support_root_sha256": _sha("support-root"),
        "tvm_site_sha256": _sha("tvm-site"),
        "nvlibs_file_sha256": _sha("nvlibs"),
        "compiler_file_sha256": sorted([_sha("lib-a"), _sha("lib-b")]),
    }
    return {**identity, "compiler_fingerprint": compiler_fingerprint(identity)}


def probe_evidence() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    profile = load_hardware_execution_profile("rtx4090")
    neutral_records = _records(NEUTRAL_PROBE_IDS)
    pruning_records = _records(PRUNING_PROBE_IDS)
    artifact_blobs = []
    manifest_digests = {}
    for family, records in (
        ("neutral", neutral_records),
        ("pruning", pruning_records),
    ):
        cells = {(row["probe_id"], row["q_mode"]): row for row in records}
        blobs, digest = build_probe_artifact_blobs(
            family=family,
            records=records,
            onnx_by_cell={cell: f"onnx:{cell[0]}:{cell[1]}".encode() for cell in cells},
            output_by_cell={cell: f"ir:{cell[0]}:{cell[1]}".encode() for cell in cells},
        )
        artifact_blobs.extend(blobs)
        manifest_digests[family] = digest
    return {
        "schema_version": "p6_rtx_compiler_capability_evidence_v1",
        "hardware_profile": "rtx4090",
        "hardware_target": "rtx4090",
        "dispatch_key": "tvm_auto",
        "tvm_arch": "sm89",
        "verified_gpu_count": 4,
        "hardware_capability_sha256": _file_sha(
            root.joinpath(*profile.hardware_capability_path.parts)
        ),
        "environment_contract_sha256": _file_sha(
            root.joinpath(*profile.environment_contract_path.parts)
        ),
        "runtime_identity": runtime_identity(),
        "probe_code_sha256": _sha("probe-code"),
        "neutral_probe_manifest_sha256": manifest_digests["neutral"],
        "pruning_probe_manifest_sha256": manifest_digests["pruning"],
        "neutral_records": neutral_records,
        "pruning_records": pruning_records,
        "artifact_blobs": artifact_blobs,
    }


def capability_context(tmp_path: Path) -> dict[str, Any]:
    profile = load_hardware_execution_profile("rtx4090")
    evidence = probe_evidence()
    return capability_context_to_mapping(
        build_rtx_capability_context(
            historical_source_bytes=historical_source_bytes(),
            evidence=evidence,
            profile=profile,
            repository_root=Path(__file__).resolve().parents[2],
            expected_probe_code_sha256=_sha("probe-code"),
            expected_support_root_sha256=runtime_identity()["support_root_sha256"],
            expected_historical_source_sha256=_sha_bytes(historical_source_bytes()),
            trusted_runtime_identity=evidence["runtime_identity"],
            trusted_probe_records={
                "neutral": evidence["neutral_records"],
                "pruning": evidence["pruning_records"],
            },
        )
    )


def validation_authority() -> dict[str, Any]:
    evidence = probe_evidence()
    return {
        "expected_historical_source_sha256": _sha_bytes(historical_source_bytes()),
        "trusted_runtime_identity": evidence["runtime_identity"],
        "trusted_probe_records": {
            "neutral": evidence["neutral_records"],
            "pruning": evidence["pruning_records"],
        },
    }


def _reseal_context(context: dict[str, Any]) -> None:
    context["context_digest"] = _sha(
        json.dumps(
            {key: value for key, value in context.items() if key != "context_digest"},
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def test_rtx_context_preserves_historical_profiles_and_rebuilds_measured_features(
    tmp_path: Path,
) -> None:
    original = historical_profiles()
    context = capability_context(tmp_path)
    validated = validate_rtx_capability_context(
        context,
        profile=load_hardware_execution_profile("rtx4090"),
        repository_root=Path(__file__).resolve().parents[2],
        expected_probe_code_sha256=_sha("probe-code"),
        expected_support_root_sha256=runtime_identity()["support_root_sha256"],
        **validation_authority(),
    )

    assert list(validated.historical_profiles) == original
    assert context["historical_profiles"] == original
    assert validated.active_profile["hardware_target"] == "rtx4090"
    assert validated.active_profile["dispatch_key"] == "tvm_auto"
    assert len(validated.active_profile["features"]) == 23
    assert set(validated.evidence["neutral_records"][0]) == RECORD_FIELDS
    assert len(validated.evidence["neutral_records"]) == 12
    assert len(validated.evidence["pruning_records"]) == 12
    assert len(validated.evidence["artifact_blobs"]) == 24
    assert not {
        "latency_ms",
        "energy_j",
        "ap30",
        "ap50",
        "ap70",
    } & set(json.dumps(context))


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("missing", id="missing-cell"),
        pytest.param("duplicate", id="duplicate-cell"),
        pytest.param("extra", id="extra-cell"),
        pytest.param("unobserved", id="unobserved-cell"),
    ],
)
def test_rtx_context_rejects_incomplete_or_nonunique_probe_partition(
    tmp_path: Path, mutation: str
) -> None:
    context = capability_context(tmp_path)
    rows = context["measurement_evidence"]["neutral_records"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "extra":
        rows.append({**copy.deepcopy(rows[0]), "probe_id": "alternate"})
    else:
        rows[0]["observation_status"] = "not_observed"
    context["context_digest"] = _sha("resealed-by-attacker")

    with pytest.raises(P6CapabilityContextError):
        validate_rtx_capability_context(
            context,
            profile=load_hardware_execution_profile("rtx4090"),
            repository_root=Path(__file__).resolve().parents[2],
            expected_probe_code_sha256=_sha("probe-code"),
            expected_support_root_sha256=runtime_identity()["support_root_sha256"],
            **validation_authority(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("feature", id="resealed-feature"),
        pytest.param("runtime", id="runtime-drift"),
        pytest.param("target", id="wrong-target"),
        pytest.param("code", id="wrong-probe-code"),
        pytest.param("path", id="private-path"),
        pytest.param("artifact", id="artifact-byte-drift"),
    ],
)
def test_rtx_context_rejects_resealed_or_private_mutations(tmp_path: Path, mutation: str) -> None:
    context = capability_context(tmp_path)
    if mutation == "feature":
        context["active_profile"]["features"]["s1q_build_success_coverage"] = 0.125
        context["active_profile"]["capability_digest"] = _sha("resealed-profile")
    elif mutation == "runtime":
        context["measurement_evidence"]["runtime_identity"]["tvm_arch"] = "sm90"
    elif mutation == "target":
        context["measurement_evidence"]["hardware_target"] = "h800"
    elif mutation == "code":
        context["measurement_evidence"]["probe_code_sha256"] = _sha("alternate-code")
    elif mutation == "path":
        context["measurement_evidence"]["runtime_identity"]["compiler_path"] = "/private/compiler"
    else:
        context["measurement_evidence"]["artifact_blobs"][0]["onnx_base64"] = context[
            "measurement_evidence"
        ]["artifact_blobs"][1]["onnx_base64"]
    context["context_digest"] = _sha("resealed-by-attacker")

    with pytest.raises(P6CapabilityContextError):
        validate_rtx_capability_context(
            context,
            profile=load_hardware_execution_profile("rtx4090"),
            repository_root=Path(__file__).resolve().parents[2],
            expected_probe_code_sha256=_sha("probe-code"),
            expected_support_root_sha256=runtime_identity()["support_root_sha256"],
            **validation_authority(),
        )


def test_compiler_fingerprint_depends_on_actual_runtime_identity() -> None:
    runtime = runtime_identity()
    fingerprint = runtime.pop("compiler_fingerprint")
    assert compiler_fingerprint(runtime) == fingerprint

    changed = {**runtime, "tvm_version": "0.21.dev0"}
    assert compiler_fingerprint(changed) != fingerprint
    changed = {**runtime, "compiler_file_sha256": [_sha("lib-c")]}
    assert compiler_fingerprint(changed) != fingerprint


def test_rtx_context_rejects_fully_resealed_non_profile_cuda_target(
    tmp_path: Path,
) -> None:
    context = capability_context(tmp_path)
    runtime = context["measurement_evidence"]["runtime_identity"]
    runtime["target"] = "cuda -arch=sm_80"
    identity = {key: value for key, value in runtime.items() if key != "compiler_fingerprint"}
    runtime["compiler_fingerprint"] = compiler_fingerprint(identity)
    active = context["active_profile"]
    active["compiler_fingerprint"] = runtime["compiler_fingerprint"]
    active["capability_digest"] = compute_capability_digest(active)
    context["context_digest"] = _sha(
        json.dumps(
            {key: value for key, value in context.items() if key != "context_digest"},
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    authority = validation_authority()
    authority["trusted_runtime_identity"] = copy.deepcopy(runtime)

    with pytest.raises(P6CapabilityContextError):
        validate_rtx_capability_context(
            context,
            profile=load_hardware_execution_profile("rtx4090"),
            repository_root=Path(__file__).resolve().parents[2],
            expected_probe_code_sha256=_sha("probe-code"),
            expected_support_root_sha256=runtime_identity()["support_root_sha256"],
            **authority,
        )


def test_rtx_context_rejects_fully_resealed_historical_profile_rewrite(
    tmp_path: Path,
) -> None:
    context = capability_context(tmp_path)
    historical = context["historical_profiles"][0]
    historical["features"]["s1q_build_success_coverage"] = 0.125
    historical["capability_digest"] = compute_capability_digest(historical)
    rewritten_source = json.dumps(
        context["historical_profiles"],
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    context["historical_source_base64"] = base64.b64encode(rewritten_source).decode(
        "ascii"
    )
    _reseal_context(context)

    with pytest.raises(P6CapabilityContextError):
        validate_rtx_capability_context(
            context,
            profile=load_hardware_execution_profile("rtx4090"),
            repository_root=Path(__file__).resolve().parents[2],
            expected_probe_code_sha256=_sha("probe-code"),
            expected_support_root_sha256=runtime_identity()["support_root_sha256"],
            **validation_authority(),
        )


def test_rtx_context_rejects_fully_resealed_external_runtime_drift(
    tmp_path: Path,
) -> None:
    context = capability_context(tmp_path)
    runtime = context["measurement_evidence"]["runtime_identity"]
    runtime["tvm_version"] = "0.21.dev0"
    identity = {key: value for key, value in runtime.items() if key != "compiler_fingerprint"}
    runtime["compiler_fingerprint"] = compiler_fingerprint(identity)
    active = context["active_profile"]
    active["compiler_fingerprint"] = runtime["compiler_fingerprint"]
    active["capability_digest"] = compute_capability_digest(active)
    _reseal_context(context)

    with pytest.raises(P6CapabilityContextError):
        validate_rtx_capability_context(
            context,
            profile=load_hardware_execution_profile("rtx4090"),
            repository_root=Path(__file__).resolve().parents[2],
            expected_probe_code_sha256=_sha("probe-code"),
            expected_support_root_sha256=runtime_identity()["support_root_sha256"],
            **validation_authority(),
        )


def test_rtx_context_rejects_fully_resealed_numeric_record_drift(
    tmp_path: Path,
) -> None:
    context = capability_context(tmp_path)
    evidence = context["measurement_evidence"]
    evidence["neutral_records"][0]["reformat_ops"] += 1
    neutral_blobs = [
        item for item in evidence["artifact_blobs"] if item["family"] == "neutral"
    ]
    _, evidence["neutral_probe_manifest_sha256"] = validate_probe_artifact_blobs(
        neutral_blobs,
        family="neutral",
        records=evidence["neutral_records"],
    )
    features, _, _ = aggregate_capability_features(
        evidence["neutral_records"], evidence["pruning_records"]
    )
    active = context["active_profile"]
    context["active_profile"] = build_capability_profile(
        capability_profile_id=active["capability_profile_id"],
        hardware_target=active["hardware_target"],
        compiler_fingerprint=active["compiler_fingerprint"],
        dispatch_key=active["dispatch_key"],
        features={name: features[name] for name in active["features"]},
    )
    _reseal_context(context)

    with pytest.raises(P6CapabilityContextError):
        validate_rtx_capability_context(
            context,
            profile=load_hardware_execution_profile("rtx4090"),
            repository_root=Path(__file__).resolve().parents[2],
            expected_probe_code_sha256=_sha("probe-code"),
            expected_support_root_sha256=runtime_identity()["support_root_sha256"],
            **validation_authority(),
        )


def test_context_validation_is_stable_after_canonical_sorted_json_round_trip(
    tmp_path: Path,
) -> None:
    context = json.loads(json.dumps(capability_context(tmp_path), sort_keys=True))

    validated = validate_rtx_capability_context(
        context,
        profile=load_hardware_execution_profile("rtx4090"),
        repository_root=Path(__file__).resolve().parents[2],
        expected_probe_code_sha256=_sha("probe-code"),
        expected_support_root_sha256=runtime_identity()["support_root_sha256"],
        **validation_authority(),
    )

    assert len(validated.historical_profiles) == 2


def test_historical_source_uses_canonical_json_digest_and_ordering(
    tmp_path: Path,
) -> None:
    profiles = historical_profiles()
    compact = historical_source_bytes()
    pretty = json.dumps(profiles, indent=2).encode("utf-8")
    expected = _sha_bytes(compact)
    profile = load_hardware_execution_profile("rtx4090")
    evidence = probe_evidence()
    arguments = {
        "evidence": evidence,
        "profile": profile,
        "repository_root": Path(__file__).resolve().parents[2],
        "expected_probe_code_sha256": _sha("probe-code"),
        "expected_support_root_sha256": runtime_identity()["support_root_sha256"],
        "expected_historical_source_sha256": expected,
        "trusted_runtime_identity": evidence["runtime_identity"],
        "trusted_probe_records": {
            "neutral": evidence["neutral_records"],
            "pruning": evidence["pruning_records"],
        },
    }

    compact_context = build_rtx_capability_context(
        historical_source_bytes=compact, **arguments
    )
    pretty_context = build_rtx_capability_context(
        historical_source_bytes=pretty, **arguments
    )
    assert compact_context.historical_profiles == pretty_context.historical_profiles

    for changed in (list(reversed(profiles)), [{**profiles[0], "features": {}}, profiles[1]]):
        with pytest.raises(P6CapabilityContextError):
            build_rtx_capability_context(
                historical_source_bytes=json.dumps(changed).encode("utf-8"),
                **arguments,
            )


def test_historical_source_accepts_exact_external_byte_authority() -> None:
    payload = (json.dumps(historical_profiles(), indent=2) + "\n").encode("utf-8")

    validated = validate_historical_capability_source(payload, expected_sha256=_sha_bytes(payload))

    assert list(validated) == historical_profiles()
