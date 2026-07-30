"""Behavioral tests for public Stage2 capability and Stage5 genome identity."""

from __future__ import annotations

import copy
import hashlib

import pytest

from framework.stage2.canonical_search_v3 import (
    CAPABILITY_PROFILE_SCHEMA,
    assign_grouped_split,
    build_active_manifest,
    build_capability_profile,
    compute_capability_digest,
    parse_width,
    route_historical_180,
    validate_capability_profile,
    width_key,
)
from framework.stage5.genome_contract_v1 import (
    canonical_group_id,
    validate_structure_identity,
    width_schema_for_model,
)


@pytest.mark.parametrize("model", ["pyramid", "codriving"])
def test_legacy_three_axis_genome_identity_is_canonical_and_input_is_immutable(
    model: str,
) -> None:
    """Catches model-specific drift or mutation of caller-owned legacy rows."""
    row = {
        "model": model,
        "width": [16, 32, 64],
        "group_id": f"{model}|16x32x64",
    }
    original = copy.deepcopy(row)

    identity = validate_structure_identity(row)

    assert identity.width_schema == ("w0", "w1", "w2")
    assert identity.structure_widths == {"w0": 16, "w1": 32, "w2": 64}
    assert identity.group_id == f"{model}|16x32x64"
    assert row == original


def test_fcooper_five_axis_genome_identity_is_named_and_order_stable() -> None:
    """Catches loss, renaming, or reordering of scanner-derived F-Cooper axes."""
    schema = width_schema_for_model("fcooper")
    widths = [64, 128, 256, 128, 256]
    row = {
        "model": "fcooper",
        "width": widths,
        "width_schema": list(schema),
        "structure_widths": dict(zip(schema, widths)),
        "group_id": canonical_group_id("fcooper", widths, schema),
    }

    identity = validate_structure_identity(row)

    assert identity.width_schema == (
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    )
    assert identity.structure_widths["neck.output"] == 256
    assert identity.group_id == (
        "fcooper|backbone.s0=64|backbone.s1=128|backbone.s2=256"
        "|neck.deblock=128|neck.output=256"
    )


def test_fcooper_rejects_missing_reordered_or_nonfinite_scanner_axes() -> None:
    """Catches ambiguous F-Cooper identities entering selection."""
    schema = list(width_schema_for_model("fcooper"))
    reordered = {
        "model": "fcooper",
        "width": [256, 128, 256, 128, 64],
        "width_schema": list(reversed(schema)),
        "group_id": (
            "fcooper|neck.output=256|neck.deblock=128|backbone.s2=256"
            "|backbone.s1=128|backbone.s0=64"
        ),
    }
    malformed = {
        "model": "fcooper",
        "width": [64, 128, float("nan"), 128, 256],
        "width_schema": schema,
        "structure_widths": dict(zip(schema, [64, 128, float("nan"), 128, 256])),
        "group_id": "fcooper|malformed",
    }

    with pytest.raises(ValueError, match="width_schema"):
        validate_structure_identity(reordered)
    with pytest.raises(ValueError, match="width|structure identity"):
        validate_structure_identity(malformed)


def test_capability_profile_schema_digest_and_dispatch_contract() -> None:
    """Catches digest instability or accidental inclusion of dispatch metadata."""
    fingerprint = hashlib.sha256(b"compiler").hexdigest()
    profile = build_capability_profile(
        capability_profile_id="h800-tvm",
        hardware_target="h800",
        compiler_fingerprint=fingerprint,
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.25, "qdq_fold": 0.5},
    )
    rerouted = {**profile, "dispatch_key": "trt_engine"}

    assert profile["schema_version"] == CAPABILITY_PROFILE_SCHEMA
    assert profile["capability_digest"] == compute_capability_digest(profile)
    assert compute_capability_digest(rerouted) == profile["capability_digest"]
    assert validate_capability_profile(rerouted)["dispatch_key"] == "trt_engine"


def test_active_manifest_keeps_dispatch_out_of_genome_and_group_split_is_stable() -> None:
    """Catches backend leakage into genomes or non-grouped/non-deterministic splits."""
    profile = build_capability_profile(
        capability_profile_id="h800-tvm",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"compiler").hexdigest(),
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.25},
    )

    manifest = build_active_manifest(
        widths_by_model={"pyramid": ["16x32x64", [24, 48, 96]]},
        capability_profiles=[profile],
    )
    first = assign_grouped_split(
        manifest["jobs"], holdout_group_count=1, seed=20260717
    )
    second = assign_grouped_split(
        manifest["jobs"], holdout_group_count=1, seed=20260717
    )

    assert parse_width("16x32x64") == (16, 32, 64)
    assert width_key([24, 48, 96]) == "24x48x96"
    assert manifest["genome_schema"] == ["p1", "p2", "p3", "q_mode"]
    assert all("dispatch_key" not in job["strategy_id"] for job in manifest["jobs"])
    assert first == second
    assert {row["group_id"] for row in first["train"]}.isdisjoint(
        {row["group_id"] for row in first["holdout"]}
    )


def test_historical_rows_are_prior_or_disagreement_evidence_never_final_frontier() -> None:
    """Catches historical proxy evidence being promoted into measured frontier rows."""
    prior = {
        "row_id": "prior",
        "can_use_as_cold_start_prior": True,
        "routeb_int8_latency_ms_if_measured": 2.0,
        "phase1_old_int8tc_latency_ms_if_available": 8.0,
    }

    routed = route_historical_180([prior], disagreement_ratio=2.0)

    assert routed["width_ap_prior"] == [prior]
    assert routed["historical_ablation"] == [prior]
    assert routed["disagreement_probe"] == [prior]
    assert routed["backend_gold"] == []
    assert routed["final_frontier"] == []


@pytest.mark.parametrize("width", ["16x32", [16, 0, 64], [1, 2, 3, 4]])
def test_width_and_split_contracts_reject_invalid_public_inputs(width) -> None:
    """Catches malformed width genomes and empty grouped holdout partitions."""
    with pytest.raises(ValueError):
        parse_width(width)
    with pytest.raises(ValueError, match="holdout_group_count"):
        assign_grouped_split(
            [{"group_id": "only"}], holdout_group_count=1, seed=1
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda profile: profile.update(schema_version="wrong"),
        lambda profile: profile.update(compiler_fingerprint="not-a-sha"),
        lambda profile: profile["features"].update(latency_ratio=1.0),
        lambda profile: profile["features"].update(qdq_fold=float("inf")),
        lambda profile: profile.update(capability_digest="0" * 64),
    ],
)
def test_capability_profile_rejects_malformed_or_label_derived_features(
    mutation,
) -> None:
    """Catches malformed capability context and target leakage at dispatch boundary."""
    profile = build_capability_profile(
        capability_profile_id="h800-tvm",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"compiler").hexdigest(),
        dispatch_key="tvm_auto",
        features={"int8_propagation": 0.25, "qdq_fold": 0.5},
    )
    mutation(profile)

    with pytest.raises(ValueError):
        validate_capability_profile(profile)
