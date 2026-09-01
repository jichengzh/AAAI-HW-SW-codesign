from __future__ import annotations

import pytest

from framework.stage6.formal_plan_v1 import build_formal_plan, fit_neutral_ap_surrogate


def _candidate(index: int, backend: str) -> dict[str, object]:
    q_mode = "int8" if index % 2 else "fp16"
    width = [16 + index, 32 + index, 64 + index]
    profile = f"h800-{backend}-probe-conditioned-v3"
    return {
        "genome": [*width, q_mode],
        "width": width,
        "q_mode": q_mode,
        "group_id": f"pyramid|{'x'.join(map(str, width))}",
        "manifest_job_id": f"pyramid|{'x'.join(map(str, width))}|q={q_mode}|profile={profile}",
        "capability_profile_id": profile,
        "dispatch_key": "tvm_auto" if backend == "tvm" else "trt_engine",
        "source_contract": {"artifact_id": f"artifact-{index}"},
        "source_evidence_sha256": str(index).zfill(64),
        "graph_features": {
            "parameter_elements": 1000 + index,
            "conv_flops": 2000 + index,
        },
        "predictions": {"ap70": 0.5 + index / 1000},
    }


def test_h800_profile_inputs_build_backend_symmetric_frozen_arm_plans() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]

    plan = build_formal_plan(tvm, trt, expected_pool_size=20)

    assert plan["passed"]
    assert plan["effective_candidate_pool_size"] == 20
    assert len(plan["arms"]["compression_only"]["selected_genomes"]) == 16
    assert len(plan["arms"]["compress_then_tune"]["screened_genomes"]) == 12
    assert len(plan["arms"]["compress_then_tune"]["locked_genomes"]) == 4
    assert len(plan["arms"]["tune_then_compress"]["attempt_genomes"]) == 16
    assert plan["backend_plans"]["tvm"]["compression_only"] == plan["backend_plans"]["trt"][
        "compression_only"
    ]


def test_rejects_backend_candidate_pool_drift() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(19)]

    with pytest.raises(ValueError, match="candidate genome sets differ"):
        build_formal_plan(tvm, trt, expected_pool_size=20)


def test_rejects_backend_dependent_ap_surrogate() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    trt[0]["predictions"]["ap70"] += 0.1  # type: ignore[index, operator]

    with pytest.raises(ValueError, match="AP surrogate drift"):
        build_formal_plan(tvm, trt, expected_pool_size=20)


def test_rejects_backend_source_evidence_drift() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    trt[0]["source_evidence_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="source evidence drift"):
        build_formal_plan(tvm, trt, expected_pool_size=20)


def test_source_evidence_is_canonical_and_bound_into_plan_semantics() -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    tvm[0]["source_evidence_sha256"] = "not-a-sha256"
    trt[0]["source_evidence_sha256"] = "not-a-sha256"

    with pytest.raises(ValueError, match="invalid source evidence binding"):
        build_formal_plan(tvm, trt, expected_pool_size=20)

    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    initial = build_formal_plan(tvm, trt, expected_pool_size=20)
    tvm[0]["source_evidence_sha256"] = "a" * 64
    trt[0]["source_evidence_sha256"] = "a" * 64
    revised = build_formal_plan(tvm, trt, expected_pool_size=20)

    binding = initial["source_evidence_binding"]
    assert binding["schema_version"] == "stage6_source_evidence_binding_v1"
    assert len(binding["evidence_version_sha256"]) == 64
    assert all("source_evidence_sha256" in record for record in binding["records"])
    assert revised["source_evidence_binding"]["evidence_version_sha256"] != binding[
        "evidence_version_sha256"
    ]
    assert revised["plan_sha256"] != initial["plan_sha256"]


@pytest.mark.parametrize(
    "genome",
    [
        [True, 32, 64, "fp16"],
        [16, 0, 64, "fp16"],
        [16, 32, 64, "/private/stage6/q-mode"],
        [16, 32, 64],
    ],
)
def test_formal_plan_rejects_untrusted_genomes_without_echoing_them(
    genome: list[object],
) -> None:
    private_marker = "/private/stage6/q-mode"
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    tvm[0]["genome"] = genome
    trt[0]["genome"] = genome

    with pytest.raises(ValueError, match="invalid candidate genome") as error:
        build_formal_plan(tvm, trt, expected_pool_size=20)

    assert private_marker not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(10**10_000, id="overflow"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_formal_plan_rejects_nonfinite_or_overflow_scalars_without_chains(
    value: object,
) -> None:
    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    tvm[0]["predictions"]["ap70"] = value  # type: ignore[index]
    trt[0]["predictions"]["ap70"] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="formal candidate contains invalid scalar") as error:
        build_formal_plan(tvm, trt, expected_pool_size=20)

    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_formal_plan_rejects_private_numeric_conversion_errors_without_chains() -> None:
    private_marker = "/private/stage6/private-float"

    class PrivateFailure:
        def __float__(self) -> float:
            raise ValueError(private_marker)

    tvm = [_candidate(index, "tvm") for index in range(20)]
    trt = [_candidate(index, "trt") for index in range(20)]
    tvm[0]["graph_features"]["parameter_elements"] = PrivateFailure()  # type: ignore[index]
    trt[0]["graph_features"]["parameter_elements"] = PrivateFailure()  # type: ignore[index]

    with pytest.raises(ValueError, match="formal candidate contains invalid scalar") as error:
        build_formal_plan(tvm, trt, expected_pool_size=20)

    assert private_marker not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


def test_neutral_surrogate_rejects_private_graph_numeric_errors_without_chains() -> None:
    private_marker = "/private/stage6/feature-float"

    class PrivateFailure:
        def __float__(self) -> float:
            raise ValueError(private_marker)

    training_rows = []
    graph_rows = []
    for index in range(16):
        width = [16 + index, 32 + index, 64 + index]
        group_id = f"pyramid|{'x'.join(map(str, width))}"
        training_rows.append(
            {
                "terminal_status": "measured_success_gold",
                "group_id": group_id,
                "ap70": 0.5 + index / 1000,
                "model": "pyramid",
                "width": width,
                "q_mode": "fp16",
            }
        )
        graph_rows.append(
            {
                "group_id": group_id,
                "parameter_elements": 1000 + index,
                "conv_flops": 2000 + index,
            }
        )
    graph_rows[0]["parameter_elements"] = PrivateFailure()
    candidates = [
        {
            "genome": [16, 32, 64, "fp16"],
            "width": [16, 32, 64],
            "q_mode": "fp16",
            "model": "pyramid",
            "graph_features": {
                "parameter_elements": 1000,
                "conv_flops": 2000,
            },
        }
    ]

    with pytest.raises(ValueError, match="formal candidate contains invalid scalar") as error:
        fit_neutral_ap_surrogate(training_rows, graph_rows, candidates)

    assert private_marker not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
