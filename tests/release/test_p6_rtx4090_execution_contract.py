"""Public RTX4090 execution and reporting-contract regression coverage."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from framework.stage6.coptv2x_h800_search_v2 import (
    METRIC_NAMES,
    P6CoptV2XContractError,
    load_public_contract,
)
from framework.stage6.hardware_execution_profile_v1 import load_hardware_execution_profile
from framework.stage6.p6_full_chain_bootstrap_v1 import (
    materialize_full_chain_binding,
)
from framework.stage6.p6_history_binding_v1 import GpuRecord
from framework.stage6.p6_history_normalization_v1 import normalize_history_inputs
from tests.stage6.test_coptv2x_h800_search import _write_yaml
from tests.stage6.test_p6_post_source_adapter_profile import v5_private_source_map
from tools.release.preflight_p6_materializer_training_bridge import (
    preflight_materializer_training_bridge,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RTX_PUBLIC_EXAMPLE = (
    REPOSITORY_ROOT / "configs/execution/p6_rtx4090_search.example.yaml"
)
RTX_FORMAL_SCAN = REPOSITORY_ROOT / "configs/stage1/p6_rtx4090_formal_scan.yaml"
RTX_GPU_INDICES = (31, 29, 23, 19)


class _RtxGpuProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def snapshot(self, indices: tuple[int, ...]) -> tuple[GpuRecord, ...]:
        self.calls.append(indices)
        return tuple(
            GpuRecord(
                index=index,
                uuid=f"GPU-public-fixture-{index}",
                model_name="NVIDIA GeForce RTX 4090",
                occupancy=0.0,
            )
            for index in indices
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _report_provenance(profile_id: str) -> dict[str, Any]:
    profile = load_hardware_execution_profile(profile_id)
    return {
        "schema_version": "p6_hardware_specific_report_provenance_v3",
        "comparison_scope": "hardware_specific",
        "hardware_profile": profile.profile_id,
        "target": profile.target_hardware_id,
        "target_model": "pyramid",
        "hardware_model_family": profile.target_hardware_id,
        "gpu_count": 4,
        "execution_backend": "tvm_auto",
        "tvm_arch": profile.tvm_arch,
        "tvm_cache_namespace": profile.tvm_cache_namespace,
        "declared_environment_contract_digest": "a" * 64,
        "runtime_observation_status": "observed",
        "observed_runtime_digest": "f" * 64,
        "code_revision": "abc123",
        "source_digest": "b" * 64,
        "compiler_toolchain_digest": "c" * 64,
        "latency_energy_hardware_profile": profile.profile_id,
        "pareto_hardware_profile": profile.profile_id,
        "ap_provenance": {
            "data_split": "coptv2x-test-v1",
            "checkpoint_initial_state": "e" * 64,
            "training_config_digest": "d" * 64,
            "seed": 73,
            "metric_protocol": "coptv2x-ap-v1",
            "dataset_snapshot_digest": "1" * 64,
            "evaluation_snapshot_digest": "2" * 64,
            "cross_hardware_comparison_status": "available",
        },
    }


def _report_contract() -> Any:
    try:
        return importlib.import_module("framework.stage6.p6_public_report_v1")
    except ModuleNotFoundError:
        pytest.fail("public hardware-specific report contract is absent")


def _prepare_rtx4090_bootstrap_inputs(tmp_path: Path) -> tuple[dict[str, Path], _RtxGpuProbe]:
    source_map, source_runner = v5_private_source_map(tmp_path)
    source_map["hardware_profile"] = "rtx4090"
    support = next(
        root
        for root in source_map["execution_code_closure"]["roots"]
        if root["closure_id"] == "tvm-support"
    )
    runtime_site = tmp_path / "approved-runtime/site-packages"
    runtime_site.mkdir(parents=True)
    runtime_nvlibs = tmp_path / "approved-runtime/nvlibs.path"
    runtime_nvlibs.write_text("/runtime/lib\n", encoding="utf-8")
    source_runner_payload = _read_yaml(source_runner)
    source_runner_payload["execution_interface"]["environment"]["values"].update(
        {
            "P6_TVM_PYTHON": {
                "kind": "external_executable",
                "value": "/usr/bin/python3.10",
            },
            "P6_TVM_SITE": {
                "kind": "external_directory",
                "value": str(runtime_site),
            },
            "P6_TVM_NVLIBS_FILE": {
                "kind": "external_file",
                "value": str(runtime_nvlibs),
            },
            "P6_TVM_SUPPORT_ROOT": {
                "kind": "private_path",
                "value": support["source_root"],
            },
            "P6_TVM_SUPPORT_ROOT_SHA256": {
                "kind": "literal",
                "value": support["sha256"],
            },
        }
    )
    _write_yaml(source_runner, source_runner_payload)
    normalized = normalize_history_inputs(
        source_map,
        Path(str(source_map["history_root"])),
        tmp_path / "normalized-private-root",
        runner_template_path=source_runner,
    )
    legacy = _read_yaml(normalized["legacy"])
    legacy.update(
        {
            "schema_version": "p6_coptv2x_local_v3",
            "target": "rtx4090",
            "hardware_profile": "rtx4090",
        }
    )
    _write_yaml(normalized["legacy"], legacy)
    runner = _read_yaml(normalized["runner_template"])
    runner["execution_interface"]["environment"]["values"][
        "CUDA_VISIBLE_DEVICES"
    ]["value"] = ",".join(str(index) for index in RTX_GPU_INDICES)
    _write_yaml(normalized["runner_template"], runner)

    output_root = tmp_path / "provisioned"
    output_root.mkdir()
    binding = output_root / "binding.json"
    local_config = output_root / "local.yaml"
    probe = _RtxGpuProbe()
    materialize_full_chain_binding(
        normalized["legacy"],
        normalized["runner_template"],
        output_root,
        binding,
        local_config,
        probe,
        source_wrapper_profile=normalized["source_wrapper_profile"],
        external_training_binding=normalized["external_training_binding"],
        post_source_adapter_profile=normalized["post_source_adapter_profile"],
    )
    return (
        {
            "public_contract_path": RTX_PUBLIC_EXAMPLE,
            "local_config_path": local_config,
            "private_binding_path": binding,
            "runner_template_path": normalized["runner_template"],
            "source_wrapper_profile_path": normalized["source_wrapper_profile"],
            "external_training_binding_path": normalized["external_training_binding"],
            "post_source_adapter_profile_path": normalized[
                "post_source_adapter_profile"
            ],
        },
        probe,
    )


def test_rtx4090_public_example_resolves_registry_and_formal_scan_config() -> None:
    contract = load_public_contract(RTX_PUBLIC_EXAMPLE)
    profile = load_hardware_execution_profile("rtx4090")
    formal_scan = _read_yaml(RTX_FORMAL_SCAN)

    assert contract.hardware_profile is profile
    assert contract.target == "rtx4090"
    assert contract.execution_backend == "tvm_auto"
    assert (contract.sample_budget, contract.batch_size, contract.round_count) == (
        16,
        4,
        4,
    )
    assert contract.metric_names == METRIC_NAMES
    assert formal_scan["hardware_precisions"] == ["FP16", "INT8"]
    assert formal_scan["backend_precisions"] == ["FP16", "INT8"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("target", "h800", id="h800-target"),
        pytest.param("hardware_profile", "h800", id="h800-profile"),
    ],
)
def test_rtx4090_public_example_rejects_h800_profile_or_target_swap(
    tmp_path: Path, field: str, value: str
) -> None:
    payload = _read_yaml(RTX_PUBLIC_EXAMPLE)
    payload[field] = value
    mutated = _write_yaml(tmp_path / "mutated-public.yaml", payload)

    with pytest.raises(P6CoptV2XContractError, match="hardware profile"):
        load_public_contract(mutated)


def test_rtx4090_public_example_drives_real_bootstrap_and_preflight_without_launches(
    tmp_path: Path,
) -> None:
    inputs, probe = _prepare_rtx4090_bootstrap_inputs(tmp_path)

    report = preflight_materializer_training_bridge(**inputs)

    assert probe.calls == [RTX_GPU_INDICES, RTX_GPU_INDICES]
    assert report.status == "accepted"
    assert report.validated_round_count == 4
    assert report.historical_process_launch_count == 0
    assert report.gpu_probe_count == 0


def test_hardware_specific_report_accepts_canonical_rtx4090_provenance() -> None:
    report_contract = _report_contract()
    provenance = report_contract.validate_hardware_specific_report_provenance(
        _report_provenance("rtx4090")
    )

    assert provenance.comparison_scope == "hardware_specific"
    assert provenance.hardware_profile == "rtx4090"
    assert provenance.target == "rtx4090"
    assert provenance.tvm_arch == "sm89"


@pytest.mark.parametrize("gpu_count", [1, 3, 7])
def test_hardware_specific_report_accepts_any_positive_gpu_count(
    gpu_count: int,
) -> None:
    report_contract = _report_contract()
    payload = _report_provenance("rtx4090")
    payload["gpu_count"] = gpu_count

    provenance = report_contract.validate_hardware_specific_report_provenance(payload)

    assert provenance.gpu_count == gpu_count


@pytest.mark.parametrize("gpu_count", [True, 0, -1])
def test_hardware_specific_report_rejects_nonpositive_or_boolean_gpu_count(
    gpu_count: object,
) -> None:
    report_contract = _report_contract()
    payload = _report_provenance("rtx4090")
    payload["gpu_count"] = gpu_count

    with pytest.raises(report_contract.P6PublicReportError, match="GPU count"):
        report_contract.validate_hardware_specific_report_provenance(payload)


def test_hardware_specific_report_rejects_combined_h800_and_rtx4090_latency_energy() -> None:
    report_contract = _report_contract()
    payload = _report_provenance("rtx4090")
    payload["latency_energy_hardware_profile"] = ["h800", "rtx4090"]

    with pytest.raises(report_contract.P6PublicReportError, match="latency.*energy"):
        report_contract.validate_hardware_specific_report_provenance(payload)


def test_hardware_specific_report_rejects_h800_target_label_for_rtx4090() -> None:
    report_contract = _report_contract()
    payload = _report_provenance("rtx4090")
    payload["target"] = "h800"

    with pytest.raises(report_contract.P6PublicReportError, match="target"):
        report_contract.validate_hardware_specific_report_provenance(payload)


def test_hardware_specific_report_rejects_cross_profile_pareto_frontier() -> None:
    report_contract = _report_contract()
    payload = _report_provenance("rtx4090")
    payload["pareto_hardware_profile"] = "h800"

    with pytest.raises(report_contract.P6PublicReportError, match="Pareto"):
        report_contract.validate_hardware_specific_report_provenance(payload)


def test_hardware_specific_cross_hardware_ap_requires_matching_external_provenance() -> None:
    report_contract = _report_contract()
    rtx = _report_provenance("rtx4090")
    h800 = _report_provenance("h800")

    validated = report_contract.validate_cross_hardware_ap_provenance([rtx, h800])
    assert validated["metric_protocol"] == "coptv2x-ap-v1"

    mismatched = copy.deepcopy(h800)
    mismatched["ap_provenance"]["data_split"] = "different-split"
    with pytest.raises(report_contract.P6PublicReportError, match="AP provenance"):
        report_contract.validate_cross_hardware_ap_provenance([rtx, mismatched])


def test_cross_hardware_ap_is_unavailable_without_exact_dataset_identity() -> None:
    report_contract = _report_contract()
    rtx = _report_provenance("rtx4090")
    h800 = _report_provenance("h800")
    for report in (rtx, h800):
        report["ap_provenance"].update(
            {
                "dataset_snapshot_digest": None,
                "evaluation_snapshot_digest": None,
                "cross_hardware_comparison_status": (
                    "unavailable_unverified_dataset_identity"
                ),
            }
        )

    with pytest.raises(report_contract.P6PublicReportError, match="unavailable"):
        report_contract.validate_cross_hardware_ap_provenance([rtx, h800])
