from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from framework.stage6 import coptv2x_h800_search_v2 as execution
from framework.stage6.coptv2x_h800_search_v2 import (
    LocalP6CoptV2XConfig,
    P6CoptV2XContractError,
    PublicP6CoptV2XContract,
    load_local_config,
    load_public_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_yaml(path: Path, content: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def _public_contract(**overrides: Any) -> dict[str, Any]:
    contract = {
        "schema_version": "p6_h800_coptv2x_search_contract_v2",
        "search_id": "p6-pyramid-h800-tvm",
        "target": "h800",
        "target_model": "pyramid",
        "execution_backend": "tvm_auto",
        "seed": 73,
        "sample_budget": 16,
        "batch_size": 4,
        "round_count": 4,
        "configuration_label": "p6-pyramid-h800-tvm",
        "candidate_space_label": "coptv2x-pyramid-width-grid-v1",
        "metric_names": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }
    return {**contract, **overrides}


def _local_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    output_root = tmp_path / "private-output"
    payload = {
        "schema_version": "p6_h800_coptv2x_local_v2",
        "target": "h800",
        "asset_paths": {
            "training-data": str(tmp_path / "training-data"),
            "model-init": str(tmp_path / "model-init"),
            "toolchain": str(tmp_path / "toolchain"),
        },
        "local_input_paths": {
            "gold176_rows": str(tmp_path / "gold176_rows.json"),
            "gold176_graph_features": str(tmp_path / "gold176_graph_features.json"),
            "capability_profiles": str(tmp_path / "capability_profiles.json"),
            "closure": str(tmp_path / "closure.json"),
        },
        "source_registry_step": {
            "name": "build_source_registry",
            "argv": [
                "python",
                "local_build_registry.py",
                "{local_output_root}",
                "{source_registry_json}",
            ],
        },
        "measurement_step": {
            "name": "measure_batch",
            "argv": [
                "python",
                "local_measure.py",
                "{measurement_request}",
                "{feedback_json}",
                "{round_output_root}",
            ],
        },
        "local_output_root": str(output_root),
    }
    return {**payload, **overrides}


def test_load_public_contract_requires_fixed_pyramid_h800_tvm_budget(tmp_path: Path) -> None:
    invalid_contracts = [
        _public_contract(target="orin"),
        _public_contract(target_model="codriving"),
        _public_contract(execution_backend="trt_engine"),
        _public_contract(sample_budget=15),
        _public_contract(batch_size=2),
        _public_contract(round_count=5),
    ]

    for index, payload in enumerate(invalid_contracts):
        with pytest.raises(P6CoptV2XContractError):
            load_public_contract(_write_yaml(tmp_path / f"contract-{index}.yaml", payload))


def test_load_public_contract_accepts_the_public_example() -> None:
    contract = load_public_contract(
        REPOSITORY_ROOT / "configs/execution/p6_h800_search.example.yaml"
    )

    assert isinstance(contract, PublicP6CoptV2XContract)
    assert contract.target == "h800"
    assert contract.target_model == "pyramid"
    assert contract.execution_backend == "tvm_auto"
    assert (contract.round_count, contract.batch_size, contract.sample_budget) == (4, 4, 16)
    assert contract.metric_names == ("latency_ms", "energy_j", "ap30", "ap50", "ap70")


@pytest.mark.parametrize("field", ["max_rounds", "public_summary", "summary_path"])
def test_load_public_contract_rejects_old_public_summary_and_round_keys(
    tmp_path: Path, field: str
) -> None:
    payload = _public_contract()
    payload[field] = "legacy"

    with pytest.raises(P6CoptV2XContractError, match="unknown|forbidden"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", payload))


def test_load_local_config_requires_source_and_measurement_steps(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    for label in ("training-data", "model-init", "toolchain"):
        (tmp_path / label).mkdir()

    loaded = load_local_config(_write_yaml(tmp_path / "local.yaml", _local_config(tmp_path)), contract)

    assert isinstance(loaded, LocalP6CoptV2XConfig)
    assert loaded.source_registry_step.name == "build_source_registry"
    assert loaded.measurement_step.name == "measure_batch"
    assert loaded.local_output_root == tmp_path / "private-output"


@pytest.mark.parametrize(
    "key",
    ["candidate_registry", "measurements", "result_path_template", "steps", "result_step"],
)
def test_load_local_config_rejects_legacy_stage5_input_and_result_keys(
    tmp_path: Path, key: str
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    payload = _local_config(tmp_path)
    payload[key] = "legacy"

    with pytest.raises(P6CoptV2XContractError, match="unknown|forbidden"):
        load_local_config(_write_yaml(tmp_path / "local.yaml", payload), contract)


def test_load_local_config_rejects_shell_or_unknown_template_tokens(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    shell_payload = _local_config(
        tmp_path,
        measurement_step={
            "name": "measure_batch",
            "argv": ["bash", "-c", "private", "{measurement_request}"],
        },
    )
    token_payload = _local_config(
        tmp_path,
        measurement_step={
            "name": "measure_batch",
            "argv": ["python", "local_measure.py", "{candidate_id}"],
        },
    )

    with pytest.raises(P6CoptV2XContractError, match="shell executable"):
        load_local_config(_write_yaml(tmp_path / "shell.yaml", shell_payload), contract)
    with pytest.raises(P6CoptV2XContractError, match="template"):
        load_local_config(_write_yaml(tmp_path / "token.yaml", token_payload), contract)


def test_v2_module_exposes_no_legacy_summary_or_execution_surface() -> None:
    assert not hasattr(execution, "H800SearchSummary")
    assert not hasattr(execution, "run_h800_search")
    assert not hasattr(execution, "write_public_summary")
