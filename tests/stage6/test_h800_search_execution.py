from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from framework.stage6.h800_search_execution_v1 import (
    H800SearchContractError,
    load_local_config,
    load_public_contract,
)


def _write_yaml(path: Path, content: Mapping[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(dict(content), sort_keys=False), encoding="utf-8")
    return path


def _public_contract(**overrides: Any) -> dict[str, Any]:
    contract = {
        "schema_version": "p6_h800_search_contract_v1",
        "search_id": "p6-h800-example",
        "target": "h800",
        "target_model": "pyramid",
        "seed": 73,
        "max_rounds": 2,
        "batch_size": 2,
        "configuration_label": "p6-h800-baseline",
        "metric_names": ["ap", "latency_ms"],
        "candidate_space_label": "pyramid-v1",
        "assets": [
            {"label": "training-data", "version": "v1", "license_status": "cleared"},
            {"label": "model-init", "version": "v2", "license_status": "cleared"},
            {"label": "toolchain", "version": "v3", "license_status": "cleared"},
        ],
    }
    return {**contract, **overrides}


def _local_config(*, asset_label: str = "training-data", **overrides: Any) -> dict[str, Any]:
    config = {
        "schema_version": "p6_h800_search_local_v1",
        "target": "h800",
        "asset_paths": {
            asset_label: "/private/p6/training-data",
            "model-init": "/private/p6/model-init",
            "toolchain": "/private/p6/toolchain",
        },
        "stage5_input_paths": {"feedback": "/private/stage5/feedback.json"},
        "steps": [
            {"name": "train", "argv": ["python", "train.py", "{candidate_request}"]},
            {
                "name": "evaluate",
                "argv": ["python", "evaluate.py", "{candidate_request}", "{result_json}"],
            },
        ],
        "result_step": "evaluate",
        "result_path_template": "/private/p6/results/final.json",
        "local_output_root": "/private/p6/output",
    }
    return {**config, **overrides}


def test_load_public_contract_requires_h800_and_public_fields(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "contract.yaml", _public_contract(target="orin"))

    with pytest.raises(H800SearchContractError, match="target must be h800"):
        load_public_contract(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_id", None),
        ("configuration_label", None),
        ("target_model", None),
        ("seed", 0),
        ("max_rounds", 0),
        ("batch_size", 0),
        ("metric_names", []),
        ("assets", []),
    ],
)
def test_load_public_contract_requires_complete_public_fields(
    tmp_path: Path, field: str, value: Any
) -> None:
    contract = _public_contract()
    if value is None:
        del contract[field]
    else:
        contract[field] = value

    with pytest.raises(H800SearchContractError):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", contract))


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("max_rounds", "2"),
        ("batch_size", -1),
        ("metric_names", ["ap", "ap"]),
        (
            "assets",
            [
                {"label": "training-data", "version": "v1", "license_status": "cleared"},
                {"label": "training-data", "version": "v2", "license_status": "cleared"},
            ],
        ),
        (
            "assets",
            [{"label": "", "version": "v1", "license_status": "cleared"}],
        ),
    ],
)
def test_load_public_contract_rejects_malformed_declared_values(
    tmp_path: Path, field: str, value: Any
) -> None:
    with pytest.raises(H800SearchContractError):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract(**{field: value})))


@pytest.mark.parametrize("forbidden_key", ["path", "command", "host", "candidate_id", "sha256"])
def test_load_public_contract_rejects_private_execution_details(
    tmp_path: Path, forbidden_key: str
) -> None:
    contract = _public_contract()
    contract[forbidden_key] = "private-value"

    with pytest.raises(H800SearchContractError, match="forbidden"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", contract))


def test_load_public_contract_rejects_unknown_and_non_mapping_roots(tmp_path: Path) -> None:
    with pytest.raises(H800SearchContractError, match="unknown"):
        load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract(extra="value")))

    path = tmp_path / "list.yaml"
    path.write_text("- not-a-contract\n", encoding="utf-8")
    with pytest.raises(H800SearchContractError, match="mapping"):
        load_public_contract(path)


def test_load_local_config_rejects_asset_label_mismatch(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    path = _write_yaml(tmp_path / "local.yaml", _local_config(asset_label="unregistered"))

    with pytest.raises(H800SearchContractError, match="asset labels"):
        load_local_config(path, contract)


def test_load_local_config_rejects_shell_string_step(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    path = _write_yaml(
        tmp_path / "local.yaml",
        _local_config(steps=[{"name": "evaluate", "argv": "python train.py"}]),
    )

    with pytest.raises(H800SearchContractError, match="argv"):
        load_local_config(path, contract)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "p6_h800_search_local_v0"),
        ("target", "a100"),
        ("asset_paths", {"training-data": "relative/data"}),
        ("stage5_input_paths", {"feedback": "relative/feedback.json"}),
        ("result_path_template", "relative/result.json"),
        ("local_output_root", "relative/output"),
        ("steps", []),
        ("result_step", "train"),
    ],
)
def test_load_local_config_rejects_contract_and_path_violations(
    tmp_path: Path, field: str, value: Any
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))

    with pytest.raises(H800SearchContractError):
        load_local_config(
            _write_yaml(tmp_path / "local.yaml", _local_config(**{field: value})), contract
        )


def test_load_local_config_rejects_duplicate_steps_unknown_placeholders_and_unknown_keys(
    tmp_path: Path,
) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    malformed_configs = (
        _local_config(
            steps=[
                {"name": "evaluate", "argv": ["python", "evaluate.py"]},
                {"name": "evaluate", "argv": ["python", "evaluate.py", "{result_json}"]},
            ],
            result_step="evaluate",
        ),
        _local_config(
            steps=[
                {"name": "evaluate", "argv": ["python", "evaluate.py", "--out={result_json}"]}
            ]
        ),
        _local_config(extra="value"),
    )

    for index, config in enumerate(malformed_configs):
        with pytest.raises(H800SearchContractError):
            load_local_config(_write_yaml(tmp_path / f"local-{index}.yaml", config), contract)


def test_load_local_config_is_pure_and_returns_immutable_values(tmp_path: Path) -> None:
    contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
    output_root = tmp_path / "not-created"
    config = _local_config(
        asset_paths={
            "training-data": "/unavailable/training-data",
            "model-init": "/unavailable/model-init",
            "toolchain": "/unavailable/toolchain",
        },
        stage5_input_paths={"feedback": "/unavailable/feedback.json"},
        result_path_template="/unavailable/final.json",
        local_output_root=str(output_root),
    )

    loaded = load_local_config(_write_yaml(tmp_path / "local.yaml", config), contract)

    assert not output_root.exists()
    assert loaded.asset_paths["training-data"] == Path("/unavailable/training-data")
    assert loaded.steps[-1].argv[-1] == "{result_json}"
    with pytest.raises(TypeError):
        loaded.asset_paths["new"] = Path("/unavailable/new")  # type: ignore[index]
