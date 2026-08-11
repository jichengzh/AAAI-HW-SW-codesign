"""Public environment contract loading tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


def _module() -> Any:
    return importlib.import_module("framework.environment_contract")


def _write_contract_tree(tmp_path: Path, *, target: str) -> tuple[Path, Path]:
    root = tmp_path / "repository"
    capability_path = root / "configs" / "hardware" / f"{target}.yaml"
    capability_path.parent.mkdir(parents=True)
    capability_path.write_text(
        yaml.safe_dump(
            {
                "name": target,
                "arch": {"family": "Ada", "sm": "sm89"},
                "ips": {"gpu": {"precisions": ["FP16"]}},
            }
        ),
        encoding="utf-8",
    )
    requires_gpu = target != "cpu_reference"
    runtime = {
        "python": "3.11",
        "cuda": "12.1" if requires_gpu else None,
        "driver": "535.104" if requires_gpu else None,
        "framework_name": "torch",
        "framework_version": "2.4",
    }
    contract_path = root / "environment-contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(
            {
                "schema": "environment_contract_v1",
                "target": target,
                "hardware_capability": f"configs/hardware/{target}.yaml",
                "requires_gpu": requires_gpu,
                "runtime": runtime,
            }
        ),
        encoding="utf-8",
    )
    observation_path = root / "environment-observation.json"
    observation_path.write_text(
        json.dumps(
            {
                "schema": "environment_observation_v1",
                "target": target,
                "runtime": runtime,
                "gpu_count": 1 if requires_gpu else 0,
            }
        ),
        encoding="utf-8",
    )
    return root, contract_path


def _observation_path(contract_path: Path) -> Path:
    return contract_path.with_name("environment-observation.json")


def _load_rtx_contract(module: Any, tmp_path: Path) -> Any:
    root, contract_path = _write_contract_tree(tmp_path, target="rtx4090")
    return module.load_environment_contract(contract_path, repository_root=root)


def _observation(
    module: Any,
    *,
    target: str = "rtx4090",
    gpu_count: int = 1,
    python: str = "3.11",
    cuda: str | None = "12.1",
    driver: str | None = "535.104",
    framework_name: str = "torch",
    framework_version: str = "2.4",
) -> Any:
    return module.EnvironmentObservation(
        target=target,
        gpu_count=gpu_count,
        runtime=module.RuntimeObservation(
            python=python,
            cuda=cuda,
            driver=driver,
            framework_name=framework_name,
            framework_version=framework_version,
        ),
    )


def test_version_satisfies_uses_numeric_zero_padded_parts() -> None:
    module = _module()

    assert module.version_satisfies("12.4", ">=12.0,<13")
    assert module.version_satisfies("12.4.0", "12.4")
    assert not module.version_satisfies("11.8", ">=12.0,<13")
    with pytest.raises(module.EnvironmentContractError, match="version"):
        module.version_satisfies("12.4rc1", ">=12.0")


@pytest.mark.parametrize("expression", [">=12..0", ">=12.0,", "~=12.0", "==12.4"])
def test_version_satisfies_rejects_invalid_expressions(expression: str) -> None:
    module = _module()

    with pytest.raises(module.EnvironmentContractError, match="version"):
        module.version_satisfies("12.4", expression)


def test_validate_environment_returns_stable_field_only_failures(tmp_path: Path) -> None:
    module = _module()
    contract = _load_rtx_contract(module, tmp_path)
    observation = _observation(module, cuda="11.8", gpu_count=0, target="h800")

    result = module.validate_environment(contract, observation)

    assert [(item.code, item.field) for item in result.failures] == [
        ("observation.target.mismatch", "target"),
        ("runtime.gpu_count.required", "gpu_count"),
        ("runtime.cuda.out_of_range", "runtime.cuda"),
    ]
    assert result.target == "rtx4090"
    assert result.passed is False
    assert not hasattr(result, "observation")
    assert not hasattr(result, "path")


def test_validate_environment_checks_cpu_gpu_framework_and_runtime_order(tmp_path: Path) -> None:
    module = _module()
    root, contract_path = _write_contract_tree(tmp_path, target="cpu_reference")
    contract = module.load_environment_contract(contract_path, repository_root=root)
    observation = _observation(
        module,
        target="cpu_reference",
        gpu_count=2,
        python="3.10",
        cuda="12.1",
        driver="535.104",
        framework_name="tensorflow",
        framework_version="2.3",
    )

    result = module.validate_environment(contract, observation)

    assert [(item.code, item.field) for item in result.failures] == [
        ("runtime.gpu_count.unexpected", "gpu_count"),
        ("runtime.python.out_of_range", "runtime.python"),
        ("runtime.cuda.out_of_range", "runtime.cuda"),
        ("runtime.driver.out_of_range", "runtime.driver"),
        ("runtime.framework.name.mismatch", "runtime.framework_name"),
        ("runtime.framework.version.out_of_range", "runtime.framework_version"),
    ]


def test_validate_environment_returns_passed_for_matching_observation(tmp_path: Path) -> None:
    module = _module()
    contract = _load_rtx_contract(module, tmp_path)

    result = module.validate_environment(contract, _observation(module))

    assert result == module.ValidationResult(target="rtx4090", passed=True, failures=())


def test_load_environment_contract_accepts_only_declared_shape(tmp_path: Path) -> None:
    module = _module()
    root, contract_path = _write_contract_tree(tmp_path, target="rtx4090")

    contract = module.load_environment_contract(contract_path, repository_root=root)

    assert contract.target == "rtx4090"
    assert contract.requires_gpu is True
    assert contract.hardware_capability == Path("configs/hardware/rtx4090.yaml")


@pytest.mark.parametrize("schema", [None, "environment_contract_v2"])
def test_load_environment_contract_requires_exact_schema_version(
    tmp_path: Path, schema: str | None
) -> None:
    module = _module()
    root, contract_path = _write_contract_tree(tmp_path, target="rtx4090")
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if schema is None:
        document.pop("schema")
    else:
        document["schema"] = schema
    contract_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(module.EnvironmentContractError):
        module.load_environment_contract(contract_path, repository_root=root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unexpected": True}),
        lambda value: value.update({"hardware_capability": "../outside.yaml"}),
        lambda value: value.update({"target": "unknown"}),
    ],
)
def test_load_environment_contract_rejects_invalid_public_shape(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    module = _module()
    root, contract_path = _write_contract_tree(tmp_path, target="rtx4090")
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    mutate(document)
    contract_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(module.EnvironmentContractError):
        module.load_environment_contract(contract_path, repository_root=root)


@pytest.mark.parametrize(
    ("target", "runtime_update"),
    [
        ("cpu_reference", {"cuda": "12.1"}),
        ("cpu_reference", {"driver": "535.104"}),
        ("rtx4090", {"cuda": None}),
        ("rtx4090", {"driver": None}),
    ],
)
def test_load_environment_contract_enforces_cpu_gpu_runtime_exclusivity(
    tmp_path: Path, target: str, runtime_update: dict[str, str | None]
) -> None:
    module = _module()
    root, contract_path = _write_contract_tree(tmp_path, target=target)
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    document["runtime"].update(runtime_update)
    contract_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(module.EnvironmentContractError):
        module.load_environment_contract(contract_path, repository_root=root)


def test_load_environment_contract_rejects_invalid_hardware_capability(tmp_path: Path) -> None:
    module = _module()
    root, contract_path = _write_contract_tree(tmp_path, target="rtx4090")
    (root / "configs" / "hardware" / "rtx4090.yaml").write_text("ips: {}\n", encoding="utf-8")

    with pytest.raises(module.EnvironmentContractError):
        module.load_environment_contract(contract_path, repository_root=root)


@pytest.mark.parametrize(
    "document",
    [
        {"target": "rtx4090", "runtime": {}, "gpu_count": 1},
        {"target": "rtx4090", "runtime": {"python": "3.11"}, "gpu_count": True},
        {"target": "cpu_reference", "runtime": {"python": "3.11"}, "gpu_count": 1},
    ],
)
def test_load_environment_observation_rejects_invalid_schema(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    module = _module()
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(module.EnvironmentContractError):
        module.load_environment_observation(observation_path)


def test_load_environment_observation_rejects_gpu_target_with_no_visible_gpu(tmp_path: Path) -> None:
    module = _module()
    _, contract_path = _write_contract_tree(tmp_path, target="rtx4090")
    observation_path = _observation_path(contract_path)
    document = json.loads(observation_path.read_text(encoding="utf-8"))
    document["gpu_count"] = 0
    observation_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(module.EnvironmentContractError):
        module.load_environment_observation(observation_path)


def test_load_environment_observation_accepts_cpu_reference_without_gpu_runtime(tmp_path: Path) -> None:
    module = _module()
    observation_path = tmp_path / "cpu-observation.json"
    observation_path.write_text(
        json.dumps(
            {
                "schema": "environment_observation_v1",
                "target": "cpu_reference",
                "runtime": {
                    "python": "3.11",
                    "cuda": None,
                    "driver": None,
                    "framework_name": "torch",
                    "framework_version": "2.4",
                },
                "gpu_count": 0,
            }
        ),
        encoding="utf-8",
    )

    observation = module.load_environment_observation(observation_path)

    assert observation.target == "cpu_reference"
    assert observation.gpu_count == 0


@pytest.mark.parametrize(
    ("runtime", "gpu_count", "message"),
    [
        (
            {
                "python": "3.11",
                "cuda": None,
                "driver": None,
                "framework_name": "torch",
                "framework_version": "2.4",
            },
            1,
            "CPU observations must report gpu_count as 0",
        ),
        (
            {
                "python": "3.11",
                "cuda": "12.1",
                "driver": "535.104",
                "framework_name": "torch",
                "framework_version": "2.4",
            },
            0,
            "CPU targets cannot declare CUDA or a driver",
        ),
    ],
)
def test_load_environment_observation_rejects_gpu_properties_for_cpu_reference(
    tmp_path: Path, runtime: dict[str, str | None], gpu_count: int, message: str
) -> None:
    module = _module()
    observation_path = tmp_path / "cpu-observation.json"
    observation_path.write_text(
        json.dumps(
            {
                "schema": "environment_observation_v1",
                "target": "cpu_reference",
                "runtime": runtime,
                "gpu_count": gpu_count,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.EnvironmentContractError, match=message):
        module.load_environment_observation(observation_path)


@pytest.mark.parametrize("schema", [None, "environment_observation_v2"])
def test_load_environment_observation_requires_exact_schema_version(
    tmp_path: Path, schema: str | None
) -> None:
    module = _module()
    _, contract_path = _write_contract_tree(tmp_path, target="rtx4090")
    observation_path = _observation_path(contract_path)
    document = json.loads(observation_path.read_text(encoding="utf-8"))
    if schema is None:
        document.pop("schema")
    else:
        document["schema"] = schema
    observation_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(module.EnvironmentContractError):
        module.load_environment_observation(observation_path)


def test_load_environment_observation_drops_extra_note(tmp_path: Path) -> None:
    module = _module()
    _, contract_path = _write_contract_tree(tmp_path, target="rtx4090")
    observation_path = _observation_path(contract_path)
    document = json.loads(observation_path.read_text(encoding="utf-8"))
    document["note"] = "collected by operator"
    observation_path.write_text(json.dumps(document), encoding="utf-8")

    observation = module.load_environment_observation(observation_path)

    assert observation.target == "rtx4090"
    assert observation.gpu_count == 1
    assert not hasattr(observation, "note")
