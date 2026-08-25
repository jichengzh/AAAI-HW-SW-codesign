"""Materialize a private P6 full-chain binding from explicit local inputs."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from framework.stage6.p6_external_training_binding_v1 import (
    P6ExternalTrainingBindingError,
    external_training_binding_from_contract,
    external_training_binding_to_mapping,
    load_external_training_binding,
    validate_external_training_binding,
)
from framework.stage6.coptv2x_h800_search_v2 import (
    P6CoptV2XContractError,
    load_local_config,
    load_public_contract,
)
from framework.stage6.p6_history_binding_v1 import (
    GpuProbe,
    LOCAL_INPUT_NAMES,
    P6HistoryBindingError,
    build_history_binding,
    prevalidate_private_binding_pair_destinations,
    validate_binding_recipe_consistency,
    validate_history_execution_binding,
    validate_ready_source_contract_template,
    write_private_binding_pair,
)
from framework.stage6.p6_history_execution_closure_v1 import (
    P6ExecutionClosureError,
    validate_post_source_wrapper_runner_binding,
)
from framework.stage6.p6_history_recipe_profiles_v1 import RECIPE_V2
from framework.stage6.p6_history_training_contract_v1 import (
    P6HistoryTrainingContractError,
)
from framework.stage6.p6_runner_template_validator_v1 import (
    EXECUTION_INTERFACE_KEYS,
    RunnerTemplateValidationError,
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    P6PostSourceAdapterProfileError,
    load_post_source_adapter_profile,
)
from framework.stage6.p6_post_source_wrapper_template_v1 import (
    P6PostSourceWrapperError,
    render_post_source_adapter_wrappers,
)
from framework.stage6.p6_source_wrapper_profile_v1 import (
    P6SourceWrapperProfileError,
    load_source_wrapper_profile,
    render_self_contained_source_wrapper,
    validate_self_contained_source_wrapper,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CONTRACT_PATH = REPOSITORY_ROOT / "configs/execution/p6_h800_search.example.yaml"
SOURCE_ADAPTER_PATH = REPOSITORY_ROOT / "tools/release/build_p6_history_registry.py"
MEASUREMENT_ADAPTER_PATH = REPOSITORY_ROOT / "tools/release/measure_p6_history_batch.py"
EXPECTED_ASSET_LABELS = ("training-data", "model-init", "toolchain")
LEGACY_LOCAL_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "asset_paths",
        "local_input_paths",
        "candidate_source_mode",
        "stage2_search_space_path",
        "stage1_scan_step",
        "source_registry_step",
        "measurement_step",
        "local_output_root",
        "history_recipe_derivation_path",
    }
)
LEGACY_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "target",
        "asset_paths",
        "local_input_paths",
        "source_registry_step",
        "measurement_step",
        "local_output_root",
    }
)
STAGE1_PLACEHOLDERS = frozenset(
    {"{stage1_partition_manifest}", "{local_output_root}"}
)
SHELL_TOKENS = frozenset({"$", "`", ";", "|", "&", "<", ">", "\n", "\r"})
MAX_PRIVATE_INPUT_SIZE = 4 * 1024 * 1024


class FullChainBootstrapError(ValueError):
    """Stable categorized failure for full-chain bootstrap operations."""

    def __init__(self, category: str, detail: str) -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")


@dataclass(frozen=True)
class _LegacyLocator:
    asset_paths: Mapping[str, Path]
    local_input_paths: Mapping[str, Path]
    expected_recipe_path: Path | None


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    keys: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise ValueError("duplicate mapping key")
        keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def materialize_full_chain_binding(
    legacy_local_config: Path,
    runner_template: Path,
    local_output_root: Path,
    binding_output: Path,
    config_output: Path,
    gpu_probe: GpuProbe,
    *,
    source_wrapper_profile: Path | None = None,
    external_training_binding: Path | None = None,
    post_source_adapter_profile: Path | None = None,
) -> dict[str, Any]:
    """Validate private inputs and atomically materialize one binding/config pair."""
    output_root, binding_path, config_path = _resolve_private_outputs(
        local_output_root, binding_output, config_output
    )
    try:
        prevalidate_private_binding_pair_destinations(
            binding_path,
            config_path,
            REPOSITORY_ROOT,
        )
    except P6HistoryBindingError as error:
        raise FullChainBootstrapError(
            error.category, "private pair destination is unsafe"
        ) from error
    locator = _load_legacy_local_locator(legacy_local_config)
    root = _unique_common_history_root(locator)
    expected_recipe_path = _validate_expected_recipe_location(locator, root)
    expected_recipe = _load_expected_recipe(expected_recipe_path)
    try:
        source_contract = validate_ready_source_contract_template(root)
    except (P6HistoryBindingError, P6HistoryTrainingContractError) as error:
        category = getattr(error, "category", "history_execution_invalid")
        raise FullChainBootstrapError(
            category, "history source contract is invalid"
        ) from error
    recipe = source_contract.get("dynamic_materialization_recipe")
    recipe_v2 = (
        isinstance(recipe, Mapping) and recipe.get("schema_version") == RECIPE_V2
    )
    if recipe_v2 and (
        source_wrapper_profile is None or external_training_binding is None
    ):
        raise FullChainBootstrapError(
            "history_execution_invalid", "recipe-v2 private inputs are required"
        )
    if recipe_v2:
        try:
            validated_external = validate_external_training_binding(
                load_external_training_binding(external_training_binding),
                code_toolchain_root=root,
                local_output_root=output_root,
                reserved_paths=(binding_path, config_path),
            )
            if external_training_binding_to_mapping(
                validated_external
            ) != external_training_binding_from_contract(source_contract):
                raise P6ExternalTrainingBindingError()
        except P6ExternalTrainingBindingError as error:
            raise FullChainBootstrapError(
                error.category, "external training binding is invalid"
            ) from error
    if source_wrapper_profile is not None:
        try:
            profile = load_source_wrapper_profile(source_wrapper_profile)
            render_self_contained_source_wrapper(profile, history_root=root)
        except P6SourceWrapperProfileError as error:
            raise FullChainBootstrapError(
                error.category, "source wrapper profile is invalid"
            ) from error
    if post_source_adapter_profile is None and (root / "post-source-adapter-profile.yaml").exists():
        raise FullChainBootstrapError(
            "history_execution_invalid", "post-source adapter profile is required"
        )
    if post_source_adapter_profile is not None:
        try:
            profile = load_post_source_adapter_profile(
                post_source_adapter_profile,
                private_root=root,
            )
            wrapper_paths = render_post_source_adapter_wrappers(profile, private_root=root)
            validate_post_source_wrapper_runner_binding(
                runner_template,
                normalized_private_root=root,
                post_source_wrapper_paths=wrapper_paths,
            )
        except (
            P6ExecutionClosureError,
            P6PostSourceAdapterProfileError,
            P6PostSourceWrapperError,
        ) as error:
            category = getattr(error, "category", "history_execution_invalid")
            raise FullChainBootstrapError(
                category, "post-source adapter profile is invalid"
            ) from error
    try:
        validated_template = validate_pre_provision_runner_template(
            runner_template,
            root,
            require_exact_history_environment=True,
        )
    except RunnerTemplateValidationError as error:
        raise FullChainBootstrapError(error.category, error.detail) from error
    if source_wrapper_profile is not None:
        try:
            validate_self_contained_source_wrapper(
                validated_template,
                source_wrapper_profile=source_wrapper_profile,
            )
        except P6SourceWrapperProfileError as error:
            raise FullChainBootstrapError(
                error.category, "source wrapper profile is invalid"
            ) from error
    stage1_scan = _load_validated_stage1_scan(runner_template)
    interface = _render_runner_interface(
        root, validated_template.execution_interface
    )
    try:
        binding = build_history_binding(
            root,
            component_paths={
                role: str(path)
                for role, path in validated_template.component_paths.items()
            },
            execution_interface=interface,
            local_input_paths={
                name: str(locator.local_input_paths[name])
                for name in LOCAL_INPUT_NAMES
            },
            gpu_probe=gpu_probe,
        )
    except (P6HistoryBindingError, P6HistoryTrainingContractError) as error:
        if isinstance(error, P6HistoryTrainingContractError):
            raise FullChainBootstrapError(
                error.category, "history binding is invalid"
            ) from error
        category = (
            "execution_interface_unavailable"
            if error.category == "execution_interface"
            else error.category
        )
        raise FullChainBootstrapError(category, "history binding is invalid") from error
    config = _render_full_chain_local_config(
        locator,
        stage1_scan,
        root,
        output_root,
        binding_path,
    )
    _validate_rendered_pair(binding, config, output_root)
    try:
        validate_binding_recipe_consistency(binding, expected_recipe)
    except P6HistoryBindingError as error:
        raise FullChainBootstrapError(
            error.category, "history binding recipe is inconsistent"
        ) from error
    try:
        write_private_binding_pair(
            binding,
            config,
            binding_path,
            config_path,
            REPOSITORY_ROOT,
        )
    except P6HistoryBindingError as error:
        raise FullChainBootstrapError(
            error.category, "private pair could not be written"
        ) from error
    return binding


def _load_legacy_local_locator(path: Path) -> _LegacyLocator:
    payload = _load_private_yaml(path, "legacy_locator_invalid")
    if (
        not LEGACY_REQUIRED_KEYS.issubset(payload)
        or not set(payload).issubset(LEGACY_LOCAL_KEYS)
        or payload.get("schema_version") != "p6_h800_coptv2x_local_v2"
        or payload.get("target") != "h800"
    ):
        raise FullChainBootstrapError(
            "legacy_locator_invalid", "legacy locator contract is invalid"
        )
    assets = _resolve_path_mapping(
        payload.get("asset_paths"), EXPECTED_ASSET_LABELS, "legacy asset"
    )
    inputs = _resolve_path_mapping(
        payload.get("local_input_paths"), LOCAL_INPUT_NAMES, "legacy input"
    )
    if any(not path.is_file() for path in inputs.values()):
        raise FullChainBootstrapError(
            "legacy_locator_invalid", "legacy local inputs are unavailable"
        )
    expected_recipe_path = _parse_expected_recipe_path(
        payload.get("history_recipe_derivation_path")
    )
    return _LegacyLocator(
        asset_paths=assets,
        local_input_paths=inputs,
        expected_recipe_path=expected_recipe_path,
    )


def _parse_expected_recipe_path(raw_path: object) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path) if isinstance(raw_path, str) else None
    if (
        path is None
        or not path.is_absolute()
        or _contains_symlink_component(path)
    ):
        raise FullChainBootstrapError(
            "history_recipe_derivation_invalid", "recipe path is invalid"
        )
    return path


def _load_expected_recipe(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    try:
        if not path.is_file() or path.stat().st_size > MAX_PRIVATE_INPUT_SIZE:
            raise OSError
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FullChainBootstrapError(
            "history_recipe_derivation_invalid", "recipe is unavailable"
        ) from error
    if not isinstance(payload, Mapping) or any(
        not isinstance(key, str) for key in payload
    ):
        raise FullChainBootstrapError(
            "history_recipe_derivation_invalid", "recipe is invalid"
        )
    return copy.deepcopy(dict(payload))


def _validate_expected_recipe_location(
    locator: _LegacyLocator, root: Path
) -> Path | None:
    path = locator.expected_recipe_path
    if path is None:
        return None
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FullChainBootstrapError(
            "history_recipe_derivation_invalid", "recipe is unavailable"
        ) from error
    if not _is_relative_to(resolved, root) or _git_root_for(resolved) != root:
        raise FullChainBootstrapError(
            "history_recipe_derivation_invalid", "recipe escapes history root"
        )
    return resolved


def _load_validated_stage1_scan(path: Path) -> Mapping[str, Any]:
    payload = _load_private_yaml(path, "execution_interface_unavailable")
    stage1_scan = payload.get("stage1_scan")
    if not isinstance(stage1_scan, Mapping):
        raise FullChainBootstrapError(
            "execution_interface_unavailable", "runner template contract is invalid"
        )
    return copy.deepcopy(dict(stage1_scan))


def _load_private_yaml(path: Path, category: str) -> dict[str, Any]:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise FullChainBootstrapError(category, "private input path is invalid")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > MAX_PRIVATE_INPUT_SIZE:
            raise OSError
        payload = yaml.load(
            resolved.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise FullChainBootstrapError(category, "private input is unavailable") from error
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        raise FullChainBootstrapError(category, "private input is invalid")
    return payload


def _resolve_path_mapping(
    raw: object,
    expected_keys: tuple[str, ...],
    label: str,
) -> dict[str, Path]:
    if not isinstance(raw, Mapping) or set(raw) != set(expected_keys):
        raise FullChainBootstrapError(
            "legacy_locator_invalid", f"{label} mapping is invalid"
        )
    resolved_paths: dict[str, Path] = {}
    for key in expected_keys:
        value = raw.get(key)
        path = Path(value) if isinstance(value, str) else None
        if (
            path is None
            or not path.is_absolute()
            or _contains_symlink_component(path)
        ):
            raise FullChainBootstrapError(
                "legacy_locator_invalid", f"{label} path is invalid"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise FullChainBootstrapError(
                "legacy_locator_invalid", f"{label} path is unavailable"
            ) from error
        resolved_paths[key] = resolved
    return resolved_paths


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )


def _unique_common_history_root(
    locator: _LegacyLocator,
) -> Path:
    roots = {_git_root_for(path) for path in locator.local_input_paths.values()}
    if len(roots) != 1:
        raise FullChainBootstrapError(
            "history_root_ambiguous", "legacy inputs do not share one Git root"
        )
    root = next(iter(roots))
    if any(
        not _is_relative_to(path, root)
        for path in (*locator.asset_paths.values(), *locator.local_input_paths.values())
    ):
        raise FullChainBootstrapError(
            "history_root_ambiguous", "legacy locator escapes the common Git root"
        )
    return root


def _git_root_for(path: Path) -> Path:
    working_directory = path if path.is_dir() else path.parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(working_directory), "rev-parse", "--show-toplevel"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FullChainBootstrapError(
            "history_root_ambiguous", "Git root resolution failed"
        ) from error
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or len(lines) != 1 or not lines[0]:
        raise FullChainBootstrapError(
            "history_root_ambiguous", "Git root resolution failed"
        )
    try:
        root = Path(lines[0]).resolve(strict=True)
    except OSError as error:
        raise FullChainBootstrapError(
            "history_root_ambiguous", "Git root is unavailable"
        ) from error
    if not root.is_dir() or not _is_relative_to(path, root):
        raise FullChainBootstrapError(
            "history_root_ambiguous", "Git root does not contain the private input"
        )
    return root


def _render_runner_interface(root: Path, raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != set(EXECUTION_INTERFACE_KEYS):
        raise FullChainBootstrapError(
            "execution_interface_unavailable", "execution interface is invalid"
        )
    interface = copy.deepcopy(dict(raw))
    try:
        controller = interface["controller"]
        chain = interface["execution_chain"]
        environment = interface["environment"]
        if not isinstance(controller, dict) or not isinstance(chain, list):
            raise TypeError
        if not isinstance(environment, dict):
            raise TypeError
        controller["argv"] = _render_executable_argv(controller.get("argv"), root)
        for entry in chain:
            if not isinstance(entry, dict):
                raise TypeError
            entry["argv"] = _render_executable_argv(entry.get("argv"), root)
        environment["activation_argv"] = _render_executable_argv(
            environment.get("activation_argv"), root
        )
        values = environment.get("values")
        if not isinstance(values, dict):
            raise TypeError
        private_root = values.get("P6_HISTORY_PRIVATE_ROOT")
        if not isinstance(private_root, dict):
            raise TypeError
        private_root["value"] = str(
            _resolve_relative_private_path(private_root.get("value"), root)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FullChainBootstrapError(
            "execution_interface_unavailable", "execution interface is invalid"
        ) from error
    return interface


def _render_executable_argv(raw: object, root: Path) -> list[str]:
    if not isinstance(raw, list) or not raw or any(
        not isinstance(token, str)
        or not token
        or any(shell_token in token for shell_token in SHELL_TOKENS)
        for token in raw
    ):
        raise ValueError("argv is invalid")
    executable = _resolve_relative_private_path(raw[0], root)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("executable is unavailable")
    return [str(executable), *raw[1:]]


def _resolve_relative_private_path(raw: object, root: Path) -> Path:
    if not isinstance(raw, str) or not raw or any(
        token in raw for token in SHELL_TOKENS
    ):
        raise ValueError("private path is invalid")
    path = Path(raw)
    if ".." in path.parts:
        raise ValueError("private path is invalid")
    candidate = path if path.is_absolute() else root / path
    if _contains_symlink_component(candidate):
        raise ValueError("private path is unsafe")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("private path is unavailable") from error
    if not _is_relative_to(resolved, root):
        raise ValueError("private path escapes root")
    return resolved


def _render_stage1_scan_step(root: Path, raw: object) -> dict[str, Any]:
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"name", "argv"}
        or raw.get("name") != "build_stage1_partition"
    ):
        raise FullChainBootstrapError(
            "execution_interface_unavailable", "Stage1 scan step is invalid"
        )
    try:
        argv = _render_executable_argv(raw.get("argv"), root)
    except ValueError as error:
        raise FullChainBootstrapError(
            "execution_interface_unavailable", "Stage1 scan step is invalid"
        ) from error
    placeholders = {token for token in argv if token in STAGE1_PLACEHOLDERS}
    if placeholders != STAGE1_PLACEHOLDERS or any(
        ("{" in token or "}" in token) and token not in STAGE1_PLACEHOLDERS
        for token in argv
    ):
        raise FullChainBootstrapError(
            "execution_interface_unavailable", "Stage1 scan step is invalid"
        )
    return {"name": "build_stage1_partition", "argv": argv}


def _render_full_chain_local_config(
    locator: _LegacyLocator,
    stage1_scan: object,
    root: Path,
    local_output_root: Path,
    binding_output: Path,
) -> dict[str, Any]:
    stage1_scan_step = _render_stage1_scan_step(root, stage1_scan)
    python_executable = str(Path(sys.executable).resolve(strict=True))
    return {
        "schema_version": "p6_h800_coptv2x_local_v2",
        "target": "h800",
        "asset_paths": {
            label: str(locator.asset_paths[label]) for label in EXPECTED_ASSET_LABELS
        },
        "local_input_paths": {
            name: str(locator.local_input_paths[name]) for name in LOCAL_INPUT_NAMES
        },
        "candidate_source_mode": "framework_stage2_search_space",
        "stage2_search_space_path": str(
            local_output_root / "stage1_partition_manifest.json"
        ),
        "stage1_scan_step": stage1_scan_step,
        "source_registry_step": {
            "name": "build_source_registry",
            "argv": [
                python_executable,
                str(SOURCE_ADAPTER_PATH),
                "--binding",
                str(binding_output),
                "--pyramid-candidate-plan",
                "{pyramid_candidate_plan}",
                "--source-registry-json",
                "{source_registry_json}",
                "--local-output-root",
                "{local_output_root}",
            ],
        },
        "measurement_step": {
            "name": "measure_batch",
            "argv": [
                python_executable,
                str(MEASUREMENT_ADAPTER_PATH),
                "--binding",
                str(binding_output),
                "--measurement-request",
                "{measurement_request}",
                "--feedback-json",
                "{feedback_json}",
                "--round-output-root",
                "{round_output_root}",
            ],
        },
        "local_output_root": str(local_output_root),
    }


def _validate_rendered_pair(
    binding: Mapping[str, Any],
    config: Mapping[str, Any],
    local_output_root: Path,
) -> None:
    try:
        validate_history_execution_binding(binding)
        contract = load_public_contract(PUBLIC_CONTRACT_PATH)
        descriptor, raw_path = tempfile.mkstemp(
            dir=local_output_root,
            prefix=".p6-full-chain-local.",
            suffix=".json",
        )
        path = Path(raw_path)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(config, handle, ensure_ascii=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            load_local_config(path, contract)
        finally:
            path.unlink(missing_ok=True)
    except P6HistoryBindingError as error:
        raise FullChainBootstrapError(
            "execution_interface_unavailable", "rendered binding is invalid"
        ) from error
    except (OSError, P6CoptV2XContractError, TypeError, ValueError) as error:
        raise FullChainBootstrapError(
            "local_config_invalid", "rendered local config is invalid"
        ) from error


def _resolve_private_outputs(
    raw_root: Path,
    raw_binding: Path,
    raw_config: Path,
) -> tuple[Path, Path, Path]:
    if (
        not all(isinstance(path, Path) and path.is_absolute() for path in (
            raw_root,
            raw_binding,
            raw_config,
        ))
        or raw_root.is_symlink()
    ):
        raise FullChainBootstrapError(
            "unsafe_destination", "private output paths are invalid"
        )
    try:
        root = raw_root.resolve(strict=True)
    except OSError as error:
        raise FullChainBootstrapError(
            "unsafe_destination", "private output root is unavailable"
        ) from error
    if not root.is_dir():
        raise FullChainBootstrapError(
            "unsafe_destination", "private output root is invalid"
        )
    destinations: list[Path] = []
    for raw_path in (raw_binding, raw_config):
        if raw_path.is_symlink():
            raise FullChainBootstrapError(
                "unsafe_destination", "private output path is unsafe"
            )
        try:
            parent = raw_path.parent.resolve(strict=True)
            destination = (parent / raw_path.name).resolve(strict=False)
        except OSError as error:
            raise FullChainBootstrapError(
                "unsafe_destination", "private output path is unavailable"
            ) from error
        if not _is_relative_to(destination, root) or destination == root:
            raise FullChainBootstrapError(
                "unsafe_destination", "private output escapes its root"
            )
        destinations.append(destination)
    if destinations[0] == destinations[1]:
        raise FullChainBootstrapError(
            "unsafe_destination", "private output paths must differ"
        )
    stage1_manifest = root / "stage1_partition_manifest.json"
    if stage1_manifest in destinations:
        raise FullChainBootstrapError(
            "unsafe_destination", "private output path is reserved"
        )
    return root, destinations[0], destinations[1]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
