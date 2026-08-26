"""Private staging helpers for P6 history normalization."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from framework.stage6.p6_external_training_binding_v1 import (
    external_training_binding_to_mapping,
    validate_external_training_binding,
)
from framework.stage6.p6_history_execution_closure_v1 import (
    copy_execution_closure,
    render_normalized_runner_template,
    validate_normalized_runner_closure,
)
from framework.stage6.p6_history_recipe_normalization_v1 import (
    P6HistoryNormalizationError,
    _invalid,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    build_post_source_adapter_profile,
    load_post_source_adapter_profile,
    post_source_adapter_profile_to_mapping,
)
from framework.stage6.p6_post_source_wrapper_template_v1 import (
    render_post_source_adapter_wrappers,
)
from framework.stage6.p6_source_wrapper_profile_v1 import (
    render_self_contained_source_wrapper,
)


INPUT_NAMES = (
    "gold176_rows",
    "gold176_graph_features",
    "capability_profiles",
    "closure",
)
REGISTRY_SCHEMA_VERSION = "stage5_candidate_source_registry_v1"
LEGACY_SCHEMA_VERSION = "p6_h800_coptv2x_local_v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )


def _copy_json_inputs(
    input_sources: Mapping[str, Path], inputs_dir: Path
) -> dict[str, Path]:
    inputs_dir.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for name in INPUT_NAMES:
        destination = inputs_dir / f"{name}.json"
        shutil.copyfile(input_sources[name], destination, follow_symlinks=False)
        json.loads(destination.read_text(encoding="utf-8"))
        paths[name] = destination
    return paths


def copy_private_tree(source: Path, destination: Path) -> None:
    if _contains_symlink_component(source):
        _invalid("private asset source is invalid")
    destination.mkdir(parents=True)
    for raw_path in source.rglob("*"):
        if _contains_symlink_component(raw_path):
            _invalid("private asset source contains a symlink")
        relative = raw_path.relative_to(source)
        target = destination / relative
        if raw_path.is_dir():
            target.mkdir()
        elif raw_path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_path, target, follow_symlinks=False)
            shutil.copymode(raw_path, target, follow_symlinks=False)
        else:
            _invalid("private asset source contains an unsupported entry")


def _initialize_private_git_root(root: Path) -> None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "init", "-q"],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "private root initialization failed"
        ) from error
    if completed.returncode != 0:
        _invalid("private root initialization failed")
    try:
        (root / ".git" / "info" / "exclude").write_text(
            "external-training-binding.yaml\n"
            "runner-template.yaml\n"
            "source-wrapper-profile.yaml\n"
            "post-source-adapter-profile.yaml\n"
            "legacy.local.yaml\n"
            "runs/\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise P6HistoryNormalizationError(
            "history_normalization_invalid", "private root initialization failed"
        ) from error


def _build_registry_v1(source_group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "groups": [copy.deepcopy(dict(source_group))],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _atomic_write_text(path, serialized)


def _atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, yaml.safe_dump(dict(payload), sort_keys=False))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _source_registry_step(python_executable: str) -> dict[str, Any]:
    return {
        "name": "build_source_registry",
        "argv": [
            python_executable,
            str(REPOSITORY_ROOT / "tools/release/build_p6_history_registry.py"),
            "--binding",
            "{binding}",
            "--pyramid-candidate-plan",
            "{pyramid_candidate_plan}",
            "--source-registry-json",
            "{source_registry_json}",
            "--local-output-root",
            "{local_output_root}",
        ],
    }


def _measurement_step(python_executable: str) -> dict[str, Any]:
    return {
        "name": "measure_batch",
        "argv": [
            python_executable,
            str(REPOSITORY_ROOT / "tools/release/measure_p6_history_batch.py"),
            "--binding",
            "{binding}",
            "--measurement-request",
            "{measurement_request}",
            "--feedback-json",
            "{feedback_json}",
            "--round-output-root",
            "{round_output_root}",
        ],
    }


def _legacy_locator(
    paths: Mapping[str, Path],
    private_root: Path,
    *,
    expected_recipe_path: Path | None = None,
    recipe_v2: bool = False,
) -> dict[str, Any]:
    python_executable = str(Path(sys.executable).resolve(strict=True))
    locator = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "target": "h800",
        "asset_paths": {
            "training-data": str(private_root / "inputs"),
            "model-init": str(paths["registry"]),
            "toolchain": str(
                private_root / ("execution-closure" if recipe_v2 else "toolchain")
            ),
        },
        "local_input_paths": {name: str(paths[name]) for name in INPUT_NAMES},
        "candidate_source_mode": "framework_stage2_search_space",
        "stage2_search_space_path": str(private_root / "stage1_partition_manifest.json"),
        "source_registry_step": _source_registry_step(python_executable),
        "measurement_step": _measurement_step(python_executable),
        "local_output_root": str(private_root / "runs"),
    }
    if expected_recipe_path is not None:
        locator["history_recipe_derivation_path"] = str(expected_recipe_path)
    return locator


def validate_destination_external(
    canonical: Mapping[str, Any], destination: Path
) -> None:
    if "external_training" not in canonical:
        return
    validate_external_training_binding(
        external_training_binding_to_mapping(canonical["external_training"]),
        code_toolchain_root=destination,
        local_output_root=destination.parent / f".{destination.name}.runtime-output",
        reserved_paths=(),
    )


def stage_base_history(
    canonical: Mapping[str, Any], staged: Path
) -> dict[str, Path]:
    _initialize_private_git_root(staged)
    paths = _copy_json_inputs(canonical["input_sources"], staged / "inputs")
    registry_path = staged / "registry" / "candidate-source-registry.json"
    _atomic_write_json(registry_path, _build_registry_v1(canonical["source_contract"]))
    return {**paths, "registry": registry_path}


def _stage_source_wrapper(
    canonical: Mapping[str, Any], staged: Path, copied_roles: Mapping[str, Path]
) -> Path:
    source_role = next(
        role
        for role in canonical["execution_closure"].roles
        if role.role == "source_materializer"
    )
    source_root = next(
        root
        for root in canonical["execution_closure"].roots
        if root.closure_id == source_role.closure_id
    )
    profile = {
        "schema_version": "p6_private_source_wrapper_profile_v2",
        "wrapper_kind": "repo_cwd_exec_v1",
        "destination_relative_path": (
            "documented-stage5-chain/stage5_materialize_round_sources_v1.sh"
        ),
        "implementation_relative_path": copied_roles["source_materializer"]
        .relative_to(staged)
        .as_posix(),
        "implementation_cwd_relative_path": (
            source_root.destination_relative_root.as_posix()
        ),
        "project_python": str(canonical["project_python"]),
    }
    profile_path = staged / "source-wrapper-profile.yaml"
    _atomic_write_yaml(profile_path, profile)
    render_self_contained_source_wrapper(profile, history_root=staged)
    return profile_path


def _stage_post_source_wrappers(
    canonical: Mapping[str, Any],
    staged: Path,
    destination: Path,
    copied_roles: Mapping[str, Path],
) -> tuple[Path | None, Mapping[str, Path] | None]:
    if "post_source_leaf_binding" not in canonical:
        return None, None
    adapter_profile = build_post_source_adapter_profile(
        private_root=staged,
        project_python=canonical["project_python"],
        copied_role_paths=copied_roles,
        execution_closure=canonical["execution_closure"],
        leaf_binding=canonical["post_source_leaf_binding"],
        adapter_python=canonical.get("adapter_python"),
    )
    profile_path = staged / "post-source-adapter-profile.yaml"
    _atomic_write_yaml(
        profile_path,
        post_source_adapter_profile_to_mapping(adapter_profile),
    )
    loaded_profile = load_post_source_adapter_profile(profile_path, private_root=staged)
    if loaded_profile != adapter_profile:
        _invalid("post-source adapter profile is inconsistent")
    wrapper_paths = render_post_source_adapter_wrappers(
        loaded_profile,
        private_root=staged,
        declared_private_root=destination,
    )
    return profile_path, wrapper_paths


def _stage_runner_template(
    canonical: Mapping[str, Any],
    staged: Path,
    copied_roles: Mapping[str, Path],
    post_source_wrapper_paths: Mapping[str, Path] | None,
) -> Path:
    payload = render_normalized_runner_template(
        canonical["source_runner"],
        normalized_private_root=staged,
        copied_role_paths=copied_roles,
        post_source_wrapper_paths=post_source_wrapper_paths,
    )
    runner_path = staged / "runner-template.yaml"
    _atomic_write_yaml(runner_path, payload)
    validate_normalized_runner_closure(
        runner_path,
        normalized_private_root=staged,
        expected_closure=canonical["execution_closure"],
        post_source_wrapper_paths=post_source_wrapper_paths,
    )
    return runner_path


def stage_recipe_v2_history(
    canonical: Mapping[str, Any],
    staged: Path,
    destination: Path,
    paths: Mapping[str, Path],
) -> dict[str, Path]:
    copied_roles = copy_execution_closure(
        canonical["execution_closure"], staged_private_root=staged
    )
    external_path = staged / "external-training-binding.yaml"
    _atomic_write_yaml(
        external_path,
        external_training_binding_to_mapping(canonical["external_training"]),
    )
    source_profile = _stage_source_wrapper(canonical, staged, copied_roles)
    adapter_profile, wrapper_paths = _stage_post_source_wrappers(
        canonical, staged, destination, copied_roles
    )
    runner_path = _stage_runner_template(canonical, staged, copied_roles, wrapper_paths)
    staged_paths = {
        **paths,
        "external_training_binding": external_path,
        "source_wrapper_profile": source_profile,
        "runner_template": runner_path,
    }
    if adapter_profile is not None:
        return {**staged_paths, "post_source_adapter_profile": adapter_profile}
    return staged_paths


def publish_normalized_history(
    canonical: Mapping[str, Any],
    staged: Path,
    destination: Path,
    paths: Mapping[str, Path],
    *,
    recipe_v2_source: bool,
) -> dict[str, Path]:
    staged_paths = dict(paths)
    if "derived_recipe" in canonical:
        derivation_path = staged / "derivation" / "recipe.json"
        _atomic_write_json(derivation_path, canonical["derived_recipe"])
        staged_paths = {**staged_paths, "derivation_recipe": derivation_path}
    public_paths = {
        name: destination / path.relative_to(staged)
        for name, path in staged_paths.items()
    }
    legacy_path = staged / "legacy.local.yaml"
    _atomic_write_yaml(
        legacy_path,
        _legacy_locator(
            public_paths,
            destination,
            expected_recipe_path=public_paths.get("derivation_recipe"),
            recipe_v2=recipe_v2_source,
        ),
    )
    final_paths = {**staged_paths, "legacy": legacy_path}
    relative_paths = {
        name: path.relative_to(staged) for name, path in final_paths.items()
    }
    os.replace(staged, destination)
    return {name: destination / relative for name, relative in relative_paths.items()}
