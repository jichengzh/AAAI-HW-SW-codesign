"""Run the private P6 derive-to-verifier chain from one explicit manifest."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT_ENTRY = str(REPOSITORY_ROOT)
sys.path = [
    _REPOSITORY_ROOT_ENTRY,
    *(entry for entry in sys.path if entry != _REPOSITORY_ROOT_ENTRY),
]

from framework.stage6.coptv2x_h800_search_v2 import (  # noqa: E402
    P6CoptV2XContractError,
    load_public_contract,
    validate_static_search_input_payloads,
)
from framework.stage6.p6_gpu_policy_v1 import parse_runtime_gpu_pool  # noqa: E402
from framework.stage6.p6_history_normalization_v1 import (  # noqa: E402
    validate_history_inputs,
)


MANIFEST_SCHEMA_VERSION = "p6_full_chain_run_manifest_v1"
MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "template_only",
        "hardware_profile",
        "contract",
        "source_map",
        "history_root",
        "pre_normalization_runner_template",
        "normalized_private_dir",
        "fresh_output_root",
    }
)
SUPPORTED_PROFILES = frozenset({"h800", "rtx4090"})
MAX_MANIFEST_SIZE = 1024 * 1024


class FullChainManifestError(ValueError):
    """Stable error for an unusable private full-chain manifest."""


@dataclass(frozen=True)
class FullChainManifest:
    """Validated immutable inputs for one fresh private execution."""

    schema_version: str
    template_only: bool
    hardware_profile: str
    contract: Path
    source_map: Path
    history_root: Path
    pre_normalization_runner_template: Path
    normalized_private_dir: Path
    fresh_output_root: Path


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.YAMLError("duplicate key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_private_location(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        raise FullChainManifestError("private location is invalid")
    if _contains_symlink_component(path):
        raise FullChainManifestError("private location is not private")
    try:
        repository = REPOSITORY_ROOT.resolve(strict=True)
        resolved = path.resolve(strict=path.exists())
    except OSError as error:
        raise FullChainManifestError("private location is unavailable") from error
    if not _is_relative_to(resolved, repository):
        return
    relative = resolved.relative_to(repository)
    candidate = relative if resolved.exists() else relative / ".p6-private-output-sentinel"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(candidate)],
            shell=False,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FullChainManifestError("private location cannot be checked") from error
    if completed.returncode != 0:
        raise FullChainManifestError("private location is not private")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file() or path.stat().st_size > MAX_MANIFEST_SIZE:
            raise OSError
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise FullChainManifestError("manifest is unavailable") from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise FullChainManifestError("manifest is invalid")
    return payload


def _absolute_existing_path(raw: object, *, directory: bool) -> Path:
    path = Path(raw) if isinstance(raw, str) else None
    if path is None or not path.is_absolute() or _contains_symlink_component(path):
        raise FullChainManifestError("private input is invalid")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FullChainManifestError("private input is unavailable") from error
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        raise FullChainManifestError("private input is invalid")
    return resolved


def _absolute_fresh_path(raw: object) -> Path:
    path = Path(raw) if isinstance(raw, str) else None
    if path is None or not path.is_absolute() or _contains_symlink_component(path):
        raise FullChainManifestError("private output is invalid")
    try:
        parent = path.parent.resolve(strict=True)
        destination = (parent / path.name).resolve(strict=False)
    except OSError as error:
        raise FullChainManifestError("private output is unavailable") from error
    if not parent.is_dir() or destination.exists():
        raise FullChainManifestError("private output is not fresh")
    return destination


def _contract_path(raw: object, profile: str) -> Path:
    expected_relative = Path(f"configs/execution/p6_{profile}_search.example.yaml")
    if raw != expected_relative.as_posix():
        raise FullChainManifestError("public contract selection is invalid")
    path = (REPOSITORY_ROOT / expected_relative).resolve(strict=True)
    try:
        contract = load_public_contract(path)
    except (OSError, P6CoptV2XContractError) as error:
        raise FullChainManifestError("public contract is invalid") from error
    if contract.hardware_profile.profile_id != profile:
        raise FullChainManifestError("hardware profile does not match public contract")
    return path


def _validate_source_authority(path: Path, history_root: Path, profile: str) -> None:
    payload = _load_yaml_mapping(path)
    source_profile = payload.get("hardware_profile", "h800")
    if (
        payload.get("schema_version") != "p6_history_normalization_source_v5"
        or source_profile != profile
        or payload.get("history_root") != str(history_root)
    ):
        raise FullChainManifestError("source-map authority does not match the run manifest")


def _validate_runner_template(path: Path) -> None:
    payload = _load_yaml_mapping(path)
    if payload.get("schema_version") != "p6_history_runner_template_v1":
        raise FullChainManifestError("runner-template authority is invalid")


def _validate_private_authorities(manifest: FullChainManifest) -> None:
    source_map = _load_yaml_mapping(manifest.source_map)
    validate_history_inputs(
        source_map,
        manifest.history_root,
        runner_template_path=manifest.pre_normalization_runner_template,
    )
    validate_static_search_input_payloads(
        source_map, load_public_contract(manifest.contract)
    )


def _load_and_validate_manifest(path: Path) -> FullChainManifest:
    """Load one private manifest without creating outputs or probing GPUs."""
    _require_private_location(path)
    payload = _load_yaml_mapping(path)
    if set(payload) != MANIFEST_KEYS:
        raise FullChainManifestError("manifest keys are invalid")
    profile = payload.get("hardware_profile")
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("template_only") is not False
        or profile not in SUPPORTED_PROFILES
    ):
        raise FullChainManifestError("manifest is not executable")

    contract = _contract_path(payload.get("contract"), profile)
    source_map = _absolute_existing_path(payload.get("source_map"), directory=False)
    history_root = _absolute_existing_path(payload.get("history_root"), directory=True)
    runner = _absolute_existing_path(
        payload.get("pre_normalization_runner_template"), directory=False
    )
    normalized = _absolute_fresh_path(payload.get("normalized_private_dir"))
    output = _absolute_fresh_path(payload.get("fresh_output_root"))
    if normalized == output:
        raise FullChainManifestError("private outputs must differ")
    for private_path in (source_map, history_root, runner, normalized, output):
        _require_private_location(private_path)
    _validate_source_authority(source_map, history_root, profile)
    _validate_runner_template(runner)
    return FullChainManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        template_only=False,
        hardware_profile=profile,
        contract=contract,
        source_map=source_map,
        history_root=history_root,
        pre_normalization_runner_template=runner,
        normalized_private_dir=normalized,
        fresh_output_root=output,
    )


def _run_command(argv: tuple[str, ...], *, env: dict[str, str]) -> int:
    completed = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        env=env,
        shell=False,
        check=False,
    )
    return completed.returncode


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "--short=12", "HEAD"],
            shell=False,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FullChainManifestError("code revision is unavailable") from error
    revision = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{12}", revision):
        raise FullChainManifestError("code revision is invalid")
    return revision


def _stage_argvs(
    manifest: FullChainManifest, recipe_path: Path, revision: str
) -> tuple[tuple[str, ...], ...]:
    normalized = manifest.normalized_private_dir
    output = manifest.fresh_output_root
    python = sys.executable
    return (
        (
            python,
            str(REPOSITORY_ROOT / "tools/release/derive_p6_history_recipe.py"),
            "--source-map",
            str(manifest.source_map),
            "--runner-template",
            str(manifest.pre_normalization_runner_template),
            "--recipe-json",
            str(recipe_path),
        ),
        (
            python,
            str(REPOSITORY_ROOT / "tools/release/normalize_p6_history_root.py"),
            "--source-map",
            str(manifest.source_map),
            "--history-root",
            str(manifest.history_root),
            "--private-dir",
            str(normalized),
            "--runner-template",
            str(manifest.pre_normalization_runner_template),
        ),
        (
            python,
            str(REPOSITORY_ROOT / "tools/release/run_p6_h800_search.py"),
            "--contract",
            str(manifest.contract),
            "--code-revision",
            revision,
            "--legacy-local-config",
            str(normalized / "legacy.local.yaml"),
            "--runner-template",
            str(normalized / "runner-template.yaml"),
            "--local-output-root",
            str(output),
            "--binding-output",
            str(output / "binding.json"),
            "--config-output",
            str(output / "local-config.yaml"),
            "--source-wrapper-profile",
            str(normalized / "source-wrapper-profile.yaml"),
            "--external-training-binding",
            str(normalized / "external-training-binding.yaml"),
            "--post-source-adapter-profile",
            str(normalized / "post-source-adapter-profile.yaml"),
        ),
        (
            python,
            str(REPOSITORY_ROOT / "tools/release/verify_p6_materializer_training_run.py"),
            "--contract",
            str(manifest.contract),
            "--local-config",
            str(output / "local-config.yaml"),
            "--binding",
            str(output / "binding.json"),
        ),
    )


def _recipes_match(first: Path, second: Path) -> bool:
    try:
        return json.loads(first.read_text(encoding="utf-8")) == json.loads(
            second.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _execute(manifest: FullChainManifest, gpu_pool: str) -> bool:
    descriptor, raw_recipe = tempfile.mkstemp(
        dir=manifest.normalized_private_dir.parent,
        prefix=f".{manifest.normalized_private_dir.name}.",
        suffix=".recipe.json",
    )
    os.close(descriptor)
    recipe_path = Path(raw_recipe)
    recipe_path.unlink(missing_ok=True)
    base_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    base_env.pop("GPU_POOL", None)
    try:
        stages = _stage_argvs(manifest, recipe_path, _git_revision())
        if _run_command(stages[0], env=base_env) != 0:
            return False
        if _run_command(stages[1], env=base_env) != 0:
            return False
        normalized_recipe = manifest.normalized_private_dir / "derivation/recipe.json"
        if not _recipes_match(recipe_path, normalized_recipe):
            return False
        manifest.fresh_output_root.mkdir()
        run_env = {**base_env, "GPU_POOL": gpu_pool}
        if _run_command(stages[2], env=run_env) != 0:
            return False
        return _run_command(stages[3], env=base_env) == 0
    finally:
        recipe_path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or run one fully provisioned private P6 hardware-search chain. "
            "This command never downloads private assets."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="validate the manifest and declared paths without writes or GPU probing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate inputs or execute derive, normalize, controller, and verifier."""
    try:
        args = _parse_args(argv)
        manifest = _load_and_validate_manifest(args.manifest)
        _validate_private_authorities(manifest)
        if args.check_inputs:
            sys.stdout.write("inputs_ready\n")
            return 0
        gpu_count = parse_runtime_gpu_pool(os.environ)
        gpu_pool = str(gpu_count)
    except (FullChainManifestError, ValueError):
        sys.stderr.write("input_error\n")
        return 2

    try:
        completed = _execute(manifest, gpu_pool)
    except (OSError, ValueError):
        completed = False
    if not completed:
        sys.stderr.write("full_chain_failed\n")
        return 1
    sys.stdout.write("completed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
