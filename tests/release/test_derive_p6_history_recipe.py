from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml
import pytest

from framework.stage6.p6_history_recipe_profiles_v1 import RECIPE_V2
from tests.stage6.test_p6_history_normalization import (
    _as_v2_procedural,
    valid_private_source_map,
)
from tests.stage6.test_p6_post_source_adapter_profile import (
    v3_private_source_map,
    v4_private_source_map,
    v5_private_source_map,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DERIVER = REPOSITORY_ROOT / "tools/release/derive_p6_history_recipe.py"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic executable\n", encoding="utf-8")
    path.chmod(0o700)


def _private_derivation_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_map, runner_template = _as_v2_procedural(
        valid_private_source_map(tmp_path), tmp_path
    )
    history_root = Path(source_map["history_root"])
    source_map_path = _write_json(
        tmp_path / "private-inputs" / "source-map.json", source_map
    )
    return source_map_path, runner_template, history_root


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DERIVER), *args],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_derives_recipe_to_absolute_private_output_without_private_echo(
    tmp_path: Path,
) -> None:
    source_map, runner_template, _ = _private_derivation_inputs(tmp_path)
    recipe_json = tmp_path / "private-output" / "derived-recipe.json"
    recipe_json.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_map),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_json),
    )

    assert result.returncode == 0
    assert result.stdout == "p6_history_recipe_derived\n"
    assert result.stderr == ""
    recipe = json.loads(recipe_json.read_text(encoding="utf-8"))
    assert recipe["schema_version"] == RECIPE_V2
    assert str(tmp_path) not in recipe_json.read_text(encoding="utf-8")


@pytest.mark.parametrize("suffix", ("yaml", "yml"))
def test_cli_loads_yaml_and_rejects_duplicate_keys(
    tmp_path: Path, suffix: str
) -> None:
    source_map, runner_template = _as_v2_procedural(
        valid_private_source_map(tmp_path), tmp_path
    )
    source_path = tmp_path / "private-inputs" / f"source-map.{suffix}"
    source_path.write_text(yaml.safe_dump(source_map), encoding="utf-8")
    recipe_path = tmp_path / "private-output" / "recipe.json"
    recipe_path.parent.mkdir()

    accepted = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )
    assert accepted.returncode == 0

    recipe_path.unlink()
    source_path.write_text(
        source_path.read_text(encoding="utf-8")
        + "schema_version: p6_history_normalization_source_v2\n",
        encoding="utf-8",
    )
    rejected = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )
    assert rejected.returncode == 1
    assert rejected.stderr == "history_recipe_derivation_invalid\n"
    assert not recipe_path.exists()


def test_cli_derives_recipe_from_v3_source_map_without_private_leaf_echo(
    tmp_path: Path,
) -> None:
    source_map, runner_template = v3_private_source_map(tmp_path)
    source_path = _write_json(
        tmp_path / "private-inputs" / "source-map-v3.json", source_map
    )
    recipe_path = tmp_path / "private-output" / "recipe.json"
    recipe_path.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )

    assert result.returncode == 0
    assert result.stdout == "p6_history_recipe_derived\n"
    assert result.stderr == ""
    assert json.loads(recipe_path.read_text(encoding="utf-8"))["schema_version"] == RECIPE_V2
    assert "quant-contract.leaf.py" not in recipe_path.read_text(encoding="utf-8")


def test_cli_derives_recipe_from_exact_v4_source_map_without_runtime_echo(
    tmp_path: Path,
) -> None:
    source_map, runner_template = v4_private_source_map(tmp_path)
    source_path = _write_json(
        tmp_path / "private-inputs" / "source-map-v4.json", source_map
    )
    recipe_path = tmp_path / "private-output" / "recipe.json"
    recipe_path.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    text = recipe_path.read_text(encoding="utf-8")
    assert json.loads(text)["schema_version"] == RECIPE_V2
    assert "adapter_python" not in text
    assert "/usr/bin/python3.10" not in text


def test_cli_derives_recipe_from_exact_v5_source_map_without_dependency_echo(
    tmp_path: Path,
) -> None:
    source_map, runner_template = v5_private_source_map(tmp_path)
    source_path = _write_json(
        tmp_path / "private-inputs" / "source-map-v5.json", source_map
    )
    recipe_path = tmp_path / "private-output" / "recipe.json"
    recipe_path.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )

    assert result.returncode == 0
    text = recipe_path.read_text(encoding="utf-8")
    assert json.loads(text)["schema_version"] == RECIPE_V2
    assert "adapter_dependency_closure_id" not in text
    assert "dependency-overlay" not in text


@pytest.mark.parametrize("hardware_profile", ("h800", "rtx4090"))
def test_cli_derives_recipe_from_v5_registered_hardware_profile_without_profile_echo(
    tmp_path: Path, hardware_profile: str,
) -> None:
    source_map, runner_template = v5_private_source_map(tmp_path)
    source_map["hardware_profile"] = hardware_profile
    source_path = _write_json(
        tmp_path / "private-inputs" / "source-map-v5-rtx.json", source_map
    )
    recipe_path = tmp_path / "private-output" / "recipe.json"
    recipe_path.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )

    assert result.returncode == 0
    assert result.stdout == "p6_history_recipe_derived\n"
    assert result.stderr == ""
    text = recipe_path.read_text(encoding="utf-8")
    assert json.loads(text)["schema_version"] == RECIPE_V2
    assert "hardware_profile" not in text
    assert hardware_profile not in text


@pytest.mark.parametrize(
    ("hardware_profile", "case"),
    (
        ("unknown-private-profile", "unknown"),
        ({"profile_id": "rtx4090"}, "non-string"),
        (" rtx4090 ", "malformed"),
    ),
)
def test_cli_rejects_unregistered_v5_hardware_profile_without_profile_echo(
    tmp_path: Path, hardware_profile: object, case: str,
) -> None:
    source_map, runner_template = v5_private_source_map(tmp_path)
    source_map["hardware_profile"] = hardware_profile
    source_path = _write_json(
        tmp_path / "private-inputs" / f"source-map-v5-profile-{case}.json",
        source_map,
    )
    recipe_path = tmp_path / "private-output" / "recipe.json"
    recipe_path.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert "unknown-private-profile" not in result.stderr
    assert "rtx4090" not in result.stderr
    assert not recipe_path.exists()


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_cli_rejects_nonexact_v5_source_map_keys(
    tmp_path: Path,
    mutation: str,
) -> None:
    source_map, runner_template = v5_private_source_map(tmp_path)
    if mutation == "missing":
        source_map.pop("adapter_dependency_closure_id")
    else:
        source_map["unexpected"] = "rejected"
    source_path = _write_json(
        tmp_path / "private-inputs" / f"source-map-v5-{mutation}.json", source_map
    )
    recipe_path = tmp_path / "private-output" / "recipe.json"
    recipe_path.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )

    assert result.returncode == 1
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert not recipe_path.exists()


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_cli_rejects_nonexact_v4_source_map_keys(
    tmp_path: Path,
    mutation: str,
) -> None:
    source_map, runner_template = v4_private_source_map(tmp_path)
    if mutation == "missing":
        source_map.pop("adapter_python")
    else:
        source_map["unexpected"] = "rejected"
    source_path = _write_json(
        tmp_path / "private-inputs" / f"source-map-v4-{mutation}.json", source_map
    )
    recipe_path = tmp_path / "private-output" / "recipe.json"
    recipe_path.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_path),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_path),
    )

    assert result.returncode == 1
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert not recipe_path.exists()


def test_cli_redacts_derivation_failure_and_preserves_existing_output(
    tmp_path: Path,
) -> None:
    source_map, runner_template, _ = _private_derivation_inputs(tmp_path)
    payload = json.loads(source_map.read_text(encoding="utf-8"))
    payload["procedural_recipe_profile"] = "private-invalid-profile"
    source_map.write_text(json.dumps(payload), encoding="utf-8")
    recipe_json = tmp_path / "private-output" / "derived-recipe.json"
    recipe_json.parent.mkdir()
    recipe_json.write_text("preserve-existing\n", encoding="utf-8")

    result = _run_cli(
        "--source-map",
        str(source_map),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_json),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert "private-invalid-profile" not in result.stderr
    assert recipe_json.read_text(encoding="utf-8") == "preserve-existing\n"
    assert not tuple(recipe_json.parent.glob(f".{recipe_json.name}.*.tmp"))


def test_cli_rejects_missing_procedural_role_refs_without_recipe(
    tmp_path: Path,
) -> None:
    source_map, runner_template, _ = _private_derivation_inputs(tmp_path)
    payload = json.loads(source_map.read_text(encoding="utf-8"))
    payload["procedural_recipe_source"] = {}
    source_map.write_text(json.dumps(payload), encoding="utf-8")
    recipe_json = tmp_path / "private-output" / "derived-recipe.json"
    recipe_json.parent.mkdir()

    result = _run_cli(
        "--source-map",
        str(source_map),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_json),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert not recipe_json.exists()


def test_cli_requires_absolute_paths_without_creating_output(tmp_path: Path) -> None:
    source_map, runner_template, _ = _private_derivation_inputs(tmp_path)

    result = _run_cli(
        "--source-map",
        str(source_map),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        "relative-recipe.json",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == "argument_error\n"


def test_cli_rejects_unignored_repository_output_without_modifying_it(
    tmp_path: Path,
) -> None:
    source_map, runner_template, _ = _private_derivation_inputs(tmp_path)
    recipe_json = REPOSITORY_ROOT / f".p6-unignored-recipe-{tmp_path.name}.json"
    recipe_json.write_text("preserve-unignored\n", encoding="utf-8")
    try:
        result = _run_cli(
            "--source-map",
            str(source_map),
            "--runner-template",
            str(runner_template),
            "--recipe-json",
            str(recipe_json),
        )
        preserved = recipe_json.read_text(encoding="utf-8")
    finally:
        recipe_json.unlink(missing_ok=True)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert preserved == "preserve-unignored\n"


def test_cli_rejects_symlinked_output_parent_without_partial_recipe(
    tmp_path: Path,
) -> None:
    source_map, runner_template, _ = _private_derivation_inputs(tmp_path)
    real_parent = tmp_path / "real-private-output"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-private-output"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    recipe_json = linked_parent / "derived-recipe.json"

    result = _run_cli(
        "--source-map",
        str(source_map),
        "--runner-template",
        str(runner_template),
        "--recipe-json",
        str(recipe_json),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "history_recipe_derivation_invalid\n"
    assert not (real_parent / recipe_json.name).exists()
