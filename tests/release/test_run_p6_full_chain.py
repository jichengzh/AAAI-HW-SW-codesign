"""Acceptance tests for the single-command P6 private full-chain orchestrator."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/run_p6_full_chain.py"


def _module() -> Any:
    return importlib.import_module("tools.release.run_p6_full_chain")


def _manifest(tmp_path: Path, *, profile: str = "rtx4090") -> SimpleNamespace:
    private_inputs = tmp_path / "private-inputs"
    private_inputs.mkdir()
    history_root = tmp_path / "history-root"
    history_root.mkdir()
    source_map = private_inputs / "source-map.yaml"
    source_map.write_text(
        yaml.safe_dump(
            {
                "schema_version": "p6_history_normalization_source_v5",
                "hardware_profile": profile,
                "history_root": str(history_root),
            }
        ),
        encoding="utf-8",
    )
    runner = private_inputs / "runner-template.yaml"
    runner.write_text(
        yaml.safe_dump({"schema_version": "p6_history_runner_template_v1"}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        schema_version="p6_full_chain_run_manifest_v1",
        template_only=False,
        hardware_profile=profile,
        contract=ROOT / f"configs/execution/p6_{profile}_search.example.yaml",
        source_map=source_map,
        history_root=history_root,
        pre_normalization_runner_template=runner,
        normalized_private_dir=tmp_path / "normalized-private",
        fresh_output_root=tmp_path / "fresh-output",
    )


def _write_manifest_file(
    tmp_path: Path,
    *,
    profile: str = "rtx4090",
    overrides: dict[str, object] | None = None,
) -> tuple[Path, SimpleNamespace]:
    manifest = _manifest(tmp_path, profile=profile)
    payload: dict[str, object] = {
        "schema_version": manifest.schema_version,
        "template_only": manifest.template_only,
        "hardware_profile": manifest.hardware_profile,
        "contract": f"configs/execution/p6_{profile}_search.example.yaml",
        "source_map": str(manifest.source_map),
        "history_root": str(manifest.history_root),
        "pre_normalization_runner_template": str(
            manifest.pre_normalization_runner_template
        ),
        "normalized_private_dir": str(manifest.normalized_private_dir),
        "fresh_output_root": str(manifest.fresh_output_root),
    }
    payload.update(overrides or {})
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path, manifest


def _argv(manifest_path: Path, *, check_inputs: bool = False) -> list[str]:
    values = ["--manifest", str(manifest_path)]
    if check_inputs:
        values.append("--check-inputs")
    return values


def test_help_exposes_manifest_and_static_input_check() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
    assert "--check-inputs" in result.stdout


def test_check_inputs_is_read_only_and_does_not_require_gpu_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _module()
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "run.yaml"
    manifest_path.write_text("fixture: delegated\n", encoding="utf-8")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    monkeypatch.delenv("GPU_POOL", raising=False)
    monkeypatch.setattr(cli, "_load_and_validate_manifest", lambda path: manifest)
    monkeypatch.setattr(cli, "_validate_private_authorities", lambda manifest: None)
    monkeypatch.setattr(
        cli,
        "_run_command",
        lambda *args, **kwargs: pytest.fail("--check-inputs launched a stage"),
    )

    assert cli.main(_argv(manifest_path, check_inputs=True)) == 0
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert not manifest.normalized_private_dir.exists()
    assert not manifest.fresh_output_root.exists()


@pytest.mark.parametrize("profile", ["h800", "rtx4090"])
def test_real_manifest_parser_accepts_complete_fresh_profile_inputs(
    tmp_path: Path, profile: str
) -> None:
    cli = _module()
    manifest_path, expected = _write_manifest_file(tmp_path, profile=profile)

    manifest = cli._load_and_validate_manifest(manifest_path)

    assert manifest.hardware_profile == profile
    assert manifest.contract == expected.contract.resolve()
    assert manifest.source_map == expected.source_map.resolve()
    assert manifest.history_root == expected.history_root.resolve()
    assert manifest.pre_normalization_runner_template == (
        expected.pre_normalization_runner_template.resolve()
    )
    assert manifest.normalized_private_dir == expected.normalized_private_dir
    assert manifest.fresh_output_root == expected.fresh_output_root


def test_real_check_inputs_validates_without_writing_or_gpu_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _module()
    manifest_path, manifest = _write_manifest_file(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    monkeypatch.delenv("GPU_POOL", raising=False)
    monkeypatch.setattr(cli, "_validate_private_authorities", lambda manifest: None)

    assert cli.main(_argv(manifest_path, check_inputs=True)) == 0

    assert capsys.readouterr().out == "inputs_ready\n"
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert not manifest.normalized_private_dir.exists()
    assert not manifest.fresh_output_root.exists()


def test_check_inputs_rejects_incomplete_private_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    manifest_path, _ = _write_manifest_file(tmp_path)
    monkeypatch.delenv("GPU_POOL", raising=False)

    assert cli.main(_argv(manifest_path, check_inputs=True)) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "input_error\n"


def test_check_inputs_rejects_malformed_subordinate_json_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _module()
    manifest = _manifest(tmp_path, profile="h800")
    input_sources: dict[str, str] = {}
    for name in (
        "gold176_rows",
        "gold176_graph_features",
        "capability_profiles",
        "closure",
    ):
        path = manifest.history_root / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        input_sources[name] = str(path)
    source_map = yaml.safe_load(manifest.source_map.read_text(encoding="utf-8"))
    source_map["input_sources"] = input_sources
    manifest.source_map.write_text(
        yaml.safe_dump(source_map, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(cli, "validate_history_inputs", lambda *args, **kwargs: None)

    with pytest.raises(ValueError):
        cli._validate_private_authorities(manifest)


def test_private_locations_must_be_outside_the_repository_or_git_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _module()
    repository = tmp_path / "public-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    visible = repository / "private-input.yaml"
    visible.write_text("private: true\n", encoding="utf-8")
    ignored = repository / "ignored-private.yaml"
    ignored.write_text("private: true\n", encoding="utf-8")
    (repository / ".gitignore").write_text("ignored-private.yaml\nignored-output/\n", encoding="utf-8")
    outside = tmp_path / "outside-private.yaml"
    outside.write_text("private: true\n", encoding="utf-8")
    visible_link = repository / "visible-private-link.yaml"
    visible_link.symlink_to(outside)
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", repository)

    with pytest.raises(cli.FullChainManifestError, match="not private"):
        cli._require_private_location(visible)

    cli._require_private_location(ignored)
    cli._require_private_location(repository / "ignored-output")
    cli._require_private_location(outside)
    with pytest.raises(cli.FullChainManifestError, match="not private"):
        cli._require_private_location(visible_link)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": "unsupported"},
        {"template_only": True},
        {"template_only": "false"},
        {"hardware_profile": "unknown"},
        {"contract": "configs/execution/p6_h800_search.example.yaml"},
    ],
)
def test_real_manifest_parser_rejects_non_executable_or_mismatched_authority(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    cli = _module()
    manifest_path, _ = _write_manifest_file(tmp_path, overrides=overrides)

    with pytest.raises(cli.FullChainManifestError):
        cli._load_and_validate_manifest(manifest_path)


@pytest.mark.parametrize("key", ["contract", "source_map", "fresh_output_root"])
def test_real_manifest_parser_requires_the_exact_key_set(
    tmp_path: Path, key: str
) -> None:
    cli = _module()
    manifest_path, _ = _write_manifest_file(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload.pop(key)
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(cli.FullChainManifestError, match="keys"):
        cli._load_and_validate_manifest(manifest_path)


def test_yaml_loader_rejects_duplicate_keys_and_non_mapping_documents(
    tmp_path: Path,
) -> None:
    cli = _module()
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: one\nschema_version: two\n", encoding="utf-8")
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("not-a-mapping\n", encoding="utf-8")

    with pytest.raises(cli.FullChainManifestError, match="unavailable"):
        cli._load_yaml_mapping(duplicate)
    with pytest.raises(cli.FullChainManifestError, match="invalid"):
        cli._load_yaml_mapping(scalar)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_map", "relative.yaml"),
        ("history_root", "relative-directory"),
        ("pre_normalization_runner_template", None),
    ],
)
def test_real_manifest_parser_rejects_non_absolute_private_inputs(
    tmp_path: Path, field: str, replacement: object
) -> None:
    cli = _module()
    manifest_path, _ = _write_manifest_file(tmp_path, overrides={field: replacement})

    with pytest.raises(cli.FullChainManifestError, match="private input"):
        cli._load_and_validate_manifest(manifest_path)


def test_real_manifest_parser_rejects_missing_and_wrong_kind_private_inputs(
    tmp_path: Path,
) -> None:
    cli = _module()
    missing = tmp_path / "missing.yaml"
    manifest_path, _ = _write_manifest_file(
        tmp_path, overrides={"source_map": str(missing)}
    )
    with pytest.raises(cli.FullChainManifestError, match="private input"):
        cli._load_and_validate_manifest(manifest_path)

    wrong_kind = tmp_path / "wrong-kind"
    wrong_kind.mkdir()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["source_map"] = str(wrong_kind)
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(cli.FullChainManifestError, match="private input"):
        cli._load_and_validate_manifest(manifest_path)


@pytest.mark.parametrize("field", ["normalized_private_dir", "fresh_output_root"])
def test_real_manifest_parser_rejects_existing_or_parentless_outputs(
    tmp_path: Path, field: str
) -> None:
    cli = _module()
    existing = tmp_path / f"existing-{field}"
    existing.mkdir()
    manifest_path, _ = _write_manifest_file(
        tmp_path, overrides={field: str(existing)}
    )
    with pytest.raises(cli.FullChainManifestError, match="not fresh"):
        cli._load_and_validate_manifest(manifest_path)

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload[field] = str(tmp_path / "missing-parent" / "output")
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(cli.FullChainManifestError, match="unavailable"):
        cli._load_and_validate_manifest(manifest_path)


def test_real_manifest_parser_requires_distinct_output_roots(tmp_path: Path) -> None:
    cli = _module()
    same = tmp_path / "same-output"
    manifest_path, _ = _write_manifest_file(
        tmp_path,
        overrides={
            "normalized_private_dir": str(same),
            "fresh_output_root": str(same),
        },
    )

    with pytest.raises(cli.FullChainManifestError, match="must differ"):
        cli._load_and_validate_manifest(manifest_path)


@pytest.mark.parametrize(
    ("source_overrides", "message"),
    [
        ({"schema_version": "old"}, "source-map authority"),
        ({"hardware_profile": "h800"}, "source-map authority"),
        ({"history_root": "/different/root"}, "source-map authority"),
    ],
)
def test_real_manifest_parser_rejects_source_map_authority_drift(
    tmp_path: Path,
    source_overrides: dict[str, object],
    message: str,
) -> None:
    cli = _module()
    manifest_path, manifest = _write_manifest_file(tmp_path)
    source = yaml.safe_load(manifest.source_map.read_text(encoding="utf-8"))
    source.update(source_overrides)
    manifest.source_map.write_text(yaml.safe_dump(source), encoding="utf-8")

    with pytest.raises(cli.FullChainManifestError, match=message):
        cli._load_and_validate_manifest(manifest_path)


def test_real_manifest_parser_rejects_runner_schema_drift(tmp_path: Path) -> None:
    cli = _module()
    manifest_path, manifest = _write_manifest_file(tmp_path)
    manifest.pre_normalization_runner_template.write_text(
        "schema_version: old\n", encoding="utf-8"
    )

    with pytest.raises(cli.FullChainManifestError, match="runner-template"):
        cli._load_and_validate_manifest(manifest_path)


@pytest.mark.parametrize("gpu_pool", [None, "", "0", "-1", "01", "1,2", " 1"])
def test_full_run_rejects_missing_or_noncanonical_gpu_pool_before_any_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gpu_pool: str | None,
) -> None:
    cli = _module()
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "run.yaml"
    manifest_path.write_text("fixture: delegated\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_load_and_validate_manifest", lambda path: manifest)
    monkeypatch.setattr(cli, "_validate_private_authorities", lambda manifest: None)
    monkeypatch.setattr(
        cli,
        "_run_command",
        lambda *args, **kwargs: pytest.fail("invalid GPU_POOL launched a stage"),
    )
    if gpu_pool is None:
        monkeypatch.delenv("GPU_POOL", raising=False)
    else:
        monkeypatch.setenv("GPU_POOL", gpu_pool)

    assert cli.main(_argv(manifest_path)) == 2
    assert not manifest.normalized_private_dir.exists()
    assert not manifest.fresh_output_root.exists()


def test_full_run_derives_normalizes_runs_and_verifies_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _module()
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "run.yaml"
    manifest_path.write_text("fixture: delegated\n", encoding="utf-8")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    monkeypatch.setenv("GPU_POOL", "3")
    monkeypatch.setattr(cli, "_load_and_validate_manifest", lambda path: manifest)
    monkeypatch.setattr(cli, "_validate_private_authorities", lambda manifest: None)
    monkeypatch.setattr(cli, "_git_revision", lambda: "abc123def456")

    def run_command(argv: tuple[str, ...], *, env: dict[str, str]) -> int:
        calls.append((argv, env))
        executable = Path(argv[1]).name
        if executable == "derive_p6_history_recipe.py":
            recipe = Path(argv[argv.index("--recipe-json") + 1])
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text('{"recipe":"same"}\n', encoding="utf-8")
        elif executable == "normalize_p6_history_root.py":
            recipe = manifest.normalized_private_dir / "derivation/recipe.json"
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text('{"recipe":"same"}\n', encoding="utf-8")
            for relative in (
                "legacy.local.yaml",
                "runner-template.yaml",
                "source-wrapper-profile.yaml",
                "external-training-binding.yaml",
                "post-source-adapter-profile.yaml",
            ):
                (manifest.normalized_private_dir / relative).write_text(
                    "fixture: true\n", encoding="utf-8"
                )
        return 0

    monkeypatch.setattr(cli, "_run_command", run_command)

    assert cli.main(_argv(manifest_path)) == 0
    names = [Path(argv[1]).name for argv, _ in calls]
    assert names == [
        "derive_p6_history_recipe.py",
        "normalize_p6_history_root.py",
        "run_p6_h800_search.py",
        "verify_p6_materializer_training_run.py",
    ]

    derive, normalize, run, verify = (argv for argv, _ in calls)
    temporary_recipe = Path(derive[derive.index("--recipe-json") + 1])
    assert temporary_recipe.parent == manifest.normalized_private_dir.parent
    assert temporary_recipe != manifest.normalized_private_dir / "derivation/recipe.json"
    assert not temporary_recipe.exists()
    assert derive == (
        sys.executable,
        str(ROOT / "tools/release/derive_p6_history_recipe.py"),
        "--source-map",
        str(manifest.source_map),
        "--runner-template",
        str(manifest.pre_normalization_runner_template),
        "--recipe-json",
        str(temporary_recipe),
    )
    assert normalize == (
        sys.executable,
        str(ROOT / "tools/release/normalize_p6_history_root.py"),
        "--source-map",
        str(manifest.source_map),
        "--history-root",
        str(manifest.history_root),
        "--private-dir",
        str(manifest.normalized_private_dir),
        "--runner-template",
        str(manifest.pre_normalization_runner_template),
    )
    assert run == (
        sys.executable,
        str(ROOT / "tools/release/run_p6_h800_search.py"),
        "--contract",
        str(manifest.contract),
        "--code-revision",
        "abc123def456",
        "--legacy-local-config",
        str(manifest.normalized_private_dir / "legacy.local.yaml"),
        "--runner-template",
        str(manifest.normalized_private_dir / "runner-template.yaml"),
        "--local-output-root",
        str(manifest.fresh_output_root),
        "--binding-output",
        str(manifest.fresh_output_root / "binding.json"),
        "--config-output",
        str(manifest.fresh_output_root / "local-config.yaml"),
        "--source-wrapper-profile",
        str(manifest.normalized_private_dir / "source-wrapper-profile.yaml"),
        "--external-training-binding",
        str(manifest.normalized_private_dir / "external-training-binding.yaml"),
        "--post-source-adapter-profile",
        str(manifest.normalized_private_dir / "post-source-adapter-profile.yaml"),
    )
    assert verify == (
        sys.executable,
        str(ROOT / "tools/release/verify_p6_materializer_training_run.py"),
        "--contract",
        str(manifest.contract),
        "--local-config",
        str(manifest.fresh_output_root / "local-config.yaml"),
        "--binding",
        str(manifest.fresh_output_root / "binding.json"),
    )
    assert calls[2][1]["GPU_POOL"] == "3"


def test_recipe_mismatch_stops_before_controller_and_cleans_temporary_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _module()
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "run.yaml"
    manifest_path.write_text("fixture: delegated\n", encoding="utf-8")
    calls: list[str] = []
    temporary_recipe: Path | None = None

    monkeypatch.setenv("GPU_POOL", "1")
    monkeypatch.setattr(cli, "_load_and_validate_manifest", lambda path: manifest)
    monkeypatch.setattr(cli, "_validate_private_authorities", lambda manifest: None)

    def run_command(argv: tuple[str, ...], *, env: dict[str, str]) -> int:
        nonlocal temporary_recipe
        name = Path(argv[1]).name
        calls.append(name)
        if name == "derive_p6_history_recipe.py":
            temporary_recipe = Path(argv[argv.index("--recipe-json") + 1])
            temporary_recipe.parent.mkdir(parents=True, exist_ok=True)
            temporary_recipe.write_text('{"recipe":"first"}\n', encoding="utf-8")
        elif name == "normalize_p6_history_root.py":
            normalized_recipe = manifest.normalized_private_dir / "derivation/recipe.json"
            normalized_recipe.parent.mkdir(parents=True, exist_ok=True)
            normalized_recipe.write_text('{"recipe":"different"}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(cli, "_run_command", run_command)

    assert cli.main(_argv(manifest_path)) != 0
    assert calls == ["derive_p6_history_recipe.py", "normalize_p6_history_root.py"]
    assert temporary_recipe is not None
    assert not temporary_recipe.exists()


@pytest.mark.parametrize("failed_stage", range(4))
def test_full_run_stops_at_the_first_failed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: int,
) -> None:
    cli = _module()
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "run.yaml"
    manifest_path.write_text("fixture: delegated\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setenv("GPU_POOL", "1")
    monkeypatch.setattr(cli, "_load_and_validate_manifest", lambda path: manifest)
    monkeypatch.setattr(cli, "_validate_private_authorities", lambda manifest: None)

    def run_command(argv: tuple[str, ...], *, env: dict[str, str]) -> int:
        name = Path(argv[1]).name
        calls.append(name)
        if len(calls) - 1 == failed_stage:
            return 1
        if name == "derive_p6_history_recipe.py":
            path = Path(argv[argv.index("--recipe-json") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"recipe":"same"}\n', encoding="utf-8")
        elif name == "normalize_p6_history_root.py":
            path = manifest.normalized_private_dir / "derivation/recipe.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"recipe":"same"}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(cli, "_run_command", run_command)

    assert cli.main(_argv(manifest_path)) != 0
    expected = [
        "derive_p6_history_recipe.py",
        "normalize_p6_history_root.py",
        "run_p6_h800_search.py",
        "verify_p6_materializer_training_run.py",
    ]
    assert calls == expected[: failed_stage + 1]


def test_run_command_uses_repository_cwd_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    recorded: dict[str, object] = {}

    def run(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        recorded.update({"argv": argv, **kwargs})
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.subprocess, "run", run)
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}

    assert cli._run_command(("python", "stage.py"), env=environment) == 7
    assert recorded == {
        "argv": ("python", "stage.py"),
        "cwd": ROOT,
        "env": environment,
        "shell": False,
        "check": False,
    }


def test_git_revision_returns_exact_lowercase_twelve_hex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123def456\n"),
    )

    assert cli._git_revision() == "abc123def456"


@pytest.mark.parametrize("revision", ["", "abc123", "ABC123DEF456", "g" * 12])
def test_git_revision_rejects_malformed_output(
    monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    cli = _module()
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=revision),
    )

    with pytest.raises(cli.FullChainManifestError, match="invalid"):
        cli._git_revision()


@pytest.mark.parametrize(
    "error",
    [OSError("git unavailable"), subprocess.CalledProcessError(1, ["git"])],
)
def test_git_revision_wraps_process_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    cli = _module()

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(cli.subprocess, "run", fail)
    with pytest.raises(cli.FullChainManifestError, match="unavailable"):
        cli._git_revision()


def test_recipe_comparison_fails_closed_for_missing_or_invalid_json(
    tmp_path: Path,
) -> None:
    cli = _module()
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    valid.write_text('{"recipe":true}\n', encoding="utf-8")
    invalid.write_text("not-json\n", encoding="utf-8")

    assert cli._recipes_match(valid, invalid) is False
    assert cli._recipes_match(valid, tmp_path / "missing.json") is False


def test_main_reports_execution_oserror_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "run.yaml"
    manifest_path.write_text("fixture: delegated\n", encoding="utf-8")
    monkeypatch.setenv("GPU_POOL", "1")
    monkeypatch.setattr(cli, "_load_and_validate_manifest", lambda path: manifest)
    monkeypatch.setattr(cli, "_validate_private_authorities", lambda manifest: None)
    monkeypatch.setattr(
        cli,
        "_execute",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("execution failed")),
    )

    assert cli.main(_argv(manifest_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "full_chain_failed\n"
