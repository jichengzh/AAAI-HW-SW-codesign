from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Callable

import pytest
import yaml

import framework.stage6.p6_history_execution_closure_v1 as closure_module
from framework.stage6.p6_external_training_binding_v1 import (
    validate_external_training_binding,
)
from framework.stage6.p6_history_execution_closure_v1 import (
    EXECUTION_CLOSURE_ROLES,
    P6ExecutionClosureError,
    copy_execution_closure,
    render_normalized_runner_template,
    validate_execution_closure_manifest,
    validate_normalized_runner_closure,
)
from framework.stage6.p6_runner_template_validator_v1 import (
    validate_pre_provision_runner_template,
)
from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_history_recipe_profiles_v1 import SHARED_SOURCE_PATH_KEYS
from framework.stage6.p6_history_training_contract_v1 import (
    P6HistoryTrainingContractError,
    validate_projected_training_contract,
)
from framework.stage6.p6_source_wrapper_profile_v1 import (
    render_self_contained_source_wrapper,
)
from tests.stage6.test_p6_runner_template_validator import _runner_template


def _tree_sha(root: Path) -> str:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append({"kind": "directory", "path": relative})
        else:
            payload = path.read_bytes()
            entries.append(
                {
                    "kind": "regular_file",
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
    encoded = json.dumps(
        entries,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o700)
    return path


def _fixture(tmp_path: Path) -> dict[str, Any]:
    source_root = tmp_path / "source-history"
    source_root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(source_root)], check=True)
    closure_root = source_root / "repo"
    source = _write_executable(
        closure_root / "source.py",
        "#!/usr/bin/env python3\nimport sibling\nassert sibling.VALUE == 'ok'\n",
    )
    source_marker = _write_executable(
        source_root / "documented-stage5-chain/stage5_materialize_round_sources_v1.sh"
    )
    (closure_root / "sibling.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    role_files = {
        "stage1_scan": _write_executable(closure_root / "scan"),
        "controller": _write_executable(closure_root / "stage5_task_round_controller_v3.sh"),
        "source_materializer": source,
        "quantization": _write_executable(closure_root / "quantize"),
        "performance": _write_executable(closure_root / "stage5_build_performance_plan_v2.py"),
        "ap": _write_executable(closure_root / "ap"),
        "finalization": _write_executable(closure_root / "stage5_finalize_feedback_v2.py"),
        "activation": _write_executable(closure_root / "activate"),
    }
    code_root = tmp_path / "code-boundary"
    output_root = tmp_path / "output"
    dataset = tmp_path / "dataset"
    stable = tmp_path / "stable"
    for path in (code_root, output_root, dataset, stable):
        path.mkdir()
    checkpoint = stable / "base.ckpt"
    config = stable / "pyramid.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_bytes(b"config")
    external = validate_external_training_binding(
        {
            "schema_version": "p6_external_training_binding_v1",
            "training_required": True,
            "training_source_kind": "selected_candidate_finetune",
            "dataset_root": str(dataset),
            "base_checkpoint_path": str(checkpoint),
            "base_checkpoint_sha256": None,
            "pyramid_config_path": str(config),
            "pyramid_config_sha256": None,
            "training_parameters": {
                "training_mode": "finetune",
                "epochs": 1,
                "seed": 0,
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "batch_size": 1,
                "dataset_split": "train",
                "checkpoint_selection": "best",
                "freeze_policy": "partial",
            },
        },
        code_toolchain_root=code_root,
        local_output_root=output_root,
        reserved_paths=(),
    )
    manifest = {
        "schema_version": "p6_execution_code_closure_v1",
        "roots": [
            {
                "closure_id": "history",
                "source_root": str(closure_root),
                "destination_relative_root": "execution-closure/history",
                "sha256": _tree_sha(closure_root),
            }
        ],
        "roles": {
            role: {
                "closure_id": "history",
                "entrypoint_relative_path": str(path.relative_to(closure_root)),
            }
            for role, path in role_files.items()
        },
    }
    staged = tmp_path / "normalized"
    staged.mkdir()
    subprocess.run(["git", "init", "-q", str(staged)], check=True)
    return {
        "source_root": source_root,
        "closure_root": closure_root,
        "manifest": manifest,
        "external": external,
        "source_marker": source_marker,
        "staged": staged,
    }


def test_closure_copies_all_roles_and_import_siblings(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )

    copied = copy_execution_closure(validated, staged_private_root=fixture["staged"])

    assert set(copied) == set(EXECUTION_CLOSURE_ROLES)
    assert (fixture["staged"] / "execution-closure/history/sibling.py").is_file()
    assert all(path.is_relative_to(fixture["staged"]) for path in copied.values())
    assert not fixture["external"].dataset_root.is_relative_to(fixture["staged"])
    assert not fixture["external"].base_checkpoint_path.is_relative_to(fixture["staged"])
    assert not fixture["external"].pyramid_config_path.is_relative_to(fixture["staged"])


def test_closure_digest_sorts_complete_posix_relative_paths_before_copy_recheck(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    closure_root = fixture["closure_root"]
    nested = closure_root / "a"
    nested.mkdir()
    (nested / "file.txt").write_text("nested\n", encoding="utf-8")
    (closure_root / "a.b").write_text("sibling\n", encoding="utf-8")
    approved_digest = _tree_sha(closure_root)
    fixture["manifest"]["roots"][0]["sha256"] = approved_digest

    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )
    copied = copy_execution_closure(validated, staged_private_root=fixture["staged"])

    copied_root = copied["source_materializer"].parent
    assert _tree_sha(copied_root) == approved_digest
    bad_manifest = copy.deepcopy(fixture["manifest"])
    bad_manifest["roots"][0]["sha256"] = "0" * 64
    with pytest.raises(P6ExecutionClosureError):
        validate_execution_closure_manifest(
            bad_manifest,
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )


def test_closure_copies_read_only_directories_and_restores_exact_modes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_root = fixture["closure_root"]
    nested = source_root / "read-only" / "nested"
    nested.mkdir(parents=True)
    payload = nested / "payload.txt"
    payload.write_text("immutable\n", encoding="utf-8")
    payload.chmod(0o440)
    nested.chmod(0o550)
    nested.parent.chmod(0o500)
    source_root.chmod(0o500)
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(source_root)
    source_modes = {
        path.relative_to(source_root).as_posix(): stat.S_IMODE(path.lstat().st_mode)
        for path in (source_root, nested.parent, nested, payload)
    }

    try:
        validated = validate_execution_closure_manifest(
            fixture["manifest"],
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )
        copied = copy_execution_closure(validated, staged_private_root=fixture["staged"])
        copied_root = copied["source_materializer"].parent
        copied_modes = {
            path.relative_to(copied_root).as_posix(): stat.S_IMODE(path.lstat().st_mode)
            for path in (
                copied_root,
                copied_root / "read-only",
                copied_root / "read-only/nested",
                copied_root / "read-only/nested/payload.txt",
            )
        }

        assert copied_modes == source_modes
        assert _tree_sha(copied_root) == fixture["manifest"]["roots"][0]["sha256"]
        assert payload.read_bytes() == (copied_root / "read-only/nested/payload.txt").read_bytes()
        assert {
            path.relative_to(source_root).as_posix(): stat.S_IMODE(path.lstat().st_mode)
            for path in (source_root, nested.parent, nested, payload)
        } == source_modes
    finally:
        for path in (
            fixture["staged"] / "execution-closure/history/read-only/nested",
            fixture["staged"] / "execution-closure/history/read-only",
            fixture["staged"] / "execution-closure/history",
            nested,
            nested.parent,
            source_root,
        ):
            if path.exists():
                path.chmod(0o700)


def test_read_only_closure_copy_failure_cleans_temporary_without_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    source_root = fixture["closure_root"]
    nested = source_root / "read-only"
    nested.mkdir()
    (nested / "payload.txt").write_text("immutable\n", encoding="utf-8")
    nested.chmod(0o500)
    source_root.chmod(0o500)
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(source_root)
    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )
    real_digest = closure_module._tree_digest

    def reject_copied_digest(root: Path) -> str:
        if root.is_relative_to(fixture["staged"]):
            return "0" * 64
        return real_digest(root)

    monkeypatch.setattr(closure_module, "_tree_digest", reject_copied_digest)
    try:
        with pytest.raises(P6ExecutionClosureError):
            copy_execution_closure(validated, staged_private_root=fixture["staged"])
        assert not (fixture["staged"] / "execution-closure").exists()
        assert not tuple(fixture["staged"].glob(".execution-closure.*.tmp"))
        assert stat.S_IMODE(source_root.lstat().st_mode) == 0o500
        assert stat.S_IMODE(nested.lstat().st_mode) == 0o500
    finally:
        nested.chmod(0o700)
        source_root.chmod(0o700)


def test_closure_copy_has_no_fallible_validation_after_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )
    calls: list[Path] = []

    def pre_publish_checks_only(path: Path) -> bool:
        calls.append(path)
        return len(calls) <= len(EXECUTION_CLOSURE_ROLES)

    monkeypatch.setattr(closure_module, "_single_link_executable", pre_publish_checks_only)

    copied = copy_execution_closure(validated, staged_private_root=fixture["staged"])

    assert len(calls) == len(EXECUTION_CLOSURE_ROLES)
    assert set(copied) == set(EXECUTION_CLOSURE_ROLES)
    assert (fixture["staged"] / "execution-closure").is_dir()


def test_closure_copy_is_owner_writable_during_restrictive_umask(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["manifest"]["roots"][0]["destination_relative_root"] = (
        "execution-closure/nested/history"
    )
    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )
    previous_umask = os.umask(0o777)
    try:
        copied = copy_execution_closure(validated, staged_private_root=fixture["staged"])
    finally:
        os.umask(previous_umask)

    assert set(copied) == set(EXECUTION_CLOSURE_ROLES)
    assert all(path.is_file() for path in copied.values())
    assert stat.S_IMODE((fixture["staged"] / "execution-closure/nested").lstat().st_mode) == 0o700


def test_temporary_cleanup_unlocks_directories_top_down(tmp_path: Path) -> None:
    temporary = tmp_path / ".execution-closure.locked.tmp"
    locked = temporary / "owner-inaccessible" / "nested"
    locked.mkdir(parents=True)
    (locked / "payload.txt").write_text("temporary\n", encoding="utf-8")
    locked.chmod(0o000)
    locked.parent.chmod(0o000)

    try:
        closure_module._remove_temporary_tree(temporary)
        assert not temporary.exists()
    finally:
        for path in (temporary, locked.parent, locked):
            try:
                path.chmod(0o700)
            except FileNotFoundError:
                pass


def test_copied_source_runs_through_wrapper_without_pythonpath(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )
    copied = copy_execution_closure(validated, staged_private_root=fixture["staged"])
    wrapper = render_self_contained_source_wrapper(
        {
            "schema_version": "p6_private_source_wrapper_profile_v1",
            "wrapper_kind": "repo_cwd_exec_v1",
            "destination_relative_path": (
                "documented-stage5-chain/stage5_materialize_round_sources_v1.sh"
            ),
            "implementation_relative_path": copied["source_materializer"]
            .relative_to(fixture["staged"])
            .as_posix(),
            "implementation_cwd_relative_path": "execution-closure/history",
        },
        history_root=fixture["staged"],
    )
    round_root = fixture["staged"] / "runs/round-00"
    round_root.mkdir(parents=True)
    environment = {
        "CUDA_VISIBLE_DEVICES": "17,19,23",
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(fixture["staged"]),
        "P6_HISTORY_TASK_STATE": str(round_root / "task-state.json"),
        "P6_HISTORY_ROUND_OUTPUT_ROOT": str(round_root),
    }
    assert set(environment) == set(EXPECTED_HISTORY_ENV_KEYS)

    completed = subprocess.run(
        [
            str(wrapper.executable),
            "--request",
            str(round_root / "request.json"),
            "--model",
            "pyramid",
            "--group-id",
            "pyramid|16x32x64",
            "--gpu",
            "0",
        ],
        cwd=round_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value, _fixture: value["roles"].pop("activation"),
        lambda value, _fixture: value["roles"].__setitem__(
            "unknown", copy.deepcopy(value["roles"]["activation"])
        ),
        lambda value, _fixture: value["roots"][0].__setitem__(
            "destination_relative_root", "/absolute"
        ),
        lambda value, _fixture: value["roots"][0].__setitem__(
            "destination_relative_root", "execution-closure/../escape"
        ),
        lambda value, _fixture: value["roles"]["source_materializer"].__setitem__(
            "entrypoint_relative_path", "../escape"
        ),
        lambda value, _fixture: value["roots"][0].__setitem__("sha256", "0" * 64),
        lambda value, _fixture: value["roles"]["activation"].__setitem__("closure_id", "missing"),
        lambda value, _fixture: value["roots"][0].__setitem__(
            "source_root", f"{value['roots'][0]['source_root']}/"
        ),
        lambda value, _fixture: value["roots"][0].__setitem__(
            "destination_relative_root", "execution-closure/./history"
        ),
        lambda value, _fixture: value["roles"]["activation"].__setitem__(
            "entrypoint_relative_path", "./activate"
        ),
    ),
)
def test_closure_rejects_invalid_manifest(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any], dict[str, Any]], object],
) -> None:
    fixture = _fixture(tmp_path)
    raw = copy.deepcopy(fixture["manifest"])
    mutate(raw, fixture)

    with pytest.raises(P6ExecutionClosureError, match="^history_normalization_invalid$"):
        validate_execution_closure_manifest(
            raw,
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )


def test_closure_rejects_source_outside_history_and_external_overlap(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw = copy.deepcopy(fixture["manifest"])
    raw["roots"][0]["source_root"] = str(fixture["external"].dataset_root)
    raw["roots"][0]["sha256"] = _tree_sha(fixture["external"].dataset_root)

    with pytest.raises(P6ExecutionClosureError):
        validate_execution_closure_manifest(
            raw,
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )


@pytest.mark.parametrize(
    "body",
    (
        "from . import absent_sibling\n",
        "import absent_sibling\n",
        "from absent_sibling import value\n",
        "import framework\n",
    ),
)
def test_closure_rejects_missing_import_sibling(tmp_path: Path, body: str) -> None:
    fixture = _fixture(tmp_path)
    source = fixture["closure_root"] / "source.py"
    source.write_text(body, encoding="utf-8")
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(fixture["closure_root"])

    with pytest.raises(P6ExecutionClosureError):
        validate_execution_closure_manifest(
            fixture["manifest"],
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )


@pytest.mark.parametrize(
    "body",
    ("import sibling.missing\n", "from sibling.missing import value\n"),
)
def test_closure_rejects_missing_dotted_import_module(tmp_path: Path, body: str) -> None:
    fixture = _fixture(tmp_path)
    (fixture["closure_root"] / "sibling.py").unlink()
    package = fixture["closure_root"] / "sibling"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (fixture["closure_root"] / "source.py").write_text(body, encoding="utf-8")
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(fixture["closure_root"])

    with pytest.raises(P6ExecutionClosureError):
        validate_execution_closure_manifest(
            fixture["manifest"],
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )


def test_closure_rejects_missing_from_package_submodule(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    closure_root = fixture["closure_root"]
    (closure_root / "sibling.py").unlink()
    package = closure_root / "sibling"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (closure_root / "source.py").write_text("from sibling import missing\n", encoding="utf-8")
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(closure_root)

    with pytest.raises(P6ExecutionClosureError):
        validate_execution_closure_manifest(
            fixture["manifest"],
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )


@pytest.mark.parametrize("member", ("missing", "VALUE"))
def test_closure_accepts_package_submodule_or_exported_symbol(tmp_path: Path, member: str) -> None:
    fixture = _fixture(tmp_path)
    closure_root = fixture["closure_root"]
    (closure_root / "sibling.py").unlink()
    package = closure_root / "sibling"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    if member == "missing":
        (package / "missing.py").write_text("READY = True\n", encoding="utf-8")
    (closure_root / "source.py").write_text(f"from sibling import {member}\n", encoding="utf-8")
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(closure_root)

    validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )


@pytest.mark.parametrize(
    "module_body",
    ("VALUE = 'ok'\n", "missing: int\n", "missing += 1\n"),
)
def test_closure_rejects_unbound_plain_module_member(tmp_path: Path, module_body: str) -> None:
    fixture = _fixture(tmp_path)
    closure_root = fixture["closure_root"]
    (closure_root / "sibling.py").write_text(module_body, encoding="utf-8")
    (closure_root / "source.py").write_text("from sibling import missing\n", encoding="utf-8")
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(closure_root)

    with pytest.raises(P6ExecutionClosureError):
        validate_execution_closure_manifest(
            fixture["manifest"],
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )


def test_closure_accepts_initialized_annotated_plain_module_member(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    closure_root = fixture["closure_root"]
    (closure_root / "sibling.py").write_text("missing: int = 1\n", encoding="utf-8")
    (closure_root / "source.py").write_text("from sibling import missing\n", encoding="utf-8")
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(closure_root)

    validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )


def test_closure_accepts_parent_relative_import_inside_declared_root(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    closure_root = fixture["closure_root"]
    (closure_root / "source.py").unlink()
    source = _write_executable(
        closure_root / "package/tools/source.py",
        "#!/usr/bin/env python3\nfrom ..common import VALUE\nassert VALUE == 'ok'\n",
    )
    for package in (closure_root / "package", closure_root / "package/tools"):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (closure_root / "package/common.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    fixture["manifest"]["roles"]["source_materializer"]["entrypoint_relative_path"] = str(
        source.relative_to(closure_root)
    )
    fixture["manifest"]["roots"][0]["sha256"] = _tree_sha(closure_root)

    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )

    assert validated.roles[2].entrypoint_relative_path == Path("package/tools/source.py")


def test_closure_rejects_symlink_hardlink_and_special_file(tmp_path: Path) -> None:
    for kind in ("symlink", "hardlink", "fifo"):
        fixture = _fixture(tmp_path / kind)
        closure_root = fixture["closure_root"]
        if kind == "symlink":
            (closure_root / "link").symlink_to(closure_root / "sibling.py")
        elif kind == "hardlink":
            os.link(closure_root / "sibling.py", closure_root / "alias.py")
        else:
            os.mkfifo(closure_root / "fifo")
        fixture["manifest"]["roots"][0]["sha256"] = "0" * 64
        with pytest.raises(P6ExecutionClosureError):
            validate_execution_closure_manifest(
                fixture["manifest"],
                source_history_root=fixture["source_root"],
                external_training=fixture["external"],
            )


def test_closure_rejects_copied_tree_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )
    real_copy = closure_module._copy_tree

    def drifting_copy(source: Path, destination: Path) -> None:
        real_copy(source, destination)
        (destination / "sibling.py").write_text("VALUE = 'drift'\n", encoding="utf-8")

    monkeypatch.setattr(closure_module, "_copy_tree", drifting_copy)

    with pytest.raises(P6ExecutionClosureError):
        copy_execution_closure(validated, staged_private_root=fixture["staged"])
    assert not (fixture["staged"] / "execution-closure").exists()


def test_closure_translates_git_timeout_to_stable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(("git",), 10)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(P6ExecutionClosureError):
        validate_execution_closure_manifest(
            fixture["manifest"],
            source_history_root=fixture["source_root"],
            external_training=fixture["external"],
        )


def test_closure_renders_and_revalidates_normalized_runner(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    validated = validate_execution_closure_manifest(
        fixture["manifest"],
        source_history_root=fixture["source_root"],
        external_training=fixture["external"],
    )
    copied = copy_execution_closure(validated, staged_private_root=fixture["staged"])
    source_payload = _runner_template()
    source_payload["stage1_scan"]["argv"][0] = str(fixture["closure_root"] / "scan")
    interface = source_payload["execution_interface"]
    interface["controller"]["argv"][0] = str(
        fixture["closure_root"] / "stage5_task_round_controller_v3.sh"
    )
    source_stage_paths = {
        "source_materialization": fixture["source_marker"],
        "quantization": fixture["closure_root"] / "quantize",
        "performance": fixture["closure_root"] / "stage5_build_performance_plan_v2.py",
        "ap": fixture["closure_root"] / "ap",
        "finalization": fixture["closure_root"] / "stage5_finalize_feedback_v2.py",
    }
    for entry in interface["execution_chain"]:
        entry["argv"][0] = str(source_stage_paths[entry["stage"]])
    interface["environment"]["activation_argv"][0] = str(fixture["closure_root"] / "activate")
    template_path = tmp_path / "runner.yaml"
    template_path.write_text(yaml.safe_dump(source_payload), encoding="utf-8")
    source_template = validate_pre_provision_runner_template(template_path, fixture["source_root"])

    rendered = render_normalized_runner_template(
        source_template,
        normalized_private_root=fixture["staged"],
        copied_role_paths=copied,
    )
    normalized_template = fixture["staged"] / "runner-template.yaml"
    wrapper = fixture["staged"] / "documented-stage5-chain/stage5_materialize_round_sources_v1.sh"
    _write_executable(wrapper)
    normalized_template.write_text(yaml.safe_dump(rendered), encoding="utf-8")

    rebound = validate_normalized_runner_closure(
        normalized_template,
        normalized_private_root=fixture["staged"],
        expected_closure=validated,
    )
    assert rebound.history_root == fixture["staged"]
    expected_non_source = {path for role, path in copied.items() if role != "source_materializer"}
    rendered_paths = {
        fixture["staged"] / Path(rendered["stage1_scan"]["argv"][0]),
        fixture["staged"] / Path(rendered["execution_interface"]["controller"]["argv"][0]),
        *(
            fixture["staged"] / Path(entry["argv"][0])
            for entry in rendered["execution_interface"]["execution_chain"]
        ),
        fixture["staged"]
        / Path(rendered["execution_interface"]["environment"]["activation_argv"][0]),
    }
    assert expected_non_source.issubset(rendered_paths)
    assert (
        fixture["staged"] / Path(rendered["execution_interface"]["execution_chain"][0]["argv"][0])
        == wrapper
    )


def test_projected_training_contract_uses_nested_binding_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    group_id = "pyramid|16x32x64"
    contract = {
        "group_id": group_id,
        "width": [16, 32, 64],
        "stage_widths": {
            "stage1_width": 16,
            "stage2_width": 32,
            "stage3_width": 64,
        },
        "external_training_binding": {
            "schema_version": fixture["external"].schema_version,
            "training_required": True,
            "training_source_kind": "selected_candidate_finetune",
            "dataset_root": str(fixture["external"].dataset_root),
            "base_checkpoint_path": str(fixture["external"].base_checkpoint_path),
            "base_checkpoint_sha256": fixture["external"].base_checkpoint_sha256,
            "pyramid_config_path": str(fixture["external"].pyramid_config_path),
            "pyramid_config_sha256": fixture["external"].pyramid_config_sha256,
            "training_parameters": dict(fixture["external"].training_parameters),
        },
        **{key: str(tmp_path / "output" / key) for key in SHARED_SOURCE_PATH_KEYS},
    }

    assert validate_projected_training_contract(contract, group_id=group_id) == contract

    flat = copy.deepcopy(contract)
    nested = flat.pop("external_training_binding")
    flat.update(
        {
            key: nested[key]
            for key in (
                "training_required",
                "training_source_kind",
                "dataset_root",
                "base_checkpoint_path",
                "pyramid_config_path",
                "training_parameters",
            )
        }
    )
    with pytest.raises(P6HistoryTrainingContractError):
        validate_projected_training_contract(flat, group_id=group_id)
