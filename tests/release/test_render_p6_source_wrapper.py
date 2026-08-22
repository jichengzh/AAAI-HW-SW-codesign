"""Black-box and unit coverage for the private P6 source-wrapper renderer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from framework.stage6 import p6_source_wrapper_profile_v1 as profile_module
from framework.stage6.p6_source_wrapper_profile_v1 import (
    P6SourceWrapperProfileError,
    expected_wrapper_bytes_from_profile,
    render_self_contained_source_wrapper,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RENDERER = REPOSITORY_ROOT / "tools/release/render_p6_source_wrapper.py"
SOURCE_MARKER = "stage5_materialize_round_sources_v1.sh"


def _private_git_root(tmp_path: Path) -> Path:
    root = tmp_path / "private-history"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def _profile() -> dict[str, str]:
    return {
        "schema_version": "p6_private_source_wrapper_profile_v1",
        "wrapper_kind": "repo_cwd_exec_v1",
        "destination_relative_path": f"documented-stage5-chain/{SOURCE_MARKER}",
        "implementation_relative_path": (
            "private-relocated-history-repo/bin/"
            "stage5_materialize_round_sources_v1.original.sh"
        ),
        "implementation_cwd_relative_path": "private-relocated-history-repo",
    }


def _write_profile(path: Path, payload: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_relocated_materializer(history_root: Path) -> Path:
    implementation = (
        history_root
        / "private-relocated-history-repo"
        / "bin"
        / "stage5_materialize_round_sources_v1.original.sh"
    )
    implementation.parent.mkdir(parents=True, exist_ok=True)
    implementation.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    implementation.chmod(0o700)
    return implementation


def test_renderer_writes_deterministic_self_contained_wrapper_from_private_profile(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    profile_path = _write_profile(
        tmp_path / "ignored-inputs" / "source-wrapper-profile.yaml",
        _profile(),
    )

    first = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    )
    first_bytes = first.executable.read_bytes()
    second = render_self_contained_source_wrapper(
        _profile(), history_root=history_root
    )

    assert first.executable == (
        history_root / "documented-stage5-chain" / SOURCE_MARKER
    )
    assert first.executable.is_file()
    assert first.executable.stat().st_mode & 0o111
    assert second.executable.read_bytes() == first_bytes
    assert expected_wrapper_bytes_from_profile(
        profile_path, history_root=history_root
    ) == first_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(
        second.executable.read_bytes()
    ).hexdigest()
    body = first_bytes.decode("utf-8")
    assert body == (
        "#!/bin/sh\n"
        "set -eu\n"
        ': "${P6_HISTORY_PRIVATE_ROOT:?}"\n'
        'root="$(cd "${P6_HISTORY_PRIVATE_ROOT}" && pwd -P)"\n'
        'implementation="${root}/private-relocated-history-repo/bin/'
        'stage5_materialize_round_sources_v1.original.sh"\n'
        'implementation_cwd="${root}/private-relocated-history-repo"\n'
        '[ -f "$implementation" ] && [ -x "$implementation" ]\n'
        'cd "$implementation_cwd"\n'
        'implementation_cwd="$(pwd -P)"\n'
        'exec "$implementation" "$@"\n'
    )
    assert "P6_HISTORY_PRIVATE_ROOT" in body
    assert 'exec "$implementation" "$@"' in body
    assert "/usr/bin/env python3" not in body
    assert "PYTHONPATH" not in body
    assert "module_name" not in body
    assert "callable_name" not in body
    assert str(history_root) not in body


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("absolute_destination", "/private/stage5_materialize_round_sources_v1.sh"),
        ("destination_parent_escape", "../stage5_materialize_round_sources_v1.sh"),
        ("destination_git_metadata", ".git/hooks/stage5_materialize_round_sources_v1.sh"),
        ("wrong_marker_basename", "documented-stage5-chain/wrong.sh"),
        ("absolute_implementation", "/private/original.sh"),
        ("implementation_parent_escape", "../original.sh"),
        ("implementation_git_metadata", ".git/hooks/original.sh"),
        ("implementation_shell_token", "private-repo/bin/source;unsafe.sh"),
        ("implementation_cwd_parent_escape", "../private-repo"),
        ("implementation_cwd_shell_token", "private-repo/$(unsafe)"),
    ],
)
def test_renderer_rejects_unsafe_private_wrapper_profile_paths(
    tmp_path: Path, mutation: str, value: str
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    profile = _profile()
    key = {
        "absolute_destination": "destination_relative_path",
        "destination_parent_escape": "destination_relative_path",
        "destination_git_metadata": "destination_relative_path",
        "wrong_marker_basename": "destination_relative_path",
        "absolute_implementation": "implementation_relative_path",
        "implementation_parent_escape": "implementation_relative_path",
        "implementation_git_metadata": "implementation_relative_path",
        "implementation_shell_token": "implementation_relative_path",
        "implementation_cwd_parent_escape": "implementation_cwd_relative_path",
        "implementation_cwd_shell_token": "implementation_cwd_relative_path",
    }[mutation]
    profile[key] = value

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        render_self_contained_source_wrapper(profile, history_root=history_root)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
    assert not (
        history_root / "documented-stage5-chain" / SOURCE_MARKER
    ).exists()


def test_renderer_rejects_implementation_at_exact_wrapper_destination(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    marker = history_root / "documented-stage5-chain" / SOURCE_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    marker.chmod(0o700)
    profile = _profile()
    profile["implementation_relative_path"] = profile["destination_relative_path"]
    profile["implementation_cwd_relative_path"] = "documented-stage5-chain"

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        render_self_contained_source_wrapper(profile, history_root=history_root)

    assert captured.value.category == "history_execution_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_profile_key",
        "missing_profile_key",
        "implementation_symlink",
        "implementation_not_executable",
        "cwd_symlink",
        "destination_symlink",
        "differing_existing_marker",
    ],
)
def test_renderer_rejects_unsafe_filesystem_or_profile_shape(
    tmp_path: Path, mutation: str
) -> None:
    history_root = _private_git_root(tmp_path)
    implementation = _write_relocated_materializer(history_root)
    profile: dict[str, Any] = _profile()
    marker = history_root / "documented-stage5-chain" / SOURCE_MARKER
    if mutation == "extra_profile_key":
        profile["module_name"] = "unsafe"
    elif mutation == "missing_profile_key":
        profile.pop("wrapper_kind")
    elif mutation == "implementation_symlink":
        real = implementation.with_name("real-original.sh")
        implementation.rename(real)
        implementation.symlink_to(real)
    elif mutation == "implementation_not_executable":
        implementation.chmod(0o600)
    elif mutation == "cwd_symlink":
        repository = history_root / "private-relocated-history-repo"
        real_repository = history_root / "real-relocated-history-repo"
        repository.rename(real_repository)
        repository.symlink_to(real_repository, target_is_directory=True)
    elif mutation == "destination_symlink":
        marker.parent.mkdir(parents=True)
        marker.symlink_to(implementation)
    else:
        marker.parent.mkdir(parents=True)
        marker.write_text("#!/bin/sh\nexec /private/raw/source \"$@\"\n", encoding="utf-8")
        marker.chmod(0o700)

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        render_self_contained_source_wrapper(profile, history_root=history_root)

    assert captured.value.category == "history_execution_invalid"
    assert str(captured.value) == "history_execution_invalid"
    if mutation == "differing_existing_marker":
        assert marker.read_text(encoding="utf-8") == (
            "#!/bin/sh\nexec /private/raw/source \"$@\"\n"
        )


def test_renderer_cli_writes_wrapper_and_redacts_invalid_profile(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    profile_path = _write_profile(
        tmp_path / "ignored-inputs" / "source-wrapper-profile.yaml",
        _profile(),
    )

    accepted = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--history-root",
            str(history_root),
            "--source-wrapper-profile",
            str(profile_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0
    assert accepted.stdout == "p6_source_wrapper_written\n"
    assert accepted.stderr == ""

    profile = _profile()
    profile["implementation_relative_path"] = "../private-secret"
    _write_profile(profile_path, profile)
    rejected = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--history-root",
            str(history_root),
            "--source-wrapper-profile",
            str(profile_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 1
    assert rejected.stdout == ""
    assert rejected.stderr == "history_execution_invalid\n"
    assert str(history_root) not in rejected.stderr


def test_profile_loader_rejects_trackable_public_repository_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_root = _private_git_root(tmp_path)
    _write_relocated_materializer(history_root)
    repository = tmp_path / "public-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    profile_path = _write_profile(repository / "local" / "profile.yaml", _profile())
    monkeypatch.setattr(profile_module, "REPOSITORY_ROOT", repository)

    with pytest.raises(P6SourceWrapperProfileError) as captured:
        expected_wrapper_bytes_from_profile(
            profile_path,
            history_root=history_root,
        )

    assert captured.value.category == "history_execution_invalid"
    (repository / ".gitignore").write_text("local/\n", encoding="utf-8")
    assert expected_wrapper_bytes_from_profile(
        profile_path,
        history_root=history_root,
    ).startswith(b"#!/bin/sh\n")
