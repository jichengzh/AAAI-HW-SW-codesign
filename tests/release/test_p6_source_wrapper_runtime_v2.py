"""Execution contracts for the v2 historical source wrapper runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework.stage6.p6_source_wrapper_profile_v1 import (
    render_self_contained_source_wrapper,
)
from tests.release.test_p6_source_wrapper_legacy_bridge import (
    DIRECTORY_OUTPUT_KEYS,
    OUTPUT_PATH_KEYS,
    _canonical_request,
    _private_git_root,
    _profile,
    _run_wrapper,
    _runtime_env,
)


def _write_project_python(tmp_path: Path) -> Path:
    executable = tmp_path / "project-env/bin/python3.9"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n", encoding="utf-8")
    executable.chmod(0o700)
    (executable.parent / "python").symlink_to(executable.name)
    return executable


def _write_conda_project_python_and_jq(tmp_path: Path) -> tuple[Path, Path]:
    conda_root = tmp_path / "miniconda3"
    executable = conda_root / "envs/project/bin/python3.9"
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/sh\nexec /usr/bin/python3 "$@"\n', encoding="utf-8")
    executable.chmod(0o700)
    (executable.parent / "python").symlink_to(executable.name)
    jq = conda_root / "bin/jq"
    jq.parent.mkdir(parents=True)
    jq.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    jq.chmod(0o700)
    return executable, jq


def _v2_profile(project_python: Path) -> dict[str, str]:
    return {
        **_profile(),
        "schema_version": "p6_private_source_wrapper_profile_v2",
        "project_python": str(project_python),
    }


def _write_runtime_materializer(
    history_root: Path,
    project_python: Path,
    *,
    fail_after_runtime_validation: int | None = None,
    dirty_failure_outputs: bool = False,
    required_jq: Path | None = None,
) -> Path:
    failure_line = (
        (
            "marker.parent.mkdir(parents=True, exist_ok=True)\n"
            "marker.write_text('dirty-marker', encoding='utf-8')\n"
            "dirty_config = Path(contract['config_path'])\n"
            "dirty_config.parent.mkdir(parents=True, exist_ok=True)\n"
            "dirty_config.write_text('dirty-config', encoding='utf-8')\n"
            f"raise SystemExit({fail_after_runtime_validation})\n"
        )
        if fail_after_runtime_validation is not None
        else ""
    )
    if fail_after_runtime_validation is not None and not dirty_failure_outputs:
        failure_line = f"raise SystemExit({fail_after_runtime_validation})\n"
    implementation = (
        history_root
        / "private-relocated-history-repo/bin/"
        "stage5_materialize_round_sources_v1.original.sh"
    )
    implementation.parent.mkdir(parents=True)
    implementation.write_text(
        "#!/usr/bin/python3\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import shutil\n"
        "import subprocess\n"
        "import sys\n"
        f"output_keys = {OUTPUT_PATH_KEYS!r}\n"
        f"directory_keys = {tuple(sorted(DIRECTORY_OUTPUT_KEYS))!r}\n"
        f"project_python = {str(project_python)!r}\n"
        f"required_jq = {str(required_jq) if required_jq is not None else None!r}\n"
        "round_root = Path(os.environ['P6_HISTORY_ROUND_OUTPUT_ROOT'])\n"
        "diagnostic = round_root / 'runtime-diagnostic.txt'\n"
        "request = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))\n"
        "contract = request['rows'][0]['source_contract']\n"
        "marker = Path(contract['training_done_marker'])\n"
        "if not marker.parent.is_dir() or marker.exists():\n"
        "    diagnostic.write_text('marker-parent-missing', encoding='utf-8')\n"
        "    raise SystemExit(31)\n"
        "path_parts = [str(Path(project_python).parent)]\n"
        "if required_jq is not None:\n"
        "    path_parts.append(str(Path(required_jq).parent))\n"
        "path_parts.extend(['/usr/bin', '/bin'])\n"
        "expected_path = ':'.join(path_parts)\n"
        "if os.environ.get('PY') != project_python or os.environ.get('PATH') != expected_path:\n"
        "    diagnostic.write_text('project-python-env-missing', encoding='utf-8')\n"
        "    raise SystemExit(32)\n"
        "if Path(shutil.which('python') or '').resolve() != Path(project_python):\n"
        "    diagnostic.write_text('bare-python-mismatch', encoding='utf-8')\n"
        "    raise SystemExit(33)\n"
        "if required_jq is not None and Path(shutil.which('jq') or '').resolve() != Path(required_jq):\n"
        "    diagnostic.write_text('jq-mismatch', encoding='utf-8')\n"
        "    raise SystemExit(34)\n"
        "literal = round_root / 'literal-python.txt'\n"
        "subprocess.run(['python', '-c', \"from pathlib import Path; "
        "Path(__import__('sys').argv[1]).write_text('literal-python-ok')\", "
        "str(literal)], check=True)\n"
        f"{failure_line}"
        "for key in output_keys:\n"
        "    target = Path(contract[key])\n"
        "    if key in directory_keys:\n"
        "        target.mkdir(parents=True, exist_ok=True)\n"
        "    else:\n"
        "        target.parent.mkdir(parents=True, exist_ok=True)\n"
        "        target.write_bytes(b'fixture:' + key.encode() + b'\\n')\n"
        "diagnostic.write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )
    implementation.chmod(0o700)
    return implementation


def test_v2_wrapper_exposes_conda_base_jq_without_ambient_path(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    project_python, jq = _write_conda_project_python_and_jq(tmp_path)
    _write_runtime_materializer(history_root, project_python, required_jq=jq)
    wrapper = render_self_contained_source_wrapper(
        _v2_profile(project_python), history_root=history_root
    ).executable
    round_root = history_root / "private-runs/0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    request = round_root / "measurement-request.json"
    request.write_text(json.dumps(canonical), encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        request,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 0, completed.stderr
    assert (round_root / "runtime-diagnostic.txt").read_text() == "ok"


@pytest.mark.parametrize("precreate_marker_parent", (False, True))
def test_v2_wrapper_prepares_training_marker_parent_and_project_python(
    tmp_path: Path,
    precreate_marker_parent: bool,
) -> None:
    history_root = _private_git_root(tmp_path)
    project_python = _write_project_python(tmp_path)
    _write_runtime_materializer(history_root, project_python)
    wrapper = render_self_contained_source_wrapper(
        _v2_profile(project_python), history_root=history_root
    ).executable
    round_root = history_root / "private-runs/0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    contract = canonical["rows"][0]["source_contract"]
    marker = Path(contract["training_done_marker"])
    if precreate_marker_parent:
        marker.parent.mkdir(parents=True)
    assert not marker.exists()
    request = round_root / "measurement-request.json"
    original_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
    request.write_bytes(original_bytes)

    completed = _run_wrapper(
        wrapper,
        request,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 0, completed.stderr
    assert (round_root / "runtime-diagnostic.txt").read_text() == "ok"
    assert (round_root / "literal-python.txt").read_text() == "literal-python-ok"
    assert request.read_bytes() == original_bytes
    assert marker.is_file() and marker.stat().st_size > 0
    for key in OUTPUT_PATH_KEYS:
        output = Path(contract[key])
        assert output.is_dir() if key in DIRECTORY_OUTPUT_KEYS else output.is_file()
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


def test_v2_wrapper_child_failure_does_not_publish_outputs_or_temp_view(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    project_python = _write_project_python(tmp_path)
    _write_runtime_materializer(
        history_root,
        project_python,
        fail_after_runtime_validation=23,
        dirty_failure_outputs=True,
    )
    wrapper = render_self_contained_source_wrapper(
        _v2_profile(project_python), history_root=history_root
    ).executable
    round_root = history_root / "private-runs/0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path, config_at_checkpoint=True)
    contract = canonical["rows"][0]["source_contract"]
    marker = Path(contract["training_done_marker"])
    request = round_root / "measurement-request.json"
    original_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
    request.write_bytes(original_bytes)

    completed = _run_wrapper(
        wrapper,
        request,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 23
    assert marker.parent.is_dir() and not marker.exists()
    assert not Path(contract["config_path"]).exists()
    assert request.read_bytes() == original_bytes
    assert not tuple(round_root.glob(".p6-legacy-source-request-*.json"))


def test_v2_wrapper_rejects_project_python_alias_drift_before_child(
    tmp_path: Path,
) -> None:
    history_root = _private_git_root(tmp_path)
    project_python = _write_project_python(tmp_path)
    _write_runtime_materializer(history_root, project_python)
    wrapper = render_self_contained_source_wrapper(
        _v2_profile(project_python), history_root=history_root
    ).executable
    alias = project_python.parent / "python"
    alternate = project_python.parent / "alternate-python"
    alternate.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    alternate.chmod(0o700)
    alias.unlink()
    alias.symlink_to(alternate.name)
    round_root = history_root / "private-runs/0"
    round_root.mkdir(parents=True)
    canonical = _canonical_request(tmp_path)
    request = round_root / "measurement-request.json"
    request.write_text(json.dumps(canonical), encoding="utf-8")

    completed = _run_wrapper(
        wrapper,
        request,
        round_root,
        _runtime_env(history_root, round_root),
    )

    assert completed.returncode == 2
    assert completed.stderr == "history_execution_invalid\n"
    assert not (round_root / "runtime-diagnostic.txt").exists()
    contract = canonical["rows"][0]["source_contract"]
    assert not Path(contract["training_done_marker"]).exists()
    assert not Path(contract["config_path"]).exists()
