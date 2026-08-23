"""Normalizer contracts for the explicit project-Python runtime binding."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from framework.stage6.p6_history_normalization_v1 import (
    P6HistoryNormalizationError,
    normalize_history_inputs,
)
from tests.stage6.test_p6_history_normalization import (
    _as_v2_procedural,
    _history_root,
    _tree_sha,
    valid_private_source_map,
)


def _write_project_python(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "project-env/bin/python3.9"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n", encoding="utf-8")
    executable.chmod(0o700)
    alias = executable.parent / "python"
    alias.symlink_to(executable.name)
    return executable, alias


def _source_with_py_assignment(
    tmp_path: Path,
    assignment_lines: tuple[str, ...],
) -> tuple[dict[str, object], Path, Path]:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    manifest = source_map["execution_code_closure"]
    root = manifest["roots"][0]
    source_root = Path(root["source_root"])
    role = manifest["roles"]["source_materializer"]
    source = source_root / role["entrypoint_relative_path"]
    source.write_text(
        "#!/bin/sh\nset -eu\n" + "\n".join(assignment_lines) + "\nexit 0\n",
        encoding="utf-8",
    )
    source.chmod(0o700)
    root["sha256"] = _tree_sha(source_root)
    return source_map, runner, source


def test_normalizer_extracts_exact_project_python_into_v2_profile(
    tmp_path: Path,
) -> None:
    project_python, alias = _write_project_python(tmp_path)
    source_map, runner, _ = _source_with_py_assignment(
        tmp_path,
        (f"PY=${{PY:-{alias}}}",),
    )
    destination = tmp_path / "private-normalized"

    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        destination,
        runner_template_path=runner,
    )

    profile = yaml.safe_load(paths["source_wrapper_profile"].read_text(encoding="utf-8"))
    assert profile["schema_version"] == "p6_private_source_wrapper_profile_v2"
    assert profile["project_python"] == str(project_python)
    assert not Path(profile["project_python"]).is_symlink()


@pytest.mark.parametrize(
    "assignment_lines",
    (
        (),
        ("PY=${PY:-/opt/project/bin/python}", "PY=${PY:-/opt/other/bin/python}"),
        ("PY=${PY:-relative/bin/python}",),
        ("PY=${PY:-${PROJECT_ROOT}/bin/python}",),
        ("PY=$(command -v python)",),
    ),
)
def test_normalizer_rejects_nonexact_project_python_assignment_before_publish(
    tmp_path: Path,
    assignment_lines: tuple[str, ...],
) -> None:
    source_map, runner, _ = _source_with_py_assignment(tmp_path, assignment_lines)
    destination = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError) as captured:
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            destination,
            runner_template_path=runner,
        )

    assert captured.value.category == "history_normalization_invalid"
    assert not destination.exists()
