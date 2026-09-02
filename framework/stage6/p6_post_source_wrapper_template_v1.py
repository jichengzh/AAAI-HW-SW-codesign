"""Render private P6 post-source adapter wrappers."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import stat
import tempfile

from framework.stage6.p6_history_binding_v1 import (
    EXPECTED_HISTORY_ENV_KEYS,
    FORMAL_TVM_ENV_KEYS,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PROFILE_SCHEMA_VERSION_V2,
    PROFILE_SCHEMA_VERSION_V3,
    PROFILE_SCHEMA_VERSION_V4,
    PROFILE_V3_KEYS,
    PROFILE_V4_KEYS,
    POST_SOURCE_ADAPTER_STAGES,
    ValidatedPostSourceAdapterProfile,
)
from framework.stage6.p6_python_runtime_v1 import (
    P6PythonRuntimeError,
    validate_adapter_python,
)


WRAPPER_RELATIVE_PATHS: Mapping[str, Path] = {
    "quantization": Path("private-runner/bin/quantize-private"),
    "performance": Path("documented-stage5-chain/stage5_build_performance_plan_v2.py"),
    "ap": Path("private-runner/bin/measure-ap-private"),
    "finalization": Path("documented-stage5-chain/stage5_finalize_feedback_v2.py"),
}


class P6PostSourceWrapperError(ValueError):
    """Stable path-free wrapper rendering failure."""

    def __init__(self) -> None:
        super().__init__("history_execution_invalid")


def render_post_source_adapter_wrappers(
    profile: ValidatedPostSourceAdapterProfile,
    *,
    private_root: Path,
    declared_private_root: Path | None = None,
) -> Mapping[str, Path]:
    """Render four deterministic wrappers from one validated adapter profile."""
    root = _private_root(profile, private_root)
    declared_root = _declared_private_root(profile, declared_private_root)
    wrappers: dict[str, Path] = {}
    for stage in POST_SOURCE_ADAPTER_STAGES:
        path = root / WRAPPER_RELATIVE_PATHS[stage]
        _write_executable(
            path,
            _wrapper_text(profile, stage, declared_private_root=declared_root),
        )
        wrappers[stage] = path
    return wrappers


def expected_post_source_wrapper_bytes_from_profile(
    profile: ValidatedPostSourceAdapterProfile,
) -> Mapping[str, bytes]:
    """Return exact final wrapper bytes for one loaded private profile."""
    _private_root(profile, profile.private_root)
    return {
        stage: _wrapper_text(profile, stage).encode("utf-8")
        for stage in POST_SOURCE_ADAPTER_STAGES
    }


def validate_post_source_adapter_wrappers(
    profile: ValidatedPostSourceAdapterProfile,
    *,
    private_root: Path,
) -> Mapping[str, Path]:
    """Prove existing generated wrapper bytes match the profile/template."""
    root = _private_root(profile, private_root)
    expected_bytes = expected_post_source_wrapper_bytes_from_profile(profile)
    wrappers: dict[str, Path] = {}
    for stage in POST_SOURCE_ADAPTER_STAGES:
        path = root / WRAPPER_RELATIVE_PATHS[stage]
        if (
            not _single_link_executable(path)
            or path.read_bytes() != expected_bytes[stage]
        ):
            raise P6PostSourceWrapperError()
        wrappers[stage] = path
    return wrappers


def _wrapper_text(
    profile: ValidatedPostSourceAdapterProfile,
    stage: str,
    *,
    declared_private_root: Path | None = None,
) -> str:
    adapter = next(item for item in profile.adapters if item.stage == stage)
    expected_adapter = {
        "implementation_relative_path": adapter.implementation.relative_to(
            profile.private_root
        ).as_posix(),
        "implementation_cwd_relative_path": adapter.implementation_cwd.relative_to(
            profile.private_root
        ).as_posix(),
    }
    expected_leaves = _expected_leaves(profile)
    environment_keys = (
        (*EXPECTED_HISTORY_ENV_KEYS, *FORMAL_TVM_ENV_KEYS)
        if profile.schema_version == PROFILE_SCHEMA_VERSION_V4
        else EXPECTED_HISTORY_ENV_KEYS
    )
    rendered = _SCRIPT_TEMPLATE.format(
        stage=stage,
        private_root=str(declared_private_root or profile.private_root),
        project_python=str(profile.project_python),
        adapter_path=expected_adapter["implementation_relative_path"],
        adapter_cwd=expected_adapter["implementation_cwd_relative_path"],
        adapter_sha256=_sha256(adapter.implementation),
        env_keys=tuple(environment_keys),
        expected_adapter=expected_adapter,
        expected_leaves=expected_leaves,
    )
    if profile.schema_version not in {
        PROFILE_SCHEMA_VERSION_V2,
        PROFILE_SCHEMA_VERSION_V3,
        PROFILE_SCHEMA_VERSION_V4,
    }:
        return rendered
    try:
        adapter_python = validate_adapter_python(profile.adapter_python)
    except P6PythonRuntimeError as error:
        raise P6PostSourceWrapperError() from error
    runtime_rendered = _v2_wrapper_text(
        rendered,
        adapter_python=adapter_python,
        project_python=profile.project_python,
        schema_version=profile.schema_version,
    )
    if profile.schema_version == PROFILE_SCHEMA_VERSION_V2:
        return runtime_rendered
    dependency_root = _dependency_root_relative_path(profile)
    profile_keys = (
        PROFILE_V4_KEYS
        if profile.schema_version == PROFILE_SCHEMA_VERSION_V4
        else PROFILE_V3_KEYS
    )
    return _v3_wrapper_text(
        runtime_rendered,
        dependency_root=dependency_root,
        schema_version=profile.schema_version,
        profile_keys=profile_keys,
    )


def _dependency_root_relative_path(
    profile: ValidatedPostSourceAdapterProfile,
) -> Path:
    root = profile.adapter_dependency_root
    if not isinstance(root, Path):
        raise P6PostSourceWrapperError()
    try:
        return root.relative_to(profile.private_root)
    except ValueError as error:
        raise P6PostSourceWrapperError() from error


def _expected_leaves(
    profile: ValidatedPostSourceAdapterProfile,
) -> dict[str, dict[str, str]]:
    return {
        leaf.name: {
            "implementation_relative_path": leaf.implementation.relative_to(
                profile.private_root
            ).as_posix(),
            "implementation_cwd_relative_path": leaf.implementation_cwd.relative_to(
                profile.private_root
            ).as_posix(),
            "sha256": leaf.sha256,
        }
        for leaf in profile.leaves
    }


def _v2_wrapper_text(
    rendered: str,
    *,
    adapter_python: Path,
    project_python: Path,
    schema_version: str,
) -> str:
    rendered = _replace_once(
        rendered,
        f"PROJECT_PYTHON = Path({str(project_python)!r})\n",
        (
            f"PROJECT_PYTHON = Path({str(project_python)!r})\n"
            f"ADAPTER_PYTHON = Path({str(adapter_python)!r})\n"
            f"PROFILE_SCHEMA_VERSION = {schema_version!r}\n"
        ),
    )
    rendered = _replace_once(
        rendered,
        "def _profile_payload() -> dict:\n",
        _V2_ADAPTER_PYTHON_RUNTIME + "def _profile_payload() -> dict:\n",
    )
    rendered = _v2_wrapper_profile_checks(rendered)
    rendered = _replace_once(
        rendered,
        "values['PATH'] = os.pathsep.join((str(PROJECT_PYTHON.parent), '/usr/bin', '/bin'))",
        "values['PATH'] = os.pathsep.join((str(ADAPTER_PYTHON.parent), '/usr/bin', '/bin'))",
    )
    rendered = _replace_once(
        rendered,
        "        if not _valid_profile(_profile_payload()):\n",
        (
            "        if not _valid_adapter_python_runtime():\n"
            "            return 1\n"
            "        if not _valid_profile(_profile_payload()):\n"
        ),
    )
    return _replace_once(
        rendered,
        "            str(PROJECT_PYTHON),\n            str(ADAPTER_IMPLEMENTATION),\n",
        "            str(ADAPTER_PYTHON),\n            str(ADAPTER_IMPLEMENTATION),\n",
    )


def _v3_wrapper_text(
    rendered: str,
    *,
    dependency_root: Path,
    schema_version: str,
    profile_keys: frozenset[str],
) -> str:
    rendered = _replace_once(
        rendered,
        f"PROFILE_SCHEMA_VERSION = {schema_version!r}\n",
        (
            f"PROFILE_SCHEMA_VERSION = {schema_version!r}\n"
            f"ADAPTER_DEPENDENCY_ROOT = PRIVATE_ROOT / {dependency_root.as_posix()!r}\n"
            f"ADAPTER_DEPENDENCY_ROOT_RELATIVE_PATH = {dependency_root.as_posix()!r}\n"
            f"EXPECTED_PROFILE_KEYS = {tuple(sorted(profile_keys))!r}\n"
        ),
    )
    rendered = _replace_once(
        rendered,
        "def _profile_payload() -> dict:\n",
        _V3_ADAPTER_DEPENDENCY_RUNTIME + "def _profile_payload() -> dict:\n",
    )
    rendered = _v3_wrapper_profile_checks(rendered)
    rendered = _replace_once(
        rendered,
        "values['PYTHONPATH'] = os.pathsep.join((str(ADAPTER_CWD), str(PRIVATE_ROOT)))",
        (
            "values['PYTHONPATH'] = os.pathsep.join((str(ADAPTER_CWD), "
            "str(ADAPTER_DEPENDENCY_ROOT), str(PRIVATE_ROOT)))"
        ),
    )
    return _replace_once(
        rendered,
        "        if not _valid_adapter_python_runtime():\n",
        (
            "        if not _valid_adapter_dependency_root():\n"
            "            return 1\n"
            "        if not _valid_adapter_python_runtime():\n"
        ),
    )


def _v3_wrapper_profile_checks(rendered: str) -> str:
    rendered = _replace_once(
        rendered,
        "    if payload.get('schema_version') != PROFILE_SCHEMA_VERSION:\n",
        (
            "    if set(payload) != set(EXPECTED_PROFILE_KEYS):\n"
            "        return False\n"
            "    if payload.get('schema_version') != PROFILE_SCHEMA_VERSION:\n"
        ),
    )
    return _replace_once(
        rendered,
        "    if _sha256(ADAPTER_IMPLEMENTATION) != ADAPTER_SHA256:\n",
        (
            "    if payload.get('adapter_dependency_root_relative_path') != "
            "ADAPTER_DEPENDENCY_ROOT_RELATIVE_PATH:\n"
            "        return False\n"
            "    if _sha256(ADAPTER_IMPLEMENTATION) != ADAPTER_SHA256:\n"
        ),
    )


def _v2_wrapper_profile_checks(rendered: str) -> str:
    rendered = _replace_once(
        rendered,
        "    if payload.get('project_python') != str(PROJECT_PYTHON):\n",
        (
            "    if payload.get('schema_version') != PROFILE_SCHEMA_VERSION:\n"
            "        return False\n"
            "    if payload.get('project_python') != str(PROJECT_PYTHON):\n"
        ),
    )
    rendered = _replace_once(
        rendered,
        "        return False\n    if _sha256(ADAPTER_IMPLEMENTATION) != ADAPTER_SHA256:\n",
        (
            "        return False\n"
            "    if payload.get('adapter_python') != str(ADAPTER_PYTHON):\n"
            "        return False\n"
            "    if _sha256(ADAPTER_IMPLEMENTATION) != ADAPTER_SHA256:\n"
        ),
    )
    return rendered


def _replace_once(rendered: str, old: str, new: str) -> str:
    if rendered.count(old) != 1:
        raise P6PostSourceWrapperError()
    return rendered.replace(old, new)


def _private_root(
    profile: ValidatedPostSourceAdapterProfile,
    private_root: Path,
) -> Path:
    if not isinstance(profile, ValidatedPostSourceAdapterProfile):
        raise P6PostSourceWrapperError()
    try:
        root = private_root.resolve(strict=True)
    except OSError as error:
        raise P6PostSourceWrapperError() from error
    if root != profile.private_root:
        raise P6PostSourceWrapperError()
    return root


def _declared_private_root(
    profile: ValidatedPostSourceAdapterProfile,
    raw: Path | None,
) -> Path:
    if raw is None:
        return profile.private_root
    if not isinstance(raw, Path) or not raw.is_absolute():
        raise P6PostSourceWrapperError()
    try:
        declared = raw.resolve(strict=False)
    except OSError as error:
        raise P6PostSourceWrapperError() from error
    if declared != raw or declared.exists():
        raise P6PostSourceWrapperError()
    return declared


def _sha256(path: Path) -> str:
    try:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise P6PostSourceWrapperError() from error


def _single_link_executable(path: Path) -> bool:
    try:
        if _contains_symlink_component(path):
            return False
        info = path.lstat()
        return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and os.access(path, os.X_OK)
    except OSError:
        return False


def _contains_symlink_component(path: Path) -> bool:
    anchor = Path(path.anchor)
    return any(
        component != anchor and component.is_symlink()
        for component in (path, *path.parents)
    )


def _write_executable(path: Path, text: str) -> None:
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
        temporary_path.chmod(0o700)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    path.chmod(stat.S_IRWXU)


_V2_ADAPTER_PYTHON_RUNTIME = '''def _valid_adapter_python_runtime() -> bool:
    import stat

    try:
        anchor = Path(ADAPTER_PYTHON.anchor)
        components = (ADAPTER_PYTHON, *ADAPTER_PYTHON.parents)
        if any(item != anchor and item.is_symlink() for item in components):
            return False
        info = ADAPTER_PYTHON.lstat()
        if (
            ADAPTER_PYTHON.resolve(strict=True) != ADAPTER_PYTHON
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or not os.access(ADAPTER_PYTHON, os.X_OK)
        ):
            return False
        completed = subprocess.run(
            [
                str(ADAPTER_PYTHON),
                '-I',
                '-S',
                '-c',
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={},
        )
        parts = completed.stdout.rstrip('\\n').split('.')
        version = tuple(int(part) for part in parts)
        return (
            completed.returncode == 0
            and len(version) == 2
            and version >= (3, 10)
        )
    except Exception:
        return False


'''


_V3_ADAPTER_DEPENDENCY_RUNTIME = '''def _valid_adapter_dependency_root() -> bool:
    try:
        anchor = Path(ADAPTER_DEPENDENCY_ROOT.anchor)
        components = (ADAPTER_DEPENDENCY_ROOT, *ADAPTER_DEPENDENCY_ROOT.parents)
        return (
            not any(item != anchor and item.is_symlink() for item in components)
            and ADAPTER_DEPENDENCY_ROOT.resolve(strict=True) == ADAPTER_DEPENDENCY_ROOT
            and ADAPTER_DEPENDENCY_ROOT.is_dir()
        )
    except Exception:
        return False


'''


_SCRIPT_TEMPLATE = """#!/usr/bin/python3
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import yaml

STAGE = {stage!r}
PRIVATE_ROOT = Path({private_root!r})
PROFILE_PATH = PRIVATE_ROOT / 'post-source-adapter-profile.yaml'
PROJECT_PYTHON = Path({project_python!r})
ADAPTER_IMPLEMENTATION = PRIVATE_ROOT / {adapter_path!r}
ADAPTER_CWD = PRIVATE_ROOT / {adapter_cwd!r}
ADAPTER_SHA256 = {adapter_sha256!r}
EXPECTED_ENV_KEYS = {env_keys!r}
EXPECTED_ADAPTER = {expected_adapter!r}
EXPECTED_LEAVES = {expected_leaves!r}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_payload() -> dict:
    payload = yaml.safe_load(PROFILE_PATH.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError
    return payload


def _valid_profile(payload: dict) -> bool:
    if payload.get('project_python') != str(PROJECT_PYTHON):
        return False
    if _sha256(ADAPTER_IMPLEMENTATION) != ADAPTER_SHA256:
        return False
    adapters = payload.get('adapters')
    leaves = payload.get('leaves')
    if not isinstance(adapters, dict) or adapters.get(STAGE) != EXPECTED_ADAPTER:
        return False
    if not isinstance(leaves, dict) or set(leaves) != set(EXPECTED_LEAVES):
        return False
    return all(_valid_leaf(name, leaves[name]) for name in EXPECTED_LEAVES)


def _valid_leaf(name: str, actual: object) -> bool:
    expected = EXPECTED_LEAVES[name]
    if actual != expected:
        return False
    path = PRIVATE_ROOT / expected['implementation_relative_path']
    return _sha256(path) == expected['sha256']


def _child_env() -> dict[str, str]:
    incoming = {{key for key in os.environ if key != 'LC_CTYPE'}}
    if incoming != set(EXPECTED_ENV_KEYS):
        raise ValueError
    values = {{key: os.environ[key] for key in EXPECTED_ENV_KEYS}}
    values['PATH'] = os.pathsep.join((str(PROJECT_PYTHON.parent), '/usr/bin', '/bin'))
    values['PYTHONPATH'] = os.pathsep.join((str(ADAPTER_CWD), str(PRIVATE_ROOT)))
    return values


def main() -> int:
    try:
        if not _valid_profile(_profile_payload()):
            return 1
        child_argv = (
            str(PROJECT_PYTHON),
            str(ADAPTER_IMPLEMENTATION),
            '--profile',
            str(PROFILE_PATH),
            *sys.argv[1:],
        )
        completed = subprocess.run(
            child_argv,
            cwd=str(ADAPTER_CWD),
            env=_child_env(),
            shell=False,
            check=False,
        )
        return int(completed.returncode)
    except Exception:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
"""
