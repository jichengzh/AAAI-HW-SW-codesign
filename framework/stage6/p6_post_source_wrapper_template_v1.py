"""Render private P6 post-source adapter wrappers."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import stat
import tempfile

from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    POST_SOURCE_ADAPTER_STAGES,
    ValidatedPostSourceAdapterProfile,
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
) -> Mapping[str, Path]:
    """Render four deterministic wrappers from one validated adapter profile."""
    root = _private_root(profile, private_root)
    wrappers: dict[str, Path] = {}
    for stage in POST_SOURCE_ADAPTER_STAGES:
        path = root / WRAPPER_RELATIVE_PATHS[stage]
        _write_executable(path, _wrapper_text(profile, stage))
        wrappers[stage] = path
    return wrappers


def _wrapper_text(profile: ValidatedPostSourceAdapterProfile, stage: str) -> str:
    adapter = next(item for item in profile.adapters if item.stage == stage)
    expected_adapter = {
        "implementation_relative_path": adapter.implementation.relative_to(
            profile.private_root
        ).as_posix(),
        "implementation_cwd_relative_path": adapter.implementation_cwd.relative_to(
            profile.private_root
        ).as_posix(),
    }
    expected_leaves = {
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
    return _SCRIPT_TEMPLATE.format(
        stage=stage,
        private_root=str(profile.private_root),
        project_python=str(profile.project_python),
        adapter_path=expected_adapter["implementation_relative_path"],
        adapter_cwd=expected_adapter["implementation_cwd_relative_path"],
        env_keys=tuple(EXPECTED_HISTORY_ENV_KEYS),
        expected_adapter=expected_adapter,
        expected_leaves=expected_leaves,
    )


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
