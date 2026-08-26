from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from framework.stage6.p6_history_binding_v1 import EXPECTED_HISTORY_ENV_KEYS
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PROFILE_SCHEMA_VERSION,
    POST_SOURCE_ADAPTER_STAGES,
    POST_SOURCE_LEAF_NAMES,
    PostSourceAdapter,
    PostSourceLeaf,
    ValidatedPostSourceAdapterProfile,
    post_source_adapter_profile_to_mapping,
)
from framework.stage6.p6_post_source_wrapper_template_v1 import (
    P6PostSourceWrapperError,
    render_post_source_adapter_wrappers,
    validate_post_source_adapter_wrappers,
)


WRAPPER_RELATIVE_PATHS = {
    "quantization": "private-runner/bin/quantize-private",
    "performance": "documented-stage5-chain/stage5_build_performance_plan_v2.py",
    "ap": "private-runner/bin/measure-ap-private",
    "finalization": "documented-stage5-chain/stage5_finalize_feedback_v2.py",
}


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o700)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_adapter_body(stage: str) -> str:
    return f"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

profile_flag = sys.argv.index("--profile")
original_argv = sys.argv[profile_flag + 2:]
if "force-nonzero" in original_argv:
    raise SystemExit(37)
report_path = Path(original_argv[-1])
report_path.write_text(json.dumps({{
    "argv": original_argv,
    "incoming_environment_keys": sorted(
        key for key in os.environ if key not in {{"PATH", "PYTHONPATH", "LC_CTYPE"}}
    ),
    "project_python": sys.executable,
    "profile": sys.argv[profile_flag + 1],
    "cwd": str(Path.cwd()),
    "stage": {stage!r},
}}, sort_keys=True), encoding="utf-8")
"""


def _profile_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, ValidatedPostSourceAdapterProfile]:
    private_root = tmp_path / "private-root"
    private_root.mkdir()
    subprocess.run(["git", "init", "-q", str(private_root)], check=True)
    cwd = private_root / "execution-closure" / "history"
    project_python = Path("/usr/bin/python3.10")
    adapters = tuple(
        PostSourceAdapter(
            stage,
            _write_executable(cwd / f"{stage}.py", _fake_adapter_body(stage)),
            cwd,
        )
        for stage in POST_SOURCE_ADAPTER_STAGES
    )
    leaves = tuple(
        PostSourceLeaf(
            name,
            _write_executable(cwd / "leaves" / f"{name}.py", f"#!/usr/bin/env python3\n# {name}\n"),
            cwd,
            _sha256(cwd / "leaves" / f"{name}.py"),
        )
        for name in POST_SOURCE_LEAF_NAMES
    )
    profile = ValidatedPostSourceAdapterProfile(
        PROFILE_SCHEMA_VERSION,
        private_root,
        project_python,
        adapters,
        leaves,
    )
    profile_path = private_root / "post-source-adapter-profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(post_source_adapter_profile_to_mapping(profile), sort_keys=False),
        encoding="utf-8",
    )
    return private_root, profile_path, profile


def _history_env(private_root: Path) -> dict[str, str]:
    values = {
        "CUDA_VISIBLE_DEVICES": "23,19,17",
        "P6_HISTORY_RUN_MODE": "bound",
        "P6_HISTORY_PRIVATE_ROOT": str(private_root),
        "P6_HISTORY_TASK_STATE": "task-state.json",
        "P6_HISTORY_ROUND_OUTPUT_ROOT": "round-root",
    }
    assert set(values) == set(EXPECTED_HISTORY_ENV_KEYS)
    return values


def _expected_wrapper_text(
    profile: ValidatedPostSourceAdapterProfile,
    stage: str,
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
    return (
        "#!/usr/bin/python3\n"
        "from __future__ import annotations\n\n"
        "import hashlib\n"
        "import os\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n\n"
        "import yaml\n\n"
        f"STAGE = {stage!r}\n"
        f"PRIVATE_ROOT = Path({str(profile.private_root)!r})\n"
        "PROFILE_PATH = PRIVATE_ROOT / 'post-source-adapter-profile.yaml'\n"
        f"PROJECT_PYTHON = Path({str(profile.project_python)!r})\n"
        f"ADAPTER_IMPLEMENTATION = PRIVATE_ROOT / {expected_adapter['implementation_relative_path']!r}\n"
        f"ADAPTER_CWD = PRIVATE_ROOT / {expected_adapter['implementation_cwd_relative_path']!r}\n"
        f"ADAPTER_SHA256 = {_sha256(adapter.implementation)!r}\n"
        f"EXPECTED_ENV_KEYS = {tuple(EXPECTED_HISTORY_ENV_KEYS)!r}\n"
        f"EXPECTED_ADAPTER = {expected_adapter!r}\n"
        f"EXPECTED_LEAVES = {expected_leaves!r}\n\n"
        "\n"
        "def _sha256(path: Path) -> str:\n"
        "    return hashlib.sha256(path.read_bytes()).hexdigest()\n\n\n"
        "def _profile_payload() -> dict:\n"
        "    payload = yaml.safe_load(PROFILE_PATH.read_text(encoding='utf-8'))\n"
        "    if not isinstance(payload, dict):\n"
        "        raise ValueError\n"
        "    return payload\n\n\n"
        "def _valid_profile(payload: dict) -> bool:\n"
        "    if payload.get('project_python') != str(PROJECT_PYTHON):\n"
        "        return False\n"
        "    if _sha256(ADAPTER_IMPLEMENTATION) != ADAPTER_SHA256:\n"
        "        return False\n"
        "    adapters = payload.get('adapters')\n"
        "    leaves = payload.get('leaves')\n"
        "    if not isinstance(adapters, dict) or adapters.get(STAGE) != EXPECTED_ADAPTER:\n"
        "        return False\n"
        "    if not isinstance(leaves, dict) or set(leaves) != set(EXPECTED_LEAVES):\n"
        "        return False\n"
        "    return all(_valid_leaf(name, leaves[name]) for name in EXPECTED_LEAVES)\n\n\n"
        "def _valid_leaf(name: str, actual: object) -> bool:\n"
        "    expected = EXPECTED_LEAVES[name]\n"
        "    if actual != expected:\n"
        "        return False\n"
        "    path = PRIVATE_ROOT / expected['implementation_relative_path']\n"
        "    return _sha256(path) == expected['sha256']\n\n\n"
        "def _child_env() -> dict[str, str]:\n"
        "    incoming = {key for key in os.environ if key != 'LC_CTYPE'}\n"
        "    if incoming != set(EXPECTED_ENV_KEYS):\n"
        "        raise ValueError\n"
        "    values = {key: os.environ[key] for key in EXPECTED_ENV_KEYS}\n"
        "    values['PATH'] = os.pathsep.join((str(PROJECT_PYTHON.parent), '/usr/bin', '/bin'))\n"
        "    values['PYTHONPATH'] = os.pathsep.join((str(ADAPTER_CWD), str(PRIVATE_ROOT)))\n"
        "    return values\n\n\n"
        "def main() -> int:\n"
        "    try:\n"
        "        if not _valid_profile(_profile_payload()):\n"
        "            return 1\n"
        "        child_argv = (\n"
        "            str(PROJECT_PYTHON),\n"
        "            str(ADAPTER_IMPLEMENTATION),\n"
        "            '--profile',\n"
        "            str(PROFILE_PATH),\n"
        "            *sys.argv[1:],\n"
        "        )\n"
        "        completed = subprocess.run(\n"
        "            child_argv,\n"
        "            cwd=str(ADAPTER_CWD),\n"
        "            env=_child_env(),\n"
        "            shell=False,\n"
        "            check=False,\n"
        "        )\n"
        "        return int(completed.returncode)\n"
        "    except Exception:\n"
        "        return 1\n\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )


def _run_wrapper(
    wrapper: Path,
    private_root: Path,
    original_argv: list[str],
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(wrapper), *original_argv],
        cwd=private_root,
        env={**_history_env(private_root), **(extra_env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def test_generated_wrappers_execute_profile_adapters_with_unchanged_stage_argv(
    tmp_path: Path,
) -> None:
    private_root, profile_path, profile = _profile_fixture(tmp_path)

    wrappers = render_post_source_adapter_wrappers(profile, private_root=private_root)

    assert wrappers == {
        stage: private_root / relative
        for stage, relative in WRAPPER_RELATIVE_PATHS.items()
    }
    assert wrappers["performance"].name == "stage5_build_performance_plan_v2.py"
    assert wrappers["finalization"].name == "stage5_finalize_feedback_v2.py"
    for stage, wrapper in wrappers.items():
        assert wrapper.read_text(encoding="utf-8") == _expected_wrapper_text(
            profile, stage
        )
        report_path = tmp_path / f"{stage}-record.json"
        original_argv = [f"--stage={stage}", "literal-token", str(report_path)]

        completed = _run_wrapper(wrapper, private_root, original_argv)

        assert completed.returncode == 0, completed.stderr
        assert json.loads(report_path.read_text(encoding="utf-8")) == {
            "argv": original_argv,
            "incoming_environment_keys": sorted(EXPECTED_HISTORY_ENV_KEYS),
            "project_python": str(profile.project_python),
            "profile": str(profile_path),
            "cwd": str(next(item.implementation_cwd for item in profile.adapters if item.stage == stage)),
            "stage": stage,
        }


def test_wrapper_rejects_extra_incoming_environment_key(tmp_path: Path) -> None:
    private_root, _, profile = _profile_fixture(tmp_path)
    wrapper = render_post_source_adapter_wrappers(profile, private_root=private_root)[
        "quantization"
    ]
    report_path = tmp_path / "extra-env-record.json"

    completed = _run_wrapper(
        wrapper,
        private_root,
        ["--stage=quantization", str(report_path)],
        extra_env={"UNEXPECTED": "private"},
    )

    assert completed.returncode == 1
    assert not report_path.exists()


def test_existing_wrapper_validator_accepts_expected_generated_bytes(
    tmp_path: Path,
) -> None:
    private_root, _, profile = _profile_fixture(tmp_path)
    wrappers = render_post_source_adapter_wrappers(profile, private_root=private_root)

    assert validate_post_source_adapter_wrappers(
        profile,
        private_root=private_root,
    ) == wrappers


def test_existing_wrapper_validator_rejects_tampered_or_missing_wrapper_bytes(
    tmp_path: Path,
) -> None:
    private_root, _, profile = _profile_fixture(tmp_path)
    wrappers = render_post_source_adapter_wrappers(profile, private_root=private_root)
    wrappers["quantization"].write_text("#!/usr/bin/python3\nraise SystemExit(0)\n", encoding="utf-8")

    with pytest.raises(P6PostSourceWrapperError):
        validate_post_source_adapter_wrappers(profile, private_root=private_root)

    render_post_source_adapter_wrappers(profile, private_root=private_root)
    wrappers["quantization"].unlink()
    with pytest.raises(P6PostSourceWrapperError):
        validate_post_source_adapter_wrappers(profile, private_root=private_root)


def test_existing_wrapper_validator_rejects_exact_byte_wrapper_symlink(
    tmp_path: Path,
) -> None:
    private_root, _, profile = _profile_fixture(tmp_path)
    wrapper = render_post_source_adapter_wrappers(profile, private_root=private_root)[
        "quantization"
    ]
    linked_target = private_root / "exact-wrapper-target"
    linked_target.write_bytes(wrapper.read_bytes())
    linked_target.chmod(0o700)
    wrapper.unlink()
    wrapper.symlink_to(linked_target)

    with pytest.raises(P6PostSourceWrapperError):
        validate_post_source_adapter_wrappers(profile, private_root=private_root)


def test_existing_wrapper_validator_rejects_symlinked_wrapper_parent(
    tmp_path: Path,
) -> None:
    private_root, _, profile = _profile_fixture(tmp_path)
    wrapper = render_post_source_adapter_wrappers(profile, private_root=private_root)[
        "quantization"
    ]
    real_parent = private_root / "real-wrapper-bin"
    wrapper.parent.rename(real_parent)
    wrapper.parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(P6PostSourceWrapperError):
        validate_post_source_adapter_wrappers(profile, private_root=private_root)


def test_existing_wrapper_validator_rejects_wrong_profile_or_root(
    tmp_path: Path,
) -> None:
    private_root, _, profile = _profile_fixture(tmp_path)
    render_post_source_adapter_wrappers(profile, private_root=private_root)
    other_root = tmp_path / "other-root"
    other_root.mkdir()

    with pytest.raises(P6PostSourceWrapperError):
        validate_post_source_adapter_wrappers(profile, private_root=other_root)
    with pytest.raises(P6PostSourceWrapperError):
        validate_post_source_adapter_wrappers(object(), private_root=private_root)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    ["interpreter_drift", "adapter_implementation_drift", "leaf_digest_drift"],
)
def test_wrapper_rejects_profile_or_leaf_drift_before_adapter_execution(
    tmp_path: Path,
    mutation: str,
) -> None:
    private_root, profile_path, profile = _profile_fixture(tmp_path)
    wrapper = render_post_source_adapter_wrappers(profile, private_root=private_root)[
        "quantization"
    ]
    payload: dict[str, Any] = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if mutation == "interpreter_drift":
        payload["project_python"] = "/usr/bin/python3"
        profile_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    elif mutation == "adapter_implementation_drift":
        alternate = _write_executable(
            private_root / "execution-closure/history/alternate.py",
            _fake_adapter_body("alternate"),
        )
        payload["adapters"]["quantization"]["implementation_relative_path"] = (
            alternate.relative_to(private_root).as_posix()
        )
        profile_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        leaf = next(item for item in profile.leaves if item.name == "quant_contract")
        leaf.implementation.write_text("#!/usr/bin/env python3\n# drift\n", encoding="utf-8")
    report_path = tmp_path / f"{mutation}-record.json"

    completed = _run_wrapper(
        wrapper,
        private_root,
        ["--stage=quantization", str(report_path)],
    )

    assert completed.returncode == 1
    assert not report_path.exists()


def test_wrapper_rejects_adapter_implementation_byte_drift_before_child_execution(
    tmp_path: Path,
) -> None:
    private_root, _, profile = _profile_fixture(tmp_path)
    wrapper = render_post_source_adapter_wrappers(profile, private_root=private_root)[
        "quantization"
    ]
    adapter = next(item for item in profile.adapters if item.stage == "quantization")
    adapter.implementation.write_text(_fake_adapter_body("tampered-byte-drift"), encoding="utf-8")
    adapter.implementation.chmod(0o700)
    report_path = tmp_path / "adapter-byte-drift-record.json"

    completed = _run_wrapper(
        wrapper,
        private_root,
        ["--stage=quantization", str(report_path)],
    )

    assert completed.returncode == 1
    assert not report_path.exists()


def test_wrapper_propagates_child_nonzero_return_code(tmp_path: Path) -> None:
    private_root, _, profile = _profile_fixture(tmp_path)
    wrapper = render_post_source_adapter_wrappers(profile, private_root=private_root)[
        "quantization"
    ]

    completed = _run_wrapper(
        wrapper,
        private_root,
        ["--stage=quantization", "force-nonzero", str(tmp_path / "nonzero.json")],
    )

    assert completed.returncode == 37
