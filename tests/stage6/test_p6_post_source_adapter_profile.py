from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Callable

import pytest
import yaml

from framework.stage6.p6_history_normalization_v1 import (
    P6HistoryNormalizationError,
    normalize_history_inputs,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    PROFILE_SCHEMA_VERSION,
    POST_SOURCE_ADAPTER_STAGES,
    POST_SOURCE_LEAF_NAMES,
    P6PostSourceAdapterProfileError,
    load_post_source_adapter_profile,
)
from tests.stage6.test_p6_history_normalization import (
    _as_v2_procedural,
    _history_root,
    _tree_sha,
    valid_private_source_map,
)


LEAF_RELATIVE_PATHS = {
    "quant_contract": "scripts/quant-contract.leaf.py",
    "performance_plan": "scripts/performance-plan.leaf.py",
    "performance_execute": "scripts/performance-execute.leaf.py",
    "ap_plan": "scripts/ap-plan.leaf.py",
    "ap_execute": "scripts/ap-execute.leaf.py",
    "feedback_finalize": "scripts/feedback-finalize.leaf.py",
    "feedback_promote": "scripts/feedback-promote.leaf.py",
}


def _write_leaf(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o700)


def _exact_leaf_binding() -> dict[str, Any]:
    return {
        "schema_version": "p6_post_source_leaf_binding_v1",
        "leaves": {
            name: {
                "closure_id": "historical-chain",
                "entrypoint_relative_path": relative_path,
            }
            for name, relative_path in LEAF_RELATIVE_PATHS.items()
        },
    }


def v3_private_source_map(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    history_root = _history_root(source_map)
    closure_root = history_root / "execution-source"
    for name, relative_path in LEAF_RELATIVE_PATHS.items():
        _write_leaf(closure_root / relative_path, f"#!/usr/bin/env python3\n# {name}\n")
    root = source_map["execution_code_closure"]["roots"][0]
    root["closure_id"] = "historical-chain"
    root["sha256"] = _tree_sha(closure_root)
    for role in source_map["execution_code_closure"]["roles"].values():
        role["closure_id"] = "historical-chain"
    source_map["schema_version"] = "p6_history_normalization_source_v3"
    source_map["post_source_leaf_binding"] = _exact_leaf_binding()
    return source_map, runner


def v4_private_source_map(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    source_map, runner = v3_private_source_map(tmp_path)
    source_map["schema_version"] = "p6_history_normalization_source_v4"
    source_map["adapter_python"] = "/usr/bin/python3.10"
    return source_map, runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_profile(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source_map, runner = v3_private_source_map(tmp_path)
    private_dir = tmp_path / "private-normalized"
    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        private_dir,
        runner_template_path=runner,
    )
    profile_path = paths["post_source_adapter_profile"]
    return private_dir, profile_path, yaml.safe_load(profile_path.read_text())


def _write_profile(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_v3_normalizer_writes_ignored_post_source_adapter_profile(
    tmp_path: Path,
) -> None:
    source_map, runner = v3_private_source_map(tmp_path)
    private_dir = tmp_path / "private-normalized"

    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        private_dir,
        runner_template_path=runner,
    )

    profile_path = paths["post_source_adapter_profile"]
    assert profile_path == private_dir / "post-source-adapter-profile.yaml"
    exclude = private_dir / ".git" / "info" / "exclude"
    assert "post-source-adapter-profile.yaml\n" in exclude.read_text(encoding="utf-8")
    profile_yaml = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    assert profile_yaml["schema_version"] == PROFILE_SCHEMA_VERSION
    assert set(profile_yaml["adapters"]) == set(POST_SOURCE_ADAPTER_STAGES)
    assert tuple(profile_yaml["leaves"]) == tuple(POST_SOURCE_LEAF_NAMES)

    profile = load_post_source_adapter_profile(profile_path, private_root=private_dir)
    assert profile.private_root == private_dir
    assert profile.project_python == Path(profile_yaml["project_python"])
    assert tuple(adapter.stage for adapter in profile.adapters) == POST_SOURCE_ADAPTER_STAGES
    assert tuple(leaf.name for leaf in profile.leaves) == POST_SOURCE_LEAF_NAMES
    for leaf in profile.leaves:
        assert leaf.implementation == private_dir / profile_yaml["leaves"][leaf.name][
            "implementation_relative_path"
        ]
        assert leaf.sha256 == _sha256(leaf.implementation)


def test_v4_normalizer_writes_two_runtime_profile_without_changing_source_wrapper(
    tmp_path: Path,
) -> None:
    source_map, runner = v4_private_source_map(tmp_path)
    private_dir = tmp_path / "private-normalized"

    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        private_dir,
        runner_template_path=runner,
    )

    post_source = yaml.safe_load(
        paths["post_source_adapter_profile"].read_text(encoding="utf-8")
    )
    source_wrapper = yaml.safe_load(
        paths["source_wrapper_profile"].read_text(encoding="utf-8")
    )
    assert post_source["schema_version"] == "p6_post_source_adapter_profile_v2"
    assert post_source["adapter_python"] == "/usr/bin/python3.10"
    assert post_source["project_python"] == source_wrapper["project_python"]
    assert Path(post_source["project_python"]).name == "python3.9"

    loaded = load_post_source_adapter_profile(
        paths["post_source_adapter_profile"], private_root=private_dir
    )
    assert loaded.adapter_python == Path("/usr/bin/python3.10")
    assert loaded.project_python == Path(post_source["project_python"])


def test_v3_rejects_adapter_python_as_an_extra_key_and_remains_profile_v1(
    tmp_path: Path,
) -> None:
    source_map, runner = v3_private_source_map(tmp_path)
    source_map["adapter_python"] = "/usr/bin/python3.10"
    destination = tmp_path / "bad-v3"

    with pytest.raises(P6HistoryNormalizationError):
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            destination,
            runner_template_path=runner,
        )

    assert not destination.exists()

    source_map.pop("adapter_python")
    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        tmp_path / "good-v3",
        runner_template_path=runner,
    )
    payload = yaml.safe_load(paths["post_source_adapter_profile"].read_text())
    assert payload["schema_version"] == "p6_post_source_adapter_profile_v1"
    assert "adapter_python" not in payload


def _bad_adapter_python(tmp_path: Path, kind: str) -> object:
    if kind == "relative":
        return "bin/python3"
    executable = tmp_path / f"{kind}-python3"
    executable.write_text(
        "#!/bin/sh\nprintf '3.9\\n'\n", encoding="utf-8"
    )
    executable.chmod(0o700)
    if kind == "symlink":
        alias = tmp_path / "linked-python3"
        alias.symlink_to(executable.name)
        return str(alias)
    if kind == "hardlink":
        linked = tmp_path / "hardlinked-python3"
        os.link(executable, linked)
        return str(executable)
    if kind == "nonexec":
        executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return str(executable)


@pytest.mark.parametrize(
    "kind", ("relative", "symlink", "hardlink", "nonexec", "python39")
)
def test_v4_rejects_invalid_adapter_python_before_publication(
    tmp_path: Path,
    kind: str,
) -> None:
    source_map, runner = v4_private_source_map(tmp_path)
    adapter_python = _bad_adapter_python(tmp_path, kind)
    source_map["adapter_python"] = adapter_python
    destination = tmp_path / f"invalid-{kind}"
    with pytest.raises(P6HistoryNormalizationError):
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            destination,
            runner_template_path=runner,
        )

    assert not destination.exists()


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_v4_requires_exact_source_map_keys(
    tmp_path: Path,
    mutation: str,
) -> None:
    source_map, runner = v4_private_source_map(tmp_path)
    if mutation == "missing":
        source_map.pop("adapter_python")
    else:
        source_map["unexpected"] = "rejected"
    destination = tmp_path / mutation

    with pytest.raises(P6HistoryNormalizationError):
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            destination,
            runner_template_path=runner,
        )

    assert not destination.exists()


def test_v2_normalizer_omits_post_source_adapter_profile(
    tmp_path: Path,
) -> None:
    source_map, runner = _as_v2_procedural(valid_private_source_map(tmp_path), tmp_path)
    private_dir = tmp_path / "private-normalized"

    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        private_dir,
        runner_template_path=runner,
    )

    assert "post_source_adapter_profile" not in paths
    assert not (private_dir / "post-source-adapter-profile.yaml").exists()


def _missing_leaf(source_map: dict[str, Any], _tmp_path: Path) -> None:
    source_map["post_source_leaf_binding"]["leaves"].pop("ap_execute")


def _extra_leaf(source_map: dict[str, Any], _tmp_path: Path) -> None:
    source_map["post_source_leaf_binding"]["leaves"]["unexpected_leaf"] = {
        "closure_id": "historical-chain",
        "entrypoint_relative_path": "scripts/unexpected.leaf.py",
    }


def _unknown_closure_id(source_map: dict[str, Any], _tmp_path: Path) -> None:
    source_map["post_source_leaf_binding"]["leaves"]["quant_contract"][
        "closure_id"
    ] = "unknown-chain"


def _escaping_relative_path(source_map: dict[str, Any], _tmp_path: Path) -> None:
    source_map["post_source_leaf_binding"]["leaves"]["quant_contract"][
        "entrypoint_relative_path"
    ] = "../quant-contract.leaf.py"


def _non_executable_leaf(source_map: dict[str, Any], _tmp_path: Path) -> None:
    history_root = _history_root(source_map)
    leaf = history_root / "execution-source" / LEAF_RELATIVE_PATHS["quant_contract"]
    leaf.chmod(0o600)
    source_map["execution_code_closure"]["roots"][0]["sha256"] = _tree_sha(
        history_root / "execution-source"
    )


def _duplicate_resolved_leaf(source_map: dict[str, Any], _tmp_path: Path) -> None:
    source_map["post_source_leaf_binding"]["leaves"]["performance_plan"][
        "entrypoint_relative_path"
    ] = LEAF_RELATIVE_PATHS["quant_contract"]


@pytest.mark.parametrize(
    "mutator",
    [
        _missing_leaf,
        _extra_leaf,
        _unknown_closure_id,
        _escaping_relative_path,
        _non_executable_leaf,
        _duplicate_resolved_leaf,
    ],
)
def test_v3_leaf_binding_failures_stop_before_private_root_publication(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any], Path], None],
) -> None:
    source_map, runner = v3_private_source_map(tmp_path)
    source_map = copy.deepcopy(source_map)
    mutator(source_map, tmp_path)
    private_dir = tmp_path / "private-normalized"

    with pytest.raises(P6HistoryNormalizationError):
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            private_dir,
            runner_template_path=runner,
        )

    assert not private_dir.exists()


def test_profile_loader_rejects_unknown_keys_without_path_disclosure(
    tmp_path: Path,
) -> None:
    private_dir, _, payload = _normalized_profile(tmp_path)
    payload["unexpected"] = "private"
    bad_profile = private_dir / "bad-post-source-adapter-profile.yaml"

    with pytest.raises(P6PostSourceAdapterProfileError) as captured:
        load_post_source_adapter_profile(
            _write_profile(bad_profile, payload), private_root=private_dir
        )

    assert str(private_dir) not in str(captured.value)


def test_profile_loader_rejects_duplicate_yaml_key(
    tmp_path: Path,
) -> None:
    private_dir, profile_path, _ = _normalized_profile(tmp_path)
    bad_profile = private_dir / "duplicate-key-profile.yaml"
    bad_profile.write_text(
        profile_path.read_text(encoding="utf-8")
        + "schema_version: p6_post_source_adapter_profile_v1\n",
        encoding="utf-8",
    )

    with pytest.raises(P6PostSourceAdapterProfileError):
        load_post_source_adapter_profile(bad_profile, private_root=private_dir)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["leaves"]["quant_contract"].update({"sha256": "0" * 64}),
        lambda payload: payload["leaves"]["quant_contract"].update({"sha256": "bad"}),
        lambda payload: payload["adapters"]["quantization"].update(
            {"implementation_relative_path": "../quantize"}
        ),
        lambda payload: payload["adapters"]["quantization"].update(
            {"implementation_cwd_relative_path": "inputs"}
        ),
        lambda payload: payload["leaves"]["performance_plan"].update(
            {
                "implementation_relative_path": payload["leaves"]["quant_contract"][
                    "implementation_relative_path"
                ]
            }
        ),
    ],
)
def test_profile_loader_rejects_invalid_profile_boundaries(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    private_dir, _, payload = _normalized_profile(tmp_path)
    mutator(payload)
    bad_profile = private_dir / "invalid-profile.yaml"

    with pytest.raises(P6PostSourceAdapterProfileError):
        load_post_source_adapter_profile(
            _write_profile(bad_profile, payload), private_root=private_dir
        )


def test_profile_loader_requires_absolute_profile_path_and_private_root(
    tmp_path: Path,
) -> None:
    private_dir, profile_path, _ = _normalized_profile(tmp_path)

    with pytest.raises(P6PostSourceAdapterProfileError):
        load_post_source_adapter_profile(
            Path(profile_path.name), private_root=private_dir
        )
    with pytest.raises(P6PostSourceAdapterProfileError):
        load_post_source_adapter_profile(
            profile_path, private_root=Path(private_dir.name)
        )
