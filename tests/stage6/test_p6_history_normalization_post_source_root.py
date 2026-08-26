from __future__ import annotations

import ast
from pathlib import Path

import pytest

from framework.stage6.p6_history_normalization_v1 import (
    P6HistoryNormalizationError,
    normalize_history_inputs,
)
from framework.stage6.p6_post_source_adapter_profile_v1 import (
    load_post_source_adapter_profile,
)
from framework.stage6.p6_post_source_wrapper_template_v1 import (
    WRAPPER_RELATIVE_PATHS,
    expected_post_source_wrapper_bytes_from_profile,
    validate_post_source_adapter_wrappers,
)
from framework.stage6.p6_source_wrapper_profile_v1 import (
    expected_wrapper_bytes_from_profile,
)
from tests.stage6.test_p6_history_normalization import _history_root
from tests.stage6.test_p6_post_source_adapter_profile import v3_private_source_map


def _normalize_v3(tmp_path: Path, name: str) -> tuple[Path, dict[str, Path]]:
    source_map, runner = v3_private_source_map(tmp_path)
    destination = tmp_path / name
    paths = normalize_history_inputs(
        source_map,
        _history_root(source_map),
        destination,
        runner_template_path=runner,
    )
    return destination, paths


def _private_root_literal(wrapper: Path) -> str:
    tree = ast.parse(wrapper.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PRIVATE_ROOT" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Call)
    return ast.literal_eval(assignment.value.args[0])


def test_v3_normalization_binds_all_post_source_wrappers_to_published_root(
    tmp_path: Path,
) -> None:
    destination, paths = _normalize_v3(tmp_path, "normalized-final")

    profile = load_post_source_adapter_profile(
        paths["post_source_adapter_profile"], private_root=destination
    )
    wrappers = validate_post_source_adapter_wrappers(profile, private_root=destination)
    expected_bytes = expected_post_source_wrapper_bytes_from_profile(profile)

    assert wrappers == {
        stage: destination / relative
        for stage, relative in WRAPPER_RELATIVE_PATHS.items()
    }
    for stage, wrapper in wrappers.items():
        text = wrapper.read_text(encoding="utf-8")
        assert wrapper.read_bytes() == expected_bytes[stage]
        assert _private_root_literal(wrapper) == str(destination)
        assert Path(_private_root_literal(wrapper)).is_dir()
        assert ".staging" not in text

    source_profile = paths["source_wrapper_profile"]
    source_wrapper = destination / "documented-stage5-chain/stage5_materialize_round_sources_v1.sh"
    assert source_wrapper.read_bytes() == expected_wrapper_bytes_from_profile(
        source_profile,
        history_root=destination,
    )


def test_v3_normalization_wrapper_bytes_follow_final_destination(tmp_path: Path) -> None:
    first_root, _ = _normalize_v3(tmp_path / "first", "normalized-one")
    second_root, _ = _normalize_v3(tmp_path / "second", "normalized-two")

    for relative in WRAPPER_RELATIVE_PATHS.values():
        first = (first_root / relative).read_bytes()
        second = (second_root / relative).read_bytes()
        assert first != second
        assert str(first_root).encode() in first
        assert str(second_root).encode() in second


def test_v3_publish_failure_leaves_no_destination_or_staging_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_map, runner = v3_private_source_map(tmp_path)
    destination = tmp_path / "normalized-final"
    import framework.stage6.p6_history_normalization_staging_v1 as staging

    real_replace = staging.os.replace

    def fail_publication(source: Path | str, target: Path | str) -> None:
        if Path(target) == destination:
            raise OSError("injected atomic publication failure")
        real_replace(source, target)

    monkeypatch.setattr(staging.os, "replace", fail_publication)

    with pytest.raises(P6HistoryNormalizationError):
        normalize_history_inputs(
            source_map,
            _history_root(source_map),
            destination,
            runner_template_path=runner,
        )

    assert not destination.exists()
    assert not tuple(destination.parent.glob(f".{destination.name}.*.staging"))
