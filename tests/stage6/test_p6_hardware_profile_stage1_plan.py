from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from framework.stage6.hardware_execution_profile_v1 import (
    HardwareExecutionProfile,
    load_hardware_execution_profile,
)
from framework.stage6.p6_formal_plan_contract_v1 import validate_p6_candidate_plan
from framework.stage6.p6_stage1_bridge_v1 import (
    P6Stage1BridgeError,
    build_p6_stage1_partition_manifest,
)
from framework.stage6.pyramid_search_space_adapter_v1 import (
    build_pyramid_candidate_plan,
)
from framework.stage1_bridge import load_stage2_search_space
from tests.stage6.pyramid_formal_space_support import (
    scanner_owned_pyramid_stage1_manifest,
    scanner_owned_pyramid_stage2_space,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PAPER_STAGE_WIDTHS = {
    "stage1": [16, 24, 32, 40, 48, 56, 64],
    "stage2": [32, 48, 64, 80, 96, 112, 128],
    "stage3": [64, 96, 128, 160, 192, 224, 256],
}


def _profile(profile_id: str) -> HardwareExecutionProfile:
    return load_hardware_execution_profile(profile_id)


def _scenario_path(profile_id: str) -> Path:
    return (
        REPOSITORY_ROOT
        / "configs"
        / "stage1"
        / f"p6_{profile_id}_formal_scan.yaml"
    )


def _hardware_path(profile: HardwareExecutionProfile) -> Path:
    return REPOSITORY_ROOT.joinpath(*profile.hardware_capability_path.parts)


def _manifest(hardware_name: str) -> dict[str, Any]:
    manifest = scanner_owned_pyramid_stage1_manifest()
    manifest["hw_capability"]["name"] = hardware_name
    manifest["formal_scan"] = {"status": "derived"}
    return manifest


def _scanner_plan() -> dict[str, Any]:
    return build_pyramid_candidate_plan(
        scanner_owned_pyramid_stage2_space(PAPER_STAGE_WIDTHS)
    )


def test_rtx_profile_accepts_committed_hardware_yaml_and_emits_canonical_target(
    tmp_path: Path,
) -> None:
    profile = _profile("rtx4090")
    source_manifest = _manifest("NVIDIA RTX 4090")
    calls: list[tuple[str, Path, str, Mapping[str, str], Path]] = []

    def scanner(
        model_name: str,
        hardware_path: Path,
        device: str,
        environment: Mapping[str, str],
        scenario_path: Path,
    ) -> Mapping[str, Any]:
        calls.append((model_name, hardware_path, device, environment, scenario_path))
        return source_manifest

    output_path = tmp_path / "manifest.json"
    result = build_p6_stage1_partition_manifest(
        output_path,
        _hardware_path(profile),
        "cuda:0",
        {},
        scanner,
        profile=profile,
        scenario_path=_scenario_path(profile.profile_id),
    )

    assert result["hw_capability"]["name"] == "rtx4090"
    assert source_manifest["hw_capability"]["name"] == "NVIDIA RTX 4090"
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert load_stage2_search_space(output_path)["hardware_target"]["name"] == (
        "rtx4090"
    )
    assert calls == [
        (
            "pyramid_lidar",
            _hardware_path(profile),
            "cuda:0",
            {},
            _scenario_path(profile.profile_id),
        )
    ]


def test_controller_local_rtx_stage1_invocation_selects_committed_profile(
    tmp_path: Path,
) -> None:
    profile = _profile("rtx4090")

    result = build_p6_stage1_partition_manifest(
        tmp_path / "manifest.json",
        _hardware_path(profile),
        "cuda:0",
        {},
        lambda *_args: _manifest("NVIDIA RTX 4090"),
        scenario_path=_scenario_path(profile.profile_id),
    )

    assert result["hw_capability"]["name"] == "rtx4090"


@pytest.mark.parametrize(
    "foreign_path",
    [
        pytest.param(
            REPOSITORY_ROOT / "configs" / "hardware" / "h800.yaml",
            id="foreign-profile-yaml",
        ),
        pytest.param(None, id="copied-rtx-yaml"),
    ],
)
def test_rtx_profile_rejects_hardware_yaml_outside_its_registry_path(
    tmp_path: Path,
    foreign_path: Path | None,
) -> None:
    profile = _profile("rtx4090")
    if foreign_path is None:
        foreign_path = tmp_path / "rtx4090.yaml"
        foreign_path.write_bytes(_hardware_path(profile).read_bytes())
    scanner_called = False

    def scanner(*_args: object) -> Mapping[str, Any]:
        nonlocal scanner_called
        scanner_called = True
        return _manifest("NVIDIA RTX 4090")

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            tmp_path / "manifest.json",
            foreign_path,
            "cuda:0",
            {},
            scanner,
            profile=profile,
            scenario_path=_scenario_path(profile.profile_id),
        )

    assert not scanner_called


def test_rtx_profile_rejects_sm90_hardware_yaml_before_scanner(
    tmp_path: Path,
) -> None:
    profile = _profile("rtx4090")
    hardware_path = tmp_path / "rtx4090-sm90.yaml"
    hardware_path.write_text(
        "name: NVIDIA RTX 4090\narch:\n  family: Ada\n  sm: sm90\n",
        encoding="utf-8",
    )
    scanner_called = False

    def scanner(*_args: object) -> Mapping[str, Any]:
        nonlocal scanner_called
        scanner_called = True
        return _manifest("NVIDIA RTX 4090")

    with pytest.raises(P6Stage1BridgeError, match="stage1_scan_invalid"):
        build_p6_stage1_partition_manifest(
            tmp_path / "manifest.json",
            hardware_path,
            "cuda:0",
            {},
            scanner,
            profile=profile,
            scenario_path=_scenario_path(profile.profile_id),
        )

    assert not scanner_called


def test_h800_profile_candidate_plan_preserves_scanner_owned_343_by_686_space(
) -> None:
    plan = _scanner_plan()

    mapping = validate_p6_candidate_plan(plan, profile=_profile("h800"))

    assert plan["structure_count"] == 343
    assert plan["candidate_count"] == 686
    assert len(mapping) == 686


def test_profile_candidate_plan_rejects_target_drift() -> None:
    plan = _scanner_plan()
    plan["hardware_target"] = "foreign"

    with pytest.raises(ValueError, match="hardware.*profile"):
        validate_p6_candidate_plan(plan, profile=_profile("h800"))


def test_profile_candidate_plan_rejects_backend_drift() -> None:
    plan = _scanner_plan()
    plan["execution_backend"] = "trt_engine"

    with pytest.raises(ValueError, match="backend.*profile"):
        validate_p6_candidate_plan(plan, profile=_profile("h800"))


def test_rtx_profile_rejects_manually_substituted_candidate_plan_target() -> None:
    plan = copy.deepcopy(_scanner_plan())
    plan["hardware_target"] = "rtx4090"

    with pytest.raises(ValueError, match="scanner.*hardware.*profile"):
        validate_p6_candidate_plan(plan, profile=_profile("rtx4090"))
