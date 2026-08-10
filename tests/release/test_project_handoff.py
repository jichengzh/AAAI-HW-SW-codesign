"""Regression checks for the maintained public-release handoff ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HANDOFF = REPOSITORY_ROOT / "docs/AAAI27_RELEASE_AUDIT.md"


def _is_anonymous_reviewer_archive() -> bool:
    root_readme = REPOSITORY_ROOT / "README.md"
    return (
        not (REPOSITORY_ROOT / "README.anonymous.md").exists()
        and root_readme.is_file()
        and root_readme.read_text(encoding="utf-8").startswith("# Anonymous AAAI Submission")
    )


def test_handoff_records_baseline_current_state_and_open_source_plan() -> None:
    """A successor must be able to find the release starting point and remaining work."""
    if _is_anonymous_reviewer_archive():
        pytest.skip("the anonymous reviewer ZIP deliberately excludes the public handoff ledger")
    handoff = HANDOFF.read_text(encoding="utf-8")

    for heading in (
        "## 使用与更新规则",
        "## 起始状态",
        "## 当前状态",
        "## 到完整开源的执行计划",
        "## 完成定义与发布门槛",
        "## 变更记录",
    ):
        assert heading in handoff
    for phase in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        assert phase in handoff
    assert "REPRODUCIBILITY.md" in handoff
    assert "ARTIFACTS.md" in handoff
    assert "每完成一项" in handoff
    assert "不记录" in handoff


def test_public_readmes_link_the_handoff_ledger() -> None:
    """The maintained plan is discoverable without depending on an old chat transcript."""
    if _is_anonymous_reviewer_archive():
        pytest.skip("the anonymous reviewer ZIP deliberately excludes public release documentation")
    for readme_name in ("README.md", "README.zh-CN.md"):
        readme = (REPOSITORY_ROOT / readme_name).read_text(encoding="utf-8")
        assert "docs/AAAI27_RELEASE_AUDIT.md" in readme


def test_p4_contract_is_discoverable_and_has_complete_coverage_summary() -> None:
    coverage = json.loads(
        (REPOSITORY_ROOT / "artifacts/external/coverage.json").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads(
        (REPOSITORY_ROOT / "artifacts/external/registry.json").read_text(
            encoding="utf-8"
        )
    )
    contract = (
        REPOSITORY_ROOT / "docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md"
    )
    assert contract.is_file()
    assert coverage["format"] == "aaai27_p4_external_resource_coverage_v1"
    assert coverage["p3_p4_candidate_count"] == 506
    assert coverage["document_candidate_count"] + coverage["asset_candidate_count"] == 506
    assert coverage["distinct_resource_count"] == len(registry["inputs"])
    assert registry["format"] == "aaai27_external_input_registry_v1"
    assert {item["input_id"] for item in registry["inputs"]} >= {
        "stage6-terminal-evidence",
        "stage7-formal-aggregate",
    }
    assert all(item["availability"] == "unavailable" for item in registry["inputs"])
    assert all(item["unavailable_reason"] for item in registry["inputs"])
    forbidden_private_fields = {
        "candidate_id",
        "candidate_path",
        "origin",
        "path",
        "private_locator",
        "private_path",
        "local_path",
    }
    assert all(
        forbidden_private_fields.isdisjoint(item) for item in registry["inputs"]
    )
    assert all(item["asset_kind"] != "document" for item in registry["inputs"])
    handoff = HANDOFF.read_text(encoding="utf-8")
    assert "P4_EXTERNAL_INPUT_CONTRACT.md" in handoff
    assert "进行中（本地）" in handoff


def test_p4_audit_links_complete_registry_and_coverage() -> None:
    """The handoff must expose the completed local P4 evidence without closure."""
    handoff = HANDOFF.read_text(encoding="utf-8")

    assert "P4_EXTERNAL_INPUT_CONTRACT.md" in handoff
    assert "artifacts/external/coverage.json" in handoff
    assert "506" in handoff
