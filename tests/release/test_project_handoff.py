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


pytestmark = pytest.mark.skipif(
    _is_anonymous_reviewer_archive(),
    reason="the anonymous reviewer ZIP deliberately excludes public release documentation",
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
    assert coverage["document_candidate_count"] == 199
    assert coverage["asset_candidate_count"] == 307
    assert coverage["distinct_resource_count"] == 284
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
    p4_plan_line = next(
        line for line in handoff.splitlines() if line.startswith("| P4 |")
    )
    assert (
        "506 条 P3 转交候选均已在本地裁决为 document 或 asset：199 条为 document、"
        "307 条为 asset，去重后为 284 项资源；284 项均为 unavailable。"
        in p4_plan_line
    )


def test_p4_audit_links_complete_registry_and_coverage() -> None:
    """The handoff must expose the complete P4 registry and coverage evidence."""
    handoff = HANDOFF.read_text(encoding="utf-8")

    assert "P4_EXTERNAL_INPUT_CONTRACT.md" in handoff
    assert "artifacts/external/coverage.json" in handoff
    assert "506" in handoff


def test_p4_closure_records_the_successful_public_ci_gate() -> None:
    """P4 closes locally only after its three public CI jobs have succeeded."""
    handoff = HANDOFF.read_text(encoding="utf-8")

    assert "| P4 | 已完成（本地） |" in handoff
    assert "31404700281" in handoff
    for job_name in ("quality (3.10)", "quality (3.11)", "public-smoke"):
        assert job_name in handoff


def test_p5_environment_contract_is_discoverable() -> None:
    contract = REPOSITORY_ROOT / "docs/release-manifests/P5_ENVIRONMENT_CONTRACT.md"

    assert contract.is_file()
    contract_text = contract.read_text(encoding="utf-8")
    assert "validate_environment_contract.py" in contract_text
    assert "已完成（本地）" in contract_text
    assert "进行中（本地）" not in contract_text
    assert "只读取这三个显式路径" not in contract_text
    assert "验证 environment contract 引用的公开 hardware capability YAML" in contract_text
    assert "不访问网络或读取外部资源" in contract_text

    handoff = HANDOFF.read_text(encoding="utf-8")
    p5_plan_line = next(
        line for line in handoff.splitlines() if line.startswith("| P5 |")
    )
    assert "[P5 环境契约](release-manifests/P5_ENVIRONMENT_CONTRACT.md)" in p5_plan_line
    assert "已完成（本地）" in p5_plan_line
    assert "进行中" not in p5_plan_line


def test_p5_local_closure_is_limited_to_offline_environment_contracts() -> None:
    handoff = HANDOFF.read_text(encoding="utf-8")
    p5_plan_line = next(
        line for line in handoff.splitlines() if line.startswith("| P5 |")
    )
    completion_heading = "**P5 环境契约完成（2026-08-11）**："
    assert completion_heading in handoff
    p5_completion_entry = handoff.split(completion_heading, maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]

    assert "| P5 | 已完成（本地） |" in p5_plan_line
    assert "P5_ENVIRONMENT_CONTRACT.md" in p5_plan_line
    assert "三份公开 hardware capability YAML" in p5_plan_line
    assert "三份环境契约" in p5_plan_line
    assert "合成正负例" in p5_plan_line
    assert "离线验证器" in p5_plan_line
    assert "不探测本机 GPU" in p5_plan_line
    assert "不运行 CUDA 编译、训练、评测、基准测试或任何外部资产" in p5_plan_line
    assert "网络下载" in p5_plan_line
    assert "远端 CI" not in p5_plan_line

    assert "三份公开 hardware capability YAML" in p5_completion_entry
    assert (
        "三份 [P5 环境契约](release-manifests/P5_ENVIRONMENT_CONTRACT.md)"
        in p5_completion_entry
    )
    assert "合成正负例" in p5_completion_entry
    assert "离线验证器" in p5_completion_entry
    assert "已完成（本地）" in p5_completion_entry
    assert "P5_ENVIRONMENT_CONTRACT.md" in p5_completion_entry
    assert "不探测本机 GPU" in p5_completion_entry
    assert "不运行 CUDA 编译、训练、评测、基准测试或任何外部资产" in p5_completion_entry
    assert "网络下载" in p5_completion_entry
    assert "真实硬件执行" in p5_completion_entry
    assert "远端 CI" not in p5_completion_entry


def test_p6_h800_execution_manifest_records_local_closure_without_public_results() -> None:
    """P6.1 history, P6.2 offline work, and the real run stay distinct."""
    manifest = REPOSITORY_ROOT / "docs/release-manifests/P6_H800_SEARCH_EXECUTION.md"

    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    handoff = HANDOFF.read_text(encoding="utf-8")
    p6_plan_line = next(
        line for line in handoff.splitlines() if line.startswith("| P6 |")
    )

    assert "P6.1 静态 343×2" in text
    assert "P6.2 动态框架候选空间" in text
    assert "不预设为 343 或 686" in text
    assert "单个候选不是 mixed-precision" in text
    assert "真实框架来源闭环仍待本地执行" in text
    assert "P6.1 静态 343×2 历史路径已完成（本地）；P6.2 动态框架候选空间离线接入与验证已完成（本地）" in p6_plan_line
    assert "/home/" not in text
    for prohibited_public_detail in (
        "candidate_id:",
        "latency_ms:",
        "energy_j:",
        "ap30:",
        "ap50:",
        "ap70:",
    ):
        assert prohibited_public_detail not in text

    assert "状态日期：2026-08-17" in handoff
    assert "P6.2 动态框架候选空间离线接入完成后的公开交接状态" in handoff
    assert "P6.2 动态框架候选空间离线接入完成" in handoff
    assert "Stage6/Stage7 论文证据" in handoff
    assert "unavailable" in handoff
    assert "/home/" not in handoff
    for prohibited_claim in (
        "P6 已完成（本地）",
        "P6 整体已关闭",
        "真实框架来源闭环已完成",
        "真实 H800 框架来源闭环已完成",
        "Stage6 或 Stage7 论文证据已完成",
        "| P7 | 已完成",
        "| P7 | 进行中",
        "| P7 | 已开始",
        "| P8 | 已完成",
        "| P8 | 进行中",
        "| P8 | 已开始",
    ):
        assert prohibited_claim not in handoff


def test_p6_framework_search_space_gate_is_documented() -> None:
    manifest = (REPOSITORY_ROOT / "docs/release-manifests/P6_H800_SEARCH_EXECUTION.md").read_text(encoding="utf-8")
    assert "从 Stage2 动态派生候选池" in manifest
    assert "不预设为 343 或 686" in manifest
    assert "P6.2 动态框架候选空间的离线接入与验证已完成" in manifest
    assert "真实框架来源闭环仍待本地执行" in manifest
