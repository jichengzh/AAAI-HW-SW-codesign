"""Regression checks for the maintained public-release handoff ledger."""

from __future__ import annotations

from collections.abc import Callable
import json
import re
from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HANDOFF = REPOSITORY_ROOT / "docs/AAAI27_RELEASE_AUDIT.md"


_TWO_GPU_INDEX_PAIR = r"(?<![{\d,])\d+\s*,\s*\d+(?!\d)(?!\s*,\s*\d)"
_TRACKED_GPU_POLICY_DISCLOSURES = (
    re.compile(
        rf"(?:CUDA_VISIBLE_DEVICES[^\n]*?{_TWO_GPU_INDEX_PAIR}|"
        rf"nvidia-smi[^\n]*?--id[^\n]*?{_TWO_GPU_INDEX_PAIR}|"
        rf"--gpus[^\n]*?{_TWO_GPU_INDEX_PAIR}|"
        rf"gpu_policy[^\n]*?[\"']indices[\"']\s*:\s*\[\s*"
        rf"{_TWO_GPU_INDEX_PAIR}\s*\]|"
        rf"(?:GPU indices|GPU编号|显卡编号)\s*[:：=]?\s*\[?\s*"
        rf"{_TWO_GPU_INDEX_PAIR}\s*\]?)"
    ),
    re.compile(
        r"CUDA_VISIBLE_DEVICES[^\n]{0,120}[\"']\d\s*,\s*\d\s*,\s*\d[\"']"
    ),
    re.compile(r"nvidia-smi[^\n]{0,120}--id(?:=|\s+)\d(?:\s*,\s*\d){2}"),
    re.compile(
        r"(?:GPU|显卡)[^\n]{0,80}(?:"
        r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]|"
        r"\d+\s*[/、]\s*\d+\s*[/、]\s*\d+)"
    ),
    re.compile(
        r"(?:DEFAULT|FIXED)[A-Z0-9_]*GPU[A-Z0-9_]*\s*=\s*"
        r"[\[(]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\])]"
    ),
    re.compile(
        r"gpu_policy[^\n]{0,160}[\"']\d[\"']\s*,\s*"
        r"[\"']\d[\"']\s*,\s*[\"']\d[\"']"
    ),
)


_P6_PUBLIC_DISCLOSURE_PATTERNS = (
    r"(?<![\w/])/(?:[^\s/]+/)+[^\s/]+",
    r"\b[A-Za-z]:[\\/]+(?:[^\s\\/]+[\\/]+)*[^\s\\/]+",
    r"[\"']?(?:candidate(?:[_ -]?id)?|候选\s*(?:ID|标识))[\"']?\s*[:：=]\s*\S+",
    r"[\"']?(?:latency|energy|ap(?:30|50|70)?)(?:_[A-Za-z0-9]+)?[\"']?\s*[:=]\s*[-+]?\d",
    r"[\"']?(?:延迟|能耗|原始结果|原始指标)[\"']?\s*[:：=]\s*[-+]?\d",
    r"[\"']?(?:checkpoint|results?|raw[_ -]?results?)[\"']?\s*[:=]\s*\S+",
    r"\b\S+\.(?:ckpt|pth|pt|onnx)\b",
    r"[\"']?(?:host|hostname|主机)[\"']?\s*[:：=]\s*\S+",
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    r"(?:^|\n)\s*(?:python(?:3)?|bash|sh|zsh)\s+\S+",
    r"(?:\b\S+\.log\b|[\"']?(?:log|日志)[\"']?\s*[:：=]\s*\S+)",
    r"\bGPU-[A-Za-z0-9][A-Za-z0-9-]*\b",
    r"[\"']?(?:command|cmd|argv|命令)[\"']?\s*[:：=]\s*\S+",
    r"[\"']?(?:measurement[_ -]?request|feedback|请求|反馈)[\"']?\s*[:：=]\s*\S+",
)


def _assert_p6_public_disclosure_safe(documents: dict[str, str]) -> None:
    """Reject concrete execution disclosures without rejecting policy prohibitions."""
    for document_name, content in documents.items():
        for pattern in _P6_PUBLIC_DISCLOSURE_PATTERNS:
            assert not re.search(pattern, content, flags=re.MULTILINE), (
                f"{document_name} exposes a P6 execution detail matching {pattern!r}"
            )

    combined = "\n".join(documents.values())
    assert "python tools/release/run_p6_h800_search.py" not in combined
    for completed_real_run_claim in (
        "真实 H800 框架来源四轮执行已完成",
        "P6.3 真实 H800 运行已完成",
        "P6.3 real H800 run completed",
    ):
        assert completed_real_run_claim not in combined


def _p6_boundary_sections(content: str) -> str:
    """Return P6 paragraphs with adjacent context, not only matching lines."""
    paragraphs = content.split("\n\n")
    p6_indices = {
        index for index, paragraph in enumerate(paragraphs) if "P6" in paragraph
    }
    selected_indices = {
        index + offset
        for index in p6_indices
        for offset in (-1, 0, 1)
        if 0 <= index + offset < len(paragraphs)
    }
    return "\n\n".join(
        paragraph
        for index, paragraph in enumerate(paragraphs)
        if index in selected_indices
    )


def _tracked_gpu_policy_disclosures() -> list[str]:
    """Return only explicit tracked fixed-policy disclosures, not unrelated numbers."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    tracked_paths = tuple(
        Path(name)
        for name in completed.stdout.decode("utf-8").split("\0")
        if name
    )
    disclosures: list[str] = []
    for relative_path in tracked_paths:
        if relative_path.suffix not in {".md", ".py", ".yaml", ".yml"}:
            continue
        content = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        for pattern in _TRACKED_GPU_POLICY_DISCLOSURES:
            if pattern.search(content):
                disclosures.append(f"{relative_path}: {pattern.pattern}")
    return disclosures


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
    reproducibility = (REPOSITORY_ROOT / "REPRODUCIBILITY.md").read_text(
        encoding="utf-8"
    )
    artifacts = (REPOSITORY_ROOT / "ARTIFACTS.md").read_text(encoding="utf-8")
    p6_plan_line = next(
        line for line in handoff.splitlines() if line.startswith("| P6 |")
    )

    assert "P6.1 静态 343×2" in text
    assert "P6.2 动态框架候选空间" in text
    assert "P6.3 历史执行适配器已实现并通过离线验证" in text
    assert "不预设为 343 或 686" in text
    assert "单个候选不是 mixed-precision" in text
    assert "真实 H800 框架来源四轮执行仍待本地运行" in text
    assert "P6.3 历史执行适配器已实现并通过离线验证" in p6_plan_line
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

    assert "状态日期：2026-08-19" in handoff
    assert "P6.3 历史执行适配器离线验证后的公开交接状态" in handoff
    assert "P6.2 动态框架候选空间离线接入完成" in handoff
    assert "P6.3 历史执行适配器已实现并通过离线验证" in handoff
    assert "真实 H800 框架来源四轮执行仍待本地运行" in handoff
    assert "Stage6/Stage7 论文证据" in handoff
    assert "unavailable" in handoff
    assert "/home/" not in handoff
    assert "P6.1 has" in reproducibility
    assert "completed a separate, Git-ignored local execution closure" in reproducibility
    assert "no checked-in result bundle" in reproducibility
    assert "Stage6 representative selection and the Stage7 formal aggregate remain" in reproducibility
    assert "P6.1 execution result bundle" in artifacts
    assert "does not add a checked-in artifact" in artifacts
    assert "a verified\nStage6/Stage7 entry" in artifacts
    for prohibited_claim in (
        "P6 已完成（本地）",
        "P6 整体已关闭",
        "真实框架来源闭环已完成",
        "真实 H800 框架来源闭环已完成",
        "Stage6 或 Stage7 论文证据已完成",
        "真实 H800 框架来源四轮执行已完成",
        "P6.3 真实 H800 运行已完成",
        "| P7 | 已完成",
        "| P7 | 进行中",
        "| P7 | 已开始",
        "| P8 | 已完成",
        "| P8 | 进行中",
        "| P8 | 已开始",
    ):
        assert prohibited_claim not in handoff

    _assert_p6_public_disclosure_safe(
        {
            "manifest": text,
            "AAAI audit P6 boundary": _p6_boundary_sections(handoff),
            "reproducibility P6 boundary": _p6_boundary_sections(reproducibility),
            "artifacts P6 boundary": _p6_boundary_sections(artifacts),
        }
    )


@pytest.mark.parametrize(
    "leaked_detail",
    (
        "/private/run/output",
        r"C:\private\p6\run",
        "candidate_id: p6-h800-001",
        "latency_ms: 12.34",
        '{"latency_ms": 12.34}',
        "results: results/p6-h800-search.json",
        '"raw-results": "results/p6-h800-search.json"',
        "checkpoint: model.ckpt",
        "host: h800-worker",
        "python tools/release/run_p6_h800_search.py",
        "round-1.log",
        "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "argv: python private_history_adapter.py",
        "measurement_request: private-request.json",
        "feedback: private-feedback.json",
        "P6.3 真实 H800 运行已完成",
    ),
)
def test_p6_public_disclosure_guard_rejects_concrete_leaks(
    leaked_detail: str,
) -> None:
    with pytest.raises(AssertionError):
        _assert_p6_public_disclosure_safe({"synthetic public P6 text": leaked_detail})


def test_p6_public_disclosure_guard_allows_policy_prohibitions() -> None:
    _assert_p6_public_disclosure_safe(
        {
            "public policy": (
                "不得公开命令、路径、候选标识、原始结果或日志。"
                "P6.3 历史执行适配器已实现并通过离线验证；"
                "真实 H800 框架来源四轮执行仍待本地运行。"
            )
        }
    )


def test_tracked_surface_has_no_fixed_private_gpu_policy_disclosure() -> None:
    """Catches fixed P6 device policies leaking into tracked source or prose."""
    assert _tracked_gpu_policy_disclosures() == []


def _synthetic_two_gpu_csv() -> str:
    return ",".join(str(index) for index in range(101, 105, 2))


@pytest.mark.parametrize(
    "render",
    (
        lambda csv: f'CUDA_VISIBLE_DEVICES="{csv}"',
        lambda csv: f"nvidia-smi --id={csv}",
        lambda csv: f"runner --gpus {csv}",
        lambda csv: f'"gpu_policy": {{"indices": [{csv}]}}',
        lambda csv: f"GPU indices: {csv}",
        lambda csv: f"GPU编号：{csv}",
        lambda csv: f"显卡编号={csv}",
    ),
)
def test_gpu_policy_guard_matches_two_gpu_execution_contexts(
    render: Callable[[str], str],
) -> None:
    leaked_policy = render(_synthetic_two_gpu_csv())

    assert any(pattern.search(leaked_policy) for pattern in _TRACKED_GPU_POLICY_DISCLOSURES)


@pytest.mark.parametrize(
    "audit_text",
    (
        "GPU count is 2; four rounds remain pending.",
        "The GPU audit contains 101 checks and 103 assertions.",
        "显卡审计记录了两次独立准入检查。",
    ),
)
def test_two_gpu_guard_does_not_match_unlabelled_audit_numbers(audit_text: str) -> None:
    assert not any(
        pattern.search(audit_text) for pattern in _TRACKED_GPU_POLICY_DISCLOSURES
    )


def test_p6_boundary_scope_includes_adjacent_non_p6_paragraphs() -> None:
    scoped = _p6_boundary_sections(
        "P6 status is pending.\n\nhost: private-worker\n\nUnrelated public text."
    )

    with pytest.raises(AssertionError):
        _assert_p6_public_disclosure_safe({"P6 boundary": scoped})


def test_p6_framework_search_space_gate_is_documented() -> None:
    manifest = (REPOSITORY_ROOT / "docs/release-manifests/P6_H800_SEARCH_EXECUTION.md").read_text(encoding="utf-8")
    assert "从 Stage2 动态派生候选池" in manifest
    assert "不预设为 343 或 686" in manifest
    assert "P6.2 动态框架候选空间的离线接入与验证已完成" in manifest
    assert "P6.3 历史执行适配器已实现并通过离线验证" in manifest
    assert "真实 H800 框架来源四轮执行仍待本地运行" in manifest
