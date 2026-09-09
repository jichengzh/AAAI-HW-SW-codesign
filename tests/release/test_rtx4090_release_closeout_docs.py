"""Release documentation contracts for the completed private RTX4090 flow."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.release.test_project_handoff import _assert_p6_public_disclosure_safe


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HANDOFF = REPOSITORY_ROOT / "docs/AAAI27_RELEASE_AUDIT.md"
PUBLIC_EXAMPLES = {
    "configs/execution/p6_rtx4090_search.example.yaml",
    "configs/execution/p6_external_training_binding.example.yaml",
}
NORMALIZED_PRIVATE_OUTPUTS = (
    "runner-template.yaml",
    "source-wrapper-profile.yaml",
    "external-training-binding.yaml",
    "post-source-adapter-profile.yaml",
)


@pytest.mark.parametrize(
    ("readme_name", "required_language_markers"),
    (
        (
            "README.md",
            (
                "User-provided private inputs",
                "private history root",
                "private source-map",
                "pre-normalization runner-template",
                "RTX local config or locator",
                "licensed datasets",
                "model sources/checkpoints",
                "ONNX or calibration inputs",
                "CUDA/TVM sm89 toolchain",
                "fresh output root",
                "Repository-provided public references",
                "null-only schema example",
                "`normalize`-generated private outputs",
            ),
        ),
        (
            "README.zh-CN.md",
            (
                "用户提供的私有输入",
                "私有 history root",
                "私有 source-map",
                "pre-normalization runner-template",
                "RTX local config 或 locator",
                "授权的数据集",
                "模型源码/checkpoint",
                "ONNX 或 calibration 输入",
                "CUDA/TVM sm89 工具链",
                "fresh output root",
                "仓库提供的公开参考",
                "仅空值 schema 示例",
                "`normalize` 自动生成的私有输出",
            ),
        ),
    ),
)
def test_bilingual_readmes_assign_every_rtx4090_material_to_its_owner(
    readme_name: str,
    required_language_markers: tuple[str, ...],
) -> None:
    """A newcomer can distinguish supplied inputs from generated private state."""
    readme = (REPOSITORY_ROOT / readme_name).read_text(encoding="utf-8")

    for marker in required_language_markers:
        assert marker in readme
    for example_path in PUBLIC_EXAMPLES:
        assert f"`{example_path}`" in readme
    for filename in NORMALIZED_PRIVATE_OUTPUTS:
        assert f"<abs-normalized-private-dir>/{filename}" in readme


def test_every_public_example_referenced_by_the_readmes_exists_and_is_tracked() -> None:
    """README starter paths must resolve without publishing private templates."""
    readmes = {
        name: (REPOSITORY_ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "README.zh-CN.md")
    }
    reference_pattern = re.compile(r"`(configs/[^`\s]+\.example\.ya?ml)`")
    references_by_readme = {
        name: set(reference_pattern.findall(content))
        for name, content in readmes.items()
    }

    assert all(PUBLIC_EXAMPLES <= references for references in references_by_readme.values())
    assert references_by_readme["README.md"] == references_by_readme["README.zh-CN.md"]

    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    for relative_path in set().union(*references_by_readme.values()):
        assert (REPOSITORY_ROOT / relative_path).is_file(), relative_path
        assert relative_path in tracked, relative_path


def test_handoff_records_a61_as_sanitized_structural_completion() -> None:
    """A61 is complete without turning private execution evidence into public data."""
    handoff = HANDOFF.read_text(encoding="utf-8")
    heading = "**A61 RTX4090 三卡四轮结构验证完成（2026-09-09）**："
    assert heading in handoff
    a61_entry = handoff.split(heading, maxsplit=1)[1].split("\n\n", maxsplit=1)[0]

    for marker in (
        "`GPU_POOL=3`",
        "4/4 轮",
        "16 条 selected",
        "16 条 measured",
        "controller 完成",
        "独立 verifier 通过",
        "不能重标记为 H800 论文数值",
    ):
        assert marker in a61_entry

    current_state = handoff.split("### 当前状态", maxsplit=1)[1].split(
        "### 到完整开源的执行计划", maxsplit=1
    )[0]
    assert "A61" in current_state
    assert "RTX4090 三卡四轮" in current_state
    assert "独立 verifier 通过" in current_state
    assert "真实 RTX4090 四卡四轮执行仍待 GPU admission 满足" not in current_state

    _assert_p6_public_disclosure_safe({"A61 handoff entry": a61_entry})
    assert not re.search(r"(?:GPU|显卡)(?:\s+(?:index|编号|索引))?\s*[:=#]?\s*\d", a61_entry)

