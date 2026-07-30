# 受限复现制品统一说明实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用审稿人无需了解内部开发阶段的语言，统一说明当前公开制品的能力、限制及发表后完整开放计划。

**Architecture:** 主 README 和匿名 README 只使用面向论文读者的功能名称；技术性证据文档可以保留内部模块路径，但必须同时给出自然语言解释。四份文档共享同一事实边界，不声明轻量策略与论文算法等价，也不声明当前制品能够复现缺失的论文结果。

**Tech Stack:** Markdown、现有 Python/pytest 发布检查、ripgrep。

## Global Constraints

- 当前仓库是论文实现的受限复现子集，不是完整训练与硬件搜索代码。
- `predicted_frontier_diversity` 只能称为确定性轻量接口演示，不得称为 NSGA-II 的等价替代。
- 当前公开在线消融模块不能称为论文 Table 2 的完整复现。
- 主 README 和匿名 README 不使用 Stage 4/5/6/7 等内部阶段编号。
- 匿名文档不得包含作者身份、组织、本地绝对路径或开发历史。
- 完整实现、执行清单与实验制品计划在论文发表后公开。

---

### Task 1: 重写审稿人入口说明

**Files:**
- Modify: `README.md`
- Modify: `README.anonymous.md`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-07-30-scoped-reproducibility-disclosure-design.md`
- Produces: 不依赖内部阶段编号、可由首次接触项目的审稿人直接理解的范围说明

- [ ] **Step 1: 记录修改前的禁止表述检查**

运行：

```bash
rg -n "Stage[ -]?[4567]|NSGA|predicted_frontier_diversity|Table 2|complete implementation" README.md README.anonymous.md
```

预期：主 README 仍含内部阶段编号；匿名 README 尚未充分区分“实现模块”和“完整论文实现”。

- [ ] **Step 2: 修改主 README**

在开头将仓库定义为“受限复现制品”，并在末尾增加
`Scope of the released implementation`。该小节必须说明：

- 已提供确定性 CPU 冒烟流程、代价模型选择审计、候选选择/结果聚合接口；
- 未提供完整候选生成器、模型物化/训练、硬件调优与正式实验轨迹；
- 轻量选择策略只验证接口与数据流，不验证论文候选生成算法或论文数值；
- 完整代码与正式实验清单计划在发表后公开。

- [ ] **Step 3: 修改匿名 README**

将 “contains the implementation modules” 改为“contains a scoped subset of
validation, selection-interface, and aggregation modules”。补充一段明确说明
smoke 不能复现硬件结果、完整搜索算法或在线消融表。

- [ ] **Step 4: 检查审稿人入口**

运行：

```bash
rg -n "Stage[ -]?[4567]" README.md README.anonymous.md
rg -n "scoped|lightweight|not.*equival|publication|hardware" README.md README.anonymous.md
```

预期：第一个命令无匹配；第二个命令能够定位范围、非等价和发表后开放说明。

- [ ] **Step 5: 提交入口文档修改**

```bash
git add README.md README.anonymous.md
git commit -m "docs: clarify reviewer-facing release scope"
```

### Task 2: 对齐证据与制品边界文档

**Files:**
- Modify: `REPRODUCIBILITY.md`
- Modify: `ARTIFACTS.md`

**Interfaces:**
- Consumes: Task 1 中确定的审稿人入口口径
- Produces: 与入口文档一致、但保留必要模块路径和证据状态的技术说明

- [ ] **Step 1: 修改复现说明**

在 evidence matrix 后增加 `Method-to-artifact mapping`，明确：

- 冒烟流程中的轻量候选选择仅验证公开接口；
- 公开在线消融代码只提供 selection-only 与统计合同；
- 缺失的正式候选生成、硬件执行和完整消融轨迹保持 unavailable；
- 现有 verified 数值范围不因接口代码存在而扩大。

- [ ] **Step 2: 修改制品清单**

在 `What is deliberately absent` 中增加论文专用候选生成器、完整在线消融
流水线、正式轨迹和 terminal evidence。注明这些内容计划在发表后与执行
清单一并发布。

- [ ] **Step 3: 运行跨文档禁用声明检查**

运行：

```bash
rg -n -i "same effect|equivalent to NSGA|reproduces? Table 2|complete paper implementation is included" \
  README.md README.anonymous.md REPRODUCIBILITY.md ARTIFACTS.md
```

预期：无匹配。

- [ ] **Step 4: 运行跨文档必需声明检查**

运行：

```bash
rg -n -i "scoped|lightweight|publication|unavailable|selection-only" \
  README.md README.anonymous.md REPRODUCIBILITY.md ARTIFACTS.md
```

预期：四份文档均有与其读者层级相适应的范围或证据边界说明。

- [ ] **Step 5: 提交证据文档修改**

```bash
git add REPRODUCIBILITY.md ARTIFACTS.md
git commit -m "docs: align reproducibility evidence boundaries"
```

### Task 3: 发布级验收

**Files:**
- Test: `tests/release/test_identity_scan.py`
- Test: `tests/integration/test_anonymous_archive.py`
- Test: `tests/integration/test_reproduce_all.py`

**Interfaces:**
- Consumes: Tasks 1–2 的四份文档
- Produces: 文档链接、匿名边界、归档边界和冒烟入口的验证结果

- [ ] **Step 1: 运行 Markdown 与差异检查**

```bash
git diff --check HEAD~2..HEAD
rg -n "/home/|jichengzh|220243603|@seu" README.anonymous.md
```

预期：无空白错误；匿名 README 无身份或本地路径匹配。

- [ ] **Step 2: 运行发布与匿名归档测试**

```bash
pytest -q \
  tests/release/test_identity_scan.py \
  tests/integration/test_anonymous_archive.py \
  tests/integration/test_reproduce_all.py
```

预期：全部通过。

- [ ] **Step 3: 运行文档命令检查**

```bash
pytest -q tests/test_package_metadata.py
```

预期：全部通过。

- [ ] **Step 4: 审阅最终差异和状态**

```bash
git diff HEAD~2..HEAD -- README.md README.anonymous.md REPRODUCIBILITY.md ARTIFACTS.md
git status --short --branch
```

预期：只包含经批准的文档范围修改；工作树无意外文件。
