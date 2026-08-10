# P4 完整外部资源登记实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 P3 转交给 P4 的 506 条候选在本地按真实外部资源去重，编译出完整、脱敏、离线可验证的公开资源注册表和覆盖摘要，并恢复 Python 3.10/3.11 CI 质量门。

**Architecture:** 受限 P3 复核 JSON 与本地 resolution JSON 始终作为显式的本机输入；候选提取器仅生成本地草案，注册表编译器要求每条 P4 候选都被标为 `document` 或关联到一个公开资源。编译器只输出现有 registry 格式和安全的聚合覆盖摘要，现有验证器继续消费 registry。匿名归档 allowlist 的递归模式先修复，使 CI 的两个 Python 版本一致。

**Tech Stack:** Python 3.10--3.13 标准库、pytest、pytest-cov、Ruff、GitHub Actions。

## Global Constraints

- 只登记数据集、模型、checkpoint、ONNX、engine、测量/证据包和可再生成大文件；纯文档 P4 候选必须标为 `document`，不能生成资产记录。
- 受限 P3 文件和本地 resolution 不进 Git；公开文件、测试 fixture、审计正文与终端摘要不得含私有路径、候选 ID、原文、受限来源细节或 HMAC key。
- 所有 CLI 都要求显式输入/输出路径；不得下载、发现、运行或迁入外部资产，也不得启动 GPU、训练、评测或硬件任务。
- registry 中 `source`、许可结论、版本、用途、相对路径和可用性必须为事实性字段；事实未公开时使用明确的 unavailable 结论，`sha256` 保持可选。
- P4 只有在 506 条候选均已裁决、公开输出一致、Python 3.10/3.11 CI 绿色后才能改为“已完成（本地）”。
- 不新增强制 checksum、文件大小、TOCTOU、特殊文件或网络下载防护。

---

### Task 1: 修复匿名归档在 Python 3.10/3.11 的递归匹配

**Files:**

- Modify: `tools/release/anonymous_allowlist.txt:10-22`
- Modify: `tests/integration/test_anonymous_archive.py`

**Interfaces:**

- Consumes: allowlist 中的 `directory/**/*` 文件递归模式。
- Produces: 在 Python 3.10--3.13 下都能让 `build_archive()` 选中 allowlisted 文件的清单。

- [ ] **Step 1: 写出跨版本模式回归测试**

在 `tests/integration/test_anonymous_archive.py` 的 `anonymous_repo` fixture 后加入：

```python
def test_anonymous_allowlist_uses_file_recursive_patterns() -> None:
    entries = {
        line.strip()
        for line in (BUILDER.parent / "anonymous_allowlist.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert {"framework/**/*", "scripts/reproduce/**/*", "data/**/*", "artifacts/**/*", "tests/**/*", "tools/release/**/*"} <= entries
    assert not any(entry.endswith("/**") for entry in entries)
```

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
python -m pytest -q tests/integration/test_anonymous_archive.py::test_anonymous_allowlist_uses_file_recursive_patterns
```

Expected: FAIL，因为当前清单使用 `directory/**`。

- [ ] **Step 3: 使用显式文件递归模式**

将以下清单项逐一改为带 `/*` 的递归文件模式，并保留其他单文件条目不变：

```text
framework/**/*
scripts/reproduce/**/*
data/**/*
artifacts/**/*
tests/**/*
tools/release/**/*
```

更新 `anonymous_repo` fixture 中写入的 allowlist 内容为 `framework/**/*`，使 fixture 与真实格式一致。

- [ ] **Step 4: 验证支持的解释器行为**

Run:

```bash
python -m pytest -q tests/integration/test_anonymous_archive.py
task_tmp="$(mktemp -d)"
python3.10 -m venv "$task_tmp/venv"
"$task_tmp/venv/bin/pip" install 'pytest>=9,<10'
"$task_tmp/venv/bin/python" -m pytest -q tests/integration/test_anonymous_archive.py
```

Expected: 两次均通过；Python 3.10 不再出现 `unsafe input: required allowlist entries are missing`。

- [ ] **Step 5: 提交任务**

```bash
git add tools/release/anonymous_allowlist.txt tests/integration/test_anonymous_archive.py
git commit -m "fix: support archive allowlists on Python 3.10"
```

### Task 2: 提取本地 P4 候选草案

**Files:**

- Create: `tools/release/extract_p4_resource_candidates.py`
- Create: `tests/release/test_p4_resource_candidate_extraction.py`

**Interfaces:**

- Consumes: 显式 `--p3-review` 的受限 JSON，其顶层含 `records`，每条记录含 `origin`、`path`、`disposition`、`actual_role`、`inputs`、`outputs`、`evidence_artifact` 与 `reason`。
- Produces: 仅写入调用者指定的本地 `--output` 草案，格式 `aaai27_p4_private_candidate_draft_v1`；每条草案含私有 `origin`/`path`、`suggested_decision`、`suggested_asset_kinds` 与匹配到的安全类别词。

- [ ] **Step 1: 写出合成 P3 草案测试**

创建 `tests/release/test_p4_resource_candidate_extraction.py`，使用如下合成记录 helper：

```python
def _record(path: str, disposition: str, text: str) -> dict[str, object]:
    return {
        "origin": "tracked",
        "path": path,
        "disposition": disposition,
        "actual_role": text,
        "inputs": text,
        "outputs": text,
        "evidence_artifact": text,
        "reason": "reviewed",
    }
```

加入测试：

```python
def test_extract_p4_candidates_classifies_assets_documents_and_ambiguities(tmp_path: Path) -> None:
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"records": [
        _record("private/checkpoint.md", "external_contract_p4", "checkpoint input"),
        _record("private/paper.md", "external_contract_p4", "paper method discussion"),
        _record("private/mixed.md", "external_contract_p4", "dataset and ONNX export"),
        _record("private/p6.md", "execution_contract_p6", "checkpoint input"),
    ]}), encoding="utf-8")

    draft = _module().extract_candidate_draft(review)

    assert [item.suggested_decision for item in draft] == ["asset", "document", "ambiguous"]
    assert draft[0].suggested_asset_kinds == ("checkpoint",)
    assert draft[2].suggested_asset_kinds == ("dataset", "onnx")
```

再加入 CLI 测试：捕获 stdout 后断言它包含 `p4 candidate draft created`，但不包含合成的 `private/checkpoint.md`；读取指定输出文件后确认该私有定位只存在于本地草案。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
python -m pytest -q tests/release/test_p4_resource_candidate_extraction.py
```

Expected: FAIL，因为提取器不存在。

- [ ] **Step 3: 实现保守的本地草案提取器**

在 `tools/release/extract_p4_resource_candidates.py` 定义：

```python
P4_DISPOSITION = "external_contract_p4"
PRIVATE_DRAFT_FORMAT = "aaai27_p4_private_candidate_draft_v1"

ASSET_TERMS = {
    "dataset": ("dataset", "data set", "数据集"),
    "model": ("model", "模型"),
    "checkpoint": ("checkpoint", "权重"),
    "onnx": ("onnx",),
    "engine": ("engine", "tensorrt", "tvm"),
    "measurement_bundle": ("latency", "energy", "telemetry", "measurement", "测量"),
    "evidence_bundle": ("evidence", "ap report", "证据"),
    "regenerable_large_file": ("lut", "cache", "artifact", "制品"),
}
```

实现 `CandidateDraft` 冻结 dataclass、`load_p4_records(path)`、`suggest_candidate(record)`、`extract_candidate_draft(path)`、`render_private_draft(draft)` 与 `main(argv)`。`suggest_candidate` 将五个文本字段小写拼接：没有资产类别时返回 `document`；恰好一个类别时返回 `asset`；多个类别返回 `ambiguous`。CLI 只接受 `--p3-review` 与 `--output`，并只打印记录数和各建议类别的计数。

- [ ] **Step 4: 验证 GREEN 和输出约束**

Run:

```bash
python -m pytest -q tests/release/test_p4_resource_candidate_extraction.py
python -m ruff check tools/release/extract_p4_resource_candidates.py tests/release/test_p4_resource_candidate_extraction.py
```

Expected: 所有测试通过，CLI 终端摘要不回显候选定位。

- [ ] **Step 5: 提交任务**

```bash
git add tools/release/extract_p4_resource_candidates.py tests/release/test_p4_resource_candidate_extraction.py
git commit -m "feat: extract local P4 resource candidates"
```

### Task 3: 编译脱敏 registry 与公开覆盖摘要

**Files:**

- Modify: `tools/release/validate_external_inputs.py`
- Create: `tools/release/compile_p4_external_registry.py`
- Create: `tests/release/test_compile_p4_external_registry.py`
- Modify: `tests/release/test_external_input_registry.py`

**Interfaces:**

- Consumes: `--p3-review`、`--resolution`、`--registry-output` 和 `--coverage-output`。
- Resolution format: `aaai27_p4_private_resource_resolution_v1`，含 `decisions` 与 `resources`。每个 decision 为 `{ "origin": str, "path": str, "decision": "document" }` 或 `{ "origin": str, "path": str, "decision": "asset", "resource_id": str }`；resources 是完整的公开 registry records。
- Produces: `aaai27_external_input_registry_v1` registry，以及 `aaai27_p4_external_resource_coverage_v1` 覆盖摘要。

- [ ] **Step 1: 让现有 registry 解析可复用**

在 `tests/release/test_external_input_registry.py` 加入：

```python
def test_parse_registry_document_accepts_the_public_shape() -> None:
    document = {"format": "aaai27_external_input_registry_v1", "registry_version": 1, "inputs": [_record()]}

    parsed = _module().parse_registry_document(document)

    assert parsed[0].input_id == "stage6-terminal-evidence"
```

把 `load_registry()` 中 document 结构与唯一 ID 检查提取为：

```python
def parse_registry_document(document: object) -> tuple[ExternalInput, ...]:
    if not isinstance(document, Mapping):
        raise RegistryError("registry must be an object")
    if document.get("format") != REGISTRY_FORMAT or document.get("registry_version") != 1:
        raise RegistryError("registry header is invalid")
    records = document.get("inputs")
    if not isinstance(records, list):
        raise RegistryError("inputs must be a list")
    inputs = tuple(_parse_record(record) for record in records)
    if len({item.input_id for item in inputs}) != len(inputs):
        raise RegistryError("input_id values must be unique")
    return inputs
```

`load_registry()` 只负责读取 JSON 后调用它。保持现有 CLI 行为不变。

- [ ] **Step 2: 写编译器的 RED 测试**

在 `tests/release/test_compile_p4_external_registry.py` 用两个 asset 候选、一个 document 候选和一个非 P4 候选构造合成 review；resolution 让两个 asset 候选都引用 `stage6-terminal-evidence`。核心断言：

```python
registry, coverage = module.compile_registry(review_path, resolution_path)

assert [item["input_id"] for item in registry["inputs"]] == ["stage6-terminal-evidence"]
assert coverage == {
    "format": "aaai27_p4_external_resource_coverage_v1",
    "p3_p4_candidate_count": 3,
    "document_candidate_count": 1,
    "asset_candidate_count": 2,
    "distinct_resource_count": 1,
    "asset_kind_counts": [{"asset_kind": "evidence_bundle", "count": 1}],
    "availability_counts": [{"availability": "unavailable", "count": 1}],
}
assert "private/one.md" not in json.dumps(registry)
assert "private/one.md" not in json.dumps(coverage)
assert "candidate_id" not in json.dumps(registry)
```

再加入两个失败案例：缺少一个 P4 decision 必须抛出 `CompilationError`；resources 中未被任何 asset decision 引用必须抛出 `CompilationError`。加入 CLI 测试，确认失败时两个公开输出路径均不存在，成功时 stdout 不含合成私有路径。

- [ ] **Step 3: 运行 RED 测试**

Run:

```bash
python -m pytest -q tests/release/test_external_input_registry.py tests/release/test_compile_p4_external_registry.py
```

Expected: FAIL，因为 `parse_registry_document` 与编译器不存在。

- [ ] **Step 4: 实现编译器**

在 `tools/release/compile_p4_external_registry.py` 定义：

```python
RESOLUTION_FORMAT = "aaai27_p4_private_resource_resolution_v1"
COVERAGE_FORMAT = "aaai27_p4_external_resource_coverage_v1"

class CompilationError(ValueError):
    pass
```

实现顺序固定如下：

1. 读取 P3 JSON，只保留 `disposition == "external_contract_p4"` 的记录，并以私有 `(origin, path)` 作为进程内键；
2. 解析 resolution，拒绝重复 decision、非 P4 decision、缺少 decision、未知 `resource_id`、未使用资源和不完整的 `document`/`asset` decision；
3. 用每个 resource 的公开字段构建 registry document，调用 `parse_registry_document()` 验证，按 `input_id` 排序；
4. 仅从数量和资源的公开 `asset_kind`/`availability` 构建 coverage，并按类别名称排序；
5. 全部校验完成后才由 CLI 写两个 JSON 文件。stdout 只打印 P4 总候选数、document 数、asset 数和去重资源数。

resolution 的私有 key 不得加入 registry 的 `consumer_ids`。`consumer_ids` 只能来自 resources 中已经公开的逻辑消费者名称，或为空数组。

- [ ] **Step 5: 验证 GREEN**

Run:

```bash
python -m pytest -q tests/release/test_external_input_registry.py tests/release/test_compile_p4_external_registry.py
python -m ruff check tools/release/validate_external_inputs.py tools/release/compile_p4_external_registry.py tests/release/test_compile_p4_external_registry.py
python -m compileall -q tools/release
```

Expected: 全部通过；失败编译不会创建部分公开输出。

- [ ] **Step 6: 提交任务**

```bash
git add tools/release/validate_external_inputs.py tools/release/compile_p4_external_registry.py tests/release/test_external_input_registry.py tests/release/test_compile_p4_external_registry.py
git commit -m "feat: compile complete P4 resource registry"
```

### Task 4: 在本地裁决真实 P3 映射并生成公开产物

**Files:**

- Modify: `artifacts/external/registry.json`
- Create: `artifacts/external/coverage.json`
- Modify: `docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md`
- Modify: `tests/release/test_project_handoff.py`

**Interfaces:**

- Consumes: 操作员显式提供的受限 P3 review 和不入库 resolution JSON。
- Produces: 完整 `registry.json`、安全 `coverage.json` 和以 registry 为权威来源的 P4 文档。

- [ ] **Step 1: 写公开产物的 RED 回归测试**

将 `test_p4_contract_is_discoverable_and_starts_unavailable` 改名为 `test_p4_contract_is_discoverable_and_has_complete_coverage_summary`，使用以下断言：

```python
coverage = json.loads((REPOSITORY_ROOT / "artifacts/external/coverage.json").read_text(encoding="utf-8"))
registry = json.loads((REPOSITORY_ROOT / "artifacts/external/registry.json").read_text(encoding="utf-8"))

assert coverage["format"] == "aaai27_p4_external_resource_coverage_v1"
assert coverage["p3_p4_candidate_count"] == 506
assert coverage["document_candidate_count"] + coverage["asset_candidate_count"] == 506
assert coverage["distinct_resource_count"] == len(registry["inputs"])
assert {item["input_id"] for item in registry["inputs"]} >= {"stage6-terminal-evidence", "stage7-formal-aggregate"}
assert all("candidate_id" not in item and "origin" not in item for item in registry["inputs"])
```

保留 registry parser 的字段验证；删除“恰好两条记录”的断言。

- [ ] **Step 2: 运行 RED 测试**

Run:

```bash
python -m pytest -q tests/release/test_project_handoff.py::test_p4_contract_is_discoverable_and_has_complete_coverage_summary
```

Expected: FAIL，因为 `coverage.json` 尚不存在。

- [ ] **Step 3: 生成并裁决本地映射**

在仓库外的受限工作目录运行：

```bash
python tools/release/extract_p4_resource_candidates.py \
  --p3-review /operator-supplied/p3_full_item_revalidation.json \
  --output /operator-supplied/p4-candidate-draft.json
```

以草案的每条候选为唯一工作清单。建议为 `document` 且没有资源类别的候选写入 `{origin, path, decision: "document"}`；建议为 `asset` 或 `ambiguous` 的候选，必要时仅检查其对应的受限局部字段，写入 `{origin, path, decision: "asset", resource_id}`。为同一真实资源复用一个 `resource_id`。

resolution 的 `resources` 采用如下完整公开 record 形状；未公开的事实必须如实写为 unavailable，而不能虚构 URL、许可、版本或 checksum：

```json
{
  "input_id": "p4-evidence-bundle-001",
  "asset_kind": "evidence_bundle",
  "source": "not-publicly-available",
  "license": {"status": "unconfirmed", "reference": "provider-terms-required"},
  "version": "unreleased",
  "relative_path": "p4/evidence_bundle/001",
  "intended_use": "p4_external_contract",
  "consumer_ids": [],
  "availability": "unavailable",
  "unavailable_reason": "source_not_publicly_available"
}
```

对已公开且有事实依据的资源，以真实的安全公开值替代以上 unavailable 字段。resolution 完成前，运行一个本地汇总脚本只输出 `document`、`asset`、`ambiguous`、资源数和缺失 decision 数；不得把草案或 resolution 加入 Git。

- [ ] **Step 4: 编译真实公开 registry 与覆盖摘要**

Run:

```bash
python tools/release/compile_p4_external_registry.py \
  --p3-review /operator-supplied/p3_full_item_revalidation.json \
  --resolution /operator-supplied/p4-resource-resolution.json \
  --registry-output artifacts/external/registry.json \
  --coverage-output artifacts/external/coverage.json
python -m pytest -q tests/release/test_project_handoff.py
```

Expected: 编译器的安全摘要显示 P4 总数 506、缺失 decision 为 0；公开回归通过。

- [ ] **Step 5: 更新 P4 契约文档**

修改 `P4_EXTERNAL_INPUT_CONTRACT.md`：将“两项初始记录”表述替换为“`registry.json` 是完整资源目录、`coverage.json` 是完整性摘要”；保留离线命令与不下载/不运行保证；明确 document 候选不作为资产记录、资源去重在本机受限 resolution 中完成、公开产物不包含候选标识或私有定位。文档只引用 coverage 中的聚合数量，不逐项复制 registry。

- [ ] **Step 6: 检查公开差异并提交**

Run:

```bash
git diff --check
git diff -- artifacts/external/ docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md tests/release/test_project_handoff.py
git status --short
```

确认暂存文件仅含公开 registry、coverage、文档和测试后运行：

```bash
git add artifacts/external/registry.json artifacts/external/coverage.json docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md tests/release/test_project_handoff.py
git commit -m "feat: register complete P4 external resources"
```

### Task 5: 记录 P4 完整性证据并完成本地验证

**Files:**

- Modify: `docs/AAAI27_RELEASE_AUDIT.md:157`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md` 的变更记录
- Modify: `tests/release/test_project_handoff.py`

**Interfaces:**

- Consumes: 已编译 registry/coverage、全套本地验证结果和 CI 状态。
- Produces: 真实反映 P4 本地状态的审计记录；远端 CI 未绿色前不宣称 P4 完成。

- [ ] **Step 1: 写审计状态 RED 测试**

在 `tests/release/test_project_handoff.py` 添加：

```python
def test_p4_audit_links_complete_registry_and_coverage() -> None:
    handoff = HANDOFF.read_text(encoding="utf-8")

    assert "P4_EXTERNAL_INPUT_CONTRACT.md" in handoff
    assert "artifacts/external/coverage.json" in handoff
    assert "506" in handoff
```

- [ ] **Step 2: 运行 RED 测试**

Run:

```bash
python -m pytest -q tests/release/test_project_handoff.py::test_p4_audit_links_complete_registry_and_coverage
```

Expected: FAIL，因为当前审计只记录初始两条记录。

- [ ] **Step 3: 记录真实本地进度**

更新 P4 表格和变更记录，写入 registry/coverage 位置、506 条 P3 候选已完成本地 `document`/`asset` 裁决、去重资源数、unavailable 数和本地测试结果。状态保持 `进行中（本地）`，直到远端 Python 3.10/3.11 CI 均成功；不得记录任何私有路径、候选 ID、原文、外部下载或真实执行。

- [ ] **Step 4: 运行完整本地门禁**

Run:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' python -m pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
python -m ruff check framework scripts tests tools
python -m compileall -q framework scripts tools
python -m pytest -q tests/release tests/integration/test_public_clean_clone.py tests/integration/test_anonymous_archive.py
```

Expected: 全部通过；全局覆盖率不低于 80%。

- [ ] **Step 5: 提交本地证据**

```bash
git add docs/AAAI27_RELEASE_AUDIT.md tests/release/test_project_handoff.py
git commit -m "docs: record complete P4 registry evidence"
```

### Task 6: 在授权推送后完成 P4 收口

**Files:**

- Modify: `docs/AAAI27_RELEASE_AUDIT.md:157`
- Modify: `docs/AAAI27_RELEASE_AUDIT.md` 的变更记录
- Modify: `tests/release/test_project_handoff.py`

**Interfaces:**

- Consumes: 已推送 release 分支对应提交的 GitHub Actions `quality (3.10)`、`quality (3.11)` 和 `public-smoke` 成功结果。
- Produces: P4 的“已完成（本地）”审计状态；不合并 `main`，不触发 P5/P6/P7/P8。

- [ ] **Step 1: 在取得推送授权后推送 release 分支**

Run only after explicit user authorization:

```bash
git push origin release/aaai27-reproducibility
```

Expected: GitHub Actions 启动 Python 3.10、3.11 和 public-smoke 作业。

- [ ] **Step 2: 确认远端 CI 绿色**

在 CI 运行页确认以下全部通过：

```text
quality (3.10)
quality (3.11)
public-smoke
```

任一失败时保留 P4 `进行中（本地）`，记录失败原因并回到对应任务；不得以本地 Python 3.13 成功替代 CI 结果。

- [ ] **Step 3: 写 P4 收口 RED 测试**

在 `tests/release/test_project_handoff.py` 添加：

```python
def test_p4_is_closed_only_after_complete_registry_evidence() -> None:
    handoff = HANDOFF.read_text(encoding="utf-8")

    assert "| P4 | 已完成（本地） |" in handoff
    assert "quality (3.10)" in handoff
    assert "quality (3.11)" in handoff
    assert "public-smoke" in handoff
```

- [ ] **Step 4: 更新审计并验证 GREEN**

将 P4 状态更新为 `已完成（本地）`，记录本次 registry/coverage 完整性与三项 CI 成功的运行链接或编号；保留“不下载、不运行真实资产”的边界。然后运行：

```bash
python -m pytest -q tests/release/test_project_handoff.py
git diff --check
git commit -am "docs: close P4 external resource registry"
```

- [ ] **Step 5: 推送最终审计提交并确认新 CI 仍绿色**

在已获授权的前提下运行：

```bash
git push origin release/aaai27-reproducibility
```

Expected: 新提交的三项 CI 再次通过。此步骤不与 `main` 合并。
