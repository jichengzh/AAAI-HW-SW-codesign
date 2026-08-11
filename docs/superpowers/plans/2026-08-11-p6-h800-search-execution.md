# P6 H800 自动搜索闭环实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在维护者已本地准备的 H800 资产上，以显式本地配置完成一次可审计的 Stage5 候选选择、训练/评测、反馈和汇总闭环；仓库仅产出脱敏、可公开的执行契约和结果摘要。

**Architecture:** 新增独立的 Stage6 本地执行控制器。受版本控制的公共 YAML 描述 H800 搜索预算、指标、随机种子与脱敏资产标签；被 Git 忽略的本地 YAML 提供资产路径、Stage5 输入、命令 argv 和本地产物目录。控制器复用 Stage5 的纯选择函数，每轮将选中的候选写入本地请求文件，按 argv（不用 shell）调用维护者配置的训练/评测命令，读取结构化结果并反馈给下一轮。它只写一份严格白名单化的公开 JSON 摘要；完整逐轮记录留在忽略的本地输出目录。

**Tech Stack:** Python 3.10+、PyYAML、现有 `framework.stage5.production_search_v1`、标准库 `subprocess`/`json`/`pathlib`、pytest、Ruff。

## Global Constraints

- P6 第一阶段的唯一执行目标是 `h800`。不得把 CPU 或 RTX 4090 作为 P6 论文执行替代，也不得把 Orin 结果混入本阶段的状态或摘要。
- 不下载模型、数据集、引擎、容器或其他外部资产；不做自动硬件发现，不调用 `nvidia-smi`。缺失本地资产、缺失命令或不符合契约时必须失败关闭。
- 本地资产绝对路径、命令 argv、主机信息、原始 stderr/stdout、checkpoint、逐轮候选和逐轮原始指标只能写入 `configs/local/p6_h800_search.local.yaml` 或 `outputs/p6-h800-search/`；二者都必须受 Git 忽略保护。
- 公开配置与 `p6_h800_search_summary_v1` 只能包含目标标签、代码版本、seed、配置标签、资产标签、版本/许可状态、计划/完成轮数、成功候选计数、状态和聚合指标。不得新增或公开 SHA-256、资产路径、候选 ID、命令、主机信息、日志、checkpoint 名称或资产字节。
- 复用 Stage5 既有 API 时可能会读取其已有的本地来源标识字段；P6 不新建、不计算、不向公开产物复制任何哈希字段。
- `completed` 仅表示显式发起的 H800 本地闭环按预算完成。它不表示 Orin 完成、远端 CI 通过、发布完成，或论文主张已经独立复现。
- 不修改既有 `framework/stage6/contracts_v1.py` 的硬件盲/in-memory 合约，也不放宽 `tests/stage6/test_stage6_contract_migration_integration.py` 对 `launch_allowed is False` 的断言。
- 自动化测试只能用临时目录、合成 Stage5 输入和假训练程序；真实 H800 作业只能在维护者明确准备了本地配置后，由显式 CLI 调用启动。

## Planned File Map

| 文件 | 责任 |
| --- | --- |
| `framework/stage6/h800_search_execution_v1.py` | 公共/本地契约解析、Stage5 闭环状态机、结构化结果校验与公开摘要生成。 |
| `tools/release/run_p6_h800_search.py` | 显式本地执行 CLI；只允许 argv 子进程调用并将失败转换为脱敏状态。 |
| `configs/execution/p6_h800_search.example.yaml` | 受版本控制的、可公开的 P6 契约样例；不含真实资产路径或运行命令。 |
| `configs/local/p6_h800_search.local.yaml` | 维护者本地输入位置；仅在工作副本创建，绝不提交。 |
| `.gitignore` | 精确忽略上述本地 YAML；复用现有 `/outputs/` 忽略规则保存完整运行记录。 |
| `tests/stage6/test_h800_search_execution.py` | 契约、循环、失败关闭、摘要脱敏的单元/集成测试。 |
| `tests/release/test_run_p6_h800_search.py` | 使用 `subprocess.run()` 的黑盒 CLI 测试，驱动临时假训练程序。 |
| `tests/release/test_project_handoff.py` | P6 文档状态与公开/本地边界回归检查。 |
| `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md` | 公开执行入口、允许字段、失败语义与实际 H800 验收说明。 |
| `docs/AAAI27_RELEASE_AUDIT.md` | 仅在真实 H800 闭环及核验完成后，将 P6 从“进行中（本地）”改为完成。 |

---

### Task 1: 建立 P6 双层环境契约和严格解析器

**Files:**

- Create: `framework/stage6/h800_search_execution_v1.py`
- Create: `configs/execution/p6_h800_search.example.yaml`
- Modify: `.gitignore`
- Create: `tests/stage6/test_h800_search_execution.py`

- [ ] **Step 1: 先写公共与本地配置加载的失败测试（RED）。**

  在 `tests/stage6/test_h800_search_execution.py` 建立最小 YAML helper，并覆盖以下边界：

  ```python
  def test_load_public_contract_requires_h800_and_public_fields(tmp_path: Path) -> None:
      path = _write_yaml(tmp_path / "contract.yaml", _public_contract(target="orin"))

      with pytest.raises(H800SearchContractError, match="target must be h800"):
          load_public_contract(path)


  def test_load_local_config_rejects_asset_label_mismatch(tmp_path: Path) -> None:
      contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
      path = _write_yaml(
          tmp_path / "local.yaml",
          _local_config(asset_label="unregistered"),
      )

      with pytest.raises(H800SearchContractError, match="asset labels"):
          load_local_config(path, contract)


  def test_load_local_config_rejects_shell_string_step(tmp_path: Path) -> None:
      contract = load_public_contract(_write_yaml(tmp_path / "contract.yaml", _public_contract()))
      path = _write_yaml(
          tmp_path / "local.yaml",
          _local_config(steps=[{"name": "evaluate", "argv": "python train.py"}]),
      )

      with pytest.raises(H800SearchContractError, match="argv"):
          load_local_config(path, contract)
  ```

  覆盖并固定：

  - 公共 schema 必须等于 `p6_h800_search_contract_v1`，`target` 必须为 `h800`；
  - `search_id`、`configuration_label`、`target_model`、正整数 `seed`、正整数 `max_rounds`、正整数 `batch_size`、已声明的公共指标及资产 `{label, version, license_status}` 都必需；
  - 本地 schema 必须等于 `p6_h800_search_local_v1`，本地 target/资产标签必须与公共契约精确一致；
  - 本地路径必须是绝对路径；`steps` 必须是非空、名称唯一的 argv 列表；`result_step` 必须引用最后一个步骤；禁止字符串 shell 命令和未知占位符；
  - 公共 YAML 不得包含路径、命令、主机、候选 ID、hash/SHA 字段；
  - 读取本地配置不访问资产、不发起子进程，也不创建输出。

- [ ] **Step 2: 运行目标测试，确认失败来自尚不存在的接口。**

  Run: `python -m pytest tests/stage6/test_h800_search_execution.py -q`

  Expected: 导入 `framework.stage6.h800_search_execution_v1` 或上述符号失败；不要为了让测试通过而弱化拒绝条件。

- [ ] **Step 3: 以不可变数据结构实现最小解析器（GREEN）。**

  在 `framework/stage6/h800_search_execution_v1.py` 定义：

  ```python
  class H800SearchContractError(ValueError):
      """Raised when a P6 public or local search contract is unsafe or incomplete."""


  @dataclass(frozen=True)
  class RegisteredAsset:
      label: str
      version: str
      license_status: str


  @dataclass(frozen=True)
  class PublicH800SearchContract:
      search_id: str
      target: str
      target_model: str
      seed: int
      max_rounds: int
      batch_size: int
      configuration_label: str
      assets: tuple[RegisteredAsset, ...]
      metric_names: tuple[str, ...]
      candidate_space_label: str


  @dataclass(frozen=True)
  class LocalExecutionStep:
      name: str
      argv: tuple[str, ...]


  @dataclass(frozen=True)
  class LocalH800SearchConfig:
      asset_paths: Mapping[str, Path]
      stage5_input_paths: Mapping[str, Path]
      steps: tuple[LocalExecutionStep, ...]
      result_step: str
      result_path_template: Path
      local_output_root: Path
  ```

  实现 `load_public_contract(path: Path) -> PublicH800SearchContract` 和 `load_local_config(path: Path, contract: PublicH800SearchContract) -> LocalH800SearchConfig`。两者均使用 `yaml.safe_load`，拒绝非 mapping 顶层对象、重复/空资产标签、非绝对本地路径与未知键。一个本地步骤包含稳定名称及其 argv，步骤可表示 train、evaluate、TVM 调优或 TensorRT 构建；`result_step` 必须是最后一个步骤并产出最终结果。允许的命令模板变量限定为完整 argv token：`{candidate_request}`, `{result_json}`, `{local_output_root}`。解析器只返回新对象，不修改 YAML dict。

  增加 `configs/execution/p6_h800_search.example.yaml`，使用确定的脱敏样例值（例如 `search_id: p6-h800-example`、`target: h800`、`target_model: pyramid`、`seed: 73`、`max_rounds: 2`、`batch_size: 2`），并用 `training-data`、`model-init`、`toolchain` 三个资产标签展示 `version`/`license_status` 字段；不得写入路径、命令或 hash。

  在 `.gitignore` 新增精确规则：

  ```gitignore
  /configs/local/p6_h800_search.local.yaml
  ```

- [ ] **Step 4: 运行 Task 1 测试并进行定向静态检查。**

  Run:

  ```bash
  python -m pytest tests/stage6/test_h800_search_execution.py -q
  ruff check framework/stage6/h800_search_execution_v1.py tests/stage6/test_h800_search_execution.py
  git check-ignore -q configs/local/p6_h800_search.local.yaml
  ```

  Expected: 测试与 Ruff 通过，`git check-ignore` 返回 0；示例公共 YAML 不被忽略。

- [ ] **Step 5: 提交契约基础。**

  ```bash
  git add framework/stage6/h800_search_execution_v1.py \
      configs/execution/p6_h800_search.example.yaml \
      tests/stage6/test_h800_search_execution.py .gitignore
  git commit -m "feat: add P6 H800 search contracts"
  ```

### Task 2: 用 Stage5 纯函数实现 H800 搜索反馈闭环

**Files:**

- Modify: `framework/stage6/h800_search_execution_v1.py`
- Modify: `tests/stage6/test_h800_search_execution.py`

- [ ] **Step 1: 先写两轮反馈闭环和失败关闭测试（RED）。**

  在单元测试中准备最小但真实结构的 Stage5 本地 JSON 输入：测量行包含 `model`、`source_contract_sha256`、`latency_ms`、`energy_j`、`ap30`、`ap50`、`ap70`、`training_source`、`terminal_status`；候选与图特征/能力 profile/closure 文件满足现有 `production_search_v1` API 的输入形状。

  使用写出结构化结果 JSON 的注入式 fake runner，而不是 GPU：

  ```python
  def test_run_h800_search_feeds_each_round_back_before_next_selection(tmp_path: Path) -> None:
      seen_requests: list[dict[str, object]] = []

      summary = run_h800_search(
          contract=_loaded_contract(tmp_path, max_rounds=2, batch_size=1),
          local=_loaded_local_config(tmp_path),
          code_revision="abc123",
          command_runner=_write_successful_measurement(seen_requests),
      )

      assert summary.status == "completed"
      assert summary.completed_rounds == 2
      assert len(seen_requests) == 2
      assert seen_requests[1]["feedback_count"] > seen_requests[0]["feedback_count"]
  ```

  再覆盖：命令非零退出、结果文件缺失、非 JSON 结果、候选 ID 不匹配、缺少任一公开指标、NaN/无限数值、Stage5 选择为空、写本地记录失败。每一种都断言：停止后续轮次、状态为 `failed`、只保留稳定的 `failure_code`、从不产生 `completed`。

- [ ] **Step 2: 运行闭环测试，确认接口尚未实现。**

  Run: `python -m pytest tests/stage6/test_h800_search_execution.py -q`

  Expected: `run_h800_search` 或 `H800SearchSummary` 缺失导致失败。

- [ ] **Step 3: 实现单向、可反馈的运行状态机（GREEN）。**

  在同一模块新增不可变结果类型和明确的注入边界：

  ```python
  @dataclass(frozen=True)
  class H800SearchSummary:
      schema: str
      target: str
      code_revision: str
      seed: int
      configuration_label: str
      assets: tuple[RegisteredAsset, ...]
      status: Literal["completed", "failed"]
      planned_rounds: int
      completed_rounds: int
      successful_candidate_count: int
      aggregate_metrics: Mapping[str, Mapping[str, float]]
      failure_code: str | None


  def run_h800_search(
      contract: PublicH800SearchContract,
      local: LocalH800SearchConfig,
      code_revision: str,
      command_runner: CommandRunner,
  ) -> H800SearchSummary:
      ...
  ```

  `CommandRunner` 是接收 `(argv: tuple[str, ...], cwd: Path)` 并返回进程返回码的窄协议。生产 CLI 传入封装 `subprocess.run(argv, cwd=cwd, check=False, shell=False)` 的实现；测试传入 fake runner。不得从控制器调用 shell、`os.system`、`nvidia-smi`、网络库或包下载器。

  每一轮严格按以下顺序执行：

  1. 在选择前确认每个资产和 Stage5 输入路径已存在且为预期文件类型；缺失时直接失败关闭。随后从本地输入读取当前已测量数据、候选 registry、图特征、能力 profiles 与 Stage4 closure；
  2. 调用 `fit_production_bundle`、`build_candidate_manifest`、`predict_candidate_rows` 与 `select_predicted_frontier_diversity`；不得改写旧的 `single_target_search_v2` 冻结 `(16, 4, 4)` 任务校验；
  3. 将当前批次的候选请求（含仅本地可见的候选 ID、轮数和 `feedback_count`）写到由 `round_index` 命名的本地目录中的 `candidate_request.json`；
  4. 对每个候选按声明顺序执行每个本地步骤；每次仅做完整 argv token 的安全占位符替换。`result_step` 成功后读取指定的本地结果 JSON，并把验证后的指标追加为新的不可变测量记录；
  5. 轮次成功后才增加 `completed_rounds`；下一轮必须基于该轮反馈重新拟合/选择；
  6. 任一失败以受控 `H800SearchExecutionError` 停止，写本地 `failure.json`（可以有详细本地错误），返回带稳定 `failure_code` 的失败摘要。

  使用数值有限性检查和精确指标键集检查。实际选择批次少于请求 batch 时，只要 Stage5 明确返回有效非空批次即可继续；累计成功候选数与实际测量数一致。

- [ ] **Step 4: 运行闭环测试、现有 Stage5/Stage6 回归。**

  Run:

  ```bash
  python -m pytest tests/stage6/test_h800_search_execution.py \
      tests/stage6/test_stage6_contract_migration_integration.py \
      tests/stage5/test_production_search.py -q
  ruff check framework/stage6/h800_search_execution_v1.py tests/stage6/test_h800_search_execution.py
  ```

  Expected: 新增两轮/失败关闭测试通过；既有 Stage6 硬件盲合约和 Stage5 选择测试不变。

- [ ] **Step 5: 提交控制器闭环。**

  ```bash
  git add framework/stage6/h800_search_execution_v1.py tests/stage6/test_h800_search_execution.py
  git commit -m "feat: run P6 H800 search feedback loop"
  ```

### Task 3: 生成白名单公开摘要并提供显式 CLI

**Files:**

- Modify: `framework/stage6/h800_search_execution_v1.py`
- Create: `tools/release/run_p6_h800_search.py`
- Modify: `tests/stage6/test_h800_search_execution.py`
- Create: `tests/release/test_run_p6_h800_search.py`

- [ ] **Step 1: 先写摘要脱敏和 CLI 黑盒测试（RED）。**

  单元测试把绝对路径、命令 token、候选 ID、checkpoint 文件名、stderr 文本和原始逐轮指标作为 sentinel 写入 local config/本地记录，然后断言 `summary_to_public_dict` 的 JSON 字符串不包含任一个 sentinel，且只含预定键。

  黑盒测试必须使用 `subprocess.run()` 启动 CLI，在 `tmp_path` 创建公共 contract、本地 contract、最小 Stage5 输入和一个临时 Python 假训练程序。假程序读取 `{candidate_request}` 并写 `{result_json}`，但不访问 GPU。至少覆盖：

  ```python
  result = subprocess.run(
      [
          sys.executable,
          "tools/release/run_p6_h800_search.py",
          "--contract", str(contract_path),
          "--local-config", str(local_path),
          "--public-summary", str(summary_path),
          "--code-revision", "test-revision",
      ],
      cwd=REPO_ROOT,
      text=True,
      capture_output=True,
      check=False,
  )
  assert result.returncode == 0, result.stderr
  ```

  再测试缺少 `--local-config`、本地配置无效和子命令失败：CLI 不能启动任何后续候选；若能安全写摘要，摘要必须是 `failed` 且没有 stderr/路径泄露。

- [ ] **Step 2: 运行这些测试，确认 CLI 和公开序列化尚不存在。**

  Run:

  ```bash
  python -m pytest tests/stage6/test_h800_search_execution.py \
      tests/release/test_run_p6_h800_search.py -q
  ```

  Expected: 缺少 CLI 或公开摘要函数造成失败。

- [ ] **Step 3: 实现固定字段摘要与唯一执行入口（GREEN）。**

  在控制器实现：

  ```python
  def summary_to_public_dict(summary: H800SearchSummary) -> dict[str, object]:
      return {
          "schema": "p6_h800_search_summary_v1",
          "target": summary.target,
          "code_revision": summary.code_revision,
          "seed": summary.seed,
          "configuration_label": summary.configuration_label,
          "assets": [
              {"label": asset.label, "version": asset.version, "license_status": asset.license_status}
              for asset in summary.assets
          ],
          "status": summary.status,
          "planned_rounds": summary.planned_rounds,
          "completed_rounds": summary.completed_rounds,
          "successful_candidate_count": summary.successful_candidate_count,
          "aggregate_metrics": summary.aggregate_metrics,
          "failure_code": summary.failure_code,
      }
  ```

  聚合指标仅以每个公开 metric 的 `count`、`min`、`max`、`mean` 组成；没有逐轮数组、候选名或原始样本。摘要先序列化到临时同目录文件，再以原子替换方式发布，避免半写入 JSON。

  创建 CLI：

  ```text
  python tools/release/run_p6_h800_search.py \
    --contract configs/execution/p6_h800_search.yaml \
    --local-config configs/local/p6_h800_search.local.yaml \
    --public-summary artifacts/p6/h800_search_summary.json \
    --code-revision "p6-test-revision"
  ```

  参数解析应使用 `allow_abbrev=False`。本地 YAML 是 `local_output_root` 的唯一权威；CLI 只在公共/本地契约都通过校验、该本地输出目录已明确且不冲突后调用控制器。正常完成返回 0；受控的执行失败返回 1；参数或契约错误返回 2。`--code-revision` 是调用者显式传入的非空标签，CLI 不自行执行 git 命令。错误 stdout/stderr 只输出稳定的错误类别，详细信息只留在忽略的本地记录。

- [ ] **Step 4: 运行 CLI、脱敏和静态回归。**

  Run:

  ```bash
  python -m pytest tests/stage6/test_h800_search_execution.py \
      tests/release/test_run_p6_h800_search.py -q
  ruff check framework/stage6/h800_search_execution_v1.py \
      tools/release/run_p6_h800_search.py \
      tests/stage6/test_h800_search_execution.py \
      tests/release/test_run_p6_h800_search.py
  python -m compileall -q framework/stage6/h800_search_execution_v1.py \
      tools/release/run_p6_h800_search.py
  ```

  Expected: 黑盒流程在无 GPU 情况下通过；不出现 shell 启动、硬件探测或公开泄露。

- [ ] **Step 5: 提交公开执行面。**

  ```bash
  git add framework/stage6/h800_search_execution_v1.py \
      tools/release/run_p6_h800_search.py \
      tests/stage6/test_h800_search_execution.py \
      tests/release/test_run_p6_h800_search.py
  git commit -m "feat: add P6 H800 search runner"
  ```

### Task 4: 固化公开边界、匿名归档与维护者运行说明

**Files:**

- Create: `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md`
- Modify: `tests/release/test_project_handoff.py`
- Modify: `tests/integration/test_anonymous_archive.py`
- Modify: `tools/release/anonymous_allowlist.txt` only if its current rules do not already include the new public config/runner/module paths

- [ ] **Step 1: 先写公开边界的回归测试（RED）。**

  在 `test_project_handoff.py` 断言 P6 审核表为“进行中（本地）”、准确描述 H800 首阶段和 Orin 后续独立线，且不把 CPU/RTX 4090 表示成 P6 目标。断言公开说明列出 `p6_h800_search_summary_v1` 的允许字段和禁止字段，并且没有 `/home/`、`checkpoint`、`SHA-256` 或真实运行命令。

  在匿名归档集成测试中从 fixture 仓库构建归档，验证：公共 runner、`framework/stage6/h800_search_execution_v1.py` 和示例公共 contract 可见；`configs/local/p6_h800_search.local.yaml`、`outputs/p6-h800-search/` 和任何 sentinel 原始日志均不可见。若既有 allowlist 已覆盖 `framework/**`、`tools/release/**`、`configs/**`，测试应明确确认它，不做无意义的 allowlist 改动。

- [ ] **Step 2: 运行回归测试，确认 P6 文档/归档保障尚不存在。**

  Run:

  ```bash
  python -m pytest tests/release/test_project_handoff.py \
      tests/integration/test_anonymous_archive.py -q
  ```

  Expected: P6 manifest/assertion 或 archive 期望失败。

- [ ] **Step 3: 编写公开运行说明并仅按需要调整 allowlist。**

  在 `docs/release-manifests/P6_H800_SEARCH_EXECUTION.md` 写明：

  - P6 仅接受明确 `--local-config` 的 H800 本地运行；不会下载、探测或自动启动硬件；
  - 公共/本地 YAML 的职责和本地文件准确位置；
  - 公共摘要的 schema、允许字段和稳定失败码；
  - Orin 不属于本次闭环；TVM/TensorRT 命令可以是本地训练/评测 argv 的维护者步骤，但其完整命令和日志不公开；
  - P6 完成的准入条件：实际 H800 运行按预算完成、公开摘要通过结构和泄露检查、全量测试/归档验证完成。

  只在归档测试证明缺少某个公共路径时修改 `anonymous_allowlist.txt`，使用最窄的精确规则；绝不放宽为 `configs/local/**` 或 `outputs/**`。

- [ ] **Step 4: 运行文档、归档与泄露检查。**

  Run:

  ```bash
  python -m pytest tests/release/test_project_handoff.py \
      tests/integration/test_anonymous_archive.py -q
  python tools/release/build_anonymous_archive.py \
      --repo-root . --output-dir /tmp/p6-anonymous-archive
  python tools/release/verify_anonymous_archive.py \
      --archive-dir /tmp/p6-anonymous-archive
  rg -n --glob '!docs/superpowers/**' '/home/|p6_h800_search\.local\.yaml|SHA-256' \
      docs/release-manifests/P6_H800_SEARCH_EXECUTION.md \
      configs/execution/p6_h800_search.example.yaml
  ```

  Expected: 测试与归档验证通过；最后一条搜索无匹配。临时归档目录只用于验证，验证后按环境的可恢复清理方式移除。

- [ ] **Step 5: 提交公开边界与说明。**

  ```bash
  git add docs/release-manifests/P6_H800_SEARCH_EXECUTION.md \
      tests/release/test_project_handoff.py \
      tests/integration/test_anonymous_archive.py \
      tools/release/anonymous_allowlist.txt
  git commit -m "docs: document P6 H800 execution boundary"
  ```

### Task 5: 用维护者已存在的 H800 资产执行实际验收并决定 P6 是否收口

**Files:**

- Create locally only: `configs/local/p6_h800_search.local.yaml`
- Create locally only: `outputs/p6-h800-search/`
- Create/Modify after review: `configs/execution/p6_h800_search.yaml`
- Create/Modify after review: `artifacts/p6/h800_search_summary.json`
- Modify only after all gates pass: `docs/AAAI27_RELEASE_AUDIT.md`
- Modify only after all gates pass: `tests/release/test_project_handoff.py`

- [ ] **Step 1: 对本地配置进行离线预检，不启动作业。**

  在已忽略的 `configs/local/p6_h800_search.local.yaml` 填入已存在资产的绝对路径、Stage5 输入 JSON 路径、训练/评测（可含维护者本地 TVM/TensorRT 步骤）argv 与结果 JSON 模板。它必须与待公开的 `configs/execution/p6_h800_search.yaml` 中的 H800、模型、seed、预算、资产标签/版本/许可状态完全一致。

  创建公共 contract 时，只由维护者确认后的脱敏标签和配置组成；不复制本地路径、命令、候选、哈希或运行日志。预检应调用 CLI 的契约加载路径或专用 `--validate-only` 模式（若 Task 3 实现），并验证：本地文件被忽略、输出根目录被忽略、公共 contract 不含禁止字段。预检失败时修正配置，不执行训练。

- [ ] **Step 2: 先以已批准的本地配置运行完整 H800 预算。**

  在目标 H800 维护者环境中明确执行：

  ```bash
  python tools/release/run_p6_h800_search.py \
    --contract configs/execution/p6_h800_search.yaml \
    --local-config configs/local/p6_h800_search.local.yaml \
    --public-summary artifacts/p6/h800_search_summary.json \
    --code-revision "$(git rev-parse HEAD)"
  ```

  这是唯一允许启动命令的步骤。控制器必须运行到 `max_rounds`，或者在首个安全失败点停止；不得重下载、不替换为 CPU/RTX 4090、不得把 Orin 工作掺入本次运行。原始输出只保存在 `outputs/p6-h800-search/`。

- [ ] **Step 3: 用自动化校验审核实际公开摘要。**

  新增或执行结构校验，至少验证：`schema == "p6_h800_search_summary_v1"`、`target == "h800"`、`status == "completed"`、完成轮数等于计划轮数、成功候选数大于零、所有公开聚合指标有限、资产字段恰为 `label`/`version`/`license_status`，并且 JSON 不含本地 sentinel 类字段（`path`、`argv`、`command`、`hostname`、`stderr`、`checkpoint`、`candidate_id`、`sha`）。

  若状态为 `failed` 或任一校验未通过：保留 P6 “进行中（本地）”，只整理本地失败记录，修复后从 Step 1 重新开始；不得提交失败的原始记录或提前关闭审计。

- [ ] **Step 4: 执行完整仓库验证。**

  Run:

  ```bash
  python -m pytest -q --cov=framework --cov=scripts/reproduce \
      --cov-report=term-missing --cov-report=xml --cov-fail-under=80
  ruff check .
  python -m compileall -q framework tools scripts
  python tools/release/build_anonymous_archive.py \
      --repo-root . --output-dir /tmp/p6-final-anonymous-archive
  python tools/release/verify_anonymous_archive.py \
      --archive-dir /tmp/p6-final-anonymous-archive
  git status --short
  ```

  Expected: 测试通过并达到 80% 覆盖率，静态检查/编译/归档验证通过，工作区中除经审核可公开的 contract、摘要、文档和测试外不出现本地配置或原始输出。

- [ ] **Step 5: 仅在真实验收全部通过后收口 P6。**

  将 P6 审核表和变更记录更新为完成，明确措辞为“已完成 H800 首阶段闭环；Orin 仍为独立后续线”。同步收紧 `test_project_handoff.py` 的状态断言，复跑该测试及全量验证。随后提交：

  ```bash
  git add configs/execution/p6_h800_search.yaml \
      artifacts/p6/h800_search_summary.json \
      docs/AAAI27_RELEASE_AUDIT.md \
      tests/release/test_project_handoff.py
  git commit -m "docs: close P6 H800 search execution"
  ```

  此提交不合并到 `main`，也不自动推送；需在验证结果和工作树审阅后取得单独授权。

## Final Verification Checklist

- [ ] 每个新增功能均按 RED → GREEN → REFACTOR 执行，新增测试覆盖解析、两轮反馈、每类失败关闭、摘要脱敏和 CLI 黑盒调用。
- [ ] 新控制器仅复用 Stage5 纯选择函数，不改变既有 Stage6 的硬件盲合约。
- [ ] 本地命令仅接受 argv，未使用 shell；代码、文档和测试没有下载或硬件探测路径。
- [ ] `configs/local/p6_h800_search.local.yaml` 和 `outputs/p6-h800-search/` 已确认受 Git 忽略，匿名归档中不存在。
- [ ] 所有公开 P6 产物不含路径、命令、候选 ID、日志、checkpoint、逐轮原始结果、主机信息、资产字节或 SHA 字段。
- [ ] P6 只有在一次真实 H800 预算完整运行且摘要审计通过后才可改为完成；Orin 继续保持单独后续状态。
