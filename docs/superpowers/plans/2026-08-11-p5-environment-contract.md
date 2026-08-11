# P5 环境契约与离线验证器实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 为 CPU、RTX 4090 与 H800 提供以既有硬件 YAML 为权威来源的最小环境契约、脱敏 JSON 观测和纯离线验证器。

**Architecture:** framework.environment_contract 只解析显式给出的契约 YAML、其仓库内的 HardwareCapability YAML 和观测 JSON；核心比较函数返回不可变的验证结果。发布 CLI 仅负责参数、调用核心函数和写入不含路径或观测值的报告。三份能力 YAML 与三份薄环境契约保持分离，JSON 只能观察环境，不能定义或覆盖能力。

**Tech Stack:** Python 3.10--3.13、标准库、PyYAML、Pydantic v2、pytest、pytest-cov、Ruff、GitHub Actions。

## Global Constraints

- framework.capability_schema.HardwareCapability 是唯一的静态硬件能力权威模型；环境契约/JSON 不复制 IP、精度、对齐或设备能力。
- 仅接受显式 --contract、--observation、--output；不得读取环境变量、探测 GPU、调用 nvidia-smi、启动子进程、CUDA 编译、训练、评测、基准或网络下载。
- 公开目标仅为 cpu_reference、rtx4090、h800。CPU 必须 gpu_count: 0 且没有 CUDA/driver；GPU 必须 gpu_count >= 1 并声明 CUDA/driver。
- 版本约束只支持逗号连接的数值比较：>=、>、<=、< 和精确值；点分数字按零补齐比较。拒绝预发布标签、发行版/内核名称和无法解析的版本。
- 契约 YAML 拒绝未知字段；观测 JSON 可含不参与判定的普通备注字段。缺字段或不匹配均失败关闭。
- 报告只能包含 schema、target、passed、稳定失败 code 和 field；不得回显原始版本值、观测内容、输入路径、SSH/主机/用户标识、序列号或 GPU UUID。
- 不增加哈希、SHA-256、供应链或不需要的运行时防御。P5 仅证明离线契约验证，不证明真实硬件、训练或性能结果。
- 所有功能从 RED 开始；每项任务通过定向测试和 Ruff 后提交。仅在最后本地全量门禁通过后将 P5 标为“已完成（本地）”；不推送、合并或访问真实硬件。

---

### Task 1: 建立环境输入模型与严格加载边界

**Files:**

- Create: framework/environment_contract.py
- Create: tests/release/test_environment_contract.py

**Interfaces:**

- Consumes: 显式环境契约 YAML、显式 JSON 观测和调用方提供的 repository_root。
- Produces: EnvironmentContractError、冻结的 RuntimeConstraint、EnvironmentContract、RuntimeObservation、EnvironmentObservation；load_environment_contract(path, *, repository_root) 与 load_environment_observation(path)。
- Depends on: HardwareCapability.from_yaml() 的既有能力 YAML 验证。

- [ ] **Step 1: 写出加载器 RED 测试**

创建临时仓库 helper：它写一个最小有效能力 YAML（name、arch、ips.gpu.precisions）及匹配的契约/观测。写入如下测试及 CPU/GPU 互斥变体：

~~~
def test_load_environment_contract_accepts_only_declared_shape(tmp_path: Path) -> None:
    module = _module()
    root, contract_path = _write_contract_tree(tmp_path, target="rtx4090")

    contract = module.load_environment_contract(contract_path, repository_root=root)

    assert contract.target == "rtx4090"
    assert contract.requires_gpu is True
    assert contract.hardware_capability == Path("configs/hardware/rtx4090.yaml")


@pytest.mark.parametrize("mutate", [
    lambda value: value.update({"unexpected": True}),
    lambda value: value.update({"hardware_capability": "../outside.yaml"}),
    lambda value: value.update({"target": "unknown"}),
])
def test_load_environment_contract_rejects_invalid_public_shape(tmp_path: Path, mutate: object) -> None:
    module = _module()
    root, contract_path = _write_contract_tree(tmp_path, target="rtx4090")
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    mutate(document)
    contract_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(module.EnvironmentContractError):
        module.load_environment_contract(contract_path, repository_root=root)
~~~

补充 CPU 含 CUDA/driver、GPU 缺 CUDA/driver、能力 YAML 解析失败、观测 schema 错误、CPU gpu_count 不为 0 的失败断言，以及带额外 note 的成功观测。

- [ ] **Step 2: 运行测试确认 RED**

Run:

~~~
python -m pytest -q tests/release/test_environment_contract.py
~~~

Expected: FAIL，因为模块与加载接口不存在。

- [ ] **Step 3: 实现最小加载模型**

在 framework/environment_contract.py 定义以下冻结类型和函数：

~~~
class EnvironmentContractError(ValueError):
    """Raised when an explicit public environment input is invalid."""

@dataclass(frozen=True)
class RuntimeConstraint:
    python: str
    cuda: str | None
    driver: str | None
    framework_name: str
    framework_version: str

@dataclass(frozen=True)
class EnvironmentContract:
    target: str
    hardware_capability: Path
    requires_gpu: bool
    runtime: RuntimeConstraint

@dataclass(frozen=True)
class RuntimeObservation:
    python: str
    cuda: str | None
    driver: str | None
    framework_name: str
    framework_version: str

@dataclass(frozen=True)
class EnvironmentObservation:
    target: str
    runtime: RuntimeObservation
    gpu_count: int

# Public functions:
# load_environment_contract(path: Path, *, repository_root: Path) -> EnvironmentContract
# load_environment_observation(path: Path) -> EnvironmentObservation
~~~

使用 yaml.safe_load() 与 json.loads()；空文件、非对象、错误 schema、未知契约字段、错误布尔/整数和缺失字段均抛 EnvironmentContractError。能力路径必须非绝对、不含 ..、解析后留在 repository_root.resolve() 内，并通过 HardwareCapability.from_yaml()。观测丢弃额外备注字段，不保存原始 document。

- [ ] **Step 4: 运行 GREEN 与静态检查**

Run:

~~~
python -m pytest -q tests/release/test_environment_contract.py
python -m ruff check framework/environment_contract.py tests/release/test_environment_contract.py
~~~

Expected: 加载边界全通过，Ruff 无诊断。

- [ ] **Step 5: 提交任务**

~~~
git add framework/environment_contract.py tests/release/test_environment_contract.py
git commit -m "feat: load P5 environment contracts"
~~~

### Task 2: 实现纯版本比较和环境判定

**Files:**

- Modify: framework/environment_contract.py
- Modify: tests/release/test_environment_contract.py

**Interfaces:**

- Consumes: Task 1 的 EnvironmentContract 与 EnvironmentObservation。
- Produces: ValidationFailure、ValidationResult、version_satisfies(actual, expression)、validate_environment(contract, observation)。
- Failure codes: observation.target.mismatch、runtime.gpu_count.required、runtime.gpu_count.unexpected、runtime.python.out_of_range、runtime.cuda.out_of_range、runtime.driver.out_of_range、runtime.framework.name.mismatch、runtime.framework.version.out_of_range。

- [ ] **Step 1: 写出判定 RED 测试**

~~~
def test_version_satisfies_uses_numeric_zero_padded_parts() -> None:
    module = _module()

    assert module.version_satisfies("12.4", ">=12.0,<13")
    assert module.version_satisfies("12.4.0", "12.4")
    assert not module.version_satisfies("11.8", ">=12.0,<13")
    with pytest.raises(module.EnvironmentContractError, match="version"):
        module.version_satisfies("12.4rc1", ">=12.0")


def test_validate_environment_returns_stable_field_only_failures(tmp_path: Path) -> None:
    module = _module()
    contract = _load_rtx_contract(module, tmp_path)
    observation = _observation(module, cuda="11.8", gpu_count=0, target="h800")

    result = module.validate_environment(contract, observation)

    assert [(item.code, item.field) for item in result.failures] == [
        ("observation.target.mismatch", "target"),
        ("runtime.gpu_count.required", "gpu_count"),
        ("runtime.cuda.out_of_range", "runtime.cuda"),
    ]
~~~

再覆盖 CPU 的非零 GPU 数、框架名称不同、Python/CUDA/driver/framework 越界、完全满足，以及结果不携带观测值或路径。

- [ ] **Step 2: 运行测试确认 RED**

Run:

~~~
python -m pytest -q tests/release/test_environment_contract.py
~~~

Expected: FAIL，因为版本函数、结果类型和比较函数不存在。

- [ ] **Step 3: 实现纯函数**

~~~
@dataclass(frozen=True)
class ValidationFailure:
    code: str
    field: str

@dataclass(frozen=True)
class ValidationResult:
    target: str
    passed: bool
    failures: tuple[ValidationFailure, ...]

# Public functions:
# version_satisfies(actual: str, expression: str) -> bool
# validate_environment(contract: EnvironmentContract, observation: EnvironmentObservation) -> ValidationResult
~~~

版本字串仅接受 ^[0-9]+(?:\.[0-9]+)*$；每个逗号项只接受五种操作符，无法解析即抛错。按接口列出的固定顺序追加失败项，最终以 tuple(failures) 返回新结果，不读写文件、不修改入参。

- [ ] **Step 4: 运行 GREEN 与静态检查**

Run:

~~~
python -m pytest -q tests/release/test_environment_contract.py
python -m ruff check framework/environment_contract.py tests/release/test_environment_contract.py
~~~

Expected: 加载、版本和比较测试都通过，且无本机环境或 GPU 依赖。

- [ ] **Step 5: 提交任务**

~~~
git add framework/environment_contract.py tests/release/test_environment_contract.py
git commit -m "feat: validate P5 environment observations"
~~~

### Task 3: 接入脱敏报告 CLI 与稳定退出码

**Files:**

- Create: tools/release/validate_environment_contract.py
- Create: tests/release/test_validate_environment_contract.py

**Interfaces:**

- Consumes: --contract Path、--observation Path、--output Path。
- Produces: environment_contract_report_v1；通过 0，已解析但不满足 1，参数/文件/schema/契约/能力 YAML 无效 2。
- Depends on: Task 1--2 的加载和 validate_environment()。

- [ ] **Step 1: 写出 CLI RED 测试**

~~~
def test_cli_writes_redacted_failure_report_and_returns_one(tmp_path: Path) -> None:
    root, contract, observation = _write_rtx_inputs(tmp_path, cuda="11.8")
    output = tmp_path / "private-marker" / "report.json"

    result = _run_cli(contract, observation, output, cwd=root)

    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": "environment_contract_report_v1",
        "target": "rtx4090",
        "passed": False,
        "failures": [{"code": "runtime.cuda.out_of_range", "field": "runtime.cuda"}],
    }
    assert "private-marker" not in result.stdout + result.stderr
~~~

增加有效输入返回 0、schema 错误返回 2（报告为 target: null 和 input.invalid）及脚本源码不含 os.environ、subprocess、nvidia-smi、torch.cuda 的断言。

- [ ] **Step 2: 运行测试确认 RED**

Run:

~~~
python -m pytest -q tests/release/test_validate_environment_contract.py
~~~

Expected: FAIL，因为 CLI 不存在。

- [ ] **Step 3: 实现 CLI 与报告渲染**

~~~
REPORT_SCHEMA = "environment_contract_report_v1"

# Public functions:
# render_report(result: ValidationResult) -> dict[str, object]
# render_invalid_report() -> dict[str, object]
# main(argv: Sequence[str] | None = None) -> int
~~~

argparse 要求三条路径。用 Path(__file__).resolve().parents[2] 作为契约的 repository_root；捕获 EnvironmentContractError、OSError、json.JSONDecodeError、yaml.YAMLError，写通用 input.invalid 报告并返回 2。成功加载后写排序、缩进 JSON；写报告失败返回 2 且仅输出固定通用错误，绝不回显路径或异常内容。

- [ ] **Step 4: 运行 GREEN 与静态检查**

Run:

~~~
python -m pytest -q tests/release/test_validate_environment_contract.py
python -m ruff check tools/release/validate_environment_contract.py tests/release/test_validate_environment_contract.py
~~~

Expected: 三种退出码、脱敏报告和无本机探测断言均通过。

- [ ] **Step 5: 提交任务**

~~~
git add tools/release/validate_environment_contract.py tests/release/test_validate_environment_contract.py
git commit -m "feat: add offline environment contract CLI"
~~~

### Task 4: 发布三套能力、契约与合成观测

**Files:**

- Create: configs/hardware/cpu_reference.yaml
- Create: configs/hardware/rtx4090.yaml
- Create: configs/hardware/h800.yaml
- Create: configs/environment/cpu_reference.yaml
- Create: configs/environment/rtx4090.yaml
- Create: configs/environment/h800.yaml
- Create: data/demo/environment_observations/cpu_reference-valid.json
- Create: data/demo/environment_observations/cpu_reference-invalid-gpu.json
- Create: data/demo/environment_observations/rtx4090-valid.json
- Create: data/demo/environment_observations/rtx4090-invalid-cuda.json
- Create: data/demo/environment_observations/h800-valid.json
- Create: data/demo/environment_observations/h800-invalid-driver.json
- Create: tests/release/test_environment_contract_profiles.py

**Interfaces:**

- Consumes: Task 1--3 的 loaders 和 CLI。
- Produces: 三个公开目标的能力 YAML、运行时契约与正负合成观测。它们不加入 data/demo/manifest.json，因为该 manifest 只列 reproduce_all.py 实际消费的 selection-only smoke 输入。

- [ ] **Step 1: 写出公开 profile RED 测试**

~~~
@pytest.mark.parametrize(("target", "invalid_name", "code"), [
    ("cpu_reference", "cpu_reference-invalid-gpu.json", "runtime.gpu_count.unexpected"),
    ("rtx4090", "rtx4090-invalid-cuda.json", "runtime.cuda.out_of_range"),
    ("h800", "h800-invalid-driver.json", "runtime.driver.out_of_range"),
])
def test_public_profiles_have_valid_and_invalid_offline_observations(
    target: str, invalid_name: str, code: str, tmp_path: Path
) -> None:
    contract = ROOT / "configs/environment" / f"{target}.yaml"
    valid = ROOT / "data/demo/environment_observations" / f"{target}-valid.json"
    invalid = ROOT / "data/demo/environment_observations" / invalid_name

    assert _run_cli(contract, valid, tmp_path / "valid.json").returncode == 0
    assert _run_cli(contract, invalid, tmp_path / "invalid.json").returncode == 1
    assert code in _read_json(tmp_path / "invalid.json")["failures"][0]["code"]
~~~

另测三份硬件 YAML 均可由 HardwareCapability.from_yaml() 解析，CPU 有 cpu IP、两 GPU 有 gpu IP，同名契约只引用同名硬件 YAML，所有公开 JSON 不含绝对路径、ssh、hostname、uuid、serial 或 paper_evidence: true。

- [ ] **Step 2: 运行测试确认 RED**

Run:

~~~
python -m pytest -q tests/release/test_environment_contract_profiles.py
~~~

Expected: FAIL，因为 P5 公开 YAML/JSON 不存在。

- [ ] **Step 3: 写入最小公开配置和 fixture**

能力 YAML 只使用既有模型字段：CPU 为 name: CPU reference、arch: generic-cpu、ips.cpu.precisions: []；4090 为 arch: Ada sm89、GPU 精度 FP32/TF32/FP16/INT8；H800 为 arch: Hopper sm90、GPU 精度 FP32/TF32/BF16/FP16/INT8/FP8。GPU profile 的 features.tensor_core 为 true；不得加入主机、占用、序列号或测量数据。

环境 YAML 都用 environment_contract_v1、framework.name: pytorch、framework.version: >=2.0。CPU 用 python: >=3.10,<3.12、requires_gpu: false；GPU 用同一 Python、requires_gpu: true、cuda: >=12.0,<13、driver: >=525。六份 JSON 均为 environment_observation_v1；正例 Python 3.11.9、PyTorch 2.4.0，GPU 正例 CUDA 12.4、driver 550.54、gpu_count: 1。每个负例只违反其测试指定的一项。

- [ ] **Step 4: 运行 GREEN 与兼容回归**

Run:

~~~
python -m pytest -q tests/release/test_environment_contract_profiles.py tests/stage1/test_capability_schema.py
python -m ruff check framework/environment_contract.py tools/release/validate_environment_contract.py tests/release/test_environment_contract_profiles.py
~~~

Expected: 三目标正例返回 0、负例返回 1，既有 capability schema 回归不变。

- [ ] **Step 5: 提交任务**

~~~
git add configs data/demo/environment_observations tests/release/test_environment_contract_profiles.py
git commit -m "feat: add public P5 environment profiles"
~~~

### Task 5: 保持发布制品和交接入口可发现

**Files:**

- Modify: tools/release/anonymous_allowlist.txt
- Modify: tests/integration/test_anonymous_archive.py
- Create: docs/release-manifests/P5_ENVIRONMENT_CONTRACT.md
- Modify: tests/release/test_project_handoff.py
- Modify: docs/AAAI27_RELEASE_AUDIT.md

**Interfaces:**

- Consumes: Task 3 CLI 和 Task 4 公开配置。
- Produces: 含 configs/**/* 的匿名审稿包，以及从交接台账发现的 P5 合同文档。

- [ ] **Step 1: 写出发布入口 RED 测试**

在 test_anonymous_allowlist_uses_file_recursive_patterns() 的集合加入 configs/**/*；在 anonymous_repo fixture 写最小 configs/hardware/rtx4090.yaml、configs/environment/rtx4090.yaml；在 ZIP 成员断言中要求两文件存在。再在交接测试中加入：

~~~
def test_p5_environment_contract_is_discoverable() -> None:
    contract = REPOSITORY_ROOT / "docs/release-manifests/P5_ENVIRONMENT_CONTRACT.md"

    assert contract.is_file()
    assert "validate_environment_contract.py" in contract.read_text(encoding="utf-8")
    assert "P5_ENVIRONMENT_CONTRACT.md" in HANDOFF.read_text(encoding="utf-8")
~~~

- [ ] **Step 2: 运行测试确认 RED**

Run:

~~~
python -m pytest -q tests/integration/test_anonymous_archive.py tests/release/test_project_handoff.py
~~~

Expected: FAIL，因为 configs allowlist、P5 合同文档和台账链接缺失。

- [ ] **Step 3: 增加 allowlist 与 P5 合同文档**

在 anonymous_allowlist.txt 的递归目录项旁增加 configs/**/*。编写 P5_ENVIRONMENT_CONTRACT.md：说明 hardware YAML 是能力权威、environment YAML 是运行时约束、观测 JSON 是显式脱敏输入；给出三参数 CLI；列出 0/1/2；明确不探测 GPU、不读取环境变量、不运行 CUDA/训练/评测/基准、不下载资产，且不构成真实硬件或论文性能证据。

在台账 P5 行增加相对链接，但保持“进行中（本地）”直到 Task 6 全量门禁通过。

- [ ] **Step 4: 运行 GREEN 与静态检查**

Run:

~~~
python -m pytest -q tests/integration/test_anonymous_archive.py tests/release/test_project_handoff.py
python -m ruff check tests/integration/test_anonymous_archive.py tests/release/test_project_handoff.py
~~~

Expected: 匿名 ZIP 含公开 configs，P5 入口可发现，且无真实环境动作。

- [ ] **Step 5: 提交任务**

~~~
git add tools/release/anonymous_allowlist.txt tests/integration/test_anonymous_archive.py \
  docs/release-manifests/P5_ENVIRONMENT_CONTRACT.md docs/AAAI27_RELEASE_AUDIT.md \
  tests/release/test_project_handoff.py
git commit -m "docs: expose P5 environment contract"
~~~

### Task 6: 完成本地门禁并收口 P5 台账

**Files:**

- Modify: tests/release/test_project_handoff.py
- Modify: docs/AAAI27_RELEASE_AUDIT.md

**Interfaces:**

- Consumes: Tasks 1--5 全部公开实现及测试结果。
- Produces: P5 “已完成（本地）”台账记录；不创建远端状态。

- [ ] **Step 1: 写出台账收口 RED 测试**

~~~
def test_p5_local_closure_is_limited_to_offline_environment_contracts() -> None:
    handoff = HANDOFF.read_text(encoding="utf-8")

    assert "| P5 | 已完成（本地） |" in handoff
    assert "P5_ENVIRONMENT_CONTRACT.md" in handoff
    assert "不探测本机 GPU" in handoff
    assert "不运行 CUDA 编译、训练、评测、基准测试或任何外部资产" in handoff
~~~

- [ ] **Step 2: 确认 RED 并运行功能门禁**

Run:

~~~
python -m pytest -q tests/release/test_project_handoff.py::test_p5_local_closure_is_limited_to_offline_environment_contracts
python -m pytest -q tests/stage1/test_capability_schema.py tests/release/test_environment_contract.py \
  tests/release/test_validate_environment_contract.py tests/release/test_environment_contract_profiles.py \
  tests/release/test_project_handoff.py tests/integration/test_anonymous_archive.py
~~~

Expected: 第一条因 P5 仍“进行中（本地）”而 FAIL；第二条通过，证明功能和制品边界可收口。

- [ ] **Step 3: 更新为本地完成**

将 P5 状态改为 已完成（本地），并准确说明三份能力 YAML、三份环境契约、合成正负例和离线验证器已通过；明确没有真实 GPU、CUDA 编译、训练、评测、基准、外部资产或网络下载。追加“P5 环境契约完成（2026-08-11）”变更记录，只写本地离线结论，不写历史测试计数、机器路径或远端 CI 结果。

- [ ] **Step 4: 执行最终本地验收**

Run:

~~~
python -m pytest -q --cov=framework --cov=scripts/reproduce --cov-report=term-missing --cov-report=xml --cov-fail-under=80
python -m ruff check framework scripts tests tools
python -m compileall -q framework scripts tools
archive_root="$(mktemp -d)"
python tools/release/build_anonymous_archive.py --output-dir "$archive_root/archive"
python tools/release/verify_archive.py "$archive_root/archive/aaai27_code_data_anonymous.zip"
git diff --check
git status --short --branch
~~~

Expected: 覆盖率不少于 80%、Ruff/compileall/diff 无错误、ZIP build/verify 成功，工作树只含待提交的收口变更。

- [ ] **Step 5: 提交本地收口，不推送**

~~~
git add docs/AAAI27_RELEASE_AUDIT.md tests/release/test_project_handoff.py
git commit -m "docs: close P5 environment contract"
git status --short --branch
~~~

Expected: p5-environment-contract 工作树干净；等待用户对推送和远端 CI 的明确授权。

## 计划自审

- 规格覆盖：Task 1 处理显式输入与既有能力 YAML；Task 2 处理版本/GPU 判定；Task 3 处理脱敏 CLI/退出码；Task 4 交付三目标配置与正负例；Task 5 保证匿名包和文档可发现；Task 6 在全量本地门禁后更新台账。
- 范围检查：无真实硬件、网络、模型/数据资产、训练/评测、哈希或供应链工作；data/demo/manifest.json 不变，因为 P5 fixture 不被 selection-only smoke 消费。
- 接口一致性：核心类型在 Task 1 定义，Task 2 定义验证入口，Task 3 只消费该入口，Task 4--6 只通过公开文件和 CLI 验证。
