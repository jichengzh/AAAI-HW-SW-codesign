# P5：纯环境契约与离线验证器设计

**日期：** 2026-08-11
**状态：** 已确认，等待实施计划
**范围：** P5 的公开 CPU、RTX 4090 与 H800 环境契约，以及只消费显式文件输入的离线验证器。

## 1. 目标与非目标

P5 将静态硬件能力、目标环境约束和一次脱敏的环境观测分开表示，使公开仓库能在没有真实 GPU、私有目录、外部资产或网络访问的条件下验证“某份声明的环境观测是否满足一个公开目标的最低要求”。

P5 不执行或声称执行以下动作：

- 探测本机 GPU、读取环境变量、调用 `nvidia-smi` 或其他系统命令；
- CUDA 编译、训练、评测、性能/功耗基准或结果汇总；
- 下载、打开或登记模型、权重、ONNX、engine、缓存或其他外部资产；
- 输出 SSH 信息、主机名、用户名、绝对路径、设备序列号、GPU UUID、原始日志或机器可识别信息；
- 用哈希、SHA-256 或复杂供应链机制替代这里的最小环境兼容性判断。

通过 P5 仅表示公开的环境契约和合成离线验证已完成；它不构成真实 4090/H800 可用性、训练成功或论文性能证据。

## 2. 既有模型与新增文件

`framework.capability_schema.HardwareCapability` 是唯一的静态硬件能力权威模型。P5 复用其现有 YAML 语义，包括 `name`、`arch`、`ips`、`alignment`、`features` 与 `toolchain`；不在 JSON 或环境契约中重复声明 IP、精度、对齐或设备能力。

P5 新增下列公开文件组：

```text
configs/hardware/
  cpu_reference.yaml
  rtx4090.yaml
  h800.yaml

configs/environment/
  cpu_reference.yaml
  rtx4090.yaml
  h800.yaml

data/demo/environment_observations/
  cpu_reference-valid.json
  rtx4090-valid.json
  h800-valid.json
  <target>-invalid-*.json
```

`configs/hardware/*.yaml` 由 `HardwareCapability` 校验，是各目标的静态能力事实来源。`configs/environment/*.yaml` 是 P5 的薄环境契约：它只选择一个已有能力 YAML，并定义运行时最低约束。`data/demo/environment_observations/*.json` 均为合成、脱敏、非论文证据的验证 fixture，不能被描述成任何维护者机器的导出结果。

## 3. 数据契约

### 3.1 环境契约 YAML

环境契约使用固定 schema 名 `environment_contract_v1`，最小形式如下：

```yaml
schema: environment_contract_v1
target: rtx4090
hardware_capability: configs/hardware/rtx4090.yaml
requires_gpu: true
runtime:
  python: ">=3.10,<3.12"
  cuda: ">=12.0,<13"
  driver: ">=525"
  framework:
    name: pytorch
    version: ">=2.0"
```

`target` 必须在 `cpu_reference`、`rtx4090`、`h800` 中取值。`hardware_capability` 必须是仓库内的相对路径，并由 `HardwareCapability.from_yaml()` 成功解析。CPU 契约固定 `requires_gpu: false`，并且不得声明 `cuda` 或 `driver`；4090 与 H800 固定 `requires_gpu: true`，并且必须声明这两个约束。

环境契约中的运行时约束仅支持以逗号连接的数值版本比较：`>=`、`>`、`<=`、`<` 或精确值。版本按点分数字比较并以零补齐，例如 `12.4` 与 `12.4.0` 相等；不支持发行版名称、预发布标签、内核版本或平台状态推断。未知契约字段应拒绝，以尽早发现公开配置的拼写错误。

### 3.2 脱敏环境观测 JSON

观测使用固定 schema 名 `environment_observation_v1`：

```json
{
  "schema": "environment_observation_v1",
  "target": "rtx4090",
  "runtime": {
    "python": "3.11.9",
    "cuda": "12.4",
    "driver": "550.54",
    "framework": {"name": "pytorch", "version": "2.4.0"}
  },
  "gpu_count": 1
}
```

CPU 观测必须给出 `gpu_count: 0`，且不得给出 CUDA 或驱动版本；GPU 目标必须给出 `gpu_count >= 1` 以及契约要求的 CUDA、驱动和框架字段。观测可以带不参与判定的普通脱敏备注字段，但验证报告既不复制这些字段，也不从中提取信息。

观测 JSON 不能定义、覆盖或推断硬件能力。目标一致性和硬件能力合法性始终由环境契约及其引用的 YAML 决定。

## 4. 验证器与数据流

新增 `tools/release/validate_environment_contract.py`，只接受显式给出的文件：

```bash
python tools/release/validate_environment_contract.py \
  --contract configs/environment/rtx4090.yaml \
  --observation data/demo/environment_observations/rtx4090-valid.json \
  --output /tmp/environment-report.json
```

处理顺序固定：

1. 读取并校验环境契约 YAML；
2. 解析契约引用的静态能力 YAML；
3. 读取并校验观测 JSON；
4. 依次判断 schema、目标一致性、GPU 要求、运行时必填字段和版本范围；
5. 输出结构化、脱敏报告。

实现应将文件加载、版本判断、环境比较和 CLI 适配分为单一职责的小函数。核心环境比较为纯函数：只以已解析的契约和观测对象为输入，返回新结果对象，不读取本机状态，也不修改输入对象。

退出码约定如下：

| 退出码 | 含义 |
| --- | --- |
| `0` | 契约和观测均有效，且环境满足契约。 |
| `1` | 输入可解析，但观测不满足契约。 |
| `2` | 参数、文件、schema、契约或能力 YAML 无效。 |

通过与不通过均写出 `environment_contract_report_v1` 报告。例如：

```json
{
  "schema": "environment_contract_report_v1",
  "target": "rtx4090",
  "passed": false,
  "failures": [
    {"code": "runtime.cuda.out_of_range", "field": "runtime.cuda"}
  ]
}
```

报告只包含 target、布尔结果和稳定的失败代码/字段名；它不得回显原始观测值、命令行路径或能力 YAML 的内容。失败应关闭，不以默认版本、局部通过或警告替代明确的失败。

## 5. 测试与验收

实施遵循 TDD，并拆为以下可独立验证的工作：

1. 为 P5 契约和观测结构建立失败测试，再实现最小 schema/文件加载逻辑；
2. 为目标、GPU 数和版本范围建立失败测试，再实现纯比较函数；
3. 为 CLI、报告脱敏、退出码和三目标公开配置建立集成测试。

P5 测试置于 `tests/release/`；原有 `tests/stage1/test_capability_schema.py` 继续保护硬件 YAML 的兼容语义。测试只能使用临时目录与公开合成 fixture，至少覆盖：

- CPU、4090、H800 的有效契约及有效观测；
- schema/必填字段错误、未知 target、目标不一致；
- 仓库外能力引用和能力 YAML 解析失败；
- CPU/GPU `gpu_count` 不符合要求；
- Python、CUDA、driver 或框架版本不在范围；
- 报告不含观测值或输入路径，且 CLI 的三种退出码稳定。

P5 的本地收口条件为：三份能力 YAML、三份环境契约及其合成正负例都能离线验证；全量 CPU pytest 与 80% 覆盖率门槛、Ruff、`compileall` 和 `git diff --check` 均通过；发布台账明确标为“已完成（本地）”并说明没有执行真实硬件操作。P5 不包含推送、合并或真实 GPU 访问。
