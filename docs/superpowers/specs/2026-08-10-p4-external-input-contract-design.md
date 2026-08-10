# P4 外部输入契约设计

## 目标

P4 为 P3 转交的 `external_contract_p4` 工作集建立轻量的外部资产注册表和纯离线验证器。它只说明完整实验所需的数据集、模型、checkpoint、ONNX/engine、测量/证据包及其可再生成大文件，并检查用户是否已提供这些输入；不下载、不迁入、不生成或执行这些资产。

## 已确认的边界

- P3 的 1,551 项候选已在本地关闭；其中 506 项转交 P4。私有候选路径、原始内容、HMAC key 和受限逐项 decisions 永不进入仓库或 validator 输出。
- P4 不运行训练、AP 评测、GPU、CUDA、TensorRT、TVM、队列或网络请求。这些运行行为属于 P6 或后续明确授权的外部动作。
- `data/demo/` 中的合成 smoke 输入始终是非论文证据，不能代替外部资产或把 `unavailable` 状态伪装成已复现。
- P4 只验证用户显式提供的本地资产根目录。默认命令不能从维护者目录、缓存、环境变量、绝对路径或网络位置发现输入。

## 方案选择

采用“按真实外部资产归一”的方案，而非将 506 个 P3 候选逐条复制到公开文件。

一个物理资产可服务多个 P3 候选或 Stage6/Stage7 消费者，因此 registry 的一条记录代表一个可获取并可校验的资产版本。P3 的公开 HMAC candidate ID 仅可作为可选的路径无关 `consumer_ids`；它不是本地路径映射，也不允许反向恢复私有内容。

被放弃的替代方案：

1. 每个 P3 候选一条 registry 记录：重复且会将私有职责结构暴露到公共面。
2. 仅按“数据集/模型/结果”大类汇总：无法落实每个实际输入的版本、大小和 SHA-256，不能失败关闭。

## 组件与职责

| 组件 | 职责 | 明确不做的事 |
| --- | --- | --- |
| `artifacts/external/registry.json` | 公开、版本化的资产声明；保存每项资产的身份、来源、许可、版本、用途、相对目录投影和可用性。 | 不保存资产字节或真实实验结果。 |
| `tools/release/validate_external_inputs.py` | 读取显式 registry 和资产根目录，检查登记的输入是否存在；若登记中已有上游校验和，则额外比对它。 | 不下载、解压、运行模型或写入资产根目录。 |
| `tests/release/test_external_input_registry.py` | 通过小型合成 registry/资产 fixture 验证必要字段、可用性和基本错误语义。 | 不接触真实数据、模型或 checkpoint。 |
| `tests/integration/test_external_input_validation.py` | 验证 CLI 的 verified/unavailable 结果与可重复输出。 | 不将成功的合成 fixture 表示为论文或正式 Stage6/Stage7 证据。 |
| `docs/release-manifests/P4_EXTERNAL_INPUT_CONTRACT.md` | 说明 public registry 格式、获取责任边界、使用命令和 P4 完成条件。 | 不记录私有资产位置或可执行下载步骤。 |

## Registry v1 数据模型

顶层对象固定为：

```json
{
  "format": "aaai27_external_input_registry_v1",
  "registry_version": 1,
  "inputs": []
}
```

每个 `inputs` 条目必须具有下列字段：

| 字段 | 约束 |
| --- | --- |
| `input_id` | 小写、安全 token；全表唯一。 |
| `asset_kind` | `dataset`、`model`、`checkpoint`、`onnx`、`engine`、`measurement_bundle`、`evidence_bundle` 或 `regenerable_large_file`。 |
| `source` | 公开来源引用和获取方式。 |
| `license` | `confirmed`、`unconfirmed` 或 `not_redistributable`，以及公开许可证/权限依据。 |
| `version` | 上游发布版本、日期版本或其他项目可识别的版本说明。 |
| `relative_path` | 相对于调用者提供 `--asset-root` 的文件位置。 |
| `intended_use` | 受控 token，说明其被 Stage6、Stage7 或完整实验中的何种非执行性输入契约消费。 |
| `consumer_ids` | 零个或多个公开 HMAC candidate ID；无候选映射时为空数组。 |
| `availability` | `available_for_verification` 或 `unavailable`。 |
| `unavailable_reason` | 仅在 `unavailable` 时要求；安全、稳定的受控 token。 |
| `sha256` | 可选。仅当上游已经发布可信摘要，或项目需要固定本地版本时记录和校验。 |

`available_for_verification` 只表示 caller 已提供本地字节，不表示论文结果已复现。`unavailable` 是预期的失败关闭状态：仍须说明来源、许可、版本、用途和缺失原因。大小和摘要不是每项登记的前置条件。

## Validator 接口与数据流

CLI：

```text
python tools/release/validate_external_inputs.py \
  --registry artifacts/external/registry.json \
  --asset-root /caller/owned/assets \
  --output /caller/owned/output/external-input-validation.json
```

`--registry` 和 `--asset-root` 都必须由调用者显式传入。输出路径也必须是调用者拥有的安全目录；输出中的路径只保留 registry 的相对投影，绝不回显调用者的绝对路径。

处理流程：

1. 读取 registry，检查 `input_id` 唯一且每条记录包含必要信息。
2. 对 `available_for_verification` 记录在 `--asset-root` 下检查 `relative_path` 是否存在且是文件；存在时输出 `verified`。
3. 对带有可选 `sha256` 的记录计算摘要并比对；不匹配时输出 `unavailable`。
4. 对 registry 已声明 `unavailable` 的记录直接输出该状态和原因。
5. 如果任一必需记录不是 `verified`，CLI 以非零退出码结束，并写出简洁的结果 manifest。

P4 validator 不修改 `scripts/reproduce/reproduce_all.py` 的 Stage6/Stage7 执行路径。未来 P6 可消费 P4 的已验证 registry 结果；缺失的正式输入仍应明确标为不可用。

## 错误处理与安全性

- 用户可见的失败使用简洁状态：`registry_invalid`、`asset_missing`、`asset_sha256_mismatch` 或 `asset_declared_unavailable`。
- registry 和输出使用普通 JSON；结果仅说明每项资产的相对位置与状态。
- 校验器只检查 `--asset-root` 下 registry 指定的文件，不递归扫描目录。

## TDD 与验收

实现必须先为以下行为编写失败测试并观察其失败：

1. 最小合法 registry 和匹配的合成文件产生 `verified` 结果。
2. 缺失文件产生 `unavailable` 状态和非零退出码。
3. 有可选摘要的文件在摘要不符时产生 `unavailable`。
4. registry 预声明 `unavailable` 时输出其原因。
5. 合成 fixtures 明确标为 `paper_evidence: false`，且集成测试确认它们无法让 Stage6/Stage7 从 `unavailable` 变成完成。

P4 的实质完成条件是：每一个将由公开完整实验消费的真实外部资产都有来源、版本、许可、用途、位置和可用性说明；上游提供摘要时可记录并校验。尚缺资产或许可的输入保持 `unavailable`，不会被 demo 代替。P4 完成不表示 P6 的训练、测量或论文证据已经执行。

## 非目标

- 不在本阶段提供下载器、镜像、凭据、认证、自动解压或缓存逻辑。
- 不在本阶段新增真实模型、数据集、checkpoint、ONNX、engine、测量日志或论文结果。
- 不在本阶段改变 Stage6/Stage7 的运行、选择、聚合或论文表格生成行为。
- 不在本阶段推进 P5、P6、P7 或发布。
