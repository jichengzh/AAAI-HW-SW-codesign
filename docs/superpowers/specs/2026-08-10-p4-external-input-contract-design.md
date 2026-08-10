# P4 外部输入契约设计

## 目标

P4 为 P3 转交的 `external_contract_p4` 工作集建立可公开审计的外部资产注册表和纯离线验证器。它只声明和验证完整实验所需的数据集、模型、checkpoint、ONNX/engine、测量/证据包及其可再生成大文件；不下载、不迁入、不生成或执行这些资产。

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
| `artifacts/external/registry.json` | 公开、版本化、路径无关的资产声明；保存每项资产的身份、来源、许可、版本、校验元数据、用途和安全目录投影。 | 不保存资产字节、私有路径、密钥、账号、下载 cookie 或真实实验结果。 |
| `tools/release/validate_external_inputs.py` | 读取显式 registry 和资产根目录，安全地验证资产存在性、常规文件属性、相对路径、大小及 SHA-256，并产生可公开的状态结果。 | 不下载、解压、运行模型、写入资产根目录、扫描其他目录或尝试网络访问。 |
| `tests/release/test_external_input_registry.py` | 通过小型合成 registry/资产 fixture 验证 schema、路径安全、许可状态及错误语义。 | 不接触真实数据、模型、checkpoint 或维护者目录。 |
| `tests/integration/test_external_input_validation.py` | 验证 CLI 的 verified/unavailable 结果、确定性输出和泄漏防护。 | 不将成功的合成 fixture 表示为论文或正式 Stage6/Stage7 证据。 |
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
| `source` | 公开来源引用和获取方式；不得含凭据、私有主机、绝对路径或用户目录。 |
| `license` | `confirmed`、`unconfirmed` 或 `not_redistributable`，以及公开许可证/权限依据。 |
| `version` | 上游发布版本、日期版本或不可变内容版本；不得使用模糊的“latest”。 |
| `expected_relative_path` | 相对于调用者提供 `--asset-root` 的非空 POSIX 相对路径；不得包含 `..`。 |
| `expected_size_bytes` | 非负整数。 |
| `expected_sha256` | 64 位小写十六进制 SHA-256。 |
| `intended_use` | 受控 token，说明其被 Stage6、Stage7 或完整实验中的何种非执行性输入契约消费。 |
| `consumer_ids` | 零个或多个公开 HMAC candidate ID；无候选映射时为空数组。 |
| `availability` | `available_for_verification` 或 `unavailable`。 |
| `unavailable_reason` | 仅在 `unavailable` 时要求；安全、稳定的受控 token。 |

`available_for_verification` 只表示 caller 已提供可按元数据校验的本地字节，不表示论文结果已复现。`unavailable` 是预期的失败关闭状态：记录仍必须提供来源、许可、版本、相对投影、大小和 SHA-256；若这些事实尚不能确定，则该输入尚不能登记为 P4 可审计条目，必须留在受限处理队列而不能伪造 metadata。

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

1. 解析 registry，拒绝重复 JSON key、未知字段、重复 `input_id`、格式不合法、未经允许的 token 和不完整 metadata。
2. 对每条记录使用 `lstat` 和不跟随符号链接的打开方式解析 `expected_relative_path`；拒绝链接、目录、设备、绝对路径及越界路径。
3. 对 `available_for_verification` 记录计算字节数和 SHA-256。存在且匹配时输出 `verified`；不存在、非普通文件、大小不符或摘要不符时输出 `unavailable`，并使用固定错误码。
4. 对 registry 已声明 `unavailable` 的记录，不读取任何资产字节，直接输出相同的失败关闭状态和该受控原因。
5. 如果任一必需记录不是 `verified`，CLI 以非零退出码结束，同时原子写出完整、可审计的结果 manifest；结果中不得出现本机路径、原始异常、资产内容、用户名、主机、IP 或密钥形态。

P4 validator 不修改 `scripts/reproduce/reproduce_all.py` 的 Stage6/Stage7 执行路径。未来 P6 只能消费带有 validator 输出 SHA-256 的已验证 registry 结果；没有该绑定即应失败关闭。

## 错误处理与安全性

- 对用户可见的失败仅使用受控错误码，例如 `registry_invalid`、`asset_missing`、`asset_not_regular_file`、`asset_size_mismatch`、`asset_sha256_mismatch`、`asset_declared_unavailable`。
- 防止 TOCTOU：校验器在读取前后比较文件 metadata，并拒绝校验过程中被替换的资产。
- Registry、资产和输出均需限制合理大小；所有 JSON 写入采取确定性序列化和原子写入。
- 校验器不读取环境变量中定义的输入位置，也不递归扫描 `--asset-root`；每次调用只访问 registry 指定的相对目标。
- 新的身份/凭据扫描必须覆盖 registry、文档、测试 fixture 及 validator 输出。

## TDD 与验收

实现必须先为以下行为编写失败测试并观察其失败：

1. 最小合法 registry 和匹配的合成文件产生确定性的 `verified` 结果与输出 digest。
2. 绝对、空、`..` 或符号链接投影被拒绝且不泄露调用者路径。
3. 重复 input ID、未知字段、非法 SHA-256、负大小、`latest` 版本和不合法的许可/availability 组合被拒绝。
4. 缺失、非普通文件、大小不符及 SHA-256 不符均产生相应 `unavailable` 状态和非零退出码。
5. registry 预声明 `unavailable` 时不得读取资产；输出必须保留其受控原因。
6. 输出确定性、可幂等、路径无关，且扫描不到绝对路径、用户标识、IP、凭据或资产正文。
7. 合成 fixtures 明确标为 `paper_evidence: false`，且集成测试确认它们无法让 Stage6/Stage7 从 `unavailable` 变成完成。

P4 的实质完成条件是：每一个将由公开完整实验消费的真实外部资产都有以上完整 registry 元数据、许可结论和离线验证结果；尚缺元数据或许可的资产保持 `unavailable`，不会被 demo 代替。P4 完成不表示 P6 的训练、测量或论文证据已经执行。

## 非目标

- 不在本阶段提供下载器、镜像、凭据、认证、自动解压或缓存逻辑。
- 不在本阶段新增真实模型、数据集、checkpoint、ONNX、engine、测量日志或论文结果。
- 不在本阶段改变 Stage6/Stage7 的运行、选择、聚合或论文表格生成行为。
- 不在本阶段推进 P5、P6、P7 或发布。
