# P4 外部输入契约

`artifacts/external/registry.json` 是轻量外部输入注册表，格式为
`aaai27_external_input_registry_v1`，当前 `registry_version` 为 `1`。每条记录声明
`input_id`、资产类别、来源、许可状态与参考、版本、预期用途、消费者、相对路径及可用性。
当 `availability` 为 `unavailable` 时，`unavailable_reason` 必填。

## 离线验证

仅对用户已经提供在本地资产根目录中的输入运行：

```bash
python tools/release/validate_external_inputs.py \
  --registry artifacts/external/registry.json \
  --asset-root /path/to/external-assets \
  --output /tmp/external-input-validation.json
```

验证器不下载任何内容，也不会运行 Stage6 或 Stage7。它只读取注册表和显式给出的本地资产根目录，
并把逐项结果写入 `--output`；存在不可用或未验证输入时以非零状态退出。

## 当前初始记录

| input_id | 相对路径 | 用途 | 状态 | 原因 |
| --- | --- | --- | --- | --- |
| `stage6-terminal-evidence` | `stage6/terminal-evidence.jsonl` | `stage6_representative_selection` | `unavailable` | `bundle_not_published` |
| `stage7-formal-aggregate` | `stage7/formal-aggregate.json` | `stage7_formal_aggregate` | `unavailable` | `bundle_not_published` |

两项均来自 `publication-pending`，版本为 `unreleased`，许可状态尚未确认，因此不应被视为可下载或可再分发的资产。

## 后续登记规则

校验和 `sha256` 是可选字段。后续记录必须填写事实性的来源、许可和版本信息；不得以占位信息
将未发布资产标记为可验证。
