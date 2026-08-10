# P4 外部输入契约

`artifacts/external/registry.json` 是完整外部资源目录，格式为
`aaai27_external_input_registry_v1`，当前 `registry_version` 为 `1`。每条记录声明
`input_id`、资产类别、来源、许可状态与参考、版本、预期用途、消费者、相对路径及可用性。
当 `availability` 为 `unavailable` 时，`unavailable_reason` 必填。

`artifacts/external/coverage.json` 是完整性摘要，格式为
`aaai27_p4_external_resource_coverage_v1`。它仅保存聚合数量：P3 转交给 P4 的候选共
506 项，其中 199 项裁决为 document、307 项裁决为 asset；asset 决策对应 284 项去重资源。

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

## 完整目录与本地裁决边界

`registry.json` 是资源事实的权威来源；本文档不逐项复制目录内容。document 候选只计入
`coverage.json` 的完整性统计，不作为资产记录。多个 asset 候选是否指向同一实际资源，仅在本机
受限 resolution 中依据证据完成去重；该 resolution 不进入 Git。

公开 registry 与 coverage 均不包含候选标识、原始来源标识或私有定位。目录中未能从公开证据确认的
来源、许可、版本和校验和均如实保持不可用或未确认状态，不以猜测值补齐，也不应被视为可下载或
可再分发的资产。

## 后续登记规则

校验和 `sha256` 是可选字段。后续更新必须填写事实性的来源、许可和版本信息；不得以占位信息
将未发布资产标记为可验证。完整性数量发生变化时，应由受限 resolution 重新编译公开目录和摘要，
而不是手工修改单个聚合值。
