# P3-31：F-Cooper 测试与证据边界逐项复审

状态：已接受（本地）。本批逐项读取 12 个候选的当前内容，并核对 P2 metadata/字节数、当前 SHA-256、实际职责、导入/调用、公开替代物、来源/许可证和安全输出。独立语义复审已通过。没有运行私有测试、脚本、模型、数据、设备、子进程或网络。

## 逐项结论

| 职责类别 | 数量 | 处置 | 复审结论 |
| --- | ---: | --- | --- |
| CoDriving ONNX export、TVM capability、heldout capture 与 Stage6 graph evidence | 4 | `external_contract_p4` | 主导依赖为外部模型、捕获、哈希或 evidence 制品。P4 必须提供可公开的 schema、固定合成输入、许可/下载说明与校验边界。 |
| 硬件能力约束、native benchmark、Orin five-config | 3 | `environment_contract_p5` | 主导依赖为设备能力、可见 GPU、基准环境或部署配置。P5 只能提供脱敏环境规范和失败关闭探测。 |
| formal round、TVM measurement、formal runner 与 GPU exclusivity | 5 | `execution_contract_p6` | 主导职责是运行器、测量、回合关闭、进程保护或实际执行调度，不能作为默认公开 CI 测试迁入。 |

当前公开树没有同路径候选，也没有可等价覆盖上述职责的公开测试。代码根许可证信号为 Apache-2.0；外部模型、制品、环境与数据的许可不由该信号覆盖，已转交相应后续阶段。

## 证据与阶段边界

12 项全部再次绑定 P2 分类/字节数和当前内容 SHA-256，并完成 AST、导入/调用、公开反证、来源/许可证和安全输出审查。受限 evidence、decision 与路径无关 HMAC ledger 均可重新计算；这些本地记录不提交，也不构成 P3 全量关闭账本。

本批没有迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果。它使已接受重审总数达到 157/1,551；后续批次的当前状态以总台账为准。P4 尚未开始。
