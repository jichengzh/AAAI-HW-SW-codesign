# P3-34：Stage2 覆盖与导出边界逐项复审

状态：已接受（本地）。本批逐项读取 12 个候选的当前内容，核对 P2 metadata/字节数、当前 SHA-256、实际职责、调用/依赖边界、公开替代物、来源/许可证和安全输出；独立语义复审通过。没有运行私有脚本、模型、数据、设备、子进程或网络。

## 逐项结论

| 职责类别 | 数量 | 处置 | 复审结论 |
| --- | ---: | --- | --- |
| checkpoint 到多尺度 ONNX export | 1 | `external_contract_p4` | 主导依赖为 checkpoint 和 ONNX 制品。P4 需建立可公开的输入获取、许可、哈希和失败关闭契约。 |
| H800 FP16 gate 与 rewrite-suite runner | 2 | `environment_contract_p5` | 主导依赖为 H800、GPU 权限、运行器环境与状态观测，不能公开本机环境假设。 |
| AP/energy coverage jobs、artifact task planner、coverage CLI 与 cold-start plan | 5 | `execution_contract_p6` | 主导职责是任务编排、外部输入消费、CLI 或执行计划生成，必须转入 P6 的受控执行契约。 |
| cost-model bundle、evidence registry、FP16 group-conv classifiers | 4 | `rewritten_public` | 可用合成 rows、profile 和 module spec 重写为纯内存逻辑；在公开实现和测试落地前不视为完成迁移。 |

12 项均无公开同路径文件；公开符号搜索未给出职责等价测试，因而没有把名称或关键词相似误判为 `duplicate_or_superseded`。根许可证信号不覆盖 checkpoint、engine、ONNX、数据或生成制品。

## 证据与阶段边界

全部候选再次绑定当前内容 SHA-256 和 P2 metadata/字节数，完成 AST、导入/调用、公开反证、来源/许可证和安全输出审查。受限 evidence、decision 与路径无关 HMAC ledger 的可重新计算检查通过；这些记录不提交，也不构成 P3 关闭账本。

本批没有迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果。本批接受时总数为 181/1,551；后续批次的当前状态以总台账为准。P4 尚未开始。
