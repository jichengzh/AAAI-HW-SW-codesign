# P3-35：Stage2 测试边界逐项裁决

状态：已接受（本地）。本批 12 个候选先后经过主审、独立语义审阅和第三次只读裁决。三次审阅都重新核对 P2 metadata/字节数与当前 SHA-256，未发现内容漂移；没有执行私有脚本、模型、数据、设备、子进程或网络。

## 分歧处理

前两次审阅的总计数相同，但 8 个候选对 P4/P5/P6 主导边界的逐项判断不同。这一状态没有生成 ledger，也没有计入 P3。第三次裁决重新比较每项的实际职责、公开实现、公开测试/API 和安全输出，确认公开 Stage2 contract/demo 测试只覆盖部分安全输入、gate 或输出语义，不能覆盖私有测试的完整制品、环境与执行组合职责。因此不将任何候选误判为 `duplicate_or_superseded`。

公开 Stage2 contract/demo 定向回归为 20 passed；它只作为“部分覆盖而非等价替代”的反证，不把 scoped demo 升级为完整实验复现。

## 逐项结论

| 职责类别 | 数量 | 处置 | 裁决结论 |
| --- | ---: | --- | --- |
| H800 latency evidence、integration/evidence delta、LUT coverage、measurement row、checkpoint selection | 5 | `external_contract_p4` | 主导依赖为外部测量、checkpoint、ONNX/engine、evidence row 或制品选择。P4 需提供公开 schema、固定合成输入、许可/下载说明和哈希校验。 |
| GPU preflight 与 FP16 launch planning | 2 | `environment_contract_p5` | 主导职责是 GPU 占用、设备映射、端口和调度资源规划。P5 应以脱敏环境规范和失败关闭探测公开。 |
| native INT8 bridge/route/worker、latency coverage queue、FP16 lane runner | 5 | `execution_contract_p6` | 主导职责是 worker、队列、运行器或训练/评测执行编排，不能直接迁入默认公开 CI。 |

根许可证信号为 Apache-2.0；该信号不覆盖模型、checkpoint、ONNX、engine、数据、测量制品或运行环境的再分发条件。所有候选均有受限的职责、调用、来源/许可证和安全输出记录；路径无关 HMAC ledger 可重新计算，但不提交，也不构成 P3 全量关闭账本。

本批没有迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果。它使已接受重审总数达到 205/1,551；P3-36 仍处于第三次裁决前的冲突状态，P4 尚未开始。
