# P3-32：F-Cooper TVM 测试边界逐项复审

状态：已接受（本地）。本批逐项读取 12 个候选的当前内容，核对其 P2 metadata/字节数、当前 SHA-256、职责、导入/调用、公开替代物、来源/许可证和安全输出。主审和独立语义复审均通过。没有执行私有测试、脚本、数据、模型、设备、子进程或生成物。

## 逐项结论

| 职责类别 | 数量 | 暂定处置 | 复审结论 |
| --- | ---: | --- | --- |
| 外部证据、制品和请求契约测试 | 5 | `external_contract_p4` | 覆盖 original admission、control closure、TVM evidence pools、INT8 quant contract 和 Stage6 request rebinding。它们定义外部 checkpoint、ONNX、calibration、AP、performance、request manifest 或 evidence bundle 的 SHA 绑定与安全投影，需在 P4 建立公开契约。 |
| GPU/调度环境协议测试 | 2 | `environment_contract_p5` | 覆盖 GPU exclusivity/runtime guard 和 round watcher/scheduler audit。P5 需要定义 GPU 池、调度状态、进程观测和环境采集的 schema 与路径安全边界。 |
| TVM/ONNX/AP bridge 与 worker 执行测试 | 5 | `execution_contract_p6` | 覆盖 source preparation、FP16/INT8 AP bridge、GPU scheduler 和 Relax worker。它们涉及 ONNX/compiled module、TVM runtime、torch tensors、worker dispatch 或真实执行器，只能进入 P6 外部执行契约。 |

逐项安全审查未发现直接凭据、邮箱或 IP。多项候选包含绝对路径样例、GPU/TVM/ONNX/engine/checkpoint/artifact 语义或进程调度面；这些是进入 P4/P5/P6 的原因，不应作为默认公开测试迁入。

## 证据与阶段边界

12 项均完成当前内容 SHA-256 与 P2 metadata/字节数对账，当前内容漂移为 0。P2b inventory 没有每文件历史 digest，当前 SHA-256 仅绑定本轮读取的内容；源码变化时必须重新审计。

本批不迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果。受限 HMAC evidence/decision records 仅用于本机核验，不构成 P3 关闭账本。主审与独立语义复审均确认 12 项当前内容无漂移，HMAC ledger 可重新计算；本批使已接受总数达到 193/1,551。P4 尚未开始。
