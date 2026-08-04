# P3-56 DP4A 与 baseline 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 434 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 7 | scheduler/feedback integrity、H800/4090 baseline、Pareto predictor 与 accuracy evidence 外部制品契约 |
| P6 | 5 | TVM worker 与 DP4A compile/tune/benchmark 执行契约 |

第三次只读裁决确认两个 Pareto 脚本虽会运行离线评分/预测，但其主导输入是私有 baseline evidence 或已训练 predictor artifact；未启动候选、GPU、测量器、worker 或模型运行闭环，故归 P4。DP4A gates 会构建、调优、编译、导出和 benchmark CUDA/TVM 代码，故归 P6。所有候选仍须在后续处理真实 GPU metrics、baseline/predictor provenance、TVM/CUDA toolchain、CUDA dump 和路径安全输出，不能直接迁入。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖 baseline 表、H800/4090 测量、trained predictor、ONNX、TVM/MetaSchedule、CUDA、GPU、scheduler 日志、AP evidence 或其派生物。后续 P4/P6 必须使用路径无关引用、显式输入契约、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **446/1,551**。
