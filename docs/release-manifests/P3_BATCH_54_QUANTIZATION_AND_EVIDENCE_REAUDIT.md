# P3-54 quantization 与 evidence 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 410 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 3 | coldstart、engine/probe 与 graph evidence 的外部制品契约 |
| P6 | 9 | pruning/quantization、GPU guard/benchmark、ONNX preparation 与 recovery training 执行契约 |

第三次只读裁决确认 GPU exclusivity gate 的 GPU telemetry、锁和 quiet-window 是准入守卫；其主导行为是启动、监控、隔离和终止被保护命令，故归 P6 而非 P5。冷启动、engine/probe 和 graph evidence 脚本消费既有结果、engine、ONNX 或 formal feedback 而不训练或测量，故归 P4。量化候选含 QuantV2X/OpenCOOD 或第三方 pruning 依赖/改写信号；后续 P6 前必须核验 NOTICE/provenance、校准数据隐私、CUDA/ONNX/ORT 可用性与路径安全失败关闭，不能直接迁入。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖 calibration 数据、checkpoint、engine、ONNX、GPU telemetry、训练日志、测量结果、外部量化来源或其派生物。后续 P4/P6 必须使用路径无关引用、显式输入契约、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **422/1,551**。
