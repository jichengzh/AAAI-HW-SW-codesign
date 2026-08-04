# P3-55 recovery 与 TVM 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 422 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 3 | baseline admission、capability rebind 与 AP feedback repair 外部 evidence 契约 |
| P6 | 9 | checkpoint recovery、replay/watchdog、GPU scheduler/measurement、ONNX quantization 与 Relax worker 执行契约 |

双审逐项一致确认：ONNX quant-contract builder 会使用 calibration 样本运行 ONNX Runtime inference；TVM/Relax worker 会加载编译模块、创建运行时并执行，因此两者均为 P6，不因制品关键词而降为 P4。baseline admission、capability rebind 与 AP feedback repair 只验证、重写或修复既有 evidence，并不启动模型、GPU 或测量，故归 P4。checkpoint、ONNX、compiled module、calibration 数据、AP report、GPU telemetry、日志和结果均仍是独立的来源/许可/隐私边界。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖 checkpoint、数据集、ONNX、compiled module、runtime weights、TVM/ORT、GPU、测量结果、AP evidence 或其派生物。后续 P4/P6 必须使用路径无关引用、显式输入契约、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **434/1,551**。
