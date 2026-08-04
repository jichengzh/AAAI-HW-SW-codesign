# P3-57 engine 与 baseline 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 446 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 9 | latency mapping、baseline integration 与 predictor/Pareto 分析的外部 evidence 契约 |
| P6 | 3 | ONNX export、TensorRT engine build 与 CUDA benchmark 的受控执行契约 |

主审与独立复审对全部 12 项逐项一致，无需第三次裁决。二者确认：消费既有 baseline、预测器或硬件测量表的映射、整合和 Pareto 分析，主导风险是输入证据的来源、许可、校准和路径无关输出，故归 P4；构建 engine、导出 ONNX 或执行 CUDA benchmark 的候选会物化运行时制品或产生真实测量，故归 P6。公开树没有承担这些完整职责的 API 或测试链，因此没有将相似的 CPU-only 选择逻辑错误认定为公开替代。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖 baseline 表、测量结果、预测器、ONNX、engine、calibration、GPU/TensorRT 运行时或其派生物。后续 P4/P6 必须使用路径无关引用、显式输入契约、受控输出根、来源/哈希验证和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **458/1,551**。
