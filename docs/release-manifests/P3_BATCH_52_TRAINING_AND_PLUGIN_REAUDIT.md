# P3-52 training 与 plugin 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 386 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 3 | tiny 训练/评估的外部数据、权重、annotation 与 checkpoint 制品配置 |
| P6 | 9 | plugin、训练、评测、loss、runner、推理与 model head 执行契约 |

第三次只读裁决确认：tiny 训练/评估配置并不以 GPU 驱动或环境探测为主；其核心是把数据 annotation、基础权重、checkpoint 和日志制品绑定到训练/评估流程，故归 P4。plugin registry 和 API export shim 会把调用方接入数据构建、CUDA/DDP 包装、runner、hook、checkpoint 与训练执行链，故归 P6 而非 P5。Motion/occupancy heads 含 ONNX/TRT 适配片段，但目前没有公开模型 head API、导出回归或完整来源 NOTICE；它们保持 P6，后续还须完成第三方来源与许可审查。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖数据集、annotation、checkpoint、日志、ONNX、engine、CUDA/DDP、MMDet/MMCV、CARLA 或第三方 TRT 适配片段。后续 P4/P6 必须使用路径无关引用、显式输入契约、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **398/1,551**。
