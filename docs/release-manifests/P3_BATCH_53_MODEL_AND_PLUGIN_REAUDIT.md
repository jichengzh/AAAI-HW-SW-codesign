# P3-53 model 与 plugin 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 398 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P6 | 12 | model head、CUDA/custom-op、attention 与 ONNX/TRT export 执行契约 |

双审逐项一致确认：planning/tracking/segmentation heads、assignment、CUDA autograd/custom-op、transformer/BEV attention 及 ONNX/TRT export 支撑都参与模型训练、推理或导出执行链。公开树没有等价 model head、CUDA plugin、ONNX/TRT export API 或测试支持链，不能以同名、目录、关键词或相邻 smoke 作为公开替代证据。候选中含 OpenMMLab 修改头或外部 TRT 适配来源的部分，后续 P6 前必须保留并核验 NOTICE、来源、CUDA/plugin 可用性、合成 tensor I/O 与路径安全失败关闭；不得直接迁入。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖 checkpoint、数据集、ONNX、engine、CUDA/TRT plugin、传感器几何、外部模型资产或第三方适配片段。后续 P6 必须使用路径无关引用、显式输入契约、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **410/1,551**。
