# P3-76 TensorRT/TVM recovery 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批 12 项完成 P2 metadata/bytes、当前 SHA-256、职责/调用、公开等价物反证、许可证/来源和安全输出的只读审查。九项双审一致；三项经第三次只读裁决。

| 处置 | 数量 |
| --- | ---: |
| P6 执行契约 | 3 |
| P7 许可/隐私阻塞 | 9 |

三项 P6 会实际进行 GPU idle gate、TensorRT engine build/inspect/CUDA re-benchmark，或加载 ONNX、编译并执行 TVM recovery measurement。其余脚本含私有模型、workdir、结果、CUDA/PTX dump 或环境文件根目录绑定，公开前保持 P7。

没有运行候选、模型、数据、设备、子进程或网络任务。受限 evidence 和 HMAC 不进入公开树；本批新增 12 个身份后真实唯一覆盖为 **686/1,551**。
