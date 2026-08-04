# P3-36：Stage2 外部执行线索逐项裁决

状态：已接受（本地）。本批 12 个候选均经主审、独立语义审阅和第三次只读裁决。每次读取均核对 P2 metadata/字节数与队列绑定的当前 SHA-256，内容漂移为 0；没有执行私有脚本、模型、数据、设备、子进程或网络。

## 分歧处理

主审将全部非 P4/P5 候选转入 P6；独立审阅认为其中两个已有公开职责等价物。第三次裁决逐项比较公开实现、核心 API、输出安全约束和公开测试，而非比较文件名或字节哈希。两个公开同路径实现保留 B4 ablation 的统计/收敛/报告责任和 three-arm search 的核心 API；公开 demo pipeline 覆盖其可复现的安全输入与结果生成。因此这两项可作为 `duplicate_or_superseded` 接受。

公开 demo pipeline 定向回归为 9 passed。该回归只证明两个安全公开职责的覆盖，不证明真实模型、测量、硬件或论文结果可复现。

## 逐项结论

| 职责类别 | 数量 | 处置 | 裁决结论 |
| --- | ---: | --- | --- |
| 实测表、artifact registry、LUT productization、AP repair、large-scale readiness | 5 | `external_contract_p4` | 主导依赖为真实测量行、ONNX/engine/checkpoint、制品 registry 或 evidence/readiness 表。P4 必须先建立公开 schema、脱敏固定输入、许可/下载说明和 SHA-256 校验。 |
| pyramid fusion 与 Orin finalization | 2 | `environment_contract_p5` | 主导依赖为硬件、GPU/部署环境或平台状态。P5 应提供环境规范、能力探测和失败关闭验证。 |
| whole-subnetwork measurement、Stage6 control request、LUT productization CLI | 3 | `execution_contract_p6` | 主导职责是测量、控制请求或 CLI/作业执行，不能直接迁入默认公开 CI。 |
| B4 ablation 与 three-arm search | 2 | `duplicate_or_superseded` | 已有公开同路径安全实现和公开 demo pipeline 测试/API 证据覆盖核心职责；私有差异不再迁入。 |

根许可证信号为 Apache-2.0；它不覆盖真实模型、checkpoint、ONNX、engine、数据、测量制品或运行环境的再分发条件。所有候选均保留受限的职责、调用、来源/许可证和安全输出记录；路径无关 HMAC ledger 可重新计算，但不提交，也不构成 P3 全量关闭账本。

本批没有迁入源码、测试、数据、模型、checkpoint、ONNX、engine 或实验结果。它使已接受重审总数达到 217/1,551；P4 尚未开始。
