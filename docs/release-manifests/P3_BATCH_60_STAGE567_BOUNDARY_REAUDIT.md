# P3-60 Stage5--Stage7 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P6 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 482 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| 已安全改写 | 7 | Stage5--Stage7 package/selection/formal-plan/protocol 与 selection-only contract |
| P4 | 2 | closure feedback 与独立 measurement request 外部 evidence 契约 |
| P5 | 3 | executor admission、recovery root 与 full frozen/sidecar/pre-scan 环境状态契约 |

第 4、10、11、12 项经第三次只读裁决。独立 measurement request 本身不启动测量、GPU 或子进程，主导职责是绑定已有 rows 的外部证据契约，故归 P4。executor admission 及 recovery/root preparation 虽读取或写入本机状态，但不运行 executor 或实际测量，主导依赖是本机 Linux/process/file-system/deployment 环境，故归 P5。公开同路径的 Stage7 selection-only 代码未覆盖完整私有 frozen/sidecar/pre-scan/state 职责，故不能将整个候选误记为已改写，也归 P5。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖测量反馈、process identity、部署状态、sidecar、executor、硬件或其派生物。后续 P4/P5 必须使用匿名制品 ID、哈希、净化身份摘要、路径无关 schema、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **494/1,551**。
