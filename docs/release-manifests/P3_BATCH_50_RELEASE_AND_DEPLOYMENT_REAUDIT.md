# P3-50 release 与 deployment 边界复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P7 启动授权。

本批从冻结 P2b 基线中确定性选取 12 个、且不在此前 362 个唯一已覆盖身份中的代码候选。所有候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/调用/公开反证、许可证/来源和安全输出的只读审查。

| 处置 | 数量 | 主导责任 |
| --- | ---: | --- |
| P4 | 3 | release asset integrity 与 checkpoint alias 的外部制品契约 |
| P6 | 5 | 实时 deployment recovery、takeover 与 source sidecar 执行契约 |
| `duplicate_or_superseded` | 2 | 已由公开 clean-clone/synthetic smoke API、测试和支持链替代 |
| `blocked_license_or_permission` | 2 | 外部 CUDA 源码 patcher 的许可与隐私阻塞 |

第三次只读裁决确认：公开 archive/manifest 测试与 release-asset verifier 相邻但职责不等价，后者仍需要 P4 的制品来源、哈希和可发布范围契约；completed-round rehydrate 与 dependency-pin replication 的环境检查只是准入保护，其主导职责是实时部署事务、状态变更、bundle 验证与回滚，故均归 P6。两个外部 CUDA patcher 会修改非公开目标源码并包含私有绝对根；在上游许可、可再分发性、路径无关重写和安全输出审查完成前，必须保持 `blocked_license_or_permission`，不得排除。

候选源码的项目受控部分继承 Apache-2.0；该许可不覆盖 checkpoint、dataset、部署 bundle、运行时审计、外部 CUDA 源码、补丁目标或其派生物。后续 P4/P6/P7 必须使用路径无关引用、显式输入契约、受控输出根和失败关闭。本批没有运行候选代码、模型、数据、设备、子进程或网络任务；受限 records 与 HMAC key 不进入公开树，HMAC ledger 已重算。本批新增 12 个不重复身份后，P3 的真实唯一覆盖为 **374/1,551**。
