# P3-150–153 Pending migrate_code 缺口复审

状态：历史缺口处置已接受（本地）；本记录形成于 P3 全量关闭前，不构成 P4--P7 启动授权。当前阶段状态以 [P3 全量关闭复核](P3_FULL_CLOSURE_REVALIDATION.md) 为准。

覆盖计算复核发现，旧选择器曾把 P3-28/29/30/33 的 pending inventory 当作已覆盖，导致 48 个 `migrate_code` 候选没有正式 acceptance manifest。该缺口已纠正并重新建立 P3-150–153 四个正式队列；不沿用 pending ledger 作为处置证据。

- P3-150：q1 因字面私有环境与临时目录引用保持 P7；q2–q12 由公开 API、职责和测试对比支持 `rewritten_public`。
- P3-151：12 项均由公开实现、API 职责和测试覆盖支持 `rewritten_public`。
- P3-152：q1 为论文产物生成执行契约 P6；q2–q12 由公开 API/测试支持 `rewritten_public`。
- P3-153：12 项测试源码均使用合成 fixture/路径断言，全部由公开测试职责支持 `rewritten_public`；不存在真实私有数据读取。

四批共 48 项均通过 P2 metadata/bytes/current SHA 绑定、双审与第三方裁决（有分歧时）、受限 ledger 和 acceptance manifest。未运行候选、测试、构建、模型、数据、GPU、网络或外部服务。

缺口补齐后，P2b 全部 **1,551/1,551** 候选均有正式 acceptance manifest；该批形成时 P3 全量关闭验证仍待完成，当前状态以全量关闭复核为准。
