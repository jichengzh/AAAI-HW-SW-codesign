# P3-137 论文、Predictor 与实验流程复审

状态：已接受（本地）；P3 继续进行中，不构成 P3 关闭或 P4--P7 启动授权。

本批 12 项 `migrate_document` 候选完成 P2 metadata/bytes、当前内容 SHA-256、职责/I-O/调用关系、公开等价物、来源/许可证与安全输出的双重逐项审查，并经第三方逐项裁决。最终为 P4=8、P5=1、P6=1、P7=0，另有 2 项 `duplicate_or_superseded`。q1 的公开 predictor/classifier/standard-conv API 与定向测试覆盖其 guardrail 和职责；q2 的公开 T1 attention TVM scanner、P/Q gate、mixed-INT8 row builder、Stop-A validator 及对应测试覆盖其职责，满足 duplicate 门槛。两项 superseded 文档仅保留为历史长版；GPU/授权 SOP 归环境前置，W_g 探针归执行管线，其余为论文或外部证据边界。未发现真实凭据、主机/IP 或可直接执行的访问秘密。

未运行候选、模型、数据、GPU、网络或外部服务；受限 evidence、逐项决策、候选 identity、HMAC key 不进入公开树。论文中的测量、图表、模型和数据主张仍须由后续外部契约约束，不能视为已复现。

本批新增 12 个冻结身份后，真实唯一覆盖为 **1,412/1,551**，剩余 **139** 项。P3 仍在进行中，P4 尚未开始；未推送、未发布。
