# P3-9：Stage4 纯内存脱敏改写

状态：首个公开模块已完成；已完成本地账本演练和 `feedback_update_eval_v1` 脱敏重写；未推送、未发布。

## 本地账本演练

本批次从 P2b 的 1,551 项冻结清单中派生了仅含 9 个候选的本地演练子集。这 9 项均是当前公开仓库中已受 Git 跟踪的同路径、同 SHA-256 代码；公开仓库采用 Apache-2.0 许可。演练 decisions 将它们标记为 `duplicate_or_superseded`，固定使用 `already_public_same_hash` 和 `path_hash_comparison` 两个安全 token。

维护者本机在仓库外保存了权限为 600 的私有 inventory、decisions 和 48-byte HMAC key；目录本身为 700。由此生成的演练 ledger 含 9 条 HMAC-SHA256 候选 ID，逐项检查确认其不含原始路径、origin 或 key。演练 ledger 不提交、不发布，也不替代必须覆盖完整 1,551 项的最终 P3 ledger。

## Stage4 改写范围

P3-8 的无路径审阅识别出 26 个 Stage4 代码候选：19 个纯内存候选需要公开脱敏改写，3 个已有公开等价物，另有 4 个必须单独审阅。P3-9 不复制私有文件；计划按以下四个最小公共模块逐一重写：

1. feedback-update evaluation：before/after 指标、区间和 fold 角色；
2. selection completion：四臂完成度、ranker 拒绝和策略 replay 摘要；
3. ranking/Pareto：只处理内存中的排序与多目标摘要；
4. uncertainty replay：校准区间和 acquisition replay 摘要。

现有的 Stage4 selection 与 closure audit 已是公开等价物，本批不重复实现。每个新模块必须有合成输入测试、字段白名单、有限数值与身份绑定校验，并且不得读取文件、网络、GPU、子进程、结果目录、模型或硬件状态。

## 首个改写单元

首个单元是 `feedback_update_eval_v1`。它只接受内存中的规范行、before/after 预测和 canonical heads，生成兼容 Stage4 closure audit 的 `stage4_feedback_update_eval_v1` 报告。实现为纯标准库逻辑，不读取文件、网络、GPU、子进程、模型、结果目录或硬件状态。

该模块采用严格字段白名单、有限数值检查、四臂完整性检查、before/after 预测集合一致性检查、fold 角色绑定和匿名 ID 契约。`group_id` 只能为 `g<number>`，`manifest_job_id` 必须由对应 `group_id` 和公开四臂派生；非匿名组、非规范行 ID、路径/URI/root/cache/terminal/checkpoint 等额外字段、重复或缺失组、预测组绑定漂移、非有限数或非法区间均失败关闭。错误信息固定为 `invalid feedback input`，不回显调用方输入。

为保持纯模块导入边界，`framework.stage4.__init__` 已改为惰性导出 `run_nested_selection`。干净 Python 子进程只导入 `framework.stage4.feedback_update_eval_v1` 时不会加载 NumPy；需要旧入口 `from framework.stage4 import run_nested_selection` 时才按需加载 selection 依赖。

验证结果：`tests/stage4/test_feedback_update_eval_v1.py` 为 8 passed；新模块覆盖率 89%。连同 `tests/stage4/test_cost_model_selection.py` 与 `tests/stage4/test_closure_audit_v1.py` 的完整 Stage4 回归为 48 passed，确认惰性导出没有破坏原有 Stage4 selection/closure 接口。当前 HEAD 的全量验收为 431 passed、85% coverage；交接/身份/账本回归为 35 passed；Ruff、`compileall` 和 `git diff --check` 通过。匿名 ZIP 已 build/verify，含 118 个成员，SHA-256 为 `1b6bebbba902a735e551b2e5c018cf3cc82c3834c9cca44573eb41da25f71180`。

Stage5 contract 的真实输入升级、Stage6/Stage7 的正式证据和硬件执行仍保持 external/unavailable；本批次只完成第一个 Stage4 公开脱敏单元，不能替代完整 P3 ledger。

## 未改变的边界

本批次不公开真实测量、真实反馈行、模型、checkpoint、engine、ONNX、校准缓存、硬件日志、服务器配置或生成物。没有完整 decisions/ledger、P4 输入契约、P5 环境契约和 P6 正式执行 manifest 时，P3 仍不能关闭。
