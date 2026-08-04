# P3-9：Stage4 纯内存脱敏改写

状态：四个最小公开模块均已完成；已完成本地账本演练、P3-10 的差异候选审计与 Stage4 脱敏改写；未推送、未发布。

## 本地账本演练

本批次从 P2b 的 1,551 项冻结清单中派生了仅含 9 个候选的本地演练子集。这 9 项均是当前公开仓库中已受 Git 跟踪的同路径、同 SHA-256 代码；公开仓库采用 Apache-2.0 许可。演练 decisions 将它们标记为 `duplicate_or_superseded`，固定使用 `already_public_same_hash` 和 `path_hash_comparison` 两个安全 token。

维护者本机在仓库外保存了权限为 600 的私有 inventory、decisions 和 48-byte HMAC key；目录本身为 700。由此生成的演练 ledger 含 9 条 HMAC-SHA256 候选 ID，逐项检查确认其不含原始路径、origin 或 key。演练 ledger 不提交、不发布，也不替代必须覆盖完整 1,551 项的最终 P3 ledger。

## Stage4 改写范围

P3-8 的无路径审阅识别出 26 个 Stage4 代码候选：19 个纯内存候选需要公开脱敏改写，3 个已有公开等价物，另有 4 个必须单独审阅。P3-9 不复制私有文件；下列四个最小公共模块均已按独立公开契约重写：

1. feedback-update evaluation：before/after 指标、区间和 fold 角色；
2. selection completion：四臂完成度、ranker 拒绝和策略 replay 摘要；
3. ranking/Pareto：只处理内存中的排序与多目标摘要；
4. uncertainty replay：校准区间和 acquisition replay 摘要。

现有的 Stage4 selection 与 closure audit 已是公开等价物，本批不重复实现。每个新模块必须有合成输入测试、字段白名单、有限数值与身份绑定校验，并且不得读取文件、网络、GPU、子进程、结果目录、模型或硬件状态。

## 四个改写单元

`feedback_update_eval_v1` 只接受内存中的规范行、before/after 预测和 canonical heads，生成兼容 Stage4 closure audit 的 `stage4_feedback_update_eval_v1` 报告。实现为纯标准库逻辑，不读取文件、网络、GPU、子进程、模型、结果目录或硬件状态。

该模块采用严格字段白名单、有限数值检查、四臂完整性检查、before/after 预测集合一致性检查、fold 角色绑定和匿名 ID 契约。`group_id` 只能为 `g<number>`，`manifest_job_id` 必须由对应 `group_id` 和公开四臂派生；非匿名组、非规范行 ID、路径/URI/root/cache/terminal/checkpoint 等额外字段、重复或缺失组、预测组绑定漂移、非有限数或非法区间均失败关闭。错误信息固定为 `invalid feedback input`，不回显调用方输入。

`selection_completion_v1` 只接受匿名完整四臂测量与严格 selection manifest；它验证候选行、折绑定、每组唯一选择、策略白名单和 replay 指标，只输出可供 closure audit 消费的匿名汇总，明确拒绝保留 ranker。`ranking_pareto_v1` 只接受预测字段，不接受实际标签或测量；它对匿名四臂候选做确定性多目标 Pareto 与排序，并把二次 Pareto 输入限制为最多 256 组、1,024 候选。`uncertainty_replay_v1` 仅处理绑定的 OOF 与无标签候选预测，输出校准 MAE、coverage、区间宽度和匿名 acquisition replay 汇总。

四个模块均采用严格字段白名单、有限数值与匿名 ID 绑定，并拒绝路径、URI、root、cache、checkpoint、terminal status、真实标签泄漏和不受限序列。三个新增模块以合成输入执行无副作用测试，明确封锁文件、网络和子进程入口；全部结果均为深层不可变或只读的匿名报告。为保持纯模块导入边界，`framework.stage4.__init__` 已改为惰性导出 `run_nested_selection`。干净 Python 子进程只导入 `framework.stage4.feedback_update_eval_v1` 时不会加载 NumPy；需要旧入口 `from framework.stage4 import run_nested_selection` 时才按需加载 selection 依赖。

当前定向验证：四个新增/重写模块与 closure audit 为 91 passed；完整 `tests/stage4` 回归为 119 passed。`feedback_update_eval_v1`、`selection_completion_v1`、`ranking_pareto_v1` 与 `uncertainty_replay_v1` 的单元覆盖率分别为 89%、88%、90% 和 90%，合计 89%。Ruff、`compileall`、`git diff --check`、交接/身份/账本回归（35 passed）均通过。当前候选 HEAD 的全仓验收为 502 passed、86% coverage；匿名 ZIP 已 build/verify，含 124 个成员，SHA-256 为 `a6951ed6310541a0266a14d6af93228fe20313f93f41508231b773e54644d264`。依赖审计没有发现已知漏洞；CPU 专用 `torch 2.9.0+cpu` 不在 PyPI 索引，因而不能由该审计器覆盖。旧的 431 passed/85% 及 ZIP SHA-256 仍仅是前一 HEAD 的历史证据。

Stage5 contract 的真实输入升级、Stage6/Stage7 的正式证据和硬件执行仍保持 external/unavailable；本批次只完成第一个 Stage4 公开脱敏单元，不能替代完整 P3 ledger。

## 未改变的边界

本批次不公开真实测量、真实反馈行、模型、checkpoint、engine、ONNX、校准缓存、硬件日志、服务器配置或生成物。没有完整 decisions/ledger、P4 输入契约、P5 环境契约和 P6 正式执行 manifest 时，P3 仍不能关闭。
