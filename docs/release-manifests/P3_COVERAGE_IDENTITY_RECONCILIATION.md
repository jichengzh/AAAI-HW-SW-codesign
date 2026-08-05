# P3 覆盖身份对账

状态：已完成（本地）；这是 P3 完整性闸门，不是 P3 关闭，也不授权启动 P4。

## 对账方法

对 P3-18--P3-42 的每一份已接受受限 inventory，以冻结 P2b 的身份键 `(origin, path)` 去重。该键只在本机受限记录中处理；公开文档不披露任何候选路径或内容。每份 inventory 先与其公开 HMAC ledger 的 inventory SHA-256 和候选数对账，再计算跨批次的交集。

## 结果

| 指标 | 数值 |
| --- | ---: |
| 已接受 records（按批次相加） | 289 |
| 不重复的冻结 P2b 身份 | 278 |
| 重复身份 | 11 |

重叠只出现在此前已接受的复审批次之间：P3-19/P3-36 为 3 项、P3-20/P3-36 为 5 项、P3-20/P3-37 为 1 项、P3-20/P3-40 为 2 项。后续批次的逐项审阅记录仍保留为审计证据，但不再增加 P3 的覆盖分子，也不能替代尚未审阅的候选。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-123 从 P3-122 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,244/1,551**，尚余 **307** 项文档候选。3 项 P4/P6 职责分歧均完成第三方只读裁决：checkpoint blocker、builder readiness、量化核对和已测 rows/telemetry/full-engine gate 的制品、scope redline 与验证记录归 P4；checkpoint recovery、多卡训练/评测、scale-aware route 修复与 completion queue/runner 编排归 P6。另有 1 项真实凭据和私有运行信息暴露保持 P7；该项需轮换失效确认、脱敏和泄露回归后重审，本记录不声称已解除。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-122 从 P3-121 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,232/1,551**，尚余 **319** 项文档候选。3 项 P4/P6 职责分歧均完成第三方只读裁决：checkpoint/route inventory 若由监控、gate 后导入和恢复编排主导则归 P6；已测 capture、smoke、numeric/trace blocker 与 readiness 制品为主导时归 P4。full-AP 监控、bridge 修复、coverage 刷新、checkpoint recovery 与 AP/energy 补点均归 P6。占位主机、环境变量和无变量名的脱敏片段不构成 P7，未发现真实凭据或连接秘密。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-121 从 P3-120 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,220/1,551**，尚余 **331** 项文档候选。7 项真实职责分歧均完成第三方只读裁决：已测 INT8 route/trace、数值与 sanity blocker、FP16 AP 来源、calibration/reference-range 制品、reference-range 实测和 calibrated AP smoke/row gate 归 P4；energy/AP 补点、completion queue/simulator、hook/reference-range capture、AP 收口和 latency 补点的执行编排归 P6。独立审查中的协议外标签未被采用；未发现真实凭据或连接秘密，故无 P7。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-120 从 P3-119 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,208/1,551**，尚余 **343** 项文档候选。10 项职责分歧均完成第三方只读裁决：已测 AP/INT8 状态、route 和 blocker 证据归 P4，实际量化补点、AP 收口、worker bridge、重测和 full AP gate 计划归 P6。未发现真实凭据或连接秘密，故无 P7。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-119 从 P3-118 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,196/1,551**，尚余 **355** 项文档候选。10 项职责分歧均完成第三方只读裁决：量化/精度/AP 的受控测量、状态表、gap 报告和制品索引归 P4，native INT8 硬件路线归 P5，实际量化生成、AP 收口和 runner 修复归 P6。占位主机、环境变量、登录规则引用和连接失败文本不足以 P7；未发现真实凭据或连接秘密。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-118 从 P3-117 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,184/1,551**，尚余 **367** 项文档候选。6 项职责分歧均完成第三方只读裁决：实际 AP replay/微调、评估和生产门禁归 P6，纯环境阻塞归 P5；泛化 PID 写法不足以 P7，明确远程登录、临时认证和访问命令归 P7。P7 文档经拆分和脱敏后才可重新按剩余主导职责审核。`migrate_code` 与 `migrate_config` 已穷尽，后续仍只按 `migrate_document` 的确定性顺序继续。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-117 从 P3-116 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,172/1,551**，尚余 **379** 项文档候选。9 项职责分歧均完成第三方只读裁决：受控测量、论文证据、制品登记和设计口径归 P4，实际调优/生成/验证交接归 P6；真实私有路径、进程标识或远程连接/认证流程归 P7。设计文本、泛化硬件词或进展文档不单独决定处置，必须由主导职责和实际内容决定。`migrate_code` 与 `migrate_config` 已穷尽，后续仍只按 `migrate_document` 的确定性顺序继续。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-116 从 P3-115 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,160/1,551**，尚余 **391** 项文档候选。6 项职责分歧均完成第三方只读裁决：论文/实验和硬件文献证据归 P4，平台能力、工具链与 TVM 硬件 gate 归 P5，实际模型/编译/评测流水线计划归 P6；泛化硬件路径不足以 P7，确认含未脱敏私有路径引用的多模型探测计划归 P7。`migrate_code` 与 `migrate_config` 已穷尽，后续仍只按 `migrate_document` 的确定性顺序继续。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段及后续旧更正的当前计数）**：P3-115 从 P3-114 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,148/1,551**，尚余 **403** 项文档候选。9 项职责分歧均完成第三方只读裁决：论文/实验制品和研究资料边界归 P4，闭环仿真编排归 P6，私有远程访问参数和口令式认证流程归 P7；只有具有公开实现与测试/API证据的 fixture 或桥接职责才关闭为已改写或已替代。旧版归档导航只有在无当前公开支持链、且其维护性职责已由既有公开发布、复现材料和测试承担时才可排除，归档中的实际研究证据仍留在 P4。`migrate_code` 与 `migrate_config` 已穷尽，后续仍只按 `migrate_document` 的确定性顺序继续。P3 尚未关闭，P4 尚未开始。

**当前更正（覆盖下段的当前计数）**：P3-114 从 P3-113 后的唯一身份集合之外新增 12 个 `migrate_document` 身份。受限 inventory、当前 SHA-256、规范 HMAC ledger 与 acceptance manifest 均已重算；真实唯一覆盖现为 **1,136/1,551**，尚余 **415** 项文档候选。4 项职责分歧均完成第三方只读裁决：外部实验/论文制品归 P4，实际修复或 GPU lane 编排归 P6；含未脱敏私有路径引用或外部执行引用的文档归 P7。`migrate_code` 与 `migrate_config` 已穷尽，后续仍只按 `migrate_document` 的确定性顺序继续。P3 尚未关闭，P4 尚未开始。

因此，截至 P3-42，P3 的真实唯一覆盖为 **278/1,551**。P3-43--P3-103 各从当时的唯一身份集合之外新增 12 个身份，P3-104 再新增最后 8 个 `migrate_code` 身份，P3-105--P3-111 各新增 12 个 `migrate_config` 身份，P3-112 再新增自然剩余的 10 个 `migrate_config` 身份，P3-113 再新增 12 个 `migrate_document` 身份，当前真实进度为 **1,124/1,551**。P3-50 入账前重建验证了全部 373 条此前 accepted records：每条均与冻结 P2 metadata、当前 SHA-256 和规范 HMAC ledger 对齐，去重结果仍为此前的 362 个身份；P3-50--P3-103 各再新增 12 个身份，P3-104 新增 8 个，P3-105--P3-111 各新增 12 个，P3-112 新增 10 个，P3-113 新增 12 个。P3-95 的一次身份不一致独立复审不计入覆盖；重审后的同一固定队列才纳入账本。P3-97 的 4 项分歧均经第三次只读裁决后才入账，其中一项非法处置/后续阶段组合已拒绝。P3-98 的 6 项分歧同样经第三次裁决，其中两项因不存在真实公开实现和测试而不得提前记为已迁入/改写。P3-99 的 8 项分歧亦经第三次裁决，五项非法 `excluded_nonessential → P7` 草案已拒绝。P3-100 的两项上游许可证/职责分歧经第三次裁决：Apache-2.0 来源证据充分，仍按 P4/P6 而非 P7 入账。P3-101 的 10 项分歧亦经第三次裁决：真实 build/run/tune/measure/calibrate 行为归 P6，本地路径/输出风险不得替代 P7。P3-102 的 7 项分歧经第三次裁决：没有公开实现和测试链的计划/私有测试归 P4，实际检查/评测/导出/测时归 P6。P3-103 的 5 项分歧经第三次裁决：纯报告解析与未验证等价性制品归 P4，实际 Orin/TensorRT/模型测量归 P6。P3-105 的 12 项边界分歧均经第三次裁决：硬件能力/工具链配置归 P5，真实剪枝/量化/评测流水线输入归 P6，无产品调用链的内部审查收据归非必要排除。P3-106--P3-110 的配置边界分歧同样经第三次裁决：主动采样、剪枝、量化路由、剪枝预设、decoder 控制面及量化模型变换配置直接控制执行链，均归 P6。P3-111 的 12 项分歧亦经第三次裁决：量化执行输入归 P6，逐层校准尺度制品归 P4。P3-112 的 10 项分歧经第三次裁决：量化执行配置归 P6，数据转换/标签更新制品归 P4。P3-113 的 4 项分歧经第三次裁决：量化执行和进展文档归 P6，公开复现边界归 P4，瞬态交接记录因无支持链且维护性职责已由既有公开发布状态与复现边界材料承担而排除；确认未脱敏私有路径引用的文档归 P7。`migrate_code` 与 `migrate_config` 均已穷尽，后续仅按 `migrate_document` 的确定性顺序继续；任何相加计数都不得作为 P3 关闭依据。
