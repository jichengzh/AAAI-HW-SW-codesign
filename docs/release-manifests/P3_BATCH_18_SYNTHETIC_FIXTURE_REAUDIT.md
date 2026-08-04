# P3-18：合成 fixture 发布级复审

状态：已完成。P2b 的 8 项 `fixture_candidate` 已逐项按 P3 重审协议复核，并形成仅本机保存的受限 evidence、decisions 和路径无关 ledger 检查点；公开树只加入已获维护者授权的脱敏小型 fixture 与其测试，未迁入数据、模型、checkpoint、engine、结果或真实硬件输入。

## 逐项结论

| 范围 | 数量 | 处置 | 复核结论 |
| --- | ---: | --- | --- |
| 合成 release-assets placeholder manifest | 1 | `migrated_public` | 仅使用 `example.invalid` 占位来源和 SHA-256 格式校验；它测试“正式运行前必须有外部制品契约”，不提供下载入口。 |
| Stage1 dense-core smoke manifests | 3 | `migrated_public` | 每项都明确为 dense-core / trace-net 范围、非完整模型测量，且保留缺 checkpoint 或未覆盖子图时的失败边界。 |
| Stage2 synthetic AP、classification、latency 输入 | 3 | `migrated_public` | 每项都有 `synthetic_fixture` 或 `fixture_only` 范围标记；仅供公开单元/集成测试。 |
| Stage1 anchor fixture | 1 | `rewritten_public` | 原文件名含“measured”；公开版本保留布尔测试语义并新增 `synthetic_fixture`、`fixture_only`、`paper_evidence: false`，防止被解释为论文证据。 |

所有 8 项在私有树中均为未跟踪的维护者授权合成 fixture。逐项记录了内容 SHA-256、原始/公开内容关系、测试消费者、授权结论、语义范围和敏感内容审查；这些逐路径信息仅保存于权限受限的本机审计目录。内容检查没有发现绝对路径、人员/机器标识、凭据、真实网络地址、真实数据或模型载荷。

## 公开验证

新增 [fixture contract tests](../../tests/stage1/test_migrated_fixture_contracts.py) 先在 fixture 缺失时失败，迁入后覆盖以下用户可见边界：

1. release-assets fixture 只能使用不可解析的占位来源，并绑定 SHA-256；
2. Stage1 manifest 只能是合成 dense-core / trace-net 范围，不能指向 `results/` 或声称完整模型延迟；
3. Stage2 输入和 anchor fixture 必须显式标为合成、fixture-only、非论文证据；
4. fixture tree 只能包含 8 个小型 JSON/YAML 文件。
5. 整个 fixture tree 会拒绝本机绝对路径、邮箱、IPv4 地址和凭据标记，并以代表性泄露值回归验证该拒绝规则本身。

定向 fixture 测试为 7 passed；Stage1 与 release 安全回归、Ruff、`compileall` 和 diff 空白检查均通过。

## 对 P4 的边界

本批的 release-assets 文件是测试 placeholder，不是数据或 checkpoint 的获取清单，因此不能作为 P4 完成证据。P4 仍等待对真实外部制品候选逐项确认来源、许可证、版本、checksum、用途和不可用失败关闭。

新的发布级 re-audit 已有 8/1,551 项可接受 decisions；此前撤回的机械分流不计入该数字。下一批将从真实外部制品候选开始，并按同一协议建立经过证据确认的 P4 工作集。
