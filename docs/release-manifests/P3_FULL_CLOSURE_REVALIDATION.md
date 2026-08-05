# P3 全量关闭复核记录

状态：已完成（本地）。本记录所述验证与本地关闭提交绑定；P4--P8 未启动，未推送、未发布，也未运行真实模型、数据、GPU、网络或外部服务。

## 冻结输入与受限复核产物

本次复核继续使用 P2b 冻结盘点 `6fafdace7dc0eae3e9982763cb1c5c098c0ae83d1715af67ff1e08d7a1d12a79`，共 1,551 项。受限目录中的三份产物分别保存：逐项当前内容绑定、去重后的本机 decisions，以及不含路径的 HMAC 公开 ledger；它们不会进入 Git。

- `p3_full_current_binding_revalidation.json`：1,551/1,551 个源文件存在，字节数与 P2b 一致，并为每项重新计算当前 SHA-256；产物 SHA-256：`d12c7e983fa20e2bddb04f1d5d9cfd2d5203e43a081c7a78b15b09b0467e2d30`。
- `p3_full_item_revalidation.json`：1,551 条逐项复核记录，包含 P2 分类/字节数、当前 SHA-256、职责、调用/输入/输出、公开等价物、来源/许可证、安全审查、处置、理由、证据 token 和后续阶段；产物 SHA-256：`8a52a66600fdc738afd6cc8f84d15cffabb817179375cfd74649c978374322d0`。
- `p3_full_public_ledger_current.json`：由当前 P2b inventory、去重后的 1,551 条 decisions 和受限 HMAC key 重新生成；产物 SHA-256：`6b41fbf7e2af5faceb32b640c1481557cb2657d4a5441da999996ee0c7b3c55a`。

P2b 冻结文件本身没有逐条历史 SHA 字段，因此本次关闭证据将“P2 分类与字节数绑定”与“当前内容 SHA-256”分开记录；不能把缺少冻结 SHA 解读成当前内容未核验。

## 对账结果

按 `(origin, path)` 去重后，唯一覆盖为 **1,551/1,551**，缺失 0，额外身份 0；批次文件共保留 1,562 条逐批记录，重复身份 11 条。旧的 P3-28/29/30/33 pending inventory 不计入 accepted-only 集合；其中 48 项代码候选已在 P3-150--153 重新逐项复核并纳入正式 acceptance。

最终去重 ledger 的处置计数为：P4 506、P5 45、P6 640、P7 144、`duplicate_or_superseded` 47、`excluded_nonessential` 79、`migrated_public` 11、`rewritten_public` 79。

历史 evidence 中有 20 条采用旧字段名（fixture 的 `semantic_scope`/`license_conclusion`，以及早期代码记录的 `license_review`）；本次受限逐项复核将这些字段显式映射到协议字段，同时以当前源文件重新计算 SHA-256。该映射不改变任何 disposition，也不把 pending 记录升级为 accepted。

## 关闭门禁结果

独立关闭复审发现并修复公开记录的门禁清单不完整、ledger 状态矛盾、P3-149 覆盖表述过期，以及受限临时路径回显。修正后的工作树已完成以下全部门禁：

- 受限 P2b 当前内容、逐项协议字段与 HMAC ledger 均为 1,551/1,551；本机 `build_ledger()` 重算与受限路径无关 ledger 完全一致。
- 文档身份/凭据扫描和公开审计/manifest 文档的绝对路径扫描均为零命中；release 定向回归为 81 passed。
- CPU-only 全量 `pytest` 为 509 passed，`framework` 与 `scripts/reproduce` 总覆盖率为 83.09%，满足 80% 门槛。
- Ruff、`compileall`、已暂存/未暂存的 `git diff --check` 均通过。
- 匿名 ZIP 本地 build/verify 通过：133 个成员，SHA-256 `306c3892f2b7f64fbb32a02b3fe3bd9ba03025afa2fcc4e0a67dcf662c2d8ee0`。
- 独立关闭复审已完成；其四项发现均已修复并重新扫描。人工差异审查确认本次提交仅包含 P3 审计/关闭文档，没有受限 evidence、candidate identity、HMAC key、模型或生成物。

所有命令均在离线 CPU 边界执行，不运行真实模型、数据、GPU、网络或外部服务。

## 关闭判断

P2b 冻结基线的 1,551 项均有按协议绑定的逐项处置，所有关闭门禁和独立复审均已完成；因此 P3 在本地关闭提交中完成。144 项 P7 许可/权限阻塞继续保留为后续公开准备事项，不被视为已解除，也不允许启动 P4--P8。任何 `git push`、发布、可见性变更、tag、Release 或真实实验仍需维护者另行明确授权。
