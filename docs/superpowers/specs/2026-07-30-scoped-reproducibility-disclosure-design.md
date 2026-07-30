# AAAI-27 受限复现制品统一说明设计

## 目标

在 `README.md`、`README.anonymous.md`、`REPRODUCIBILITY.md` 和
`ARTIFACTS.md` 中采用一致、可审计的制品边界说明，使审稿人能够准确
区分：

1. 当前公开的输入校验、候选选择、聚合和审计接口；
2. 仅用于确定性冒烟测试的轻量实现与合成数据；
3. 尚未纳入当前制品的论文级候选生成、完整消融和硬件执行代码。

该说明不得把不同算法描述为等价实现，也不得暗示当前制品能够复现
未提供的论文数值结果。

## 统一口径

四份文档统一使用以下事实边界：

- 当前仓库是论文实现的受限复现子集，不是完整训练与硬件搜索代码。
- `predicted_frontier_diversity` 是用于检查候选选择与反馈接口的确定性
  轻量策略；它不是论文所述 NSGA-II 的替代实现，也不构成效果等价声明。
- 当前 Stage 7 模块只提供 selection-only 合同与统计接口，不能据此复现
  论文 Table 2 的完整搜索轨迹或消融数值。
- 当前制品不包含论文专用的完整候选生成器、完整消融流水线、模型训练/
  物化栈、硬件执行器和正式 terminal evidence。
- 完整实现、执行清单与相应实验制品计划在论文发表后公开；在此之前，
  缺失材料必须保持 `external` 或 `unavailable`，不得用 demo 或代理结果替代。

## 分文件修改

### `README.md`

在末尾增加 “Scope of the released implementation” 小节，面向公开 GitHub
用户说明完整代码与当前受限制品的区别，并明确轻量策略的用途及非等价边界。

### `README.anonymous.md`

补充最短但完整的匿名审稿人提示，避免现有
“contains the implementation modules” 被理解为完整论文实现。明确 smoke
只验证公开接口，不能复现论文硬件结果或 Table 2。

### `REPRODUCIBILITY.md`

在 evidence matrix 后增加方法映射说明：

- Stage 5 smoke 的轻量策略不映射为论文 NSGA-II 结果；
- Stage 7 公开代码不映射为论文完整消融；
- 当前唯一可验证的论文相关数值制品仍是已列出的 Stage 4 小型审计材料。

在 Known limitations 中同步完整代码发表后公开的计划。

### `ARTIFACTS.md`

在 “What is deliberately absent” 中列明论文专用候选生成器、完整消融
流水线及正式 trajectory/terminal evidence，并保持其状态为
`external`/`unavailable`。

## 防误导约束

下列表述禁止出现在发布文档中：

- “`predicted_frontier_diversity` 与 NSGA-II 效果相同/等价”；
- “`backend_blind` 与 `w/o graph features` 相同”；
- “当前仓库可以复现 Table 2”；
- “缺失代码不影响全部论文结果复现”。

允许说明轻量策略能够确定性地演示相同的接口形状、轮次预算、反馈边界和
输出结构，但必须同时说明其不验证论文候选生成算法或最终数值。

## 验收

1. 四份文档对公开范围、轻量策略、Stage 7 和发表后开放计划表述一致。
2. 匿名 README 不包含作者、组织、本地路径或开发历史。
3. 文档中的所有命令和相对链接继续有效。
4. 现有 identity scan、匿名归档和文档命令检查通过。
5. 搜索四份文档时，不存在算法等价、Table 2 已复现或完整实现已包含的
   无证据声明。
