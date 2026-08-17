# P6.2 动态框架候选空间设计

## 目标

把现有 `stage2_search_space_v1` 的 Pyramid/H800/TVM 搜索空间直接接入 CoptV2X 的完整四轮搜索。候选集合必须由 Stage2 的实际合法输出决定，不得要求旧 P6.1 静态网格的 343 个结构或 686 个候选。

本设计只改变 P6.2 的 framework-source 路径。P6.1 已完成的静态 343×2 本地闭环仍是历史执行事实，不被改写。

## 既有方法口径

1. Stage2 对每个 Pyramid dense stage 按 manifest 的合法宽度、`round_to`、`max_rate`、硬件锚点与 INT8 可构建性产生软件点；每阶段的点数可以变化。
2. 一个 CoptV2X 候选有一个全局 `q_mode`。候选基因为三段 `width` 加 `q_mode`；单个候选不是逐 stage 的 mixed-precision 配置。
3. 对每个全局 `q_mode`，将三个 stage 的 active、buildable 点做笛卡尔积。FP16 与 INT8 两个集合独立生成，再合并为候选池；不要求每个宽度同时具备两种精度。
4. `predicted_frontier_diversity` 是候选预测后的每轮选点策略，不是搜索空间预剪枝。不得另加静态 FLOPs、代理时延、宽度范围或 Pareto 规则来删除合法候选。

因此，候选池可同时含纯 FP16 候选和纯 INT8 候选，但不会产生“一个候选内 FP16/INT8 混合”的新方法。

## 范围与不变量

- 固定实验协议不变：H800、TVM、Pyramid、Gold176、4 轮、每轮 4 个真实测量槽位，共 16 个候选。
- Gold176 的 176 条冻结输入重新训练 cold-start cost model；174 条成功行进入回归，两条真实可行性失败保留为证据但不伪造指标或重测。
- Round 0 使用 cold-start bundle；后续轮以 Gold 成功行和已验证 online feedback 重训，再预测当前所有未测候选并选择 4 个。
- framework-source 候选清单在 round 0 之前必须含至少 16 个未测、可物化候选；否则失败关闭，且不得启动测量适配器。
- public contract 继续不含路径、候选 ID、命令、原始指标或本地结果。真实输入、local adapter、计划、registry、请求、反馈与产物保持 Git 忽略。
- 不下载资产、不自动发现硬件、不自动启动 Orin、TensorRT 或其他后端。

## 数据契约

### 动态候选计划

将当前 `p6_pyramid_structure_plan_v1` 替换为局部计划 `p6_pyramid_candidate_plan_v2`：

```json
{
  "schema_version": "p6_pyramid_candidate_plan_v2",
  "source_schema": "stage2_search_space_v1",
  "target_model": "pyramid",
  "hardware_target": "h800",
  "execution_backend": "tvm_auto",
  "candidate_source_mode": "framework_stage2_search_space",
  "structure_count": 0,
  "candidate_count": 0,
  "candidates": [
    {
      "width": [0, 0, 0],
      "q_mode": "fp16",
      "source_point_ids": ["stage1-point", "stage2-point", "stage3-point"]
    }
  ]
}
```

上例中的零值和点名只是 schema 形状说明，不是允许的候选值。实际计划必须按 `(width, q_mode)` 排序、去重，并只含 `fp16` 或 `int8`。`structure_count` 是候选中不同 `width` 的数量；`candidate_count` 必须等于 `candidates` 长度，二者不带预设常数。

### 本地 source registry

framework materializer 产出的本地 registry 使用新 schema 版本，仍按结构 group 保存 graph/source contract，但显式带有：

- `available_q_modes`：该结构实际可物化的全局精度集合；
- `source_point_ids_by_q_mode`：每个可用精度对应的三个 Stage2 点来源；
- 与计划完全相同的 `(width, q_mode)` 身份集合。

静态 P6.1 registry 保持原 schema 及其“每个 group 展开 FP16/INT8”的兼容行为。Stage5 manifest 在读取 framework registry 时只能展开 `available_q_modes`，不能凭空补充另一精度。

## 控制器行为

framework mode 的顺序为：

```text
Stage2 search space
  -> dynamic candidate plan
  -> Git-ignored local registry materializer
  -> registry/plan exact-identity validation
  -> Stage5 dynamic candidate manifest
  -> plan/manifest exact-identity validation and >=16 preflight
  -> Gold176 fit, predict, select 4, measure, feedback
  -> repeat four rounds
```

控制器在 framework mode 只使用动态计划的精确候选身份检查，不检查 343/686。静态 mode 继续保留 343/686 门禁，以保护 P6.1 已完成的历史协议。若 local materializer 漏掉、重复、替换或新增任一计划候选，或者合法候选数不足 16，控制器以稳定的 `source_registry_invalid` 在测量前失败。

## 测试与验收

1. adapter 测试验证 Stage2 非 7×7×7 输出得到相应动态候选数，FP16 与 INT8 独立笛卡尔积，且不要求 q-mode 成对。
2. adapter 拒绝非法模型、非 H800、缺少 tuned TVM、未知量化策略、active-but-nonbuildable 点、重复 `(width, q_mode)` 或不存在的 stage 来源。
3. Stage5 测试验证 framework registry 仅展开实际 `available_q_modes`，静态 registry 仍展开两种精度。
4. controller 黑盒测试验证非 343/686、但不少于 16 的 framework candidate pool 可以完成四轮合成反馈；计划/registry/manifest 任一身份漂移或少于 16 个候选会在测量前拒绝。
5. 保留 P6.1 静态 343/686 回归，证明动态改造不改变历史执行路径。
6. release/document 测试声明：P6.2 动态离线改造进行中，真实 H800 框架来源闭环待离线验证成功后执行；不得把这一步升级为论文结果或 P6 总体关闭。

真实 H800 启动前必须通过完整离线质量套件、公开边界扫描、动态 local-config 预检和 registry/manifest 计数预检。真实运行成功条件是 4 轮均有匹配反馈、16 个候选均被正确计量、控制器写出 completed 状态；结果仍只保留在 Git 忽略的本地边界。

## 非目标

- 不发明 mixed-precision 或逐 stage quantization candidate。
- 不增加候选空间预剪枝策略。
- 不改变 P6.1 静态闭环、Gold176 证据、轮数、批量大小、测量指标或 acquisition policy。
- 不执行真实 H800、训练、下载、推送或合并，直到离线改造通过并进入已授权的实际启动阶段。
