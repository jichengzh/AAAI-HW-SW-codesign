# P6 历史执行根归一桥接设计

## 目标

让已有的 H800 历史资产在不公开路径、设备编号、checkpoint、数据或测量结果的前提下，能够生成可由 P6.4 消费的私有 binding 与 local YAML，并先完成真实 Stage1→Stage2 预检。

## 现状与问题

P6.4 的公开控制器已经实现了动态候选、Gold176 冷启动和四轮每轮四条实际反馈。但历史 H800 执行树不是一个可直接验证的 P6 root：其 Git 元数据不可用，输入分散，旧结果中存在重复组件，且没有可执行的 P6 Stage1 JSON launcher。

不能通过手写一个看似完整的 YAML 绕过这些问题。这样的配置会在 Stage1 或 source-registry 阶段失败，也不能证明是完整闭环。

## 架构

新增两个公开、无状态的桥接边界，所有具体定位留在 Git 忽略的私有 source-map 与 runner 文件中：

1. `build_p6_stage1_manifest.py` 在已激活的私有环境中调用既有 `framework.stage1.graph_scan`，以 Pyramid LiDAR、H800 capability 和私有模型资产输出 JSON 格式的 `stage1_partition_manifest_v1`。它不选择候选数量；Stage2 仍从扫描产物自主导出 active/buildable 的 FP16 与 INT8 组合。
2. `p6_history_normalization_v1.py` 将显式选定的一份历史 source contract 与一个动态物化 recipe 规范化为唯一 `stage5_candidate_source_registry_v1`。它不读取训练数据、checkpoint 或结果；只校验由私有 source-map 提供的结构化元数据、相对输出模板及来源证据。

私有部署侧建立一个受控 root：它是有效 Git worktree，内含四个规范化输入副本、唯一 source registry、以及对冻结历史 Stage5 组件和 Stage1 launcher 的显式绑定。大模型、训练数据和历史结果不复制、不移动；它们仍在既有私有资产位置，并仅以受控 root 内的私有路径供 runner 使用。

## 数据流

```text
private source-map + existing assets
  -> normalized private root (ignored)
  -> Stage1 bridge emits JSON manifest
  -> existing Stage2 loader derives dynamic plan
  -> existing registry adapter materializes registry-v2
  -> existing measurement adapter executes four rounds × four candidates
```

Stage1/Stage2 预检必须在任何 Stage5 测量之前完成，且必须证明：manifest schema/status 正确、动态 plan 至少包含 16 个 eligible candidates、registry-v2 与 plan identity 一致。只有通过这些门后才允许启动真实四轮。

## 私有输入契约

私有 source-map 只存在于 H800 的 Git 忽略位置，至少提供：

- Pyramid LiDAR 的 Stage1 repo、HEAL/模型配置与 checkpoint 根、H800 capability 和环境 activation argv；
- 三个已确认的资产目录与 Gold176、graph features、capability profile、Stage4 closure 的唯一来源文件；
- 冻结历史 Stage5 组件的明确路径、一个不含测量结果的基线 source contract，以及包含训练/checkpoint/ONNX/calibration 相对模板的动态物化 recipe；
- 私有输出根与 runner 的三卡 CUDA policy。

source-map、runner template、binding、local YAML、Stage1 manifest、registry、日志、结果与候选 ID 均不得进入 Git 或公开产物。

## 失败语义

- source-map 中的路径不存在、越界、为软链接、或不能归入单一私有 Git root：稳定失败，不写 binding/config；
- Stage1 scan 输出不是可加载的 `stage1_partition_manifest_v1`：`stage1_scan_invalid`；
- registry 基线契约或 recipe 不完整、输出模板碰撞、或可用候选不足 16：`source_registry_invalid`；
- GPU policy、UUID、型号或占用发生漂移：沿用现有 binding/measurement 准入失败；
- 任一预检失败都不启动 Stage5、训练或真实测量。

## 不做的事

- 不固定或公开任何物理 GPU 编号、私有路径、资产 ID、候选 ID 或测量值；
- 不下载、安装或重建 TVM；
- 不复制或迁移训练数据、checkpoint 与大型历史结果；
- 不把历史 Stage7 的旧 registry 伪装成 P6 registry-v2；
- 不改变 P6 的 Gold176、TVM、四轮、每轮四条、16 次实际测量口径。

## 验证

公开测试覆盖：Stage1 bridge JSON 产物、错误输出不落盘、normalizer 的 recipe/来源/输出模板校验，以及 provision→Stage1→Stage2 的无测量端到端路径。H800 验证只写 Git 忽略目录，先运行预检，再在用户已授权的三张私有 H800 上启动完整闭环。
