# P3-5：Stage2 脱敏延迟异常策略

状态：已完成本地迁移、验证与独立复审；未推送、未发布。

## 目的与范围

本批次从私有 Stage2 延迟异常策略中迁入重复稳定性、多次运行离散度和历史锚点漂移的纯计算逻辑。公开实现为 `framework/stage2/latency_outlier_policy_v1.py`；相应测试为 `tests/stage2/test_latency_outlier_policy_v1.py`。两者只使用内存中的脱敏合成字典和 Python 标准库，不读取 JSONL、CSV、结果目录、模型、checkpoint、engine、ONNX 或硬件状态，也不迁入私有的文件读写 CLI。

## 脱敏与契约改造

私有原实现会把 `row_id`、`run_id`、`config_id`、模型、候选和调度标识原样写入报告，也会把未受约束的测量状态插入原因文本。公开版本不保留这些字段：调用方标识仅在内存中形成重复组键，输出使用稳定的 `row_index`，且所有原因均为固定代码。报告、异常和统计键不回显调用方标识、路径或 URI。

公开版本还将数值边界收紧为失败关闭：测量值和锚点必须是有限正数，最大延迟不得小于最小延迟，grade 只能为 `calibration` 或 `paper`，分组值、行位置、anchor 映射和行容器都经过形状验证。非测量行、缺失/畸形数值和畸形行会得到固定的 `no_claim` 结论；空输入或不支持的 grade 以不含调用方内容的 `ValueError` 拒绝。

## 测试优先与当前结果

测试先以缺少模块的 `ModuleNotFoundError` 失败。定向结果为 `8 passed`，覆盖稳定/重复异常、paper 级多次运行离散与锚点漂移、畸形行和数值、公开边界、输入不变性及调用方标识不回显。复审新增的回归进一步固定：缺失、空白或非字符串的 `model`、`config_id`、`schedule_policy` 只能得到 `invalid_group`/`no_claim`，超大数值不会抛出异常而是分别失败关闭为 `invalid_latency` 或 `invalid_anchor`。

最终全量验证为 `386 passed`，总覆盖率为 81.72%，新模块覆盖率为 93%；Ruff、`compileall`、diff 空白检查和匿名 ZIP build/verify 均通过。独立复审确认该模块不进行文件、网络、GPU 或子进程操作，畸形 status、直接传入的 group 值、私有路径和 URI 标记均不从报告或异常回显。最终匿名 ZIP 已独立 build/verify：113 个成员，SHA-256 为 `6d821d88ab4915a586192b0e38592dfea22c51a8541fef194487576f0a3196b4`。

## 非范围与后续边界

本策略只产生协议质量状态，不读取实际测量，也不把任何输出升级为论文证据。真实测量、原始重复数据、历史锚点来源、外部制品许可、数据/模型获取、硬件执行和结果落盘仍属于 P4--P6。P3 的其他候选继续逐批审阅，不因 P3-5 自动获准。
