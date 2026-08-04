# P3-1：Stage6 纯 Python 合约层

状态：已完成（本地）；未推送、未发布。

## 目的与边界

本批次只补齐当前公开复现包缺失的、确定性的 Stage6 计划与验证逻辑；它不发起训练、模型导出、GPU 测量、AP 评测、TVM/TensorRT 调度或网络访问。它也不将论文数值升级为 `verified`。

范围是三个私有源候选的脱敏迁移：

- `framework/stage6/contracts_v1.py`
- `framework/stage6/formal_plan_v1.py`
- `framework/stage6/protocol_smoke_v1.py`

对应的公开测试位于 `tests/stage6/`，只构造内存中的合成字典；没有 fixture、数据集、模型、checkpoint、engine、ONNX、缓存或 `results/` 文件被迁入。

## 审阅结论

P2 的重新只读盘点将上述三个源文件归为 `migrate_code`。在私有与公开树的 Stage6 相对路径比较中，私有候选为 8 个、公开已有 2 个；本批次选择其中 3 个公开独有文件，未覆盖或替换任何已有公开 Stage6 模块。

三个模块的运行时依赖仅为 Python 标准库、NumPy 和 scikit-learn；后两者已由公开 `pyproject.toml` 的基础或 `repro` 依赖声明。源树和公开树均有 Apache-2.0 许可证文本；这只是本批次的许可证线索，不代替 P7 对第三方来源、署名和数据许可的最终审查。

## 脱敏与设计决定

`contracts_v1.py` 的原始静态合约含有生成结果目录的固定位置。公开版本改为结构化的 `availability: external` 和 `required_artifact` 描述符：它定义后续 P4/P6 必需的证据种类，却不假定维护者本机路径、更不读取任何外部文件。验证器拒绝非结构化描述符。

`formal_plan_v1.py` 只根据传入的候选、图特征与后端无关 AP 代理生成计划；它会拒绝两后端候选集、源证据 SHA-256、AP 代理或图特征发生漂移的输入。`protocol_smoke_v1.py` 只执行内存排序和计数，并以字段白名单输出硬件无关的候选信息。两者均不执行硬件动作。

这些模块暂不接入默认 CPU smoke 的论文证据链；它们的用途是让后续 P3 批次能够以公开、可测试的合约建立 Stage6 输入/输出。缺少外部证据时，后续执行仍应失败关闭。

## 验收与后续

迁移测试先以缺少模块的 `ModuleNotFoundError` 失败，再在实现后通过。覆盖内容包括六臂合约、后端无关候选计划、禁用后端标签、串行锁定、逆向迁移失败记录，以及一个纯内存跨模块集成测试。

本批次最终验证为：`366 passed`、Ruff、`compileall` 与 diff 空白检查通过；匿名 ZIP 独立 build/verify 通过，含 109 个成员，SHA-256 为 `72b8f3c813fd29407044c863c62d66f50ae7c8a4296ae43727bdce1ea75f1b9f`。待提交内容的禁用模式扫描为零命中；`pip-audit -r requirements.txt` 未发现已知漏洞，带 `+cpu` 标记的 PyTorch 仍因不在 PyPI 审计索引中而无法被该工具审计。

以下内容明确留给后续批次，不因 P3-1 自动批准：

- `evidence_bundle_v1.py`：会读取显式外部证据路径，须先完成 P4 输入/哈希契约；
- Stage6 的执行器、硬件配置、真实 AP/延迟/能耗输入和 shell 脚本：须经 P4--P6 与人工许可审阅；
- 其余私有 Stage6、Stage5、Stage7、Stage2 模块：须逐批重新盘点、脱敏、测试和记录决定。
