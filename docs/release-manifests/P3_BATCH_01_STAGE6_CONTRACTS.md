# P3-1：Stage6 纯 Python 合约层

状态：已完成本地加固、独立复审与当前全树验收；未推送、未发布。

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

## 复审加固

后续独立复审发现原始迁入后仍存在调用方内容回显或异常链泄露的入口；本轮没有迁入任何新私有源，而是以新增失败测试加固既有公开实现。

- 硬件盲选、串行锁定和逆向迁移记录均改为严格公开字段白名单。路径型或畸形 candidate ID 被拒绝，逆向记录只使用位置索引、固定状态和原因码，不再展开输入行。
- 双后端 `source_evidence_sha256` 必须均为规范的小写 SHA-256 且一致；绑定记录及其版本哈希成为计划哈希语义的一部分。
- Stage6 manifest、候选 genome、图特征、训练/预测标量及嵌套容器均使用失败关闭的形状和有限数值校验。非法 q-mode、超大数值、NaN/Inf、私有 `__float__` 异常和非对象嵌套输入不会进入返回值、错误文本、`__cause__` 或 `__context__`。

复审最终确认本层没有文件、网络、子进程或硬件 I/O；路径标记探针不从返回、错误或异常链回显。Stage6 定向测试为 `72 passed`，Stage6 覆盖率为 89%，Ruff、`compileall` 与 diff 空白检查通过。当前全树回归为 `405 passed`、总覆盖率 81.95%；匿名 ZIP 已独立 build/verify（113 成员，SHA-256 `96a3bc9b6980e2f25bb9ab102cc950ba2ff57f9fb7c2f35142d2367818b319f6`）。

## 历史验收与后续

迁移测试先以缺少模块的 `ModuleNotFoundError` 失败，再在实现后通过。覆盖内容包括六臂合约、后端无关候选计划、禁用后端标签、串行锁定、逆向迁移失败记录，以及一个纯内存跨模块集成测试。

首次迁入时的历史验证为：`366 passed`、Ruff、`compileall` 与 diff 空白检查通过；匿名 ZIP 独立 build/verify 通过，含 109 个成员，SHA-256 为 `72b8f3c813fd29407044c863c62d66f50ae7c8a4296ae43727bdce1ea75f1b9f`。它不能外推到本轮加固后的 HEAD。待提交内容的禁用模式扫描为零命中；`pip-audit -r requirements.txt` 未发现已知漏洞，带 `+cpu` 标记的 PyTorch 仍因不在 PyPI 审计索引中而无法被该工具审计。

以下内容明确留给后续批次，不因 P3-1 自动批准：

- `evidence_bundle_v1.py`：会读取显式外部证据路径，须先完成 P4 输入/哈希契约；
- Stage6 的执行器、硬件配置、真实 AP/延迟/能耗输入和 shell 脚本：须经 P4--P6 与人工许可审阅；
- 其余私有 Stage6、Stage5、Stage7、Stage2 模块：须逐批重新盘点、脱敏、测试和记录决定。
