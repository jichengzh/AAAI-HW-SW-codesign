# P3-7：P3 关闭验证与远端更新判定

状态：已完成本轮关闭验证；P3 未关闭，未推送、未发布。

## 验证目标

本批次不迁移新的私有源码，也不复制私有路径清单。目标是验证 P3 关闭所需的公开机制是否具备，并判断当前工作树是否已经可以作为远端更新候选。

P3 的完整基线、允许的处置决定、公开执行链要求和最终验证门见 [P3 关闭计划](P3_CLOSURE_PLAN.md)。

## 已完成项

- 新增 `tools/release/build_p3_disposition_ledger.py`，用于从本地私有 inventory 与 decisions 文件生成公开、路径无关的 P3 disposition ledger。
- 新增 `tests/release/test_p3_disposition_ledger.py`，覆盖 HMAC-SHA256 不可逆候选 ID、逐候选完整对账、固定 disposition/reason/evidence token、符号链接和硬链接拒绝、HMAC key 权限检查、输出路径不覆盖以及错误不回显私有标识。
- 新增 `tests/release/test_public_execution_surface.py`，固定公开执行入口的失败关闭边界：Stage1 S2 anchor runner 不再包含个人绝对路径示例，缺少 `CUDA_VISIBLE_DEVICES` 或输出参数时在导入 TVM 前失败且不回显调用方路径；Stage2 demo 说明与实际显式 `--output-root` 行为一致；生成物忽略规则不影响公开 fixture、verified artifacts 或源码。
- 更新 `.gitignore`，忽略本地实验结果、模型、checkpoint、engine、ONNX、TensorRT 产物和分区生成目录。

公开 CPU smoke 仍只覆盖 scoped demo；Stage6 正式终端证据、Stage7 正式聚合、训练物化、完整数据和硬件执行均保持 `external` 或 `unavailable`，由 P4--P6 处理，不能由 P3 或 demo 伪装为论文复现。

## 关闭结果

P3 当前不能标记为完成。原因是尚未提供本地逐路径 decisions 文件，也未生成可提交的公开 disposition ledger。按 P3 关闭计划，ledger 必须覆盖 P2b 基线的 1,551 个候选，并把每个候选恰好归入 `migrated_public`、`rewritten_public`、`external_contract_p4`、`environment_contract_p5`、`execution_contract_p6`、`duplicate_or_superseded`、`excluded_nonessential` 或 `blocked_license_or_permission`。没有该账本时，无法证明“其余模块”已经全部处置。

因此，本批次的结论是：

- P3 关闭基础设施已具备；
- P3 全部候选处置尚未完成；
- 当前 HEAD 不应作为“完整可复现公开版本”推送；
- 即使后续要把阶段性成果同步到远端，也只能作为私有协作分支备份，且需要维护者对 `git push` 的明确授权。

## 验证记录

定向验证：

- `python -m pytest tests/release/test_p3_disposition_ledger.py tests/release/test_public_execution_surface.py -q`：18 passed；
- `python -m ruff check tools/release/build_p3_disposition_ledger.py tests/release/test_p3_disposition_ledger.py tests/release/test_public_execution_surface.py scripts/phase2/stage1_s2_anchor_runner.py scripts/prepare_stage2_demo_data.py`：通过；
- `python -m compileall -q tools/release/build_p3_disposition_ledger.py scripts/phase2/stage1_s2_anchor_runner.py scripts/prepare_stage2_demo_data.py`：通过；
- `git diff --check`：通过。

针对同一候选工作树的全量验证为 `423 passed`，总覆盖率为 81.95%；交接、身份和匿名 archive 回归为 `52 passed`。匿名 ZIP 已独立 build/verify，含 116 个成员，SHA-256 为 `707dbe2ac21dd276d8eb25192527c0486a8ccf86f73afc50174e2a849e8fb06a`。依赖审计未发现已知漏洞；带 `+cpu` 本地版本标记的 PyTorch 仍不在 PyPI 审计索引，保留为 P7 供应链边界。

后续若要关闭 P3，下一步必须在公开仓库外生成并维护本地 decisions 文件，再用当前 ledger 工具生成公开脱敏账本；随后对同一 HEAD 运行全量 pytest、Ruff、`compileall`、身份/凭据扫描和匿名 ZIP build/verify。
