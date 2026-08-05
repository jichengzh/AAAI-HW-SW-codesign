# P3：完整源码处置与验证关闭计划

状态：进行中；此前机械分流形成的 P3-13--P3-17 结论与全量 ledger 已撤回，不能作为公开发布或 P4 的依据。真实逐项复审按 [P3_REAUDIT_PROTOCOL.md](P3_REAUDIT_PROTOCOL.md) 执行；本文件定义其关闭条件，不代表已完成公开发布。

P3-18--P3-39 的已接受 records 按批次相加曾为 253 条，但该数字不是去重后的覆盖数：P3-36 与 P3-37 中有 9 条重复 P2 身份，因此截至 P3-39 的真实唯一覆盖为 244 项。P3-25 中 4 项在入队快照后发生内容变化，已按当前内容重新读取、重新绑定受限 SHA-256 证据后才保留；P3-26/P3-27 的冲突本地 records 已由保留旧记录的 supersession manifest 解决，唯一规范 records 已重新验证当前内容并重建账本；P3-28--P3-39 中已接受批次当前内容漂移均为 0，且均已有独立语义复审。P3-35 的前两次审阅有 8 个边界分歧，已由第三次逐项裁决、公开 Stage2 契约回归和重新计算账本解决；P3-36 的两项公开替代分歧也已由第三次职责/API/测试裁决解决；P3-38 的两个 P4/P6 finalizer 分歧已由逐项读取测试与被测入口的裁决解决。所有受限 evidence/decision/HMAC ledger 检查点均不构成关闭账本；P4 尚未开始。

## 当前更正（以本节为准）

P3-40 经第三次逐项裁决、P3-41 经公开职责与定向回归验证、P3-42 经主审与独立复审后，三批均完成当前内容绑定和路径无关 HMAC ledger 重算。P3-41 的 6 项 `duplicate_or_superseded` 已逐项由公开 Stage4 实现与测试/API 职责证据支持，定向回归为 119 passed。P3-18--P3-42 records 合计 289 条，但 P3-40 的 2 条重复 P2 身份与此前 9 条重叠合并后，基线唯一覆盖为 278。P3-43--P3-47 依次新增 12 个身份；P3-48 再以 3 项 P4、8 项 P6、1 项非必要排除新增 12 个身份。P3-49 以 2 项 P4、1 项 P5、8 项 P6、1 项非必要排除新增 12 个身份。P3-50 以 3 项 P4、5 项 P6、2 项有公开 API/测试职责证据的 `duplicate_or_superseded` 和 2 项 `blocked_license_or_permission` 新增 12 个身份。P3-51 以 10 项 P4（闭环结果、图表与 TRT/ONNX 制品）、1 项 P5（分布式运行配置）和 1 项 P6（CARLA/CUDA 全流程运行时）新增 12 个身份；P3-52 以 3 项 P4（训练/评估数据、权重与 checkpoint 配置）和 9 项 P6（插件、训练、评测、损失与模型 head 执行面）新增 12 个身份；P3-53 以 12 项 P6（planning/tracking/segmentation heads、CUDA/custom-op 和 ONNX/TRT attention/export 支撑链）新增 12 个身份。P3-50 的 release-asset verifier、completed-round deployment rehydrate 与 dependency-pin transaction 均经第三次只读裁决，分别归 P4、P6、P6；P3-51 的两个 TRT/ONNX 配置也经第三次裁决归 P4，因为主导的是 checkpoint-to-ONNX/engine 制品边界而非环境探测；P3-52 的三份训练配置和 plugin/API 入口也分别经第三次裁决归 P4/P6。P3-53 双审逐项一致，但其多个 OpenMMLab/外部 TRT 适配来源仍须在 P6 前完成 NOTICE/provenance 审查。两个外部 CUDA patcher 保持 P7 许可/隐私阻塞，不能降为非必要。历史 373 条 accepted records 亦已按其规范 manifest/ledger 组合重建：P2 metadata、当前 SHA-256 和 HMAC 均重新验证，去重为 362；P3-53 后当前真实唯一覆盖为 **410/1,551**。P3 仍在进行中，P4 尚未开始。详见 [覆盖身份对账](P3_COVERAGE_IDENTITY_RECONCILIATION.md)、[P3-52 记录](P3_BATCH_52_TRAINING_AND_PLUGIN_REAUDIT.md) 与 [P3-53 记录](P3_BATCH_53_MODEL_AND_PLUGIN_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-54 完成 3 项 P4（coldstart、engine/probe、graph evidence 外部制品）和 9 项 P6（pruning/quantization、GPU guard、GPU benchmark、ONNX preparation 与 recovery training）处置，并新增 12 个身份。GPU exclusivity gate 经第三次只读裁决归 P6：GPU 准入是守卫，而其主导行为是被保护命令的启动、监控、隔离与终止。当前真实唯一覆盖为 **422/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-54 记录](P3_BATCH_54_QUANTIZATION_AND_EVIDENCE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-55 完成 3 项 P4（baseline admission、capability rebind 与 AP feedback repair 外部证据）和 9 项 P6（checkpoint recovery、replay/watchdog、GPU scheduler/measurement、ONNX quantization 与 Relax worker 执行链）处置，并新增 12 个身份。双审逐项一致：含 ONNX Runtime calibration/inference 的 quant-contract builder 与实际编译模块执行均归 P6；只绑定、重写或修复已有外部 evidence 的项目归 P4。当前真实唯一覆盖为 **434/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-55 记录](P3_BATCH_55_RECOVERY_AND_TVM_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-56 完成 7 项 P4（scheduler/feedback integrity、H800/4090 baseline、Pareto predictor 与 accuracy evidence）和 5 项 P6（TVM worker 与 DP4A compile/tune/benchmark gates）处置，并新增 12 个身份。第 10、11 项经第三次只读裁决归 P4：虽然会运行离线评分/预测，但主导依赖是真实 baseline/predictor evidence，未执行候选、硬件或模型运行闭环。当前真实唯一覆盖为 **446/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-56 记录](P3_BATCH_56_DPA4_AND_BASELINE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-57 完成 9 项 P4（latency mapping、baseline integration 与 predictor/Pareto analysis 外部 evidence）和 3 项 P6（ONNX export、TensorRT engine build 与 CUDA benchmark 执行）处置，并新增 12 个身份。双审逐项一致：只消费既有测量、baseline 或 predictor 的映射/整合/预测分析归 P4；会物化 runtime engine、导出 ONNX 或执行 CUDA benchmark 的候选归 P6。当前真实唯一覆盖为 **458/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-57 记录](P3_BATCH_57_ENGINE_AND_BASELINE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-58 完成 7 项 P4（measurement budget、baseline/predictor search、ratio lookup 与 ablation evidence），确认 2 项已公开迁入、2 项已安全改写并排除 1 项无运行表面的 legacy marker；本批新增 12 个身份。第 1、2、11、12 项经第三次只读裁决：同字节公开 package marker 记为迁入，受测试覆盖的更窄安全导出面记为改写，未被受支持流程调用的纯说明 marker 可排除。当前真实唯一覆盖为 **470/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-58 记录](P3_BATCH_58_PUBLIC_SURFACE_AND_EVIDENCE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-59 完成 5 项 P4（cost bundle、evidence registry、native route、probe profile 与 measurement feedback evidence），确认 6 项已有安全公开改写，并将 1 项会回显私有 probe/feature 标识、尚无公开安全改写和测试链的候选记为 P7 阻塞；本批新增 12 个身份。第 7 项经第三次只读裁决：未来可改写不是当前已改写证据，必须保持 P7。当前真实唯一覆盖为 **482/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-59 记录](P3_BATCH_59_CONTRACT_AND_EVIDENCE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-60 完成 2 项 P4（closure feedback 与 independent measurement request evidence）、3 项 P5（executor admission、recovery root 与 full frozen/sidecar/pre-scan 环境状态）并确认 7 项安全公开改写；本批新增 12 个身份。第 4、10、11、12 项经第三次裁决：不启动测量的 request builder 归 P4，进程/根目录/状态准备而不运行 executor 的候选归 P5，公开 selection-only 子集不替代完整私有 stateful 角色。当前真实唯一覆盖为 **494/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-60 记录](P3_BATCH_60_STAGE567_BOUNDARY_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-61 确认 11 项当前公开流程不调用的 empty/legacy pytest、执行辅助测试和展示图回归为非必要排除，并确认 1 项由公开 Stage1 coupling predictor API/workflow tests 替代；本批新增 12 个身份。第 2--5、8--12 项经第三次裁决：旧测试本身未启动被测脚本所含的模型、GPU、TVM 或子进程，不可仅因其被测目标而升为 P4/P6。当前真实唯一覆盖为 **506/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-61 记录](P3_BATCH_61_LEGACY_TEST_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-62 确认 11 项遗留 replay/watchdog/TVM/TRT/硬件配置测试为非必要排除，并确认 1 项由公开 Stage1 autoscan/census/predictor workflow 测试替代；本批新增 12 个身份。第 1--9、11、12 项经第三次裁决：旧测试不执行被测运行工具，且不在当前公开支持链中，不能仅因被测目标具有执行边界而转 P4--P6。当前真实唯一覆盖为 **518/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-62 记录](P3_BATCH_62_LEGACY_EXECUTION_TEST_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-63 确认 7 项由公开 Stage1 predictor/classifier/trace-plan 或 Stage2 canonical-search/genome API/tests 替代，并排除 5 项不在当前公开支持链中的遗留测试；本批新增 12 个身份。第 3、4、6、10 项经第三次裁决：公开 API/tests 覆盖净化后的 predictor/classifier/architecture 合同，但不替代完整私有 QxS/SMBO entry。当前真实唯一覆盖为 **530/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-63 记录](P3_BATCH_63_STAGE12_LEGACY_TEST_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-64 确认 2 项由公开 bridge/Stage4 selection-completion API/tests 替代、排除 9 项不在当前公开支持链中的遗留测试，并将 1 项未净化 probe/feature 标识回显候选保持为 P7 阻塞；本批新增 12 个身份。第 4--10、12 项经第三次裁决：旧测试不执行被测 AP/TVM/TRT/repair 工具，不能仅凭底层工具边界转 P4--P6。当前真实唯一覆盖为 **542/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-64 记录](P3_BATCH_64_STAGE35_LEGACY_TEST_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-65 确认 2 项由公开 Stage5 genome/Stage6 six-arm contracts 及测试替代，并排除 10 项当前公开链未调用的 AP plan、repair、validation、audit、TVM helper 和 controller 测试；本批新增 12 个身份。双审逐项一致：旧测试不执行被测运行工具，不能仅由其潜在硬件/AP/TVM 边界转 P4--P6。当前真实唯一覆盖为 **554/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-65 记录](P3_BATCH_65_STAGE56_LEGACY_TEST_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-66 确认 2 项由公开 Stage6 INT8 dispatch/protocol-smoke contracts 及测试替代，并排除 10 项当前公开链未调用的 Stage6/Stage7 遗留测试；本批新增 12 个身份。第 3 项经第三次裁决：公开 manifest validator 已可校验且失败关闭地承担 INT8 route/legacy 禁止职责。当前真实唯一覆盖为 **566/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-66 记录](P3_BATCH_66_STAGE67_LEGACY_TEST_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-67 完成 2 项 P4（未公开论文 latency/curve 证据图输入）、1 项 P5（部署 diagnostic bytecode archive）、8 项 P6（Stage7 transaction deployment/runtime repair）和 1 项 P7（外部 process patcher 的路径与再分发权限阻塞）处置，并新增 12 个身份。第 2--12 项经第三次只读裁决：未公开证据制品不可仅因不在公开调用链而排除；部署目录归档是环境状态；会修改 deployed/frozen runtime、scheduler 或 live wrapper 的候选属于执行链。当前真实唯一覆盖为 **578/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-67 记录](P3_BATCH_67_DEPLOYMENT_AND_EVIDENCE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-68 完成 2 项 P4（未公开 measured CSV 的图/插值证据）、5 项 P6（Stage7 transactional deployment 与 live orchestrator 维护）、4 项 P7（外部 patch/config 与第三方派生 metrics 的许可/路径/输出阻塞）和 1 项非必要 legacy synthetic test 排除，并新增 12 个身份。第 1--5、11、12 项经第三次只读裁决：会替换 deployed/frozen runtime、更新 deployment state 或控制 live 进程的包装器属于执行链；未公开 measured CSV→论文图的职责属于外部制品契约。当前真实唯一覆盖为 **590/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-68 记录](P3_BATCH_68_DEPLOYMENT_METRICS_AND_EVIDENCE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-69 完成 1 项 P4（TRT Phase-2 config 的 checkpoint/ONNX/engine/dataset/anchor 制品来源）、9 项 P6（UniV2X/MMDet registry、assigner、coder、match-cost 和 tensor 执行支持）和 2 项非必要排除，并新增 12 个身份。第 2、3 项经第三次只读裁决：零字节 private package marker 不承担可发布职责；模型 config 的主导边界是外部制品身份而非环境或执行器。当前真实唯一覆盖为 **602/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-69 记录](P3_BATCH_69_UNIV2X_MODEL_SUPPORT_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-70 完成 12 项 P6（UniV2X/MMDet distributed evaluation、training loss 与 model plugin import 支持）处置，并新增 12 个身份。主审初稿的队列身份错误已作废；重新绑定实际队列后双审逐项一致。当前真实唯一覆盖为 **614/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-70 记录](P3_BATCH_70_UNIV2X_EXECUTION_SUPPORT_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-71 完成 12 项 P6（UniV2X 分割、tracking、detector、ONNX/TRT plugin、fusion、hook 与 attention 执行支持）处置，并新增 12 个身份。双审逐项一致；公开 CPU-safe 流程没有同职责 API/test 支持链。当前真实唯一覆盖为 **626/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-71 记录](P3_BATCH_71_UNIV2X_TRACKING_EXECUTION_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-72 完成 6 项 P6（transformer、pruning 与 temporal cache 执行支持）和 6 项 P7（QuantV2X/OpenCOOD 量化栈 copy/port 的再分发许可阻塞）处置，并新增 12 个身份。后六项经第三次裁决：项目 Apache-2.0 信号不覆盖明确的上游 copy/port，且未见可再分发豁免或干净公开替代。当前真实唯一覆盖为 **638/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-72 记录](P3_BATCH_72_PRUNING_AND_QUANTIZATION_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-73 完成 6 项 P4（真实测量、formal feedback、hardware baseline 与结果制品）、2 项 P6（formal feedback 原子发布与 TVM remeasurement 控制）、3 项 P7（QuantV2X/OpenCOOD copy/port 许可阻塞）和 1 项非必要排除，并新增 12 个身份。q6、q8、q11 经第三次裁决：atomic finalizer/remeasurement request 归 P6，私有早期 sanity 不在公开支持链可排除。当前真实唯一覆盖为 **650/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-73 记录](P3_BATCH_73_EVIDENCE_AND_QUANTIZATION_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-74 完成 5 项 P4（AP/latency/H800/SMBO 外部结果制品）、3 项 P6（模型评估与 TVM/CUDA 执行）、2 项 P7（私有根目录与破坏性工作目录语义）和 3 项已安全公开改写，并新增 12 个身份。六项经第三次裁决：执行 runner 的实际加载/编译/计时职责归 P6；嵌入路径仅为结果来源示例的 builder 归 P4；当前回显私有根目录或破坏性工作目录的 runner 保持 P7。当前真实唯一覆盖为 **662/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-74 记录](P3_BATCH_74_ATTENTION_EXECUTION_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-75 完成 4 项 P4（dataset、coverage、active-sample 与网格外部制品）、6 项 P6（ONNX/TRT/TVM/CUDA/NVML 执行与能耗测量）和 2 项 P7（私有根目录或外部 OpenCOOD 来源绑定），并新增 12 个身份。七项经第三次裁决：真实 load/compile/measure/NVML runner 均归 P6，纯配置计划归 P4。当前真实唯一覆盖为 **674/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-75 记录](P3_BATCH_75_DATASET_AND_ENERGY_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-76 完成 3 项 P6（TensorRT engine repair/build/inspect/CUDA re-benchmark 与 TVM recovery measurement）和 9 项 P7（私有模型、workdir、结果、CUDA/PTX dump 或环境文件根目录绑定），并新增 12 个身份。q4、q5、q12 经第三次裁决：消费 engine/cache 或 workdir 不改变其实际 build/inspect/compile/execute/measure 的主导执行职责。当前真实唯一覆盖为 **686/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-76 记录](P3_BATCH_76_TRT_TVM_RECOVERY_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-77 完成 9 项 P4（latency/AP/预测器、ONNX 或校准制品及其派生产物）、1 项 P6（TensorRT/CUDA engine build/inspect）、1 项 P7（私有运行工作目录绑定）和 1 项公开实现已替代处置，并新增 12 个身份。q3、q12 经第三次只读裁决：公开 latency-LUT API、调用链及 fixture 测试已完整承担查询/插值职责；构建和检查 TensorRT engine 的主导边界仍为执行。当前真实唯一覆盖为 **698/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-77 记录](P3_BATCH_77_PARETO_ARTIFACT_AND_TRT_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-78 完成 2 项 P4（外部评测日志与多模型 Pareto 结果制品）、9 项 P6（TensorRT engine build/load/execute/inspect 及 CUDA、NVML、CUDA Graph、设备采样或能耗/延迟测量）和 1 项 P7（第三方 NMS monkey-patch 来源、许可证和 NOTICE 链路），并新增 12 个身份。q2--q10 经第三次只读裁决：即使消费外部 ONNX、engine 或校准制品，实际 build/execute/measure/inspect 脚本的主导边界仍是 P6。当前真实唯一覆盖为 **710/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-78 记录](P3_BATCH_78_TRT_CUDA_MEASUREMENT_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-79 完成 1 项 P4（外部 screening 制品解析）、6 项 P6（TensorRT/TVM build/load/execute/inspect、CUDA、MetaSchedule 或测量）和 5 项 P7（私有模型、数据、工作目录或第三方来源/许可边界），并新增 12 个身份。q1、q2、q3、q7、q8、q10、q11 经第三次只读裁决：私有输入/许可边界优先 P7；合成算子只要实际执行 CUDA/TVM 调优、编译或运行，主导边界仍为 P6。当前真实唯一覆盖为 **722/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-79 记录](P3_BATCH_79_COUPLING_AND_TVM_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-80 完成 10 项 P6（TVM/CUDA compile、MetaSchedule tune、VM run、计时、TIR 导出或结果记录）和 2 项 P7（私有调优工作目录或外部 ONNX 输入绑定），并新增 12 个身份。q1--q6、q8、q9、q11、q12 经第三次只读裁决：环境依赖本身不足以归 P5，脚本若自身承担 compile/tune/run/measure 即归 P6。当前真实唯一覆盖为 **734/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-80 记录](P3_BATCH_80_TVM_KERNEL_EXECUTION_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-81 完成 2 项公开迁入、1 项安全公开改写、5 项 P6（ONNX/TVM build/run 或 LightGBM 训练、评估、模型/指标导出）和 4 项 P7（私有工作目录、第三方训练/算子来源或错误输出回显风险），并新增 12 个身份。q2、q9--q12 经第三次只读裁决：读取数据后自行训练、评估和导出模型的脚本属于 P6，而非仅外部制品整合。当前真实唯一覆盖为 **746/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-81 记录](P3_BATCH_81_PREDICTOR_AND_STAGE1_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-82 完成 7 项 P4（外部数据、日志、硬件结果、配置和指标制品的校验、解析或聚合）、4 项 P6（LightGBM 训练、交叉验证/评估与模型/指标导出）和 1 项 P7（第三方 VFE 微基准来源与改写许可风险），并新增 12 个身份。q1--q4 经第三次只读裁决：外部训练数据不改变脚本自身训练、评估和模型导出的 P6 主导职责。当前真实唯一覆盖为 **758/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-82 记录](P3_BATCH_82_PHASE4_AND_TRAINING_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-83 完成 3 项安全公开改写、4 项 P4（历史测量、AP 来源或制品 registry 的脱敏公开契约）和 5 项 P6（LightGBM 训练、交叉验证/评估与模型/指标导出），并新增 12 个身份。q2--q6 经第三次只读裁决：外部 CSV 只是输入依赖，不改变训练/验证/模型导出脚本的 P6 主导职责。当前真实唯一覆盖为 **770/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-83 记录](P3_BATCH_83_PHASE4_STAGE1_STAGE2_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-84 完成 3 项 P4（历史架构审计、诊断报告或 INT8 来源证明等外部证据制品）和 9 项 P6（ONNXRuntime/TVM import、build、compile、tune、VM run、数值对齐、计时或能耗测量），并新增 12 个身份。q3、q4、q5、q7--q12 经第三次只读裁决：外部 ONNX、校准或参考制品属于输入约束，不改变脚本自身执行职责；OpenCOOD 标记仅指形状/配置来源，静态未见明确代码复制或移植证据。当前真实唯一覆盖为 **782/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-84 记录](P3_BATCH_84_CODRIVING_TVM_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-85 完成 1 项非必要私有 CLI 排除、5 项 P4（外部测量、制品 registry、探测运行或候选计划）和 6 项 P6（TVM/CUDA worker、外部 telemetry/latency 命令或硬件测量执行调度队列），并新增 12 个身份。q1、q4--q7、q10、q11 经第三次只读裁决：公开 API/测试与既有范围策略完整覆盖私有 outlier 文件 I/O CLI；只生成候选计划而没有命令链的脚本归 P4，具有硬件测量调度语义的队列归 P6。当前真实唯一覆盖为 **794/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-85 记录](P3_BATCH_85_STAGE2_COVERAGE_AND_LUT_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-86 完成 1 项安全公开改写、4 项 P4（外部 profile、测量回放或 LUT 导入契约）、3 项 P6（可执行 H800 测量调度或 TVM/CUDA 编译/worker 执行）和 4 项 P7（私有运行环境、个人路径、任意命令入口或未脱敏命令/主机/traceback 日志），并新增 12 个身份。q1、q8、q9、q11 经第三次只读裁决：可执行硬件命令链属 P6；带私有 runtime 与任意命令/日志风险的 worker 优先 P7。当前真实唯一覆盖为 **806/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-86 记录](P3_BATCH_86_H800_WORKERS_AND_LUT_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-87 完成 6 项 P4（外部制品、证据、probe、readiness 与历史回放报告）、3 项 P6（实时 GPU launch gate、可执行 LUT 命令链或 TVM/TensorRT build）和 3 项 P7（私有制品或 TVM/CUDA 环境根及未脱敏主机/日志输出），并新增 12 个身份。q2、q6、q12 经第三次只读裁决：实时 GPU 调度、可执行命令链和实际编译均属 P6，而非纯环境描述。当前真实唯一覆盖为 **818/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-87 记录](P3_BATCH_87_ARTIFACT_AND_PROBE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-88 完成 1 项安全公开改写、4 项 P4、4 项 P6 和 3 项 P7，并新增 12 个身份。当前真实唯一覆盖为 **830/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-88 记录](P3_BATCH_88_SMBO_AND_CODRIVING_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-89 完成 10 项 P4 外部制品契约和 2 项 P7 许可/隐私阻塞，并新增 12 个身份。12 项经双独立只读审查逐项一致，当前真实唯一覆盖为 **842/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-89 记录](P3_BATCH_89_ARTIFACT_AND_PERMISSION_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-90 完成 4 项 P4 外部制品契约和 8 项 P6 执行契约，并新增 12 个身份。6 项经第三次只读裁决：可执行性能/AP 计划及实际离线充分性模型训练均属 P6，而非纯环境或制品汇总。当前真实唯一覆盖为 **854/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-90 记录](P3_BATCH_90_GOLD_PLAN_AND_SUFFICIENCY_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-91 完成 1 项安全公开改写、6 项 P4 外部制品契约、4 项 P6 执行契约和 1 项 P7 许可/隐私阻塞，并新增 12 个身份。6 项经第三次只读裁决：执行计划、模型训练/推理、staging contract、私有运行环境与公开 closure rewrite 分别按其主导边界处置。当前真实唯一覆盖为 **866/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-91 记录](P3_BATCH_91_GOLD_AND_CLOSURE_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-92 完成 1 项 P4 外部制品契约和 11 项 P6 执行契约，并新增 12 个身份。6 项经第三次只读裁决：公开 API 的存在不足以将实际驱动训练/评估或作业调度的候选认定为公开改写，均按执行边界归 P6。当前真实唯一覆盖为 **878/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-92 记录](P3_BATCH_92_STAGE4_STAGE5_EXECUTION_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-93 完成 7 项 P4 外部制品契约和 5 项 P6 执行契约，并新增 12 个身份。4 项经第三次只读裁决：执行状态修复、性能计划和 teacher-forced replay 含可执行作业或实际拟合/预测/选择，均归 P6。当前真实唯一覆盖为 **890/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-93 记录](P3_BATCH_93_STAGE5_EVIDENCE_AND_REPLAY_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-94 完成 6 项 P4 外部制品契约和 6 项 P6 执行契约，并新增 12 个身份。2 项经第三次只读裁决：独立验证性能计划与 hidden-label replay 分别生成可执行 jobs 或实际拟合/预测/选择，均归 P6。当前真实唯一覆盖为 **902/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-94 记录](P3_BATCH_94_STAGE5_STAGE6_REAUDIT.md)。

**最新更正（覆盖上段的当前计数）**：P3-95 完成 10 项 P4 外部制品契约和 2 项 P6 执行契约，并新增 12 个身份。独立复审的首次候选身份集合与固定队列不一致，已整体作废并重新逐项审阅；q1、q4、q8、q11 经第三次只读裁决：私有证据 bundle、既有 smoke evidence 与 formal 状态均归 P4，TVM/CUDA 编译导出归 P6。当前真实唯一覆盖为 **914/1,551**；P3 仍在进行中，P4 尚未开始。详见 [P3-95 记录](P3_BATCH_95_STAGE6_EVIDENCE_REAUDIT.md)。

## 冻结基线

P3 以 P2b 的私有源只读盘点为唯一候选基线：原始盘点 SHA-256 为 `6fafdace7dc0eae3e9982763cb1c5c098c0ae83d1715af67ff1e08d7a1d12a79`，共 1,551 个候选：1,008 个代码、96 个配置、439 个 Markdown 和 8 个小 fixture。另有 3,664 个生成物、外部输入或敏感/特殊项，它们不直接进入 P3。两个嵌套仓库保持独立边界，不能默认合并。

每一次 P3 批次开始前必须用 v1.1.0 inventory 工具在受控临时目录重跑盘点；若基线哈希或计数改变，先记录增量并处置新增/删除项。原始逐路径报告、私有路径和其映射绝不提交。静态扫描、AST、目录和扩展名仅可用于生成复审线索，不能代替逐项职责、依赖、来源和许可证证据。

## 处置账本

关闭前必须由 `build_p3_disposition_ledger.py` 校验一个本地逐路径 decisions 文件，并输出可提交的脱敏账本。维护者提供私有 HMAC key；公开账本只保存 HMAC-SHA256 派生的不可逆 `candidate_id`，不得保存路径、内容、地址、凭据、人员标识或该 key。

每个候选恰好一条、且仅可使用以下决定之一：

- `migrated_public`：可公开迁入并有测试；
- `rewritten_public`：语义保留但已脱敏改写；
- `external_contract_p4`：外部数据、模型或大文件已有获取/校验契约；
- `environment_contract_p5`：环境、驱动或硬件约束已有公开配置；
- `execution_contract_p6`：训练、测量或汇总执行器已有公开执行契约；
- `duplicate_or_superseded`：已有公开等价物；
- `excluded_nonessential`：与支持的复现链无关，且已记录依赖反证；
- `blocked_license_or_permission`：不可公开；若其为复现关键依赖，P3 不得关闭。

每条公开记录还须绑定 P2 分类、批次、固定 reason code、许可结论、公开替代物/契约、测试或审阅证据及后续阶段。每个本机 decision 还必须遵循 P3_REAUDIT_PROTOCOL 的内容 hash、职责、调用关系、来源/许可证和安全输出证据要求。账本不得含 `pending` 或 `reviewing`；计数必须精确回填至 1,551。

## 执行链关闭

每一项支持的公开流程必须建立“设计说明 → 模块 → 命令/API → 合成测试 → 输入契约”的映射。迁入或改写的模块只能使用脱敏小输入；默认 CPU smoke 不读取维护者目录、`results/`、缓存、GPU 或外部硬件。无法提供的外部输入必须以明确的 `unavailable` 失败关闭，而不是以 demo 替代论文结果。

当前 CPU smoke 已满足上述离线边界，但只复现 scoped smoke。正式 Stage6 终端证据、Stage7 轨迹/聚合、训练物化、完整数据和硬件执行仍为 external/unavailable；这些事项不因 P3 完成而被视为已复现，分别由 P4--P6 处理。

公开执行面仍须逐一处置非 smoke 历史入口：每个入口必须被标为有完整公开契约，或被明确外置/延后。P3-7 已修复已发现的个人绝对路径、生成物忽略缺口和过时输出说明；其余入口的全量处置仍由本账本约束。

## P3 关闭验证

只有同时满足以下条件才可将 P3 标记为“已完成（本地）”：

1. 脱敏账本由当前 P2b 基线生成，1,551 条全部处置且分类/总数对账通过；
2. 所有 `migrated_public`、`rewritten_public`、`duplicate_or_superseded` 和 `excluded_nonessential` 都有公开证据；转交 P4--P6 的关键项已有可验证契约；
3. 账本机械测试、全量 pytest（覆盖率不少于 80%）、Ruff、`compileall`、身份/凭据扫描、匿名 ZIP build/verify 均针对同一 HEAD 通过；
4. 独立复审确认不存在路径/身份/凭据回显、隐式本机依赖或未处置候选。

完成后在总台账记录候选基线、提交、命令、测试数、覆盖率和 ZIP 哈希，并保留 P4--P8 的未完成边界。

## 远端更新判定

当前不允许将远端称为完整公开发布版本：P3 的发布级逐项处置账本尚未重新生成，P4--P8 也未完成。只有完成逐项复审及 P4--P7 实际交付后，才可以评估将通过同一 HEAD 验证的分支同步到已确认私有的协作远端；这只是阶段性备份，不是开源发布。任何 `git push`、可见性变更、tag 或 Release 都需要维护者对此次外部动作的明确授权。
