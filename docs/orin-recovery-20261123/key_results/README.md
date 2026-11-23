# Orin 最小原始结果备份

此目录保留从 `jichengzhi@172.16.62.222` 只读复制的原始结果。除本导航外，文件内容未改写；路径相对远端 `/home/jichengzhi/` 保留。源绝对路径、字节数及 SHA256 记录在恢复包 `manifests/key_results_manifest.json`。已复制 111 个原始文件，共 4,852,498 字节；逐文件本地校验与源 SHA256 一致。

## 阅读入口

| 实验 | 入口 | 口径与边界 |
| --- | --- | --- |
| FCooper 正式五臂 | `V2X/results/lane_c_orin_fcooper_stage6_five_config_20260724/06_final/table2_values.json`、同目录 `evidence_manifest.json` | 表格原文 status=complete、row_count=5。五臂 latency report、AP metrics/run_manifest 已保留；AP 为 OPV2V test 2170 样本、post-scatter backbone/shrinker 范围。能耗值与口径保存在正式表格和 latency report 内，不是完整车路系统能耗。Joint 一臂明确为 FP16 control。 |
| Pyramid / CoDriving | `lane_c_orin_stage6_five_config_20260724/{pyramid,codriving}/` | 各配置的 `primary_latency_power*.json` 与存在的 `ap_full/report.json` 原样保存。AP full 报告为 DAIR-V2X 1789 样本；时延仅相应 backbone/多尺度范围。无 sudo 的 power 报告可明确为 unavailable，优先结合 `_sudo` 原文判断，不能把目录存在当作通过。Pyramid `joint_shcosearch` 没有发现 full AP，另保留 `joint_shcosearch_fp16` 的 full AP，不将其替代为 INT8 的成功证据。 |
| P4 recovery | `coptv2x_hil_p4_recovery_20261003T054210Z/final-bundle/README.md`、`evidence/p4-bundle-audit.json`、`p4-five-arms-20261003T054210Z/paired_results.csv` | 仅接口/生命周期 HIL 联调冒烟；原文 integration_smoke_only=true、performance_conclusion_allowed=false。五臂原始最终 CARLA `results.json` 已按 audit 中的路径及 SHA256 保留。paired 表五臂均为 FailedDriving，audit 中驾驶终态为 route_timeout；不能据此声称五臂闭环性能成功或发表性能比较。state 保留 attempt 元信息，未复制失败 attempt 文件树。 |
| Pipeline 去重重放 / ROS | `coptv2x_hil_pipeline_20261010/evidence/paired_{source,saturated}_dedup_5arms_20.jsonl`、`pipeline_ros_smoke.json` | 两份 JSONL 末条为去重摘要：20 个不同输入、100 对比较、200 completed。原文 is_hil=false、is_carla=false、communication_measured=false，通信延迟为 modeled；这属于静态重放。ROS smoke 的 passed=true 仅说明单独 ROS 接口冒烟，is_hil_driving=false。 |
| Quant actual | `resnet_compare_s1_20261013/quant_measurement_actual_r01/summary.json`、`manifest.json`、`target_evidence/ap_evidence.json`、`full1789_completion_acceptance.json` | ONNX Runtime CPU / CPUExecutionProvider，非 TensorRT/GPU。时延/能耗 independent_samples=2；AP qualification count=1789、host_equivalence=false。指定 VIN_SYS_5V0 电源轨积分含 reader 起停开销，不是整板或 GPU 能耗。保留审计、目标身份、评估代码身份及 qualification 证据。 |
| Target fullval | `resnet_compare_s1_20261013/target_fullval_r02/completed.json`、`manifest.json`、`post_fault_cpu_resume_r02.json` | 仅按原文记录完成与恢复；与 quant_actual 的 qualification、CPU 边界结合读取。 |
| CPU 50000 | `resnet_compare_s1_20261013/cpu_ort_full50000_{dense_r02,geta_r01,copt_r01}/run/target/completion.json` | 三份原文 status=STREAM_COMPLETE、count=50000、provider=CPUExecutionProvider；保留 job/identity/receipt/times。未找到对应 top-1/top-5 或其他准确率汇总，因此这里只能证明流处理完成，不能给出准确率。 |
| 旧 m4_8 | `m4_8_orin/p03_build/orin_p03_measure.csv`、`m4_8_orin/tools/orin_deploy/m1_standalone/*.csv` | 仅保留六份小 CSV；按原始字段及旧实验范围解读。 |

## 有意未复制

- 数据集、输入缓存、NPY/NPZ、模型/权重、ONNX、TensorRT engine、环境、凭据、大日志和历史 attempt 文件树。
- `classification_target_accuracy_r01/final.json` 与 `s7_trt_target_accuracy_r01/final.json` 均为约 14.6 MB 的 50000 样本清单，只有 count/samples/sha256/split，不是准确率结果；其路径、文件 SHA256、字节数和计数已登记在 manifest，未复制。
- Quant 的逐帧大数组、输出 NPZ、profile 大数组及请求流水未纳入此最小包。需要重新分析逐帧结果或复现实验时，须另行保留对应原始证据和模型，不能从本包反推缺失数据。
