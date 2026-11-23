# CoptV2X P4 五臂在线 HIL 冒烟恢复实验

## 范围与限制

固定 `Town05:_1:r2`，五个冻结配置：original_default、compression_only、schedule_only、compress_then_tune、joint_shcosearch。seed=traffic seed=2000、repeat=0、cadence=2、profile=5_10、固定仿真步长50ms。推理通过既有Orin本地缓存、ROS trigger/result和异步融合链路执行。

本包仅用于接口及生命周期集成冒烟：`integration_smoke_only=true`、`performance_conclusion_allowed=false`。预录输入不是当前车辆重新采集的输入；压缩通道通过分段前缀裁取现有训练投影接入规划器，未经重新联合校准。不得据此比较五臂闭环驾驶性能或形成论文结论。425项全量实验属于P5，未纳入本包。

部署边界明确为：Orin读取预录`spatial_features [2,64,256,512]`，执行冻结`backbone.resnet`/对应TensorRT engine，回传`res0/res1/res2`；服务器保留当前ego分支、协同融合、检测、规划和控制。因此本轮Orin计算计时不是“从原始RSU点云到控制指令”的完整推理时间，也未证明全部车端/路侧计算已经部署在Orin。原始点云前端与后续模块的正式计时边界必须另行覆盖；这一限制只记录，不扩张或冻结P4。

## 文件导航

- `p4-five-arms-20261003T054210Z/manifest.json`：冻结任务与协议。
- 同目录`state.json`：所有attempt、基础设施重试及最终结果。
- 同目录`runs/`：每次尝试的配置、CARLA checkpoint、fusion审计和Orin服务日志；失败尝试不删除、不补造驾驶结果。
- `evidence/INPUT_VERIFICATION.md`、host/Orin input manifest及逐字节验证日志：2225帧缓存完整性证据。
- `evidence/worktree-head.txt`、`worktree.patch`、`code-snapshot.tgz`：执行代码基线、已有及本轮未提交差异、新文件快照。
- `evidence/p4-bundle-audit.json`：仅在真实五臂结束后生成正式验收报告；不存在或状态不是P4_PASS时，本包未完成。

## 已有输入和产物位置

Host历史缓存根：`/data/jichengzhi_v2x/ditl/coptv2x_hil_p4_smoke_20261001T074559Z`。

Orin完整2225帧缓存：`/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z/input`。本包复用该既有约149GB缓存，不重复复制；各NPY逐字节校验全部OK。

Orin冻结模型/engine根：`/home/jichengzhi/lane_c_orin_stage6_five_config_20260724`。Host相应产物根：`/home/jichengzhi/V2X/results/lane_c_orin_stage6_five_config_20260724`。没有重新搜索、训练、构建engine或修改冻结权重。

Orin新归档根：`/home/jichengzhi/coptv2x_hil_p4_recovery_20261003T054210Z`。逐attempt远端服务配置位于`episodes/`；既有runtime/ROS容器路径继续复用历史部署。

## 执行环境说明

工作树：`/home/jichengzhi/V2X/.worktrees/V2Xverse-ditl`，branch `feature/coptv2x-live-hil`，base HEAD `af1565f9cd4b79c9aa47200261e74588c3c95fcf`。

实际沿用CARLA Python API egg 0.9.10，服务返回构建标识`784d9b9f`；不是论文预期0.9.15环境已经验证的声明。Orin ROS Foxy容器与原部署沿用。主机共享GPU4且不测时延/能耗比较，五臂串行复用同一Orin和ROS域71。

本轮墙钟episode上限7200秒、最多6次基础设施尝试，CPU线程预算OMP/MKL/OpenBLAS均4；这些是执行资源设置，不改变路线、模型、仿真步长或统计口径。中途预算调整的attempt-03作为明确operator基础设施重试保留，不计入驾驶性能。

P4完成条件：五臂合法驾驶终态和HIL审计、精确输入映射、真实resume不重复执行、最终数据包在Orin逐文件校验通过。合法路线超时可为驾驶终态，sensor/agent/transport崩溃不可冒充驾驶终态。
