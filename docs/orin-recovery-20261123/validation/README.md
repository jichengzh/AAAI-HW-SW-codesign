# 上传后的真实恢复副本验收

本目录保留2026-11-23上传四份冻结engine后的最小关键结果。没有恢复历史大缓存。

- [公开下载校验](public_engine_downloads_20261123.json)：commit固定URL、每文件字节数与SHA256；实际HTTP下载后重新计算。
- [五配置实际推理回执](engine_revalidation_20261123.json)：逐臂退出码、样本/完成事件、模型身份、输入/输出合同、feature hash和wire大小/摘要。
- [软件与隔离快照](isolation_software.json)：既有SDK/conda、791份传输文件、临时目录及20份原生模块来源。快照时CUDA尚未初始化，后续probe实际运行模型。
- 原始JSONL：[original_default](original_default.jsonl)、[compression_only](compression_only.jsonl)、[schedule_only](schedule_only.jsonl)、[compress_then_tune](compress_then_tune.jsonl)、[joint_shcosearch](joint_shcosearch.jsonl)。这些文件从Orin直接复制，字节未改写，SHA256写入推理回执。

每个配置实际执行一次冷启动首帧，并检查TensorRT/原生模型输出、协议编解码、finite/shape/feature hash。探针沿用 `static_replay_same_host_process_cpu_diagnostic` 的scope标签；backend字段记录实际native_pytorch或tensorrt，时间字段仅保留诊断语境。100ms通信是建模值，没有测量网络或启动ROS/CARLA/底盘。仅optimized运行，baseline/optimized pair比较未执行；不作准确率或性能结论。

此检查沿用原机安装的软件栈，没有测试fresh OS重装、ROS往返或完整HIL。原1.60GiB离线材料仍需另行保存。完整wire、stderr和命令退出文件在私有验收目录；公开回执保留wire摘要，不上传约45.8MiB输出副本。复验命令见 [COPT_ENGINE_RESTORE](../COPT_ENGINE_RESTORE.md)，完整范围见 [VALIDATION](../VALIDATION.md)。


## 2026-11-24补包、服务与私有网络配置完善

- [空目录完整还原回执](replay_final_receipt_20261124.json)：23,480个普通文件、4,329,722,681 B；30,759项恢复后核验，缺SDK/坏图反向测试及复原后再次通过。
- [归档、路径与压缩流检查](replay_prepare_receipt_20261124.json)：实际解包的5组/6分卷（不含Docker内部层）。
- [私有配置备份状态](private_configuration_backup_20261124.json)：25份网络/SSH/启动配置已在服务器受限目录保存并试还原；实体、详细manifest和摘要不在Git中。

空目录回执记录发布前工作树（基线commit e896723，已含两份服务修复）；其source_manifest/checker/补包manifest与逐脚本SHA固定实际验收字节，不把该基线HEAD当作未修改的旧版本。此检查仅还原文件，没有安装包、构建或启动设备。完整服务与ROS/ACK的实际回执另按完成状态记录。


- [本轮真实ROS/ACK回执](service_revalidation_20261124.json)及[原始小证据摘要](service_ack_20261124/evidence_manifest.json)：保留首次完整性异常及拒ACK事实，随后组件一次和完整supervised三次成功。根因未定位，不能把重试成功当作已修复损坏原因。
- [定向代码覆盖](recovery_code_coverage_20261124.json)：100项测试，合计82.57%；各文件与新增helper覆盖率分别列出。

本轮实际ROS域为173、独立话题，默认业务域71未验证；接收端为真实协议消费者，非CARLA agent。新系统安装、底盘/ROS1、外部H800/CARLA及性能均不在这次通过范围。
