# 上传后的真实恢复副本验收

本目录保留2026-11-23上传四份冻结engine后的最小关键结果。没有恢复历史大缓存。

- [公开下载校验](public_engine_downloads_20261123.json)：commit固定URL、每文件字节数与SHA256；实际HTTP下载后重新计算。
- [五配置实际推理回执](engine_revalidation_20261123.json)：逐臂退出码、样本/完成事件、模型身份、输入/输出合同、feature hash和wire大小/摘要。
- [软件与隔离快照](isolation_software.json)：既有SDK/conda、791份传输文件、临时目录及20份原生模块来源。快照时CUDA尚未初始化，后续probe实际运行模型。
- 原始JSONL：[original_default](original_default.jsonl)、[compression_only](compression_only.jsonl)、[schedule_only](schedule_only.jsonl)、[compress_then_tune](compress_then_tune.jsonl)、[joint_shcosearch](joint_shcosearch.jsonl)。这些文件从Orin直接复制，字节未改写，SHA256写入推理回执。

每个配置实际执行一次冷启动首帧，并检查TensorRT/原生模型输出、协议编解码、finite/shape/feature hash。探针沿用 `static_replay_same_host_process_cpu_diagnostic` 的scope标签；backend字段记录实际native_pytorch或tensorrt，时间字段仅保留诊断语境。100ms通信是建模值，没有测量网络或启动ROS/CARLA/底盘。仅optimized运行，baseline/optimized pair比较未执行；不作准确率或性能结论。

此检查沿用原机安装的软件栈，没有测试fresh OS重装、ROS往返或完整HIL。原1.60GiB离线材料仍需另行保存。完整wire、stderr和命令退出文件在私有验收目录；公开回执保留wire摘要，不上传约45.8MiB输出副本。复验命令见 [COPT_ENGINE_RESTORE](../COPT_ENGINE_RESTORE.md)，完整范围见 [VALIDATION](../VALIDATION.md)。
