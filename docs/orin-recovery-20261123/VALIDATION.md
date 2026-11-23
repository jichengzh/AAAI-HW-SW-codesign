# 备份验收记录

核验日期：2026-11-23。本页记录资料与资产备份的完整性；不表示格式化后的系统已经安装、编译或通过推理验收。冻结Copt engine补备份与后续原机单帧检查单独记录如下。

**精选备份已全部离开Orin原盘，保存到当前主机 `bm-2s55he` 的 `/home/jichengzhi/V2X/docs/orin-recovery-20261123`，全部资产包已完成逐文件流式SHA256校验。** 本恢复包未执行系统重装或新系统推理验收。

| 离线类别 | 文件数 | 原始字节 | 压缩字节 |
|---|---:|---:|---:|
| 主模型/安装资产 | 362 | 3,610,840,548 | 1,218,492,636 |
| FCooper定制模型 | 8 | 63,156,308 | 36,717,260 |
| ResNet候选/UniAD单帧 | 18 | 245,280,961 | 171,561,243 |
| DAIR split/校准/标签 | 14315 | 116,821,737 | 37,097,047 |
| ROS2 Docker共享层归档 | 3个镜像 | 共享层不重复相加 | 254,241,601 |

离线归档共 **1,718,109,787 B（1.600 GiB）**。逐分卷与逐文件SHA256、原安装相对路径见各manifest；总表见 `manifests/backup_summary.json`。


## 已完成的检查

- 7,176 份原始源码/配置，73,192,234 B；从Orin逐文件计算SHA256，并在当前主机重新计算比对，全部一致。
- 公开源码为73,192,275 B；3个文件的5处密码式SSH/sudo命令已替换，公开哈希与变化位置已记录；这3个原文件的摘要/大小只留私有备份。私有原始树不上传。
- 111份原始关键结果，4,852,498 B；全部与原端SHA256一致，未改写实验字段。
- FCooper8份定制ONNX/checkpoint，63,156,308 B；压缩包36,717,260 B。远端流式哈希、本地解包流哈希一致；按恢复文档实测临时解包，8文件路径/内容再次一致。
- FCooper归档内模型的10项五臂引用哈希与正式canonical_manifest完全一致；3个ResNet候选与原目标accuracy链接指向文件哈希一致。
- DAIR原split/校准/标注：14,315份JSON，116,821,737 B；压缩包37,097,047 B，流式逐文件SHA256验证通过；不含点云或图像。
- ResNet三个独立候选模型、UniAD单帧与可视化字体另存补充资产包，精确文件与哈希见 `manifests/supplemental_assets_manifest.json`。
- ROS2三个镜像以单个共享层归档保留。压缩包254,241,601 B；镜像tag与原端一致，逐配置SHA256/镜像ID、37个layer摘要引用均通过校验。
- 原设备15份串口/雷达/相机udev规则原样保存，14,435 B，附逐文件SHA256与用户组恢复步骤。
- 全部6份离线归档/分卷的SHA256再次核验通过；主包XZ完整性检查通过，并实际解出pruned75 checkpoint，路径/大小/SHA256一致。
- 早期工具曾通过7,293项文件核验；本次工具默认模式核验7,291项（含4个冻结engine），相关106项回归测试通过。删除主manifest而保留补充manifest、缺失/损坏engine和checkpoint别名均有失败测试。
- 恢复文档Bash片段通过 `bash -n`；公开树未纳入数据集、大缓存或大于100MiB的Git普通文件。公开内容做过私钥/token/URL凭据/密码命令等静态模式检查；该检查不构成穷尽的安全证明。
- 只读校验工具验证了正常文件，并确认可拒绝损坏哈希、大小不符、路径越界、缺失和软链接文件。没有执行payload中的程序。

## 复核命令

仓库与所有离线包已收齐后，在保存材料的机器执行（只读）：

```bash
python3 -B github/tools/verify_recovery.py --repo github --offline offline_assets
```

在新Orin将源码与模型恢复原路径后：

```bash
python3 -B /home/jichengzhi/orin-recovery/github/tools/verify_recovery.py \
 --repo /home/jichengzhi/orin-recovery/github --home /home/jichengzhi
```

该工具检查文件和SHA256，不启动推理、不加载Docker、不操作机器人。原始环境版本对照及新系统功能验收另见 [恢复步骤](RESTORE_CHECKLIST.md)。

## 本包有意不覆盖的内容

1. 约139GiB预录BEV输入、重复NPZ/中间输出、全部失败attempt/大日志、旧整目录备份；不会恢复约200GB历史数据。
2. 完整venv/conda/site-packages。保存版本、少量特定wheel与源码，需重装依赖；旧EOL仓库/conda构建长期可取性未保证。
3. 整盘镜像、原刷机介质/启动分区、H800/仿真主机和真实车辆设备状态。UniAD真实tiny原训练pth在其它host的副本未确认；已保存实际部署ONNX，公开dummy不能替代。
4. CPU50000准确率汇总尚未找到；只保存完成记录。P4五臂route_timeout、Pipeline静态重放、CPU ORT范围按原文保留，不能转述为成功的GPU/驾驶性能结果。
5. 下载文档的官方URL、文件大小/数量是核验日记录；未声称已重新下载和解压全量数据。自定义模型没有已发布的个人Release URL。

分支发布额外处理：6份历史脚本的GPU分配说明/诊断字符串已去除固定分配文字；代码AST除字符串外一致，公开哈希已更新。其原始摘要/大小仅留私有。


## 本次冻结Copt engine补备份

- 4份原始engine，共56,499,229 B，已从私有逐文件校验补包提取到本仓库 `engine_payload/`；全部SHA256匹配冻结registry。逐文件下载路径见 [engine manifest](manifests/copt_engine_backup_manifest.json)。
- 另在原盘之外保存独立压缩副本 `copt-frozen-engines-20261123.tar.gz`，48,338,884 B，SHA256=`1764bc848a9b2f59442745f0cbcce1113d4448194b77e6918055817d46b7dbe2`。此压缩副本不重复纳入Git。
- 恢复说明已补入4个models→source checkpoint路径映射，并要求显式Stage6 artifact root。旧默认路径不能代表本备份的实际安装位置。
- 此版本提交时，GitHub发布后的重新下载及Orin真实单帧测试尚待执行；完成后的独立回执将记录实际提交、下载SHA256与每臂退出状态。不得把本段的文件一致性检查解读为已运行推理。
- 当前仅证明现有SDK环境的材料可验收。新JetPack安装、重新编译、ROS往返、外部HIL及车辆闭环仍需另行验收；其他约1.60GiB离线资产仍不在Git仓库。
