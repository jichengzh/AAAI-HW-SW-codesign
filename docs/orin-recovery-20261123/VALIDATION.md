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

原6份离线归档/分卷共 **1,718,109,787 B（1.600 GiB）**。逐分卷与逐文件SHA256、原安装相对路径见各manifest；总表见 `manifests/backup_summary.json`。


## 已完成的检查

- 7,176 份原始源码/配置，73,192,234 B；从Orin逐文件计算SHA256，并在当前主机重新计算比对，全部一致。
- 2026-11-23隐私处理后的公开源码为73,192,275 B（2026-11-24服务修复后为73,195,374 B，published哈希为当前验收依据）；3个文件的5处密码式SSH/sudo命令及6份脚本的固定GPU分配说明已处理，公开哈希与变化位置已记录；这9个原文件的摘要/大小只留私有备份。私有原始树不上传。
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
- 只读校验工具验证了正常文件，并确认可拒绝损坏哈希、大小不符、路径越界、缺失和软链接文件。该工具本身不执行payload；后续实际推理见下节独立记录。

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
- 已发布commit `aa3d68132c6c8b2e3bc74ac03162585cb758fae3`，随后从该不可变commit的公开raw URL实际重新下载4份engine，56,499,229 B，大小及SHA256全部一致。见 [公开下载回执](validation/public_engine_downloads_20261123.json)。
- 当前仅证明现有SDK环境的材料可验收。新JetPack安装、重新编译、ROS往返、外部HIL及车辆闭环仍需另行验收；其他约1.60GiB离线资产仍不在Git仓库。


## 上传后Orin独立恢复副本真实单帧复验

2026-11-23 12:12:13–12:12:36 UTC（北京时间20:12），在Orin新建独立 `/tmp` 目录执行。runtime/native来自恢复包，4份checkpoint与单帧输入从已备份归档选择性提取，4份engine使用**上传后GitHub重新下载**的字节，未借用原部署源码、权重或输入。传输791份文件逐项SHA256一致、0软链接；20个原生模型模块导入路径均来自临时副本。

| 配置 | 实际backend | 推理样本数 | 退出码 | 结果 |
|---|---|---:|---:|---|
| original_default | native_pytorch | 1 | 0 | PASS |
| compression_only | tensorrt | 1 | 0 | PASS |
| schedule_only | tensorrt | 1 | 0 | PASS |
| compress_then_tune | tensorrt | 1 | 0 | PASS |
| joint_shcosearch | tensorrt | 1 | 0 | PASS |

- 源码自己的preflight通过5臂、14项冻结资产hash；实际probe每臂仅1个sample、1个probe_complete。输入float32 `[2,64,256,512]` 的SHA256为 `d685d156c22e59506a436ed8baccdee5ca093739f1ee94c7da01a89fcd802103`。
- 实际执行模型、协议编码/解码、RSU-only多尺度feature合同、shape/float32/finite/身份及解码feature hash检查；wire大小/SHA256另在远端重新计算核对。没有将仅存在wire文件视为通过。
- 逐臂未经改写的原始JSONL、公开回执、依赖及隔离信息见 [validation/](validation/README.md)；完整wire与stderr留私有验收目录，不上传大输出。
- 使用既有L4T35.6.5、Python3.8.19、Torch1.12.0a0+2c916ef.nv22.3/CUDA11.4、TensorRT8.5.2.2、NumPy1.22.4；这证明独立恢复材料在**原机现有软件栈**上可以加载和执行。未测试新系统安装/编译。
- 每臂是冷启动首帧，warm sample=0；仅optimized，`pair_hash_validation=not_run`。通信100ms是模型项，`communication_measured=false`、`is_hil=false`。不据此报告吞吐、网络时延、AP准确率或闭环驾驶性能。
- 五次运行均出现既有Torch“只读NumPy转换”UserWarning；无异常，未改动冻结源码压制警告。ROS、CARLA、外部ACK、车辆与约200GB数据没有启动或恢复。
- 此次发布前本地全量测试 **1,042 passed**；仓库+原离线归档的只读工具 **7,297项通过**；后续GitHub CI与fresh OS验收仍分别记录。


## 2026-11-24恢复流程完善与实际验收

软件采用固定版本适配说明联网重装；无需另存完整公开依赖闭包。版本下载/安装和新系统功能验收仍是不同步骤。

### 已完成的修复

- 公开 `restore_audit_assets_manifest.json` 纳入1581份审计补包文件；`--offline`强制验该归档，`--home`强制验其恢复位置。统一解包包含UniAD默认121帧、847份bin、726张图和Astra三库。
- 安装说明补齐 `libuvc-dev`、三个底盘ROS包和系统/venv的 `python3-tk`；conda沿用自己的tk。增加包发现、SDK路径与输出文件验收要求。
- Copt使用显式P4 input/native/state与pipeline runtime，以及匹配的容器/work、/state。新增 [完整服务说明](COPT_SERVICE_RESTORE.md)，不依赖不适用的父目录默认值。
- 修复两份服务脚本的有限任务生命周期：worker正常退出后等待publisher真实ACK并以0退出，随后成功清理才写completed.json；非零、缺就绪、proxy退出、排空超时和中途停止均不会误报完成。source manifest保留原采集SHA并更新公开修复SHA；原版仍在私有采集树。
- ZeroTier、SSH私钥/认证/host keys和shell设置共25文件已在独立服务器的仓库外私有备份，0700/0600；实际试还原并重算SHA通过。Git只发布 [状态](validation/private_configuration_backup_20261124.json) 和 [恢复方法](environment/PRIVATE_NETWORK_RESTORE.md)。ZeroTier原状态文件归zerotier-one账号，恢复时按新安装包生成状态文件的实际owner处理，不照抄旧UID或无条件改root。

### 从空目录实际还原

在服务器全新临时home实际还原23,480个普通文件，4,329,722,681 B；仓库加七份所需归档/分卷核验 **7,298项通过**，恢复位置核验 **30,759项通过**。实际移除Astra库、破坏一张图时均按预期失败，恢复后再次30,759项通过。121帧info、全部metadata与工作目录下726张图片相对引用均验证。临时4.3GB副本已清理。

[还原回执](validation/replay_final_receipt_20261124.json)和[归档/路径回执](validation/replay_prepare_receipt_20261124.json)固定了实际source/checker/manifest摘要；其中HEAD为发布前工作树的基线，包含两份已明确记录的恢复修复，不能解释为未修改旧commit的测试。该项仅执行文件还原，不安装或运行payload。

### 真实ROS与最终ACK

在原Orin既有SDK、独立容器/state/端口中执行：

| 验收 | 结果 |
|---|---|
| ROS单帧payload往返 | 1次通过 |
| 组件链（worker→publisher→ROS proxy→SSH→真实接收端→ACK） | 首次失败、第二次通过 |
| 修复后的完整监督流程，真实1帧/1触发 | 连续3次通过 |
| 后两次完整监督重复 | 无消费者capture monkeypatch，生产路径及哈希校验不变 |

三次完整流程均实际核对trigger、冻结身份、shape/finite与payload SHA后，由生产transport发送ACK；ROS ACK和ACK后trace被实际观察，supervisor/consumer退出0，completed.mode=publisher_ack、stopped.cleaned=true，无failure.json。原始ACK/trace、consumer、worker/proxy与完成日志见 [实际回执](validation/service_revalidation_20261124.json) 和 [逐文件证据摘要](validation/service_ack_20261124/evidence_manifest.json)。新测试容器全部停删，原业务容器保持原停止状态。

**首轮异常没有隐去：** 初次组件测试收到的wire有连续16字节完整性差异，生产decoder拒绝校验，未产生最终ACK，publisher失败退出。跨环境复算排除了简单NumPy摘要差异，但缺少该次各边界快照，根因仍未定位。后续组件成功和3次完整监督成功不能证明已修复这个偶发损坏原因；原失败日志和wire SHA保留在同一回执。所有成功判据均保留严格校验，没有跳过坏数据或把ready当ACK。

消费端是本地主机的真实协议验证程序，**不是CARLA驾驶agent**。实际测试为隔离ROS domain173和独立话题，默认业务domain71没有在本轮验收。新系统安装、ROS1/底盘、外部H800/CARLA驾驶和性能仍未验收，不据本次结果作这些结论。

### 回归与覆盖

补包回归72项、服务有限完成回归10项及既有相关回归通过；122项恢复/身份/handoff组合通过。定向子进程覆盖100项测试通过，三个改动文件总statement覆盖 **82.57%**；单个Orin监督文件71.57%，新增完成helper合计87.5%，不宣称每个文件都达80%。详见 [覆盖记录](validation/recovery_code_coverage_20261124.json)。完整仓库测试由最终推送前钩子执行，不能以定向测试代替。
