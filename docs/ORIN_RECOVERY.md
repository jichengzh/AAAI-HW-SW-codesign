# Orin 恢复 CoptV2X 部署

本分支 `orin恢复CoptV2X部署分支` 保存2026-11-23从Jetson AGX Orin整理的恢复资料。完整快照位于 [orin-recovery-20261123/](orin-recovery-20261123/README.md)，包括7,176份源码/配置、111份关键结果（4,852,498字节）、环境版本、安装目录、数据/权重下载来源及SHA256。原约200GB实验缓存不纳入恢复。本次另将4份冻结Copt TensorRT engine（56,499,229字节）放入普通Git的 `engine_payload/`，不使用LFS或Release，并补齐4个checkpoint路径修复步骤。

## 资料入口

| 内容 | 入口 |
|---|---|
| 总说明与保留范围 | [README](orin-recovery-20261123/README.md) |
| 格式化后的执行顺序 | [RESTORE_CHECKLIST](orin-recovery-20261123/RESTORE_CHECKLIST.md) |
| Copt冻结engine、checkpoint路径修复与单帧验收 | [COPT_ENGINE_RESTORE](orin-recovery-20261123/COPT_ENGINE_RESTORE.md) |
| Copt完整服务路径、挂载、有限完成与最终ACK | [COPT_SERVICE_RESTORE](orin-recovery-20261123/COPT_SERVICE_RESTORE.md) |
| ZeroTier、SSH及启动设置的私有恢复 | [PRIVATE_NETWORK_RESTORE](orin-recovery-20261123/environment/PRIVATE_NETWORK_RESTORE.md) |
| 安装目录及项目映射 | [DIRECTORY_MAP](orin-recovery-20261123/DIRECTORY_MAP.md) |
| 数据集/权重下载地址、字节数与样本量 | [DATASETS_AND_WEIGHTS](orin-recovery-20261123/DATASETS_AND_WEIGHTS.md) |
| 环境安装与版本账本 | [ENVIRONMENT_RESTORE](orin-recovery-20261123/environment/ENVIRONMENT_RESTORE.md) |
| 历史关键结果与范围 | [key_results/README](orin-recovery-20261123/key_results/README.md) |
| 备份校验记录与缺口 | [VALIDATION](orin-recovery-20261123/VALIDATION.md) |

系统基线为Ubuntu20.04.6 / aarch64 / L4T R35.6.5（JetPack5.1.7）/ CUDA11.4 / TensorRT8.5.2。宿主ROS为Noetic，Foxy位于Docker中。新系统的安装、编译和运行验收尚需按清单执行。

## 从本分支取得恢复资料

在恢复资料的机器上执行；下列克隆与文件检查不启动推理或底盘：

```bash
git clone --depth 1 --single-branch \
 --branch 'orin恢复CoptV2X部署分支' \
 https://github.com/jichengzh/AAAI-HW-SW-codesign.git
cd AAAI-HW-SW-codesign

# 检查完整快照，以及源码/结果的原始字节校验。
(cd docs/orin-recovery-20261123 && sha256sum -c SHA256SUMS)
python3 -B docs/orin-recovery-20261123/tools/verify_recovery.py \
 --repo docs/orin-recovery-20261123
```

原清单约定业务用户为`jichengzhi`、恢复根目录为`/home/jichengzhi/orin-recovery`。在格式化后的Orin上，把本仓库的快照目录放到清单预期位置：

```bash
mkdir -p /home/jichengzhi/orin-recovery/github
rsync -a docs/orin-recovery-20261123/ /home/jichengzhi/orin-recovery/github/
```

然后按`/home/jichengzhi/orin-recovery/github/RESTORE_CHECKLIST.md`执行。运行历史小车/HIL脚本前先做离线检查；不能用发布控制命令的脚本验证安装。

## 必须另外保存的离线资产

模型/剪枝权重、特定aarch64安装包、单帧输入、DAIR原标注metadata和ROS2镜像共有6个归档/分卷，合计 **1,718,109,787字节（1.600GiB）**，未加入普通Git，也没有为它们创建GitHub Release。

它们已保存在原备份主机的`/home/jichengzhi/V2X/docs/orin-recovery-20261123/offline_assets/`。应从该备份或你另外保存的介质取得完整目录，放到新Orin的`/home/jichengzhi/orin-recovery/offline_assets/`，再按清单校验与解包。仅克隆本分支不能恢复这些原离线包中的自定义二进制资产；本次新增4个Copt engine已在Git内，逐文件大小、SHA256与下载URL单独见 [engine manifest](orin-recovery-20261123/manifests/copt_engine_backup_manifest.json)。

逐文件/分卷SHA256、大小和原安装路径见 [manifests/](orin-recovery-20261123/manifests/)。原离线包自定义资产的公开下载URL仍为null，表示尚未发布；官方可下载数据集与基础权重有独立链接。不要用随机UniAD dummy替代真实部署模型。

## 快照与现有复现工具的关系

该目录保留原始备份包的关键结果及源码；发布时仅移除6份历史脚本中的GPU分配说明/诊断文字，AST除字符串外不变。3个曾含密码命令文件与这6个脚本的原始摘要/大小仅留私有；公开副本摘要及总校验表重新生成，完整原始版本仍在私有备份。包内本地路径、“未上传”及`public_upload_performed=false`描述的是原始采集时状态；本分支提供公开副本与本次4个engine，原1.60GiB离线资产仍单独保存。156MiB审计补包实体仍私有保存；其1581项公开manifest已纳入统一恢复、--offline及--home校验，默认121帧输入和相机SDK可按清单还原。ZeroTier/SSH配置也已私有备份，身份与密钥不发布。

2026-11-24进一步修复两份pipeline服务脚本的有限任务生命周期：worker正常结束后继续等待publisher真实ACK完成，再清理并写completed.json；公开source manifest保留原始SHA并更新published SHA，完整原版仍在私有采集树。公开依赖允许联网按版本重新安装，无需全部离线保存。

本目录中的历史部署源码、设备路径、依赖版本和结果边界独立于仓库现有CPU smoke入口，不作为匿名审稿ZIP的新输入，也不将历史CPU ORT/静态重放/route_timeout记录改写为GPU或闭环驾驶成功。上游及厂商代码保留原有许可证/版权说明；恢复包不重新许可这些第三方代码。

## 既有依赖审计记录

提交前对仓库原有根`requirements.txt`做了直接固定依赖的pip-audit检查：`torch==2.9.0`返回6条风险记录，其中含重复关联，共5个唯一CVE，见[审计记录](ORIN_RECOVERY_DEPENDENCY_AUDIT.json)。这对应原CPU smoke环境，不是对归档Jetson环境或完整依赖闭包的安全认证。本分支未修改这些既有依赖，也未将历史重装版本未经验证地升级；需要部署运行时应单独评估并验证依赖修复。

上一版提交前本地全套测试通过986项，恢复包总表7,346项SHA256与源码/结果7,287项核验通过；这些是历史记录。本次新增engine后的当前文件核验、源码5臂14项冻结哈希及实际推理回执以 [VALIDATION](orin-recovery-20261123/VALIDATION.md) 为准。GitHub CI、原机同栈测试、新Orin重装、ROS往返及完整HIL分别验收，不由文件检查相互替代。UniAD/ROS1默认输入/SDK补包和底盘额外apt/Tk安装步骤现已纳入；PL/torchmetrics兼容及新机编译、设备功能仍单独验收。
