# Orin 格式化后恢复清单

2026-11-23 从原 Jetson AGX Orin 只读整理。目标是保住自定义代码、最关键原始结果和无法再次下载的模型，以原目录恢复运行环境；大数据集按需重下；仅额外保留本项目DAIR split/标注/校准metadata，不搬回约 200 GB 的历史实验缓存。本次没有格式化、安装或启动推理，也未上传 GitHub。

## 从这里开始

1. [逐步恢复操作](RESTORE_CHECKLIST.md)：系统、源码、离线资产、环境、数据与验收的执行顺序。
2. [具体安装目录及项目内容](DIRECTORY_MAP.md)。
3. [环境版本与安装命令](environment/ENVIRONMENT_RESTORE.md)：原设备实际为 **Ubuntu 20.04.6 / L4T R35.6.5（JetPack 5.1.7）/ CUDA 11.4 / TensorRT 8.5.2**。ROS1 为宿主 Noetic；ROS2 Foxy 在 Docker 中。
4. [数据集、权重下载信息](DATASETS_AND_WEIGHTS.md)：官方网站、具体文件、字节数/样本量、访问条件、原安装路径及未核实信息。
5. [串口/雷达/相机规则](environment/DEVICE_RULES.md)：保存了15份设备规则及设备组恢复步骤。
6. [关键实验结果入口](key_results/README.md)：111 份原始结果约 4.63 MiB，逐文件 SHA256 和有效性边界见 manifest。
7. [自定义源码及恢复说明](SOURCE_PAYLOAD.md)。
8. [备份验收记录](VALIDATION.md)：最终文件数量、大小、校验情况及缺口。

## GitHub 与离线文件如何放

当前 `github/` 目录可以作为独立仓库根目录。**只上传这个目录内经过检查的内容**；没有替你执行 git add/commit/push，也没有创建远端仓库。

```text
orin-recovery-20261123/
├── github/                  ← 上传到你的GitHub仓库
│   ├── README.md
│   ├── RESTORE_CHECKLIST.md
│   ├── DIRECTORY_MAP.md
│   ├── DATASETS_AND_WEIGHTS.md
│   ├── SOURCE_PAYLOAD.md
│   ├── VALIDATION.md
│   ├── environment/          环境快照、版本与安装步骤
│   ├── payload/              以原home为根的源码/配置副本
│   ├── key_results/          关键原始实验结果
│   └── manifests/            文件来源、下载来源及校验值
├── offline_assets/          ← 单独保管；不属于普通Git提交
│   ├── critical-assets.tar.xz.part*
│   ├── fcooper-assets.tar.gz.part*
│   ├── supplemental-assets.tar.gz.part*
│   ├── dataset-metadata.tar.gz.part*
│   ├── ros2-foxy-images.tar.gz
│   ├── SHA256SUMS
│   └── SHA256SUMS.docker
└── work/                    私有采集过程/未经公开整理的原始源码；不上传
```

离线包包含真实部署的 UniAD ONNX、必要剪枝权重/导出、一个测试输入、特定 aarch64 安装包和本地 ROS2 镜像。它们没有已发布的个人下载 URL；**仅上传普通 Git 仓库不足以保全它们**。请将 `offline_assets/` 另外存到不会格式化的盘，或将有再分发许可的模型包自行上传 GitHub Release/其他备份存储（ROS2镜像建议保留为私有备份），再把真实下载 URL 填入 manifests。分卷名称、大小和 SHA256 已留在仓库中；此刻 URL 为 null，表示尚未发布。

## 保留与舍弃

| 内容 | 处理 |
|---|---|
| 自定义部署代码、模型结构/配置、DAIR原split、关键原始结果 | 保留在本仓库；发布副本如有凭据会替换，并记录变化 |
| 无公开替代的训练/剪枝权重、UniAD实际ONNX、少量必要离线包 | 放入独立离线压缩包，逐文件校验 |
| DAIR / OPV2V / ImageNet / 基础预训练权重 | 留下载信息；按需重新取得，不恢复原整套数据 |
| P4约139GiB输入缓存、重复NPZ/中间输出、大量历史attempt、旧备份压缩包 | 不保留；P4只留1份64MiB输入用于静态冒烟 |
| venv/conda整个目录、旧build/devel、大多数TensorRT engines | 不复制；依据版本清单重建。保留1个实际UniAD FP16 engine作为同栈参考 |
| 密钥、密码、令牌、shell历史、原始环境变量 | 不纳入公开恢复包 |

下载重点：DAIR点云五卷加metadata约 **20.43 GiB**，本项目验证split是 **1789**，当前上游split为1783，不能互换。OPV2V test 下载包约 **19.70 GiB**，原项目使用 **2170** 个ego时刻。详细体量、解压占用和原目录差异见下载文档，不把压缩包大小当作解压大小。

## 恢复边界

这是“可审查的恢复资料与精选资产”，不是整盘镜像。恢复命令需要在新系统执行并验收，尚未证明重装后的推理成功。公开 UniAD tiny dummy 是随机权重，不能替代离线包中的真实自定义模型。P4 的 P4_PASS 只代表联调验收，五臂均 route_timeout；CPU ORT、静态重放和GPU/TRT结果在关键结果导航中分别标明。缺失的50000样本准确率汇总、原训练host、完整输入流等不能由完成标记推导出来。

所有公开文件的总校验表为 `SHA256SUMS`；上传前或重新克隆后，在本仓库根目录运行 `sha256sum -c SHA256SUMS`。
