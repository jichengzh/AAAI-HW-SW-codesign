# 格式化后恢复检查单

目标是恢复固定版本的软件、项目源码、定制模型与少量关键结果，**不恢复原约 200GB 的数据/输入缓存**。本包是恢复材料；下列安装、编译和功能检查均须在新系统执行，尚不代表已完成重装验收。实际文件覆盖范围以 `manifests/` 为准。

## 1. 格式化前确认材料可取

- [ ] `github/` 已完整保存到另一台机器/存储介质，源码和结果的 SHA256 核对通过。
- [ ] 与 `github/` 同级的 `offline_assets/critical-assets.tar.xz.part000` 等**全部连续分卷**已保存，并按最终资产清单核对大小与哈希。Git 仓库本身不包含这些大文件。
- [ ] 如需 ROS2 tracing，另有 `offline_assets/ros2-foxy-images.tar.gz`；Docker 镜像不由普通源码目录恢复。
- [ ] 已阅读 [数据与权重来源](DATASETS_AND_WEIGHTS.md) 的自定义模型缺口。基础公开 checkpoint、dummy ONNX 不能替代离线定制权重与真实 tiny ONNX。
- [ ] `key_results/` 和结果 manifest 已保存。保留历史结果只证明当时相应范围的记录存在，不表示新系统已通过测试。

## 2. 刷入匹配系统，固定目录

按 [环境恢复文档](environment/ENVIRONMENT_RESTORE.md) 第 1 节使用官方 **AGX Orin / aarch64 / L4T R35.6.5** 映像与配套固件；R35.6.5 对应 JetPack5.1.7，旧教程中的“5.1.5”不能作为刷机版本依据。刷机目标为 eMMC 还是 NVMe 由实际介质决定。

- [ ] 用户名为 `jichengzhi`，家目录为 `/home/jichengzhi`。
- [ ] 在新 Orin 上只读核对以下结果；先不要安装其他大版本 SDK。

```bash
uname -m                         # aarch64
uname -r                         # 目标历史内核5.10.216-tegra
cat /etc/nv_tegra_release         # R35 / REVISION:6.5
cat /etc/os-release              # Ubuntu20.04 focal
```

将恢复仓库和离线目录放在独立位置，下面示例约定：

```text
/home/jichengzhi/orin-recovery/
├── github/                     # 本仓库检出/复制的内容
└── offline_assets/             # 单独保存的大文件，非Git仓库内容
```

```bash
export RECOVERY_ROOT=/home/jichengzhi/orin-recovery
export RECOVERY_REPO="$RECOVERY_ROOT/github"
export OFFLINE="$RECOVERY_ROOT/offline_assets"
export RECOVERY_ENV="$RECOVERY_REPO/environment"
test -d "$RECOVERY_REPO/payload"
test -f "$OFFLINE/critical-assets.tar.xz.part000"
```

如实际仓库放在别处，只改这几个变量；业务目录仍恢复到原 `/home/jichengzhi/<relative_path>`，见 [目录映射](DIRECTORY_MAP.md)。

## 3. 恢复源码，再解开必要离线资产

先执行完整性检查（含源码、结果和全部离线包；只读）：

```bash
python3 "$RECOVERY_REPO/tools/verify_recovery.py" --repo "$RECOVERY_REPO" --offline "$OFFLINE"
```

这些命令**会写入家目录，但不启动项目/底盘/推理**。先核对分卷完整性，再执行。

```bash
# payload按原home目录布局，只复制已收录源码/配置；不使用--delete。
rsync -a "$RECOVERY_REPO/payload/" /home/jichengzhi/

# 分卷按000、001…顺序拼接；先只读检查压缩流与归档目录。
set -o pipefail
cat "$OFFLINE"/critical-assets.tar.xz.part[0-9][0-9][0-9] | xz -t
cat "$OFFLINE"/critical-assets.tar.xz.part[0-9][0-9][0-9] | tar -tJ | sed -n '1,40p'

# 归档第一层为打包前缀；去掉该层后回到原home relative_path。
cat "$OFFLINE"/critical-assets.tar.xz.part[0-9][0-9][0-9] \
 | tar -xJ --strip-components=1 --exclude=__ASSET_MANIFEST__.json -C /home/jichengzhi
```

FCooper、ResNet/单帧与DAIR标签/校准metadata补充包也应恢复（其中metadata包没有点云/图片；单帧包含六相机图片），归档布局相同：

```bash
for name in fcooper-assets supplemental-assets dataset-metadata; do
 cat "$OFFLINE"/"$name".tar.gz.part[0-9][0-9][0-9] \
  | tar -xz --strip-components=1 --exclude=__ASSET_MANIFEST__.json -C /home/jichengzhi
done
python3 "$RECOVERY_REPO/tools/verify_recovery.py" --repo "$RECOVERY_REPO" --home /home/jichengzhi
```

- [ ] 按资产 manifest 对恢复到 `/home/jichengzhi/` 的文件重新核对 SHA256，尤其自定义 ONNX、checkpoint、engine、校准输入与 NVIDIA Torch wheel。
- [ ] 按 `manifests/source_capture_manifest.json` 核对源码、原 Git commit、补丁与软链接。`rsync -a` 只能恢复 payload 实际收录的链接，链接目标的数据/外部目录仍可能缺失。
- [ ] 不要以 `payload/` 有某个实验目录判断整套原实验树已保留；该目录可能只收录配置/脚本，模型由离线包恢复，完整数据需另下。

## 4. 先装系统组件，按需建Python环境

依 [环境文档](environment/ENVIRONMENT_RESTORE.md) **逐节执行**，不要整篇无检查地粘贴运行：

1. 配置对应 R35 Jetson 源，安装/核对 CUDA11.4、cuDNN8.6、TensorRT8.5.2，编译基础库。
2. ROS1 Noetic 安装到 `/opt/ros/noetic`，先保留 `/usr/bin/python3` 的系统身份。
3. `~/uniad_venv` 用系统 Python3.8 创建，恢复原 Torch1.12；按备份 commit/patch 重编 torchvision、MMCV1.5、mmdet2.25.1。
4. 如确需训练/导出，再恢复 `~/miniforge3` 和 `~/miniforge3/envs/uniad`，使用该环境自己的 MMCV1.6、mmdet2.26、mmdet3d1.0.0rc6。不要与 venv 的包混用。
5. 如需 ResNet 对照实验，再建 `isolated_torch_compat_r01/venv` 的 Torch2.1 与 ORT 的两个 `--target` 目录；r01用Python3.8，r02用Python3.12。

**离线包已恢复原位置，环境文档中的 wheel 变量需要这样设置：**

```bash
export WHEEL_ROOT=/home/jichengzhi
export TORCH112_WHEEL=/home/jichengzhi/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
test -f "$TORCH112_WHEEL"
# 若从环境文档复制了示例WHEEL_ROOT/TORCH112_WHEEL赋值，以这里的实际原路径为准。
```

ResNet wheel 已在 `~/resnet_compare_s1_20261013/{isolated_torch_compat_r01/wheels,ort_cpu_20261026_r01/wheels,ort127_cp312}/`，无需搬到新的统一 wheel 目录。具体有哪些依赖 wheel 已保存以资产 manifest 为准；未保存的公开包仍需联网安装。完整包版本表是审计账本，不等于可直接安装的 requirements。

- [ ] 核对三处 MMCV/mmdet editable 源路径，重新编译 CUDA 扩展；不要直接使用旧环境/旧 `.so`。
- [ ] 原conda环境内 `cudatoolkit11.1.1` 与宿主 CUDA11.4 是两层历史状态；核对实际动态库解析，不能用 conda CUDA 覆盖 Jetson BSP。
- [ ] 记录新装版本与历史快照的差异；下载不到旧构建时保留“未恢复”状态，不能静默替换为最新版本。

## 5. 重新编译本地代码

以下是**编译步骤**，会生成新的 `build/devel` 等目录。不会自动启动旧实验。

真实小车还需要先按 [设备规则恢复](environment/DEVICE_RULES.md) 安装15份已保存规则并恢复设备访问组；完全重新登录后检查别名。

ROS工作区：先完成依赖安装和源文件/patch恢复，再按环境文档运行 `rosdep` 与：

```bash
source /opt/ros/noetic/setup.bash
cd /home/jichengzhi/ROS1/wheeltec_robot
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -j2
source devel/setup.bash
```

完整软链接与具体CMake编译命令见 [源码恢复说明](SOURCE_PAYLOAD.md)。

UniAD C++：在 `/home/jichengzhi/liruozhan/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2` 使用备份源码及其 `CMakeLists.txt` 重建 `uniad_udp` 和 `libuniad_plugin.so`。先检查源码要求的 CUDA/TRT/OpenCV路径和构建选项，再执行项目构建命令；**不要把原 build 目录或另一个 TensorRT 大版本编出的插件当作已恢复产物**。引擎发生版本不兼容时从保存的 ONNX 重新构建，按原 shape、精度、plugin 和校准设置执行，不能简单用通用 trtexec 命令替代原流程。

Pyramid / CoDriving / FCooper / ResNet：按各目录保存的模型定义、配置、adapter 与构建命令恢复；优先使用已保存自定义 ONNX 重建对应引擎。上游公开基础权重不等同于原剪枝宽度/获胜点。

## 6. ROS2 tracing按需恢复

宿主实际是 ROS1 Noetic；Foxy/tracing 在 Docker 内。先按 [Docker精确恢复步骤](environment/DOCKER_RESTORE.md) 安装26.1.3及配套组件；原daemon数据目录为 `/var/lib/docker`。仅需要相关实验时，在 Docker/NVIDIA容器运行时安装后执行：

```bash
test -f "$OFFLINE/ros2-foxy-images.tar.gz"
set -o pipefail
gzip -dc "$OFFLINE/ros2-foxy-images.tar.gz" | docker load
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
```

期望历史tag包括 `ditl/ros-foxy:arm64-exact`、`ditl/ros-foxy-tracing:arm64`、`ditl/ros-foxy-tracing-enabled:arm64`，以实际导出清单为准。`/opt/ros/foxy`、`/opt/ros2-tracing-overlay`、`/workspace` 是容器内路径，不在宿主手工创建空目录冒充安装。先核对原 Dockerfile、运行命令中的挂载、网络和tracing设置；本检查单不自动启动旧trace/HIL脚本。

## 7. 只读检查通过，再补数据与做功能验收

下列是**只读版本/导入/路径检查**，不是模型精度或闭环验收：

```bash
/usr/local/cuda/bin/nvcc --version
source /opt/ros/noetic/setup.bash
rosversion -d
rospack find uniad_bridge
/usr/bin/python3 -B -c 'import rospy,tensorrt; from geometry_msgs.msg import Twist; print(tensorrt.__version__)'
/home/jichengzhi/uniad_venv/bin/python -B -c \
 'import torch,torchvision,mmcv; print(torch.__version__,torchvision.__version__,mmcv.__version__)'
# 只列出断链；根据source manifest及数据下载情况逐项处理。
find /home/jichengzhi/ROS1 /home/jichengzhi/heal_research \
 /home/jichengzhi/liruozhan/toOrinV2 -xtype l -print 2>/dev/null
```

- [ ] 用 [数据与权重来源](DATASETS_AND_WEIGHTS.md) 下载当前要运行的那套数据。**不下载、不恢复旧约200GB缓存、所有中间输入、失败attempt、旧环境目录和大日志。**
- [ ] DAIR完整评测核对项目1789条split及两侧对应PCD；原60对smoke目录不能代替。官方数据全部解压完成后，再回放本项目metadata，避免官方包覆盖旧split/标注：

```bash
cat "$OFFLINE"/dataset-metadata.tar.gz.part[0-9][0-9][0-9] \
 | tar -xz --strip-components=1 --exclude=__ASSET_MANIFEST__.json -C /home/jichengzhi
python3 -B "$RECOVERY_REPO/tools/verify_recovery.py" --repo "$RECOVERY_REPO" --home /home/jichengzhi
```

这份metadata只对应原`dair_val_full`评测子集；不替代完整训练集metadata。

- [ ] OPV2V完整评测核对实际loader2170条，不以实验内小子集替代 `/data/lsw-data/OPV2V/test`。
- [ ] ImageNet全量50000张若不在本机，从合法来源重下或重新连接原host；stream completion不是准确率。
- [ ] UniAD输入帧及图片相对路径需按需恢复；单个预录BEV输入只用于相应离线smoke，不代表已恢复整个CARLA/HIL场景缓存。
- [ ] 先手动执行项目离线小输入验收，分别记录模块导入、算子、模型输出、数值/精度、shape和动态库问题；新结果写新目录。
- [ ] 最后才按小车/HIL项目流程连接真实设备。旧 `uniad_closedloop.sh` 会发布控制并启动底盘，不应拿它做安装检测。

历史关键结果保留在仓库 `key_results/`，读 [结果边界说明](key_results/README.md) 后引用。恢复完成须有**新系统的新验收记录**；原 `route_timeout/FailedDriving`、静态重放、CPUExecutionProvider结果，不能改写成GPU或闭环性能成功。
