# AGX Orin 环境恢复（2026-11-23 只读快照）

本目录是版本账本与重装步骤，不是系统镜像。采集仅运行版本/包元数据查询、读取脚本依赖和 Git 标识；没有安装、修改远端或运行推理。**“已采集安装版本”不等于本次已重新验证推理成功**。以下重装命令尚未在格式化后的机器执行。

## 1. 固定系统基线，再恢复软件

| 层 | 当前实际安装 | 安装/运行位置 |
|---|---|---|
| 板卡/架构 | Jetson AGX Orin Developer Kit / aarch64 | Jetson，非 x86_64、非 ARM SBSA 服务器 |
| OS / BSP | Ubuntu 20.04.6 focal；L4T **35.6.5**；5.10.216-tegra | 系统映像、内核、固件、QSPI 必须配套 |
| CUDA | toolkit 11.4.19；nvcc **11.4.315**；cudart 11.4.298 | `/usr/local/cuda-11.4`；`/usr/local/cuda` 指向该版本 |
| cuDNN | 8.6.0.166-1+cuda11.4 | `/usr/lib/aarch64-linux-gnu` |
| TensorRT | libnvinfer 8.5.2-1+cuda11.4；Python/元包 8.5.2.2 | `/usr/lib/aarch64-linux-gnu`、`/usr/src/tensorrt`、`/usr/lib/python3.8/dist-packages` |
| ROS1 | Noetic；ros_comm/rospy 1.17.4；catkin 0.8.12 | `/opt/ros/noetic` |
| ROS2（可选） | **Docker 内 Foxy**；宿主未发现 `/opt/ros/foxy` | `ditl/ros-foxy-tracing-enabled:arm64` 等本地镜像 |

旧教程称“JetPack 5.1.5 = R35.6.5”，应纠正：官方 **R35.6.5 属于 JetPack 5.1.7**；官方 5.1.5 使用 R35.6.1。当前没有 `nvidia-jetpack` 元包，恢复以 BSP 和各组件版本为准。[NVIDIA R35.6.5](https://developer.nvidia.com/embedded/jetson-linux-r3565)、[JetPack 5.1.5 官方说明](https://developer.nvidia.com/embedded/jetpack-sdk-515)

优先用官方 R35.6.5 BSP + 对应 sample rootfs/SDK Manager 给 **AGX Orin** 刷机，保留用户名 `jichengzhi` 与家目录，原机实测根目录位于 `/dev/nvme0n1p1`（ext4），板卡兼容规格为 `3701--0005--1--jetson-agx-orin-devkit-`，见 `storage_layout.json`。新机仍需核对实际接入NVMe和启动介质后选择刷机目标。不要直接照抄另一板型或磁盘的 flash 命令。`flash.sh` 与 initrd flash 的选择和板型参数以该下载包内说明为准；本清单没有执行刷机，也没有保存整盘克隆。[官方刷机说明](https://docs.nvidia.com/jetson/archives/r35.6.0/DeveloperGuide/SD/FlashingSupport.html)

刷机后先检查，再安装缺少的 Jetson 组件（以下在 **Orin** 执行）：

```bash
uname -m
cat /etc/nv_tegra_release
cat /etc/os-release
apt-cache policy cuda-toolkit-11-4 libcudnn8 libnvinfer8 python3-libnvinfer
# 只有已配置对应R35仓库、这些精确版本可见时执行；不可用则回到匹配的SDK/BSP安装源。
sudo apt-get update
sudo apt-get install cuda-toolkit-11-4=11.4.19-1 \
  libcudnn8=8.6.0.166-1+cuda11.4 libcudnn8-dev=8.6.0.166-1+cuda11.4 \
  libnvinfer8=8.5.2-1+cuda11.4 libnvinfer-dev=8.5.2-1+cuda11.4 \
  libnvinfer-plugin8=8.5.2-1+cuda11.4 libnvinfer-plugin-dev=8.5.2-1+cuda11.4 \
  libnvonnxparsers8=8.5.2-1+cuda11.4 libnvonnxparsers-dev=8.5.2-1+cuda11.4 \
  python3-libnvinfer=8.5.2-1+cuda11.4 libnvinfer-bin=8.5.2-1+cuda11.4
sudo apt-get install build-essential cmake git curl wget pkg-config \
  python3-dev python3-pip python3-venv python3-tk libopenblas-dev libopenmpi-dev openmpi-bin \
  libjpeg-dev zlib1g-dev libeigen3-dev libgoogle-glog-dev libuvc-dev
export CUDA_HOME=/usr/local/cuda-11.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

禁止用 Ubuntu x86 CUDA deb、桌面显卡驱动或 `pip install tensorrt` 替换 Jetson BSP 组件。不要执行整机 `apt upgrade` 来“修复”版本不一致。CUDA 11.4/TRT 8.5 引擎和插件在新大版本环境不算已验证；升级 JetPack/CUDA/TRT/PyTorch 是单独的迁移任务。

## 2. 先恢复最小业务链：ROS1 + C++ UniAD

固定工作区 `/home/jichengzhi/ROS1/wheeltec_robot`；应用和资产根目录 `/home/jichengzhi/liruozhan/toOrinV2`。`uniad_bridge` 只依赖 `rospy`、`geometry_msgs`、`nav_msgs`、`std_msgs`、`std_srvs`，使用 `/usr/bin/python3`；无需先恢复训练环境。C++ UDP 程序使用 TensorRT 引擎与 `libuniad_plugin.so`，不是 Python mmcv 推理。

ROS Noetic 已 EOL；命令能否取到**相同构建号**取决于仓库保留情况。先配置 ROS 官方 arm64 focal 源，再看 `apt-cache policy`，不能把最新可取版本写成旧实验版本。

```bash
sudo apt-get install ca-certificates curl gnupg
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg >/dev/null
printf '%s\n' 'deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://packages.ros.org/ros/ubuntu focal main' \
  | sudo tee /etc/apt/sources.list.d/ros1.list
sudo apt-get update
sudo apt-get install ros-noetic-ros-base ros-noetic-geometry-msgs ros-noetic-nav-msgs \
  ros-noetic-std-msgs ros-noetic-std-srvs python3-catkin-tools python3-rosdep \
  ros-noetic-robot-pose-ekf ros-noetic-joint-state-publisher ros-noetic-robot-state-publisher \
  python3-rospkg python3-empy python3-serial
source /opt/ros/noetic/setup.bash
# 将备份的已修复src放回原目录；不要复制旧build/devel作为重建结果。
cd /home/jichengzhi/ROS1/wheeltec_robot
# 首次初始化用 sudo rosdep init；已初始化则跳过。
rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src --rosdistro noetic -r -y
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -j2
source devel/setup.bash
rosversion -d
rospack find uniad_bridge
for package in robot_pose_ekf joint_state_publisher robot_state_publisher; do
  rospack find "$package"
done
```

完整小车工作区还含雷达、相机、导航厂商包，应使用备份的**已裁剪/修复 src**和对应项目手册；`rosdep` 可能因 EOL/厂商包缺项失败，应逐项查明，不以 `-r` 的继续执行视为全部成功。全套已装 ROS 包及构建号在 `dpkg_versions.tsv`，无需为最小桥接链复制原来所有桌面组件。

Astra相机的CMake通过 `pkg-config` 查找 `libuvc`，故上面显式安装 `libuvc-dev`。私有 `restore-audit-supplement.tar.gz.part000` 保存了 `ROS1/wheeltec_robot/src/ros_astra_camera-main/include/openni2_redist/arm64/` 下的 `libOpenNI2.so`、`OpenNI2/Drivers/libOniFile.so` 和 `liborbbec.so`。先按 [恢复检查单](../RESTORE_CHECKLIST.md) 第3节解包并通过 `--home` 校验，再运行catkin构建；只复制文本源码无法补出这些厂商库。编译前可只读检查：

```bash
pkg-config --modversion libuvc
ASTRA=/home/jichengzhi/ROS1/wheeltec_robot/src/ros_astra_camera-main/include/openni2_redist/arm64
test -f "$ASTRA/libOpenNI2.so"
test -f "$ASTRA/OpenNI2/Drivers/libOniFile.so"
test -f "$ASTRA/OpenNI2/Drivers/liborbbec.so"
```

C++ 应用先在原 `.../DL4AGX_min/uniad-trt/inference_app/enqueueV2` 重建，使用配套源文件/CMake。引擎与插件、输入 `info.txt` 中相对图片路径必须同时到位；运行脚本硬编码该路径。具体资产及 CMake 见代码/资产恢复清单。引擎/插件跨 TensorRT 版本要重新构建并验收，不能只改文件名。

## 3. Python 环境分开恢复，禁止混装

| 环境 | Python / 关键安装版本 | 原位置和用途 |
|---|---|---|
| 系统 | 3.8.10；numpy 1.17.4；TRT 8.5.2.2 | `/usr/bin/python3`；ROS、系统工具 |
| UniAD venv | 3.8.10；torch 1.12.0a0+2c916ef.nv22.3；numpy 1.22.4；mmcv-full 1.5.0；mmdet 2.25.1；mmseg 0.30.0 | `~/uniad_venv`；**system-site-packages=true** |
| UniAD conda | 3.8.19；同版 torch；numpy 1.22.4；mmcv-full 1.6.0；mmdet 2.26.0；mmdet3d 1.0.0rc6；mmseg 0.29.1 | `~/miniforge3/envs/uniad` |
| Miniforge base | 3.12.10；conda 25.3.0 | `~/miniforge3`；ORT r02 使用的解释器 |
| 独立 Torch | 3.8.10；torch 2.1.0a0+41361538.nv23.6；numpy 1.22.4 | `~/resnet_compare_s1_20261013/isolated_torch_compat_r01/venv`；不继承系统包 |
| ORT r01 | onnxruntime 1.19.2，cp38 aarch64 | `~/resnet_compare_s1_20261013/ort_cpu_20261026_r01/python`，pip `--target` 目录，无独立解释器 |
| ORT r02 | onnxruntime 1.27.0，numpy 1.26.4，protobuf 7.36.2，cp312 | `~/resnet_compare_s1_20261013/ort_cpu_20261026_r02/python`，由 Miniforge Python3.12 使用 |

两套 UniAD 当前都通过用户目录引用 `torchvision 0.13.0a0+da3794e` egg；不能只重建 venv 就认为 torchvision 齐全。MMCV 三处 editable 源是：`~/mmcv_build/mmcv`（v1.5.0，235c0253ab8806a2a2ee6954b4258a95358497ac）、`~/mmcv_build/mmdetection`（v2.25.1，3b72b12fe9b14de906d1363982b9fba05e7d47c1）、`~/mmcv`（v1.6.0，f4167fe1e3d106cd641708d269ddbff568393437）。采集时这三个 Git 工作区干净，仍以代码备份中的补丁与锁定记录为准。

下列 `WHEEL_ROOT` 为恢复到 Orin 的离线安装资产目录；按实际备份目录设定。本清单不存 200GB 缓存，也不要求复制虚拟环境。`RECOVERY_ENV` 指本目录克隆/复制后的绝对路径。

公开apt/pip/conda依赖允许联网按固定版本重新安装，无需另存完整离线闭包。`python3-tk` 为系统Python及其venv提供Tk支持；原conda清单已含 `tk=8.6.13`，分别在各自解释器检查，不向conda环境注入系统Tk库。

```bash
export RECOVERY_ENV="${RECOVERY_ENV:-/home/jichengzhi/orin-recovery/github/environment}"
export WHEEL_ROOT="${WHEEL_ROOT:-/home/jichengzhi}"
export TORCH112_WHEEL="$WHEEL_ROOT/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl"
test -f "$TORCH112_WHEEL"
/usr/bin/python3 -m venv --system-site-packages /home/jichengzhi/uniad_venv
P=/home/jichengzhi/uniad_venv/bin/python
"$P" -B -c 'import tkinter; print(tkinter.TkVersion)'
"$P" -m pip install 'pip==24.3.1' 'setuptools==75.3.0' wheel
"$P" -m pip install --no-cache-dir "$TORCH112_WHEEL"
"$P" -m pip install --no-cache-dir -r "$RECOVERY_ENV/requirements_uniad_venv_runtime.txt"
```

Torch 使用原 `cp38-linux_aarch64` NVIDIA wheel，来源/JetPack兼容要求以 [NVIDIA 安装说明](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/) 与[版本矩阵](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)为准。1.12 wheel 是旧实验现状，不把官方新 wheel 当作已测替代。不要照 UniAD 2.0 的通用文档安装 x86 `cu118` torch。

将 torchvision 源码放在原目录 `~/vision`（包版本指向短 commit `da3794e`，已核验完整 HEAD `da3794e90c7cf69348f5446471926729c55f243e`，工作区干净；按源码 manifest 锁定该 commit；无备份时先 `git clone https://github.com/pytorch/vision.git ~/vision` 并 checkout 对应 commit）；在**每套 torch1.12 环境分别**执行：

```bash
cd /home/jichengzhi/vision
# 没有源码补丁时才checkout；已有备份源码时保留备份状态。
# git checkout da3794e90c7cf69348f5446471926729c55f243e
export BUILD_VERSION=0.13.0a0+da3794e
"$P" -m pip install --no-build-isolation --no-deps .
```

这会改为各环境内自包含安装，避免继续依赖 `~/.local` egg；这项重建方法尚未在新系统执行。仅保留该 egg 可用于应急同路径恢复，但不替代源码重编译验收。

用保存的源目录（或按下列 commit 重新 checkout 并应用保存补丁）重编 MMCV，不能沿用旧 editable 路径或原 `.so`：

```bash
mkdir -p /home/jichengzhi/mmcv_build
# 恢复备份源码；若无本地源码才执行以下clone，并保留备份补丁。
git clone https://github.com/open-mmlab/mmcv.git /home/jichengzhi/mmcv_build/mmcv
cd /home/jichengzhi/mmcv_build/mmcv
git checkout 235c0253ab8806a2a2ee6954b4258a95358497ac
export CUDA_HOME=/usr/local/cuda-11.4
export TORCH_CUDA_ARCH_LIST=8.7
export MMCV_WITH_OPS=1
export MAX_JOBS=2
"$P" -m pip install --no-build-isolation --no-deps -v -e .
git clone https://github.com/open-mmlab/mmdetection.git /home/jichengzhi/mmcv_build/mmdetection
cd /home/jichengzhi/mmcv_build/mmdetection
git checkout 3b72b12fe9b14de906d1363982b9fba05e7d47c1
"$P" -m pip install --no-deps -e .
"$P" -m pip install mmsegmentation==0.30.0
```

运行子集pin已与PyPI官方 `requires_python` 元数据逐项核对，均允许Python3.8.10，见 `requirements_compatibility_audit.json`；此检查不验证CUDA ABI或完整依赖兼容。

MMCV/MMDet最小导入依赖、PyYAML的apt继承以及必要的wheel/源码构建说明见 [IMPORT_DEPENDENCY_NOTES.md](IMPORT_DEPENDENCY_NOTES.md)。

`requirements_*_runtime.txt` 仅是已采集的第三方运行包子集，不是完整依赖闭包；pip resolver/依赖冲突需按 TSV 与源 `requirements` 补齐。完整审计表不能直接 `pip install -r`：含系统 apt 包、重复继承包、editable 源及失效元数据。`uniad_venv` 当前未检测到 mmdet3d 的有效发行元数据；不要凭猜测在该环境补装某一版。导出/训练应优先复原有完整 mmdet3d 的 conda 环境。

恢复 conda 环境（可选，主要用于导出/训练）：

```bash
# 下载Miniforge Linux-aarch64历史安装器到本地，再安装到固定路径；原安装器修订号未采到。
# 25.3.0-3是可选重建安装器，不声称是原机器的安装器。
curl -fL -o /tmp/Miniforge3-Linux-aarch64.sh \
 https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-Linux-aarch64.sh
bash /tmp/Miniforge3-Linux-aarch64.sh -b -p /home/jichengzhi/miniforge3
C=/home/jichengzhi/miniforge3/bin/conda
"$C" install -n base -c conda-forge python=3.12.10 conda=25.3.0
"$C" create -y -p /home/jichengzhi/miniforge3/envs/uniad -c conda-forge \
 --file "$RECOVERY_ENV/conda_uniad_specs.txt"
P=/home/jichengzhi/miniforge3/envs/uniad/bin/python
"$P" -B -c 'import tkinter; print(tkinter.TkVersion)'
"$P" -m pip install --no-cache-dir "$TORCH112_WHEEL"
"$P" -m pip install --no-cache-dir -r "$RECOVERY_ENV/requirements_miniforge_uniad_runtime.txt"
# 在此环境按前述步骤重编torchvision。
git clone https://github.com/open-mmlab/mmcv.git /home/jichengzhi/mmcv
cd /home/jichengzhi/mmcv
git checkout f4167fe1e3d106cd641708d269ddbff568393437
CUDA_HOME=/usr/local/cuda-11.4 TORCH_CUDA_ARCH_LIST=8.7 MMCV_WITH_OPS=1 MAX_JOBS=2 \
 "$P" -m pip install --no-build-isolation --no-deps -v -e .
"$P" -m pip install -c "$RECOVERY_ENV/requirements_miniforge_uniad_runtime.txt" \
 mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6 \
 pytorch-lightning==1.2.5 nuscenes-devkit==1.1.11
```

[Miniforge 官方历史发行](https://github.com/conda-forge/miniforge/releases/tag/25.3.0-3)。conda pin 依据 `conda-meta` 的 linux-aarch64 名称/版本/build，故意不保存可能含认证信息的 channel URL；旧构建下架时需要历史包/改用同版本重新解算并标记差异。原 conda 环境还装有 **cudatoolkit 11.1.1**（h3b7cbac_13），这是历史环境内的包，并非宿主 Jetson CUDA/驱动版本；conda清单按历史现状记录它，不能用它覆盖 `/usr/local/cuda-11.4` 或替换BSP。重建后必须核对实际动态库解析。Numba/llvmlite 等优先用此 conda 清单恢复；不要静默换成最新版本。Conda 环境看不到系统 TRT 时，**仅给需要 TRT 的命令**附加 `PYTHONPATH=/usr/lib/python3.8/dist-packages`，并再次检查包解析顺序，勿污染整个 conda base。

## 4. ResNet 比较专用环境（可选）

```bash
R=/home/jichengzhi/resnet_compare_s1_20261013
/usr/bin/python3 -m venv "$R/isolated_torch_compat_r01/venv"
P="$R/isolated_torch_compat_r01/venv/bin/python"
# 把该目录原来的10个wheel恢复到原wheels目录；所有依赖从此处安装。
"$P" -m pip install --no-index --find-links "$R/isolated_torch_compat_r01/wheels" \
 numpy==1.22.4 sympy==1.12 mpmath==1.3.0 Jinja2==3.1.6 MarkupSafe==2.1.5 \
 fsspec==2025.3.0 networkx==3.1 typing_extensions==4.13.2 filelock==3.16.1
"$P" -m pip install --no-index --no-deps \
 "$R/isolated_torch_compat_r01/wheels/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
# ORT r01是CPU wheel，不是CUDA EP或TensorRT EP。
/home/jichengzhi/uniad_venv/bin/python -m pip install --no-deps \
 --target "$R/ort_cpu_20261026_r01/python" \
 "$R/ort_cpu_20261026_r01/wheels/onnxruntime-1.19.2-cp38-cp38-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
# ORT r02必须Python3.12，不能把cp312包加到Python3.8路径。
/home/jichengzhi/miniforge3/bin/python -m pip install --no-index --no-deps \
 --target "$R/ort_cpu_20261026_r02/python" "$R"/ort127_cp312/*.whl
```

r01 单包 `--target` 目录的 numpy/flatbuffers/packaging/protobuf 等来自调用解释器/其他路径，需恢复调用脚本相同的解释器和 `PYTHONPATH`。r02 的 `PYTHONPATH` 仅指向 r02/python，以覆盖 base 中 numpy 2.4.6。原 wheel 名写 `nv23.06`，装入后标准化为 `nv23.6`，这是同一个 Torch。

ORT1.19.2 cp38 aarch64 wheel 的本地安装元数据 SHA256 `5bd8b875757ea941cbcfe01582970cc299893d1b65bd56731e326a8333f638a3` 与 [PyPI 官方1.19.2元数据](https://pypi.org/pypi/onnxruntime/1.19.2/json)一致，属于公开发行 wheel。1.27.0 cp312 aarch64 的官方哈希 `c6fddce0539a4898c7bef35b052ffd37935b2190e35488eab99ce91887743ea1` 见 [PyPI1.27.0](https://pypi.org/pypi/onnxruntime/1.27.0/json)；采集时未重新计算远端 r02 wheel 哈希，交由资产清单校验。目录中的“wheels”是有限离线安装资产，不是建议恢复的 pip/conda/模型大缓存。

## 5. HIL/DITL 与 ROS2 的边界

只读 AST 核对见快照 `dependency_imports`：Orin worker 使用 torch、numpy、项目内 `coptv2x_*`/`ditl_*` 模块；TRT backend 使用 torch+numpy+tensorrt；Python `uniad_cmdvel_bridge.py` 使用 ROS1 消息；`ditl_ros2_edge_bridge.py` 使用 `rclpy`、`std_msgs`、numpy。HIL源码要和 worker、协议、服务、配置、模型/引擎一起恢复；第三方 `msgpack/msgpack_numpy/zmq/yaml/PIL` 出现在整套运行目录的可选路径，不能据此声称系统解释器已安装它们。

原HIL `runtime/pydeps` 的发行包元数据另存 `hil_pydeps_snapshot.json`：P4/P5各有asttokens3.0.2、colorama0.4.6、executing2.2.1、icecream2.2.0、Pygments2.19.2；Pipeline的pydeps链接到P4。可在Python3.8及主运行依赖已经就绪后重建这部分小型target目录：

```bash
P=/home/jichengzhi/uniad_venv/bin/python
D=/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z/runtime/pydeps
"$P" -m pip install --target "$D" asttokens==3.0.2 colorama==0.4.6 \
 executing==2.2.1 icecream==2.2.0 Pygments==2.19.2
D5=/home/jichengzhi/coptv2x_hil_p5_preparation_20261004/runtime/pydeps
mkdir -p "$D5"
rsync -a "$D/" "$D5/"
# Pipeline目录恢复后、目标路径尚不存在时执行：
ln -s "$D" /home/jichengzhi/coptv2x_hil_pipeline_20261010/runtime/pydeps
```

这仅恢复可识别的五个发行包，仍须保留本项目worker/协议等源码与正确解释器，不表示整套HIL依赖闭包或新机功能已验收。

ROS2 trace 脚本 source `/opt/ros/foxy` 与 `/opt/ros2-tracing-overlay`，并使用 `/workspace`，这些是容器路径。宿主已安装 **Noetic**，不能在宿主执行这些 source 后认为装好了 Foxy。Docker26.1.3的精确包版本、安装目录、加载命令及GPU runtime边界见 [DOCKER_RESTORE.md](DOCKER_RESTORE.md)。原容器镜像：

| 镜像 | 本地ID | 显示大小 |
|---|---|---|
| `ditl/ros-foxy:arm64-exact` | cc26bc4823a8 | 662 MB |
| `ditl/ros-foxy-tracing:arm64` | 311954d34644 | 669 MB |
| `ditl/ros-foxy-tracing-enabled:arm64` | 9ad55751d6d3 | 674 MB |

优先 `docker load -i` 恢复单独保存的最终镜像（仅当资产清单确实保存了它）。重建时需要基础镜像 `ditl/ros-foxy:arm64-exact`，原两个 Dockerfile、`ros2_tracing_foxy_20260922_v1.tar.gz`；后两者已经不足以从空白构建自定义基础镜像。若这些基础依赖缺失，则 ROS2 tracing 是明确恢复缺口；普通 ROS1/UDP/C++ 链可先恢复。

```bash
cd /home/jichengzhi/ditl_rsu_edge_v1
# 仅在基础镜像已恢复时构建：
docker build -f Dockerfile.ros2trace -t ditl/ros-foxy-tracing:arm64 .
docker build -f Dockerfile.ros2trace-enabled -t ditl/ros-foxy-tracing-enabled:arm64 .
```

### 功耗与时钟状态

`power_mode_snapshot.json` 保存当前只读查询：nvpmodel可读为 **MAXN / mode 0**，但EMC字段返回权限错误；普通用户执行 `jetson_clocks --show` 返回需root，故没有完整锁频快照。当前模式也不自动等于所有历史实验设置。新系统复现实验前，用 `sudo nvpmodel -q`、`sudo jetson_clocks --show` 只读查询并记入新结果，结合旧报告中的功耗轨/时钟/样本范围核对。本恢复清单未自动修改功耗模式或开启锁频。

## 6. 重装后的验收与清单含义

本次只验证快照文件能解析、版本可追溯；**没有在新系统安装或跑任何推理/机器人测试**。重装后先做导入/动态库检查，再按项目的离线小输入验收，最后才连真实底盘；不要把历史结果表当作恢复后新测结果。

```bash
/usr/local/cuda/bin/nvcc --version
/usr/src/tensorrt/bin/trtexec --version
source /opt/ros/noetic/setup.bash
/usr/bin/python3 -B -c 'import rospy; from geometry_msgs.msg import Twist; import tensorrt; print(tensorrt.__version__)'
/home/jichengzhi/uniad_venv/bin/python -B -c \
 'import torch,torchvision,mmcv; print(torch.__version__,torchvision.__version__,mmcv.__version__); print(torch.cuda.is_available())'
# 导出环境再单独检查mmdet3d和mmcv ops；不得仅凭pip metadata判定编译成功。
```

- `environment_snapshot.json`：机器、dpkg、五套解释器元数据、ORT目标包、conda包build、source revision与AST依赖、Docker镜像；只存允许字段。
- `dpkg_versions.tsv`、`python_*_versions.tsv`：完整采集版本审计表。继承系统包时允许同名多版本；以实际 `sys.path` 与导入位置验收。快照含少量 Name=null 的失效元数据，不能拿来当安装包名。
- `requirements_*_runtime.txt`：运行包子集 pin；`conda_uniad_specs.txt`：无URL的已装conda构建pin。它们不是已实测重装锁文件。
- `collect_environment_readonly.py`：远端只读的SSH采集器，使用现有agent认证，不含密码、私钥、token、原始环境变量全集。它会在脚本所在本地目录重写快照；复验时先复制脚本到新的日期目录运行，保留本历史清单不变。

已知缺口：原 BSP 刷机介质/启动分区未克隆；Noetic EOL仓库和旧conda构建的长期可取性未知；完整Python依赖闭包未离线打包；原UniAD仓库无可用Git HEAD；系统Python用户级torchvision需要定向安装资产或源码恢复；TRT插件与引擎需匹配版本复验；ROS2基础镜像应单独备份。任何新 JetPack、Torch、MMCV、ORT 版本都应在新目录验证后记录，不能覆盖本快照的历史版本。
