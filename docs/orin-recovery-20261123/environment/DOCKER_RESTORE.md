# Docker / Foxy 容器恢复

原宿主为Ubuntu20.04 arm64，`/usr/bin/docker` 来自 `docker.io=26.1.3-0ubuntu1~20.04.1`；containerd为 `1.7.24-0ubuntu1~20.04.2`，runc为 `1.1.12-0ubuntu2~20.04.1`。存储目录 `/var/lib/docker`，driver为overlay2，cgroupfs/v1，默认runtime=runc。完整允许字段见 `docker_runtime_snapshot.json`。

三个ROS2 Foxy镜像已合在 `offline_assets/ros2-foxy-images.tar.gz`，254,241,601 B，共享层只存一次。镜像标签/ID、压缩包SHA256、配置与layer摘要见 `manifests/docker_images_manifest.json`；镜像内容作为私有离线备份，不放普通Git中。

## 新系统安装目录和命令

以下在完成匹配JetPack的**新Orin**执行，会安装服务；本次没有在原机执行。旧Ubuntu仓库若已移除这些构建号，先记录缺口并取得匹配arm64包，不把新候选版本当作原版本。

```bash
sudo apt-get update
apt-cache policy docker.io containerd runc
# 仅精确版本仍可见时安装：
sudo apt-get install docker.io=26.1.3-0ubuntu1~20.04.1 \
 containerd=1.7.24-0ubuntu1~20.04.2 runc=1.1.12-0ubuntu2~20.04.1
sudo systemctl enable --now docker
sudo usermod -aG docker jichengzhi
# 完全注销SSH/终端并重新登录，再执行后续命令。
docker version
docker info --format '{{.ServerVersion}} {{.Driver}} {{.CgroupDriver}} {{.CgroupVersion}} {{.DefaultRuntime}}'
```

Docker CLI在 `/usr/bin/docker`，daemon数据目录默认 `/var/lib/docker`，不复制原整个daemon目录。若新系统已有其他容器业务，先核对现有服务/存储配置，不覆盖其配置文件。本次保存的是镜像，不含旧容器可写层、volume或宿主输入数据；本项目代码和配置由payload恢复，输入缓存有意省略。

## 校验后加载镜像

```bash
export RECOVERY_REPO=/home/jichengzhi/orin-recovery/github
export OFFLINE=/home/jichengzhi/orin-recovery/offline_assets
python3 -B "$RECOVERY_REPO/tools/verify_recovery.py" --repo "$RECOVERY_REPO" --offline "$OFFLINE"
set -o pipefail
gzip -dc "$OFFLINE/ros2-foxy-images.tar.gz" | docker load
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
```

期望标签：

- `ditl/ros-foxy:arm64-exact` → `cc26bc4823a8`
- `ditl/ros-foxy-tracing:arm64` → `311954d34644`
- `ditl/ros-foxy-tracing-enabled:arm64` → `9ad55751d6d3`

`/opt/ros/foxy`、`/opt/ros2-tracing-overlay`、`/workspace` 是容器内路径。原Dockerfile和tracing源码包可用于重建增量层，基础镜像仍需上述离线包；加载后再按原脚本核对挂载/网络/trace权限。本清单不自动运行HIL或底盘节点。

原系统装有 `nvidia-container-runtime=3.9.0-1`、`nvidia-container-toolkit=1.11.0~rc.1-1`、`nvidia-docker2=2.11.0-1`，但本次 `docker info` 仅发现runc相关runtime，未发现已注册nvidia runtime。因此不能把这些包存在视为GPU容器可用；Foxy桥接/tracing与宿主TRT worker分开恢复。若以后需要GPU容器，按匹配JetPack的 [NVIDIA容器安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 单独配置和验收，不能以ROS2镜像成功load代替GPU检查。
