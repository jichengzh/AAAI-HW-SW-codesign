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

原系统装有 `nvidia-container-runtime=3.9.0-1`、`nvidia-container-toolkit=1.11.0~rc.1-1`、`nvidia-docker2=2.11.0-1`；快照 `docker_runtime_snapshot.json` 的 `Runtimes` 明确包含 `nvidia`（path=`nvidia-container-runtime`），`DefaultRuntime` 为 `runc`。此前“未发现已注册nvidia runtime”的描述有误。注册runtime和安装包存在都不能证明GPU容器实际可运行；Foxy桥接/tracing与宿主TRT worker分开恢复，宿主TRT不需要GPU容器。若以后需要GPU容器，按匹配JetPack的 [NVIDIA容器安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 单独配置和验收，不能以ROS2镜像成功load代替GPU检查。


## 仅在容器名不存在时重建空闲骨架

源码 `coptv2x_hil_episode_services.py` 使用 `docker exec`，不会自动创建默认容器。新系统加载镜像后，可按原inspect恢复下面的骨架：选择的runtime挂到 `/work`，独立状态目录挂到 `/state`，host网络、runc。命令只在容器名不存在时创建；若已有同名容器，检查其挂载、状态和配置，不自动删除或覆盖。

```bash
export R=/home/jichengzhi/coptv2x_hil_pipeline_20261010/runtime
export OLD=/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z
mkdir -p "$OLD/runtimestate"
if docker container inspect coptv2x-hil-p4-smoke-ros >/dev/null 2>&1; then
  docker container inspect coptv2x-hil-p4-smoke-ros
else
  docker run -d --name coptv2x-hil-p4-smoke-ros \
    --runtime runc --network host \
    --mount "type=bind,src=$R,dst=/work" \
    --mount "type=bind,src=$OLD/runtimestate,dst=/state" \
    ditl/ros-foxy-tracing-enabled:arm64 bash -lc 'sleep infinity'
fi
docker inspect --format '{{.State.Status}} {{.HostConfig.Runtime}} {{.HostConfig.NetworkMode}} {{json .Mounts}}' \
  coptv2x-hil-p4-smoke-ros
```

已有容器若为exited，先核对上述配置，再手动 `docker start coptv2x-hil-p4-smoke-ros`。该命令仅恢复空闲容器，不启动底盘或闭环。实际episode服务在每次 `docker exec` 时 `source /opt/ros/foxy/setup.bash`，并传入 `ROS_DOMAIN_ID=71` 和 `/work` 的PYTHONPATH；只设置容器名字不足以启动worker/publisher/proxy或完成最终ACK。

## 后续单帧ROS往返检查

先完成 [Copt宿主单帧真实推理](../COPT_ENGINE_RESTORE.md)，沿用其中 `PROOF` 和 `ARM`。wire位于宿主独立runtimestate，映射为 `/state/<新目录名>/<arm>.wire`。以下命令实际发布并接收容器内ROS话题，需已有运行中的正确骨架；应单独记录结果，不将它当作新系统安装、外部HIL或底盘闭环已验收：

```bash
export PROOF ARM
PROOF_NAME=$(basename "$PROOF")
docker exec -e ROS_DOMAIN_ID=71 \
  -e PYTHONPATH=/work:/work/simulation/leaderboard/team_code/closedloop \
  -e WIRE="/state/$PROOF_NAME/$ARM.wire" \
  coptv2x-hil-p4-smoke-ros bash -lc \
  'source /opt/ros/foxy/setup.bash && exec python3 /work/scripts/coptv2x_hil_serial_ros_smoke.py --wire "$WIRE" --config-dir /work/simulation/leaderboard/team_code/agent_config/coptv2x_hil'
```

此为后续可执行的验收步骤；是否已经运行及通过以 [VALIDATION](../VALIDATION.md) 的实际回执为准。完整HIL还需恢复宿主worker、容器publisher/proxy、反向SSH与外部最终ACK，以及原H800/CARLA输入。加载镜像或单个ROS话题成功均不能推出完整链路已恢复。
