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


## 服务启动入口与挂载闭合

完整worker/publisher/proxy启动、SSH转发、context、ready及最终ACK命令见 [COPT_SERVICE_RESTORE](../COPT_SERVICE_RESTORE.md)。该流程使用新命名容器、独立attempt状态和62610–62613端口，显式指定全部业务路径；不启动、删除或覆盖旧命名容器。旧容器为exited不构成恢复障碍。

路径固定为：pipeline runtime挂 `/work`，P4的独立runtimestate挂 `/state`。宿主worker显式读取P4的input/native和Stage6模型，`--host-state-root P4/runtimestate/p4-recovery`对应 `--container-state-root /state/p4-recovery`。不能由pipeline runtime父目录推断这些路径。容器仅以runc/host网络保持 `bash -lc 'sleep infinity'`；实际服务由episode启动器执行docker exec，并source Foxy、设置ROS_DOMAIN_ID=71及PYTHONPATH。

## 仅用于离线wire的独立Foxy骨架

若只检查已有单帧wire，可使用下列新容器；完整服务流程会自行创建另一个新容器，无须先做本节。遇到同名容器时先inspect并另选新名，不自动删除或覆盖：

```bash
export R=/home/jichengzhi/coptv2x_hil_pipeline_20261010/runtime
export OLD=/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z
export CONTAINER=coptv2x-recovery-20261124-wire
mkdir -p "$OLD/runtimestate"
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "容器名称已存在：请inspect后另选新名" >&2
  exit 1
fi
docker run -d --name "$CONTAINER" --runtime runc --network host \
  --mount "type=bind,src=$R,dst=/work" \
  --mount "type=bind,src=$OLD/runtimestate,dst=/state" \
  ditl/ros-foxy-tracing-enabled:arm64 bash -lc 'sleep infinity'
docker inspect --format '{{.State.Status}} {{.HostConfig.Runtime}} {{.HostConfig.NetworkMode}} {{json .Mounts}}' \
  "$CONTAINER"
```

## 单帧wire的ROS往返检查

先完成 [Copt宿主单帧真实推理](../COPT_ENGINE_RESTORE.md)，沿用其中PROOF和ARM。wire映射为 `/state/<新目录名>/<arm>.wire`。以下实际发布并接收ROS话题；使用独立的smoke话题，不启动完整worker/publisher/proxy、底盘或闭环：

```bash
export PROOF ARM
PROOF_NAME=$(basename "$PROOF")
docker exec -e ROS_DOMAIN_ID=71 \
  -e PYTHONPATH=/work:/work/simulation/leaderboard/team_code/closedloop \
  -e WIRE="/state/$PROOF_NAME/$ARM.wire" \
  "$CONTAINER" bash -lc \
  'source /opt/ros/foxy/setup.bash && exec python3 /work/scripts/coptv2x_hil_serial_ros_smoke.py --wire "$WIRE" --config-dir /work/simulation/leaderboard/team_code/agent_config/coptv2x_hil --topic /coptv2x/recovery_20261124/wire_smoke'
docker stop "$CONTAINER"
```

wire往返和完整服务最终ACK是不同范围，实际测试设置与回执见 [VALIDATION](../VALIDATION.md)。完整部署使用源码配套的域71和trigger/result/trace/ack默认话题；隔离审计若给全套节点显式换域及话题，回执须披露该差异，不能只改一个节点。H800/CARLA外部最终消费、新系统安装和车辆闭环仍另行验收。
