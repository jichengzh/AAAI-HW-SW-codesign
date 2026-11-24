# CoptV2X 服务路径、启动与最终ACK验收

本说明衔接pipeline源码、P4输入/native/state及Foxy容器。先完成 [冻结资产恢复](COPT_ENGINE_RESTORE.md) 和 [Docker环境](environment/DOCKER_RESTORE.md)。依赖可在联网的新Orin安装，不要求整个Python/系统依赖闭包离线保存；JetPack/TensorRT及模型身份仍须匹配并实际验收。这里只覆盖边缘服务与单帧传输，不启动CARLA、车辆或底盘；实际测试记录见 [VALIDATION](VALIDATION.md)。

## 路径和独立试验范围

在恢复后的Orin操作。`ORIN_REMOTE`设为本机可用的SSH alias或`user@host`，用于既有episode启动器的SSH/rsync；示例的消费端也在本机，通过真实SSH反向转发连接。远端CARLA接入时，消费端应移至该主机并单独验证网络及其依赖。

```bash
set -euo pipefail
: "${ORIN_REMOTE:?先设置可用的Orin SSH目标，如SSH config中的alias}"
export R=/home/jichengzhi/coptv2x_hil_pipeline_20261010/runtime
export P4=/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z
export STAGE6=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
export ORIN_PYTHON=/home/jichengzhi/miniforge3/envs/uniad/bin/python
export CONTAINER=coptv2x-recovery-20261124
export HOST_STATE_ROOT="$P4/runtimestate/p4-recovery"
export CONTAINER_STATE_ROOT=/state/p4-recovery
export LOCAL_PORT=62610 FORWARD_PORT=62611 WORKER_PORT=62612 RESULT_PORT=62613
mkdir -p "$P4/runtimestate"
export PROOF
PROOF=$(mktemp -d "$P4/runtimestate/recovery-service.XXXXXX")
export REMOTE_ROOT="$PROOF/remote-control"
export COPT_HIL_RUN_DIR="$PROOF/runner"
export COPT_HIL_TASK_ID=recovery-oneframe
export COPT_HIL_ARM_ID=compression_only
export COPT_HIL_COHORT_ID=recovery-20261124
export COPT_HIL_ATTEMPT=1
export RESULT_COUNT=1
mkdir -p "$COPT_HIL_RUN_DIR"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$ORIN_REMOTE" true
"$ORIN_PYTHON" -B -c 'import numpy,yaml,tensorrt; print(tensorrt.__version__)'
```

`ORIN_PYTHON`须换为实际通过依赖检查的宿主环境；原生`original_default`另需Torch和native项目依赖。SSH用户还须能直接运行Docker。每次试验新建PROOF，不复用旧context/token；不要指向原部署的控制目录。

| 内容 | 宿主路径 | 容器路径 |
|---|---|---|
| pipeline运行源码 | `$R` | `/work` |
| 单帧与原生源码 | `$P4/input`、`$P4/native/V2Xverse` | worker在宿主读取，无需挂载 |
| 试验/owner/ready文件 | `$P4/runtimestate` | `/state` |
| 每次episode的服务状态 | `$HOST_STATE_ROOT/<token>` | `/state/p4-recovery/<token>` |
| 冻结模型 | `$STAGE6` | worker在宿主读取 |

`pipeline/runtime`的父目录没有P4的input/native/state。不能省略下面的显式路径，也不能用镜像内的空目录代替资产。

## 单帧manifest与trigger-plan

下面生成1样本manifest及恰好1个trigger。样本、route、scenario和batch_order保留原身份，输入改为已备份preflight；输入SHA必须相同。`result-count`、worker的max-samples和publisher的max-results均为1。扩大试验时，必须同时提供相应数量的样本和递增sequence_no的trigger；不能保留旧2225默认值来跑单帧。

```bash
"$ORIN_PYTHON" -B - <<'PYCODE'
import hashlib
import json
import os
from pathlib import Path
import sys
r, p4, proof = [Path(os.environ[k]) for k in ("R", "P4", "PROOF")]
sys.path.insert(0, str(r / "simulation/leaderboard/team_code/closedloop"))
from coptv2x_hil_trigger import parse_hil_trigger
from coptv2x_hil_artifacts import preflight_hil_artifacts
for path in (r / "scripts/coptv2x_hil_episode_services.py", p4 / "input/preflight.npy", p4 / "native/V2Xverse"):
    assert path.exists(), str(path)
records = preflight_hil_artifacts(r / "simulation/leaderboard/team_code/agent_config/coptv2x_hil", os.environ["STAGE6"])
assert len(records) == 5 and sum(len(x["artifact_sha256"]) for x in records) == 14
original = json.loads((r.parent / "evidence/worker_smoke_manifest_20.json").read_text())
first = original["samples"][0]
spec = first["inputs"]["spatial_features"]
assert hashlib.sha256((p4 / "input/preflight.npy").read_bytes()).hexdigest() == spec["sha256"]
new_first = dict(first, inputs=dict(first["inputs"], spatial_features=dict(spec, path="preflight.npy")))
one = dict(original, samples=[new_first])
(proof / "one-frame.json").write_text(json.dumps(one, indent=2) + "\n")
trigger = dict(schema_version="coptv2x.hil-trigger.v1", run_id="recovery-" + proof.name,
    route_id=one["route_id"], scenario_id=one["scenario_id"], sample_id=first["sample_id"],
    sequence_no=0, source_frame=first["source_frame"], source_sim_time_ns=0,
    arm_id=os.environ["COPT_HIL_ARM_ID"])
plan = [parse_hil_trigger(trigger).to_metadata()]
assert len(one["samples"]) == len(plan) == int(os.environ["RESULT_COUNT"]) == 1
(proof / "trigger-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
(proof / "frozen-preflight.json").write_text(json.dumps(records, indent=2) + "\n")
PYCODE
```

`source_sim_time_ns=0`是离线健康检查的显式时间，不冒充CARLA测量。真实HIL须由真实source_frame/仿真时间驱动trigger。

## 新容器与源码生成的布局检查

先确认62610–62613未被使用；下列只绑定检查，不保留监听端口。域71及默认话题由现有episode源码固定使用，运行前须确认没有其他活跃部署占用这些话题。旧命名容器即使exited也不影响本例；不要启动、删除或重命名它。

```bash
"$ORIN_PYTHON" -B - <<'PYCODE'
import os
import socket
for key in ("LOCAL_PORT", "FORWARD_PORT", "WORKER_PORT", "RESULT_PORT"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", int(os.environ[key])))
PYCODE
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "同名容器已存在：先检查并选择一个新的CONTAINER名称" >&2
  exit 1
fi
docker run -d --name "$CONTAINER" --runtime runc --network host \
  --mount "type=bind,src=$R,dst=/work" \
  --mount "type=bind,src=$P4/runtimestate,dst=/state" \
  ditl/ros-foxy-tracing-enabled:arm64 bash -lc 'sleep infinity'
docker exec -e ROS_DOMAIN_ID=71 "$CONTAINER" bash -lc \
  'source /opt/ros/foxy/setup.bash && ros2 topic list'
```

如果已有活跃`/coptv2x/hil/trigger`或`/ditl/rsu/perception_result`、`perception_trace`、`perception_result_ack`端点，停止本次试验并先安排独立环境；不同TCP端口不能隔离相同ROS域和话题。Foxy内部话题须保持源码配套默认值；不能只给一个节点换域或话题。

以下只调用纯函数，不启动服务。核对容器实际mount及生成的worker参数、publisher ready路径与owner路径，将预览保存以便审查：

```bash
"$ORIN_PYTHON" -B - <<'PYCODE'
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
r, p4, proof = [Path(os.environ[k]) for k in ("R", "P4", "PROOF")]
spec = importlib.util.spec_from_file_location("episode_services", str(r / "scripts/coptv2x_hil_episode_services.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args = argparse.Namespace(runtime=str(r), data_root=str(p4 / "input"), input_manifest=str(proof / "one-frame.json"),
    native_repo_root=str(p4 / "native/V2Xverse"), artifact_root=os.environ["STAGE6"],
    orin_python=os.environ["ORIN_PYTHON"], container=os.environ["CONTAINER"], container_runtime="/work",
    host_state_root=os.environ["HOST_STATE_ROOT"], container_state_root=os.environ["CONTAINER_STATE_ROOT"],
    forward_port=int(os.environ["FORWARD_PORT"]), worker_port=int(os.environ["WORKER_PORT"]),
    result_port=int(os.environ["RESULT_PORT"]), result_count=int(os.environ["RESULT_COUNT"]), ready_timeout_s=180,
    serial_optimized=True, pipeline_depth=0, runtime_warmup_count=0, wire_compression="none", trigger_policy="fifo",
    completion_drain_timeout_s=135)
context = module.episode_context(os.environ)
config = module.service_configuration(args, context)
assert config["completion_policy"] == dict(mode="publisher_ack", worker="worker", publisher="publisher", drain_timeout_s=135)
mounts = json.loads(subprocess.check_output(["docker", "inspect", os.environ["CONTAINER"]]))[0]["Mounts"]
actual = {x["Destination"]: Path(x["Source"]).resolve() for x in mounts}
assert actual["/work"] == r.resolve() and actual["/state"] == (p4 / "runtimestate").resolve()
state = Path(config["state_dir"])
assert state == Path(os.environ["HOST_STATE_ROOT"]) / context["token"]
assert not state.exists(), "必须使用全新context/token"
worker = next(x for x in config["services"] if x["name"] == "worker")
for option, expected in (("--data-root", str(p4 / "input")), ("--manifest", str(proof / "one-frame.json")),
        ("--native-repo-root", str(p4 / "native/V2Xverse")), ("--artifact-root", os.environ["STAGE6"]),
        ("--ready-file", str(state / "worker.ready")), ("--result-count", "1")):
    assert worker["argv"][worker["argv"].index(option) + 1] == expected
publisher = next(x for x in config["services"] if x["name"] == "publisher")
assert publisher["ready_file"] == str(state / "publisher.ready")
for service in config["services"]:
    if service["container"]:
        assert service["container_state_dir"] == "/state/p4-recovery/" + context["token"]
        assert "ROS_DOMAIN_ID=71" in service["argv"]
(proof / "services.preview.json").write_text(json.dumps(config, indent=2) + "\n")
(proof / "token").write_text(context["token"] + "\n")
print(json.dumps({"layout": "PASS", "state_dir": str(state), "token": context["token"]}))
PYCODE
export TOKEN
TOKEN=$(cat "$PROOF/token")
export STATE="$HOST_STATE_ROOT/$TOKEN"
```

## 启动：消费端、反向转发、episode服务

先准备消费端。它只在本机loopback监听，使用仓库已有transport解码真实wire；handler通过trigger、冻结模型、shape/dtype/finite及重新计算的输出SHA检查后，生产transport才发送`ACK <request_id>`。收到空结果或任一检查失败均不能ACK。下面将脚本写入新PROOF，不修改payload：

```bash
cat > "$PROOF/consume_one.py" <<'PYCODE'
import json
import os
from pathlib import Path
import sys
import threading
import time
r, proof = Path(os.environ["R"]), Path(os.environ["PROOF"])
sys.path.insert(0, str(r / "simulation/leaderboard/team_code/closedloop"))
from coptv2x_hil_proxy_transport import ReverseSshRosProxyTransport
from coptv2x_hil_backend import validate_feature_payload_contract
from ditl_edge_protocol import canonical_feature_payload_sha256
trigger = json.loads((proof / "trigger-plan.json").read_text())[0]
complete = threading.Event()
receipt = {}
def accept(envelope):
    if envelope is None:
        complete.set()
        return
    result = envelope.feature_result
    assert dict(result.trigger_identity) == trigger
    assert tuple(envelope.batch_order) == ("rsu",)
    validate_feature_payload_contract(dict(result.artifact_identity), result.outputs,
        r / "simulation/leaderboard/team_code/agent_config/coptv2x_hil", result.payload_semantics)
    assert canonical_feature_payload_sha256(result.outputs) == result.payload_sha256
    receipt.update(request_id=trigger["run_id"] + ":0", trigger=trigger,
        payload_sha256=result.payload_sha256, contract="PASS", consumer="PASS")
    complete.set()
transport = ReverseSshRosProxyTransport(listen_port=int(os.environ["LOCAL_PORT"]),
    connect_timeout_s=180, socket_timeout_s=180, readonly_results=True)
try:
    transport.start(accept)
    (proof / "consumer.ready").write_text("ready\n")
    while not (proof / "send-trigger").exists():
        time.sleep(0.1)
    transport.publish_trigger(trigger)
    assert complete.wait(180) and receipt, "未收到通过校验的真实结果"
    # callback返回后transport才发送ACK；保留连接，直到后续ROS trace回执证明ACK被接收。
    deadline = time.monotonic() + 180
    while not (proof / "trace-after-ack.json").exists():
        if transport.last_error is not None or time.monotonic() >= deadline:
            raise RuntimeError("未取得ACK后的ROS trace")
        time.sleep(0.1)
    (proof / "consumer-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
finally:
    transport.close()
PYCODE
cleanup_recovery() {
  trap - EXIT
  if declare -p SERVICE_ARGS >/dev/null 2>&1; then
    "$ORIN_PYTHON" -B "$R/scripts/coptv2x_hil_episode_services.py" \
      --mode stop "${SERVICE_ARGS[@]}" || true
  fi
  ssh -S "$PROOF/ssh-control" -O exit "$ORIN_REMOTE" 2>/dev/null || true
  for owned_pid in "${CONSUMER_PID:-}" "${TRACE_PID:-}"; do
    if [ -n "$owned_pid" ]; then kill "$owned_pid" 2>/dev/null || true; fi
  done
  docker stop "$CONTAINER" >/dev/null || true
}
trap cleanup_recovery EXIT
"$ORIN_PYTHON" -B "$PROOF/consume_one.py" > "$PROOF/consumer.log" 2>&1 &
CONSUMER_PID=$!
for check in $(seq 1 300); do
  [ -f "$PROOF/consumer.ready" ] && break
  kill -0 "$CONSUMER_PID"
  sleep 0.1
done
test -f "$PROOF/consumer.ready"
ssh -M -S "$PROOF/ssh-control" -fNT \
  -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -R "127.0.0.1:$FORWARD_PORT:127.0.0.1:$LOCAL_PORT" "$ORIN_REMOTE" \
  > "$PROOF/tunnel.log" 2>&1
ssh -S "$PROOF/ssh-control" -O check "$ORIN_REMOTE"
```

下面的数组是同一次start/stop的完整参数，不依赖历史默认目录、容器、端口、帧数或Python路径。`--remote-root`是Orin上新控制目录，写入`episodes/<token>/services.json`；`COPT_HIL_RUN_DIR`是启动器所在机的新证据目录，启动/停止都会rsync状态到它的`evidence/orin/`。

```bash
SERVICE_ARGS=(
  --remote "$ORIN_REMOTE" --remote-root "$REMOTE_ROOT"
  --runtime "$R" --container "$CONTAINER" --container-runtime /work
  --orin-python "$ORIN_PYTHON" --artifact-root "$STAGE6"
  --data-root "$P4/input" --input-manifest "$PROOF/one-frame.json"
  --native-repo-root "$P4/native/V2Xverse"
  --host-state-root "$HOST_STATE_ROOT" --container-state-root "$CONTAINER_STATE_ROOT"
  --forward-port "$FORWARD_PORT" --worker-port "$WORKER_PORT" --result-port "$RESULT_PORT"
  --result-count "$RESULT_COUNT" --ready-timeout-s 180 --completion-drain-timeout-s 135
  --serial-optimized --pipeline-depth 0 --runtime-warmup-count 0
  --wire-compression none --trigger-policy fifo
)
"$ORIN_PYTHON" -B "$R/scripts/coptv2x_hil_episode_services.py" \
  --mode start "${SERVICE_ARGS[@]}"
"$ORIN_PYTHON" -B - <<'PYCODE'
import json
import os
from pathlib import Path
import time
state = Path(os.environ["STATE"])
proxy_deadline = time.monotonic() + 180
while True:
    assert not (state / "failure.json").exists(), "服务启动失败，检查failure.json"
    log = state / "proxy.log"
    if log.exists() and '"event": "coptv2x_hil_ros_proxy_ready"' in log.read_text():
        break
    assert time.monotonic() < proxy_deadline, "proxy未在期限内就绪"
    time.sleep(0.1)
ready = json.loads((state / "ready.json").read_text())
assert ready["ready"] is True and set(ready["services"]) == {"worker", "publisher", "proxy"}
assert json.loads((state / "worker.ready").read_text())["expected_results"] == 1
assert json.loads((state / "publisher.ready").read_text())["result_port"] == int(os.environ["RESULT_PORT"])
assert '"event": "coptv2x_hil_ros_proxy_ready"' in (state / "proxy.log").read_text()
print("SERVICE_READY_PASS")
PYCODE
```

`ready.json`只汇总worker/publisher的ready文件并监测子进程存活；proxy没有ready_file，所以另查proxy日志。容器端每条服务命令都会source Foxy，固定`ROS_DOMAIN_ID=71`。ready证明启动，不证明完成推理或ACK。

## 发送前启动ACK后的trace观察，再触发单帧

观察器订阅原`/ditl/rsu/perception_trace`。源码publisher仅在收到匹配ROS ACK后发布该trace；这是“host ACK已经回到ROS publisher”的可复查证据。观察器只接受本次request_id，180秒无匹配消息退出失败。

```bash
cat > "$PROOF/capture_trace.py" <<'PYCODE'
import json
import os
import time
import rclpy
from std_msgs.msg import String
rclpy.init()
node = rclpy.create_node("copt_recovery_trace")
received = []
def receive(message):
    value = json.loads(message.data)
    if value.get("request_id") == os.environ["EXPECTED_REQUEST"]:
        received.append(value)
subscription = node.create_subscription(String, "/ditl/rsu/perception_trace", receive, 10)
from pathlib import Path
discovery_deadline = time.monotonic() + 30
while node.count_publishers("/ditl/rsu/perception_trace") == 0:
    assert time.monotonic() < discovery_deadline, "trace publisher发现超时"
    rclpy.spin_once(node, timeout_sec=0.1)
Path(os.environ["OBSERVER_READY"]).write_text("ready\n")
deadline = time.monotonic() + 180
try:
    while not received and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    assert received, "未收到匹配request_id的ACK后trace"
    print(json.dumps(dict(received[0], ack_observed=True)), flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
PYCODE
EXPECTED_REQUEST=$("$ORIN_PYTHON" -B -c \
  'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["PROOF"])/"trigger-plan.json").read_text())[0]["run_id"]+":0")')
docker exec -e ROS_DOMAIN_ID=71 -e EXPECTED_REQUEST="$EXPECTED_REQUEST" \
  -e OBSERVER_READY="/state/$(basename "$PROOF")/trace-observer.ready" \
  "$CONTAINER" bash -lc \
  "source /opt/ros/foxy/setup.bash && exec python3 /state/$(basename "$PROOF")/capture_trace.py" \
  > "$PROOF/trace-after-ack.json.tmp" 2> "$PROOF/trace.log" &
TRACE_PID=$!
# 确认订阅者已创建，再发出本次唯一trigger。
for check in $(seq 1 300); do
  [ -f "$PROOF/trace-observer.ready" ] && break
  kill -0 "$TRACE_PID"
  sleep 0.1
done
test -f "$PROOF/trace-observer.ready"
touch "$PROOF/send-trigger"
wait "$TRACE_PID"
mv "$PROOF/trace-after-ack.json.tmp" "$PROOF/trace-after-ack.json"
wait "$CONSUMER_PID"
"$ORIN_PYTHON" -B - <<'PYCODE'
import json
import os
from pathlib import Path
import time
proof = Path(os.environ["PROOF"])
state = Path(os.environ["STATE"])
deadline = time.monotonic() + 150
while not (state / "completed.json").exists():
    assert not (state / "failure.json").exists(), "服务未完成，检查failure和日志"
    assert time.monotonic() < deadline, "服务未在drain期限内完成"
    time.sleep(0.1)
completed = json.loads((state / "completed.json").read_text())
assert completed["completed"] is True and completed["mode"] == "publisher_ack"
assert set(completed["services"]) == {"worker", "publisher"}
assert not (state / "failure.json").exists()
receipt = json.loads((proof / "consumer-receipt.json").read_text())
trace = json.loads((proof / "trace-after-ack.json").read_text())
assert receipt["consumer"] == receipt["contract"] == "PASS"
assert trace["ack_observed"] and trace["request_id"] == receipt["request_id"]
print("FINAL_HOST_AND_ROS_ACK_PASS", receipt["request_id"], receipt["payload_sha256"])
PYCODE
```

各级判据分别是：14项冻结文件哈希通过；启动ready通过；consumer收到真实输出且contract/hash/trigger通过；生产transport发送host ACK；publisher收到ROS ACK后发布匹配trace。worker的`DONE`表示本地处理完成，pipeline模式的`ACCEPTED`只表示准入，两者均不能替代最终ACK。

本次发布副本修复了有限帧数生命周期：episode生成 `completion_policy=publisher_ack`，worker达到result-count并正常退出后，supervisor继续保留publisher/proxy，等待最后host ACK经ROS返回。worker与publisher都以0退出、proxy未提前退出并清理成功后才写 `completed.json`。`--completion-drain-timeout-s 135`从第一个有限服务正常退出开始计时，不限制完整episode总时长；非零退出、proxy提前退出或排空超时仍失败。SIGTERM手动停止不会制造completed标记。原始源码仍私有保留，旧的无policy配置保留原语义；验收必须使用本次公开副本，不能提高result-count却少发trigger来绕过。

## 仅停止本次owner、转发与新容器

保留同一context、SERVICE_ARGS和PROOF。失败时也执行以下清理，先停episode再停隧道和辅助consumer。EXIT trap也会尝试清理本次服务；隧道通过本次专属SSH control socket停止。辅助consumer和观察器只使用本shell保存的PID。不要使用pkill、docker prune或删除原命名容器。

```bash
"$ORIN_PYTHON" -B "$R/scripts/coptv2x_hil_episode_services.py" \
  --mode stop "${SERVICE_ARGS[@]}"
cleanup_recovery
"$ORIN_PYTHON" -B - <<'PYCODE'
import json
import os
from pathlib import Path
state = Path(os.environ["STATE"])
assert json.loads((state / "stopped.json").read_text())["cleaned"] is True
print("OWNED_SERVICES_STOPPED", str(state))
PYCODE
```

新容器保留为stopped供inspect，PROOF保留全部输入计划、服务参数与日志。重新验收使用新PROOF/context和新容器名。上述例子证明本机边缘协议链时，也不能据此声称外部H800/CARLA、真实感知输入流、新系统重装或车辆闭环已通过；这些须按各自范围另有回执。
