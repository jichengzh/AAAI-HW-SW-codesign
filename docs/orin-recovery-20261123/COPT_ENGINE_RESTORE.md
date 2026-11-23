# CoptV2X 冻结 engine 恢复与最小验收

本次补入4份冻结 TensorRT engine，共 **56,499,229 B（53.88 MiB）**，最大文件48,406,458 B，直接保存于普通 Git 的 `engine_payload/`；不使用 Git LFS 或 Release。逐文件原路径、发布路径、字节数、SHA256和下载URL，以及4个checkpoint路径映射见 [copt_engine_backup_manifest.json](manifests/copt_engine_backup_manifest.json)。这修补了原恢复包遗漏engine及checkpoint路径不匹配的问题；实际执行记录与尚未验证范围见 [VALIDATION](VALIDATION.md)。

## 资产范围与不可替代性

下列路径均相对于 `lane_c_orin_stage6_five_config_20260724/`：

| engine路径 | 字节数 |
|---|---:|
| `codriving/compression_only/backbone.engine` | 2,694,290 |
| `codriving/schedule_only/backbone.engine` | 48,406,458 |
| `codriving/compress_then_tune/backbone.engine` | 4,220,604 |
| `codriving/joint_shcosearch_fp16/backbone.engine` | 1,177,877 |

冻结backend核对engine原字节SHA256。从ONNX重建的同名engine不能直接替代这些文件；若软件栈变化需要重建，应建立新的部署身份并重新验收。原1.60GiB离线资产（权重、ONNX、输入、安装包与镜像）仍须另外取得。本次没有发布含UniAD 121帧/相机SDK的156MiB私有审计补包。

## 复制与checkpoint路径修复

先按 [RESTORE_CHECKLIST](RESTORE_CHECKLIST.md) 恢复 `payload/` 并解开原离线资产，再执行以下命令；只写入文件，不启动推理或车辆：

```bash
export RECOVERY_REPO=/home/jichengzhi/orin-recovery/github
export STAGE6=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
rsync -a "$RECOVERY_REPO/engine_payload/" /home/jichengzhi/
for pair in \
  016x032x064/net_epoch_bestval_at9.pth \
  032x064x096/net_epoch_bestval_at11.pth \
  032x080x128/net_epoch_bestval_at5.pth \
  064x128x256/net_epoch29.pth; do
  mkdir -p "$STAGE6/source/codriving/${pair%/*}"
  cp -n "$STAGE6/models/codriving/$pair" "$STAGE6/source/codriving/$pair"
done
python3 -B "$RECOVERY_REPO/tools/verify_recovery.py" --repo "$RECOVERY_REPO" --home /home/jichengzhi
```

`cp -n`保留已有目标；若哈希不符，应先检查和另存已有文件再修复。默认仓库校验包含4个公开engine；`--home`额外核对恢复位置的4个engine及4个checkpoint别名。源码恢复使用manifest的**公开副本哈希**；9份隐私处理文件的原摘要已从公开元数据移除，不要求它们存在。

## 调用源码自己的只读冻结文件检查

显式设置 `artifact_root` 为Stage6目录。registry历史默认值指向 `V2X/results/...` 的另一棵树，不能直接照用。下面只读取5个arm的config/checkpoint/engine，校验共14项哈希，不加载TensorRT或运行模型：

```bash
export R=/home/jichengzhi/coptv2x_hil_pipeline_20261010/runtime
export OLD=/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z
export STAGE6=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
export ORIN_PYTHON=/usr/bin/python3
"$ORIN_PYTHON" -B - <<'PYCODE'
import json
import os
from pathlib import Path
import sys
runtime = Path(os.environ["R"])
sys.path.insert(0, str(runtime / "simulation/leaderboard/team_code/closedloop"))
from coptv2x_hil_artifacts import preflight_hil_artifacts
records = preflight_hil_artifacts(
    config_dir=runtime / "simulation/leaderboard/team_code/agent_config/coptv2x_hil",
    artifact_root=os.environ["STAGE6"],
)
assert len(records) == 5
assert sum(len(record["artifact_sha256"]) for record in records) == 14
print(json.dumps(records, indent=2))
PYCODE
```

此步骤需要PyYAML；后续推理还需匹配JetPack的宿主TensorRT、NumPy，以及原生arm使用的Torch与项目依赖。`ORIN_PYTHON`应指向实际完成依赖检查的宿主Python环境，不据历史版本表推定已安装。

## 用已备份单帧准备新的真实推理验收

不必恢复139GiB缓存即可检查最小输入链。离线包中的 `OLD/input/preflight.npy` 为float32、shape `[2,64,256,512]`，其SHA256是 `d685d156c22e59506a436ed8baccdee5ca093739f1ee94c7da01a89fcd802103`，等于pipeline `evidence/worker_smoke_manifest_20.json` 的第一份样本。以下生成新的单帧manifest，保留原schema、route、scenario、batch_order与样本身份，只截取首样本并改输入路径；输出放到新目录，原结果不变。

```bash
mkdir -p "$OLD/runtimestate"
export PROOF
PROOF=$(mktemp -d "$OLD/runtimestate/recovery-oneframe.XXXXXX")
"$ORIN_PYTHON" -B - <<'PYCODE'
import hashlib
import json
import os
from pathlib import Path
old = Path(os.environ["OLD"])
runtime = Path(os.environ["R"])
manifest = json.loads((runtime.parent / "evidence/worker_smoke_manifest_20.json").read_text())
first = manifest["samples"][0]
input_spec = first["inputs"]["spatial_features"]
assert hashlib.sha256((old / "input/preflight.npy").read_bytes()).hexdigest() == input_spec["sha256"]
new_first = dict(first, inputs=dict(first["inputs"], spatial_features=dict(input_spec, path="preflight.npy")))
oneframe = dict(manifest, samples=[new_first])
(Path(os.environ["PROOF"]) / "one-frame.json").write_text(json.dumps(oneframe, indent=2) + "\n")
PYCODE
```

下一条命令**实际执行宿主模型推理**及协议编码/解码、shape/finite/身份哈希检查，但不启动ROS、CARLA、网络服务或底盘。先完成上述只读检查和环境恢复；可将 `ARM`换为另一个冻结arm，逐个验收：

```bash
ARM=compression_only
set -o pipefail
"$ORIN_PYTHON" -B "$R/scripts/coptv2x_hil_serial_probe.py" \
  --runtime-root "$R" --data-root "$OLD/input" \
  --manifest "$PROOF/one-frame.json" --artifact-root "$STAGE6" \
  --native-repo-root "$OLD/native/V2Xverse" \
  --arm "$ARM" --variant optimized --count 1 \
  --last-wire "$PROOF/$ARM.wire" | tee "$PROOF/$ARM.jsonl"
```

保留命令退出状态与新JSONL/wire输出；检查sample事件和 `probe_complete`，不得把仅有wire文件视为通过。单臂optimized检查不包含baseline/optimized成对一致性检验，也不等于完整AP、HIL或闭环性能验收。不要直接运行固定另一输入SHA、固定20+100次循环的旧 `coptv2x_hil_orin_smoke.py`。

## Foxy与完整部署的后续门槛

容器的镜像加载、可恢复骨架及ROS命令见 [DOCKER_RESTORE](environment/DOCKER_RESTORE.md)。宿主TRT推理不依赖GPU容器；Foxy容器负责ROS通信。单帧wire通过后，可用源码 `coptv2x_hil_serial_ros_smoke.py --wire`独立检验ROS往返；它仍不能替代worker/publisher/proxy、反向SSH与外部最终ACK。完整HIL还需要原H800/CARLA和新输入流。

本次修复范围不包括可选UniAD/ROS1链的121帧默认输入、Astra SDK/libuvc、底盘额外apt包或PL/torchmetrics API冲突。空目录文件恢复、同栈原机测试、新系统安装、ROS往返与真实闭环是不同验收范围，逐项状态以 [VALIDATION](VALIDATION.md) 为准。匿名制品allowlist不因本恢复目录而扩大。
