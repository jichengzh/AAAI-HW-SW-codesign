# 源码恢复与构建入口

`payload/` 以原 `/home/jichengzhi/` 为根，保存部署代码、模型定义、配置、导出/构建脚本和原评测 split。逐文件公开副本SHA256、原权限与排除规则见 [源码清单](manifests/source_capture_manifest.json)。原结果单独放 `key_results/`，不覆盖源码树。4份冻结Copt engine单独放 `engine_payload/`，按同样的原home相对布局复制；原离线包中的4个checkpoint需从 `models/codriving/` 补到 `source/codriving/`，精确命令见 [COPT_ENGINE_RESTORE](COPT_ENGINE_RESTORE.md)，并须在 `--home` 核验前完成。

UniAD默认121帧输入及Astra arm64 SDK由私有 `offline_assets/restore-audit-supplement.tar.gz.part000` 补齐；公开 [审计补包清单](manifests/restore_audit_assets_manifest.json) 只保存1581个文件的相对路径、大小、SHA256与种类及归档哈希，不含归档实体。按 [恢复检查单](RESTORE_CHECKLIST.md) 第3节校验并解包到原home布局。`--offline` 必须检查该归档，`--home` 必须检查补包中全部文件；默认仓库检查无需私有归档。

## 源码复制与凭据处理

```bash
export RECOVERY_REPO=/home/jichengzhi/orin-recovery/github
python3 "$RECOVERY_REPO/tools/verify_recovery.py" --repo "$RECOVERY_REPO"
rsync -a "$RECOVERY_REPO/payload/" /home/jichengzhi/
```

公开副本去掉旧文档中的密码式 SSH 和脚本中的密码式 sudo 命令，改用 SSH key/交互 sudo；变化记录在 manifest 的 `publication_sanitization`。私有 `work/raw_payload/` 保留原字节，不应上传。历史实验文件可能仍包含旧地址或目录，应在运行前核对；不将旧脚本直接当安装检测命令执行。

上游源码若不需随包保存，以 manifest 的 `git_repositories` 固定 commit 重新克隆：MMCV1.5/1.6、MMDetection2.25.1、TorchVision、Pruner。采集时这些仓库 `git status` 与 `git diff HEAD` 均为空；manifest 保留原路径、完整 commit 和不含凭据的上游URL。不是将今日主分支视为历史版本。UniAD 的本地 Git 元数据不可用，其文本源码按原样保留。

## 只恢复当前用途所需的软链接

manifest 的 `symlinks` 是原盘观察记录，包含大量历史 attempt/模型/数据链接，**不应整表盲目重建**。以下是新系统的主要链接；命令仅用于对应目标已经恢复、链接路径尚不存在的情况，已有实体目录时先检查，不覆盖它。

```bash
# 系统先装好ROS Noetic。
ln -s /opt/ros/noetic/share/catkin/cmake/toplevel.cmake \
 /home/jichengzhi/ROS1/wheeltec_robot/src/CMakeLists.txt
mkdir -p /home/jichengzhi/heal_research/HEAL
ln -s /home/jichengzhi/m4_8_orin/HEAL/opencood \
 /home/jichengzhi/heal_research/HEAL/opencood
ln -s /home/jichengzhi/ROS1/wheeltec_robot /home/jichengzhi/wheeltec_robot
```

DAIR完整评测应指向重新下载并配好本项目1789 split的根目录；旧 `m4_8_orin/dair_val` 只有60对点云，不用空链接冒充完整数据。HIL runtime 的 `pydeps`、ResNet candidate 链接等按实际恢复环境逐项建立；旧NPZ/失败attempt未恢复，对应断链是有意舍弃。

FCooper四种结构的离线模型放回 `lane_c_orin_fcooper_stage6_five_config_20260724/source/search_artifacts/sources/`。正式结果的 `canonical_manifest.json` 引用 `V2X/results/.../00_source_audit/recovered_h800/search_artifacts/sources/`；运行原脚本前将上述 `sources/` 同步到该位置（或配置中明确改用已恢复路径），不要把目录名当作模型一致性的证据。四种宽度为 `32x64x64x32x64`、`64x64x64x32x64`、`32x32x64x32x64`、`64x128x256x128x256`，分别保留ONNX与recovered_checkpoint。

## UniAD C++ 原路径构建

原路径：`/home/jichengzhi/liruozhan/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2`。自定义 `main_udp.cpp`、UniAD kernels、BEVFormer TensorRT plugins、cuOSD 和 STB 必须保持原相对层级。具体说明也见 payload 中 `README_ORIN.md` / `TUTORIAL_ORIN_build.md`；其中版本号以本恢复包环境快照为准。

依赖和资产恢复后，以下命令只编译，不启动底盘或推理。旧脚本使用的build路径保持不变：

```bash
cd /home/jichengzhi/liruozhan/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
test -f ../../../common/dependencies/stb/stb_image.h
test -f ../../../common/dependencies/stb/stb_image_write.h
# Jetson apt原机没有/usr/src/tensorrt/include或lib目录；CMake要求同根include/lib。
test -f /usr/include/aarch64-linux-gnu/NvInfer.h
test -f /usr/lib/aarch64-linux-gnu/libnvinfer.so
TRT_ROOT=/home/jichengzhi/tensorrt-8.5.2
mkdir -p "$TRT_ROOT"
test -e "$TRT_ROOT/include" || ln -s /usr/include/aarch64-linux-gnu "$TRT_ROOT/include"
test -e "$TRT_ROOT/lib" || ln -s /usr/lib/aarch64-linux-gnu "$TRT_ROOT/lib"
cmake -S . -B build -DTENSORRT_PATH="$TRT_ROOT" -DTARGET_GPU_SM=87
cmake --build build --parallel 2
# 只读确认产物与库解析，不执行应用。
test -x build/uniad_udp
test -f build/libuniad_plugin.so
ldd build/uniad_udp
```

`/home/jichengzhi/tensorrt-8.5.2` 是本清单新增的头文件/库兼容入口（两个软链接），不是重复安装SDK。已核实原机没有旧教程假定的 `/usr/src/tensorrt/include`/`lib`，不能照抄旧教程路径。新系统若布局不同，应依据实际包布局调整，并记录差异。旧CMake还有DRIVE来源的 `-Xlinker -lsocket`，当前Ubuntu未发现独立libsocket（BSD socket由libc提供）；若新机链接报 `cannot find -lsocket`，需在新系统构建副本移除该遗留行并记录diff，不改原始备份。模型 engine 对TensorRT/CUDA/plugin敏感；不能靠复制旧engine或改文件名证明兼容。

## 最小输入与完整实验的区别

离线包保留一个P4预录BEV输入（float32，shape `[2,64,256,512]`，64MiB；实读NPY头见 `manifests/smoke_input_shape.json`）。原单帧补包保存UniAD第0帧的7份metadata、6张相机图与原info文件；新增审计补包另保存默认 `uniad_trt_input_119plus/info.txt` 的121行、7类共847份metadata，以及其引用的726张六相机图片。图片恢复到 `DL4AGX_min/uniad-trt/inference_app/enqueueV2/data/nuscenes/samples/`，保留info中的相对路径，从 `enqueueV2` 工作目录解析。该121帧集合不代表完整nuScenes、CARLA场景或139GiB BEV缓存已备份。

UniAD单帧仍可只取原info首行（保留行尾分号）与第0帧metadata，按下节命令建立独立smoke入口；只做离线数值/输出验收，不启动UDP底盘闭环。可视化字体在离线包中；也可按源码支持的 `UNIAD_SKIP_VIS=1` 跳过可视化。默认121帧的逐文件完整性由 `--home` 核验，不要用第0帧smoke替代它。

ROS1恢复 `/opt/ros/noetic` 后在 `ROS1/wheeltec_robot` 重建catkin。DITL/Foxy按保存的Docker镜像与挂载配置恢复。Pyramid/FCooper/ResNet等按对应模型结构、config、ONNX、原shape/精度重建TensorRT engine；Copt冻结CoDriving服务按新增engine清单恢复原字节，显式使用Stage6 artifact_root，不能以不同SHA的重建文件替代。完整AP需要另取完整评测数据。

## 仍需外部材料

- CARLA主机、H800训练/导出主机、真实车辆设备、ImageNet原输入流，不由本Orin源码树自动恢复。
- Python包和编译依赖由环境清单重新安装；`pydeps`/venv/conda的完整目录不复制。
- 保留第三方源码的原许可证/版权说明。本恢复目录不重新许可上游代码，数据和模型的再分发条件见各提供方。

## 正式模型目录与单帧输入的具体准备

离线文件校验通过后，FCooper原五臂入口可这样补齐正式路径：

```bash
FC=/home/jichengzhi/lane_c_orin_fcooper_stage6_five_config_20260724/source/search_artifacts/sources
FD=/home/jichengzhi/V2X/results/lane_c_orin_fcooper_stage6_five_config_20260724/00_source_audit/recovered_h800/search_artifacts/sources
mkdir -p "$FD"
rsync -a "$FC/" "$FD/"
```

以下显式构造UniAD第0帧smoke入口，不修改原info。原 `uniad_trt_input/` 与默认121帧的 `uniad_trt_input_119plus/` 是不同输入入口，info和metadata必须成套使用：

```bash
U=/home/jichengzhi/liruozhan/toOrinV2
mkdir -p "$U/uniad_trt_input_smoke"
head -n 1 "$U/uniad_trt_input/info.txt" > "$U/uniad_trt_input_smoke/info.txt"
for d in timestamp l2g_r_mat l2g_t command img_metas_can_bus img_metas_lidar2img img_metas_scene_token; do
 mkdir -p "$U/uniad_trt_input_smoke/$d"
 cp "$U/uniad_trt_input/$d/0.bin" "$U/uniad_trt_input_smoke/$d/0.bin"
done

# 默认121帧材料由审计补包提供；这里只读核对info行数，不运行模型。
test "$(grep -c '[^[:space:]]' "$U/uniad_trt_input_119plus/info.txt")" -eq 121
```

运行前仍须按 `uniad`/`uniad_udp` 源码的实际参数格式检查engine/plugin，先显式建立所选OUTPUT目录（默认脚本使用 `output_udp`）；应用只创建dump子目录，不能依赖exit0判断数值输出存在。离线验收必须核对新生成的数值文件、帧数和错误日志。本清单仅准备输入，没有代你执行推理或向车辆发送控制。

发布隐私处理：3个曾含密码命令的原文件摘要/大小只留在私有备份，不在公开清单中提供；其脱敏公开副本的SHA256与大小完整保留，恢复校验仍可执行。

仓库发布规则额外移除6个历史脚本中的固定GPU分配说明/诊断文字；AST验证确认除了字符串内容外结构完全相同。对应原文件摘要/大小仅留私有，公开副本哈希已更新。恢复后应按本机设备配置选择GPU。


## 2026-11-24服务恢复修复

pipeline runtime的 `scripts/coptv2x_hil_orin_services.py` 与 `coptv2x_hil_episode_services.py` 为有限任务补入显式publisher_ack完成策略：worker正常退出后不提前清理，等待publisher完成真实ACK并正常退出，再记录completed。原始采集文件仍保存在私有备份，source manifest分别保留原SHA与公开修复SHA；恢复到home以published哈希验收。旧无该策略配置的监督语义保持原样，非零退出与超时仍失败。新测试和实际回执见 [服务恢复](COPT_SERVICE_RESTORE.md) 及 [验收记录](VALIDATION.md)。
