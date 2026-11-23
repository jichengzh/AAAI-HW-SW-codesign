# UniAD-tiny → Orin 端到端部署：总结报告与复现流程

> 实机：Jetson Orin（aarch64, SM 87），TensorRT **8.5.2.2**（CUDA 11.4），主机 `10.201.79.25`
> 模型：UniAD-tiny「固定 buffer 全静态」版（track 状态恒 1150 行 = 901 fresh + 249 carry，无 DDS/动态 TopK/If）
> 数据：nuScenes 69 帧 × 6 相机（414 张 jpg）+ 配套元数据 .bin
> 日期：引擎构建 2026-06-13；端到端验证 2026-09-23

---

## 1. 结论速览

| 项 | 结果 |
|---|---|
| FP32 引擎 | ✅ 构建成功（1949.7s），360 MB |
| FP16 引擎 | ✅ 构建成功（3701.8s），189 MB —— 最快，但**第 0/39 帧整头输出 NaN**（见 §8），交付前须修 |
| INT8 (QDQ) 引擎 | ⚠️ 原图 ❌；**cnn 变体 + `--int8` 单精度 ✅ 可建可跑**（慢于 FP16，但**数值最稳、0 NaN**，见 §7/§8） |
| 纯 GPU 推理（trtexec） | FP16 **509.9 ms** / INT8 606.3 ms / FP32 769.4 ms（median） |
| 端到端 GPU 推理（app 埋点） | FP16 **528.3 ms**（≈1.89 FPS）/ INT8 613.0 ms（≈1.63 FPS）/ FP32 785.2 ms（≈1.27 FPS） |
| 端到端功能验证 | ✅ 69/69 帧跑通，可视化正确（检测框/跟踪/规划轨迹/命令） |
| 精度（vs FP32，见 §8） | planning 轨迹 L2：FP16 **0.0153 m** / INT8 0.0372 m；检测匹配召回 0.93/0.89、类别一致 99.9%/99.7% |
| 数值鲁棒性 | FP32/INT8 全 69 帧干净；**FP16 有 2 帧 NaN** |
| 实时性结论 | FP16 ≈1.9 FPS，**不足以实时闭环控制**；离线回放/低速演示可用 |

FP16 相对 FP32 提速 **1.49×**，逐帧方差极小（sd≈2.5 ms）。**但精度量化发现 FP16 在第 0/39 帧整头 NaN**，
若作小车交付须先修（§8.4）；INT8(cnn) 数值最稳但慢于 FP16。绝对官方指标（mAP/NDS/avg.L2）需 x86 全 val 另测。

---

## 2. 流水线各阶段状态

| # | 阶段 | 状态 | 产物/证据 |
|---|---|---|---|
| 1 | ONNX 静态模型 | ✅ | `onnx/`：noaten_simp(fp32)、int8、int8_qdq |
| 2 | TRT 插件编译 | ✅ | `DL4AGX_min/.../enqueueV2/build/libuniad_plugin.so` |
| 3 | 推理 app 编译 | ✅ | `build/uniad`；`uniad.hpp` 已设 `TRACK_INS_MIN=1150`、`TRACK_MAX=300` |
| 4 | 建引擎 | ✅fp32 ✅fp16 ❌int8 | `uniad_tiny_fp32.engine`、`uniad_tiny_fp16.engine` |
| 5 | trtexec 基准 | ✅ | `perf_fp32.log`、`perf_fp16.log` |
| 6 | 端到端 69 帧 | ✅ | `enqueueV2/output/dumped_video_results/*.jpg`（69 张） |
| 7 | 小车实时接入 | ⏳ 未做 | 需相机流/CAN 实时源（§6） |

---

## 3. 量化数据

### 3.1 纯 GPU 计算（trtexec，`--iterations=50 --warmUp=500`）

| 指标 | FP16 | FP32 |
|---|---|---|
| GPU Compute median | **509.92 ms** | 769.38 ms |
| mean / p90 / p99 | 510.26 / 511.57 / 513.31 | 769.43 / 771.84 / 774.37 |
| Throughput | 1.95 qps | 1.30 qps |

### 3.2 端到端 GPU 推理（app 内 `cudaEvent` 埋点，69 帧）

| 指标 | FP16 | FP32 |
|---|---|---|
| mean | **528.3 ms** | 785.2 ms |
| sd | 2.5 | 3.4 |
| min / max | 516.5 / 532.1 | 780.0 / 796.1 |
| p50 / p90 / p99 | 528.7 / 529.9 / 532.1 | 784.9 / 789.7 / 796.1 |
| 吞吐 | **1.89 FPS** | 1.27 FPS |

> app enqueue 相对 trtexec 纯 GPU 仅多 ~16–18 ms/帧，host 侧开销很小。

### 3.3 整进程 wall time（含图片 IO + 预处理 + 推理 + 后处理 + 调试可视化）

| | FP16 | FP32 |
|---|---|---|
| 69 帧总 wall | 96.50 s | 117.79 s |
| 每帧 | 1398.6 ms | 1707.1 ms |
| 折合吞吐 | 0.72 FPS | 0.59 FPS |
| 非 GPU 部分 | ~870 ms | ~922 ms |

> ⚠️ 非 GPU 部分**主要是调试可视化**（每帧读 6×1600×900 jpg + 画 BEV 合成 + 编码 ~1.7MB jpg），
> **不属于模型运行时**。小车上去掉可视化后，单帧成本 ≈ §3.2 的推理时间 + 少量 CPU 预处理。

### 3.4 建引擎耗时

| 引擎 | 耗时 | 大小 |
|---|---|---|
| fp32 | 1949.7 s | 360 MB |
| fp16 | 3701.8 s | 189 MB |

---

## 4. INT8 失败说明（已知问题，非图问题）

`build_int8_qdq.log` 关键错误：

```
Error[2]: [copyNode.cpp::checkSanity::17] Assertion phylum(*inputs[0]) == phylum(*outputs[0]) failed.
          Use CastNode for cross-phyla conversions.
Error[2]: [builder.cpp::buildSerializedNetwork::751] Assertion engine != nullptr failed.
```

- 发生在 QDQ 优化 pass（`QDQToCopy` 把 Reshape/Cast 后的 Quantize 转 kQDQ 时跨 phylum）。
- 与 x86 TRT 8.6.1 上 INT8/FP16 的 Myelin 缺口同源：**8.5.2 vendor build 的降精度 QDQ 路径缺陷**。
- INT8 图在 x86 TRT 10.9 可完整 PASSED，进一步佐证是 8.5.2 版本限制而非模型图错误。
- 结论：8.5.2 上 **FP16 为最优可用精度**；若需 INT8 提速须升级 TRT 或改量化图。

> 所有 trtexec 日志结尾的 `free(): invalid pointer` 出现在 `&&&& PASSED` 之后的析构阶段，**无害**。

---

## 5. 复现流程（从零到端到端）

### 5.0 前置
- Orin：TRT 8.5.2（`/usr/src/tensorrt`）、CUDA 11.4、cmake/make。
- 本仓库 `toOrinV2/` 已含 onnx、元数据、插件+app 源码。
- nuScenes 图片**不在包内**，需从 x86 拉取（§5.4）。

### 5.1 编译插件 + app
```bash
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
mkdir -p build && cd build
cmake .. -DTENSORRT_PATH=/usr/src/tensorrt -DTARGET_GPU_SM=87 && make -j$(nproc)
# 产物: ./libuniad_plugin.so  ./uniad
```

### 5.2 建引擎
```bash
cd ~/toOrinV2
bash build_engine_orin.sh fp32    # 基线，验证 8.5.2 可解析
bash build_engine_orin.sh fp16    # 推荐交付（~1h）
# bash build_engine_orin.sh int8_qdq   # 已知在 8.5.2 失败，可跳过
```
成功标志：日志结尾 `&&&& PASSED` + 生成 `*.engine`。

### 5.3 trtexec 基准
```bash
N=1150
S=prev_track_intances0:${N}x512,prev_track_intances1:${N}x3,prev_track_intances3:${N},prev_track_intances4:${N},prev_track_intances5:${N},prev_track_intances6:${N},prev_track_intances8:${N},prev_track_intances9:${N}x10,prev_track_intances11:${N}x4x256,prev_track_intances12:${N}x4,prev_track_intances13:${N}
LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH \
/usr/src/tensorrt/bin/trtexec --loadEngine=uniad_tiny_fp16.engine \
  --plugins=./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so \
  --shapes=$S --iterations=50 --warmUp=500 --separateProfileRun
```

### 5.4 准备 nuScenes 图片（414 张）
清单 `required_images.txt`（414 个 `data/nuscenes/samples/CAM_*/*.jpg` 相对路径）已在仓库根。
x86 上 nuScenes 根为 `/data/nuscenes`（即清单路径相对 `/`）。

**方式 A：密钥 rsync（推荐，无需密码落盘）**
```bash
# Orin 上生成密钥并把公钥加到 x86 的 ~/.ssh/authorized_keys（一次性，人工输密码）
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_orin_to_x86
# 把 ~/.ssh/id_orin_to_x86.pub 内容追加到 x86:~/.ssh/authorized_keys

# Orin 上拉取（SRC=/ 因清单路径相对 /）
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
rsync -a --files-from=~/toOrinV2/required_images.txt \
  -e "ssh -i ~/.ssh/id_orin_to_x86 -p <PORT> -o BatchMode=yes" \
  <USER>@<X86_HOST>:/ ./
```

**方式 B：x86 打包后 scp**
```bash
# x86 上（根 / 下含 data/nuscenes）
tar czf nuscenes_69.tar.gz -T required_images.txt
scp -P <PORT> nuscenes_69.tar.gz <orin>:~/toOrinV2/
# Orin 上在 enqueueV2/ 内解压
```

校验：`find data/nuscenes/samples -name '*.jpg' | wc -l` 应为 **414**。

### 5.5 端到端运行
```bash
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2   # CWD 必须含 data/nuscenes
mkdir -p output
LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH \
./build/uniad ~/toOrinV2/uniad_tiny_fp16.engine ./build/libuniad_plugin.so \
  ~/toOrinV2/uniad_trt_input ./output 69
# 产物: output/dumped_video_results/{0..68}.jpg
```

### 5.6 统计每帧推理
```bash
grep -a "timer:  Inference" <log> | awk '{print $3}' | sort -n > inf.txt
n=$(wc -l < inf.txt)
awk '{s+=$1; ss+=$1*$1; if(NR==1||$1<mn)mn=$1; mx=$1} END{printf "mean=%.1f sd=%.1f min=%.1f max=%.1f\n", s/NR, sqrt(ss/NR-(s/NR)^2), mn, mx}' inf.txt
sed -n "$(( n*50/100+1 ))p;$(( n*90/100+1 ))p;$(( n*99/100+1 ))p" inf.txt   # p50/p90/p99
```

### 5.7 验收标准
- 建引擎日志 `&&&& PASSED`。
- 端到端 exit=0，69 张可视化 jpg。
- 可视化含：6 路相机 3D 检测框（带置信度）、BEV 距离圈(15/30/45/60)、跟踪框+速度矢量、绿色规划轨迹、命令文本（如 TURN LEFT）。

---

## 6. 后续工作（小车实测）

当前为**离线回放**验证。上真车还需：
1. **实时相机源**：把 `load_images`(读 jpg) 换成订阅 6 路相机（ROS topic / GStreamer / V4L2）。
2. **实时位姿/CAN**：`l2g_r_mat / l2g_t / img_metas_can_bus / command` 从离线 .bin 换成车上定位+CAN 实时源。
3. **去掉调试可视化**：消除 ~870ms/帧 的非模型开销。
4. **闭环测速**：持续循环测稳态 FPS、传感器→轨迹端到端延迟、显存/功耗。
5. **若需实时（≥10FPS）**：攻 INT8（修 §4 跨 phylum 断言 / 升级 TRT）或降分辨率、抽帧。

---

## 7. INT8 攻关结果（2026-09-23 追加）

原 INT8 QDQ 图在 8.5.2 构建失败（§4）。经两层根因剥离后**已可构建并端到端跑通**：

1. **QDQ 跨 phylum 断言** ← 直接跟在 Reshape/Cast 后的 QuantizeLinear（67 个）触发 8.5.2
   `QDQToCopy` 生成跨 phylum copy 节点。外科移除这些 Q/DQ 对后断言消失。
2. **Myelin 无 kernel** ← `--int8 --fp16` 使 encoder-attention 融合块内出现 f32/f16 混杂
   （`verify_output_type: f32 vs expected f16`），唯一 tactic 被跳过。改用 **`--int8` 单精度**
   （非量化部分 fp32，Myelin 块全 f32）即有 kernel。

**可建变体**：`onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_cnn.onnx`（移除 191 对 Q/DQ，
INT8 仅留 CNN backbone/neck/heads 共 321 个 Q；attention/transformer 留 fp32）。

```bash
bash build_engine_orin.sh int8_qdq_cnn_nofp16   # EXTRA=--int8 → uniad_tiny_int8_cnn.engine
```

**结果**：构建 2415.7s `&&&& PASSED`（273MB）；端到端 69 帧跑通、可视化与 FP16 几乎一致
（精度保持良好）；但性能 **慢于 FP16**：

| 指标 | INT8(cnn) | FP16 |
|---|---|---|
| trtexec GPU median | 606.3 ms（1.65 qps） | 509.9 ms（1.95 qps） |
| 端到端推理 mean | 613.0 ms（1.63 FPS） | 528.3 ms（1.89 FPS） |

原因：8.5.2 迫使 attention 留 fp32，其代价超过 CNN 卷积 int8 的收益。
**结论**：INT8 可行性已证明；**交付推荐仍为 FP16**。要让 INT8 真正提速需升级 TRT（≥10.x）或拆解该 Myelin 块。

---

## 附：本仓库关键文件
- 引擎：`uniad_tiny_fp32.engine`、`uniad_tiny_fp16.engine`、`uniad_tiny_int8_cnn.engine`
- 日志：`build_{fp32,fp16,int8_qdq}.log`、`perf_{fp32,fp16,int8}.log`、`bench/e2e_{fp16,fp32,int8}.log`、
  `build_int8_qdq_{norc,cnn,bb,cnn_nofp16}.log`
- 图片清单：`required_images.txt`（414 条）
- 端到端产物：`DL4AGX_min/.../enqueueV2/output/dumped_video_results/`（FP16）、
  `.../output_int8/dumped_video_results/`（INT8）

## 8. 精度指标量化（三引擎，2026-09-24 追加）

给 app 打补丁把每帧输出张量落盘，FP32/FP16/INT8 用**同一 69 帧、同一自由递归**各跑一遍，以 **FP32 为同平台参考**对比。脚本 `tools_acc/analyze_acc.py` + `tools_acc/analyze_det_match.py`；原始结果 `acc_results.txt` / `det_match_results.txt`。

### 8.1 口径与边界
- 这是**相对精度**（vs 同 Orin/同 TRT8.5.2/同递归的 FP32），隔离"精度格式"单一变量。
- **官方绝对指标（mAP/NDS/AMOTA/minADE/mIoU/avg.L2/avg.Col）测不了**：需全 val(6019)+GT+nuscenes-devkit，且 Orin(aarch64/8.5.2) 引擎不能在 x86 跑；x86 官方指标只反映 x86(TRT10.9) 引擎。
- 自由递归下检测 query 跨帧漂移，`bboxes` 逐索引原始 MSE 无意义 → 检测用**贪心最近中心匹配**(thr=0.25, gate=2m)。

### 8.2 任务级精度（vs FP32）
| 指标 | FP16 | INT8(cnn) |
|---|---|---|
| planning 平均轨迹 L2 | **0.0153 m** | **0.0372 m** |
| planning 逐路点 L2(1→6) | 0.005→0.026 m | 0.010→0.065 m |
| 检测匹配召回(中心≤2m) | 0.926 | 0.894 |
| 匹配框中心距 mean/median | 0.297 / 0.176 m | 0.262 / 0.166 m |
| 匹配框 \|Δscore\| / 尺寸L1 / \|Δyaw\| | 0.068 / 0.0125 / 0.176rad | 0.074 / 0.0155 / 0.178rad |
| 匹配框类别一致率 | 99.9% | 99.7% |
| seg 逐格类别不一致率 | 0.77% | 1.62% |
| bev_embed cos / MSE | 0.9972 / 9.5e-4 | 0.9945 / 2.0e-3 |
| **NaN 帧** | **第 0、39 帧整头 NaN** | **无（69 帧全干净）** |

### 8.3 关键发现
1. **FP16 有 NaN 缺陷**：第 0、39 帧 planning + 全部 300 检测框/分数输出 NaN（bev 仍有限），FP32/INT8 无此问题。目视可视化此前未发现（NaN 框不绘制），是数值 dump 才暴露。**2/69≈2.9% 帧检测+规划完全失效**。
2. **INT8(cnn) 数值最稳**：0 NaN；planning 仅偏 3.7 cm；检测质量与 FP16 同级。因其只量化 CNN 卷积、attention/heads 留 fp32，规避了全量 INT8 的 planning 失真。
3. FP16 与 INT8 检测质量基本相当（召回 0.89–0.93、中心误差 0.17–0.30 m、类别一致 99.7%+）。

### 8.4 交付建议（更新）
- FP16 最快（510 ms）但**须先修 2 帧 NaN**（定位溢出头层单独保 fp32）才可作小车交付，否则不可接受。
- INT8(cnn) 是当前**数值最稳**的低精度方案（606 ms），适合鲁棒性优先场景。
- FP32（769 ms）为精度参考基线。
- 复现：`bash run_acc_dump.sh`（三引擎 dump）→ `python3 tools_acc/analyze_acc.py .` + `python3 tools_acc/analyze_det_match.py .`。
