# Orin 上构建 UniAD-tiny TensorRT 引擎 —— 详细教程

> 适用对象:拿到 `toOrinV2` 包、需要在 NVIDIA DRIVE Orin 上把固定 buffer 全静态 UniAD-tiny
> 模型构建成 TensorRT 引擎的工程师。
> 本教程比 `README_ORIN.md` 更细,逐条命令 + 每步预期输出 + 出错怎么办。

---

## 0. 先理解你手里的东西(30 秒)

这个模型经过"固定 buffer 重构":递归 track 状态被**钉死成 1150 行**(901 个检测 query + 249 个
carry-over 槽),因此 ONNX 图里**没有动态形状(DDS/NonZero)、没有动态 TopK、没有 If 分支**。
这正是老版本(v1)在 Orin TRT 8.5.2 上**建不出来**的根因 —— 老图有动态 TopK(K 必须是常量)和
DDS,8.5.2 的解析器直接拒绝。新图把这些全消除了。

> x86 侧已验证:FP32 静态图在 TRT 8.6 一次建成;INT8 图在 TRT 10.9 一次建成。详见
> `x86_build_logs/` 和 `QUANTIZE_PROGRESS.md` §十六~十八。

三种精度,三个 ONNX:
| 文件 | 精度 | 何时用 |
|---|---|---|
| `onnx/uniad_tiny_imgx0.25_cp.noaten_simp.onnx` | FP32 | 第一步必建,验证 8.5.2 能解析+建图 |
| `onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx` | INT8(标准 QDQ) | INT8 首选(权重 FP+Q/DQ,规避 Myelin kINT8 坑) |
| `onnx/uniad_tiny_imgx0.25_cp.noaten.int8.onnx` | INT8(dq_only) | 备选(权重直存 INT8,体积小) |

---

## 1. 传输到 Orin

```bash
# 在 x86 机器上(包约 1.1GB)
scp -r toOrinV2 <user>@<orin_ip>:~/
# 或:tar 已经打好了
scp toOrinV2.tar <user>@<orin_ip>:~/ && ssh <user>@<orin_ip> "cd ~ && tar xf toOrinV2.tar"
```

---

## 2. 确认 Orin 环境

```bash
# TensorRT 版本(本项目实测实机为 8.5.2.2)
/usr/src/tensorrt/bin/trtexec --version
# CUDA
nvcc --version            # DRIVE OS 通常 11.4
# GPU 算力(Orin = SM 87)
nvidia-smi --query-gpu=compute_cap --format=csv 2>/dev/null || echo "Orin 固定 SM 8.7"
```
记下:
- `TRT_PATH`:trtexec 所在根目录(默认 `/usr/src/tensorrt`,即含 `bin/trtexec` 和 `lib/`)。
- 若不在默认位置,后面所有命令前加 `TRT_PATH=/你的/路径`。

---

## 3. 编译插件 libuniad_plugin.so(必须,aarch64 原生编译)

模型用到 3 个自定义算子(`RotateTRT` / `InverseTRT` / `MultiScaleDeformableAttnTRT`),
建引擎时 trtexec 要用 `--plugins` 加载它们。**固定 buffer 重构只改了 Python 前向,这 3 个 C++ 插件
算子没变**,直接编 v1 的源码即可。

```bash
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
mkdir -p build && cd build

cmake .. \
  -DTENSORRT_PATH=/usr/src/tensorrt \
  -DTARGET_GPU_SM=87

make -j$(nproc)
```

**预期产物**(在 `build/` 下):
- `libuniad_plugin.so` ← 建引擎要用的就是它
- `uniad` ← 推理应用(见第 7 节,端到端跑还需 S7 改造)

**编译排错**:
| 报错 | 原因 / 处理 |
|---|---|
| `Need: -DTENSORRT_PATH=...` | 没传 TensorRT 路径,按上面加 `-DTENSORRT_PATH` |
| `Need: -DTARGET_GPU_SM=...` | 没传算力,Orin 用 `-DTARGET_GPU_SM=87` |
| 找不到 `cudnn`/`nvinfer`/`cublas` | TensorRT/CUDA 不在标准路径,确认 `/usr/src/tensorrt/lib` 和 `/usr/local/cuda` 存在;必要时 `export CPATH/LIBRARY_PATH` |
| `cuOSD`/`stb` 头文件找不到 | **不要移动 `DL4AGX_min` 内的目录层级**,CMake 用相对路径 `../../../common/dependencies/...` 找它们 |

> 注:CMakeLists 是官方原版(x86 上为 conda cudnn 做的临时改动已还原),Orin 系统自带 cudnn 即可。

---

## 4. 构建引擎(核心步骤)

`build_engine_orin.sh` 已经把所有 trtexec 参数封装好:固定 N=1150、8.5.2 兼容旗标
(`--buildOnly` / `--plugins`)、自动选 ONNX 和精度。

### 4.1 第一步:FP32 基线(必做)
这一步验证 Orin 8.5.2 能否解析 + 建这张静态图。**x86 TRT 8.6 上这步已 PASSED**,Orin 应同样能过。

```bash
cd ~/toOrinV2
bash build_engine_orin.sh fp32
```
**成功标志**:`build_fp32.log` 末尾 `&&&& PASSED`,生成 `uniad_tiny_fp32.engine`。

> ❗ 如果 FP32 这步就报 `checkSanity` 或别的解析/建图错:把 `build_fp32.log` 完整带回,
> 说明 8.5.2 还有更深的算子限制,需进一步分析(理论上不该发生 —— 图已全静态)。

### 4.2 第二步:INT8(标准 QDQ)
```bash
bash build_engine_orin.sh int8_qdq
```
**成功标志**:`build_int8_qdq.log` 末尾 `&&&& PASSED`,生成 `uniad_tiny_int8qdq_fp16.engine`。

### 4.3 降精度预案(重要)
x86 公版 TRT **8.6.1** 在 INT8/FP16 都卡在 Myelin 优化器(版本缺口,不是图问题 —— FP32 和
TRT 10.9 都能建)。**Orin 8.5.2 是不同的 vendor build,必须实测**。建议按优先级试:

```
int8_qdq   →  失败?  →  int8(dq_only)  →  失败?  →  fp16  →  失败?  →  先交 fp32 + 带回所有日志
```
对应命令:
```bash
bash build_engine_orin.sh int8        # dq_only 版
bash build_engine_orin.sh fp16        # 安全提速退路(无 INT8 优化器风险)
```

### 4.4 路径不在默认位置时
```bash
TRT_PATH=/your/trt \
PLUGIN=/your/path/libuniad_plugin.so \
bash build_engine_orin.sh int8_qdq
```

### 4.5 建引擎常见错误对照
| 日志里出现 | 含义 | 处理 |
|---|---|---|
| `only supports input K as an initializer`(TopK) | 用错 ONNX 了(老动态图) | 必须用本包 `onnx/noaten*.onnx`(K 已常量化) |
| `convertMyelinWeightsToFp32 ... kINT8` 断言 | dq_only 的 INT8 权重进 Myelin | 改用 `int8_qdq` 模式 |
| `Could not find any implementation for /Resize_x` | 该版本无 INT8 Resize kernel | 用本包 `int8_qdq`(已把 Resize 排除量化) |
| `Impossible to reformat` / `ForeignNode[...]` | Myelin 降精度融合缺口 | 退 `fp16`;或换更高 TRT(10.7+) |
| `Plugin not found: MultiScaleDeformableAttnTRT` | 插件没加载 | 检查 `--plugins` 路径、插件是否编译成功 |
| trtexec 不认 `--buildOnly`/`--plugins` | 极老 trtexec | 脚本里换回 `--skipInference`/`--staticPlugins` |

---

## 5. 验证引擎能加载 + 测性能

```bash
cd ~/toOrinV2
N=1150
S=prev_track_intances0:${N}x512,prev_track_intances1:${N}x3,prev_track_intances3:${N},prev_track_intances4:${N},prev_track_intances5:${N},prev_track_intances6:${N},prev_track_intances8:${N},prev_track_intances9:${N}x10,prev_track_intances11:${N}x4x256,prev_track_intances12:${N}x4,prev_track_intances13:${N}

LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH \
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=uniad_tiny_int8qdq_fp16.engine \
  --plugins=./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so \
  --shapes=$S \
  --iterations=100 --separateProfileRun
```
看 `median GPU Compute Time`,对比 fp32 / int8 引擎。
官方参考(DRIVE Orin,TRT 10.7):FP32 64.0ms / INT8 39.3ms(~1.63× 加速);8.5.2 数值会不同。

---

## 6. 准备 nuScenes 运行数据(端到端推理才需要)

```bash
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
ln -s <你的nuscenes根目录> ./data     # data/ 下需有 nuscenes/samples/CAM_*/...
# uniad_trt_input/info.txt 里按 data/nuscenes/samples/... 相对路径引用图片
```

---

## 7. ⚠️ 端到端推理(enqueueV2)—— 需先做 S7

**只建引擎到第 4 节就完成了。** 但如果要跑 `uniad` 应用做端到端推理 + 可视化,有个**已知待办 S7**:

固定 buffer 重构改了模型的 I/O 契约:
- 输入 track 张量:动态 901..1150 → **固定 1150**;
- 输出 `bboxes_dict_bboxes / scores / labels / bbox_index / obj_idxes`:动态 `[-1,...]` → **固定 `[300,...]` + mask**。

v1 的 `enqueueV2/src/main.cpp`(约 230–292 行)还是按**旧动态契约**读写 I/O,直接跑会维度对不上。
**S7 = 把 C++ 端 I/O 改成定长 1150/300,并用 mask(或 obj_idxes<0)过滤无效行。**

S7 未做不影响"建引擎"(第 4 节),只影响"跑应用"。S7 改造点详见 `QUANTIZE_PROGRESS.md` §十二/§十三。

S7 完成后:
```bash
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
mkdir -p output_int8
LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH \
./build/uniad \
  ~/toOrinV2/uniad_tiny_int8qdq_fp16.engine \
  ./build/libuniad_plugin.so \
  ~/toOrinV2/uniad_trt_input \
  ./output_int8 \
  69
```

---

## 8. 结果回传清单(便于后续分析)

1. `build_fp32.log`(必须,确认 `&&&& PASSED`)
2. `build_int8_qdq.log` / `build_int8.log` / `build_fp16.log`(哪个试了就带哪个)
3. 第 5 节的 trtexec 性能输出(各精度 median GPU Compute Time)
4. 若 FP32 都建不出:完整日志 + `trtexec --version` 输出

---

## 9. 一页速查(TL;DR)

```bash
# 1) 解包
tar xf toOrinV2.tar && cd toOrinV2
# 2) 编插件
cd DL4AGX_min/uniad-trt/inference_app/enqueueV2 && mkdir -p build && cd build
cmake .. -DTENSORRT_PATH=/usr/src/tensorrt -DTARGET_GPU_SM=87 && make -j$(nproc)
cd ~/toOrinV2
# 3) 建引擎(按优先级)
bash build_engine_orin.sh fp32        # 必过
bash build_engine_orin.sh int8_qdq    # INT8 首选
bash build_engine_orin.sh fp16        # INT8 不行时的退路
# 4) 验证
/usr/src/tensorrt/bin/trtexec --loadEngine=uniad_tiny_int8qdq_fp16.engine \
  --plugins=DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so \
  --shapes=<见第5节> --iterations=100
```
