# UniAD-tiny 固定 buffer 全静态 INT8 模型 → Orin 部署指南(toOrinV2)

> **本包目标**:在 Orin 上为「固定 buffer 全静态」UniAD-tiny 构建 TensorRT 引擎。
> 这是 v1(toOrin)的继任版本。v1 的动态形状模型在 Orin TRT 8.5.2 上**无法解析**
> (动态 TopK 的 K 必须为常量 + DDS 不成熟,见 `QUANTIZE_PROGRESS.md` §十/§十一)。
> 本包的模型经 S1–S6 固定 buffer 重构,**图中已无 DDS(NonZero=0)、无动态 TopK、无 If**,
> 递归 track 状态恒为 **1150 行(901 fresh + 249 carry)**,单一静态 profile 即可建引擎。

## 0. x86 已完成的验证(交付信心来源)

| 构建 | TensorRT | 结果 |
|---|---|---|
| FP32 静态图(`noaten_simp.onnx`) | **8.6**(x86 公版) | ✅ **PASSED**,引擎 363 MB |
| INT8(`noaten.int8.onnx`) | **10.9**(x86) | ✅ PASSED,引擎 171 MB |
| FP16(`noaten_simp.onnx`) | 8.6(x86) | ❌ Myelin ForeignNode 无 kernel |
| INT8 dq_only / qdq / qdq-noResize | 8.6(x86) | ❌ Myelin kINT8 / 无 INT8-Resize / reformat |

**关键结论(两条)**:
1. **静态图本身合法可建**:老动态图在 x86 TRT 8.6 死于 `checkSanity`/动态 TopK 解析;
   **新静态图在同一 8.6 上 FP32 一次 PASSED**(解析 0.76s 干净)。§十/§十一 两大死结已根除。
2. **x86 公版 8.6.1 的降精度(FP16/INT8)Myelin 实现有缺口**:FP32 过,但 FP16 与三种 INT8
   全部在**解析之后**的 Myelin 优化器阶段失败(每次报不同 ForeignNode/算子)。这是 **x86 TRT
   版本特性,不是图问题**(INT8 图在 TRT 10.9 完整 PASSED)。降精度能否建引擎**须由 Orin 实机
   裁决**——本仓库官方也只在 Orin 上验证过降精度,x86 8.6.1 不构成实机结论(见 §九/§十八)。

## 1. 包内容

```
toOrinV2/
├── README_ORIN.md                  # 本文档
├── QUANTIZE_PROGRESS.md            # 量化 + 固定 buffer 重构全过程报告
├── build_engine_orin.sh            # 引擎构建脚本(fp32 / int8 / int8_qdq;8.5.2 兼容旗标)
├── onnx/
│   ├── uniad_tiny_imgx0.25_cp.noaten.int8.onnx       # INT8(--dq_only)
│   ├── uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx   # ★ INT8 标准 QDQ(8.5/8.6 首选)
│   └── uniad_tiny_imgx0.25_cp.noaten_simp.onnx       # FP32 静态基线(先验证可建)
├── calib_data_fixed_1150.npz       # N=1150 校准数据(透明可复现,Orin 上不需要)
├── x86_build_logs/                 # x86 上 FP32@8.6 / INT8@10.9 / int8_qdq@8.6 的 trtexec 日志
├── uniad_trt_input/                # 69 帧推理元数据(timestamp/l2g/can_bus/lidar2img/command)
└── DL4AGX_min/                     # 编译插件 + 推理应用的最小源码树(目录层级勿动)
    └── uniad-trt/inference_app/enqueueV2/   # enqueueV2 应用 + 插件源码
```

Orin 上自备(包里没有):**nuScenes 图片数据**(`uniad_trt_input/info.txt` 按 `data/nuscenes/samples/...` 引用)。

## 2. Orin 环境确认

```bash
/usr/src/tensorrt/bin/trtexec --version   # 实机实测为 TensorRT 8.5.2.2(CUDA 11.4)
```
若 TensorRT 不在 `/usr/src/tensorrt`,后续以 `TRT_PATH=<路径>` 传入。

## 3. 编译插件(Orin,aarch64,SM 87)

custom 算子(RotateTRT / InverseTRT / MultiScaleDeformableAttnTRT)由插件提供;
固定 buffer 重构**只改了 Python 前向**,这 3 个 C++ 插件算子未变,v1 的插件源码可直接复用。

```bash
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
mkdir -p build && cd build
cmake .. -DTENSORRT_PATH=/usr/src/tensorrt -DTARGET_GPU_SM=87 && make -j$(nproc)
# 产物:./libuniad_plugin.so(建引擎用);./uniad(推理应用,见第 6 节 S7 说明)
```

## 4. 构建引擎(核心交付)

```bash
cd ~/toOrinV2
bash build_engine_orin.sh fp32        # ① 必做:FP32 静态基线,验证 8.5.2 能解析+建图(x86 8.6 已 PASS)
bash build_engine_orin.sh int8_qdq    # ② 试 INT8 标准 QDQ(Resize 不量化)+ FP16
bash build_engine_orin.sh fp16        # ③ 若 INT8 在 8.5.2 也遇 Myelin 坑,退回 FP16(安全提速)
# 备选:bash build_engine_orin.sh int8  # --dq_only 版
```
- 成功标志:日志结尾 `&&&& PASSED`,生成对应 `*.engine`。
- 脚本已用 8.5.2 兼容旗标(`--buildOnly` / `--plugins`,非 v1 的 `--skipInference` / `--staticPlugins`)。
- 全静态:`min=opt=max=1150`,无需动态范围。
- **降精度预案**:x86 8.6.1 的 INT8/FP16 都卡在 Myelin(版本缺口);Orin 8.5.2 是不同 vendor build,
  须实测。优先级:int8_qdq → 不行则 fp16 → 都不行则先交 fp32 + 带回日志。
- 路径不同:`TRT_PATH=/your/trt PLUGIN=/your/libuniad_plugin.so bash build_engine_orin.sh int8_qdq`

## 5. 准备运行数据

```bash
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
ln -s <你的nuscenes根目录> ./data        # data/ 下需有 nuscenes/samples/CAM_*/...
```

## 6. 运行推理(enqueueV2)—— ✅ S7 已完成

> 固定 buffer 重构改变了模型 I/O 契约,**S7 已在 x86 完成并验证**(全 69 帧 FP32 端到端跑通,
> 见 `QUANTIZE_PROGRESS.md` §二十)。实际只需改 enqueueV2 的 `include/uniad.hpp` 两个 `#define`:
> - `TRACK_INS_MIN` 901 → **1150**(frame-0 输入形状匹配静态引擎);
> - `TRACK_MAX` 200 → **300**(输出 bbox buffer 容纳固定 300 行)。
>
> **本包内的 `DL4AGX_min/.../enqueueV2/include/uniad.hpp` 已经是改好的 1150/300 版本**,
> Orin 上按第 3 节正常编译即可,无需再改。
> (recursion 与输出解码是 shape-transparent 的:`temporal_info_assign` 透传 runtime shape、
> `decode_bbox` 已用 score<0.25 过滤 padding 行,故除这两个 define 外无需其他改动。)

运行(引擎建好后):
```bash
# CWD 需含 data/nuscenes/(info.txt 用相对路径引用图片);或在 enqueueV2/ 下 ln -s 你的 nuscenes
cd ~/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
mkdir -p output
LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH \
./build/uniad ~/toOrinV2/uniad_tiny_int8qdq_fp16.engine ./build/libuniad_plugin.so \
  ~/toOrinV2/uniad_trt_input ./output 69
# 输出:output/dumped_video_results/*.jpg(6 路相机+BEV 可视化)
```
> x86 参考:FP32 引擎 69 帧端到端 mean 36.3ms/帧,可视化正确(检测框/规划轨迹/命令/距离圈)。

## 7. 性能测量

```bash
N=1150
S=prev_track_intances0:${N}x512,prev_track_intances1:${N}x3,prev_track_intances3:${N},prev_track_intances4:${N},prev_track_intances5:${N},prev_track_intances6:${N},prev_track_intances8:${N},prev_track_intances9:${N}x10,prev_track_intances11:${N}x4x256,prev_track_intances12:${N}x4,prev_track_intances13:${N}
LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH \
/usr/src/tensorrt/bin/trtexec --loadEngine=$HOME/toOrinV2/uniad_tiny_int8qdq_fp16.engine \
  --plugins=./build/libuniad_plugin.so --shapes=$S --iterations=100 --separateProfileRun
```
对比 fp32 / int8qdq 引擎的 median GPU Compute Time。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `This version of TensorRT only supports input K as an initializer`(TopK) | 用错了模型——必须用本包 noaten*.onnx(K 已常量化);勿用 v1 的旧 onnx |
| `convertMyelinWeightsToFp32 ... kINT8` 断言 | 用 `int8_qdq` 模式(标准 QDQ),勿用 `--dq_only` 的 int8 |
| `Plugin not found: MultiScaleDeformableAttnTRT...` | `--plugins` 路径错误或插件没编成功 |
| trtexec 不认 `--buildOnly`/`--plugins` | 极老版本可换 `--skipInference`/`--staticPlugins`;实机 8.5.2 用新名 |
| 应用读不到图片 / I/O 维度不符 | 见第 6 节:enqueueV2 需先做 S7 定长 I/O 改造 |

## 9. 结果回传清单

- `build_fp32.log` / `build_int8_qdq.log`(trtexec 完整日志,确认 `&&&& PASSED`)
- 两个引擎的 trtexec 性能输出(median GPU Compute Time)
- 若 FP32 也建不出来:保留日志带回,排查 8.5.2 是否有更深的算子限制
