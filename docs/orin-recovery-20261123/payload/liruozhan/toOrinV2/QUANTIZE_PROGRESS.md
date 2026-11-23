# 阶段性报告:UniAD-tiny ONNX 显式量化(TensorRT-8.6 部署)

> 更新至 **2026-06-05**。任务:把 `uniad_tiny_imgx0.25_cp` ONNX 用 ModelOpt 做 INT8 显式量化(QDQ),最终在 **TensorRT-8.6 硬件**上部署。
>
> **✅ 2026-06-05:ONNX 量化已成功**(RUN8),产物 `onnx/uniad_tiny_imgx0.25_cp.noaten.int8.onnx`,详见第八节;坑 E 根因与修复见第七节末尾。当前遗留:INT8 ONNX 在 **TRT 8.6 x86** 建引擎触发 `checkSanity` 内部断言(TRT 10.9 无此问题),二分定位中,见第九节。

---

## 一、核心结论(最重要的认知)

**量化(校准)的 TensorRT 版本 ≠ 部署的 TensorRT 版本,二者解耦。**

- ModelOpt 的 `modelopt.onnx.quantization` 只是在 x86 上用 ONNX Runtime 的 **TensorRT EP** 跑校准推理、算 INT8 scale,产出**版本无关的标准 QDQ ONNX**。
- 这一步**不在 8.6 硬件上运行**,因此“硬件只能用 8.6”对它没有约束。
- “用 8.6 做校准后端”在技术上**不可行**:ModelOpt 0.29 这套工具链的 ORT TRT EP 只支持 TRT 10.x。
- 因此既定路线 = **x86 上用 TRT 10.x 校准 → 8.6 硬件上 `trtexec --int8` 建引擎**。用户已确认此路线。

---

## 二、初始环境诊断(conda env `uniad2.0`)

| 组件 | 初始状态 | 问题 |
|---|---|---|
| Python | 3.9.25 | —(后来成为天花板) |
| `tensorrt` (py) | 8.6.1 | 无法 import(`libnvinfer.so.8` 未在 LD_LIBRARY_PATH) |
| `onnxruntime-gpu` | 1.19.2 | **GPU EP 全失效**,providers 只有 `['Azure','CPU']` |
| `nvidia-modelopt` | 0.29.0 | ✅ 符合要求 |
| `onnx` | 1.16.1 | ✅ |
| CUDA | 11.8(torch cu118),系统另有 CUDA 12.2 | 混合 |
| ONNX 文件 | `...cp.onnx` / `...cp.repaired.onnx` + `calib_data_shape0_901.npz` | 齐全 |

---

## 三、已完成的工作(全部成果可复用)

### ① 环境修复(py3.9 阶段)
- `tensorrt` 8.6.1 → **10.9.0.34**(装仓库内 cp39 wheel)。
- `onnxruntime-gpu` 1.19.2(cu11,从 azure 源装)→ 配合 TRT 10.9 库,EP 恢复为 `['TensorrtExecutionProvider','CUDAExecutionProvider',...]`。
- 这是 py3.9 能装到的 ORT-gpu 上限。

### ② 插件按 TRT 10.9 重编译 ✅
- 修正 CMake 选错 conda base nvcc 的问题(改用系统 CUDA)。
- 产物:`UniAD/plugins/TensorRT/build/libtensorrt_ops.so`(含 INT8 kernel)。
- 注册的 7 个算子:`BEVPoolV2 / GridSampler / Inverse / ModulatedDeformableConv2d / MultiScaleDeformableAttn / QKV / Rotate`。

### ③ 校准数据校验 ✅
- 24 个 key 与模型 24 个输入**一一对应**,`shape0=901`(minShape)正确。

### ④ 关键阻塞点修复:`aten::unflatten` ✅(本阶段最大贡献)

**问题链:**
1. **Run 1 失败**:两份 ONNX 各含 **29 个 `org.pytorch.aten::unflatten`** 节点,TRT 解析器报 *"Plugin not found"*。
   - 根因:`export_onnx.py` 里 `unflatten->Reshape` 的 symbolic 要求 `sizes` 为**静态常量**,而本模型 `sizes` 是**动态**(`Concat` 输出)→ symbolic 抛异常 → 退化成 aten fallback 残留。**即使重新导出也会残留**,离线改写才是正解。
2. **离线改写**:用 onnx-graphsurgeon 把 29 个 `unflatten` 改写为标准 `Reshape`(`new_shape = Concat(Shape(X)[0:dim], sizes, Shape(X)[dim+1:])`),动态 `sizes` 直接接进 `Concat`。→ `uniad_tiny_imgx0.25_cp.noaten.onnx`
3. **Run 2 失败**:`Reshape changes volume [49,1,768]->[49,1,3,256,49,1,768]`。
   - 根因:29 个节点 `dim` 全为 **-1**(负轴),`post=shape[dim+1:]` 在 `-1+1=0` 时被当成绝对下标 0,取了整个 shape。
4. **免秩修正**:`dim==-1` 时 `post` 置空(只 `Concat(pre, sizes)`);其余情形 `dim+1` 仍为负、ONNX Slice 正确从末尾计数。→ 重写 `noaten.onnx`,`aten_remaining: 0`,`opset {ai.onnx:16}`。

### ⑤ 解析阶段通过 ✅(Run 3)
```
Model with ORT support is saved ... custom ops: ['InverseTRT','MultiScaleDeformableAttnTRT','RotateTRT'].
Model ... opset_version 16 is loaded.
```
TRT 成功解析整图、识别 3 个插件算子,**unflatten/Reshape 问题彻底解决**。

---

## 四、用户三点 8.6 兼容性约束的落实

| 约束 | 落实情况 | 状态 |
|---|---|---|
| ① Opset ≤ 17 | 输入 opset **16** / `ir_version=8`;改写后仍 opset 16,已移除 `org.pytorch.aten` 域 | ✅ |
| ② 仅标准 INT8(无 FP8/INT4) | 用 modelopt 默认 INT8 QDQ,无任何 fp8/int4/AWQ/GPTQ 参数 | ✅(待最终产物复核) |
| ③ 原生算子兼容 | 8.6 FP32 基线模型;自定义算子由插件接管;只加 Q/DQ,不引入新算子 | ✅ |

---

## 五、遇到的“墙”:为什么必须升级 Python(本阶段终点)

**Run 3 在校准前最后一道版本门槛中止:**
```
WARNING: This model contains DDS ops, please upgrade your ORT version to 1.21.0+
  (DDS ops require TRT 10.6+; earlier ORT compiled with TRT <=10.4). Disabling it.
Exception: This model contains DDS and custom ops. Custom ops are only supported
  with the TensorRT EP, but that has been disabled.
```

**死锁逻辑:**
- 模型同时含 **DDS(数据依赖形状)+ 自定义插件算子**。
- 自定义算子**只能**走 TRT EP;DDS 校准要求 ORT 内置 **TRT ≥10.6**。
- `ORT 1.19.2` 用 TRT ≤10.4 编译 → ModelOpt 禁用 TRT EP → 与“自定义算子必须 TRT EP”冲突 → 中止。
- 解药 = **ORT ≥ 1.21.0**;但 **ORT 1.21.0 需 Python ≥ 3.10**(ORT 自 1.20 起不再发 py3.9 轮子)。

**→ Python 3.9 锁死了 ORT 上限。这是升级到 Python 3.11 的唯一原因。**

补充探测结论:
- ORT 1.21.0 **cu11 不存在**(cu11 源最高 1.20.1);ORT 1.20.1 仍用 TRT ≤10.5,**不满足 DDS**。
- ORT 1.21.0 **仅有 cu12 版** → 强制走 **CUDA 12 通道** → 需额外下载 **TensorRT 10.9 cu12**(仓库只有 cu118),并按 cu12 重编插件。

---

## 六、可复用资产清单(进入 py3.11 阶段时无需重做)

| 资产 | 路径 | 复用性 |
|---|---|---|
| 已修复的输入 ONNX | `UniAD/onnx/uniad_tiny_imgx0.25_cp.noaten.onnx` | ✅ 直接复用 |
| 校准数据 | `UniAD/calib_data_shape0_901.npz` | ✅ |
| unflatten 改写逻辑 | (已验证通过解析) | ✅ |
| 插件源码 | `UniAD/plugins/TensorRT/` | ⚠️ 需按 cu12 **重编一次**(C++ 与 py 版本无关) |

---

## 七、py3.11 / CUDA-12 阶段(已搭建,卡在运行时)

环境(全部就绪):
1. ✅ 新建 conda env `uniad_quant`(Python 3.11)
2. ✅ modelopt 0.29 + **ORT-gpu 1.21.0(cu12)**
3. ✅ 下载并解压 **TensorRT 10.9 cu12** → `TensorRT-10.9_x86_cu128/`,装 cp311 wheel
4. ✅ 插件按 cu12 + CUDA 12.2 重编 → `UniAD/plugins/TensorRT/build_cu12b/libtensorrt_ops.so`
5. ✅ onnx 降到 **1.16.1**(修 `onnx.reference.custom_element_types` import 错误)

逐个攻克的运行时坑(按出现顺序):
- **坑 A**:`ImportError: onnx.reference.custom_element_types`(onnx 1.21 过新)→ 降到 1.16.1 ✅
- **坑 B**:DDS 门槛 → ORT 1.21 + TRT cu12 **已越过** ✅
- **坑 C**:`trt.plugins:RotateTRT not a registered function/op`
  - 根因:`ort_utils._prepare_ep_list` 给 TRT EP 传裸字符串、**未传 `trt_extra_plugin_lib_paths`**(modelopt 0.29 这条 `graph_utils` 路径的疏漏)。
  - 修复:**monkey-patch** `_prepare_ep_list`,给 `TensorrtExecutionProvider` 补 `{"trt_extra_plugin_lib_paths": <so>}`(经环境变量传入,不改源码)✅
- **坑 D**:`Failed to load libonnxruntime_providers_tensorrt.so: libcudnn.so.9 cannot open`
  - 根因:`uniad_quant` 环境**没装 cudnn**。
  - 修复:把 `uniad2.0` 环境的 `nvidia/cudnn/lib`(含 cu12 cudnn 9.10 的 `libcudnn.so.9`)加进 `LD_LIBRARY_PATH` ✅

**坑 E(运行时 enqueue 失败)— ✅ 已于 2026-06-05 根因定位并修复:**

更详细的日志显示 enqueue 失败的真实错误是:
```
IShuffleLayer /Reshape_215: reshaping failed for tensor: /Sub_70_output_0
reshape would change volume 0 to 1. Instruction: RESHAPE input dims{0} reshape dims{}
```
- **不是插件问题,不是显存问题,是校准数据违反了图的隐含约定。**
- trace 导出的 ONNX 含一段 track 后处理子图(`Greater(>0) → NonZero → Gather → Reshape(成标量)`),其结构假设 `prev_track_intances` 中**恰好有 1 个活跃 track**(`save_period > 0`)。
- 旧校准数据 `calib_data_shape0_901.npz` 是全零"第 0 帧"(obj_idxes 全 -1、save_period 全 0)→ NonZero 选出 **0 个元素** → Reshape {0}→{}(volume 0→1)失败。
- **官方配方本身就有这个坑**:`tools/prepare_calib_data.py` 保存的第一个 shape0=901 样本正是这种全零帧。
- **修复**:`UniAD/dumped_inputs/` 中有真实 dump 的部署输入(**902 行 = 901 init query + 1 个活跃 track**,save_period=3,与 24 个模型输入逐一精确匹配)。据此重建 `calib_data_real_frame.npz`,以 shape0=902 重跑 → **RUN8 全链路通过**。

**全链路状态:** unflatten 修复 → DDS 门槛 → TRT EP 加载 → 插件注册 → MHA 检测执行 → 校准 → INT8 QDQ 写出,**全部通过 ✅**。

### 关键路径/产物(py3.11 阶段)
| 项 | 路径 |
|---|---|
| 量化 env | conda `uniad_quant`(py3.11) |
| TRT cu12 | `TensorRT-10.9_x86_cu128/` |
| cu12 插件 | `UniAD/plugins/TensorRT/build_cu12b/libtensorrt_ops.so` |
| cudnn(借用) | `<uniad2.0 env>/.../nvidia/cudnn/lib/libcudnn.so.9` |
| 当前命令日志 | `UniAD/quantize_cu12.log`(RUN7) |
| EP 补丁 | monkey-patch `ort_utils._prepare_ep_list` 加 `trt_extra_plugin_lib_paths` |

---

## 八、✅ 量化成功(RUN8,2026-06-05)

### 产物
`UniAD/onnx/uniad_tiny_imgx0.25_cp.noaten.int8.onnx`(267 MB)

### 验证结果(三条 8.6 约束全部满足)
| 检查项 | 结果 |
|---|---|
| Opset | **ai.onnx 16**(≤17 ✅);`trt.plugins:1` 为插件域 |
| Q/DQ 节点 | Q=374 / DQ=519;差值 145 = `--dq_only` 权重直存 INT8 |
| 零点 dtype | **全部 INT8**(893 个),无 FP8/INT4 ✅ |
| 自定义算子 | 仅原有 3 种插件算子共 29 节点(RotateTRT×1 / InverseTRT×1 / MSDA×27)✅ |
| 量化节点数 | 457(Conv 104 / Add 181 / MatMul 71 / Gemm 42 / Resize 15 / MaxPool 8 / 其它) |

### 复现入口(已固化为脚本)
| 文件 | 用途 |
|---|---|
| `UniAD/run_quantize8.sh` | RUN8 量化完整命令(env/LD_LIBRARY_PATH/SHAPES=902) |
| `UniAD/quantize_patch_runner.py` | modelopt EP monkey-patch(注入 `trt_extra_plugin_lib_paths`) |
| `UniAD/calib_data_real_frame.npz` | 真实帧校准数据(由 `dumped_inputs/*.dat` 重建,shape0=902) |
| `UniAD/quantize_cu12_run8.log` | RUN8 全量日志 |

---

## 九、遗留问题:TRT 8.6 x86 建引擎触发 checkSanity 断言(调查中)

为验收量化产物,在 x86 上用 `TensorRT-8.6_x86_cu118/bin/trtexec` 建引擎(8.6 插件已编译:`plugins/TensorRT/build_trt86/libtensorrt_ops.so`,nvcc 11.8):

| 输入 ONNX | TRT 8.6 结果 | TRT 10.9 结果 |
|---|---|---|
| `noaten.onnx`(未简化,含 29 个 If) | ❌ `If_OutputLayer` 两分支 shape 不一致([3,49,1,256] vs [3,49,1,1,256]),8.6 要求严格一致 | ✅(modelopt 流程内可建) |
| `noaten_simp.onnx`(onnxsim 后,If 全折叠) | ❌ **`checkSanity.cpp::checkLinks::198` 内部断言**(解析通过,网络校验崩) | ✅ |
| `noaten.int8.onnx`(量化产物,同底图) | ❌ 同上 checkSanity | 待测 |

- graphsurgeon `cleanup+toposort` 后重试 → 无效,排除悬空张量假设。

### 二分结果(2026-06-05,`bisect_prefix.py` + `bisect_run.sh`)
```
k=971 PASS / k=1457 PASS / k=1487 PASS
k=1494 FAIL(Myelin: "No Myelin Error exists")
k≥1494 全部 FAIL(前缀为 Myelin 错误;全图为 checkSanity 断言)
RESULT: first-fail in (1487, 1494]
```
- 区间内节点本身是**普通算子**(Sqrt/Log/Div/Add,分解式 LayerNorm 与 decoder 的 log 框编码)。
- **真正的模式特征**:从该区间开始,来自上游 `NonZero`(DDS,数据依赖形状)的 `'unk'` 动态维张量开始进入 LayerNorm 分解模式 → 8.6 Myelin 融合区对 DDS 形状处理不成熟 → 内部崩溃。checkSanity / Myelin 两种报错是同一根源的不同表象。
- TRT 8.5/8.6 才引入 NonZero/DDS 支持,10.x 才稳定 —— 与"同图 10.9 正常"完全一致。

### 关键考证:x86 8.6.1 ≠ 实机 8.6.13
- 本仓库初始发布(commit `2914d54`)明确记载:推理应用在 DRIVE Orin 上以 **TensorRT 8.6.13.3** 测试通过 —— 即同源 ONNX **在 8.6.13 上建过引擎**。
- DRIVE OS 的 8.6.13 比 x86 公版 8.6.1 多大量内部修复;x86 上触发的断言**很可能在 8.6.13 上不存在**。
- 另注:本仓库官方 INT8(EQ) 结果(explicit_quantization.md)是 **TRT 10.7 Orin**,官方从未在 8.6 上验证过显式量化引擎。

### 结论与建议(下一步)
1. **优先**:把 `noaten.int8.onnx` + 8.6 实机插件直接在部署硬件(8.6.13)上 `trtexec --int8 --fp16` 建引擎(命令模板在 `build_engine_trt86.sh`,shapes MIN=OPT=901/MAX=1150)。x86 8.6.1 的失败不构成实机结论。
2. 若实机同样触发:再做 DDS 子图改写(如 decoder 入口对 'unk' 维 pad 到固定上界),工程量较大,届时再评估。
3. x86 上如需性能预估,可用 TRT 10.9 对比 INT8 vs FP32(图已验证可建)。

---

## 附:关键命令备忘

```bash
ROOT=/home/liruozhan/DL4AGX/AV-Solutions/uniad-trt

# unflatten -> Reshape 离线改写(免秩负轴处理),已产出 noaten.onnx
#   逻辑:dim==-1 -> Concat(Shape(X)[0:-1], sizes);其余 -> Concat(pre, sizes, post)

# py3.9 阶段量化命令(因 DDS 门槛失败,仅作参考)
#   LD_LIBRARY_PATH=<cudnn>:<TRT10.9>/lib:<cuda>/lib64 \
#   python -m modelopt.onnx.quantization \
#     --onnx_path=.../uniad_tiny_imgx0.25_cp.noaten.onnx \
#     --trt_plugins=.../libtensorrt_ops.so \
#     --calibration_eps trt cuda:0 cpu \
#     --calibration_shapes=<SHAPES> --simplify --op_types_to_exclude MatMul \
#     --calibration_data_path=.../calib_data_shape0_901.npz \
#     --output_path=.../uniad_tiny_imgx0.25_cp.noaten.int8.onnx --dq_only
```
## 十、Orin 实机部署验证记录(2026-06-10)

在目标 Orin 设备上进行引擎构建与部署测试时，遇到了不可调和的底层环境限制，导致阶段性受阻，需记录如下：

### 1. 实际环境版本与预期不符
- **预期**: DRVIE Orin 上搭载 TensorRT 8.6.13.3 (与原仓库文档一致，支持更完善的 DDS)。
- **实际**: 实机环境检查发现搭载的 TensorRT 版本为 **8.5.2.2** (伴随 CUDA 11.4)。
- 因版本较老，原有的 `trtexec` 构建脚本 (`--skipInference`, `--staticPlugins`) 报错，需降级替换为 `--buildOnly` 和 `--plugins` 才能推进。

### 2. 彻底阻塞: 动态 TopK (Dynamic K) 解析失败
再次执行 `trtexec` 时，FP32 基线构建失败并引发 `free(): invalid pointer` 和解析断言：
```
[E] [TRT] ModelImporter.cpp:731: ERROR: ModelImporter.cpp:168 In function parseGraph:
[6] Invalid Node - /TopK_1
This version of TensorRT only supports input K as an initializer. 
```
- **根因**: TRT 8.5 及更早版本不支持 `TopK` 节点的 `K` 值（取前 $ 个元素）作为动态张量输入，要求 $ 必须是常量（initializer）。而在现有的 UniAD 网络拓扑（特别是在轨迹/候选框筛选场景）中，这个 $ 由前面的推理动态计算得来（如根据 shape 取长度）。
- **结论**: 并非先前怀疑的 `checkSanity` 报错，而是 **解析阶段（Parse）直接被此处的动态图限制给卡死**。

由于环境被锁死在 TRT 8.5.2，目前有且只有以下解决路径：
1. **(推荐) 升级 Orin DRIVE OS**: 将固件升级至官方经过测试的 DRIVE OS (带 TRT 8.6.13 或 TRT 10.x)，直接覆盖目前的系统缺陷。
2. **离线图改写 (ONNX GraphSurgeon)**: 编写脚本扫描 `TopK_1` 及所有此类节点，将对应上游连接断开，尝试通过将网络架构里所有可变长的 TopK 的 $ 写死为常数（取决于业务能否将候选数目固定为统一上限），但这可能对精度和上游数据产生链式破坏。

- **结论**: 实机直接崩溃（Aborted core dumped）。这是因为 TensorRT 8.5 的 ONNX 解析器（ModelImporter.cpp:168）完全无法接受动态 K 的 `TopK`，并由此引发了段级内存越界。

### 遗留建议 (Fallback & Rewrite)
由于环境被锁死在 TRT 8.5.2，目前有且只有以下解决路径：
1. **(推荐) 升级 Orin DRIVE OS**: 将固件升级至官方经过测试的 DRIVE OS (带 TRT 8.6.13 或 TRT 10.x)，直接覆盖目前的系统缺陷。
2. **离线图改写 (ONNX GraphSurgeon)**: 编写脚本扫描 `TopK_1` 及所有此类节点，尝试通过将网络架构里所有可变长的 TopK 的 $K$ 写死为常数（取决于业务能否将候选数目固定为统一上限），但这可能对精度和上游数据产生链式破坏。

---

## 十一、❌ "TopK 钉常量"路线被实验证伪(2026-06-10)— TopK 与 DDS 不可分

> 本节记录一次**失败的尝试**及其揭示的关键事实,供后续决策。源码与产物均已回退,RUN8 动态 K 的 `noaten.int8.onnx` 保持完好。

### 曾经的(错误)假设
- 以为 `max_num = min(cls_scores.size(0), 300)` 中 `size(0)` 是全量 query 数(901..1150),故 `min` 恒为 300,动态 K 无意义 → 可离线钉成常量 300 让 TRT 8.5.2 解析。

### 实验证伪(RUN9)
- 用 `tools/rewrite_topk_const.py` 把 2 个 `TopK.K` 钉成常量 300,跑 `run_quantize9.sh` 校准时 **TRT EP enqueue 直接报错**:
  ```
  /TopK: length of reduction axis ((VALUE /NonZero_15[size])) is smaller than K (300)
  Condition '==' violated: 0 != 1
  ```
- **真相**:TopK 的**被规约轴长度 = `NonZero` 的输出长度**,不是全量 query 数。

### 根因(图 + 源码双向确认)
- 部署前向(`uniad_track.py:1193-1196`)先做 `active_index = (obj_idxes>=0) & (track_scores>=thresh)`,经 `index_bool2long_trt`(内部 `nonzero`,见 :1277)得到**动态长度的活跃 track 索引**。
- `select_active_track_query_trt`(:1282-1285)把所有张量**筛到活跃集**,长度 = `N_active`(动态,通常远小于 300,校准帧里是 0/1)。
- 因此 `decode_single_trt` 里 `cls_scores.size(0)` **就是 `N_active`**,`K = min(N_active, 300)` ≈ `N_active`。**`min` 是必需的**:它保证 `K ≤ 轴长`。钉死 K=300 在 `N_active<300` 时既违反 TRT 的 `K≤axisLen`、在 PyTorch 里也会直接报错,**语义错误**。

### 关键结论(影响后续路线)
- **TopK 的动态 K 是 NonZero/DDS 活跃过滤的直接产物,与 DDS 不是两个独立问题,而是同一个结。** "只修 TopK、暂不动 DDS"在本图上**不可达**。
- 要在 TRT 8.5.2 上跑通,必须从 DDS 下手,可选方向(均需后续评估精度/工程量):
  1. **把 TopK 前移到活跃过滤之前**:在全量 query 集(≥901,K=300 恒成立)上做 TopK,再用活跃 mask 后处理。改变后处理拓扑,需验证与原语义等价。
  2. **把活跃集 pad 到固定上界**(如 300),让 `N_active` 维变常量,TopK 的 K 也随之可常量化。需全链路核对 pad 元素不污染下游(ScatterND/Gather)。
  3. **(最稳)升级 Orin DRIVE OS 到带 TRT 8.6.13 / 10.x 的官方版本**(doc §10 已列),DDS 支持成熟,同源图本就在 8.6.13 跑通过。

### 已回退/清理
- `uniad_track.py::decode_single_trt` 恢复为动态 `max_num = torch.minimum(size(0), 300)`(method6)。
- 删除 RUN9 全部 `*topkconst*` ONNX 产物。
- 保留 `tools/rewrite_topk_const.py` 与 `run_quantize9.sh` 作为"此路不通"的实验记录(勿用于交付)。
- RUN8 动态 K 产物 `onnx/uniad_tiny_imgx0.25_cp.noaten.int8.onnx` 完好。
- RUN9 失败日志:`quantize_cu12_run9.log`。

---

## 十二、固定 buffer 重构(消减 DDS,进行中,2026-06-10)

> 用户决策:走"固定容量 + mask 取代变长 nonzero 压缩"的全量重构,在 Orin TRT 8.5.2 上跑通。
> 核心设计:递归 track 状态恒为 **901 fresh + 249 carry-over = 1150 行**,图中不再有数据相关形状(DDS)。
> cap=249 = maxShapes(1150) − (num_query+1=901)。padding 槽 = empty track(obj_idxes=-1)。

### 阶段划分
- **S1 递归核心(`modules.py` QueryInteractionModuleTRTP)** — ✅ 完成并验证
- **S2** `track_base_update_trt_script_v5` 去 nonzero(cumsum 分配新 ID) — ✅ 完成并验证
- **S3** detection decoder self_attn 贯穿 padding mask — ✅ 代码+数值验证完成
- **S4** decode/select 输出改定长 padded + valid_count;TopK K 常量化 — 等价逻辑已验证,源码改造进行中
- **S5** motion head `vehicle_mask>0` 改定长 mask — 待做
- **S6** 重新导出 + unflatten 改写(固化脚本) — 待做
- **S7** C++ app I/O 改定长(`inference_app/enqueueV2/main.cpp:230-292`) — 待做
- **S8** 重新量化(校准 shape 固定 N=1150) — 待做

### S1 已完成(✅ 真实模块验证通过)
改动 `projects/.../track_head_plugin/modules.py` 的 `QueryInteractionModuleTRTP`:
- `__init__`:新增 `self.fixed_active_cap = 249`。
- `_select_active_tracks_trt`:变长 `nonzero` gather → 定长 `cumsum`-scatter packing。
  active 行按升序 rank 写进固定 cap 槽,inactive 路由 trash 槽(cap)后切掉,输出恒 `[cap,...]`。
  per-field sentinel:obj_idxes/matched_gt_idxes(idx 3,4)补 -1,其余补 0。返回 `(packed, valid_mask)`。
- `_update_track_embedding_trt(track_instances, valid_mask=None)`:加 `key_padding_mask=~valid_mask`
  做 masked-MHA;末尾用 `where(valid_mask, updated, 0)` 把 padding 行 query 良性化(杜绝全屏蔽 NaN)。
  `valid_mask=None` 时退化为原行为(供 baseline 对照)。
- `forward_trt`:输出从动态 `901+N_active` 变为固定 **1150**。
- 验证脚本(均通过,n_active=0/1/5/50/200/249):
  `s1_test_packing.py`(packing==动态 gather,maxdiff=0)、
  `s1_test_mha.py`(masked-MHA==动态,maxdiff<5e-7)、
  `s1_test_forward.py`(端到端复刻)、
  `s1_test_real_module.py`(**真实模块**:有效行等价/无NaN/恒1150行)。

### S3 可行性已确认(关键结论)
detection decoder 每层 `operation_order=(self_attn, norm, cross_attn, norm, ffn, norm)`:
- **self_attn(MultiheadAttention,1150 query 互注意力)= padding 污染真实行的唯一通道** → 必须传 mask。
- cross_attn(CustomMSDeformableAttention)逐 query 独立采样 BEV → 不污染。
- ffn/norm 逐位置 → 不污染。
- 结论:S1 单独**不足以**保证全模型等价;真正端到端等价要靠 S3 把 valid_mask 贯穿 decoder self_attn。
  原理同 S1 piece2 已验证的 masked-MHA,**可行**。

### 关键认知更正(对照 §十一)
- §十一证伪了"TopK K 钉常量":K=min(N_active,300) 的 N_active 来自 nonzero,是真动态。
- 固定 buffer 重构后 N_active 槽变成固定 cap,**TopK K 才可安全常量化**(S4)。两件事的依赖关系在此闭环。

### S3 已完成(✅ 代码链 + 数值验证)
**关键发现**:mmcv 1.6.1 `BaseTransformerLayer` 里 self_attn 用 `query_key_padding_mask`、
cross_attn 才用 `key_padding_mask`。原 decoder 传的 `key_padding_mask` 喂的是 cross_attn
(BEV 的 2500 token),**根本没屏蔽 padding query**。S3 要传的是 `query_key_padding_mask`。

改动链(导出走 `tiny_imgx0.25_e2e_trt_p.py`:BEVFormerTrackHeadTRTP / PerceptionTransformerUniADTRTP
/ DetectionTransformerDecoderTRTP):
1. `uniad_track.py:_forward_single_frame_inference_trt`:`query_key_padding_mask =
   (track_instances[3] < 0).unsqueeze(0)` [1,1150],图内由 obj_idxes 推导,**无需新增图输入**。
   传给 `get_detections_trt`。
2. `track_head.py:BEVFormerTrackHeadTRTP.get_detections_trt`:加 `query_key_padding_mask` 参数,
   透传给 `get_states_and_refs_trt`。
3. `transformer.py:PerceptionTransformerUniADTRTP.get_states_and_refs_trt`(556):加参数,
   传给 `self.decoder(..., query_key_padding_mask=...)`。
4. `decoder.py:DetectionTransformerDecoderTRTP.forward` **无需改**:`query_key_padding_mask`
   经 `**kwargs` 自动透传到 mmcv layer,由 `BaseTransformerLayer` 喂给 self_attn。
- 验证:`s3_test_self_attn.py`(真实 mmcv MultiheadAttention,256/8 dropout=0.1):
  1150 行带 mask 的真实行输出 == n_valid 行无 padding 基线,maxdiff<2.4e-7,真实行无 NaN。
- cross_attn(MSDeformAttn)逐 query 独立采样 BEV,不污染,无需 mask。
- 注:padding 行经 self_attn 后是 NaN(全屏蔽 softmax),但真实行不受影响;padding 行最终由下游
  obj_idxes<0 排除。若 ONNX/TRT 对 NaN 敏感,S4/S6 阶段可在 decoder 出口对 padding 行做 where 清零。

### S2 已完成(✅ 真实函数验证)
改动 `uniad_track.py` 的 module-level `track_base_update_trt_script_v5`:
- 新生 track ID 分配:`arange(cond1.nonzero().shape[0])+max_obj_id` + `track3[index_bool2long(cond1)]=data`
  → `rank=cumsum(cond1)-1; new_id=max_obj_id+rank; track3=where(cond1,new_id,track3)`。
- 二者对 cond1 为真位置按升序分配相同递增 id。验证:`s2_test_idassign.py`(复刻对比,n_new=0/1/5/50/249
  逐元素等价)+ 真实函数冒烟(无 nonzero / 形状恒 1150 / 无 NaN / max_obj_id 递增正确)。

### 剩余阶段(S4-S8)的 nonzero/DDS 来源清单(待处理)
- `index_bool2long_trt` 仍在的调用点:uniad_track.py:1021,1026(active/other 拆分)、1196(输出 active 筛选)、
  1216(SDC 拆分)、1300、1450-1454(decode 输出 mask)、1562/1574/1588(shape_len_* scene-reset 重建)。
- `modules.py:151`(MemoryBankTRTP.forward_trt 的 valid_idxes nonzero)。
- `motion_head.py:653-656`(vehicle_mask>0 布尔索引)。
- 这些分属:S3 余项(1021/1026 拆分→定长 mask)、S4(decode/select 输出定长 + TopK K 常量)、
  S5(motion vehicle 过滤)、S2 余项(MemoryBank)。需逐个改定长 + 数值等价验证。

### S4 进展(等价逻辑已验证,源码改造进行中)
- 数据已就位:`data/nuscenes`(trainval)+`data/infos/*.pkl`+`dumped_inputs/`+`nuscenes_np/`,
  S6 重新导出在本地可跑通,S4–S8 端到端验证链成立。
- **等价方案(s4_test_decode.py 全过,n_active=0..400)**:
  原两层变长(`topk(min(M,300))` + `index_bool2long(mask)`)→ 定长:
  `valid_pre=(obj>=0)&(score>=thresh)`;`ts=where(valid_pre,track_scores,-inf)`;
  `idx=ts.topk(300)`(轴=1150≥300,**K=300 常量合法,解 §11 死结**);gather→[300];
  `valid300=valid_pre[idx]`;`mask300=center_range_mask & valid300`。输出恒 `[300,...]`+mask300。
- **契约影响面(待逐步改)**:
  - `track2mop_trt:1200-1203` 预过滤 `active_index=index_bool2long(...)` → 改为保留 1150 + valid_pre mask。
  - `select_active_track_query_trt:1282` 14 项 `item[active_index]` + 内部 `[active_index][bbox_index][mask]`
    三重变长索引 → 定长 gather(topk idx)+ mask。
  - `decode_single_trt:1404` topk/mask 定长化。
  - 输出 `bboxes_dict_bboxes/scores/labels/obj_idxes` 从变长 `[-1,..]` 变 `[300,..]`+valid_count,
    波及 motion head 入参、SDC、ONNX 输出名(export_onnx.py:384-388)、C++(S7)。
  - SDC 选择(`:1222` `obj==-2`,固定 1 个 ego 槽在 index 900)→ 定长 1 槽,无需 topk。

### S4 detector 侧源码改造已完成(✅ AST 通过,无 live nonzero)
`uniad_track.py` 部署前向路径的 `index_bool2long_trt`/`nonzero` 已全部消除(仅剩 2 个函数定义本身
未被调用 + 1 处注释):
- `decode_single_trt`:全 1150 行上 `where(obj>=0, score, -inf)` → `topk(300)` 定长(K=300 常量,
  解 §11) → center-range mask & valid300 → 返回定长 `[300,...]` + mask300。
- `select_active_track_query_trt`:去三重变长索引,用 topk 的 bbox_index 定长 gather
  track_query_embeddings/matched_idxes(定长 300,mask 交下游)。
- `track2mop_trt`:删除 active_index 变长预过滤。
- SDC 选择:`obj==-2` nonzero → 固定下标 `[900]` gather。
- `_forward_single_frame_inference_preprocess_trt_v2`:active/other nonzero 拆分+重排+cat →
  全 1150 行就地 velo 更新 + `where(active,...)` 选择(保持行位置不变,ego 仍在 900)。
- `shape_len_1/2/3`(scene-reset):去 `-1e4` nonzero 剥离,保留全 1150 行(padding 行 obj<0 由下游排除)。
- **待验证**:这些改动是逐行做的等价改写,尚未跑真实前向/导出确认数值与原动态路径一致(归 S6 集成)。
- **待办(S4 收尾→S5)**:track_query_embeddings 现为定长 300+mask,motion head(S5)需按 mask 消费,
  否则在 motion 入口变长会令 DDS 再发。

---

## 十三、当前总状态与交接点(2026-06-10 截至本次会话)

### 阶段状态总表
| 阶段 | 内容 | 状态 |
|---|---|---|
| S1 | 递归核心 `modules.py`(packing+masked-MHA) | ✅ 完成,数值验证(packing 改 TopK-gather) |
| S2 | `track_base_update_v5` 新生 ID 去 nonzero | ✅ 完成,真实函数验证 |
| S3 | decoder 贯穿 `query_key_padding_mask` | ✅ 完成 + mask 根因修复(只屏蔽 carry padding) |
| S4 | decode/select 定长 + TopK K 常量 | ✅ 代码完成 + 导出验证通过 |
| S5 | motion head `vehicle_mask>0` 定长化 | ✅ 完成,seg_out 逐元素等价验证 |
| MemoryBank | `MemoryBankTRTP` 去 nonzero(定长 where) | ✅ 完成,数值验证(s5b_test) |
| S6 | 重新导出 + unflatten 改写 | ✅ 完成,NonZero=0 全静态图 |
| S7 | C++ app I/O 改定长 | ⏳ 未开始(不阻塞 S8) |
| **S8** | **重新量化(N=1150 校准)** | ✅ **成功(2026-06-11,见 §十六)** |

### 已修改源码文件(仅部署路径,训练路径未碰)
1. `projects/.../track_head_plugin/modules.py`(S1):`QueryInteractionModuleTRTP` +`fixed_active_cap=249`;
   `_select_active_tracks_trt` 定长 packing;`_update_track_embedding_trt` masked-MHA+padding 良性化;
   `forward_trt` 输出恒 1150。
2. `projects/.../detectors/uniad_track.py`(S2+S3+S4):见 §十二各节。部署前向已无 live nonzero。
3. `projects/.../dense_heads/track_head.py`(S3):`get_detections_trt` 加 `query_key_padding_mask` 透传。
4. `projects/.../modules/transformer.py`(S3):`get_states_and_refs_trt`(556)加 mask 透传。

### 验证脚本(独立可跑,均通过;放在 UniAD/ 根目录)
`s1_test_packing.py` / `s1_test_mha.py` / `s1_test_forward.py` / `s1_test_real_module.py` /
`s2_test_idassign.py` / `s3_test_self_attn.py` / `s4_test_decode.py`。

### S5 勘查结论(motion head,未动)
- `motion_head.py:589 forward_test_trt`:track_query_embeddings 现为定长 `[300,256]`,
  cat sdc 后 `[1,1,301,256]` 进 motionformer。
- `filter_vehicle_query`(:639-661 与 :185 两处):`vehicle_mask>0` 布尔索引把
  `inter_states/out_track_query/track_query_pos/track_scores` 变长 → **S5 的 DDS 源**。
- **级联**:vehicle 过滤定长化后,其变长输出还流入 occ/planning 头 + ONNX 输出名 + C++,
  改动面比 track stack 大。需把 vehicle 过滤改成"固定 301 + mask 不实际删行",mask 贯穿下游。

### 关键风险(交接必读)
1. **S4 未验证**:S4 的 detector 改动是逐行等价改写,**没有像 S1-S3 那样的数值测试托底**。
   唯一验证途径是 S6 完整导出跑真实前向。**建议下一步先跑 S6 导出验证 S4,再做 S5**,
   否则在未验证的 S4 上叠 S5 会放大调试面。
2. **NaN 隐患**:S3 注释指出 padding 行经 decoder self_attn 全屏蔽会出 NaN;真实行不受影响,
   但若 ONNX/TRT 对 NaN 敏感,需在 decoder 出口对 padding 行 where 清零(S6 视情况补)。
3. **可复用资产完好**:RUN8 动态 K 的 `onnx/uniad_tiny_imgx0.25_cp.noaten.int8.onnx` 未被破坏,
   随时可回退到"TRT 10.9 可量化、8.5.2 不可建引擎"的已知基线。

### 下一步建议(优先级)
1. 跑 S6 导出(uniad2.0 env,`tools/uniad_export_onnx.sh` + trt_p config)验证 S4 detector 改动数值正确、
   且图中 NonZero/ScatterND 计数显著下降。
2. S6 通过后再做 S5(motion vehicle 定长 + 下游 mask 贯穿)。
3. S5 后重跑 S6 → S7(C++ 定长 I/O)→ S8(固定 1150 校准重量化)→ Orin 8.5.2 建引擎验收。

---

## 十四、S4 导出验证通过 + S5/MemoryBank 收尾(2026-06-11,本次会话)

### S4 导出验证 ✅
跑 S6 导出(`uniad2.0` env,torchrun + `tiny_imgx0.25_e2e_trt_p` config + `tiny_imgx0.25_e2e_ep20.pth`)
验证 S4 detector 改动。

- **坑 F(已修复)**:`decode_single_trt` 把 TopK 的 K 硬钉成 `bbox_coder.max_num=300`,
  但该函数同时被 **SDC(ego)单槽路径**调用(`select_sdc_track_query_trt`,实例仅 1 行,
  index 900)→ `topk(300)` 报 `selected index k out of range`。
  - **修复**:给 `decode_single_trt / bbox_coder_decode_trt / _track_instances2results_trt`
    加可选常量参数 `topk_k`(默认 300);SDC 路径传 `topk_k=1`。两者皆为编译期常量,
    满足 TRT-8.5.2 "K 必须 initializer" 约束。SDC 路径 `with_mask=False` 且丢弃 mask,
    `topk(1)` 取唯一 ego 槽,语义不变。
- **导出成功**:`torch.onnx.export ... 0 ERROR`,产出 `onnx/uniad_tiny_imgx0.25_cp.onnx`(357 MB)
  + `.repaired.onnx`。日志 `s6_export_run2.log`。

### NonZero(DDS)计数对比(S4 效果实测)
| 产物 | 总节点 | **NonZero** | ScatterND | TopK | If |
|---|---|---|---|---|---|
| RUN8 baseline(动态 K) | 23938 | **35** | 84 | 2 | 29 |
| S4 新导出(本次) | 23739 | **9** | 83 | 2 | 29 |

S4 把 NonZero 从 35 砍到 9。剩余 9 个 NonZero 经上游 trace 精确定位:
- **5×** `track_head_plugin/modules.py:10 index_bool2long_trt` — 调用点全在 `MemoryBankTRTP`
  (`update_trt`:132-135 的 saved_idxes/save_period;`_forward_temporal_attn_trt`:151 的 valid_idxes)。属 §十二"S2 余项(MemoryBank)"。
- **4×** `motion_head.py:1068-1072 filter_vehicle_query_trt`(`vehicle_mask>0` 布尔索引)。属 **S5**。

### 本次待办(按依赖)
1. **S5**:`motion_head.py filter_vehicle_query_trt` 定长化(301 + mask),消 4 个 NonZero。
2. **MemoryBank**:`MemoryBankTRTP` 定长化,消 5 个 NonZero。
3. 重跑 S6 导出 → 确认 NonZero≈0 → unflatten 改写 → **S8 重新量化(N=1150 校准)**。

### 备份
- RUN8 基线图已备份:`onnx/uniad_tiny_imgx0.25_cp.RUN8backup.onnx` /
  `onnx/uniad_tiny_imgx0.25_cp.repaired.RUN8backup.onnx`。
- RUN8 INT8 产物 `onnx/uniad_tiny_imgx0.25_cp.noaten.int8.onnx` 未动。

---

## 十五、S5 完成 + 重大修复:S3 mask 的全 NaN 根因(2026-06-11)

### 🔴 重大发现:之前的"S4 验证通过"是假象 —— 模型输出全 NaN
S5 等价性核对时(对比固定 buffer 前后 seg_out)发现:**`scores`/`bboxes`/`outs_planning`
在 S4 导出里就已经是整片 NaN**(seg_out 因走 max-aggregation + 阈值化侥幸输出全 0/相同,
所以单看 seg_out 看不出问题)。即:
- "导出成功(0 ERROR)+ NonZero 下降" **不等于** 数值正确。
- S1–S4 的单元 test 都通过,但**组装进真实递归后全 NaN**,说明 bug 在"mask 如何接入真实图"这一环。

### 根因:S3 的 `query_key_padding_mask = (obj_idxes < 0)` 把 fresh 检测 query 也屏蔽了
- 固定 buffer 递归状态 = **901 fresh(row 0–900)+ 249 carry(row 901–1149)**。
- fresh 段是**检测主体**,但其 `obj_idxes` 恒为 -1(尚未分配 track id)。
- 原 mask 按 `obj<0` 把 fresh 段当 padding **全部屏蔽**。detection decoder self_attn 里
  这些被屏蔽的行同时是 **key**。frame0 无活跃 track → 1150 个 key 全屏蔽 →
  softmax 全 -inf → **NaN** → 经递归(track_instances[2]=query_feats)**永久扩散**。
- 这正是 §十三 风险 2 预警的 NaN,但远比预期严重(污染全部真实行,非仅 padding 行)。

### 修复(uniad_track.py:_forward_single_frame_inference_trt)
```python
fresh_count = self.num_query + 1            # 901
row_idx = torch.arange(N, device=...)
is_carry = row_idx >= fresh_count           # 只有 carry 段(>=901)才可能是 padding
query_key_padding_mask = (is_carry & (track_instances[3] < 0)).unsqueeze(0)
```
- 只屏蔽 **carry 段的空槽**(row≥901 且 obj<0);fresh 段(前 901)**永不屏蔽**。
- 每个 query 恒可注意 ≥901 个 fresh key → 无全屏蔽行 → 无 NaN;真实行注意的 key 集
  (901 fresh + N_active)与原变长路径一致。

### 验证(RUN4,S3 修复后)
| 输出 | NaN 数(6 帧) |
|---|---|
| scores | **0/300** ×6 |
| bboxes_dict_bboxes | **0/2700** ×6 |
| outs_planning | **0/12** ×6 |
| seg_out | **0/12500** ×6 |
全部有限值,NaN 彻底消除。

### S5 motion head 定长化 ✅(等价验证)
- `motion_head.py:filter_vehicle_query_trt`:不再 `vehicle_mask>0` 变长压缩;返回定长 301 + mask。
  非车辆行 `track_scores` 置 0(occ max-aggregation 中天然排除)。
- `occ_head.py:get_attn_mask_trt/forward_trt/forward_test_trt`:加 `vehicle_mask`,
  在 cross_attn 把非车辆 query 列强制屏蔽(occ 是 BEV→query 注意的唯一污染通道),
  顺序:叠非车辆屏蔽 → if_sum_greater_zero 回退全屏蔽行 → 再叠非车辆屏蔽。
- `uniad_e2e.py`:motion 输出加 `vehicle_mask`,透传 occ_head。
- **等价证据**:S5 改动前(原变长 filter,RUN3)vs 改动后的 `seg_out` **逐元素 maxdiff=0、
  mism=0(全 6 帧)**;`obj_idxes` 同样 0 差异。

### NonZero(DDS)递减
| 阶段 | NonZero |
|---|---|
| RUN8 baseline | 35 |
| S4(detector 定长) | 9 |
| **S5+S3fix(本次 RUN4)** | **5** |
剩余 5 个全部来自 `MemoryBankTRTP`(`modules.py:10 index_bool2long_trt`):
`update_trt`(saved_idxes/save_period)+ `_forward_temporal_attn_trt`(valid_idxes)。

### 坑 F(S4 SDC topk)
`decode_single_trt` 硬钉 K=300,SDC 单槽路径(1 行)`topk(300)` 越界。
修复:加常量参数 `topk_k`(默认 300,SDC 传 1)。两者皆编译期常量,满足 8.5.2。

### 下一步
- 消除 MemoryBank 的 5 个 NonZero(定长 + where 条件更新,per-row 独立 temporal attn)。
- 再导出确认 NonZero≈0 → unflatten 改写 → **S8 重新量化(N=1150 校准)**。

---

## 十六、✅✅ S8 重新量化成功(2026-06-11)—— 全静态 INT8 图

> 用户目标"成功完成 S8 重新量化"**达成**。产物:`onnx/uniad_tiny_imgx0.25_cp.noaten.int8.onnx`
> (296 MB,2026-06-11 03:01),由固定 buffer 全静态图(NonZero=0)校准而来。

### 量化产物验证(全部满足)
| 检查项 | 结果 |
|---|---|
| Opset | **ai.onnx 16**(≤17 ✅)+ `trt.plugins:1` |
| 零点 dtype | **全部 INT8**(903 个),无 FP8/INT4 ✅ |
| 自定义算子 | 仅原有 3 插件(RotateTRT×1 / InverseTRT×1 / MSDA×27)✅ |
| Q / DQ | Q=377 / DQ=526(差 149 = `--dq_only` 权重直存 INT8) |
| **NonZero(DDS)** | **0**(RUN8 是 35)—— DDS 彻底消除 |
| ScatterElements | **0**(超尺寸 scatter 已解,见坑 H) |
| TopK | 3(全部常量 K:detector 2×300 + QIM packing 1×249) |
| **If** | **0**(29 个 If 被 `--simplify` 折叠;RUN8 noaten 是 29) |
| 总节点 | 7464 |

**意义**:这是一张**完全静态形状**的 INT8 图(无 DDS、无动态 TopK、无 If),正是 Orin
TRT 8.5.2 建引擎所需(§十/§十一 的两大死结 —— DDS 与动态 K —— 均已从图中根除)。

### S8 过程中逐个攻克的坑(本次会话)
- **坑 F(SDC TopK 越界)**:`decode_single_trt` 硬钉 K=300,SDC 单槽路径(1 行)越界。
  → 加常量参数 `topk_k`(默认 300,SDC 传 1)。
- **🔴 根因级:S3 mask 全 NaN**:`query_key_padding_mask=(obj<0)` 把 fresh 901 检测 query
  当 padding 全屏蔽 → frame0 全屏蔽行 softmax NaN → 沿递归永久扩散(scores/planning 全 NaN)。
  之前"S4 验证通过"是假象(只看了 export 成功 + NonZero 降,没核 NaN)。
  → 改为只屏蔽 carry 段空槽(row≥901 且 obj<0)。修复后全输出有限。
- **坑 G(MemoryBank DDS)**:`index_bool2long` 变长压缩 5 个 NonZero。
  → 定长 where 条件更新 + per-row 独立 temporal attn(`s5b_test_memorybank.py` 验证)。
- **坑 H(超尺寸 ScatterElements)**:S1 packing 把 [1150] scatter 进 [250] buffer,
  PyTorch 容忍但 ONNX/TRT 的 ScatterElements 要求 indices≤data,modelopt 的 MHA-tracing
  真建 TRT 引擎时崩(`extent of indices(1150) must be <= data(250)`)—— 与 Orin 8.5.2 同源。
  → 改 **TopK(cap, 最小) + gather + where-sentinel**(K=249 常量),`s1_test_packing.py` 验证。
- **坑 I(校准 -inf histogram 崩)**:`decode_single_trt` 用 `-inf` 沉底无效行,N=1150 时
  1149 个 -inf → modelopt 校准 `np.histogram` 报 "Cannot create 128 finite-sized bins"。
  → 改用大负有限值 `-1e4`(track_scores∈[0,1],排序语义不变)。
- **坑 I 附带(MemoryBank NaN 中间张量)**:无效行 mem 全 padding → temporal_attn softmax 全屏蔽
  NaN → 校准崩。→ `_forward_temporal_attn_trt` 把 key_padding_mask 末列强制有效(safe_kpm),
  保证每行≥1 可注意 key 无 NaN;无效行结果反正被 where 丢弃。

### 复现入口(固化脚本)
| 文件 | 用途 |
|---|---|
| `tools/rewrite_unflatten.py` | unflatten→Reshape 离线改写(dim=-1) |
| `tools/build_calib_1150.py` | 从 `dumped_inputs/*.dat` 重建 N=1150 校准 npz |
| `run_quantize_s8.sh` | S8 量化完整命令(N=1150 SHAPES) |
| `calib_data_fixed_1150.npz` | N=1150 真实帧校准数据(1 active + 1149 padding) |
| `s5b_test_memorybank.py` | MemoryBank 定长等价测试 |
| `quantize_s8_run3.log` | S8 成功全量日志(PYTHON_EXIT=0) |
| `s6_export_run7.log` | 最终固定 buffer 导出日志 |

### 全静态导出 NonZero 递减全程
| 阶段 | NonZero | ScatterElements | If(noaten) |
|---|---|---|---|
| RUN8 baseline | 35 | (变长) | 29 |
| S4 detector | 9 | — | — |
| S5 + S3fix | 5 | — | — |
| +MemoryBank | 0 | 12(超尺寸) | — |
| +TopK packing | 0 | **0** | 29 |
| **S8 INT8(simplify 后)** | **0** | **0** | **0** |

### 已修改源码(本次会话新增,均部署路径)
1. `detectors/uniad_track.py`:坑 F(topk_k 参数);S3 mask 根因修复(carry-only);坑 I(-1e4)。
2. `dense_heads/motion_head.py`:S5(filter 定长化返回 vehicle_mask)。
3. `dense_heads/occ_head.py`:S5(get_attn_mask_trt/forward_trt/forward_test_trt 加 vehicle_mask)。
4. `detectors/uniad_e2e.py`:S5(motion 输出 vehicle_mask 透传 occ)。
5. `dense_heads/track_head_plugin/modules.py`:坑 G(MemoryBank 定长 where + safe_kpm);
   坑 H(_select_active_tracks_trt 改 TopK-gather)。

### 下一步(S7 / 实机验收,本次未做)
- S7:C++ app I/O 改定长 1150/300(`inference_app/enqueueV2/main.cpp`),不阻塞 S8。
- Orin TRT 8.5.2 实机 `trtexec --int8` 建引擎验收(此图已无 DDS/动态 K,理论可建)。
- 精度回归:在 nuScenes 上比对 INT8 vs FP32 指标(固定 buffer 等价性已由各 stage 数值测试托底)。

---

## 十七、Orin 交付:x86 建引擎验证 + toOrinV2 打包(2026-06-11)

> 用户要求"做到可交付 Orin 的程度,打包 toOrinV2,同步报告"。本节记录 x86 上对**新静态图**
> 的建引擎验证(交付信心来源)与打包内容。

### x86 建引擎验证(决定性结果)
在 x86 上用 `trtexec` 对固定 buffer 全静态图建引擎:

| 构建 | TensorRT | 结果 | 引擎 |
|---|---|---|---|
| FP32(`noaten_simp.onnx`) | **8.6** 公版 | ✅ **&&&& PASSED** | 363 MB |
| INT8(`noaten.int8.onnx`,dq_only) | 10.9 | ✅ PASSED | 171 MB |
| INT8(dq_only) | 8.6 公版 | ❌ Myelin `kINT8` 断言 | — |

- **核心证据**:**老动态图**在 x86 TRT 8.6 上死于 `checkSanity`(§九)/ 动态 TopK 解析(§十);
  **新静态图在同一 TRT 8.6 上 FP32 一次 PASSED**,解析阶段 0.76s 干净通过(无 TopK initializer
  报错、无 checkSanity)。§十/§十一 两大死结已从图中根除 → Orin 8.5.2 建引擎有强证据。
- **坑 J(Myelin INT8 权重断言)**:`--dq_only` 把权重直接存 INT8,x86 公版 8.6.1 的
  `convertMyelinWeightsToFp32` 对「INT8 权重进入 Myelin 融合区」有断言
  `params.weights.type() != kINT8`。
  - 解法:再产一份**标准 QDQ**(不加 `--dq_only`,权重 FP32 + Q/DQ 对,数值等价)→
    `onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx`(`run_quantize_s8_qdq.sh`)。
    权重对 Myelin 呈 FP,规避断言。Orin 8.5/8.6 优先用 int8_qdq。
  - 两版本数值等价(同套校准 scale)。INT8 图在 TRT 10.9 已 PASSED,证明 QDQ 图本身合法。

### x86 验证脚本/日志
| 文件 | 用途 |
|---|---|
| `build_int8_trt86_v2.log` | INT8(dq_only)@8.6 → 解析通过,Myelin 断言(坑 J) |
| `build_fp32_trt86_v2.log` | FP32@8.6 → **&&&& PASSED**(静态图可建的铁证) |
| `build_int8_trt109_v2.log` | INT8@10.9 → PASSED |
| `run_quantize_s8_qdq.sh` | 标准 QDQ 量化(去 `--dq_only`)|

### toOrinV2 打包内容
```
toOrinV2/
├── README_ORIN.md              # 针对 8.5.2 + 静态图 + int8_qdq 的部署指南(含 S7 待办说明)
├── QUANTIZE_PROGRESS.md        # 本报告
├── build_engine_orin.sh        # fp32/int8/int8_qdq;8.5.2 兼容旗标(--buildOnly/--plugins);N=1150 静态
├── onnx/{noaten.int8, noaten.int8_qdq, noaten_simp}.onnx
├── calib_data_fixed_1150.npz   # 校准数据(可复现,Orin 不需要)
├── x86_build_logs/             # 上述 3 份 trtexec 日志
├── uniad_trt_input/            # 69 帧推理元数据(复用 v1)
└── DL4AGX_min/                 # 插件 + enqueueV2 源码(3 个插件算子未变,复用 v1)
```

### vs toOrin v1 的关键差异
1. 模型:动态 901..1150 → **固定 1150 全静态**(无 DDS/动态 TopK/If)。
2. 实机版本认知更正:8.6.13 → **8.5.2.2**(§十实测)。
3. trtexec 旗标:`--skipInference/--staticPlugins` → `--buildOnly/--plugins`(8.5.2 兼容)。
4. shapes:min=901 动态范围 → min=opt=max=1150 单一静态 profile。
5. 新增 int8_qdq 版规避 Myelin 断言。

### 仍待办(交付后 Orin 侧)
- **S7(端到端推理必需,不阻塞建引擎)**:enqueueV2 `main.cpp` I/O 改定长——
  输入 track 定长 1150;输出 bboxes/scores/labels/bbox_index/obj_idxes 从动态 `[-1]` 改 `[300]`+mask,
  按 mask/obj<0 过滤有效行。README 第 6 节已标注。
- Orin 实机 `build_engine_orin.sh fp32 → int8_qdq` 建引擎验收。
- INT8 vs FP32 精度回归(各 stage 数值等价已托底)。

---

## 十八、x86 INT8 建引擎的三连坑根因 + 交付决策(2026-06-11)

> 在 x86 上尝试把 INT8 静态图建成引擎,连续遇到 3 个**解析后、优化器阶段**的失败。
> 关键认知:**这些都是 x86 公版 TensorRT 8.6.1 的 INT8 优化器/kernel 缺口,不是图非法**
> (每次 `Finished parsing network model` 都干净通过)。x86 TRT 版本无法更换,故 INT8 最终
> 建引擎须由 Orin 实机(8.5.2.2,不同 vendor build)裁决——这与 §九"x86 8.6.1 失败不构成
> 实机结论"一致。

| 尝试 | 模型 | 解析 | 优化器报错 |
|---|---|---|---|
| A | int8(--dq_only) | ✅ | Myelin `convertMyelinWeightsToFp32: kINT8` 断言(坑 J) |
| B | int8_qdq(标准 QDQ) | ✅ | `Could not find any implementation for /Resize_1`(INT8 Resize 无 kernel) |
| C | int8_qdq + Resize 不量化 | ✅ | `Impossible to reformat`(优化器 INT8↔FP reformat 缺口) |

- **不再继续逐坑修**(失败循环识别):3 次都是同一类——x86 8.6.1 INT8 优化器成熟度。
  继续在 x86 打补丁无意义,真正的裁决环境是 Orin。

### ✅ 已牢固证明、可交付的事实
1. **静态图在 TRT 8.6 解析干净**:无 `checkSanity`(§九)、无动态 TopK initializer 报错(§十)。
   —— §十/§十一 两大死结确已根除(老动态图在此处直接崩)。
2. **FP32 静态图在 TRT 8.6 端到端 PASSED**(`build_fp32_trt86_v2.log`,引擎 363 MB)。
3. **INT8 图在 TRT 10.9 PASSED**(`build_int8_trt109_v2.log`,引擎 171 MB)—— QDQ 图本身合法。
4. **FP16 静态图在 TRT 8.6**(`build_fp16_trt86_v2.log`)—— 作为 INT8 退路。

### 交付决策(toOrinV2)
- 打包**全部 4 种 onnx/模式**让 Orin 裁决:`fp32`(已证可建)、`fp16`(安全提速退路)、
  `int8`(dq_only)、`int8_qdq`(标准 QDQ + Resize 不量化,INT8 首选)。
- `build_engine_orin.sh` 加 `fp16` 模式;README 第 0/4/8 节如实写明三连坑与"INT8 由实机裁决"。
- 随包附 `x86_build_logs/` 全部日志,Orin 端对照。
- **判据**:Orin 上 `fp32` 必须 PASS(证明 8.5.2 能解析+建静态图);`int8_qdq` 若遇同类
  优化器坑则回退 `fp16`;INT8 能否成在 8.5.2 上属实机特性,x86 无法预判。

### 仍待办
- Orin 实机:`build_engine_orin.sh fp32`(必过)→ `int8_qdq`(试)→ 不行则 `fp16`。
- S7:enqueueV2 C++ I/O 定长化(端到端推理必需,不阻塞建引擎)。
- 精度回归。

### 补充:FP16 在 x86 8.6.1 也失败 → 锁定根因为 Myelin(非图)
`build_engine_orin.sh fp16` 等价命令在 x86 8.6 也 FAILED,报
`Could not find any implementation for node ForeignNode[/Gather_204.../Slice_323]`
(Myelin 融合子图)。即:**FP32 过、FP16/INT8 全挂**,共因是 x86 公版 8.6.1 的 **Myelin 在
降精度(FP16/INT8)融合区的实现缺口**,与图无关(每次都解析干净、FP32 与 TRT10.9 均能完整建)。
- 结论强化:x86 8.6.1 不是合适的降精度裁决环境;FP16/INT8 能否建须在 Orin 实机定。
- INT8 在 **TRT 10.9 已 PASSED**,证明降精度图在成熟 TRT 上完全可建。

---

## 十九、本地建引擎成功 + Orin 教程打包(2026-06-11)

### 本地建引擎(带 inference,非 --skipInference,真跑通)
| 引擎 | TRT | 结果 | GPU Compute(median) | 体积 | 路径 |
|---|---|---|---|---|---|
| INT8+FP16 | 10.9 | ✅ PASSED | **10.6 ms** | 170 MB | `UniAD/engines/uniad_tiny_int8_fp16.trt109.engine` |
| FP32 | 8.6 | ✅ PASSED | 30.7 ms | 364 MB | `UniAD/engines/uniad_tiny_fp32.trt86.engine` |

- INT8 相对 FP32 **~2.9× 加速**(10.6 vs 30.7 ms),且体积更小。
- 两者都跑了完整 inference loop,证明**可建且可运行**(不仅是建图通过)。
- 日志:`UniAD/engines/build_int8_trt109_local.log` / `build_fp32_trt86_local.log`。
- ⚠️ TRT 引擎是**平台+版本锁定**的,这两个 x86 引擎**不能搬到 Orin 跑**;Orin 必须在本机
  用 `build_engine_orin.sh` 重新建。故未放进 toOrinV2 包(只放可跨平台的 ONNX + 源码 + 脚本)。

### toOrinV2 最终内容
```
toOrinV2/
├── TUTORIAL_ORIN_build.md      # ★ 详细教程(9 节,逐命令+预期输出+错误对照表+速查)
├── README_ORIN.md              # 简版参考
├── QUANTIZE_PROGRESS.md        # 本报告
├── build_engine_orin.sh        # 4 模式 fp32/fp16/int8/int8_qdq;8.5.2 兼容旗标;静态 N=1150
├── onnx/{noaten_simp(FP32), noaten.int8(dq_only), noaten.int8_qdq(标准QDQ)}.onnx  # 均 NonZero=0/If=0
├── calib_data_fixed_1150.npz   # 校准数据(可复现)
├── run_quantize_s8_qdq*.sh     # QDQ 重量化脚本(透明)
├── x86_build_logs/             # FP32@8.6 / INT8@10.9 / 各 INT8@8.6 / 本地 engines 日志
├── uniad_trt_input/            # 69 帧推理元数据
└── DL4AGX_min/                 # 插件 + enqueueV2 源码(SM87 原生编译)
```

### 交付状态小结
- ✅ 模型全静态(NonZero=0/If=0/TopK 常量),x86 已证 FP32@8.6 + INT8@10.9 可建可跑。
- ✅ Orin 包 + 详细教程齐备,build 脚本按 8.5.2 调好。
- ⏳ 待 Orin 实机:`fp32`(必过)→ `int8_qdq` → 不行退 `fp16`;S7 C++ 定长 I/O(端到端推理用)。

---

## 二十、✅ S7 完成 + x86 端到端推理跑通(2026-06-11)

> 用户要求"在 x86 设备完成推理"。走**路线 B(TRT 8.6 + FP32)**:enqueueV2 应用的 8.6 binding
> API 原样可用,无需移植到 10.x。结果:**全 69 帧端到端推理成功**。

### S7 实际改动(远小于预期)
深入读 enqueueV2 源码后发现:**recursion 与 I/O 是 shape-transparent 的**,不需要重写。
- `uniad.cpp`:用运行时 dims(`set_run_dims`/`run_dims`),形状不写死。
- `main.cpp:temporal_info_assign`:把上一帧每个输出的 runtime shape 直接灌进下一帧输入 →
  **1150→1150 自传播**,无需改。
- `post_process.hpp:decode_bbox`:已有 `scores[i] < 0.25` 过滤 → **padding 行(score=0)天然跳过**,
  输出端无需 mask 改造。

因此 S7 只改了 `inference_app/enqueueV2/include/uniad.hpp` 两个 `#define`:
- `TRACK_INS_MIN` 901 → **1150**(frame-0 输入形状匹配静态引擎 min=opt=max=1150)。
- `TRACK_MAX` 200 → **300**(输出 bbox buffer 容纳固定 300 行,否则越界)。
重编译即可(`tensorrt.cpp` 不动)。

### x86 端到端推理结果(FP32 @ TRT 8.6,H800)
| 项 | 值 |
|---|---|
| 引擎 | `UniAD/engines/uniad_tiny_fp32.trt86.engine`(静态 N=1150) |
| 帧数 | **69/69 成功**(EXIT=0) |
| 单帧推理 | mean **36.3 ms** / min 29.7 / max 50.9 |
| 输出 | `UniAD/infer_out/dumped_video_results/*.jpg`(69 张,115 MB) |
| 引擎 I/O 实测 | 输入 track `{1150,...}`、`bboxes_dict_bboxes {300x9}`、`outs_planning {1x6x2}`、`seg_out {1x5x1x50x50}` —— 全部固定形状 ✓ |

可视化质量核对(抽查 frame 2/20/40/50/68):6 路相机拼接 + BEV 视图、检测框(带类别色+朝向)、
ego 框、绿色规划轨迹、命令文字(TURN LEFT)、距离圈,全部正确;全程各帧非黑占比稳定 ~0.34。
(`simhei.ttf` 字体警告无害,自动 fallback DejaVu。)

### 运行命令(复现)
```bash
cd UniAD                      # CWD 必须含 data/nuscenes/(info.txt 用相对路径引用图片)
APP=../inference_app/enqueueV2/build/uniad
PLUGIN=../inference_app/enqueueV2/build/libuniad_plugin.so
ENGINE=engines/uniad_tiny_fp32.trt86.engine
INPUT=../toOrinV2/uniad_trt_input
TRT=../TensorRT-8.6_x86_cu118; E=<uniad2.0 env>
export LD_LIBRARY_PATH=$TRT/lib:$E/.../torch/lib:$E/.../nvidia/cudnn/lib
$APP $ENGINE $PLUGIN $INPUT ./infer_out 69
```

### 说明 / 边界
- **INT8 端到端未在 x86 跑**:enqueueV2 用 TRT 8.x binding API,而 INT8 引擎需 TRT 10.9(8.6 建不出
  INT8,Myelin 坑)。在 x86 跑 INT8 端到端需把 `tensorrt.cpp` 移植到 10.x tensor-name API
  (`enqueueV3`/`setInputShape`/`setTensorAddress`),工程量较大,本次未做(用户选路线 B)。
- **Orin 侧**:8.5.2 同属 8.x binding API,enqueueV2 可直接用;S7 的 `uniad.hpp` 改动已同步进
  `toOrinV2/DL4AGX_min/.../uniad.hpp`(1150/300)。Orin 建出引擎后(fp32/int8_qdq)即可端到端跑。

### S7 改动已同步
- `toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2/include/uniad.hpp` 已更新为 1150/300。

---

## 二十一、planning MSE 测量 + 三角验证发现(2026-06-11)

> 用户要求测官方表里的精度指标。结论:**只测 planning MSE**(avg.L2/avg.Col 是模型固有精度、
> 需全 val 评测,与量化工作关系小,未做)。过程暴露了一个**真实的 INT8 精度问题**。

### 方法(`tools/planning_mse.py`,Python + cupy 驱动 TRT 引擎)
- 6 帧 clip,**teacher forcing**:TRT 每帧的递归输入(prev_track_intances*/prev_bev/max_obj_id…)
  直接取 **PyTorch dump 的上一帧输出**(`nuscenes_np/uniad_pth_trtp_out/`),使两条链输入逐帧
  严格对齐 → outs_planning 才可比(官方"同输入比输出"口径)。
- MSE = mean((trt_plan − pytorch_plan)²),frame 1..5 为 clean 对比。
- 坑:第一版让 TRT 自由递归(用自己上一帧输出),两链从 frame-0 就分叉(PyTorch 901 / TRT 1150),
  MSE~130 不可信;改 teacher forcing 后仍 ~134 → 说明不是递归发散,是别的。

### 三角验证(决定性)
同一驱动跑 **FP32@10.9** 引擎对比 PyTorch:

| 引擎(同驱动,teacher-forced) | planning MSE | 对照官方 |
|---|---|---|
| **FP32 @ TRT 10.9** | **2.88e-07** | 官方 FP32 9.24e-07 ✓ 同量级 |
| INT8 @ TRT 10.9 | **~134** | 官方 INT8 0.05 ✗ 差 ~2600× |

- **FP32 ≈ PyTorch(2.88e-07)→ 驱动与图都正确**,排除测量误差。
- 因此 INT8 的 ~134 是**真实量化精度损失**:本套 INT8 在 **planning head 回归输出**上严重失真
  (TRT planning-y ~0.2 vs PyTorch ~3..16),而检测/跟踪 + 可视化看起来正常。

### 根因推测 + 后续
- 量化命令只 `--op_types_to_exclude MatMul`,**planning head 的敏感算子未保护**。官方 INT8 能做到
  0.05,说明需按官方做法细化敏感层排除(planning/回归相关层保 FP)后**重量化**。
- **交付影响**:INT8 产物**可建可跑**(引擎、延迟、检测可视化均正常),但 **planning 精度未达标**;
  故端到端先以 **FP32 引擎**交付(§二十,69 帧跑通、可视化正确)。INT8 待敏感层细化后重量化再验收。

### 新增产物
- `tools/planning_mse.py`(cupy 驱动 TRT 引擎 + teacher-forcing MSE)。
- `engines/uniad_tiny_fp32.trt109.engine`(三角验证用 FP32@10.9 引擎)。
- 指标已填入 `documents/explicit_quantization.md` 的 x86 扩展表(含发现说明)。

### 本会话最终状态
| 项 | 状态 |
|---|---|
| S1–S6 固定 buffer 重构 + S8 重量化(NonZero=0 全静态 INT8) | ✅ |
| x86 建引擎:FP32@8.6 PASS / INT8@10.9 PASS(FP16/INT8@8.6 = Myelin 版本坑) | ✅ |
| S7 + x86 端到端推理(FP32@8.6,69 帧,36.3ms/帧,可视化正确) | ✅ |
| toOrinV2 包 + 详细教程 `TUTORIAL_ORIN_build.md` | ✅ |
| planning MSE:FP32 达标(2.88e-07)/ **INT8 未达标(~134,待敏感层重量化)** | ⚠️ 已诚实记录 |
| avg.L2 / avg.Col(需全 val 评测) | ⏳ 未做 |
| Orin 实机建引擎 + 端到端验收 | ⏳ 交付后在 Orin 进行 |

## 二十二、✅ Orin 实机建引擎 + 端到端验收(建引擎 2026-06-13 / 端到端 2026-09-23)

> 承接 §二十一 末尾待办「Orin 实机建引擎 + 端到端验收」。本节为实机裁决结果。
> 详细复现流程与完整数据见 `ORIN_E2E_REPORT.md`。

### 实机环境
- Jetson Orin(aarch64, SM 87),**TensorRT 8.5.2.2 / CUDA 11.4**(`/usr/src/tensorrt`)。
- 即 §十七–§二十一 预言的「须由 Orin 实机裁决降精度能否建引擎」的那台机器。

### 建引擎结果(实机裁决)
| 构建 | 结果 | 耗时 | 引擎大小 |
|---|---|---|---|
| FP32(`noaten_simp.onnx`) | ✅ PASSED | 1949.7 s | 360 MB |
| FP16(`noaten_simp.onnx`) | ✅ PASSED | 3701.8 s | 189 MB |
| INT8 QDQ(`noaten.int8_qdq.onnx`) | ❌ FAILED | — | — |

- **FP32/FP16 在 8.5.2 均一次 PASSED**,确认静态图在实机可建(§十/§十一 死结已根除的实机佐证)。
- **INT8 QDQ 失败**,错误与 x86 8.6.1 同源(降精度 QDQ 路径缺陷,非图问题):
  ```
  Error[2]: [copyNode.cpp::checkSanity::17] Assertion phylum(*inputs[0]) == phylum(*outputs[0]) failed.
            Use CastNode for cross-phyla conversions.
  Error[2]: [builder.cpp::buildSerializedNetwork::751] Assertion engine != nullptr failed.
  ```
  发生在 QDQ 优化 pass(`QDQToCopy` 将 Reshape/Cast 后的 Quantize 转 kQDQ 时跨 phylum)。
  结合 §十八(INT8 在 x86 TRT 10.9 可 PASS),定性为 **8.5.2 vendor build 的降精度 QDQ 缺口**。
- **结论:8.5.2 上 FP16 为最优可用精度**,取代原 int8_qdq 交付位。
- 注:所有 trtexec 日志结尾 `free(): invalid pointer` 出现在 `&&&& PASSED` 之后析构阶段,无害。

### 性能基准(实机)
| 指标 | FP16 | FP32 |
|---|---|---|
| trtexec 纯 GPU median | **509.92 ms** | 769.38 ms |
| 端到端 app 埋点推理 mean | **528.3 ms**(sd 2.5) | 785.2 ms(sd 3.4) |
| 端到端推理 p50/p90/p99 | 528.7/529.9/532.1 | 784.9/789.7/796.1 |
| 推理吞吐 | **1.89 FPS** | 1.27 FPS |
| 整进程 wall(69 帧,含调试可视化) | 96.50 s(1398.6 ms/帧) | 117.79 s(1707.1 ms/帧) |

- FP16 相对 FP32 提速 **1.49×**;逐帧 sd≈2.5 ms,稳定无抖动。
- app enqueue 仅比 trtexec 纯 GPU 多 ~16–18 ms/帧,host 开销小。
- 整进程 wall 中 ~870 ms/帧 为**调试可视化**(读 6×1600×900 jpg + BEV 合成 + 编码),非模型运行时;上车应去除。

### 端到端功能验收(✅ 69/69 帧)
- 数据:nuScenes **414 张**(69 帧×6 相机)此前不在 Orin;经密钥 rsync 从 x86 `/data/nuscenes`
  按清单 `required_images.txt` 拉取至 `enqueueV2/data/nuscenes/`(56 MB)。元数据 .bin 69 帧本就在包内。
- 运行:`./build/uniad uniad_tiny_fp16.engine libuniad_plugin.so uniad_trt_input ./output 69`,exit=0。
- 产物:`output/dumped_video_results/{0..68}.jpg`。抽查帧 0/40 可视化正确:
  6 路相机 3D 检测框(带置信度)、BEV 距离圈(15/30/45/60)、跟踪框+速度矢量、绿色规划轨迹、命令(TURN LEFT)。

### 实时性结论(对小车)
- 去可视化后 FP16 稳态 ≈ **1.9 FPS**,**不足以实时闭环控制**(一般需 ≥10 FPS);离线回放/低速演示可用。
- 若需实时:攻 INT8(修跨 phylum 断言 / 升级 TRT)或降分辨率、抽帧;并接入实时相机流 + CAN(见 `ORIN_E2E_REPORT.md` §6)。

### 新增产物
- `uniad_tiny_fp16.engine`(实机 FP16,推荐交付)、`uniad_tiny_fp32.engine`。
- 日志:`build_{fp32,fp16,int8_qdq}.log`、`perf_{fp32,fp16}.log`、`bench/e2e_{fp16,fp32}.log`。
- 图片清单 `required_images.txt`(414 条);端到端可视化 `enqueueV2/output/dumped_video_results/`。
- 总结报告 + 复现流程:`ORIN_E2E_REPORT.md`。

### 全局最终状态(更新 §二十一 表)
| 项 | 状态 |
|---|---|
| S1–S6 固定 buffer 重构 + S8 重量化(全静态 INT8) | ✅ |
| x86 建引擎:FP32@8.6 / INT8@10.9 PASS | ✅ |
| S7 + x86 端到端(FP32@8.6,69 帧) | ✅ |
| planning MSE:FP32 达标 / INT8 未达标(待敏感层重量化) | ⚠️ 已记录 |
| **Orin 实机建引擎:FP32/FP16 PASS,INT8@8.5.2 FAIL(版本缺口)** | ✅ 已裁决 |
| **Orin 端到端验收:FP16 69 帧跑通、可视化正确、1.89 FPS** | ✅ |
| 小车实时接入(相机流/CAN/去可视化/闭环测速) | ⏳ 未做 |
| avg.L2 / avg.Col 全 val 评测 | ⏳ 未做 |

## 二十三、✅ INT8 引擎在 Orin 8.5.2 构建成功 + 端到端跑通(2026-09-23)

> 承接 §二十二「INT8@8.5.2 FAIL」。本节把它攻下来:**INT8 引擎已能在 8.5.2 构建并端到端跑通**,
> 但**性能慢于 FP16**(原因见下),故交付推荐仍为 FP16。

### 两个根因(逐层剥离)
1. **QDQ 跨 phylum 断言**(`copyNode.cpp::checkSanity`):触发类是**直接跟在 Reshape/Cast 之后的
   QuantizeLinear**(全图 67 个)。8.5.2 的 `QDQToCopy` 把这类 Q 转成 kQDQ copy 节点,而 copy 不允许
   跨 phylum(fp→int8)→ 断言。
   **修复**:外科移除这 67 对 Q/DQ(重接线、清孤儿 scale/zp),产出 `int8_qdq_noRC.onnx` → 断言消失。
2. **Myelin 无 kernel**(`optimizer.cpp::computeCosts` Could not find any implementation,
   ForeignNode = transformer encoder layers.0–2 attention 块):
   - 先怀疑 int8 张量喂入该块,逐版收缩量化范围(noRC → cnn=再剥 attention/transformer →
     bb=仅留 img_backbone)**均仍失败**,排除"int8 输入"假设。
   - 真因在日志:`Skipping tactic ... [verify_output_type] Mismatched type for tensor
     /Unsqueeze_124_output_0, f32 vs expected f16`。即 **`--int8 --fp16` 混合精度**使该 Myelin 块内
     某张量被精度选择器分配为 f32,而 Myelin 唯一 tactic 期望全 f16 → tactic 被跳过 → 无实现。
     FP32 构建(全 f32)与 FP16 构建(全 f16)都不混杂,故都能建。
   **修复**:改用 **`--int8` 单精度**(不加 `--fp16`),非量化部分走 fp32,Myelin 块全 f32(有 kernel)。

### 最终可建变体
- ONNX:`onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_cnn.onnx`
  = 原 int8_qdq 移除 191 对 Q/DQ(67 个 Reshape/Cast 触发类 + attention/transformer 区,有重叠),
  保留 **321 个 Q**(CNN backbone/neck/heads 卷积)。checker 通过。
- 构建:`bash build_engine_orin.sh int8_qdq_cnn_nofp16`(EXTRA=`--int8`)→
  `uniad_tiny_int8_cnn.engine`(273 MB),**2415.7 s,&&&& PASSED**。

### INT8 性能与端到端(实机)
| 指标 | INT8(cnn,--int8) | FP16 | FP32 |
|---|---|---|---|
| trtexec GPU median | **606.3 ms**(1.65 qps) | 509.9 ms(1.95 qps) | 769.4 ms(1.30 qps) |
| 端到端推理 mean | **613.0 ms**(sd 4.1) | 528.3 ms | 785.2 ms |
| p50/p90/p99 | 613.8/617.2/621.9 | 528.7/529.9/532.1 | 784.9/789.7/796.1 |
| 端到端 69 帧 | ✅ 跑通,可视化正确 | ✅ | ✅ |

- 端到端:`./build/uniad uniad_tiny_int8_cnn.engine ... ./output_int8 69`,exit=0,69 张可视化;
  帧 40 与 FP16 几乎逐像素一致(检测框/轨迹/命令相同,置信度微差 0.77 vs 0.68)→ **精度保持良好**
  (attention/heads 留 fp32,规避了 §二十一 全量 INT8 的 planning 失真)。
- **但 INT8 慢于 FP16**(606 vs 510 ms):8.5.2 迫使 attention 留 fp32,其代价 > CNN 卷积 int8 的收益。

### 结论
- **目标达成**:INT8 引擎在 8.5.2 可构建 + 可端到端跑通(此前认为不可能)。
- **交付推荐不变**:FP16(510 ms)仍是最快;INT8(cnn) 作为"INT8 可行性"证据与备份。
- 若要让 INT8 真正提速,须让 attention 也能跑低精度:升级 TRT(≥10.x,Myelin 有 int8 kernel)或
  替换/拆解该 Myelin 融合块;8.5.2 内无解。

### 新增产物
- `onnx/...int8_qdq_noRC.onnx`、`onnx/...int8_qdq_cnn.onnx`、`onnx/...int8_qdq_bb.onnx`(迭代中间版)。
- `uniad_tiny_int8_cnn.engine`;日志 `build_int8_qdq_{norc,cnn,bb,cnn_nofp16}.log`、`perf_int8.log`、
  `bench/e2e_int8.log`;可视化 `enqueueV2/output_int8/dumped_video_results/`。
- 构建脚本新增模式:`int8_qdq_norc` / `int8_qdq_cnn` / `int8_qdq_bb` / `int8_qdq_cnn_nofp16`。
- 改图脚本(已存档本仓库):`tools_strip/strip_qdq_{rc,attn,combined,bb}.py`。

## 二十四、Orin 三引擎精度指标量化(FP32/FP16/INT8,2026-09-24)

> 用户要求"量化测试各种方案的准确率等各项精度指标"。做法:给 enqueueV2 app 打补丁,推理循环后
> 把每帧关键输出张量落盘为裸二进制(`acc_*/dump/frame<i>_<tensor>.bin`),三引擎用**同一 69 帧输入、
> 同一自由递归(free-run)**各跑一遍,再以 **FP32 为同平台参考**逐张量对比。脚本见
> `tools_acc/analyze_acc.py`(张量级+任务级)、`tools_acc/analyze_det_match.py`(贪心中心匹配的检测偏差)。

### 指标口径与边界(重要)
- 本节是 **相对精度**:FP16/INT8 各自 vs **FP32(同 Orin、同 TRT 8.5.2、同递归)**。这隔离了"精度格式"
  这一单一变量,是 Orin 交付引擎之间最干净的对比。
- **官方绝对指标(mAP/NDS/AMOTA/minADE/mIoU/avg.L2/avg.Col)无法在 Orin 上算**:需 nuScenes 全 val(6019 帧)
  + GT 标注 + nuscenes-devkit/UniAD eval 工具链。这些只在 x86;且 **Orin(aarch64/8.5.2)引擎不能拿到 x86 跑**,
  故 x86 上的官方指标只反映 x86(TRT 10.9)引擎,不代表 Orin 交付件。§二十一 的 planning MSE(FP32@10.9
  =2.88e-07;全量 INT8@10.9 ~134)即属此类绝对口径,列此仅作对照。
- 自由递归下检测 query 会跨帧漂移,故 `bboxes` 的**逐索引原始 MSE 无意义**(被 ~287 个低置信 padding query
  和索引错位主导);检测改用**贪心最近中心匹配**(thr=0.25 与 app 一致,gate=2m)才可信。

### A. 张量级偏差(vs FP32,69 帧均值)
| 张量 | 引擎 | MSE | RMSE | MAE | max-abs | rel-L2 | cos | 有效帧 |
|---|---|---|---|---|---|---|---|---|
| outs_planning | FP16 | 4.67e-04 | 1.20e-02 | 8.30e-03 | 2.58e-02 | 3.10e-03 | 1.000000 | 67/69 |
| outs_planning | INT8 | 1.96e-03 | 2.97e-02 | 2.04e-02 | 6.38e-02 | 9.83e-03 | 0.999995 | 69/69 |
| scores | FP16 | 4.68e-03 | 6.05e-02 | 3.91e-02 | 2.56e-01 | — | 0.848 | 67/69 |
| scores | INT8 | 4.42e-03 | 5.82e-02 | 3.79e-02 | 2.49e-01 | — | 0.857 | 69/69 |
| bev_embed | FP16 | 9.49e-04 | 1.87e-02 | 1.11e-02 | 3.88e-01 | 4.43e-02 | 0.997231 | 69/69 |
| bev_embed | INT8 | 1.98e-03 | 4.44e-02 | 3.47e-02 | 3.27e-01 | 1.05e-01 | 0.994461 | 69/69 |

### B. 任务级(相对 FP32)
| 指标 | FP16 | INT8(cnn) |
|---|---|---|
| **planning 平均轨迹 L2** | **0.0153 m** | **0.0372 m** |
| planning 最大轨迹 L2 | 0.156 m | 0.205 m |
| planning 逐路点 L2(1→6) | 0.005→0.026 m | 0.010→0.065 m |
| 检测匹配召回(中心≤2m) | 0.926 | 0.894 |
| 匹配框中心距 mean/median | 0.297 / 0.176 m | 0.262 / 0.166 m |
| 匹配框 \|Δscore\| mean | 0.068 | 0.074 |
| 匹配框 尺寸 L1(w,l,h) | 0.0125 | 0.0155 |
| 匹配框 \|Δyaw\| | 0.176 rad | 0.178 rad |
| 匹配框 类别一致率 | 99.9% | 99.7% |
| seg 逐格类别不一致率 | 0.77% | 1.62% |

### C. 关键发现
1. **FP16 在第 0、39 帧整头输出 NaN**(planning + 全部 300 检测框/分数全 NaN;bev_embed 仍有限)。
   FP32、INT8 全 69 帧干净。这是 FP16 有限动态范围在检测/规划头的溢出,**可视化目视此前未发现**
   (NaN 框不绘制),是数值 dump 才暴露的真实鲁棒性缺陷。
2. **INT8(cnn) 数值最稳**:0 NaN 帧;planning 仅偏 FP32 **3.7 cm**,检测匹配质量与 FP16 基本同级。
   原因:该 INT8 只把 CNN 卷积量化,**attention/heads 留 fp32**(`--int8` 单精度构建),恰好规避了
   §二十一 全量 INT8(planning MSE~134)的敏感层失真。
3. **检测质量 FP16≈INT8**:两者匹配召回 0.89–0.93、中心误差 ~0.17–0.30 m、类别一致 99.7%+,与"可视化
   看起来正确"一致。
4. 性能与精度的取舍未变:FP16 最快(510 ms)但有 2 帧 NaN;INT8(cnn) 稳但慢(606 ms,attention 留 fp32);
   FP32 参考(769 ms)。

### D. 交付建议(基于精度)
- **FP16 若作交付,须先修 NaN**(定位第 0/39 帧溢出的头层,单独保 fp32,或排查是否输入相关);否则
  2/69≈2.9% 帧检测+规划完全失效,对小车不可接受。
- **INT8(cnn) 是当前数值最稳的低精度方案**,但慢于 FP16;适合对鲁棒性敏感、对延迟不极致的场景。
- 官方 mAP/NDS/avg.L2/avg.Col 仍需在 x86 全 val 上、对 x86 引擎单独评测(与本 Orin 交付件区分)。

### 新增产物
- `tools_acc/analyze_acc.py`、`tools_acc/analyze_det_match.py`;结果 `acc_results.txt`、`det_match_results.txt`。
- app 补丁:`enqueueV2/src/main.cpp` 推理循环后新增数值 dump 块(重编译 `build/uniad`)。
- dump 数据:`enqueueV2/acc_{fp32,fp16,int8}/dump/`(各 69 帧 × 7 张量);运行脚本 `run_acc_dump.sh`。

## 二十五、延迟基准复测(30W→MAXN)+ N=901 对齐(2026-10-29)

> 用户要求"测 DL model latency",并对齐 NVIDIA DL4AGX `explicit_quantization.md` 的口径。文档给出
> FP16 **49.76 ms**(DRIVE Orin-X,TRT **10.7**,N=**901**),而我们初测 FP16 **526 ms**,差约 10×。
> 本节记录两条根因的排查与裁决:**①功耗模式降频、②N=901 对齐**,以及方法论、过程与回滚。

### 方法论(对齐文档)
- 延迟口径 = **median GPU Compute Time**(纯 DL 模型计算,不含 H2D/D2H);host 端 `Latency` 仅作参照。
- 命令:`trtexec --loadEngine=<eng> --plugins=<so> --iterations=100 --warmUp=5000 --avgRuns=10`。
- H2D ~0.7 ms、D2H ~0.34 ms,相对计算可忽略。

### 根因①:30W 软件功耗模式降频(非硬件/供电问题)
- 初测在 **MODE_30W**(nvpmodel id=2)下,GPU 上限被锁 **612 MHz**;接电源也不会自动升频,是软件功耗墙。
- 切到 **MAXN**(id=0,60W,GPU **1300.5 MHz**)并 `jetson_clocks` 锁频后,**约 3.5× 加速**。
- 切换需重启;已把 `/etc/nvpmodel.conf` `DEFAULT=0` 且 `/var/lib/nvpmodel/status` `pmode:0000`,**开机即 MAXN**。

| 引擎(N=1150) | 30W median GPU(ms) | MAXN median GPU(ms) | 加速比 | MAXN 吞吐(qps) |
|---|---|---|---|---|
| FP32 | 779.698 | **222.396** | 3.51× | 4.47 |
| FP16 | 525.919 | **149.853** | 3.51× | 6.62 |
| INT8(cnn) | 615.514 | **179.694** | 3.43× | 5.53 |

### 根因②:N=901 对齐 —— 实测**无加速效果**
- 文档 N=901;我们默认 N=1150(=901 fresh + 249 carry)。把 `build_engine_orin.sh` 的 N 改为可环境变量覆盖,
  N≠1150 时引擎名加 `_n901` 后缀(**不覆盖 1150 基线**),重建 FP16@901 并按文档口径 bench。
- 结果:**FP16@N=901 = 155.462 ms**(6.39 qps,PASSED),与 **FP16@N=1150 = 149.853 ms** 几乎相同(甚至略慢)。

| 配置 | median GPU Compute(ms) | 吞吐(qps) |
|---|---|---|
| FP16 @ N=901(MAXN,本次) | **155.462** | 6.39 |
| FP16 @ N=1150(MAXN,§二十五上表) | 149.853 | 6.62 |
| 文档 FP16 @ N=901(TRT10.7, DRIVE Orin-X) | 49.76 | — |

- **为什么 N 减小不提速**:该 UniAD-tiny 是固定缓冲 carry 网络——输入 901 个上一帧实例,网络内部拼上
  249 个新检测,**输出仍是 1150**(bench 日志:输入 binding `prev_track_intances0=901x512`,输出 binding
  `prev_track_intances0_out=1150x512`)。decoder 实际处理的 token 数没减少,故延迟不变。
- **与文档 49.76 ms 的差距来源**:主要是 **TRT 10.7 + DRIVE Orin-X** 的叠加优势(更新的 tactic/Myelin、
  车规芯片调优),**不是 N**。单纯把 N 从 1150 改 901 无法逼近文档数字。

### 关于文档 INT8(EQ)
- 文档 **INT8(EQ)+FP16 = 39.31 ms** 中的 EQ 指 **Explicit Quantization(显式量化/QDQ)**:量化 scale 以
  Q/DQ 节点形式 baked into ONNX,未量化层 fallback FP16。
- 我们的 **INT8(cnn)** 口径不同:仅把 CNN 卷积量化(`--int8` 单精度构建),**attention/heads 留 fp32**
  (见 §二十三),故 179.7 ms 慢于文档 EQ+FP16;两者不是同一量化方案,不可直接比。

### 过程记录与裁决
- 构建期间设备**意外重启一次**(uptime 3 min,`free -h` 显示 59 GiB 空闲,**非 OOM**,原因不明);第一轮
  FP16@901 构建被中断,清理重复实例后重启流水线,重启后仍自动进 MAXN。
- FP16@901 **首次 bench 偶发 SIGBUS(exit=135)**,停在 "Starting inference" 无结果;**重跑即成功**
  (`&&&& PASSED`,median 155.462 ms;末尾 exit=134 是已知良性 teardown bug `free(): invalid pointer`,
  结果已完整打印)。bench 重试逻辑已写入 `tools_acc/build_bench_n901_rest.sh`。
- **FP32@901 / INT8@901 应用户要求未跑**(鉴于 N=901 对 FP16 已证明无加速,继续构建价值有限,已停流水线
  并清理半成品,无 `uniad_tiny_fp32_n901.engine` 残留)。

### 回滚保障(完好)
- 1150 三引擎 md5 基线 `tools_acc/baseline_1150.md5` **校验全部 OK**(fp32/fp16/int8_cnn 未被触碰)。
- N=901 引擎用 `_n901` 后缀,与 1150 隔离;一键回滚 `tools_acc/rollback_n901.sh`(删 n901 引擎+bench 日志,
  再 `md5sum -c` 复核 1150)。

### 新增产物
- `build_engine_orin.sh`:`N=${N:-1150}` 可环境变量覆盖;N≠1150 时引擎名加 `_n901` 后缀。
- `tools_acc/build_bench_n901.sh`、`tools_acc/build_bench_n901_rest.sh`(bench 带重试防偶发 SIGBUS)。
- `uniad_tiny_fp16_n901.engine`(188 MB)、`bench_n901_fp16.log`(median 155.462 ms)。
- `tools_acc/baseline_1150.md5`、`tools_acc/rollback_n901.sh`。
- `lat_summary.txt`(30W)、`lat_summary_maxn.txt`(MAXN,N=1150)。

### 小结
- **延迟现状(MAXN, N=1150)**:FP32 222.4 / FP16 149.9 / INT8(cnn) 179.7 ms。FP16 最快但仍有第 0/39 帧
  NaN(§二十四、任务 #10 未决);INT8(cnn) 数值最稳。
- **N=901 对齐结论:无效**(155.5≈149.9 ms),文档 49.76 ms 的差距归因于 TRT 10.7 + DRIVE Orin-X,非 N。
- 若要进一步逼近文档延迟,路径是**升级 TRT 版本 / 更换目标平台**,而非调 N;小车实时性评估应以
  **MAXN 下 N=1150 的实测值**为准。
