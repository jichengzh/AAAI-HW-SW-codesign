#!/bin/bash
# 在 Orin 上为「固定 buffer 全静态」UniAD-tiny 构建 TensorRT 引擎。
# 关键变化(vs toOrin v1):
#   - 模型已是固定形状 N=1150(901 fresh + 249 carry),图中无 DDS(NonZero=0)、无动态 TopK、无 If。
#     因此 min=opt=max=1150 单一静态 profile 即可,不再需要 901..1150 动态范围。
#   - 实机经核实为 TensorRT 8.5.2.2(非 8.6.13)。8.5.2 的 trtexec 不支持 --skipInference/--staticPlugins,
#     改用 --buildOnly / --plugins。
#   - x86 已验证:FP32 静态图在公版 TRT 8.6 上 &&&& PASSED(老动态图在此处死于 checkSanity/动态 TopK)。
#     INT8 在 TRT 10.9 上 PASSED;INT8(--dq_only) 在 8.6 触发 Myelin INT8-weight 断言 →
#     本包提供 int8_qdq(标准 QDQ,权重 FP+Q/DQ)规避该断言,见 README 第 4 节。
#
# 用法:
#   bash build_engine_orin.sh fp32        # FP32 基线(先验证可建)
#   bash build_engine_orin.sh fp16        # FP16(安全提速,无 INT8 优化器风险)← INT8 建不出时的退路
#   bash build_engine_orin.sh int8        # INT8(dq_only)+FP16
#   bash build_engine_orin.sh int8_qdq    # INT8(标准 QDQ,Resize 不量化)+FP16  ← INT8 首选
#   bash build_engine_orin.sh int8_qdq_norc  # INT8(再移除 Reshape/Cast 后 QDQ)+FP16 ← 8.5.2 可建版
#
# 可用环境变量覆盖:TRT_PATH / PLUGIN / ONNX_DIR / TRTEXEC_EXTRA
set -u

TRT_PATH=${TRT_PATH:-/usr/src/tensorrt}
PLUGIN=${PLUGIN:-./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so}
ONNX_DIR=${ONNX_DIR:-./onnx}
MODE=${1:-fp32}

# 全静态:单一 profile。N 可用环境变量覆盖(默认 1150=901 fresh+249 carry;N=901 为无 carry 配置,对齐 DL4AGX 文档)。
N=${N:-1150}
S="prev_track_intances0:${N}x512,prev_track_intances1:${N}x3,prev_track_intances3:${N},prev_track_intances4:${N},prev_track_intances5:${N},prev_track_intances6:${N},prev_track_intances8:${N},prev_track_intances9:${N}x10,prev_track_intances11:${N}x4x256,prev_track_intances12:${N}x4,prev_track_intances13:${N}"

case "$MODE" in
  fp32)
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten_simp.onnx
    ENGINE=uniad_tiny_fp32.engine; EXTRA="" ;;
  fp16)
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten_simp.onnx
    ENGINE=uniad_tiny_fp16.engine; EXTRA="--fp16" ;;
  int8)
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten.int8.onnx
    ENGINE=uniad_tiny_int8_fp16.engine; EXTRA="--int8 --fp16" ;;
  int8_qdq)
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx
    ENGINE=uniad_tiny_int8qdq_fp16.engine; EXTRA="--int8 --fp16" ;;
  int8_qdq_norc)
    # noRC = 移除 Reshape/Cast 之后的 Q/DQ 对(8.5.2 QDQToCopy 跨 phylum 断言的触发类),见 QUANTIZE_PROGRESS §二十二
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_noRC.onnx
    ENGINE=uniad_tiny_int8_norc_fp16.engine; EXTRA="--int8 --fp16" ;;
  int8_qdq_cnn)
    # cnn = noRC + 再移除 attention/transformer 区域 Q/DQ(8.5.2 Myelin 无 INT8 kernel),INT8 仅留 CNN
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_cnn.onnx
    ENGINE=uniad_tiny_int8_cnn_fp16.engine; EXTRA="--int8 --fp16" ;;
  int8_qdq_bb)
    # bb = INT8 仅留 img_backbone 卷积;neck 及之后全 fp16,保证 Myelin 块输入为 fp16(8.5.2 可建)
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_bb.onnx
    ENGINE=uniad_tiny_int8_bb_fp16.engine; EXTRA="--int8 --fp16" ;;
  int8_qdq_cnn_nofp16)
    # 关键:--int8 单精度(不加 --fp16)。--int8 --fp16 会让 Myelin attention 块内出现 f32/f16 混杂
    # (verify_output_type f32 vs f16) 而无 kernel;全 f32 的 Myelin 块在 FP32 构建已验证可建。
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_cnn.onnx
    ENGINE=uniad_tiny_int8_cnn.engine; EXTRA="--int8" ;;
  int8_qdq_cnn_fp16u)
    # 把 Bug2 的 f32 残留强制回 fp16:逐层输出类型约束 → CNN=int8 且 attention=fp16(目标:快于纯 fp16)
    ONNX=$ONNX_DIR/uniad_tiny_imgx0.25_cp.noaten.int8_qdq_cnn.onnx
    ENGINE=uniad_tiny_int8_cnn_fp16u.engine
    EXTRA="--int8 --fp16 --precisionConstraints=obey --layerOutputTypes=/Unsqueeze_124:fp16" ;;
  *) echo "unknown mode: $MODE (fp32|fp16|int8|int8_qdq)"; exit 2 ;;
esac

# N!=1150 时给引擎名加后缀,避免覆盖默认 1150 引擎。
if [ "$N" != "1150" ]; then ENGINE="${ENGINE%.engine}_n${N}.engine"; fi

# 8.5.2 兼容:--plugins(非 --staticPlugins)、--buildOnly(非 --skipInference)。
# 若实机是 8.6.x 也接受这两个旧旗标的新名,无需改。
LD_LIBRARY_PATH=$TRT_PATH/lib:${LD_LIBRARY_PATH:-} \
$TRT_PATH/bin/trtexec \
  --onnx=$ONNX \
  --saveEngine=$ENGINE \
  --plugins=$PLUGIN \
  --minShapes=$S --optShapes=$S --maxShapes=$S \
  --buildOnly \
  --verbose \
  $EXTRA ${TRTEXEC_EXTRA:-} 2>&1 | tee build_${MODE}.log
echo "EXIT=${PIPESTATUS[0]}  engine=$ENGINE  log=build_${MODE}.log"
