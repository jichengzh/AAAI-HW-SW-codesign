#!/bin/bash
# FP16 engine with all numerically-fragile elementwise/reduction ops
# (Log/Div/Sub/Clip/Sqrt/Pow/Exp/Reciprocal/ReduceMean/ReduceSum/ReduceMax/Neg)
# forced to fp32 to eliminate the frame-0/39 head NaN. Conv/MatMul/Softmax stay fp16.
set -u
TRT_PATH=${TRT_PATH:-/usr/src/tensorrt}
PLUGIN=./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so
ONNX=./onnx/uniad_tiny_imgx0.25_cp.noaten_simp.onnx
ENGINE=uniad_tiny_fp16_fixnan.engine
N=1150
S="prev_track_intances0:${N}x512,prev_track_intances1:${N}x3,prev_track_intances3:${N},prev_track_intances4:${N},prev_track_intances5:${N},prev_track_intances6:${N},prev_track_intances8:${N},prev_track_intances9:${N}x10,prev_track_intances11:${N}x4x256,prev_track_intances12:${N}x4,prev_track_intances13:${N}"
LP=$(cat tools_acc/layerprecisions_fp32.txt)
CONSTRAINTS=${CONSTRAINTS:-obey}
LD_LIBRARY_PATH=$TRT_PATH/lib:${LD_LIBRARY_PATH:-} \
$TRT_PATH/bin/trtexec \
  --onnx=$ONNX --saveEngine=$ENGINE --plugins=$PLUGIN \
  --minShapes=$S --optShapes=$S --maxShapes=$S \
  --buildOnly --verbose --fp16 \
  --precisionConstraints=$CONSTRAINTS \
  --layerPrecisions="$LP"
echo "EXIT=$? engine=$ENGINE constraints=$CONSTRAINTS"
