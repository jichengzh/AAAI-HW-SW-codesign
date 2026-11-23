#!/bin/bash
# S8: ModelOpt INT8 QDQ quantization of the FIXED-BUFFER UniAD-tiny ONNX (N=1150).
# Input graph: onnx/uniad_tiny_imgx0.25_cp.noaten.onnx  (NonZero=0, fully static-shaped,
#   produced by the S1-S6 fixed-buffer refactor + unflatten->Reshape rewrite).
# Calib: calib_data_fixed_1150.npz (frame-5 real state, 1 active track + 1149 padding).
# Stack: uniad_quant (py3.11) + ModelOpt 0.29 + ORT-gpu 1.21 cu12 + TensorRT 10.9 cu12.
set -u

ROOT=/home/liruozhan/DL4AGX/AV-Solutions/uniad-trt
UNIAD=$ROOT/UniAD
PY=/home/liruozhan/miniconda3/envs/uniad_quant/bin/python

CUDNN_LIB=/home/liruozhan/miniconda3/envs/uniad2.0/lib/python3.9/site-packages/nvidia/cudnn/lib
TRT_LIB=$ROOT/TensorRT-10.9_x86_cu128/lib
CUDA_LIB=/usr/local/cuda-12.2/lib64

export LD_LIBRARY_PATH=$CUDNN_LIB:$TRT_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}
export TRT_PLUGIN_SO=$UNIAD/plugins/TensorRT/build_cu12b/libtensorrt_ops.so
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

N=1150
SHAPES="prev_track_intances0:${N}x512,prev_track_intances1:${N}x3,prev_track_intances3:${N},prev_track_intances4:${N},prev_track_intances5:${N},prev_track_intances6:${N},prev_track_intances8:${N},prev_track_intances9:${N}x10,prev_track_intances11:${N}x4x256,prev_track_intances12:${N}x4,prev_track_intances13:${N},prev_timestamp:1,prev_l2g_r_mat:1x3x3,prev_l2g_t:1x3,prev_bev:2500x1x256,timestamp:1,l2g_r_mat:1x3x3,l2g_t:1x3,img:1x6x3x256x416,img_metas_can_bus:18,img_metas_lidar2img:1x6x4x4,command:1,use_prev_bev:1,max_obj_id:1"

cd $UNIAD
$PY quantize_patch_runner.py \
  --onnx_path=$UNIAD/onnx/uniad_tiny_imgx0.25_cp.noaten.onnx \
  --trt_plugins=$TRT_PLUGIN_SO \
  --calibration_eps trt cuda:0 cpu \
  --calibration_shapes=$SHAPES \
  --simplify \
  --op_types_to_exclude MatMul \
  --calibration_data_path=$UNIAD/calib_data_fixed_1150.npz \
  --output_path=$UNIAD/onnx/uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx
echo "PYTHON_EXIT=$?"
