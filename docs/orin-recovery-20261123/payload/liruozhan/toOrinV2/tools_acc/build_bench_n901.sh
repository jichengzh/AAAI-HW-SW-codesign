#!/bin/bash
# 对齐 DL4AGX 文档配置: N=901 静态重建 + 文档口径延迟复测(median GPU Compute, iter=100).
set -u
TRT=/usr/src/tensorrt
PLUGIN=./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so
for mode in fp16 fp32 int8_qdq_cnn_nofp16; do
  echo "##### BUILD N=900+1 mode=$mode #####"
  N=901 bash build_engine_orin.sh $mode
  case $mode in
    fp16) ENG=uniad_tiny_fp16_n901.engine;;
    fp32) ENG=uniad_tiny_fp32_n901.engine;;
    int8_qdq_cnn_nofp16) ENG=uniad_tiny_int8_cnn_n901.engine;;
  esac
  if [ -f "$ENG" ]; then
    echo "##### BENCH $ENG #####"
    LD_LIBRARY_PATH=$TRT/lib:${LD_LIBRARY_PATH:-} \
    $TRT/bin/trtexec --loadEngine=./$ENG --plugins=$PLUGIN \
      --iterations=100 --warmUp=5000 --avgRuns=10 > bench_n901_${mode}.log 2>&1
    echo "bench exit=$? -> bench_n901_${mode}.log"
  else
    echo "##### $ENG NOT BUILT, skip bench #####"
  fi
done
echo "ALL N901 DONE"
