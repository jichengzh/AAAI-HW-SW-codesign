#!/bin/bash
# N=901 剩余两个引擎(fp32 / int8_cnn)的构建 + 文档口径 bench。
# fp16@901 已完成(median 155.462ms),此处跳过。
# bench 带重试:偶发 SIGBUS(exit=135,设备重启后 GPU 不稳)时自动重跑,直到拿到 GPU Compute Time。
set -u
TRT=/usr/src/tensorrt
PLUGIN=./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so
for mode in fp32 int8_qdq_cnn_nofp16; do
  case $mode in
    fp32) ENG=uniad_tiny_fp32_n901.engine;;
    int8_qdq_cnn_nofp16) ENG=uniad_tiny_int8_cnn_n901.engine;;
  esac
  if [ ! -f "$ENG" ]; then
    echo "##### BUILD N=901 mode=$mode ##### $(date)"
    N=901 bash build_engine_orin.sh $mode
  else
    echo "##### $ENG 已存在,跳过 build #####"
  fi
  if [ -f "$ENG" ]; then
    for attempt in 1 2 3 4; do
      echo "##### BENCH $ENG attempt=$attempt ##### $(date)"
      LD_LIBRARY_PATH=$TRT/lib:${LD_LIBRARY_PATH:-} \
      $TRT/bin/trtexec --loadEngine=./$ENG --plugins=$PLUGIN \
        --iterations=100 --warmUp=5000 --avgRuns=10 > bench_n901_${mode}.log 2>&1
      if grep -aq "GPU Compute Time:" bench_n901_${mode}.log; then
        echo "bench OK attempt=$attempt -> bench_n901_${mode}.log"
        grep -aE "Throughput:|GPU Compute Time:" bench_n901_${mode}.log
        break
      fi
      echo "bench attempt=$attempt 无结果(可能SIGBUS),3s后重试"
      sleep 3
    done
  else
    echo "##### $ENG 未构建,跳过 bench #####"
  fi
done
echo "ALL REST DONE $(date)"
