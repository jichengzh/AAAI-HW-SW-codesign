#!/bin/bash
# DL model latency benchmark: pure-GPU inference per engine via trtexec.
set -u
TRT=/usr/src/tensorrt
PLUGIN=./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so
ITER=${ITER:-60}
WARM=${WARM:-5000}
for eng in uniad_tiny_fp32 uniad_tiny_fp16 uniad_tiny_int8_cnn; do
  LOG=lat_${eng}.log
  echo "=== $eng ==="
  LD_LIBRARY_PATH=$TRT/lib:${LD_LIBRARY_PATH:-} \
  $TRT/bin/trtexec --loadEngine=./${eng}.engine --plugins=$PLUGIN \
    --iterations=$ITER --warmUp=$WARM --avgRuns=10 \
    > $LOG 2>&1
  echo "exit=$? -> $LOG"
done
echo "ALL DONE"
