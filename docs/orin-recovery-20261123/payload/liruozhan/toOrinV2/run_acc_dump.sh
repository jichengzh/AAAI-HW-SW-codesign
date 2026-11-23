#!/bin/bash
set -x
cd /home/jichengzhi/liruozhan/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
IN=/home/jichengzhi/liruozhan/toOrinV2/uniad_trt_input
for pair in "uniad_tiny_fp32.engine:acc_fp32" "uniad_tiny_fp16.engine:acc_fp16" "uniad_tiny_int8_cnn.engine:acc_int8"; do
  ENG=${pair%%:*}; OUT=${pair##*:}
  echo "===== RUN $ENG -> $OUT ====="; date +%H:%M:%S
  ./build/uniad /home/jichengzhi/liruozhan/toOrinV2/$ENG ./build/libuniad_plugin.so $IN ./$OUT 69
  echo "===== DONE $ENG exit=$? ====="; date +%H:%M:%S
done
echo "ALL_DUMP_DONE"
