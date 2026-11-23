#!/bin/bash
# 启动 UniAD FP16 推理（默认121帧=原240帧集的119-239段，4场景），UDP 发送 planning 到 127.0.0.1:9876
# 可覆盖：INPUT=/path/to/input_dir bash run_uniad_udp.sh [帧数]
# 注意：必须在 enqueueV2 目录下运行（info.txt 图片为相对路径 data/nuscenes/...）
set -u
V2=/home/jichengzhi/liruozhan/toOrinV2
APP=$V2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
INPUT=${INPUT:-$V2/uniad_trt_input_119plus}
FRAMES=${1:-121}
cd "$APP"
UNIAD_SKIP_VIS=1 ./build/uniad_udp \
  "$V2/uniad_tiny_fp16.engine" ./build/libuniad_plugin.so \
  "$INPUT" "$V2/output_udp" "$FRAMES"
