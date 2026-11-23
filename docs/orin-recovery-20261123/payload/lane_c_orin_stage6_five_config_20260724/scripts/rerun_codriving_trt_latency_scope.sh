#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
PY=/home/jichengzhi/uniad_venv/bin/python
RUNNER="$ROOT/code/lane_c_backbone_parity_runner.py"
INPUT="$ROOT/contracts/codriving_latency_input.npy"

measure() {
  arm="$1"
  channels="$2"
  artifact="$ROOT/codriving/$arm"
  stale="$artifact/stale_pyramid_scope_latency"
  mkdir -p "$stale"
  for name in primary_latency_power.json primary_measure.log primary_tegrastats.log; do
    if [ -f "$artifact/$name" ]; then
      cp -p "$artifact/$name" "$stale/$name"
    fi
  done
  "$PY" "$RUNNER" measure-primary \
    --artifact-root "$artifact" \
    --engine "$artifact/backbone.engine" \
    --inputs-npy "$INPUT" \
    --expected-input-shape 2,64,256,512 \
    --expected-output-channels "$channels" \
    --scope-model codriving \
    --warmup 20 --iters 300 --repeat 5 \
    --power-log "$artifact/primary_tegrastats.log" \
    --output-json "$artifact/primary_latency_power.json" \
    --allow-missing-power-rails \
    >"$artifact/primary_measure.log" 2>&1
  "$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["scope"].startswith("codriving_backbone_resnet_"); assert x["sample_count"]==1500' "$artifact/primary_latency_power.json"
  printf 'PASS codriving/%s\n' "$arm"
}

measure compression_only 32,64,96
measure schedule_only 64,128,256
measure compress_then_tune 32,80,128
measure joint_shcosearch 16,32,64
