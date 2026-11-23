#!/usr/bin/env bash
set -u

ROOT=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
PY=/home/jichengzhi/uniad_venv/bin/python
NATIVE="$ROOT/code/lane_c_native_backbone_orin.py"
HEAL=/home/jichengzhi/lane_c_orin_backbone_parity_20260723/code/HEAL
V2XVERSE="$ROOT/code/V2Xverse"

run_one() {
  model="$1"
  repo="$2"
  width="$3"
  checkpoint="$4"
  input_file="$5"
  artifact="$ROOT/$model/original_default"
  failed="$artifact/failed_attempts/input_shape_gate_1"
  mkdir -p "$failed"
  for name in primary_measure.log primary_tegrastats.log; do
    if [ -f "$artifact/$name" ]; then
      mv "$artifact/$name" "$failed/$name"
    fi
  done
  "$PY" "$NATIVE" \
    --model "$model" \
    --repo-root "$repo" \
    --model-dir "$ROOT/models/$model/$width" \
    --checkpoint "$ROOT/models/$model/$width/$checkpoint" \
    --inputs-npy "$ROOT/contracts/$input_file" \
    --warmup 20 --iters 300 --repeat 5 \
    --power-log "$artifact/primary_tegrastats.log" \
    --output-json "$artifact/primary_latency_power.json" \
    >"$artifact/primary_measure.log" 2>&1
}

if run_one pyramid "$HEAL" 064x128x256 net_epoch_bestval_at23.pth pyramid_latency_input.npy; then
  printf 'PASS pyramid/original_default\n'
else
  printf 'FAIL pyramid/original_default\n'
fi
if run_one codriving "$V2XVERSE" 064x128x256 net_epoch29.pth codriving_latency_input.npy; then
  printf 'PASS codriving/original_default\n'
else
  printf 'FAIL codriving/original_default\n'
fi
