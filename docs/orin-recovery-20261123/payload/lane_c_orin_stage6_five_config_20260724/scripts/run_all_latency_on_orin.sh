#!/usr/bin/env bash
set -u

ROOT=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
PY=/home/jichengzhi/uniad_venv/bin/python
RUNNER="$ROOT/code/lane_c_backbone_parity_runner.py"
NATIVE="$ROOT/code/lane_c_native_backbone_orin.py"
HEAL=/home/jichengzhi/lane_c_orin_backbone_parity_20260723/code/HEAL
V2XVERSE="$ROOT/code/V2Xverse"

while kill -0 627458 2>/dev/null; do
  sleep 10
done

"$PY" -c 'import numpy as np; a=np.load("'"$ROOT"'/calibration/pyramid_joint/batch2_014.npy",allow_pickle=False); np.save("'"$ROOT"'/contracts/pyramid_latency_input.npy",a[None])'
"$PY" -c 'import numpy as np; a=np.load("'"$ROOT"'/calibration/codriving_joint/batch2_015.npy",allow_pickle=False); np.save("'"$ROOT"'/contracts/codriving_latency_input.npy",a[None])'

measure_trt() {
  model="$1"
  arm="$2"
  input_shape="$3"
  output_channels="$4"
  input_file="$5"
  artifact="$ROOT/$model/$arm"
  if [ ! -f "$artifact/backbone.engine" ]; then
    printf 'SKIP %s/%s missing_engine\n' "$model" "$arm"
    return 1
  fi
  "$PY" "$RUNNER" measure-primary \
    --artifact-root "$artifact" \
    --engine "$artifact/backbone.engine" \
    --inputs-npy "$ROOT/contracts/$input_file" \
    --expected-input-shape "$input_shape" \
    --expected-output-channels "$output_channels" \
    --warmup 20 --iters 300 --repeat 5 \
    --power-log "$artifact/primary_tegrastats.log" \
    --output-json "$artifact/primary_latency_power.json" \
    --allow-missing-power-rails \
    >"$artifact/primary_measure.log" 2>&1
}

run_trt() {
  model="$1"
  arm="$2"
  shift 2
  if measure_trt "$model" "$arm" "$@"; then
    printf 'PASS %s/%s\n' "$model" "$arm"
  else
    printf 'FAIL %s/%s\n' "$model" "$arm"
  fi
}

run_trt pyramid compression_only 2,64,128,256 24,128,64 pyramid_latency_input.npy
run_trt pyramid schedule_only 2,64,128,256 64,128,256 pyramid_latency_input.npy
run_trt pyramid compress_then_tune 2,64,128,256 32,48,128 pyramid_latency_input.npy
run_trt pyramid joint_shcosearch 2,64,128,256 16,32,64 pyramid_latency_input.npy
run_trt codriving compression_only 2,64,256,512 32,64,96 codriving_latency_input.npy
run_trt codriving schedule_only 2,64,256,512 64,128,256 codriving_latency_input.npy
run_trt codriving compress_then_tune 2,64,256,512 32,80,128 codriving_latency_input.npy
run_trt codriving joint_shcosearch 2,64,256,512 16,32,64 codriving_latency_input.npy

native_one() {
  model="$1"
  repo="$2"
  width="$3"
  checkpoint="$4"
  input_file="$5"
  artifact="$ROOT/$model/original_default"
  mkdir -p "$artifact"
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

if native_one pyramid "$HEAL" 064x128x256 net_epoch_bestval_at23.pth pyramid_latency_input.npy; then
  printf 'PASS pyramid/original_default\n'
else
  printf 'FAIL pyramid/original_default\n'
fi
if native_one codriving "$V2XVERSE" 064x128x256 net_epoch29.pth codriving_latency_input.npy; then
  printf 'PASS codriving/original_default\n'
else
  printf 'FAIL codriving/original_default\n'
fi
