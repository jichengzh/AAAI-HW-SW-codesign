#!/usr/bin/env bash
set -euo pipefail
set +x

: "${ORIN_SUDO_PW:?ORIN_SUDO_PW must be supplied through the process environment}"

ROOT=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
PY=/home/jichengzhi/uniad_venv/bin/python
RUNNER="$ROOT/code/lane_c_backbone_parity_runner.py"
NATIVE="$ROOT/code/lane_c_native_backbone_orin.py"
HEAL=/home/jichengzhi/lane_c_orin_backbone_parity_20260723/code/HEAL
V2XVERSE="$ROOT/code/V2Xverse"

sudo -v

validate_energy_report() {
  report="$1"
  "$PY" - "$report" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
power = report.get("power_measurement", {})
if power.get("measurement_status") != "available":
    raise SystemExit("power measurement is not available")
if power.get("energy_rail") != "VIN_SYS_5V0":
    raise SystemExit("unexpected Orin energy rail")
if not power.get("used_sudo"):
    raise SystemExit("report does not bind privileged tegrastats")
if float(power.get("energy_j", 0.0)) <= 0.0:
    raise SystemExit("energy_j must be positive")
PY
}

measure_trt() {
  model="$1"
  result_arm="$2"
  engine="$3"
  input_shape="$4"
  output_channels="$5"
  input_file="$6"
  artifact="$ROOT/$model/$result_arm"
  mkdir -p "$artifact"
  test -f "$engine"
  "$PY" "$RUNNER" measure-primary \
    --artifact-root "$artifact" \
    --engine "$engine" \
    --inputs-npy "$ROOT/contracts/$input_file" \
    --expected-input-shape "$input_shape" \
    --expected-output-channels "$output_channels" \
    --scope-model "$model" \
    --warmup 20 --iters 300 --repeat 5 \
    --power-log "$artifact/primary_tegrastats_sudo.log" \
    --output-json "$artifact/primary_latency_power_sudo.json" \
    >"$artifact/primary_measure_sudo.log" 2>&1
  validate_energy_report "$artifact/primary_latency_power_sudo.json"
  printf 'PASS %s/%s\n' "$model" "$result_arm"
}

native_one() {
  model="$1"
  repo="$2"
  width="$3"
  checkpoint="$4"
  input_file="$5"
  artifact="$ROOT/$model/original_default"
  "$PY" "$NATIVE" \
    --model "$model" \
    --repo-root "$repo" \
    --model-dir "$ROOT/models/$model/$width" \
    --checkpoint "$ROOT/models/$model/$width/$checkpoint" \
    --inputs-npy "$ROOT/contracts/$input_file" \
    --warmup 20 --iters 300 --repeat 5 \
    --power-log "$artifact/primary_tegrastats_sudo.log" \
    --output-json "$artifact/primary_latency_power_sudo.json" \
    >"$artifact/primary_measure_sudo.log" 2>&1
  validate_energy_report "$artifact/primary_latency_power_sudo.json"
  printf 'PASS %s/original_default\n' "$model"
}

native_one \
  pyramid "$HEAL" 064x128x256 net_epoch_bestval_at23.pth \
  pyramid_latency_input.npy
measure_trt \
  pyramid compression_only \
  "$ROOT/pyramid/compression_only/backbone.engine" \
  2,64,128,256 24,128,64 pyramid_latency_input.npy
measure_trt \
  pyramid schedule_only \
  "$ROOT/pyramid/schedule_only/backbone.engine" \
  2,64,128,256 64,128,256 pyramid_latency_input.npy
measure_trt \
  pyramid compress_then_tune \
  "$ROOT/pyramid/compress_then_tune/backbone.engine" \
  2,64,128,256 32,48,128 pyramid_latency_input.npy
measure_trt \
  pyramid joint_shcosearch_fp16 \
  /home/jichengzhi/lane_c_calibration_locked_attribution_20260724/fp16/backbone_fp16.engine \
  2,64,128,256 16,32,64 pyramid_latency_input.npy

native_one \
  codriving "$V2XVERSE" 064x128x256 net_epoch29.pth \
  codriving_latency_input.npy
measure_trt \
  codriving compression_only \
  "$ROOT/codriving/compression_only/backbone.engine" \
  2,64,256,512 32,64,96 codriving_latency_input.npy
measure_trt \
  codriving schedule_only \
  "$ROOT/codriving/schedule_only/backbone.engine" \
  2,64,256,512 64,128,256 codriving_latency_input.npy
measure_trt \
  codriving compress_then_tune \
  "$ROOT/codriving/compress_then_tune/backbone.engine" \
  2,64,256,512 32,80,128 codriving_latency_input.npy
measure_trt \
  codriving joint_shcosearch_fp16 \
  "$ROOT/codriving/joint_shcosearch_fp16/backbone.engine" \
  2,64,256,512 16,32,64 codriving_latency_input.npy
