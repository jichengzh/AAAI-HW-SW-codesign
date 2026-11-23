#!/usr/bin/env bash
set -u

ROOT=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
PY=/home/jichengzhi/uniad_venv/bin/python
RUNNER="$ROOT/code/lane_c_backbone_parity_runner.py"

build_one() {
  model="$1"
  arm="$2"
  source_dir="$3"
  onnx_name="$4"
  precision="$5"
  input_shape="$6"
  output_channels="$7"
  calibration_dir="${8:-}"
  calibration_count="${9:-15}"
  artifact_dir="$ROOT/$model/$arm"
  mkdir -p "$artifact_dir"
  command=(
    "$PY" "$RUNNER" build
    --artifact-root "$artifact_dir"
    --onnx "$ROOT/source/$source_dir/$onnx_name"
    --engine "$artifact_dir/backbone.engine"
    --precision "$precision"
    --expected-input-shape "$input_shape"
    --expected-output-channels "$output_channels"
    --inspector-json "$artifact_dir/engine_inspector.json"
    --artifact-json "$artifact_dir/build_report.json"
  )
  if [ "$precision" = int8 ]; then
    command+=(
      --expected-calibration-count "$calibration_count"
      --calibration-dir "$ROOT/calibration/$calibration_dir"
      --calibration-manifest "$ROOT/calibration/$calibration_dir/manifest.json"
      --calibration-cache "$artifact_dir/calibration.cache"
    )
  fi
  {
    printf 'START %s %s %s\n' "$model" "$arm" "$(date --iso-8601=seconds)"
    "${command[@]}"
    status=$?
    printf 'END %s %s status=%s %s\n' \
      "$model" "$arm" "$status" "$(date --iso-8601=seconds)"
    return "$status"
  } >"$artifact_dir/build.log" 2>&1
}

run_and_record() {
  model="$1"
  arm="$2"
  shift 2
  if build_one "$model" "$arm" "$@"; then
    printf 'PASS %s/%s\n' "$model" "$arm"
  else
    status=$?
    printf 'FAIL %s/%s status=%s\n' "$model" "$arm" "$status"
  fi
}

run_and_record pyramid compression_only \
  pyramid/024x128x064 pyramid_024x128x064_multiscale.onnx \
  fp16 2,64,128,256 24,128,64
run_and_record pyramid schedule_only \
  pyramid/064x128x256 pyramid_064x128x256_multiscale.onnx \
  fp32 2,64,128,256 64,128,256
run_and_record pyramid compress_then_tune \
  pyramid/032x048x128 pyramid_032x048x128_multiscale.onnx \
  fp16 2,64,128,256 32,48,128
run_and_record pyramid joint_shcosearch \
  pyramid/016x032x064 pyramid_016x032x064_multiscale.onnx \
  int8 2,64,128,256 16,32,64 pyramid_joint 15

run_and_record codriving compression_only \
  codriving/032x064x096 resnet_multiscale_32x64x96_final_fp32.onnx \
  fp16 2,64,256,512 32,64,96
run_and_record codriving schedule_only \
  codriving/064x128x256 resnet_multiscale_64x128x256_final_fp32.onnx \
  fp32 2,64,256,512 64,128,256
run_and_record codriving compress_then_tune \
  codriving/032x080x128 resnet_multiscale_32x80x128_final_fp32.onnx \
  fp16 2,64,256,512 32,80,128
run_and_record codriving joint_shcosearch \
  codriving/016x032x064 resnet_multiscale_16x32x64_final_fp32.onnx \
  int8 2,64,256,512 16,32,64 codriving_joint 16
