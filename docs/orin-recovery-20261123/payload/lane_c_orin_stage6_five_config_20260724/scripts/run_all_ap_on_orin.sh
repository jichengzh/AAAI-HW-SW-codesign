#!/usr/bin/env bash
set -u

ROOT=/home/jichengzhi/lane_c_orin_stage6_five_config_20260724
PY=/home/jichengzhi/uniad_venv/bin/python
HEAL=/home/jichengzhi/lane_c_orin_backbone_parity_20260723/code/HEAL
V2XVERSE="$ROOT/code/V2Xverse"
PYR_BRIDGE="$ROOT/code/scripts/stage3_trt_multiscale_ap_bridge_v3.py"
COD_BRIDGE="$ROOT/code/scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py"

while kill -0 628436 2>/dev/null; do
  sleep 10
done

json_true() {
  path="$1"
  expression="$2"
  "$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if eval(sys.argv[2],{"x":x}) else 1)' "$path" "$expression"
}

full_report_complete() {
  label="$1"
  report="$ROOT/$label/ap_full/report.json"
  [ -f "$report" ] || return 1
  json_true "$report" 'x.get("status")=="success" and x.get("ap50") is not None and x.get("ap70") is not None and x.get("processed_samples",x.get("num_samples"))==1789 and x.get("failed_samples",0)==0 and x.get("fallback_samples",0)==0 and (x.get("num_samples")==1789 or x.get("engine_ap_claim") is True or x.get("gates",{}).get("full_1789") is True)'
}

run_pyramid_native() {
  artifact="$ROOT/pyramid/original_default"
  model_dir="$ROOT/models/pyramid/064x128x256"
  mkdir -p "$artifact/ap_sanity" "$artifact/ap_full"
  "$PY" "$ROOT/scripts/run_pyramid_native_ap_on_orin.py" \
    --heal-root "$HEAL" --model-dir "$model_dir" \
    --checkpoint "$model_dir/net_epoch_bestval_at23.pth" \
    --num-samples 16 --raw-dir "$artifact/ap_sanity" \
    --report-json "$artifact/ap_sanity/report.json" \
    >"$artifact/ap_sanity/run.log" 2>&1 || return 1
  "$PY" "$ROOT/scripts/run_pyramid_native_ap_on_orin.py" \
    --heal-root "$HEAL" --model-dir "$model_dir" \
    --checkpoint "$model_dir/net_epoch_bestval_at23.pth" \
    --num-samples 1789 --raw-dir "$artifact/ap_full" \
    --report-json "$artifact/ap_full/report.json" \
    >"$artifact/ap_full/run.log" 2>&1
}

run_codriving_native() {
  artifact="$ROOT/codriving/original_default"
  model_dir="$ROOT/models/codriving/064x128x256"
  mkdir -p "$artifact/ap_sanity" "$artifact/ap_full"
  "$PY" "$ROOT/scripts/run_codriving_native_ap_on_orin.py" \
    --repo-root "$V2XVERSE" --model-dir "$model_dir" \
    --ap-samples 16 --num-workers 0 \
    --ap-out "$artifact/ap_sanity/report.json" \
    >"$artifact/ap_sanity/run.log" 2>&1 || return 1
  "$PY" "$ROOT/scripts/run_codriving_native_ap_on_orin.py" \
    --repo-root "$V2XVERSE" --model-dir "$model_dir" \
    --ap-samples 1789 --num-workers 0 \
    --ap-out "$artifact/ap_full/report.json" \
    >"$artifact/ap_full/run.log" 2>&1
}

run_pyramid_trt() {
  arm="$1"
  width="$2"
  channels="$3"
  precision="$4"
  artifact="$ROOT/pyramid/$arm"
  model_dir="$ROOT/models/pyramid/$width"
  mkdir -p "$artifact/ap_sanity" "$artifact/ap_full"
  "$PY" "$PYR_BRIDGE" \
    --label "pyramid_${arm}_orin" \
    --ckpt-dir "$model_dir" --engine "$artifact/backbone.engine" \
    --precision-tag "$precision" --heal-root "$HEAL" --gpu-id 0 \
    --num-samples 16 --num-workers 2 --raw-dir "$artifact/ap_sanity" \
    --eval-range 102.4,51.2 --full-ap-min-samples 1789 \
    --expected-output-channels "$channels" \
    --report-json "$artifact/ap_sanity/report.json" \
    >"$artifact/ap_sanity/run.log" 2>&1 || return 1
  json_true "$artifact/ap_sanity/report.json" 'x.get("status")=="success" and x.get("processed_samples")==16 and x.get("failed_samples")==0 and x.get("fallback_samples")==0 and x.get("pred_nonempty_count",0)>0' || return 1
  "$PY" "$PYR_BRIDGE" \
    --label "pyramid_${arm}_orin" \
    --ckpt-dir "$model_dir" --engine "$artifact/backbone.engine" \
    --precision-tag "$precision" --heal-root "$HEAL" --gpu-id 0 \
    --num-samples 1789 --num-workers 2 --raw-dir "$artifact/ap_full" \
    --eval-range 102.4,51.2 --full-ap-min-samples 1789 \
    --expected-output-channels "$channels" \
    --report-json "$artifact/ap_full/report.json" \
    >"$artifact/ap_full/run.log" 2>&1 || return 1
  json_true "$artifact/ap_full/report.json" 'x.get("status")=="success" and x.get("engine_ap_claim") is True and x.get("processed_samples")==1789 and x.get("failed_samples")==0 and x.get("fallback_samples")==0'
}

run_codriving_trt() {
  arm="$1"
  width="$2"
  precision="$3"
  artifact="$ROOT/codriving/$arm"
  model_dir="$ROOT/models/codriving/$width"
  mkdir -p "$artifact/ap_sanity" "$artifact/ap_full"
  "$PY" "$COD_BRIDGE" \
    --repo-root "$V2XVERSE" --model-dir "$model_dir" \
    --engine "$artifact/backbone.engine" --precision-tag "$precision" \
    --gate sanity --n-samples 16 --num-workers 2 \
    --eval-dir "$artifact/ap_sanity/eval" \
    --out-json "$artifact/ap_sanity/report.json" \
    >"$artifact/ap_sanity/run.log" 2>&1 || return 1
  json_true "$artifact/ap_sanity/report.json" 'x.get("gates",{}).get("sanity_16") is True' || return 1
  "$PY" "$COD_BRIDGE" \
    --repo-root "$V2XVERSE" --model-dir "$model_dir" \
    --engine "$artifact/backbone.engine" --precision-tag "$precision" \
    --gate full --n-samples 1789 --num-workers 2 \
    --eval-dir "$artifact/ap_full/eval" \
    --out-json "$artifact/ap_full/report.json" \
    >"$artifact/ap_full/run.log" 2>&1 || return 1
  json_true "$artifact/ap_full/report.json" 'x.get("gates",{}).get("full_1789") is True'
}

run_and_record() {
  label="$1"
  shift
  if full_report_complete "$label"; then
    printf 'PASS %s existing_full_1789\n' "$label"
    return 0
  fi
  if "$@"; then
    printf 'PASS %s\n' "$label"
  else
    status=$?
    printf 'FAIL %s status=%s\n' "$label" "$status"
  fi
}

run_and_record pyramid/original_default run_pyramid_native
run_and_record pyramid/compression_only run_pyramid_trt compression_only 024x128x064 24,128,64 fp16
run_and_record pyramid/schedule_only run_pyramid_trt schedule_only 064x128x256 64,128,256 fp32
run_and_record pyramid/compress_then_tune run_pyramid_trt compress_then_tune 032x048x128 32,48,128 fp16
run_and_record pyramid/joint_shcosearch run_pyramid_trt joint_shcosearch 016x032x064 16,32,64 int8

run_and_record codriving/original_default run_codriving_native
run_and_record codriving/compression_only run_codriving_trt compression_only 032x064x096 fp16
run_and_record codriving/schedule_only run_codriving_trt schedule_only 064x128x256 fp32
run_and_record codriving/compress_then_tune run_codriving_trt compress_then_tune 032x080x128 fp16
run_and_record codriving/joint_shcosearch run_codriving_trt joint_shcosearch 016x032x064 int8
