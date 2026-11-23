#!/usr/bin/env bash
set -euo pipefail

root=/home/jichengzhi/lane_c_orin_backbone_parity_20260723
code_root="$root/code/V2X"
heal_root="$root/code/HEAL"
checkpoint=/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10
dataset_link=/home/jichengzhi/heal_research/HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure
full_dataset="$root/dair_val_full/cooperative-vehicle-infrastructure"
python_bin=/home/jichengzhi/uniad_venv/bin/python
log_root="$root/logs/orin"

old_dataset_target="$(readlink "$dataset_link")"

restore_dataset_link() {
    ln -sfn "$old_dataset_target" "$dataset_link"
}
trap restore_dataset_link EXIT

mkdir -p "$log_root" "$root/ap_full/fp16" "$root/ap_full/int8"
ln -sfn "$full_dataset" "$dataset_link"

cd "$code_root"
PYTHONPATH="$code_root" "$python_bin" scripts/stage3_trt_multiscale_ap_bridge_v3.py \
    --ckpt-dir "$checkpoint" \
    --engine "$root/fp16/backbone_fp16.engine" \
    --precision-tag fp16 \
    --heal-root "$heal_root" \
    --raw-dir "$root/ap_full/fp16" \
    --report-json "$root/ap_full/fp16/stage3_trt_multiscale_ap_bridge_report.json" \
    --eval-range 102.4,51.2 \
    --full-ap-min-samples 1789 \
    >"$log_root/fp16_full1789_ap.log" 2>&1

PYTHONPATH="$code_root" "$python_bin" scripts/stage3_trt_multiscale_ap_bridge_v3.py \
    --ckpt-dir "$checkpoint" \
    --engine "$root/int8/backbone_int8.engine" \
    --precision-tag int8 \
    --heal-root "$heal_root" \
    --raw-dir "$root/ap_full/int8" \
    --report-json "$root/ap_full/int8/stage3_trt_multiscale_ap_bridge_report.json" \
    --eval-range 102.4,51.2 \
    --full-ap-min-samples 1789 \
    >"$log_root/int8_full1789_ap.log" 2>&1
