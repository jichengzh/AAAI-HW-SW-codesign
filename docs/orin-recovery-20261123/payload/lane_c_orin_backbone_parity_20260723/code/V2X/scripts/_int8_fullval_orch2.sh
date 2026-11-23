#!/usr/bin/env bash
# 监控已在跑的 4 个 full-val(s0_024→3,s0_040→4,s0_056→5,s1_048→6)。
# 任一报告生成(卡空出)时启动排队的 s2_160 到该卡。5 个报告全生成后退出。
set -uo pipefail
export SSHPASS="$(printf '\061\062\063\064\065\066\067\070')"
SSH="sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 30001 jichengzhi@222.95.84.215"
RP=/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route

remote_launch_s2 () {
  local GPU=$1
  $SSH "GPU=$GPU bash -s" <<'EOF'
set -uo pipefail
V2X=/home/jichengzhi/V2X
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
HEAL=/home/jichengzhi/heal_research/HEAL
RP=$V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
LBL=s2_160; PGPU=gpu7
CKPT=/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_${LBL}_2026_06_26
ROUTE=$RP/20260628_${LBL}_scaleaware_bnaware_bias_v1/${LBL}
PARAMS=$RP/20260628_${LBL}_bnaware_tensor_range_capture_v1_${PGPU}/tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json
ART=$ROUTE/${LBL}_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
INV=$ROUTE/tvm_operator_inventory.json
WEIGHTS=$ROUTE/runtime_weights_int8.npz
FD=$ROUTE/ap_fullval_1789_bnaware_bias_v1_gpu${GPU}_20260629
rm -rf $FD; mkdir -p $FD
CMD="cd $V2X && PYTHONPATH=$V2X:$HEAL CUDA_VISIBLE_DEVICES=$GPU $PY scripts/stage2_h800_native_int8_real_activation_bridge.py --label $LBL --ckpt-dir $CKPT --raw-dir $FD --route-dir $ROUTE --artifact-path $ART --inventory-path $INV --runtime-weight-archive-path $WEIGHTS --gpu-id $GPU --num-samples 1789 --full-ap-min-samples 1789 --ap-row-min-samples 1789 --keep-detailed-samples 20 --tensor-quant-params-path $PARAMS"
printf '%s\n' "$CMD" > $FD/command.txt; date -Is > $FD/started_at.txt
setsid bash -lc "$CMD" > $FD/stdout.txt 2> $FD/stderr.txt < /dev/null &
echo $! > $FD/runner.pid
echo "  launched s2_160 gpu=$GPU pid=$(cat $FD/runner.pid)"
EOF
}

declare -A LGPU=( [s0_024]=3 [s0_040]=4 [s0_056]=5 [s1_048]=6 )
rp () { local L=$1 G=$2; echo "$RP/20260628_${L}_scaleaware_bnaware_bias_v1/${L}/ap_fullval_1789_bnaware_bias_v1_gpu${G}_20260629/full_ap_eval_report.json"; }
S2_STARTED=0; S2_GPU=""

for ((c=1;c<=400;c++)); do
  # 用一次 SSH 批量查 5 个报告是否存在
  CHECK=$($SSH "bash -lc '
for spec in s0_024:3 s0_040:4 s0_056:5 s1_048:6; do
  L=\${spec%%:*}; G=\${spec##*:}
  P=$RP/20260628_\${L}_scaleaware_bnaware_bias_v1/\${L}/ap_fullval_1789_bnaware_bias_v1_gpu\${G}_20260629/full_ap_eval_report.json
  [ -f \$P ] && echo \$L:DONE || echo \$L:RUN
done
if [ -n \"$S2_GPU\" ]; then
  P=$RP/20260628_s2_160_scaleaware_bnaware_bias_v1/s2_160/ap_fullval_1789_bnaware_bias_v1_gpu${S2_GPU}_20260629/full_ap_eval_report.json
  [ -f \$P ] && echo s2_160:DONE || echo s2_160:RUN
fi
'" 2>/dev/null)
  echo "[cycle $c] $(echo "$CHECK" | tr '\n' ' ')  S2_STARTED=$S2_STARTED S2_GPU=$S2_GPU"
  done4=$(echo "$CHECK" | grep -cE '^(s0_024|s0_040|s0_056|s1_048):DONE' || true)

  # 首个完成 -> 该卡启 s2_160
  if [ "$S2_STARTED" -eq 0 ]; then
    for L in s0_024 s0_040 s0_056 s1_048; do
      if echo "$CHECK" | grep -q "^${L}:DONE"; then
        S2_GPU=${LGPU[$L]}
        echo "  -> $L 完成, 卡 $S2_GPU 空出, 启动 s2_160"
        remote_launch_s2 $S2_GPU
        S2_STARTED=1
        break
      fi
    done
  fi

  s2done=0; echo "$CHECK" | grep -q "^s2_160:DONE" && s2done=1
  if [ "$done4" -eq 4 ] && [ "$S2_STARTED" -eq 1 ] && [ "$s2done" -eq 1 ]; then
    echo "=== 5 个报告全部生成 (s2_160_gpu=$S2_GPU) ==="
    exit 0
  fi
  sleep 90
done
echo "ORCH2 TIMEOUT (S2_GPU=$S2_GPU S2_STARTED=$S2_STARTED)"
exit 0
