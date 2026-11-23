#!/usr/bin/env bash
# 编排 5 个 INT8 full-val(仅 GPU 3/4/5/6)。先启 4 个, 空卡自动补 s2_160, 5 个报告全生成后退出。
set -uo pipefail
export SSHPASS="$(printf '\061\062\063\064\065\066\067\070')"
SSH="sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 30001 jichengzhi@222.95.84.215"
V2X=/home/jichengzhi/V2X
RP=$V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route

# 远端启动一个 label 到指定卡。参数: LABEL GPU PGPU
remote_launch () {
  local LABEL=$1 GPU=$2 PGPU=$3
  $SSH "LBL=$LABEL GPU=$GPU PGPU=$PGPU bash -s" <<'EOF'
set -uo pipefail
V2X=/home/jichengzhi/V2X
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
HEAL=/home/jichengzhi/heal_research/HEAL
RP=$V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
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
echo "  launched $LBL gpu=$GPU pid=$(cat $FD/runner.pid)"
EOF
}

# label:gpu:pgpu   (s2_160 gpu 待定, 先占位 QUEUE)
declare -A PGPU=( [s0_024]=gpu7 [s0_040]=gpu5 [s0_056]=gpu7 [s1_048]=gpu7 [s2_160]=gpu7 )
declare -A LGPU=( [s0_024]=3 [s0_040]=4 [s0_056]=5 [s1_048]=6 )   # s2_160 动态
ALL_LABELS="s0_024 s0_040 s0_056 s1_048 s2_160"

echo "=== 启动 4 个全量 ==="
for L in s0_024 s0_040 s0_056 s1_048; do remote_launch $L ${LGPU[$L]} ${PGPU[$L]}; done
S2_STARTED=0

report_path () { local L=$1 G=$2; echo "$RP/20260628_${L}_scaleaware_bnaware_bias_v1/${L}/ap_fullval_1789_bnaware_bias_v1_gpu${G}_20260629/full_ap_eval_report.json"; }

for ((c=1;c<=300;c++)); do
  # 各卡占用
  MEM=$($SSH 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits' 2>/dev/null)
  done_cnt=0
  for L in s0_024 s0_040 s0_056 s1_048; do
    RPph=$(report_path $L ${LGPU[$L]})
    if $SSH "test -f $RPph" 2>/dev/null; then done_cnt=$((done_cnt+1)); fi
  done
  # s2_160
  s2done=0
  if [ $S2_STARTED -eq 1 ]; then
    RPs2=$(report_path s2_160 ${LGPU[s2_160]:-X})
    if $SSH "test -f $RPs2" 2>/dev/null; then s2done=1; fi
  fi
  echo "[cycle $c] done4=$done_cnt s2_started=$S2_STARTED s2done=$s2done"
  echo "$MEM" | sed 's/^/    mem: /'

  # 若 s2 未启动, 找一张空闲目标卡(3/4/5/6, mem<500)启动
  if [ $S2_STARTED -eq 0 ]; then
    for G in 3 4 5 6; do
      m=$(echo "$MEM" | awk -F', ' -v g=$G '$1==g{print $2}')
      if [ -n "$m" ] && [ "$m" -lt 500 ]; then
        echo "  -> 卡 $G 空闲, 启动 s2_160"
        LGPU[s2_160]=$G
        remote_launch s2_160 $G ${PGPU[s2_160]}
        S2_STARTED=1
        break
      fi
    done
  fi

  if [ $done_cnt -eq 4 ] && [ $S2_STARTED -eq 1 ] && [ $s2done -eq 1 ]; then
    echo "=== 5 个报告全部生成 ==="
    echo "s2_160_gpu=${LGPU[s2_160]}"
    exit 0
  fi
  sleep 60
done
echo "ORCH TIMEOUT"
echo "s2_160_gpu=${LGPU[s2_160]:-NOT_STARTED}"
exit 0
