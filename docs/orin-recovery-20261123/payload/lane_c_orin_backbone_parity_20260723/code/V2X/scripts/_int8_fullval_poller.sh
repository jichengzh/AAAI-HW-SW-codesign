#!/usr/bin/env bash
# Background poller: waits until all 5 INT8 full-val jobs finish (report present or PID dead).
# Re-SSHes each cycle so a dropped connection just retries. Exits 0 when batch settled.
set -uo pipefail
export SSHPASS="$(printf '\061\062\063\064\065\066\067\070')"
SSH="sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 30001 jichengzhi@222.95.84.215"
SPECS="s0_024:1 s0_040:2 s0_056:3 s1_048:4 s2_160:5"
MAX_CYCLES=120   # 120 * 120s = 4h cap
for ((i=1;i<=MAX_CYCLES;i++)); do
  OUT=$($SSH 'bash -lc "
RP=/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
done=0; alive=0
for spec in s0_024:1 s0_040:2 s0_056:3 s1_048:4 s2_160:5; do
  L=\${spec%%:*}; G=\${spec##*:}
  FD=\$RP/20260628_\${L}_scaleaware_bnaware_bias_v1/\${L}/ap_fullval_1789_bnaware_bias_v1_gpu\${G}_20260629
  if [ -f \$FD/full_ap_eval_report.json ]; then done=\$((done+1)); fi
  PID=\$(cat \$FD/runner.pid 2>/dev/null)
  if kill -0 \$PID 2>/dev/null; then alive=\$((alive+1)); fi
done
echo \"DONE=\$done ALIVE=\$alive\"
"' 2>/dev/null)
  echo "[cycle $i] $OUT"
  DONE=$(echo "$OUT" | grep -oP 'DONE=\K[0-9]+' || echo 0)
  ALIVE=$(echo "$OUT" | grep -oP 'ALIVE=\K[0-9]+' || echo 9)
  if [ "${DONE:-0}" = "5" ]; then echo "ALL REPORTS PRESENT"; exit 0; fi
  if [ "${ALIVE:-9}" = "0" ]; then echo "ALL PROCESSES SETTLED (some may lack report)"; exit 0; fi
  sleep 120
done
echo "POLLER TIMEOUT after $MAX_CYCLES cycles"
exit 0
