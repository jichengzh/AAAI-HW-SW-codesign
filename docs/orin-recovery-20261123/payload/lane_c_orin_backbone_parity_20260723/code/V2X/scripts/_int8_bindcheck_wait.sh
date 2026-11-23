#!/usr/bin/env bash
# 等待 s0_024 绑卡验证进程分配 GPU 或结束, 然后输出全卡落点 + 报告摘要。
set -uo pipefail
export SSHPASS="$(printf '\061\062\063\064\065\066\067\070')"
SSH="sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 30001 jichengzhi@222.95.84.215"
MINIDIR=/tmp/int8_bindcheck_s0_024_gpu3
for ((i=1;i<=40;i++)); do
  OUT=$($SSH "bash -lc '
PID=\$(cat $MINIDIR/runner.pid 2>/dev/null)
ALIVE=NO; kill -0 \$PID 2>/dev/null && ALIVE=YES
M3=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null)
REP=NO; [ -f $MINIDIR/full_ap_eval_report.json ] && REP=YES
echo \"ALIVE=\$ALIVE M3=\$M3 REP=\$REP\"
'" 2>/dev/null)
  echo "[wait $i] $OUT"
  M3=$(echo "$OUT" | grep -oP 'M3=\K[0-9]+' || echo 0)
  ALIVE=$(echo "$OUT" | grep -oP 'ALIVE=\K\w+' || echo NO)
  REP=$(echo "$OUT" | grep -oP 'REP=\K\w+' || echo NO)
  if [ "${M3:-0}" -gt 500 ] || [ "$REP" = "YES" ] || [ "$ALIVE" = "NO" ]; then
    echo "=== TRIGGER: 输出全卡落点 ==="
    $SSH 'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; echo "---MEM---"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
    echo "=== bindcheck stderr tail ==="
    $SSH "tail -n 15 $MINIDIR/stderr.txt 2>/dev/null"
    echo "=== report (if any) ==="
    $SSH "cat $MINIDIR/full_ap_eval_report.json 2>/dev/null || echo NO_REPORT_YET"
    exit 0
  fi
  sleep 10
done
echo "WAIT TIMEOUT"
exit 0
