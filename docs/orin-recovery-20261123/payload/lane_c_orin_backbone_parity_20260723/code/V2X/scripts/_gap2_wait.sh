#!/usr/bin/env bash
# 等 lhc_17/s2_096 FP32 latency job 完成(result.json 出现或进程退出), 输出结果摘要。
set -uo pipefail
export SSHPASS="$(printf '\061\062\063\064\065\066\067\070')"
SSH="sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 30001 jichengzhi@222.95.84.215"
RAWROOT=/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/raw/original60_latency_fp32_gap2_20260702
OUTJ=/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/rows/latency_lut_rows_fp32_gap2_20260702.jsonl
for ((i=1;i<=90;i++)); do
  ST=$($SSH "bash -lc '
alive=\$(ps -eo cmd | grep stage2_h800_run_measurement_job | grep -v grep | grep -oE \"label [a-z0-9_]+\" | wc -l)
lhc=\$(grep -oE \"latency_(succeeded|failed|done|written)\" $RAWROOT/lhc_17_launch.log 2>/dev/null | tail -1)
s2=\$(grep -oE \"latency_(succeeded|failed|done|written)\" $RAWROOT/s2_096_launch.log 2>/dev/null | tail -1)
rows=\$(wc -l < $OUTJ 2>/dev/null || echo 0)
echo \"alive=\$alive lhc=\${lhc:-run} s2=\${s2:-run} outrows=\$rows\"
'" 2>/dev/null)
  echo "[wait $i] $ST"
  alive=$(echo "$ST" | grep -oP 'alive=\K[0-9]+' || echo 9)
  if [ "${alive:-9}" = "0" ]; then
    echo "=== 两 job 均退出, 输出结果 ==="
    $SSH "tail -n 4 $RAWROOT/lhc_17_launch.log; echo '---'; tail -n 4 $RAWROOT/s2_096_launch.log; echo '---OUTJ---'; cat $OUTJ 2>/dev/null"
    exit 0
  fi
  sleep 60
done
echo "WAIT TIMEOUT"
exit 0
