#!/bin/bash
# =============================================================================
# scripts/collect_rsu_latency_trace_live.sh  --  §3.8 R-1 live τ-trace collect
#
# Runs 1 route with trace_collect_out mode:
#   - CARLA server on RENDER_GPU (default GPU3)
#   - Leaderboard evaluator + perception on INFER_GPU (default GPU4)
#   - Agent config: pnp_config_codriving_trace_collect.yaml
#     (sets simulation.trace_collect_out=results/rsu_latency_trace.csv, warmup=20)
#   - Output: results/rsu_latency_trace.csv (call_idx, tau_ms)
#
# 算力隔离 (§3.1): CARLA render GPU ≠ inference GPU. Both must be idle before start.
#
# Usage:
#   bash scripts/collect_rsu_latency_trace_live.sh [render_gpu] [infer_gpu] [carla_port]
#   bash scripts/collect_rsu_latency_trace_live.sh 3 4 40100
#
# ★ DISCIPLINE: run ONLY after team-lead authorization + nvidia-smi guard passes.
# =============================================================================
set -u
set -o pipefail

RENDER_GPU=${1:-3}
INFER_GPU=${2:-4}
PORT=${3:-40100}
TM_PORT=$((PORT + 5))
ROUTE=${4:-0}     # route 0 = town05_short_r0 (short, stable for τ collection)
SCEN_CONFIG="_1"
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || { echo "[ABORT] cannot cd to $REPO"; exit 2; }

AGENT_CONFIG="codriving_trace_collect"   # eval_driving_e2e.sh prepends pnp_config_ automatically
TRACE_OUT="results/rsu_latency_trace.csv"

# ── conda env: V2Xverse requires Python 3.7 + CARLA 0.9.10 ──────────────────
# Prepend v2xverse bin so `python` resolves to Python 3.7 (not base 3.13).
# Must be set BEFORE eval_driving_e2e.sh is sourced.
V2XVERSE_PY_BIN="/home/jichengzhi/miniconda3/envs/v2xverse/bin"
if [ -d "$V2XVERSE_PY_BIN" ]; then
    export PATH="$V2XVERSE_PY_BIN:$PATH"
    echo "[r1] Python env: $("$V2XVERSE_PY_BIN/python" --version 2>&1)"
else
    echo "[WARN] v2xverse env not found at $V2XVERSE_PY_BIN — using system python"
fi
export RECORD_PATH=""  # suppress --record= warning; leaderboard accepts empty

echo "============================================================"
echo "§3.8 R-1 LIVE τ-TRACE COLLECT"
echo "  render_gpu=$RENDER_GPU  infer_gpu=$INFER_GPU  port=$PORT"
echo "  route=$ROUTE  agent=$AGENT_CONFIG"
echo "  trace_out=$TRACE_OUT"
echo "============================================================"

# ── gpu_guard: abort if GPU is busy ──────────────────────────────────────────
gpu_guard() {
    local gid="$1"
    local line util mem
    line=$(nvidia-smi --id="$gid" \
        --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "$line" ]; then
        echo "[ABORT] nvidia-smi returned nothing for GPU $gid"; exit 3
    fi
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    if [ "$util" -gt 5 ] || [ "$mem" -gt 50 ]; then
        echo "[ABORT] GPU $gid BUSY (util=${util}% mem=${mem}MiB). Occupants:"
        nvidia-smi --id="$gid" \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        echo "[ABORT] compute isolation requires idle GPU. Stop."
        exit 3
    fi
    echo "[guard] GPU $gid idle: util=${util}%  mem=${mem}MiB OK"
}

# ── teardown ──────────────────────────────────────────────────────────────────
CARLA_PID=""
teardown() {
    pkill -9 -f "leaderboard_evaluator_parameter.*--port=${PORT}\b" 2>/dev/null
    pkill -9 -f "CarlaUE4.*world-port=${PORT}\b" 2>/dev/null
    pkill -9 -f "CarlaUE4.sh.*world-port=${PORT}\b" 2>/dev/null
    if [ -n "${CARLA_PID:-}" ]; then
        kill -9 -- "-${CARLA_PID}" 2>/dev/null
        kill -9 "$CARLA_PID" 2>/dev/null
    fi
    CARLA_PID=""
    sleep 5
}
trap teardown EXIT

# ── guard check ───────────────────────────────────────────────────────────────
if [ "$RENDER_GPU" = "$INFER_GPU" ]; then
    echo "[ABORT] render_gpu == infer_gpu ($RENDER_GPU). R-1 requires compute isolation."
    exit 2
fi
gpu_guard "$RENDER_GPU"
gpu_guard "$INFER_GPU"

mkdir -p results

# ── launch CARLA server ───────────────────────────────────────────────────────
echo "[r1] Launching CARLA on GPU $RENDER_GPU port $PORT ..."
CUDA_VISIBLE_DEVICES=$RENDER_GPU setsid \
    ./external_paths/carla_root/CarlaUE4.sh \
    --world-port="$PORT" -prefer-nvidia -opengl -RenderOffScreen \
    > "results/carla_r1_collect_${PORT}.log" 2>&1 &
CARLA_PID=$!

# poll until CARLA accepts connections
for i in $(seq 1 90); do
    if python3 - "$PORT" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1]))); s.close(); sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then echo "[r1] CARLA ready after ${i}s (pid $CARLA_PID)"; break; fi
    sleep 1
    if [ "$i" -eq 90 ]; then
        echo "[ABORT] CARLA did not become ready in 90s"; exit 1
    fi
done

# ── run leaderboard evaluator ─────────────────────────────────────────────────
echo "[r1] Running evaluator on GPU $INFER_GPU ..."
TAG="r1_trace_collect"
CUDA_VISIBLE_DEVICES=$INFER_GPU \
    bash scripts/eval_driving_e2e.sh "$ROUTE" "$PORT" "$TAG" 0 "$AGENT_CONFIG" "$SCEN_CONFIG"
eval_status=$?

echo "[r1] eval_driving_e2e.sh exit status: $eval_status"

# ── verify trace CSV was written ──────────────────────────────────────────────
echo "[r1] Checking trace CSV ..."
if [ ! -f "$TRACE_OUT" ]; then
    echo "[WARN] $TRACE_OUT not found — trace collection may not have run!"
else
    N_ROWS=$(tail -n +2 "$TRACE_OUT" | wc -l)
    echo "[r1] trace CSV: $N_ROWS rows written"
    python3 - "$TRACE_OUT" <<'PY'
import csv, sys, math
rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        rows.append(float(r['tau_ms']))
if not rows:
    print("  [WARN] empty trace")
    sys.exit(1)
import numpy as np
a = np.array(rows)
deltas = sorted(set(min(10, int(math.ceil(t/50.0))) for t in a))
print("  n_calls : %d (warmup already excluded)" % len(rows))
print("  tau_ms  : min=%.2f  median=%.2f  p95=%.2f  max=%.2f" % (
    a.min(), float(np.median(a)), float(np.percentile(a,95)), a.max()))
print("  Δ values: %s  (=ceil(τ/50), cap 10)" % deltas)
if len(deltas) < 2:
    print("  [WARN] Only %d distinct Δ value(s) — trace variation low" % len(deltas))
else:
    print("  [OK] %d distinct Δ values — trace ready for R-3 replay" % len(deltas))
PY
fi

teardown
echo "[r1] Done."
