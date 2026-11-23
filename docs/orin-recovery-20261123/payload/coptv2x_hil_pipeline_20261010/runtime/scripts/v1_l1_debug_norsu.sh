#!/bin/bash
# =============================================================================
# scripts/v1_l1_debug_norsu.sh  --  Task#11 I-1 L1-DIAG 诊断 run
#
# Single route r146, norsu arm (disable_rsu=true, l1_debug=true)
# Captures [L1DBG] per-tick trace to reveal wp degeneration / speed instability
#
# 口径: I-1 / L1-DIAG / diagnostic / norsu / τ_ego=0 / n_repeat=1
#
# Usage:
#   bash scripts/v1_l1_debug_norsu.sh <render_gpu> <infer_gpu> [carla_port]
#   bash scripts/v1_l1_debug_norsu.sh 3 4 40200
#
# ★ DISCIPLINE: Only run after team-lead GO + nvidia-smi guard confirms idle.
# =============================================================================
set -u
set -o pipefail

RENDER_GPU=${1:?need render_gpu (CARLA server)}
INFER_GPU=${2:?need infer_gpu (evaluator+inference)}
PORT=${3:-40200}

if [ "$RENDER_GPU" = "$INFER_GPU" ]; then
    echo "[ABORT] render_gpu == infer_gpu ($RENDER_GPU). Split-GPU required."; exit 2
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || { echo "[ABORT] cannot cd $REPO"; exit 2; }

# Python env
V2XVERSE_PY_BIN="/home/jichengzhi/miniconda3/envs/v2xverse/bin"
if [ -d "$V2XVERSE_PY_BIN" ]; then
    export PATH="$V2XVERSE_PY_BIN:$PATH"
    echo "[diag] Python: $("$V2XVERSE_PY_BIN/python" --version 2>&1)"
fi
export RECORD_PATH=""

ROUTE_ID=146
SCEN_CONFIG="_1"
OUT_DIR="results/v1_l1_diag"
TRACE_LOG="$OUT_DIR/l1dbg_norsu_r${ROUTE_ID}.log"
mkdir -p "$OUT_DIR"

# ── GPU guard ─────────────────────────────────────────────────────────────────
gpu_guard() {
    local gid="$1"
    local line
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                      --format=csv,noheader,nounits -i "$gid" 2>&1)
    [ -z "$line" ] && { echo "[ABORT] nvidia-smi empty for GPU $gid"; exit 3; }
    local util mem
    util=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
    mem=$(echo  "$line" | cut -d',' -f2 | tr -d ' ')
    if [ "$util" -gt 5 ] || [ "$mem" -gt 50 ]; then
        echo "[ABORT] GPU $gid BUSY (util=${util}% mem=${mem}MiB). Stop."; exit 3
    fi
    echo "[guard] GPU $gid idle: util=${util}%  mem=${mem}MiB OK"
}

# ── CARLA lifecycle ───────────────────────────────────────────────────────────
CARLA_PID=""
teardown() {
    if [ -n "${CARLA_PID:-}" ]; then
        kill -9 -- "-${CARLA_PID}" 2>/dev/null
        kill -9 "$CARLA_PID" 2>/dev/null
    fi
    CARLA_PID=""; sleep 5
}
trap teardown EXIT

kill_carla_on_port() {
    pkill -9 -f "CarlaUE4.*world-port=${PORT}\b" 2>/dev/null
    pkill -9 -f "CarlaUE4.sh.*world-port=${PORT}\b" 2>/dev/null
    [ -n "${CARLA_PID:-}" ] && { kill -9 -- "-${CARLA_PID}" 2>/dev/null; kill -9 "$CARLA_PID" 2>/dev/null; }
    for _w in $(seq 1 15); do nc -z -w 1 127.0.0.1 "$PORT" 2>/dev/null || break; sleep 1; done
}

start_carla() {
    kill_carla_on_port
    echo "[carla] launching on GPU $RENDER_GPU port $PORT ..."
    CUDA_VISIBLE_DEVICES=$RENDER_GPU setsid \
        ./external_paths/carla_root/CarlaUE4.sh \
        --world-port="$PORT" -prefer-nvidia -opengl -RenderOffScreen \
        > "$OUT_DIR/carla_${PORT}.log" 2>&1 &
    CARLA_PID=$!
    for i in $(seq 1 360); do
        if nc -z -w 1 127.0.0.1 "$PORT" 2>/dev/null; then
            echo "[carla] ready after ${i}s (pid $CARLA_PID)"
            return 0
        fi
        sleep 1
        [ "$i" -eq 360 ] && { echo "[ABORT] CARLA not ready in 360s"; kill_carla_on_port; return 1; }
    done
}

# ============================================================
echo ""
echo "================================================================"
echo "L1-DIAG: r${ROUTE_ID} norsu_l1_debug (l1_debug=true)"
echo "  render_gpu=$RENDER_GPU  infer_gpu=$INFER_GPU  port=$PORT"
echo "  trace → $TRACE_LOG"
echo "  $(date)"
echo "================================================================"
echo ""

gpu_guard "$RENDER_GPU"
gpu_guard "$INFER_GPU"

start_carla || { echo "[ABORT] CARLA failed to start"; exit 5; }

echo "[diag] running r${ROUTE_ID} with codriving_norsu_l1_debug ..."
CUDA_VISIBLE_DEVICES=$INFER_GPU \
    bash scripts/eval_driving_e2e.sh "$ROUTE_ID" "$PORT" "v1_l1_diag_norsu" 0 \
        "codriving_norsu_l1_debug" "$SCEN_CONFIG" \
    2>&1 | tee "$TRACE_LOG"
RET=${PIPESTATUS[0]}

echo ""
echo "================================================================"
echo "L1-DIAG done (exit=$RET): $(date)"
echo "Trace log: $TRACE_LOG"
echo "[L1DBG] line count: $(grep -c '\[L1DBG\]' "$TRACE_LOG" 2>/dev/null || echo 0)"
echo ""
echo "First 20 [L1DBG] lines:"
grep '\[L1DBG\]' "$TRACE_LOG" 2>/dev/null | head -20
echo ""
echo "Last 20 [L1DBG] lines:"
grep '\[L1DBG\]' "$TRACE_LOG" 2>/dev/null | tail -20
echo "================================================================"
