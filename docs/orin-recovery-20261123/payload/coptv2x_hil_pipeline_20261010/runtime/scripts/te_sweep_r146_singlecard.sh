#!/bin/bash
# =============================================================================
# scripts/te_sweep_r146_singlecard.sh  --  I-2 τ_ego sweep (单卡版)
#
# r146 × {τ_ego ∈ arm list}  (l1_control=true, latency_inject_ms=0, V2 speed)
# Produces the "加速档 → 驾驶分" curve: each accel档 maps to a real-measured
# Orin τ_ego (sim_ego_rsu_latency_strategy_v1 §5):
#   FP32 219ms (Δ5) / FP16 136ms (Δ3) / INT8 108ms (Δ3) / pruned <108 / te0 ideal.
#
# Mechanism: I-2 delays ego plan availability by Δ_ego=ceil(τ_ego/50) frames
#   (L1PlanStore.select_active). τ_ego=0 reproduces the I-1 baseline (DS≈65).
#
# Expected: DS monotonically DECREASES as τ_ego increases (slower reaction →
#   later braking/avoidance → more collisions) = "加速→更快反应→更安全" 闭环证据.
#
# 口径: I-2 / L1-V2 / injected_sim_arm / single_card / RSU present, τ_RSU=0 / n_repeat=1
#
# Usage:
#   bash scripts/te_sweep_r146_singlecard.sh <gpu> [port] [arm_list]
#   bash scripts/te_sweep_r146_singlecard.sh 0 40300 "0 219"        # 2-pt validate
#   bash scripts/te_sweep_r146_singlecard.sh 0 40300 "0 108 136 219" # full curve
#
# ★ Shared-GPU: set SKIP_GPU_GUARD=1 (DS/RC validation, not latency timing).
# =============================================================================
set -u
set -o pipefail

GPU=${1:?need gpu id (single card for CARLA+inference)}
PORT=${2:-40300}
ARMS=${3:-"0 108 136 219"}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || { echo "[ABORT] cannot cd $REPO"; exit 2; }

V2XVERSE_PY_BIN="/home/jichengzhi/miniconda3/envs/v2xverse/bin"
if [ -d "$V2XVERSE_PY_BIN" ]; then
    export PATH="$V2XVERSE_PY_BIN:$PATH"
    echo "[te_sweep] Python: $("$V2XVERSE_PY_BIN/python" --version 2>&1)"
fi
export RECORD_PATH=""

SCEN_CONFIG="_1"
ROUTE_ID=146
OUT_DIR="results/te_sweep_sc"
FAIL_LOG="$OUT_DIR/failures.log"
mkdir -p "$OUT_DIR"

# ── GPU guard (SKIP_GPU_GUARD=1 bypasses for shared-GPU DS runs) ──────────────
gpu_guard() {
    local gid="$1" line util mem
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                      --format=csv,noheader,nounits -i "$gid" 2>&1)
    [ -z "$line" ] && { echo "[ABORT] nvidia-smi empty for GPU $gid"; exit 3; }
    util=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
    mem=$(echo  "$line" | cut -d',' -f2 | tr -d ' ')
    if [ "${SKIP_GPU_GUARD:-0}" = "1" ]; then
        echo "[guard] SKIP_GPU_GUARD=1 → shared GPU $gid (util=${util}% mem=${mem}MiB) — DS-only run, ok"
        return 0
    fi
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
    echo "[carla] launching on GPU $GPU port $PORT (single-card) ..."
    CUDA_VISIBLE_DEVICES=$GPU setsid \
        ./external_paths/carla_root/CarlaUE4.sh \
        --world-port="$PORT" -prefer-nvidia -opengl -RenderOffScreen \
        > "$OUT_DIR/carla_${PORT}.log" 2>&1 &
    CARLA_PID=$!
    for i in $(seq 1 360); do
        nc -z -w 1 127.0.0.1 "$PORT" 2>/dev/null && { echo "[carla] ready after ${i}s (pid $CARLA_PID)"; return 0; }
        sleep 1
        [ "$i" -eq 360 ] && { echo "[ABORT] CARLA not ready in 360s"; kill_carla_on_port; return 1; }
    done
}

# ── run_one <tag> <agent_name> ────────────────────────────────────────────────
run_one() {
    local tag="$1" agent="$2"
    CUDA_VISIBLE_DEVICES=$GPU \
        bash scripts/eval_driving_e2e.sh "$ROUTE_ID" "$PORT" "$tag" 0 "$agent" "$SCEN_CONFIG"
    local ego_json="results/results_driving_${tag}/v2x_final/town05_short_collab/r${ROUTE_ID}_repeat0/ego_vehicle_0/results.json"
    python3 - "$ego_json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    sys.exit(0 if d["_checkpoint"]["global_record"]["status"] == "Completed" else 1)
except: sys.exit(1)
PY
}

report_ds() {
    local tag="$1" tau="$2"
    local ego_json="results/results_driving_${tag}/v2x_final/town05_short_collab/r${ROUTE_ID}_repeat0/ego_vehicle_0/results.json"
    python3 - "$ego_json" "$tag" "$tau" <<'PY'
import json, sys, math
tag, tau = sys.argv[2], sys.argv[3]
delta = math.ceil(int(tau)/50.0)
try:
    d = json.load(open(sys.argv[1]))
    rec = d["_checkpoint"]["global_record"]
    ds = rec.get("scores", {}).get("score_composed", rec.get("score_composed", "?"))
    rc = rec.get("scores", {}).get("score_route",    rec.get("score_route",    "?"))
    status = rec.get("status", "?")
    infrac = rec.get("infractions", {})
    def _c(v): return len(v) if isinstance(v, list) else float(v)
    coll_lay = sum(_c(v) for k,v in infrac.items() if "layout"  in k)
    coll_veh = sum(_c(v) for k,v in infrac.items() if "vehicle" in k)
    print(f"{tag}: tau_ego={tau}ms Delta_ego={delta}  DS={ds}  RC={rc}  status={status}  coll_lay={coll_lay:.4f}  coll_veh={coll_veh:.4f}")
except Exception as e:
    print(f"{tag}: tau_ego={tau}ms parse error {e}")
PY
}

# ── main ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "I-2 τ_ego SWEEP (single-card GPU${GPU}) — r${ROUTE_ID} × τ_ego{$ARMS}ms"
echo "  $(date)"
echo "================================================================"
gpu_guard "$GPU"

for TAU in $ARMS; do
    TAG="te${TAU}"
    AGENT="codriving_te${TAU}_l1"
    echo ""
    echo "--- arm: r${ROUTE_ID} τ_ego=${TAU}ms (agent=${AGENT}) ---"
    start_carla || { echo "r146,te${TAU},carla_fail" >> "$FAIL_LOG"; continue; }
    if run_one "$TAG" "$AGENT"; then
        echo "[ok] r${ROUTE_ID} te${TAU} Completed"
    else
        echo "[retry] r${ROUTE_ID} te${TAU}"; teardown
        start_carla || { echo "r146,te${TAU},restart_fail" >> "$FAIL_LOG"; continue; }
        run_one "$TAG" "$AGENT" && echo "[ok-retry] te${TAU}" \
            || { echo "r146,te${TAU},x2" >> "$FAIL_LOG"; echo "[SKIP] te${TAU} failed twice"; }
    fi
    report_ds "$TAG" "$TAU"
    teardown
done

echo ""
echo "================================================================"
echo "I-2 τ_ego SWEEP SUMMARY (single-card GPU${GPU})  r${ROUTE_ID}"
echo "  expected: DS monotonically DECREASES as τ_ego increases"
echo "================================================================"
for TAU in $ARMS; do report_ds "te${TAU}" "$TAU"; done
echo ""
echo "τ_ego sweep done: $(date)"
