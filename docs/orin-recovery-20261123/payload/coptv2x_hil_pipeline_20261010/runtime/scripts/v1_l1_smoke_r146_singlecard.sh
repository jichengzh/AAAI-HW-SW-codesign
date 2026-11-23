#!/bin/bash
# =============================================================================
# scripts/v1_l1_smoke_r146_singlecard.sh  --  Task#11 I-1 V1 L1 smoke (单卡版)
#
# r146 × {d0_l1, d500_l1, norsu_l1}  (l1_control=true, desired_speed V2)
#
# 单卡模式: CARLA + 推理同跑在一张 GPU 上。
# 适用场景: split-GPU 不可得时(另一张卡被他人训练占用)。
# 注意: 单卡不影响 driving score / DS 结果正确性(仅影响推理时延测量纯净度,
#       但 V1 验收用的是 DS/RC/碰撞率而非时延绝对值)。
#
# 验收条件(同 split-GPU 版):
#   1. d0_l1  DS≥90 + RC=100 + 无 layout 碰撞
#   2. norsu_l1  RC=100 (Fix-B + V2 速度不退化)
#   3. d500_l1 ≤ d0_l1 (陈旧信号方向正确)
#
# 口径: I-1 / L1-V2 / injected_sim_arm / single_card / τ_ego=0 / n_repeat=1
#
# Usage:
#   bash scripts/v1_l1_smoke_r146_singlecard.sh <gpu> [carla_port]
#   bash scripts/v1_l1_smoke_r146_singlecard.sh 4 40200
#
# ★ DISCIPLINE: Only run after team-lead GO + nvidia-smi guard confirms idle.
# =============================================================================
set -u
set -o pipefail

GPU=${1:?need gpu id (single card for CARLA+inference)}
PORT=${2:-40200}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || { echo "[ABORT] cannot cd $REPO"; exit 2; }

# Python env
V2XVERSE_PY_BIN="/home/jichengzhi/miniconda3/envs/v2xverse/bin"
if [ -d "$V2XVERSE_PY_BIN" ]; then
    export PATH="$V2XVERSE_PY_BIN:$PATH"
    echo "[v1_sc] Python: $("$V2XVERSE_PY_BIN/python" --version 2>&1)"
fi
export RECORD_PATH=""

AGENT_DIR="simulation/leaderboard/team_code/agent_config"
SCEN_CONFIG="_1"
ROUTE_ID=146
OUT_DIR="results/v1_l1_smoke_sc"
FAIL_LOG="$OUT_DIR/failures.log"
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
    # SKIP_GPU_GUARD=1: allow shared-GPU runs. Safe for V1 DS/RC validation
    # (driving-score, NOT latency) — see CLAUDE.md latency-idle discipline applies
    # only to timing runs. Needs ~8GB free for CARLA+CoDriving single-card.
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
    echo "[carla] launching on GPU $GPU port $PORT (single-card mode) ..."
    CUDA_VISIBLE_DEVICES=$GPU setsid \
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

# ── run_one <tag> <agent_name> ────────────────────────────────────────────────
run_one() {
    local tag="$1" agent="$2"
    # single card: inference also on same GPU
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

# ── quick DS report ───────────────────────────────────────────────────────────
report_ds() {
    local tag="$1"
    local ego_json="results/results_driving_${tag}/v2x_final/town05_short_collab/r${ROUTE_ID}_repeat0/ego_vehicle_0/results.json"
    python3 - "$ego_json" "$tag" <<'PY'
import json, sys
tag = sys.argv[2]
try:
    d = json.load(open(sys.argv[1]))
    rec = d["_checkpoint"]["global_record"]
    ds = rec.get("scores", {}).get("score_composed", rec.get("score_composed", "?"))
    rc = rec.get("scores", {}).get("score_route",    rec.get("score_route",    "?"))
    status = rec.get("status", "?")
    infrac = rec.get("infractions", {})
    def _c(v): return len(v) if isinstance(v, list) else float(v)
    coll_lay = sum(_c(v) for k,v in infrac.items() if "layout" in k)
    coll_veh = sum(_c(v) for k,v in infrac.items() if "vehicle" in k)
    print(f"{tag}: DS={ds}  RC={rc}  status={status}  coll_lay={coll_lay:.4f}  coll_veh={coll_veh:.4f}")
except Exception as e:
    print(f"{tag}: parse error {e}")
PY
}

# ============================================================
echo ""
echo "================================================================"
echo "V1 L1-V2 SMOKE TEST (SINGLE-CARD) — r${ROUTE_ID} × {d0_l1, d500_l1, norsu_l1}"
echo "  gpu=$GPU  port=$PORT  mode=single-card"
echo "  $(date)"
echo "================================================================"

gpu_guard "$GPU"

# ── arm 1: d0_l1 ─────────────────────────────────────────────────────────────
echo ""
echo "--- arm1: r${ROUTE_ID} d0_l1 (latency_inject_ms=0, l1_control=true, V2) ---"
start_carla || { echo "r146,d0_l1,carla_fail" >> "$FAIL_LOG"; exit 5; }
if run_one "v1_l1_d0" "codriving_lat_d0_l1"; then
    echo "[ok] r${ROUTE_ID} d0_l1 Completed"
else
    echo "[retry] r${ROUTE_ID} d0_l1"
    teardown
    start_carla || { echo "r146,d0_l1,restart_fail" >> "$FAIL_LOG"; exit 5; }
    run_one "v1_l1_d0" "codriving_lat_d0_l1" \
        && echo "[ok-retry] r${ROUTE_ID} d0_l1" \
        || { echo "r146,d0_l1,x2" >> "$FAIL_LOG"; echo "[SKIP] d0_l1 failed twice"; }
fi
report_ds "v1_l1_d0"
teardown

# ── arm 2: d500_l1 ───────────────────────────────────────────────────────────
echo ""
echo "--- arm2: r${ROUTE_ID} d500_l1 (latency_inject_ms=500, l1_control=true, V2) ---"
start_carla || { echo "r146,d500_l1,carla_fail" >> "$FAIL_LOG"; exit 5; }
if run_one "v1_l1_d500" "codriving_lat_d500_l1"; then
    echo "[ok] r${ROUTE_ID} d500_l1 Completed"
else
    echo "[retry] r${ROUTE_ID} d500_l1"
    teardown
    start_carla || { echo "r146,d500_l1,restart_fail" >> "$FAIL_LOG"; exit 5; }
    run_one "v1_l1_d500" "codriving_lat_d500_l1" \
        && echo "[ok-retry] r${ROUTE_ID} d500_l1" \
        || { echo "r146,d500_l1,x2" >> "$FAIL_LOG"; echo "[SKIP] d500_l1 failed twice"; }
fi
report_ds "v1_l1_d500"
teardown

# ── arm 3: norsu_l1 ──────────────────────────────────────────────────────────
echo ""
echo "--- arm3: r${ROUTE_ID} norsu_l1 (disable_rsu=true, l1_control=true, V2) ---"
start_carla || { echo "r146,norsu_l1,carla_fail" >> "$FAIL_LOG"; exit 5; }
if run_one "v1_l1_norsu" "codriving_norsu_l1"; then
    echo "[ok] r${ROUTE_ID} norsu_l1 Completed"
else
    echo "[retry] r${ROUTE_ID} norsu_l1"
    teardown
    start_carla || { echo "r146,norsu_l1,restart_fail" >> "$FAIL_LOG"; exit 5; }
    run_one "v1_l1_norsu" "codriving_norsu_l1" \
        && echo "[ok-retry] r${ROUTE_ID} norsu_l1" \
        || { echo "r146,norsu_l1,x2" >> "$FAIL_LOG"; echo "[SKIP] norsu_l1 failed twice"; }
fi
report_ds "v1_l1_norsu"
teardown

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "V1 L1-V2 SMOKE SUMMARY (single-card GPU${GPU})"
echo "================================================================"
report_ds "v1_l1_d0"
report_ds "v1_l1_d500"
report_ds "v1_l1_norsu"
echo ""
echo "Failures:"
cat "$FAIL_LOG" 2>/dev/null | grep . | sed 's/^/  FAIL: /' || echo "  (none)"
echo ""
echo "V1 L1-V2 smoke done: $(date)"
