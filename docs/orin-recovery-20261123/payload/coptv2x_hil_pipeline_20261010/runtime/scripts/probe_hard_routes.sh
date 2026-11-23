#!/bin/bash
# =============================================================================
# scripts/probe_hard_routes.sh  --  §4.6 Task#9 难场景 early probe
#
# 3-arm × 2-route sequential probe (证伪条件1 check):
#   routes: r146 (最密路口 3-junction 88°) → r160 (最长221m 3-junction)
#   arms  : 0ms (ideal) | 500ms (max-delay) | norsu (no RSU)
#
# Split-GPU: CARLA=render_gpu, leaderboard+inference=infer_gpu.
# 口径: injected_from_E7 / separate_gpu / injected_sim_arm / n_repeat=1
#
# Usage:
#   bash scripts/probe_hard_routes.sh <render_gpu> <infer_gpu> [carla_port]
#   bash scripts/probe_hard_routes.sh 3 4 40100
#
# ★ DISCIPLINE: Only run after team-lead authorization + nvidia-smi guard.
# =============================================================================
set -u
set -o pipefail

RENDER_GPU=${1:?need render_gpu (CARLA server)}
INFER_GPU=${2:?need infer_gpu (evaluator+inference)}
PORT=${3:-40100}

if [ "$RENDER_GPU" = "$INFER_GPU" ]; then
    echo "[ABORT] render_gpu == infer_gpu ($RENDER_GPU). Split-GPU required."; exit 2
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || { echo "[ABORT] cannot cd $REPO"; exit 2; }

# ── Python env (v2xverse / Python 3.7) ───────────────────────────────────────
V2XVERSE_PY_BIN="/home/jichengzhi/miniconda3/envs/v2xverse/bin"
if [ -d "$V2XVERSE_PY_BIN" ]; then
    export PATH="$V2XVERSE_PY_BIN:$PATH"
    echo "[probe] Python: $("$V2XVERSE_PY_BIN/python" --version 2>&1)"
fi
export RECORD_PATH=""

# ── Probe config ──────────────────────────────────────────────────────────────
PROBE_ROUTES=(146 160)
PROBE_INJECT_MS=(0 500)          # arms 1+2; arm 3 = norsu

SCEN_CONFIG="_1"
AGENT_DIR="simulation/leaderboard/team_code/agent_config"
BASE_CONFIG="$AGENT_DIR/pnp_config_codriving_lat_d0.yaml"
NORSU_CONFIG="$AGENT_DIR/pnp_config_codriving_norsu.yaml"
ISOLATION_TAG="separate_gpu"
AGG="scripts/closedloop_aggregate_v1.py"
SWEEP_CSV="results/closedloop_sweep_v1.csv"
FAIL_LOG="results/probe_r146_r160_failures.log"

mkdir -p results
> "$FAIL_LOG"

echo "======================================================================"
echo "§4.6 Task#9 PROBE r146+r160 × {0ms, 500ms, norsu}"
echo "  render_gpu=$RENDER_GPU  infer_gpu=$INFER_GPU  port=$PORT"
echo "  isolation=$ISOLATION_TAG  scen=$SCEN_CONFIG"
echo "  口径: injected_from_E7 / separate_gpu / injected_sim_arm"
echo "======================================================================"

# ── gpu_guard ─────────────────────────────────────────────────────────────────
gpu_guard() {
    local gid="$1"
    local line util mem
    line=$(nvidia-smi --id="$gid" \
        --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -z "$line" ] && { echo "[ABORT] nvidia-smi empty for GPU $gid"; exit 3; }
    util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
    mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
    if [ "$util" -gt 5 ] || [ "$mem" -gt 50 ]; then
        echo "[ABORT] GPU $gid BUSY (util=${util}% mem=${mem}MiB). Stop."; exit 3
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
    CARLA_PID=""; sleep 5
}
trap teardown EXIT

# ── write_inject_config <inject_ms> → prints bare agent name ─────────────────
write_inject_config() {
    local L="$1"
    local name="codriving_probe_p${PORT}_d${L}"
    local out="$AGENT_DIR/pnp_config_${name}.yaml"
    [ "$out" -ef "$BASE_CONFIG" ] && { echo "[ABORT] config aliases template"; exit 4; }
    sed -E "s/^([[:space:]]*latency_inject_ms:)[[:space:]]*[0-9]+.*$/\1 ${L}          # ★probe-injected (Task#9)/" \
        "$BASE_CONFIG" > "$out"
    local got
    got=$(grep -E "^[[:space:]]*latency_inject_ms:" "$out" | grep -oE '[0-9]+' | head -1)
    [ -z "$got" ] || [ "$got" != "$L" ] && { echo "[ABORT] inject failed: want $L got '$got'"; exit 4; }
    echo "$name"
}

# ── start_carla ───────────────────────────────────────────────────────────────
kill_carla_on_port() {
    # ★orphan fix: kill the actual Shipping binary (not just wrapper) holding $PORT
    pkill -9 -f "CarlaUE4.*world-port=${PORT}\b" 2>/dev/null
    pkill -9 -f "CarlaUE4.sh.*world-port=${PORT}\b" 2>/dev/null
    [ -n "${CARLA_PID:-}" ] && { kill -9 -- "-${CARLA_PID}" 2>/dev/null; kill -9 "$CARLA_PID" 2>/dev/null; }
    # wait until port is actually free (orphan fully gone) before relaunching
    for _w in $(seq 1 15); do nc -z -w 1 127.0.0.1 "$PORT" 2>/dev/null || break; sleep 1; done
}
start_carla() {
    kill_carla_on_port            # ★clear any orphan CARLA + free port BEFORE launch
    echo "[carla] launching on GPU $RENDER_GPU port $PORT ..."
    CUDA_VISIBLE_DEVICES=$RENDER_GPU setsid \
        ./external_paths/carla_root/CarlaUE4.sh \
        --world-port="$PORT" -prefer-nvidia -opengl -RenderOffScreen \
        > "results/carla_probe_${PORT}.log" 2>&1 &
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

# ── run_one <route_id> <tag> <agent_name> → 0=Completed 1=fail ───────────────
run_one() {
    local rid="$1" tag="$2" agent="$3"
    CUDA_VISIBLE_DEVICES=$INFER_GPU \
        bash scripts/eval_driving_e2e.sh "$rid" "$PORT" "$tag" 0 "$agent" "$SCEN_CONFIG"
    local ego_json="results/results_driving_${tag}/v2x_final/town05_short_collab/r${rid}_repeat0/ego_vehicle_0/results.json"
    python3 - "$ego_json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    sys.exit(0 if d["_checkpoint"]["global_record"]["status"]=="Completed" else 1)
except: sys.exit(1)
PY
}

# ── report_route_ds <route_id> — quick DS print after each route ──────────────
report_route_ds() {
    local rid="$1"
    echo ""
    echo "=== r${rid} probe DS summary ==="
    python3 - results "$rid" <<'PY'
import json, os, sys, glob

results_root, rid = sys.argv[1], sys.argv[2]
arms = [("d0ms","lat_d0"), ("d500ms","lat_d500"), ("norsu","lat_norsu")]
print("  %-10s %-8s %-8s %s" % ("arm","DS","RC","status"))
for label, tag in arms:
    pattern = os.path.join(results_root,
        "results_driving_%s/v2x_final/town05_short_collab/r%s_repeat0/ego_vehicle_0/results.json" % (tag, rid))
    matches = glob.glob(pattern)
    if not matches:
        print("  %-10s %-8s %-8s %s" % (label, "N/A", "N/A", "not found"))
        continue
    try:
        d = json.load(open(matches[0]))
        gr = d["_checkpoint"]["global_record"]
        scores = gr.get("scores", {})
        ds = scores.get("score_composed", "?")
        rc = scores.get("score_route", "?")
        st = gr.get("status","?")
        # collision count from infractions
        infr = gr.get("infractions", {})
        coll_ped = len(infr.get("collisions_pedestrian",[]))
        coll_veh = len(infr.get("collisions_vehicle",[]))
        coll_lay = len(infr.get("collisions_layout",[]))
        print("  %-10s DS=%-6s RC=%-6s coll(ped=%d,veh=%d,lay=%d)  status=%s" % (
            label, str(ds)[:6], str(rc)[:6], coll_ped, coll_veh, coll_lay, st))
    except Exception as e:
        print("  %-10s [error: %s]" % (label, e))
PY
    echo ""
}

# ── Guard GPUs ────────────────────────────────────────────────────────────────
gpu_guard "$RENDER_GPU"
gpu_guard "$INFER_GPU"

# ── Main probe loop: r146 first, then r160 ────────────────────────────────────
for rid in "${PROBE_ROUTES[@]}"; do
    echo ""
    echo "################################################################"
    echo "### PROBE ROUTE r${rid}"
    echo "################################################################"

    # arm 1+2: injected 0ms + 500ms
    for L in "${PROBE_INJECT_MS[@]}"; do
        TAG="lat_d${L}"
        AGENT_NAME=$(write_inject_config "$L") || exit 4
        echo ""
        echo "--- r${rid} inject=${L}ms  tag=${TAG} ---"
        start_carla || { echo "$rid,d${L},carla_fail" >> "$FAIL_LOG"; teardown; continue; }
        if run_one "$rid" "$TAG" "$AGENT_NAME"; then
            echo "[ok] r${rid} d${L}ms Completed"
        else
            echo "[retry] r${rid} d${L}ms"
            teardown
            start_carla || { echo "$rid,d${L},restart_fail" >> "$FAIL_LOG"; teardown; continue; }
            run_one "$rid" "$TAG" "$AGENT_NAME" \
                && echo "[ok-retry] r${rid} d${L}ms" \
                || { echo "$rid,d${L},x2" >> "$FAIL_LOG"; echo "[skip] r${rid} d${L}ms"; }
        fi
        teardown
    done

    # arm 3: norsu
    [ ! -f "$NORSU_CONFIG" ] && { echo "[ABORT] missing $NORSU_CONFIG"; exit 5; }
    TAG="lat_norsu"
    echo ""
    echo "--- r${rid} NO-ROADSIDE (norsu)  tag=${TAG} ---"
    start_carla || { echo "$rid,norsu,carla_fail" >> "$FAIL_LOG"; teardown; continue; }
    if run_one "$rid" "$TAG" "codriving_norsu"; then
        echo "[ok] r${rid} norsu Completed"
    else
        echo "[retry] r${rid} norsu"
        teardown
        start_carla || { echo "$rid,norsu,restart_fail" >> "$FAIL_LOG"; teardown; continue; }
        run_one "$rid" "$TAG" "codriving_norsu" \
            && echo "[ok-retry] r${rid} norsu" \
            || { echo "$rid,norsu,x2" >> "$FAIL_LOG"; echo "[skip] r${rid} norsu"; }
    fi
    teardown

    # ★ Per-route DS report (r146 first, then r160)
    report_route_ds "$rid"
done

# ── Aggregate all results ─────────────────────────────────────────────────────
echo "[probe] Aggregating → $SWEEP_CSV"
python3 "$AGG" --results-root results --out "$SWEEP_CSV" --isolation "$ISOLATION_TAG"

echo ""
echo "[probe] DONE. Failures:"
cat "$FAIL_LOG" 2>/dev/null | grep . | sed 's/^/  FAIL: /' || echo "  (none)"
