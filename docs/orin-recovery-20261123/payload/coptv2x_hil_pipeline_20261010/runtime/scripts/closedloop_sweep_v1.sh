#!/bin/bash
# =============================================================================
# scripts/closedloop_sweep_v1.sh  --  Sim-C latency-sweep closed-loop driver
# Design: multi_agent/methods/design/sim_closedloop_arch_v1.md §4.E
#
# Realises the §4.E skeleton into executable commands:
#   - nvidia-smi twin-GPU guard (abort if either render/infer GPU is busy)
#   - per-latency temp agent config injection (simulation.latency_inject_ms)
#   - separate-GPU compute isolation (CARLA server = render_gpu,
#     evaluator+perception = infer_gpu) per §3.1
#   - leaderboard run via scripts/eval_driving_e2e.sh
#   - status verification (ego results.json status==Completed) + 1 retry
#   - hard teardown (kill -9 CarlaUE4 + pkill evaluator, sleep 5) per §2.5/§4.E
#   - per-(route,arm) row appended to results CSV via the aggregate script
#
# 8 arms (design §4.A): 7 injected-latency arms + 1 no-roadside (C-5 disable_rsu).
#
# Usage:
#     bash scripts/closedloop_sweep_v1.sh <render_gpu> <infer_gpu> <carla_port>
#
# ★ DISCIPLINE: this script TOUCHES GPU + LAUNCHES CARLA. Only run AFTER
#   team-lead authorization (D3 GPU coordination) + nvidia-smi twin guard passes.
#   It is the SINGLE start action once cards are freed.
# =============================================================================
set -u
set -o pipefail

# ---- args ----
RENDER_GPU=${1:?need render_gpu (CARLA server)}    # e.g. 2
INFER_GPU=${2:?need infer_gpu (evaluator+perception)}  # e.g. 0
PORT=${3:-40000}
TM_PORT=$((PORT + 5))

# ---- P1 single-card mode (team-lead authorized 2026-06-09) -------------------
# SIM_SINGLE_CARD=1 lets CARLA + inference SHARE one GPU. This is VALID for the
# injected-latency sweep (latency_ms_source=injected_from_E7): in CARLA sync
# mode the perception result is deterministic, control is bit-identical, and
# Delta=ceil(L/50) is computed from the injected value, NOT wallclock -- so
# sharing a card does NOT corrupt the sweep. It only inflates wallclock and
# removes the compute-isolation that protects MEASURED-latency purity (deferred
# to P2 Orin). The result CSV is tagged isolation=single_card_shared so this is
# never silently confused with the separate-GPU runs.
SIM_SINGLE_CARD=${SIM_SINGLE_CARD:-0}
if [ "$SIM_SINGLE_CARD" = "1" ]; then
    ISOLATION_TAG="single_card_shared"
    echo "############################################################"
    echo "# SINGLE-CARD MODE: CARLA+inference share GPU $RENDER_GPU"
    echo "#   valid for INJECTED-latency sweep ONLY, NOT compute-isolated;"
    echo "#   wallclock inflated. Measured-latency anchors -> P2 Orin."
    echo "#   result tagged isolation=single_card_shared"
    echo "############################################################"
    if [ "$RENDER_GPU" != "$INFER_GPU" ]; then
        echo "[note] single-card mode but render_gpu($RENDER_GPU)!=infer_gpu($INFER_GPU);"
        echo "       forcing infer_gpu=$RENDER_GPU so both pin the same card."
        INFER_GPU=$RENDER_GPU
    fi
else
    ISOLATION_TAG="separate_gpu"
    if [ "$RENDER_GPU" = "$INFER_GPU" ]; then
        echo "[ABORT] render_gpu == infer_gpu ($RENDER_GPU). Compute isolation"
        echo "        (§3.1) requires DISTINCT GPUs: CARLA render must not pollute"
        echo "        perception latency. Pass two different GPU ids, OR set"
        echo "        SIM_SINGLE_CARD=1 for the injected-latency single-card sweep."
        exit 2
    fi
fi

# ---- config ----
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || { echo "[ABORT] cannot cd to repo $REPO"; exit 2; }

ROUTES=(0 146 28 160 141 135)             # 6 routes (design §4.B, junction-geometry selected)
# Temporary route override for PILOT runs WITHOUT editing the canonical 6-route
# definition above. e.g. SWEEP_ROUTES="0" for the r0-only pilot. Whitespace-sep.
if [ -n "${SWEEP_ROUTES:-}" ]; then
    # shellcheck disable=SC2206
    ROUTES=(${SWEEP_ROUTES})
    echo "[sweep] ROUTES overridden via SWEEP_ROUTES -> ${ROUTES[*]} (pilot mode)"
fi
INJECT_MS=(0 50 108 150 219 300 500)       # 7 injected-latency arms (§4.A)
# 8th arm = no-roadside (C-5 disable_rsu:true), handled separately below.

SCEN_CONFIG="_1"                            # scenario_parameter_1.yaml (smoke-validated)
AGENT_DIR="simulation/leaderboard/team_code/agent_config"
BASE_CONFIG="$AGENT_DIR/pnp_config_codriving_lat_d0.yaml"   # template (has sim block)
NORSU_CONFIG="$AGENT_DIR/pnp_config_codriving_norsu.yaml"   # C-5 canonical no-RSU
# SWEEP_CSV is the canonical full-table output. Each instance also writes a
# per-PORT shard so parallel fan-out doesn't race on one file; the final
# aggregate re-scans ALL result dirs on disk (route/arm keyed) -> the canonical
# table is correct regardless of which instance writes it last.
SWEEP_CSV="results/closedloop_sweep_v1.csv"
SWEEP_CSV_SHARD="results/closedloop_sweep_v1_p${PORT}.csv"
FAIL_LOG="results/sweep_failures_p${PORT}.log"   # per-PORT (no cross-instance truncate)
AGG="scripts/closedloop_aggregate_v1.py"

mkdir -p results
: > "$FAIL_LOG"   # truncate THIS instance's failure log
echo "[sweep] repo=$REPO render_gpu=$RENDER_GPU infer_gpu=$INFER_GPU port=$PORT"
echo "[sweep] routes=${ROUTES[*]}  arms=ms{${INJECT_MS[*]}}+norsu  scen=$SCEN_CONFIG"
echo "[sweep] isolation=$ISOLATION_TAG (single_card_shared=injected-only / separate_gpu=isolated)"

# -----------------------------------------------------------------------------
# gpu_guard <gpu_id> : abort whole sweep if the GPU is busy (util>5% or mem>50MiB)
# -----------------------------------------------------------------------------
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
        echo "[ABORT] latency must be measured on an idle GPU (CLAUDE.md). Stop."
        exit 3
    fi
    echo "[guard] GPU $gid idle (util=${util}% mem=${mem}MiB) OK"
}

# -----------------------------------------------------------------------------
# write_inject_config <inject_ms> <config_name_out>
#   Produces $AGENT_DIR/pnp_config_<name>.yaml from the d0 template with
#   simulation.latency_inject_ms set to <inject_ms>. Returns the bare name
#   (without 'pnp_config_' prefix / '.yaml') that eval_driving_e2e.sh expects.
# -----------------------------------------------------------------------------
write_inject_config() {
    local L="$1"
    # Use a DISTINCT generated-config namespace so the output never collides with
    # the template (BASE_CONFIG = pnp_config_codriving_lat_d0.yaml). For L=0 the
    # old name 'codriving_lat_d0' aliased the template, and `sed BASE > BASE`
    # truncated it to empty. Namespaced by PORT too, so PARALLEL instances (one
    # per GPU/port) never clobber each other's per-arm config mid-run.
    local name="codriving_sweep_p${PORT}_d${L}"
    local out="$AGENT_DIR/pnp_config_${name}.yaml"
    if [ "$out" -ef "$BASE_CONFIG" ]; then
        echo "[ABORT] generated config $out aliases the template $BASE_CONFIG"; exit 4
    fi
    # replace the latency_inject_ms value (keep indentation + trailing comment)
    sed -E "s/^([[:space:]]*latency_inject_ms:)[[:space:]]*[0-9]+.*$/\1 ${L}          # ★sweep-injected (design §4.A)/" \
        "$BASE_CONFIG" > "$out"
    # sanity: the line must exist AND the field must now equal L
    local got
    got=$(grep -E "^[[:space:]]*latency_inject_ms:" "$out" | grep -oE '[0-9]+' | head -1)
    if [ -z "$got" ] || [ "$got" != "$L" ]; then
        echo "[ABORT] config injection failed: wanted ${L} got '${got}' in $out"; exit 4
    fi
    echo "$name"
}

# -----------------------------------------------------------------------------
# start_carla : launch CARLA server on render_gpu (background), wait until ready
# -----------------------------------------------------------------------------
start_carla() {
    echo "[carla] launching on GPU $RENDER_GPU port $PORT ..."
    # setsid -> CARLA gets its own process group so teardown can kill the whole
    # group (launcher + shipping-binary child) by -PGID without touching siblings.
    CUDA_VISIBLE_DEVICES=$RENDER_GPU setsid \
        ./external_paths/carla_root/CarlaUE4.sh \
        --world-port="$PORT" -prefer-nvidia -opengl -RenderOffScreen \
        > "results/carla_render_${PORT}.log" 2>&1 &
    CARLA_PID=$!
    # poll the RPC port until it accepts connections (max ~90s)
    local i
    for i in $(seq 1 90); do
        if python3 - "$PORT" <<'PY' 2>/dev/null
import socket,sys
s=socket.socket(); s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1]))); s.close(); sys.exit(0)
except Exception:
    sys.exit(1)
PY
        then echo "[carla] ready after ${i}s (pid $CARLA_PID)"; return 0; fi
        sleep 1
    done
    echo "[ABORT] CARLA did not become ready in 90s (pid $CARLA_PID)"; return 1
}

# -----------------------------------------------------------------------------
# teardown : hard-kill CARLA + evaluator, free the port (design §2.5 B-3 / §4.E)
# -----------------------------------------------------------------------------
teardown() {
    # ★PORT-SCOPED: only kill THIS instance's CARLA + evaluator, identified by
    # the world-port / --port=$PORT on their command lines. A blanket
    # `pkill -f CarlaUE4` would cross-kill sibling instances on other GPUs/ports
    # during parallel fan-out. CARLA shipping binary carries '--world-port=$PORT';
    # evaluator carries '--port=$PORT'.
    pkill -9 -f "leaderboard_evaluator_parameter.*--port=${PORT}\b" 2>/dev/null
    pkill -9 -f "CarlaUE4.*world-port=${PORT}\b" 2>/dev/null
    pkill -9 -f "CarlaUE4.sh.*world-port=${PORT}\b" 2>/dev/null
    # also kill the tracked launcher PID's whole process group (covers the
    # CarlaUE4.sh -> CarlaUE4-Linux-Shipping child if the port match missed).
    if [ -n "${CARLA_PID:-}" ]; then
        kill -9 -- "-${CARLA_PID}" 2>/dev/null   # process group (set via setsid)
        kill -9 "$CARLA_PID" 2>/dev/null
    fi
    CARLA_PID=""
    sleep 5   # let the RPC + TM ports release before the next arm
}
trap teardown EXIT

# -----------------------------------------------------------------------------
# run_one <route_id> <exp_tag> <agent_name>
#   Runs a single (route, arm). Returns 0 on Completed, 1 otherwise.
#   <exp_tag> -> RESULT_ROOT=results/results_driving_<exp_tag>
# -----------------------------------------------------------------------------
run_one() {
    local rid="$1" tag="$2" agent="$3"
    # eval_driving_e2e.sh <route> <port> <exp_name> <repeat> <agent_name> <scen>
    # repeat=0 (deterministic, design §4.C); evaluator pinned to infer_gpu.
    CUDA_VISIBLE_DEVICES=$INFER_GPU \
        bash scripts/eval_driving_e2e.sh "$rid" "$PORT" "$tag" 0 "$agent" "$SCEN_CONFIG"
    # verify ego results.json status==Completed
    local ego_json="results/results_driving_${tag}/v2x_final/town05_short_collab/r${rid}_repeat0/ego_vehicle_0/results.json"
    python3 - "$ego_json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    st=d["_checkpoint"]["global_record"]["status"]
    sys.exit(0 if st=="Completed" else 1)
except Exception:
    sys.exit(1)
PY
}

# =============================================================================
# MAIN SWEEP
# =============================================================================
# guard GPU(s) ONCE up front (whole-sweep gate). Single-card mode: one card.
gpu_guard "$RENDER_GPU"
if [ "$SIM_SINGLE_CARD" != "1" ]; then
    gpu_guard "$INFER_GPU"
fi

# ---- 7 injected-latency arms ----
for rid in "${ROUTES[@]}"; do
    for L in "${INJECT_MS[@]}"; do
        TAG="lat_d${L}"
        AGENT_NAME=$(write_inject_config "$L") || exit 4
        echo "=== route r${rid}  arm inject=${L}ms  agent=${AGENT_NAME} ==="
        start_carla || { echo "$rid,$L,carla_start_fail" >> "$FAIL_LOG"; teardown; continue; }
        if run_one "$rid" "$TAG" "$AGENT_NAME"; then
            echo "[ok] r${rid} d${L} Completed"
        else
            echo "[retry] r${rid} d${L} not Completed -> retry once"
            teardown; start_carla || { echo "$rid,$L,carla_restart_fail" >> "$FAIL_LOG"; teardown; continue; }
            if run_one "$rid" "$TAG" "$AGENT_NAME"; then
                echo "[ok-retry] r${rid} d${L} Completed on retry"
            else
                echo "$rid,$L,status_not_completed_x2" >> "$FAIL_LOG"
                echo "[skip] r${rid} d${L} failed twice, logged"
            fi
        fi
        teardown
    done
done

# ---- 8th arm: no-roadside (C-5 disable_rsu:true) ----
# uses the canonical norsu config directly (no latency_inject_ms -> no aligner).
NORSU_NAME="codriving_norsu"
if [ ! -f "$NORSU_CONFIG" ]; then
    echo "[ABORT] missing $NORSU_CONFIG (C-5). Cannot run no-roadside arm."; exit 5
fi
for rid in "${ROUTES[@]}"; do
    TAG="lat_norsu"
    echo "=== route r${rid}  arm NO-ROADSIDE (disable_rsu) ==="
    start_carla || { echo "$rid,norsu,carla_start_fail" >> "$FAIL_LOG"; teardown; continue; }
    if run_one "$rid" "$TAG" "$NORSU_NAME"; then
        echo "[ok] r${rid} norsu Completed"
    else
        echo "[retry] r${rid} norsu -> retry once"
        teardown; start_carla || { echo "$rid,norsu,carla_restart_fail" >> "$FAIL_LOG"; teardown; continue; }
        run_one "$rid" "$TAG" "$NORSU_NAME" && echo "[ok-retry] r${rid} norsu" \
            || { echo "$rid,norsu,status_not_completed_x2" >> "$FAIL_LOG"; echo "[skip] r${rid} norsu failed twice"; }
    fi
    teardown
done

# ---- aggregate -> per-instance shard + canonical full table ----
# The aggregator scans ALL result dirs on disk keyed by (route,arm), so it is
# idempotent across parallel instances: whoever runs it last produces the
# correct complete table. The per-PORT shard is this instance's own view.
echo "[sweep] aggregating shard -> $SWEEP_CSV_SHARD (isolation=$ISOLATION_TAG)"
python3 "$AGG" --results-root results --out "$SWEEP_CSV_SHARD" --isolation "$ISOLATION_TAG"
echo "[sweep] aggregating canonical full table -> $SWEEP_CSV"
python3 "$AGG" --results-root results --out "$SWEEP_CSV" --isolation "$ISOLATION_TAG"

echo "[sweep] DONE. csv=$SWEEP_CSV shard=$SWEEP_CSV_SHARD failures=$FAIL_LOG"
echo "[sweep] (failures, if any):"
cat "$FAIL_LOG" 2>/dev/null | sed 's/^/    /'
