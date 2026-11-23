#!/bin/bash
# Run one fixed action-FIFO arm on a single 4090 over matched Town05 routes.
# Usage: bash scripts/action_fifo_multiroute_sc.sh GPU PORT DELAY_MS "ROUTES" [REPEAT]
# Fast contract check: bash scripts/action_fifo_multiroute_sc.sh --check-config DELAY_MS

set -u
set -o pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PY_BIN=/home/jichengzhi/miniconda3/envs/v2xverse/bin/python
CONFIG_DIR="$REPO/simulation/leaderboard/team_code/agent_config"

check_delay() {
    case "$1" in
        0|200|500|800) return 0 ;;
        *) echo "unsupported action FIFO delay: $1 (allowed: 0 200 500 800)"; return 2 ;;
    esac
}

if [ "${1:-}" = "--check-config" ]; then
    DELAY=${2:-}
    check_delay "$DELAY" || exit $?
    AGENT="codriving_fifo${DELAY}_l1"
    CONFIG="$CONFIG_DIR/pnp_config_${AGENT}.yaml"
    "$PY_BIN" - "$CONFIG" "$DELAY" "$AGENT" <<'PY'
import math
import sys
import yaml

path, expected, agent = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path) as stream:
    simulation = yaml.safe_load(stream)["simulation"]
actual = int(simulation["action_fifo_delay_ms"])
if actual != expected:
    raise SystemExit("config delay mismatch: expected %d got %d" % (expected, actual))
frames = int(math.ceil(actual * 20.0 / 1000.0))
print("delay_ms=%d delay_frames=%d agent=%s" % (actual, frames, agent))
PY
    exit $?
fi

route_result_exists() {
    local ego_json=$1
    "$PY_BIN" - "$ego_json" <<'PY'
import json
import sys

try:
    record = json.load(open(sys.argv[1]))["_checkpoint"]["global_record"]
    status = record.get("status")
except Exception:
    print("route_result_missing_or_unparseable")
    raise SystemExit(1)

if not status:
    print("route_result_missing_or_unparseable")
    raise SystemExit(1)

print("route_result_status=%s" % status)
raise SystemExit(0)
PY
}

if [ "${1:-}" = "--check-result" ]; then
    route_result_exists "${2:-/nonexistent}"
    exit $?
fi

GPU=${1:?need GPU index}
PORT=${2:?need CARLA port}
DELAY=${3:?need action FIFO delay in ms}
ROUTES=${4:?need space-separated route ids}
REPEAT=${5:-0}
check_delay "$DELAY" || exit $?

cd "$REPO" || exit 2
V2XVERSE_PY_BIN=/home/jichengzhi/miniconda3/envs/v2xverse/bin
export PATH="$V2XVERSE_PY_BIN:$PATH"
export RECORD_PATH=""

SCEN_CONFIG="_1"
AGENT="codriving_fifo${DELAY}_l1"
OUT_DIR="results/action_fifo_pilot"
SUMMARY="$OUT_DIR/summary_fifo${DELAY}_g${GPU}_n${REPEAT}.csv"
FAIL_LOG="$OUT_DIR/failures_fifo${DELAY}_g${GPU}_n${REPEAT}.log"
mkdir -p "$OUT_DIR"
echo "route,repeat,action_fifo_delay_ms,delay_frames,DS,RC,status,coll_lay,coll_veh,ego_json,audit_csv" > "$SUMMARY"
: > "$FAIL_LOG"

gpu_guard() {
    local line util mem
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU" 2>&1)
    util=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
    mem=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
    if [ "$util" -gt 5 ] || [ "$mem" -gt 50 ]; then
        echo "[ABORT] GPU $GPU busy (util=${util}% mem=${mem}MiB)"
        return 3
    fi
    echo "[guard] GPU $GPU idle (util=${util}% mem=${mem}MiB)"
}

CARLA_PID=""
CURRENT_ROUTE="none"
teardown() {
    pkill -9 -f "leaderboard_evaluator_parameter.*--port=${PORT}\\b" 2>/dev/null || true
    pkill -9 -f "CarlaUE4.*world-port=${PORT}\\b" 2>/dev/null || true
    pkill -9 -f "CarlaUE4.sh.*world-port=${PORT}\\b" 2>/dev/null || true
    if [ -n "${CARLA_PID:-}" ]; then
        kill -9 -- "-${CARLA_PID}" 2>/dev/null || true
        kill -9 "$CARLA_PID" 2>/dev/null || true
    fi
    CARLA_PID=""
    sleep 4
}
trap teardown EXIT

start_carla() {
    teardown
    local log="$OUT_DIR/carla_fifo${DELAY}_r${CURRENT_ROUTE}_p${PORT}_n${REPEAT}.log"
    CUDA_VISIBLE_DEVICES=$GPU setsid ./external_paths/carla_root/CarlaUE4.sh \
        --world-port="$PORT" -prefer-nvidia -opengl -RenderOffScreen > "$log" 2>&1 &
    CARLA_PID=$!
    for i in $(seq 1 180); do
        if nc -z -w1 127.0.0.1 "$PORT" 2>/dev/null; then
            echo "[carla] FIFO ${DELAY}ms r${CURRENT_ROUTE} ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[carla] start timeout: $log"
    return 1
}

ego_json_for() {
    local route=$1
    local tag="fifo_d${DELAY}_r${route}_n${REPEAT}"
    echo "results/results_driving_${tag}/v2x_final/town05_short_collab/r${route}_repeat${REPEAT}/ego_vehicle_0/results.json"
}

run_one() {
    local route=$1
    local tag="fifo_d${DELAY}_r${route}_n${REPEAT}"
    local eval_log="$OUT_DIR/eval_fifo${DELAY}_r${route}_p${PORT}_n${REPEAT}.log"
    CUDA_VISIBLE_DEVICES=$GPU bash scripts/eval_driving_e2e.sh \
        "$route" "$PORT" "$tag" "$REPEAT" "$AGENT" "$SCEN_CONFIG" > "$eval_log" 2>&1
    local ego_json
    ego_json=$(ego_json_for "$route")
    route_result_exists "$ego_json"
}

record_one() {
    local route=$1
    local ego_json
    ego_json=$(ego_json_for "$route")
    "$PY_BIN" scripts/action_fifo_result.py \
        "$ego_json" "$route" "$REPEAT" "$DELAY" "$SUMMARY"
}

gpu_guard || exit $?
echo "[worker] GPU=$GPU port=$PORT FIFO=${DELAY}ms routes={$ROUTES} repeat=$REPEAT"
for route in $ROUTES; do
    CURRENT_ROUTE=$route
    if ! start_carla; then
        echo "$route,carla_start_failed" >> "$FAIL_LOG"
        record_one "$route"
        continue
    fi
    if ! run_one "$route"; then
        echo "[retry] r${route} FIFO=${DELAY}ms"
        teardown
        if ! start_carla || ! run_one "$route"; then
            echo "$route,evaluation_failed_twice" >> "$FAIL_LOG"
        fi
    fi
    record_one "$route"
    teardown
done

echo "[done] $SUMMARY"
