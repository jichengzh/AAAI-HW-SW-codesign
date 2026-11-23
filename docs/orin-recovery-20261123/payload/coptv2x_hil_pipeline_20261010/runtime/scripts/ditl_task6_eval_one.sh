#!/bin/bash
# Launch one Task 6 CARLA group and one evaluator group. Cleanup belongs to the Python owner.

set -euo pipefail

CONFIG="" RESULT_ROOT="" ROUTE="" SCENARIO="" REPEAT=""
CARLA_SEED="" TRAFFIC_SEED="" RENDER_GPU="" INFER_GPU=""
WORLD_PORT="" TRAFFIC_PORT="" PROCESS_DIR="" LOG_DIR=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config) CONFIG=${2:?}; shift 2 ;;
        --result-root) RESULT_ROOT=${2:?}; shift 2 ;;
        --route) ROUTE=${2:?}; shift 2 ;;
        --scenario) SCENARIO=${2:?}; shift 2 ;;
        --repeat) REPEAT=${2:?}; shift 2 ;;
        --carla-seed) CARLA_SEED=${2:?}; shift 2 ;;
        --traffic-seed) TRAFFIC_SEED=${2:?}; shift 2 ;;
        --render-gpu) RENDER_GPU=${2:?}; shift 2 ;;
        --infer-gpu) INFER_GPU=${2:?}; shift 2 ;;
        --world-port) WORLD_PORT=${2:?}; shift 2 ;;
        --traffic-port) TRAFFIC_PORT=${2:?}; shift 2 ;;
        --process-dir) PROCESS_DIR=${2:?}; shift 2 ;;
        --log-dir) LOG_DIR=${2:?}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for value in CONFIG RESULT_ROOT ROUTE SCENARIO REPEAT CARLA_SEED TRAFFIC_SEED \
             RENDER_GPU INFER_GPU WORLD_PORT TRAFFIC_PORT PROCESS_DIR LOG_DIR; do
    [ -n "${!value}" ] || { echo "missing --${value,,}" >&2; exit 2; }
done

[ "$ROUTE" = 146 ] && [ "$SCENARIO" = _1 ] && [ "$REPEAT" = 0 ] || exit 2
[ "$CARLA_SEED" = 2000 ] && [ "$TRAFFIC_SEED" = 2000 ] || exit 2
[ "$RENDER_GPU" = 6 ] && [ "$INFER_GPU" = 7 ] || exit 2
[ "$WORLD_PORT" = 46000 ] && [ "$TRAFFIC_PORT" = 46005 ] || exit 2

canonical_file=$(readlink -e -- "$CONFIG")
canonical_result=$(readlink -e -- "$RESULT_ROOT")
canonical_process=$(readlink -e -- "$PROCESS_DIR")
canonical_log=$(readlink -e -- "$LOG_DIR")
[ "$CONFIG" = "$canonical_file" ] && [ -f "$CONFIG" ] && [ ! -L "$CONFIG" ] || exit 2
[ "$RESULT_ROOT" = "$canonical_result" ] && [ -d "$RESULT_ROOT" ] && [ ! -L "$RESULT_ROOT" ] || exit 2
[ "$PROCESS_DIR" = "$canonical_process" ] && [ -d "$PROCESS_DIR" ] && [ ! -L "$PROCESS_DIR" ] || exit 2
[ "$LOG_DIR" = "$canonical_log" ] && [ -d "$LOG_DIR" ] && [ ! -L "$LOG_DIR" ] || exit 2
[ -z "$(find "$RESULT_ROOT" -mindepth 1 -print -quit)" ] || { echo "stale result root" >&2; exit 3; }
[ -z "$(find "$PROCESS_DIR" -mindepth 1 -print -quit)" ] || { echo "stale process output" >&2; exit 3; }
[ -z "$(find "$LOG_DIR" -mindepth 1 -print -quit)" ] || { echo "stale log output" >&2; exit 3; }

PY_BIN=/home/jichengzhi/miniconda3/envs/v2xverse/bin/python
SUPERVISOR=scripts/dual_fifo_result.py
PENDING_SIGNAL=0 PENDING_PID="" PENDING_START=""
CARLA_PID="" CARLA_START="" EVAL_PID="" EVAL_START=""

stop_pending_child() {
    if [ -n "$PENDING_PID" ] && [ -n "$PENDING_START" ]; then
        "$PY_BIN" "$SUPERVISOR" --stop-child "$PENDING_PID" "$PENDING_START" "$$" || true
    fi
    PENDING_PID="" PENDING_START=""
}

stop_owned_child() {
    local pid=$1 start=$2
    if [ -n "$pid" ] && [ -n "$start" ]; then
        "$PY_BIN" "$SUPERVISOR" --stop-child "$pid" "$start" "$$" || true
    fi
}

stop_all_owned() {
    stop_pending_child
    stop_owned_child "$EVAL_PID" "$EVAL_START"
    stop_owned_child "$CARLA_PID" "$CARLA_START"
}

handle_signal() {
    local status=$1
    trap '' INT TERM
    stop_all_owned
    exit "$status"
}

begin_owned_launch() {
    PENDING_SIGNAL=0 PENDING_PID="" PENDING_START=""
    trap 'PENDING_SIGNAL=130' INT
    trap 'PENDING_SIGNAL=143' TERM
}

finish_owned_launch() {
    trap 'handle_signal 130' INT
    trap 'handle_signal 143' TERM
    if [ "$PENDING_SIGNAL" -ne 0 ]; then
        handle_signal "$PENDING_SIGNAL"
    fi
}

clear_pending_child() {
    PENDING_PID="" PENDING_START=""
}

claim_owned_identity() {
    local pid=$1 start=$2 identity=$3
    "$PY_BIN" "$SUPERVISOR" --await-owned "$identity" "$pid" "$start" "$$" 5
}

export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=simulation/leaderboard
export SCENARIO_RUNNER_ROOT=simulation/scenario_runner
export DATA_ROOT=external_paths/data_root
export PYTHONPATH=${PYTHONPATH:-}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${LEADERBOARD_ROOT}:${LEADERBOARD_ROOT}/team_code:${SCENARIO_RUNNER_ROOT}
export PORT=$WORLD_PORT TM_PORT=$TRAFFIC_PORT CARLA_SEED TRAFFIC_SEED
export CHALLENGE_TRACK_CODENAME=SENSORS DEBUG_CHALLENGE=0 REPETITIONS=1 RESUME=0 EGO_NUM=1 SKIP_EXISTED=0
export ROUTES=${LEADERBOARD_ROOT}/data/evaluation_routes/town05_short_r${ROUTE}.xml
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town05_all_scenarios_2.json
export SCENARIOS_PARAMETER=${LEADERBOARD_ROOT}/leaderboard/scenarios/scenario_parameter${SCENARIO}.yaml
export EVAL_SETTING=v2x_final/town05_short_collab/r${ROUTE}_repeat${REPEAT}
export CHECKPOINT_ENDPOINT=${RESULT_ROOT}/${EVAL_SETTING}/results.json
export SAVE_PATH=${RESULT_ROOT}/image/${EVAL_SETTING}
export TEAM_AGENT=simulation/leaderboard/team_code/pnp_agent_e2e.py
export TEAM_CONFIG=$CONFIG RECORD_PATH=${RECORD_PATH:-}
mkdir -p "$SAVE_PATH" "$(dirname "$CHECKPOINT_ENDPOINT")"

trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

begin_owned_launch
CUDA_VISIBLE_DEVICES=$RENDER_GPU setsid "$PY_BIN" \
    "$SUPERVISOR" --supervise "$PROCESS_DIR/carla.exit" "$PROCESS_DIR/carla.identity.json" \
    "$CARLA_ROOT/CarlaUE4.sh" --world-port="$WORLD_PORT" -prefer-nvidia -opengl -RenderOffScreen \
    > "$LOG_DIR/carla.log" 2>&1 &
CARLA_PID=$!
PENDING_PID=$CARLA_PID
CARLA_START=$("$PY_BIN" "$SUPERVISOR" --process-start-token "$CARLA_PID" "$$")
PENDING_START=$CARLA_START
finish_owned_launch
if claim_owned_identity "$CARLA_PID" "$CARLA_START" "$PROCESS_DIR/carla.identity.json"; then
    clear_pending_child
else
    claim_status=$?
    stop_pending_child
    exit "$claim_status"
fi

ready=0
for _ in $(seq 1 180); do
    if [ -f "$PROCESS_DIR/carla.exit" ] || ! kill -0 "$CARLA_PID" 2>/dev/null; then break; fi
    if /home/jichengzhi/miniconda3/envs/v2xverse/bin/python - "$WORLD_PORT" <<'PY'
import socket, sys
sock = socket.socket()
sock.settimeout(.2)
try:
    sock.connect(('127.0.0.1', int(sys.argv[1])))
finally:
    sock.close()
PY
    then ready=1; break; fi
    sleep 1
done
[ "$ready" -eq 1 ] || { echo "CARLA readiness failed" >&2; exit 4; }

begin_owned_launch
CUDA_VISIBLE_DEVICES=$INFER_GPU setsid "$PY_BIN" \
    "$SUPERVISOR" --supervise "$PROCESS_DIR/evaluator.exit" "$PROCESS_DIR/evaluator.identity.json" \
    /home/jichengzhi/miniconda3/envs/v2xverse/bin/python \
    "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_parameter.py" \
    --scenarios="$SCENARIOS" --scenario_parameter="$SCENARIOS_PARAMETER" \
    --routes="$ROUTES" --repetitions="$REPETITIONS" --track="$CHALLENGE_TRACK_CODENAME" \
    --checkpoint="$CHECKPOINT_ENDPOINT" --agent="$TEAM_AGENT" --agent-config="$TEAM_CONFIG" \
    --debug="$DEBUG_CHALLENGE" --record="$RECORD_PATH" --resume="$RESUME" \
    --port="$PORT" --trafficManagerPort="$TM_PORT" --carlaProviderSeed="$CARLA_SEED" \
    --trafficManagerSeed="$TRAFFIC_SEED" --ego-num="$EGO_NUM" --timeout 600 \
    --skip_existed="$SKIP_EXISTED" > "$LOG_DIR/evaluator.log" 2>&1 &
EVAL_PID=$!
PENDING_PID=$EVAL_PID
EVAL_START=$("$PY_BIN" "$SUPERVISOR" --process-start-token "$EVAL_PID" "$$")
PENDING_START=$EVAL_START
finish_owned_launch
if claim_owned_identity "$EVAL_PID" "$EVAL_START" "$PROCESS_DIR/evaluator.identity.json"; then
    clear_pending_child
else
    claim_status=$?
    stop_pending_child
    exit "$claim_status"
fi
while [ ! -f "$PROCESS_DIR/evaluator.exit" ]; do
    kill -0 "$EVAL_PID" 2>/dev/null || exit 5
    [ ! -f "$PROCESS_DIR/carla.exit" ] || exit 4
    sleep 1
done
rc=$(cat "$PROCESS_DIR/evaluator.exit")
printf '%s\n' "$rc" > "$LOG_DIR/evaluator.exit"
exit "$rc"
