#!/bin/bash
# Town05 official 85-cell x four-delay action-FIFO sweep.
# Dry run: bash scripts/action_fifo_full_sweep.sh --dry-run --gpus 4,7
# Execute: bash scripts/action_fifo_full_sweep.sh --run --gpus 4,7

set -u
set -o pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST="$REPO/scripts/action_fifo_full_manifest.py"
EVAL_ONE="$REPO/scripts/action_fifo_eval_one.sh"
PY_BIN=${V2XVERSE_PYTHON:-/home/jichengzhi/miniconda3/envs/v2xverse/bin/python}
MODE=""
GPU_CSV="4,7"
CARLA_SEED_VALUE=${CARLA_SEED:-2000}
TRAFFIC_SEED_VALUE=${TRAFFIC_SEED:-2000}
REPEAT=0
BASE_PORT=44000
MAX_ATTEMPTS=2
SMOKE=0

usage() {
    echo "usage: $0 (--dry-run|--check|--check-results|--run) [--smoke] [--gpus 4,7] [--seed N] [--carla-seed N] [--traffic-seed N] [--repeat N]"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run|--check|--check-results|--run) MODE=$1; shift ;;
        --gpus) GPU_CSV=${2:?need GPU CSV}; shift 2 ;;
        --seed)
            CARLA_SEED_VALUE=${2:?need seed}
            TRAFFIC_SEED_VALUE=$CARLA_SEED_VALUE
            shift 2
            ;;
        --carla-seed) CARLA_SEED_VALUE=${2:?need CARLA seed}; shift 2 ;;
        --traffic-seed) TRAFFIC_SEED_VALUE=${2:?need traffic seed}; shift 2 ;;
        --repeat) REPEAT=${2:?need repeat}; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --base-port) BASE_PORT=${2:?need base port}; shift 2 ;;
        *) usage; echo "unknown argument: $1"; exit 2 ;;
    esac
done

if [ -z "$MODE" ]; then
    usage
    exit 2
fi
if ! [[ "$CARLA_SEED_VALUE" =~ ^[0-9]+$ && "$TRAFFIC_SEED_VALUE" =~ ^[0-9]+$ && "$REPEAT" =~ ^[0-9]+$ ]]; then
    echo "seed and repeat must be non-negative integers"
    exit 2
fi
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
if [ "${#GPUS[@]}" -lt 1 ]; then
    echo "GPU list must contain unique numeric IDs"
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]] || [ -n "${SEEN_GPUS[$gpu]:-}" ]; then
        echo "GPU list must contain unique numeric IDs"
        exit 2
    fi
    SEEN_GPUS[$gpu]=1
done

manifest_count() {
    local smoke_args=()
    if [ "$SMOKE" -eq 1 ]; then
        smoke_args+=(--smoke)
    fi
    "$PY_BIN" "$MANIFEST" --format tsv "${smoke_args[@]}" --carla-seed "$CARLA_SEED_VALUE" --traffic-seed "$TRAFFIC_SEED_VALUE" --repeat "$REPEAT" "$@" | wc -l
}

check_static_contract() {
    local profile delay config expected_count label
    for profile in 8_10 5_10 5_50; do
        for delay in 0 200 500 800; do
            config="$REPO/simulation/leaderboard/team_code/agent_config/pnp_config_codriving_${profile}_fifo${delay}_l1.yaml"
            [ -f "$config" ] || { echo "missing config: $config"; return 2; }
        done
    done
    expected_count=340
    label="FULL_SWEEP_CHECK_OK base_cells=85 delays=4 tasks=340"
    if [ "$SMOKE" -eq 1 ]; then
        expected_count=6
        label="PROFILE_SMOKE_CHECK_OK base_cells=3 delays=2 tasks=6"
    fi
    [ "$(manifest_count)" -eq "$expected_count" ] || { echo "manifest task count mismatch"; return 2; }
    echo "$label"
}

dry_run() {
    local worker gpu port count first last
    local smoke_args=()
    local dry_label="FULL_SWEEP_DRY_RUN base_cells=85 delays=4 tasks=340"
    if [ "$SMOKE" -eq 1 ]; then
        smoke_args+=(--smoke)
        dry_label="PROFILE_SMOKE_DRY_RUN base_cells=3 delays=2 tasks=6"
    fi
    local seed_label=mixed
    if [ "$CARLA_SEED_VALUE" = "$TRAFFIC_SEED_VALUE" ]; then
        seed_label=$CARLA_SEED_VALUE
    fi
    echo "$dry_label seed=$seed_label repeat=$REPEAT carla_seed=$CARLA_SEED_VALUE traffic_seed=$TRAFFIC_SEED_VALUE"
    for worker in "${!GPUS[@]}"; do
        gpu=${GPUS[$worker]}
        port=$((BASE_PORT + gpu * 100))
        count=$(manifest_count --shard-index "$worker" --shard-count "${#GPUS[@]}")
        first=$("$PY_BIN" "$MANIFEST" --format tsv "${smoke_args[@]}" --carla-seed "$CARLA_SEED_VALUE" --traffic-seed "$TRAFFIC_SEED_VALUE" --repeat "$REPEAT" --shard-index "$worker" --shard-count "${#GPUS[@]}" | sed -n '1p' | cut -f10)
        last=$("$PY_BIN" "$MANIFEST" --format tsv "${smoke_args[@]}" --carla-seed "$CARLA_SEED_VALUE" --traffic-seed "$TRAFFIC_SEED_VALUE" --repeat "$REPEAT" --shard-index "$worker" --shard-count "${#GPUS[@]}" | tail -1 | cut -f10)
        echo "SHARD worker=$worker gpu=$gpu port=$port tasks=$count first=$first last=$last"
    done
    echo "DRY_RUN_ONLY carla_started=0 evaluator_started=0"
}

result_gate() {
    local smoke_args=()
    if [ "$SMOKE" -eq 1 ]; then
        smoke_args+=(--smoke)
    fi
    "$PY_BIN" "$MANIFEST" --summarize-results \
        "${smoke_args[@]}" \
        --carla-seed "$CARLA_SEED_VALUE" \
        --traffic-seed "$TRAFFIC_SEED_VALUE" \
        --repeat "$REPEAT"
}

CARLA_PID=""
CURRENT_PORT=""
teardown_carla() {
    if [ -n "$CURRENT_PORT" ]; then
        pkill -9 -f "leaderboard_evaluator_parameter.*--port=${CURRENT_PORT}\\b" 2>/dev/null || true
        pkill -9 -f "CarlaUE4.*world-port=${CURRENT_PORT}\\b" 2>/dev/null || true
    fi
    if [ -n "$CARLA_PID" ]; then
        kill -9 -- "-$CARLA_PID" 2>/dev/null || true
        kill -9 "$CARLA_PID" 2>/dev/null || true
    fi
    CARLA_PID=""
}

start_carla() {
    local gpu=$1 port=$2 log=$3 i
    teardown_carla
    CURRENT_PORT=$port
    mkdir -p "$(dirname "$log")"
    CUDA_VISIBLE_DEVICES=$gpu setsid ./external_paths/carla_root/CarlaUE4.sh \
        --world-port="$port" -prefer-nvidia -opengl -RenderOffScreen > "$log" 2>&1 &
    CARLA_PID=$!
    for i in $(seq 1 180); do
        if nc -z -w1 127.0.0.1 "$port" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

gpu_guard() {
    local gpu=$1 line util mem
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$gpu" 2>&1) || return 3
    util=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
    mem=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
    if [ "$util" -gt 5 ] || [ "$mem" -gt 50 ]; then
        echo "[ABORT] GPU $gpu busy (util=${util}% mem=${mem}MiB)"
        return 3
    fi
}

run_worker() {
    local worker=$1 gpu=$2 port=$3
    local subset scenario route profile delay frames task_carla_seed task_traffic_seed task_repeat key tag agent result_json eval_log carla_log eval_setting
    local attempt worker_status=0
    local smoke_args=()
    if [ "$SMOKE" -eq 1 ]; then
        smoke_args+=(--smoke)
    fi
    gpu_guard "$gpu" || return $?
    trap teardown_carla EXIT
    while IFS=$'\t' read -r subset scenario route profile delay frames task_carla_seed task_traffic_seed task_repeat key tag agent result_json eval_log carla_log eval_setting; do
        if "$PY_BIN" "$MANIFEST" --check-result "$result_json" >/dev/null 2>&1; then
            echo "[resume] worker=$worker skip=$key"
            continue
        fi
        for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
            if ! start_carla "$gpu" "$port" "$carla_log"; then
                echo "[infra] worker=$worker key=$key attempt=$attempt carla_start_failed"
                continue
            fi
            CUDA_VISIBLE_DEVICES=$gpu bash "$EVAL_ONE" \
                "$route" "$port" "$tag" "$task_repeat" "$agent" "$scenario" \
                "$eval_setting" "$task_carla_seed" "$task_traffic_seed" > "$eval_log" 2>&1 || true
            teardown_carla
            if "$PY_BIN" "$MANIFEST" --check-result "$result_json" >/dev/null 2>&1; then
                echo "[result] worker=$worker key=$key valid"
                break
            fi
            echo "[infra] worker=$worker key=$key attempt=$attempt result_missing"
        done
        if ! "$PY_BIN" "$MANIFEST" --check-result "$result_json" >/dev/null 2>&1; then
            echo "[failed-infrastructure] worker=$worker key=$key attempts=$MAX_ATTEMPTS"
            worker_status=4
        fi
    done < <("$PY_BIN" "$MANIFEST" --format tsv "${smoke_args[@]}" --carla-seed "$CARLA_SEED_VALUE" --traffic-seed "$TRAFFIC_SEED_VALUE" --repeat "$REPEAT" --shard-index "$worker" --shard-count "${#GPUS[@]}")
    teardown_carla
    trap - EXIT
    return "$worker_status"
}

cd "$REPO" || exit 2
export PATH=/home/jichengzhi/miniconda3/envs/v2xverse/bin:$PATH
export RECORD_PATH=${RECORD_PATH:-}

case "$MODE" in
    --check) check_static_contract ;;
    --check-results) result_gate ;;
    --dry-run) dry_run ;;
    --run)
        check_static_contract || exit $?
        worker_pids=()
        for worker in "${!GPUS[@]}"; do
            gpu=${GPUS[$worker]}
            port=$((BASE_PORT + gpu * 100))
            run_worker "$worker" "$gpu" "$port" &
            worker_pids+=("$!")
        done
        status=0
        for pid in "${worker_pids[@]}"; do
            wait "$pid" || status=$?
        done
        if ! result_gate; then
            status=4
        fi
        exit "$status"
        ;;
esac
