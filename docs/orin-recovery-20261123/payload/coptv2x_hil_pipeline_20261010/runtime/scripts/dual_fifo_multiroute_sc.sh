#!/bin/bash
# Usage: bash scripts/dual_fifo_multiroute_sc.sh [OPTIONS] GPU PORT RSU_MS EGO_MS "ROUTES" [REPEAT]
# Fast contract check: [--joint|--joint-c2] --check-config RSU_MS EGO_MS (no CARLA or GPU access).
# DUAL_FIFO_PYTHON overrides the verified local Python executable.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY_BIN=${DUAL_FIFO_PYTHON:-/home/jichengzhi/miniconda3/envs/v2xverse/bin/python}
PY_BIN=$(command -v "$PY_BIN") || { echo "Python executable not found" >&2; exit 2; }
# Resolve a relative override before evaluation changes into its owned cwd.
PY_BIN=$(cd "$(dirname "$PY_BIN")" && pwd)/$(basename "$PY_BIN")
CONFIG_DIR="$REPO/simulation/leaderboard/team_code/agent_config"
PARSER="$REPO/scripts/dual_fifo_result.py"

usage() {
    echo 'usage: dual_fifo_multiroute_sc.sh [--joint|--joint-c2] [--scenario _N] [--carla-seed N] [--traffic-seed N] [--cohort LABEL] [--max-attempts 1|2] GPU PORT RSU_MS EGO_MS "ROUTES" [REPEAT]' >&2
    echo '       dual_fifo_multiroute_sc.sh [--joint|--joint-c2] --check-config RSU_MS EGO_MS' >&2
    exit 2
}

check_delay() {
    case "$1" in
        0|200|500|800) return 0 ;;
        *) echo "unsupported FIFO delay: $1" >&2; return 2 ;;
    esac
}

select_arm() {
    local rsu=$1 ego=$2
    if [ "$rsu" -ne 0 ] && [ "$ego" -ne 0 ]; then
        echo "joint arm is not enabled in the single-variable phase" >&2
        return 2
    fi
    if [ "$rsu" -eq 0 ] && [ "$ego" -eq 0 ]; then
        echo baseline
    elif [ "$rsu" -ne 0 ]; then
        echo "rsu$rsu"
    else
        echo "ego$ego"
    fi
}

select_joint_arm() {
    local rsu=$1 ego=$2
    case "$rsu" in 100|200|300|400) ;; *) echo "unsupported joint RSU delay: $rsu" >&2; return 2 ;; esac
    case "$ego" in 200|300|400|500) ;; *) echo "unsupported joint Action delay: $ego" >&2; return 2 ;; esac
    if [ "$ego" -le "$rsu" ]; then
        echo "joint Action delay must exceed RSU delay" >&2
        return 2
    fi
    echo "joint_rsu${rsu}_ego${ego}"
}

check_config() {
    "$PY_BIN" - "$AGENT_CONFIG" "$RSU_DELAY" "$EGO_DELAY" "$EXPECTED_SKIP_FRAMES" <<'PY'
import sys
import yaml

path, rsu, ego, skip = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(path) as stream:
    simulation = yaml.safe_load(stream)["simulation"]
for key, expected in (("rsu_fifo_delay_ms", rsu), ("ego_output_fifo_delay_ms", ego),
                      ("tau_ego_ms", 0), ("latency_audit", True), ("ego_num", 1),
                      ("skip_frames", skip)):
    actual = simulation.get(key)
    if actual != expected or (key == "latency_audit" and actual is not True):
        raise SystemExit("%s mismatch: expected %r got %r" % (key, expected, actual))
print("rsu_delay_ms=%d ego_delay_ms=%d rsu_delay_frames=%d ego_delay_frames=%d agent_config=%s"
      % (rsu, ego, rsu // 50, ego // 50, path))
PY
}

JOINT_MODE=0
JOINT_C2_MODE=0
case "${1:-}" in
    --joint) JOINT_MODE=1; shift ;;
    --joint-c2) JOINT_MODE=1; JOINT_C2_MODE=1; shift ;;
esac
if [ "${1:-}" = "--check-config" ]; then
    [ "$#" -eq 3 ] || usage
    RSU_DELAY=$2 EGO_DELAY=$3
else
    SCENARIO=_1 CARLA_SEED=2000 TRAFFIC_SEED=2000 COHORT=pilot MAX_ATTEMPTS=2
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --scenario|--carla-seed|--traffic-seed|--cohort|--max-attempts)
                [ "$#" -ge 2 ] || usage
                case "$1" in
                    --scenario) SCENARIO=$2 ;;
                    --carla-seed) CARLA_SEED=$2 ;;
                    --traffic-seed) TRAFFIC_SEED=$2 ;;
                    --cohort) COHORT=$2 ;;
                    --max-attempts) MAX_ATTEMPTS=$2 ;;
                esac
                shift 2
                ;;
            --) shift; break ;;
            -*) usage ;;
            *) break ;;
        esac
    done
    [ "$#" -ge 5 ] && [ "$#" -le 6 ] || usage
    GPU=$1 PORT=$2 RSU_DELAY=$3 EGO_DELAY=$4 ROUTES=$5 REPEAT=${6:-0}
    [[ "$GPU" =~ ^(0|[1-9][0-9]{0,3})$ ]] || usage
    [[ "$PORT" =~ ^[1-9][0-9]{0,4}$ ]] || usage
    [ "$PORT" -le 65530 ] || usage # traffic manager uses PORT + 5
    [[ "$REPEAT" =~ ^(0|[1-9][0-9]{0,8})$ ]] || usage
    [[ "$SCENARIO" =~ ^_[1-5]$ ]] || usage
    [[ "$CARLA_SEED" =~ ^(0|[1-9][0-9]{0,9})$ ]] || usage
    [[ "$TRAFFIC_SEED" =~ ^(0|[1-9][0-9]{0,9})$ ]] || usage
    [[ "$COHORT" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || usage
    [[ "$MAX_ATTEMPTS" =~ ^[12]$ ]] || usage
    [[ "$ROUTES" =~ ^[0-9[:blank:]]+$ ]] || usage
    read -r -a ROUTE_IDS <<< "$ROUTES"
    [ "${#ROUTE_IDS[@]}" -gt 0 ] || usage
    SEEN_ROUTES=" "
    for route in "${ROUTE_IDS[@]}"; do
        [[ "$route" =~ ^(0|[1-9][0-9]{0,8})$ ]] || usage
        [[ "$SEEN_ROUTES" != *" $route "* ]] || usage
        SEEN_ROUTES="$SEEN_ROUTES$route "
    done
    BASE_PORT=$PORT
    [ $((BASE_PORT + (${#ROUTE_IDS[@]} - 1) * 10 + 5)) -le 65535 ] || usage
fi
if [ "$JOINT_MODE" -eq 1 ]; then
    ARM=$(select_joint_arm "$RSU_DELAY" "$EGO_DELAY") || exit $?
    if [ "$JOINT_C2_MODE" -eq 1 ]; then
        ARM="jointc2_${ARM#joint_}"
    fi
else
    check_delay "$RSU_DELAY" || exit $?
    check_delay "$EGO_DELAY" || exit $?
    ARM=$(select_arm "$RSU_DELAY" "$EGO_DELAY") || exit $?
fi
AGENT=$(printf 'codriving_dualfifo_%s_l1' "$ARM")
AGENT_CONFIG="$CONFIG_DIR/pnp_config_$AGENT.yaml"
EXPECTED_SKIP_FRAMES=$([ "$JOINT_C2_MODE" -eq 1 ] && echo 2 || echo 4)
check_config
[ "${1:-}" != "--check-config" ] || exit 0

OUT_DIR=${DUAL_FIFO_RUN_ROOT-"$REPO/results/dual_fifo_pilot"}
if [[ "$OUT_DIR" != /* ]] || [ "$OUT_DIR" = / ]; then
    echo "DUAL_FIFO_RUN_ROOT must be a non-empty absolute directory other than /" >&2
    exit 2
fi

# Capture provenance before switching to the private evaluation workspace.
GIT_COMMIT=$(git -C "$REPO" rev-parse HEAD)
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi

gpu_guard() {
    local line util mem
    if ! line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU" 2>&1); then
        echo "[ABORT] GPU $GPU query failed: $line" >&2
        return 3
    fi
    if [[ ! "$line" =~ ^[[:space:]]*([0-9]+),[[:space:]]*([0-9]+)[[:space:]]*$ ]]; then
        echo "[ABORT] GPU $GPU unreadable: $line" >&2
        return 3
    fi
    util=${BASH_REMATCH[1]} mem=${BASH_REMATCH[2]}
    # A just-finished joint batch can leave a brief utilization sample while
    # CUDA memory is already released.  The four-card controller owns these
    # slots, so memory occupancy is the reliable conflict signal between its
    # consecutive batches.  Preserve the stricter legacy admission rule.
    if { [ "$JOINT_MODE" -eq 0 ] && [ "$util" -gt 5 ]; } || [ "$mem" -gt 50 ]; then
        echo "[ABORT] GPU $GPU busy (util=${util}% mem=${mem}MiB)" >&2
        return 3
    fi
    echo "[guard] GPU $GPU idle (util=${util}% mem=${mem}MiB)"
}
gpu_guard || exit $?
# Joint batches can span many routes; only inspect each exact port immediately
# before its CARLA launch, so transient future-port users cannot reject a batch.
# Preserve the legacy all-route preflight for single-variable runs.
if [ "$JOINT_MODE" -eq 0 ]; then
    for index in "${!ROUTE_IDS[@]}"; do
        "$PY_BIN" "$PARSER" --ports-free "$((BASE_PORT + index * 10))" || exit $?
    done
fi

mkdir -p "$OUT_DIR"
RUN_DIR=$(mktemp -d "$OUT_DIR/${COHORT}_${ARM}_sc${SCENARIO#_}_cseed${CARLA_SEED}_tseed${TRAFFIC_SEED}_g${GPU}_p${PORT}_n${REPEAT}.XXXXXX")
SUMMARY="$RUN_DIR/summary.csv"
FAIL_LOG="$RUN_DIR/failures.log"
touch "$FAIL_LOG"
WORKSPACE="$RUN_DIR/workspace"
mkdir "$WORKSPACE"
# The legacy shell fixes SAVE_DIR=results. Isolate cwd instead of editing it.
# Do not link caches/results: all newly generated evaluation data stays here.
for input in scripts simulation external_paths checkpoints codriving common opencood; do
    if [ -e "$REPO/$input" ]; then
        ln -s "$REPO/$input" "$WORKSPACE/$input"
    fi
done
export RECORD_PATH="" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# The unchanged evaluator invokes `python`; an exported function honors the
# selected executable without replacing system PATH tools or requiring a cwd.
export DUAL_FIFO_PYTHON="$PY_BIN"
python() { "$DUAL_FIFO_PYTHON" "$@"; }
export -f python
cd "$WORKSPACE"

CARLA_PID="" CARLA_START_TOKEN="" CARLA_OWNED_TOKEN="" CARLA_TOKEN_FILE=""
EVAL_PID="" EVAL_START_TOKEN="" EVAL_OWNED_TOKEN="" EVAL_TOKEN_FILE=""
CURRENT_ROUTE=none ATTEMPT=0 PENDING_SIGNAL=0
stop_slot() {
    local pid=$1 start=$2 owned=$3 deadline=$((SECONDS + 5))
    [ -n "$pid" ] || return 0
    if [ -n "$owned" ]; then
        "$PY_BIN" "$PARSER" --stop-owned "$pid" "$owned" || true
    elif [ -n "$start" ]; then
        "$PY_BIN" "$PARSER" --stop-child "$pid" "$start" "$$" || true
    fi
    while [ "$SECONDS" -lt "$deadline" ]; do
        if "$PY_BIN" "$PARSER" --identity-gone "$pid" "$start"; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep .05
    done
    echo "[cleanup] identity still live/unreadable after 5s: pid=$pid; refusing wait" >&2
    return 1
}

teardown() {
    local failed=0
    if stop_slot "$EVAL_PID" "$EVAL_START_TOKEN" "$EVAL_OWNED_TOKEN"; then
        EVAL_PID="" EVAL_START_TOKEN="" EVAL_OWNED_TOKEN="" EVAL_TOKEN_FILE=""
    else failed=1; fi
    if stop_slot "$CARLA_PID" "$CARLA_START_TOKEN" "$CARLA_OWNED_TOKEN"; then
        CARLA_PID="" CARLA_START_TOKEN="" CARLA_OWNED_TOKEN="" CARLA_TOKEN_FILE=""
    else failed=1; fi
    return "$failed"
}

finish() {
    local status=$?
    trap - EXIT
    trap '' INT TERM
    if ! teardown && [ "$status" -eq 0 ]; then status=1; fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

begin_launch() {
    # Defer exit until both $! and the pre-setsid start identity are recorded.
    PENDING_SIGNAL=0
    trap 'PENDING_SIGNAL=130' INT
    trap 'PENDING_SIGNAL=143' TERM
}

finish_launch() {
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [ "$PENDING_SIGNAL" -ne 0 ]; then exit "$PENDING_SIGNAL"; fi
}

await_slot() {
    local slot=$1 pid=$2 start=$3 file=$4 token deadline=$((SECONDS + 5))
    # Short bounded probes let Bash dispatch INT/TERM promptly; a single 5s
    # foreground command substitution would defer the owner's signal trap.
    while [ "$SECONDS" -lt "$deadline" ]; do
        if token=$("$PY_BIN" "$PARSER" --await-owned "$file" "$pid" "$start" "$$" .1 2>"$file.error"); then
            printf -v "${slot}_OWNED_TOKEN" '%s' "$token"
            return 0
        fi
        if "$PY_BIN" "$PARSER" --identity-gone "$pid" "$start"; then break; fi
        sleep .02
    done
    echo "[ownership] bounded handshake failed: pid=$pid identity=$file" >&2
    return 1
}

start_carla() {
    teardown || return 1
    "$PY_BIN" "$PARSER" --ports-free "$PORT" || return 1
    local log="$RUN_DIR/carla_r${CURRENT_ROUTE}_a${ATTEMPT}.log" i
    CARLA_STATUS="$RUN_DIR/carla_r${CURRENT_ROUTE}_a${ATTEMPT}.status"
    CARLA_TOKEN_FILE="$RUN_DIR/carla_r${CURRENT_ROUTE}_a${ATTEMPT}.identity.json"
    begin_launch
    CUDA_VISIBLE_DEVICES=$GPU setsid "$PY_BIN" "$PARSER" --supervise "$CARLA_STATUS" \
        "$CARLA_TOKEN_FILE" \
        ./external_paths/carla_root/CarlaUE4.sh \
        --world-port="$PORT" -prefer-nvidia -opengl -RenderOffScreen > "$log" 2>&1 &
    CARLA_PID=$!
    local captured=0
    CARLA_START_TOKEN=$("$PY_BIN" "$PARSER" --process-start-token "$CARLA_PID" "$$") || captured=$?
    finish_launch
    [ "$captured" -eq 0 ] || return 1
    await_slot CARLA "$CARLA_PID" "$CARLA_START_TOKEN" "$CARLA_TOKEN_FILE" || return 1
    for i in $(seq 1 180); do
        if [ -f "$CARLA_STATUS" ] || ! "$PY_BIN" "$PARSER" --owned-alive "$CARLA_PID" "$CARLA_OWNED_TOKEN"; then
            echo "[carla] launched child exited: $log"
            return 1
        fi
        if "$PY_BIN" "$PARSER" --owned-ready "$CARLA_PID" "$CARLA_OWNED_TOKEN" "$PORT" && [ ! -f "$CARLA_STATUS" ]; then
            echo "[carla] $ARM r${CURRENT_ROUTE} ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[carla] start timeout: $log"
    return 1
}

run_one() {
    local route=$1 tag=$2
    local status="$RUN_DIR/eval_r${route}_a${ATTEMPT}.status"
    EVAL_TOKEN_FILE="$RUN_DIR/eval_r${route}_a${ATTEMPT}.identity.json"
    begin_launch
    CUDA_VISIBLE_DEVICES=$GPU setsid "$PY_BIN" "$PARSER" --supervise "$status" "$EVAL_TOKEN_FILE" bash scripts/action_fifo_eval_one.sh \
        "$route" "$PORT" "$tag" "$REPEAT" "$AGENT" "$SCENARIO" "$EVAL_SETTING" \
        "$CARLA_SEED" "$TRAFFIC_SEED" > "$EVAL_LOG" 2>&1 &
    EVAL_PID=$!
    local captured=0
    EVAL_START_TOKEN=$("$PY_BIN" "$PARSER" --process-start-token "$EVAL_PID" "$$") || captured=$?
    finish_launch
    [ "$captured" -eq 0 ] || return 1
    await_slot EVAL "$EVAL_PID" "$EVAL_START_TOKEN" "$EVAL_TOKEN_FILE" || return 1
    EVAL_RC=0
    while [ ! -f "$status" ]; do
        if ! "$PY_BIN" "$PARSER" --owned-alive "$EVAL_PID" "$EVAL_OWNED_TOKEN" || [ -f "$CARLA_STATUS" ]; then
            EVAL_RC=1
            break
        fi
        sleep 1
    done
    if [ "$EVAL_RC" -eq 0 ]; then EVAL_RC=$(< "$status"); fi
    local check_rc=0
    CHECK_MESSAGE=$("$PY_BIN" "$PARSER" --check-result "$EGO_JSON" \
        --eval-log "$EVAL_LOG" --process-exit "$EVAL_RC" --route "$route" --repeat "$REPEAT" \
        --rsu-delay "$RSU_DELAY" --ego-delay "$EGO_DELAY" --agent-config "$AGENT_CONFIG" 2>&1) || check_rc=$?
    echo "$CHECK_MESSAGE"
    return "$check_rc"
}

record_one() {
    "$PY_BIN" "$PARSER" "$EGO_JSON" "$CURRENT_ROUTE" "$REPEAT" \
        "$RSU_DELAY" "$EGO_DELAY" "$SUMMARY" "$AGENT_CONFIG" "$GIT_COMMIT" "$GIT_DIRTY" \
        --eval-log "$EVAL_LOG" --process-exit "$EVAL_RC" --infra-error "$INFRA_ERROR"
}

EXIT_STATUS=0
echo "[worker] GPU=$GPU port=$PORT cohort=$COHORT arm=$ARM scenario=$SCENARIO carla_seed=$CARLA_SEED traffic_seed=$TRAFFIC_SEED routes={$ROUTES} repeat=$REPEAT run=$RUN_DIR"
ROUTE_INDEX=0
for route in "${ROUTE_IDS[@]}"; do
    CURRENT_ROUTE=$route
    PORT=$((BASE_PORT + ROUTE_INDEX * 10))
    echo "[route] r${route} port=$PORT tm_port=$((PORT + 5))"
    ROUTE_INDEX=$((ROUTE_INDEX + 1))
    for ((ATTEMPT=1; ATTEMPT<=MAX_ATTEMPTS; ATTEMPT++)); do
        tag="dualfifo_${COHORT}_${ARM}_sc${SCENARIO#_}_r${route}_cseed${CARLA_SEED}_tseed${TRAFFIC_SEED}_n${REPEAT}_a${ATTEMPT}"
        EVAL_SETTING="v2x_final/town05_short_collab/r${route}_repeat${REPEAT}"
        EGO_JSON="$WORKSPACE/results/results_driving_${tag}/${EVAL_SETTING}/ego_vehicle_0/results.json"
        EVAL_LOG="$RUN_DIR/eval_r${route}_a${ATTEMPT}.log"
        EVAL_RC=0 INFRA_ERROR="" CHECK_MESSAGE=""
        if ! start_carla; then
            INFRA_ERROR=carla_start_failed
        elif run_one "$route" "$tag"; then
            break
        fi
        echo "route=$route attempt=$ATTEMPT status=INFRA_INVALID reason=${INFRA_ERROR:-$CHECK_MESSAGE} log=$EVAL_LOG" >> "$FAIL_LOG"
        if [ "$ATTEMPT" -eq "$MAX_ATTEMPTS" ]; then
            EXIT_STATUS=1
        else
            echo "[retry] r${route} $ARM infrastructure failure"
        fi
    done
    record_one
    teardown
done
echo "[done] $SUMMARY"
exit "$EXIT_STATUS"
