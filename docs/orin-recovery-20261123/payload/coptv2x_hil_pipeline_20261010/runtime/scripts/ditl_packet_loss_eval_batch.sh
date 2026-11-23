#!/bin/bash
# Execute one canonical packet-loss arm/scenario batch on one fixed worker.
set -euo pipefail

ORIGINAL_ARGV=("$0" "$@")
COHORT="" BATCH_ID="" ATTEMPT="" WORKER="" GPU=""
WORLD_PORT="" STREAMING_PORT="" SECONDARY_PORT="" TRAFFIC_PORT=""
declare -a TASK_KEYS=() ROUTES_LIST=() SCENARIOS_LIST=() PROFILES_LIST=() CONFIGS_LIST=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cohort) COHORT=${2:?}; shift 2 ;;
        --batch-id) BATCH_ID=${2:?}; shift 2 ;;
        --attempt) ATTEMPT=${2:?}; shift 2 ;;
        --worker) WORKER=${2:?}; shift 2 ;;
        --gpu) GPU=${2:?}; shift 2 ;;
        --world-port) WORLD_PORT=${2:?}; shift 2 ;;
        --streaming-port) STREAMING_PORT=${2:?}; shift 2 ;;
        --secondary-port) SECONDARY_PORT=${2:?}; shift 2 ;;
        --traffic-port) TRAFFIC_PORT=${2:?}; shift 2 ;;
        --task)
            [ "$#" -ge 6 ] || { echo "--task requires five values" >&2; exit 2; }
            TASK_KEYS+=("$2"); ROUTES_LIST+=("$3"); SCENARIOS_LIST+=("$4")
            PROFILES_LIST+=("$5"); CONFIGS_LIST+=("$6"); shift 6 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for value in COHORT BATCH_ID ATTEMPT WORKER GPU WORLD_PORT STREAMING_PORT SECONDARY_PORT TRAFFIC_PORT; do
    [ -n "${!value}" ] || { echo "missing $value" >&2; exit 2; }
done
[ "${#TASK_KEYS[@]}" -ge 1 ] && [ "${#TASK_KEYS[@]}" -le 9 ] || exit 2
[ "$ATTEMPT" = 1 ] || [ "$ATTEMPT" = 2 ] || exit 2
case "$ATTEMPT:$WORKER:$GPU:$WORLD_PORT:$STREAMING_PORT:$SECONDARY_PORT:$TRAFFIC_PORT" in
    1:worker6:6:62000:62001:62002:62005|1:worker7:7:62100:62101:62102:62105|\
    2:worker6:6:63000:63001:63002:63005|2:worker7:7:63100:63101:63102:63105) ;;
    *) echo "invalid fixed launch coordinates" >&2; exit 2 ;;
esac
canonical_cohort=$(readlink -e -- "$COHORT")
[ "$canonical_cohort" = "$COHORT" ] && [ -d "$COHORT" ] && [ ! -L "$COHORT" ] || exit 2
case "$COHORT" in
    /var/tmp/codriving-ditl/packet-loss-stage1-*|/data/jichengzhi_v2x/ditl/packet-loss-stage1-*) ;;
    *) exit 2 ;;
esac

PY_BIN=/home/jichengzhi/miniconda3/envs/v2xverse/bin/python
SUPERVISOR=scripts/dual_fifo_result.py
CARLA_ROOT=external_paths/carla_root
LEADERBOARD_ROOT=simulation/leaderboard
SCENARIO_RUNNER_ROOT=simulation/scenario_runner
DATA_ROOT=external_paths/data_root
export CARLA_ROOT LEADERBOARD_ROOT SCENARIO_RUNNER_ROOT DATA_ROOT
export PYTHONPATH=${PYTHONPATH:-}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${LEADERBOARD_ROOT}:${LEADERBOARD_ROOT}/team_code:${SCENARIO_RUNNER_ROOT}
export CUDA_VISIBLE_DEVICES=$GPU PORT=$WORLD_PORT TM_PORT=$TRAFFIC_PORT
export CHALLENGE_TRACK_CODENAME=SENSORS DEBUG_CHALLENGE=0 REPETITIONS=1 RESUME=0 EGO_NUM=1 SKIP_EXISTED=0
export DITL_BATCH_ARGV_JSON
DITL_BATCH_ARGV_JSON=$("$PY_BIN" - "${ORIGINAL_ARGV[@]}" <<'PY'
import json, sys
print(json.dumps(sys.argv[1:], separators=(',', ':')))
PY
)

stop_child() {
    local pid=${1:-} token=${2:-}
    if [ -n "$pid" ] && [ -n "$token" ]; then
        "$PY_BIN" "$SUPERVISOR" --stop-child "$pid" "$token" "$$" >/dev/null 2>&1 || true
    fi
}

ACTIVE_CARLA_PID="" ACTIVE_CARLA_TOKEN="" ACTIVE_EVAL_PID="" ACTIVE_EVAL_TOKEN=""
stop_active_task() {
    stop_child "$ACTIVE_EVAL_PID" "$ACTIVE_EVAL_TOKEN"
    stop_child "$ACTIVE_CARLA_PID" "$ACTIVE_CARLA_TOKEN"
    ACTIVE_CARLA_PID=""; ACTIVE_CARLA_TOKEN=""
    ACTIVE_EVAL_PID=""; ACTIVE_EVAL_TOKEN=""
}
trap 'stop_active_task; exit 130' INT
trap 'stop_active_task; exit 143' TERM

write_evidence() {
    local attempt_dir=$1 task_key=$2 route=$3 scenario=$4 profile=$5 config=$6 rc=$7
    local result_source=$8 audit_source=$9
    "$PY_BIN" - "$attempt_dir" "$task_key" "$route" "$scenario" "$profile" "$config" \
        "$rc" "$result_source" "$audit_source" "$ATTEMPT" "$WORKER" "$GPU" \
        "$WORLD_PORT" "$STREAMING_PORT" "$SECONDARY_PORT" "$TRAFFIC_PORT" <<'PY'
import hashlib, json, os, pathlib, shutil, sys
(root, key, route, scenario, profile, config, rc, result_source, audit_source,
 attempt, worker, gpu, *ports) = sys.argv[1:]
root = pathlib.Path(root); result = root/'result.json'; audit = root/'event_audit.jsonl'
if result_source and pathlib.Path(result_source).is_file(): shutil.copyfile(result_source, result)
else: result.write_text('{}\n', encoding='utf-8')
if audit_source and pathlib.Path(audit_source).is_file(): shutil.copyfile(audit_source, audit)
else: audit.write_text('{"schema_version":"missing"}\n', encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
config_sha = sha(pathlib.Path(config))
policy_hash = '0'*64
try:
    first = json.loads(audit.read_text(encoding='utf-8').splitlines()[0])
    value = first.get('policy_config_hash')
    if isinstance(value, str) and len(value) == 64: policy_hash = value
except (ValueError, IndexError): pass
command = json.loads(os.environ['DITL_BATCH_ARGV_JSON'])
arm = key.split(':', 2)[1]
shared = {'task_key': key, 'attempt': int(attempt), 'worker': worker,
          'gpu': int(gpu), 'ports': [int(p) for p in ports], 'command_argv': command}
identity = dict(shared, schema='ditl.packet-loss-stage1-command-identity.v1')
status = dict(shared, schema='ditl.packet-loss-stage1-task-status.v1', arm=arm,
              scenario=scenario, route=int(route), carla_seed=2000, traffic_seed=2000,
              replay_seed=2000, repeat=0, config_sha256=config_sha,
              returncode=int(rc), state='completed' if int(rc) == 0 else 'failed')
for name, value in (('command_identity.json', identity), ('status.json', status)):
    (root/name).write_text(json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n', encoding='ascii')
artifacts = {name: {'path': name, 'sha256': sha(root/name)} for name in
             ('result.json', 'event_audit.jsonl', 'status.json', 'command_identity.json')}
evidence = {'schema': 'ditl.packet-loss-stage1-task-evidence.v1',
            'cohort_path': str(root.parents[2].resolve()),
            'task_identity': {'key': key, 'arm': arm,
              'drop_rate': {'clean':0.0,'drop_010':0.1,'drop_020':0.2,'drop_030':0.3}[arm],
              'subset': {'_1':'1s','_2':'2s','_3':'3s','_4':'4s','_5':'5s'}[scenario],
              'scenario': scenario, 'profile': profile, 'route': int(route),
              'carla_seed': 2000, 'traffic_seed': 2000, 'replay_seed': 2000,
              'repeat': 0, 'skip_frames': 2},
            'attempt': int(attempt), 'worker': worker, 'gpu': int(gpu),
            'ports': [int(p) for p in ports], 'command_argv': command,
            'config_sha256': config_sha, 'policy_config_hash': policy_hash,
            'artifacts': artifacts}
(root/'evidence.json').write_text(json.dumps(evidence, sort_keys=True, separators=(',', ':'))+'\n', encoding='ascii')
for path in root.iterdir():
    if path.is_file(): path.chmod(0o400)
PY
}

run_task() {
    local task_key=$1 route=$2 scenario=$3 profile=$4 config=$5
    case "$task_key" in stage1:*:${scenario}:r${route}:cseed2000:tseed2000:replay2000:rep0) ;; *) return 2 ;; esac
    case "$scenario" in _1|_2|_3|_4|_5) ;; *) return 2 ;; esac
    case "$profile" in 8_10|5_10|5_50) ;; *) return 2 ;; esac
    [[ "$route" =~ ^[0-9]+$ ]] || return 2
    local canonical_config
    canonical_config=$(readlink -e -- "$config")
    [ "$canonical_config" = "$config" ] && [ -f "$config" ] && [ ! -L "$config" ] || return 2
    local safe=${task_key//:/__}
    local attempt_dir="$COHORT/attempts/attempt-$ATTEMPT/$safe"
    mkdir "$attempt_dir"
    local work="$attempt_dir/work" logs="$attempt_dir/logs" processes="$attempt_dir/processes"
    mkdir "$work" "$logs" "$processes"
    local eval_setting="v2x_final/town05_short_collab/r${route}_repeat0"
    local checkpoint="$work/$eval_setting/results.json"
    local save_path="$work/image/$eval_setting"
    mkdir -p "$save_path" "$(dirname "$checkpoint")"
    export ROUTES="$LEADERBOARD_ROOT/data/evaluation_routes/town05_short_r${route}.xml"
    export SCENARIOS="$LEADERBOARD_ROOT/data/scenarios/town05_all_scenarios_2.json"
    export SCENARIOS_PARAMETER="$LEADERBOARD_ROOT/leaderboard/scenarios/scenario_parameter${scenario}.yaml"
    export EVAL_SETTING=$eval_setting CHECKPOINT_ENDPOINT=$checkpoint SAVE_PATH=$save_path
    export TEAM_AGENT=simulation/leaderboard/team_code/pnp_agent_e2e.py TEAM_CONFIG=$config
    export CARLA_SEED=2000 TRAFFIC_SEED=2000 RECORD_PATH=${RECORD_PATH:-}
    local carla_pid="" carla_token="" eval_pid="" eval_token="" rc=0 ready=0
    CUDA_VISIBLE_DEVICES=$GPU setsid "$PY_BIN" "$SUPERVISOR" --supervise \
        "$processes/carla.exit" "$processes/carla.identity.json" \
        "$CARLA_ROOT/CarlaUE4.sh" --world-port="$WORLD_PORT" \
        -prefer-nvidia -opengl -RenderOffScreen >"$logs/carla.log" 2>&1 &
    carla_pid=$!
    carla_token=$("$PY_BIN" "$SUPERVISOR" --process-start-token "$carla_pid" "$$") || rc=4
    ACTIVE_CARLA_PID=$carla_pid; ACTIVE_CARLA_TOKEN=$carla_token
    if [ "$rc" -eq 0 ]; then
        "$PY_BIN" "$SUPERVISOR" --await-owned "$processes/carla.identity.json" \
            "$carla_pid" "$carla_token" "$$" 5 || rc=4
    fi
    if [ "$rc" -eq 0 ]; then
        for _ in $(seq 1 180); do
            if [ -f "$processes/carla.exit" ] || ! kill -0 "$carla_pid" 2>/dev/null; then break; fi
            if "$PY_BIN" - "$WORLD_PORT" <<'PY' >/dev/null 2>&1
import socket, sys
s=socket.socket(); s.settimeout(.2)
try: s.connect(('127.0.0.1', int(sys.argv[1])))
finally: s.close()
PY
            then ready=1; break; fi
            sleep 1
        done
        [ "$ready" -eq 1 ] || rc=4
    fi
    if [ "$rc" -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU setsid "$PY_BIN" "$SUPERVISOR" --supervise \
            "$processes/evaluator.exit" "$processes/evaluator.identity.json" \
            "$PY_BIN" "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_parameter.py" \
            --scenarios="$SCENARIOS" --scenario_parameter="$SCENARIOS_PARAMETER" \
            --routes="$ROUTES" --repetitions=1 --track=SENSORS --checkpoint="$checkpoint" \
            --agent="$TEAM_AGENT" --agent-config="$config" --debug=0 --record="$RECORD_PATH" \
            --resume=0 --port="$WORLD_PORT" --trafficManagerPort="$TRAFFIC_PORT" \
            --carlaProviderSeed=2000 --trafficManagerSeed=2000 --ego-num=1 --timeout 600 \
            --skip_existed=0 >"$logs/evaluator.log" 2>&1 &
        eval_pid=$!
        eval_token=$("$PY_BIN" "$SUPERVISOR" --process-start-token "$eval_pid" "$$") || rc=5
        ACTIVE_EVAL_PID=$eval_pid; ACTIVE_EVAL_TOKEN=$eval_token
        if [ "$rc" -eq 0 ]; then
            "$PY_BIN" "$SUPERVISOR" --await-owned "$processes/evaluator.identity.json" \
                "$eval_pid" "$eval_token" "$$" 5 || rc=5
        fi
        if [ "$rc" -eq 0 ]; then
            while [ ! -f "$processes/evaluator.exit" ]; do
                if ! kill -0 "$eval_pid" 2>/dev/null || [ -f "$processes/carla.exit" ]; then rc=5; break; fi
                sleep 1
            done
            if [ -f "$processes/evaluator.exit" ]; then rc=$(cat "$processes/evaluator.exit"); fi
        fi
    fi
    stop_active_task
    local result_source="" audit_source=""
    if ! result_source=$("$PY_BIN" -m scripts.ditl_packet_loss_result \
            --resolve-ego-result "$checkpoint"); then
        result_source=""
        [ "$rc" -ne 0 ] || rc=5
    fi
    mapfile -t audits < <(find "$save_path" -type f -name ditl_event_audit.jsonl -print 2>/dev/null)
    [ "${#audits[@]}" -eq 1 ] && audit_source=${audits[0]}
    write_evidence "$attempt_dir" "$task_key" "$route" "$scenario" "$profile" \
        "$config" "$rc" "$result_source" "$audit_source"
    chmod -R a-w "$attempt_dir"
    return 0
}

batch_rc=0
for index in "${!TASK_KEYS[@]}"; do
    if ! run_task "${TASK_KEYS[$index]}" "${ROUTES_LIST[$index]}" \
        "${SCENARIOS_LIST[$index]}" "${PROFILES_LIST[$index]}" "${CONFIGS_LIST[$index]}"; then
        batch_rc=1
    fi
done
exit "$batch_rc"
