#!/bin/bash
# Evaluate one manifest task while preserving caller-provided seeds.

set -u
set -o pipefail

ROUTE=${1:?need route}
PORT=${2:?need CARLA port}
TAG=${3:?need result tag}
REPEAT=${4:?need repeat}
AGENT=${5:?need agent config stem}
SCENARIO=${6:?need scenario suffix}
EVAL_SETTING=${7:?need unique eval setting}
CARLA_SEED=${8:-2000}
TRAFFIC_SEED=${9:-2000}

export CARLA_ROOT=external_paths/carla_root
export LEADERBOARD_ROOT=simulation/leaderboard
export SCENARIO_RUNNER_ROOT=simulation/scenario_runner
export DATA_ROOT=external_paths/data_root
export SAVE_DIR=results
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=${PYTHONPATH:-}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:${LEADERBOARD_ROOT}:${LEADERBOARD_ROOT}/team_code:${SCENARIO_RUNNER_ROOT}
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT
export TM_PORT=$((PORT + 5))
export DEBUG_CHALLENGE=0
export CARLA_SEED
export TRAFFIC_SEED
export REPETITIONS=1
export ROUTES=${LEADERBOARD_ROOT}/data/evaluation_routes/town05_short_r${ROUTE}.xml
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town05_all_scenarios_2.json
export SCENARIOS_PARAMETER=${LEADERBOARD_ROOT}/leaderboard/scenarios/scenario_parameter${SCENARIO}.yaml
export RESULT_ROOT=${SAVE_DIR}/results_driving_${TAG}
export EVAL_SETTING
export CHECKPOINT_ENDPOINT=${RESULT_ROOT}/${EVAL_SETTING}/results.json
export SAVE_PATH=${RESULT_ROOT}/image/${EVAL_SETTING}
export TEAM_AGENT=simulation/leaderboard/team_code/pnp_agent_e2e.py
export TEAM_CONFIG=simulation/leaderboard/team_code/agent_config/pnp_config_${AGENT}.yaml
export RESUME=0
export EGO_NUM=1
# The outer manifest accepts only final Completed/Failed checkpoints.  Keeping
# evaluator-level skipping off prevents stale Started checkpoints from blocking
# recovery after an infrastructure interruption.
export SKIP_EXISTED=0
export RECORD_PATH=${RECORD_PATH:-}

mkdir -p "$SAVE_PATH" "${RESULT_ROOT}/${EVAL_SETTING}"

python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_parameter.py \
    --scenarios="${SCENARIOS}" \
    --scenario_parameter="${SCENARIOS_PARAMETER}" \
    --routes="${ROUTES}" \
    --repetitions="${REPETITIONS}" \
    --track="${CHALLENGE_TRACK_CODENAME}" \
    --checkpoint="${CHECKPOINT_ENDPOINT}" \
    --agent="${TEAM_AGENT}" \
    --agent-config="${TEAM_CONFIG}" \
    --debug="${DEBUG_CHALLENGE}" \
    --record="${RECORD_PATH}" \
    --resume="${RESUME}" \
    --port="${PORT}" \
    --trafficManagerPort="${TM_PORT}" \
    --carlaProviderSeed="${CARLA_SEED}" \
    --trafficManagerSeed="${TRAFFIC_SEED}" \
    --ego-num="${EGO_NUM}" \
    --timeout 600 \
    --skip_existed="${SKIP_EXISTED}"
