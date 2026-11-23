#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/foxy/setup.bash
source /opt/ros2-tracing-overlay/local_setup.bash
set -u
out=/workspace/run/ros2_trace_on_v1
mkdir -p "$out"
lttng create ditl-task3-v1 --output="$out/lttng"
lttng enable-event --userspace "ros2:*"
lttng add-context --userspace --type=vpid --type=vtid --type=procname
lttng start
python3 scripts/ditl_ros2_edge_bridge.py edge-relay --edge-host 127.0.0.1 --edge-port 55020 --max-requests 4 --timeout-s 120 --max-payload-mib 128 >"$out/relay.log" 2>&1 &
relay_pid=$!
sleep 2
for i in 0 1 2 3; do
  python3 scripts/ditl_ros2_edge_bridge.py probe-client \
    --input-npz run/ros2_smoke_v1/pyramid_input_real.npz \
    --output-npz "$out/output_${i}.npz" \
    --trace-jsonl "$out/trace_${i}.jsonl" \
    --latency-csv "$out/latency_${i}.csv" \
    --request-id "ros-trace-on-${i}" \
    --run-id task3-orin-ros2-trace-on-v1 \
    --route-id pyramid-dair-m1-collab-n2 \
    --source-id rsu-orin-agx-01 \
    --target-id server-v2xverse-01 \
    --frame-id "$i" --sequence-id "$i" \
    --observer-host-id jetson-orin-agx-01 \
    --discovery-timeout-s 15 --timeout-s 120 --max-payload-mib 128
done
wait "$relay_pid"
lttng stop
lttng destroy ditl-task3-v1
