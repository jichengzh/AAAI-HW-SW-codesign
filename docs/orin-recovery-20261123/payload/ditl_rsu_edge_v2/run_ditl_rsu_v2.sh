#!/usr/bin/env bash
set -euo pipefail

codec="$1"
run_id="$2"
ros_domain_id="$3"
root=/home/jichengzhi/ditl_rsu_edge_v2
repo="$root/repo"
run_dir="$root/run/$run_id"
image=ditl/ros-foxy-tracing-enabled:arm64
sink_name="ditl-v2-${codec}-sink"
publisher_name="ditl-v2-${codec}-publisher"

cleanup() {
  docker rm -f "$sink_name" "$publisher_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup
rm -rf "$run_dir"
mkdir -p "$run_dir"

cat > "$run_dir/command_manifest.txt" <<EOF
source_commit=$(cat "$root/SOURCE_COMMIT")
codec=$codec
run_id=$run_id
ros_domain_id=$ros_domain_id
warmup_frames=20
measured_frames=100
remote_input_tensor_bytes=0
EOF

docker run -d \
  --name "$sink_name" \
  --network host --ipc host \
  -e ROS_DOMAIN_ID="$ros_domain_id" -e ROS_LOCALHOST_ONLY=1 \
  -v "$repo:/workspace/V2Xverse:ro" \
  -v "$run_dir:/artifacts" \
  "$image" bash -lc \
  "source /opt/ros/foxy/setup.bash; python3 /workspace/V2Xverse/scripts/ditl_ros2_edge_bridge.py result-sink --max-results 120 --timeout-s 600 --receiver-host-id orin-agx-01 --trace-jsonl /artifacts/trace.jsonl --output-npz /artifacts/output.npz --ready-file /artifacts/sink_ready.json" \
  > "$run_dir/sink_container_id.txt"

for _ in $(seq 1 100); do
  test -f "$run_dir/sink_ready.json" && break
  sleep 0.1
done
test -f "$run_dir/sink_ready.json"

docker run -d \
  --name "$publisher_name" \
  --network host --ipc host \
  -e ROS_DOMAIN_ID="$ros_domain_id" -e ROS_LOCALHOST_ONLY=1 \
  -v "$repo:/workspace/V2Xverse:ro" \
  -v "$run_dir:/artifacts" \
  "$image" bash -lc \
  "source /opt/ros/foxy/setup.bash; python3 /workspace/V2Xverse/scripts/ditl_ros2_edge_bridge.py result-publisher --bind 127.0.0.1 --result-port 55021 --max-results 120 --discovery-timeout-s 60 --socket-timeout-s 600 --ready-file /artifacts/publisher_ready.json" \
  > "$run_dir/publisher_container_id.txt"

for _ in $(seq 1 100); do
  test -f "$run_dir/publisher_ready.json" && break
  sleep 0.1
done
test -f "$run_dir/publisher_ready.json"

/home/jichengzhi/miniforge3/envs/uniad/bin/python \
  "$repo/scripts/ditl_rsu_edge_publisher.py" \
  --data-root /home/jichengzhi/m4_8_orin \
  --manifest "$root/manifest.json" \
  --engine /home/jichengzhi/m4_8_orin/models/pyramid_dair_m1_collab_n2_fp16_orin.engine \
  --run-id "$run_id" --count 120 --warmup-count 20 \
  --compression "$codec" --result-host 127.0.0.1 --result-port 55021 \
  --host-id orin-agx-01 --output-npz "$run_dir/native_last_output.npz" \
  > "$run_dir/native.log" 2>&1

publisher_status=$(docker wait "$publisher_name")
sink_status=$(docker wait "$sink_name")
docker logs "$publisher_name" > "$run_dir/publisher.log" 2>&1 || true
docker logs "$sink_name" > "$run_dir/sink.log" 2>&1 || true
printf '%s\n' "$publisher_status" > "$run_dir/publisher.exit"
printf '%s\n' "$sink_status" > "$run_dir/sink.exit"
test "$publisher_status" = 0
test "$sink_status" = 0

/home/jichengzhi/miniforge3/envs/uniad/bin/python \
  "$repo/scripts/ditl_summarize_stages.py" \
  --trace-jsonl "$run_dir/trace.jsonl" \
  --output-json "$run_dir/summary.json" --warmup-count 20 \
  > "$run_dir/summary.log"

/home/jichengzhi/miniforge3/envs/uniad/bin/python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

run_dir = Path(sys.argv[1])
expected_path = Path('/home/jichengzhi/m4_8_orin/p03_build/oc_outputs_orin_base_fp16.npz')
result = {}
with np.load(str(expected_path), allow_pickle=False) as expected, np.load(str(run_dir / 'output.npz'), allow_pickle=False) as actual:
    for name in sorted(actual.files):
        expected_value = expected[name]
        if expected_value.shape != actual[name].shape:
            expected_value = expected_value[0]
        difference = np.abs(actual[name].astype(np.float64) - expected_value.astype(np.float64))
        result[name] = {
            'shape': list(actual[name].shape),
            'max_abs': float(difference.max()) if difference.size else 0.0,
            'exact': bool(np.array_equal(actual[name], expected_value)),
        }
(run_dir / 'output_comparison.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
if not result or not all(item['exact'] for item in result.values()):
    raise SystemExit('output mismatch')
PY

{
  uname -a
  cat /etc/nv_tegra_release
  /usr/sbin/nvpmodel -q 2>/dev/null || true
} > "$run_dir/device_snapshot.txt"

cd "$run_dir"
find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\0' | sort -z | xargs -0 sha256sum > SHA256SUMS
printf 'run_dir=%s\n' "$run_dir"
cat summary.log
cat output_comparison.json
