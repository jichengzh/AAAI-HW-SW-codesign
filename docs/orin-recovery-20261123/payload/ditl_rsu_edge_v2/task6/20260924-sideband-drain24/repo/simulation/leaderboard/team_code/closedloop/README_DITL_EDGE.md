# DitL RSU edge model access and communication

This module implements the corrected Deployment-in-the-Loop measurement path
for an RSU perception model on Jetson AGX Orin. Development stays in the
`feature/deployment-in-the-loop` worktree. The v2 deployment is isolated at:

```text
/home/jichengzhi/ditl_rsu_edge_v2
```

## Formal measurement boundary

The formal data source is local to the Orin. The server never uploads
perception tensors. An optional control command may start a run, but its time is
testbed overhead and is excluded from the perception metric.

```text
Orin-local RSU frame available
  -> local read and preprocessing
  -> H2D
  -> TensorRT inference
  -> D2H and identity postprocess
  -> pack, hash, optional compression and protocol serialization
  -> localhost result handoff to the Orin ROS 2 publisher
  -> ROS message construction and publish
  -> ROS receive, frame parse, decompression, verification and reconstruction
  -> fusion-ready
```

The main metric is `rsu_to_fusion_latency`. It is a one-way result-age
measurement and is never reported as RTT. `ditl.trace.v2` records each stage
separately. A cross-host total is headline eligible only when both hosts have
comparable PTP clocks. The verified same-host Orin containers share the host
kernel `CLOCK_MONOTONIC_RAW` and boot identity.

The current source is `local_feature_replay`: it replays model input features
already stored on the Orin. It does not claim physical LiDAR capture or raw
point-cloud preprocessing. The current TensorRT engine returns dense heads, so
the trace labels its payload `dense_head_debug` and its postprocess `identity`.

## Components

- `ditl_rsu_source.py` reads a fixed local manifest and rejects remote paths,
  pickle data, shape mismatches, and dtype mismatches.
- `ditl_rsu_pipeline.py` performs one local inference per sequence and records
  edge stages.
- `ditl_trt_backend.py` records input validation, H2D, CUDA-event inference,
  and D2H separately.
- `ditl_edge_protocol.py` supports bounded raw, LZ4, and legacy zlib payloads.
- `scripts/ditl_rsu_edge_publisher.py` owns the local source and fixed engine.
- `scripts/ditl_ros2_edge_bridge.py result-publisher` publishes result data and
  a small timestamp sideband from the Orin.
- `scripts/ditl_ros2_edge_bridge.py result-sink` receives the result once and
  writes the v2 trace. It bounds unmatched result and sideband count, result
  bytes, and sideband size so malformed ROS traffic cannot grow memory without
  limit.
- `scripts/ditl_summarize_stages.py` reports p50, p90, p95, p99, mean, extrema,
  payload size, stage share, bottleneck, and stage-sum residual.

## Codec policy

The formal default is `--compression none` on the current 1 GbE testbed. It
removes the Python zlib hot path and preserves every float32 output exactly.
Bounded lossless LZ4 is the bandwidth-oriented comparison. It uses the existing
`liblz4.so.1` with fixed acceleration 2, records that value in the bundle,
validates the declared raw size before allocation, and verifies SHA-256 after
decompression. Zlib remains available only for legacy comparison.

The pre-correction Orin microbenchmark for the 2,621,440-byte dense output was:

| Codec | Compression median | Wire payload |
| --- | ---: | ---: |
| zlib level 6 | 575.46 ms | 1,283,815 bytes |
| zlib level 1 | 78.41 ms | 1,389,363 bytes |
| LZ4 default | 8.76 ms | 1,967,436 bytes |
| none | no compression stage | 2,621,440 bytes |

## Orin formal results

The paired runs used source commit `f4b8c61`, 20 warmup frames, and 100
measured frames per codec. Every measured record was headline eligible in the
same-host `CLOCK_MONOTONIC_RAW` domain. `remote_input_tensor_bytes` was zero,
and `cls_preds`, `reg_preds`, and `dir_preds` matched the fixed TensorRT
reference exactly (`max_abs=0`). Times below are milliseconds.

| Stage or total | Raw p50 | Raw p95 | LZ4 p50 | LZ4 p95 |
| --- | ---: | ---: | ---: | ---: |
| Local source read | 4.282 | 6.513 | 5.080 | 5.268 |
| Preprocess / input validation | 2.212 | 3.418 | 2.037 | 2.125 |
| H2D | 3.808 | 4.593 | 3.905 | 3.950 |
| GPU inference | 49.822 | 50.255 | 50.273 | 50.372 |
| D2H | 2.511 | 2.869 | 2.646 | 2.903 |
| Identity postprocess | 0.008 | 0.012 | 0.011 | 0.013 |
| Pack | 0.614 | 1.049 | 0.949 | 1.017 |
| SHA-256 | 2.503 | 5.732 | 4.672 | 5.187 |
| Compression | 0.000 | 0.000 | 13.198 | 13.318 |
| Complete encode | 3.394 | 7.300 | 18.890 | 19.571 |
| Local result handoff | 3.658 | 5.144 | 2.287 | 3.269 |
| ROS message construction | 0.555 | 0.891 | 0.480 | 0.546 |
| ROS publish call | 1.339 | 6.277 | 0.882 | 0.990 |
| Receiver decode | 5.020 | 9.980 | 11.377 | 13.644 |
| RSU device pipeline | 79.354 | 85.496 | 90.251 | 91.194 |
| ROS and network interval | 4.101 | 6.743 | 2.914 | 3.119 |
| `rsu_to_fusion_latency` | 85.281 | 99.539 | 104.080 | 107.206 |

GPU inference is the largest individual edge stage in both runs. LZ4 reduced
the wire payload from 2,621,440 to 2,043,030 bytes; its compression p95 was
13.318 ms and complete encoding p95 was 19.571 ms, both below GPU inference
p50. The p95 stage-sum residuals were 4.197 ms for raw and 3.168 ms for LZ4,
within 5% of their device-pipeline p95 values.

Artifacts are stored on the Orin and mirrored locally:

```text
/home/jichengzhi/ditl_rsu_edge_v2/run/formal_raw_v2
/home/jichengzhi/ditl_rsu_edge_v2/run/formal_lz4_v2
/var/tmp/codriving-ditl/task3-rsu-local-v2/formal_raw_v2
/var/tmp/codriving-ditl/task3-rsu-local-v2/formal_lz4_v2
```

Security hardening commit `1586778` added bounded unmatched ROS result state,
sideband size and identity checks, a non-loopback allowlist requirement for the
legacy service, and archive-safe tensor names. A subsequent 120-frame raw ROS
integration run retained exact outputs and zero remote input bytes at:

```text
/home/jichengzhi/ditl_rsu_edge_v2/run/security_smoke_raw_v2
```

## Canonical formal commands

Create one manifest under the v2 deployment root:

```json
{
  "spatial_features": {
    "path": "p03_build/oc_inputs_8.npy", "frame_axis": 0, "frame_index": 0
  },
  "t_ego": {
    "path": "p03_build/oc_tego_8.npy", "frame_axis": 0, "frame_index": 0
  }
}
```

Start the ROS result sink and result publisher in the tracing-enabled Foxy
image with host network and host IPC:

```bash
python3 scripts/ditl_ros2_edge_bridge.py result-sink \
  --max-results 120 --receiver-host-id orin-agx-01 \
  --trace-jsonl /artifacts/trace.jsonl --output-npz /artifacts/output.npz

python3 scripts/ditl_ros2_edge_bridge.py result-publisher \
  --bind 127.0.0.1 --result-port 55021 --max-results 120
```

Then run the native TensorRT process. It sends results to the localhost
publisher and sends zero remote input tensor bytes:

```bash
/home/jichengzhi/miniforge3/envs/uniad/bin/python \
  scripts/ditl_rsu_edge_publisher.py \
  --data-root /home/jichengzhi/m4_8_orin \
  --manifest /home/jichengzhi/ditl_rsu_edge_v2/manifest.json \
  --engine /home/jichengzhi/m4_8_orin/models/pyramid_dair_m1_collab_n2_fp16_orin.engine \
  --run-id rsu-local-raw-v2 --count 120 --warmup-count 20 \
  --compression none --result-host 127.0.0.1 --result-port 55021
```

Repeat with `--compression lz4` for the paired codec comparison. Summarize the
100 measured frames after excluding the 20 warmups:

```bash
python3 scripts/ditl_summarize_stages.py \
  --trace-jsonl /artifacts/trace.jsonl \
  --output-json /artifacts/summary.json --warmup-count 20
```

## Verified hardware and model

- Jetson: `jichengzhi@172.16.62.222:22`, AGX Orin, L4T R35.6.5.
- Native stack: Python 3.8, CUDA 11.4, TensorRT 8.5.2.2.
- ROS stack: pinned arm64 Foxy container with tracing enabled.
- Engine: `pyramid_dair_m1_collab_n2_fp16_orin.engine`.
- Engine SHA-256:
  `a4a89392033e39fd6e754b9c1caff2b0a3c87cc4f035c6ef73fe15322af476e0`.
- Inputs: `spatial_features (2,64,128,256) float32` and
  `t_ego (2,2,3) float32`.
- Outputs: `cls_preds (1,2,128,256)`, `reg_preds (1,14,128,256)`, and
  `dir_preds (1,4,128,256)`, all float32.

## Legacy connectivity measurements

The v1 client-driven TCP and ROS request/response runs are retained as a
**legacy testbed probe**. They uploaded the full model input to the Orin and
therefore do not measure the RSU-local deployment path. Their 0.90-0.97 second
round-trip values must not be used as `rsu_to_fusion_latency`.

Legacy artifacts remain under:

```text
/home/jichengzhi/ditl_rsu_edge_v1/run
```

## Cleanup

After a run, stop the v2 publisher and sink containers, remove their temporary
containers, and verify that ports 55021 and ROS processes are no longer active.
Do not alter the v1 deployment, active V2Xverse experiment worktrees, firewall,
or system packages.
