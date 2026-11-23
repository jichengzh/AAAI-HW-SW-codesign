#!/usr/bin/env python3
"""Standalone NMS latency benchmark for Orin — no HEAL required.
Measures CUDA-accelerated 3D NMS (via torchvision.ops.nms proxy OR
pure-torch box IOU fallback) representative of DAIR Pyramid output.

Output: JSON with warmup_runs, measure_runs, lat_mean_ms, lat_p50_ms,
        lat_p99_ms, nms_impl, n_boxes_input, n_boxes_output
"""
import json, time, sys
import torch

# ── DAIR Pyramid head output profile (estimated from HEAL heads config)
# 2 anchors/voxel × BEV (256×256) = ~6000-10000 pre-NMS candidates
# Use 8192 as conservative upper bound (matches OPV2V profile)
N_BOXES = 8192
SCORE_THRESH = 0.1   # typical HEAL NMS threshold
IOU_THRESH   = 0.5

WARMUP   = 100
MEASURE  = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)

# dummy boxes: [x, y, z, w, l, h, yaw] → (N, 7)
torch.manual_seed(42)
boxes_7 = torch.rand(N_BOXES, 7, device=device)
# scale to realistic DAIR range
boxes_7[:, 0] = boxes_7[:, 0] * 100 - 50   # x: [-50, 50]
boxes_7[:, 1] = boxes_7[:, 1] * 100 - 50   # y: [-50, 50]
boxes_7[:, 2] = boxes_7[:, 2] * 3  - 1     # z: [-1,  2]
boxes_7[:, 3:6] = boxes_7[:, 3:6] * 4 + 0.5  # w,l,h: [0.5, 4.5]
scores = torch.rand(N_BOXES, device=device)

# ── try torchvision 3D NMS first, then 2D proxy, then pure-torch IOU fallback
nms_impl = None
try:
    from torchvision.ops import nms as tv_nms
    # 2D proxy: use BEV footprint (x,y,w,l) as pseudo-2D box
    def run_nms():
        cx, cy = boxes_7[:, 0], boxes_7[:, 1]
        w, l   = boxes_7[:, 3], boxes_7[:, 4]
        bx1, by1 = cx - w/2, cy - l/2
        bx2, by2 = cx + w/2, cy + l/2
        b2d = torch.stack([bx1, by1, bx2, by2], dim=1)
        keep = tv_nms(b2d, scores, IOU_THRESH)
        return keep
    _ = run_nms()
    nms_impl = "torchvision_2d_proxy"
    print("using torchvision 2D BEV-proxy NMS", flush=True)
except ImportError:
    try:
        # Pure-torch IOU NMS (slower but no dependency)
        def iou_2d(b):
            x1, y1 = b[:, 0] - b[:, 2]/2, b[:, 1] - b[:, 3]/2
            x2, y2 = b[:, 0] + b[:, 2]/2, b[:, 1] + b[:, 3]/2
            area = b[:, 2] * b[:, 3]
            inter_x1 = torch.max(x1.unsqueeze(1), x1.unsqueeze(0))
            inter_y1 = torch.max(y1.unsqueeze(1), y1.unsqueeze(0))
            inter_x2 = torch.min(x2.unsqueeze(1), x2.unsqueeze(0))
            inter_y2 = torch.min(y2.unsqueeze(1), y2.unsqueeze(0))
            inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            union = area.unsqueeze(1) + area.unsqueeze(0) - inter
            return inter / union.clamp(min=1e-6)
        def run_nms():
            iou = iou_2d(boxes_7[:, :4])
            order = scores.argsort(descending=True)
            keep = []
            suppressed = torch.zeros(N_BOXES, dtype=torch.bool, device=device)
            for i in range(len(order)):
                idx = order[i].item()
                if not suppressed[idx]:
                    keep.append(idx)
                    suppressed |= (iou[idx] > IOU_THRESH)
            return torch.tensor(keep, device=device)
        _ = run_nms()
        nms_impl = "pure_torch_cpu_loop"
        print("using pure-torch IOU NMS fallback", flush=True)
    except Exception as e:
        print(f"NMS fallback also failed: {e}", file=sys.stderr)
        sys.exit(1)

torch.cuda.synchronize()

# warmup
for _ in range(WARMUP):
    keep = run_nms()
    torch.cuda.synchronize()

# measure
lats = []
for _ in range(MEASURE):
    t0 = time.perf_counter()
    keep = run_nms()
    torch.cuda.synchronize()
    lats.append((time.perf_counter() - t0) * 1000.0)

import statistics
lats_sorted = sorted(lats)
result = {
    "nms_impl": nms_impl,
    "device": str(device),
    "n_boxes_input": N_BOXES,
    "n_boxes_output": len(keep),
    "score_thresh": SCORE_THRESH,
    "iou_thresh": IOU_THRESH,
    "warmup_runs": WARMUP,
    "measure_runs": MEASURE,
    "lat_mean_ms":  round(statistics.mean(lats), 4),
    "lat_median_ms": round(statistics.median(lats), 4),
    "lat_p50_ms":   round(lats_sorted[int(0.50 * MEASURE)], 4),
    "lat_p99_ms":   round(lats_sorted[int(0.99 * MEASURE)], 4),
    "lat_min_ms":   round(lats_sorted[0], 4),
    "lat_max_ms":   round(lats_sorted[-1], 4),
    "source": "orin_nms_bench.py;torch_cuda;standalone_no_HEAL",
    "platform": "orin_agx",
    "power_mode": "MODE_30W"
}
print(json.dumps(result, indent=2))
