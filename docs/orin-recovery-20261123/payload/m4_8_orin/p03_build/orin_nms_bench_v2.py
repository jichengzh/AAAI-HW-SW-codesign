#!/usr/bin/env python3
"""Standalone NMS latency benchmark for Orin v2 — no HEAL, no torchvision import.
Uses torchvision.ops C++ extension directly (avoids datasets/urllib3 path).
Falls back to pure-torch IOU loop if ops also unavailable.
Representative of DAIR Pyramid post-threshold detection count (~300-500 boxes).
"""
import json, time, sys, statistics
import torch

# ── Profile: DAIR Pyramid output after score threshold
# ~300-500 candidates survive score>0.1 per frame (2-agent BEV 256x256)
N_PRE_THRESH = 8192
SCORE_THRESH = 0.1
IOU_THRESH   = 0.5
WARMUP  = 200
MEASURE = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)
torch.manual_seed(42)

# Create representative boxes (BEV 2D footprint, DAIR spatial range)
boxes_all = torch.rand(N_PRE_THRESH, 4, device=device)
boxes_all[:, 0] = boxes_all[:, 0] * 100 - 50   # cx
boxes_all[:, 1] = boxes_all[:, 1] * 100 - 50   # cy
boxes_all[:, 2] = boxes_all[:, 2] * 3 + 0.5    # w
boxes_all[:, 3] = boxes_all[:, 3] * 3 + 0.5    # h
scores_all = torch.rand(N_PRE_THRESH, device=device)

# Apply score threshold → ~N_PRE_THRESH * SCORE_THRESH * (1-SCORE_THRESH) subset
mask = scores_all > SCORE_THRESH
boxes  = boxes_all[mask].contiguous()
scores = scores_all[mask].contiguous()
n_post = len(boxes)
print(f"post-thresh boxes: {n_post}", flush=True)

# Convert cx,cy,w,h → x1,y1,x2,y2 for torchvision
boxes_xyxy = torch.stack([
    boxes[:, 0] - boxes[:, 2]/2,
    boxes[:, 1] - boxes[:, 3]/2,
    boxes[:, 0] + boxes[:, 2]/2,
    boxes[:, 1] + boxes[:, 3]/2,
], dim=1).contiguous()

# ── Try torchvision C++ ops directly (skip top-level import)
nms_impl = None
try:
    import importlib.util
    # Try the ops submodule path directly
    tv_egg = "/home/jichengzhi/.local/lib/python3.8/site-packages/torchvision-0.13.0a0+da3794e-py3.8-linux-aarch64.egg"
    if tv_egg not in sys.path:
        sys.path.insert(0, tv_egg)
    from torchvision.ops import nms as tv_nms
    def run_nms():
        return tv_nms(boxes_xyxy, scores, IOU_THRESH)
    _ = run_nms()
    nms_impl = "torchvision.ops.nms_cuda"
    print("torchvision.ops.nms loaded OK", flush=True)
except Exception as e1:
    print(f"torchvision.ops failed ({e1}), trying torch._C._torchvision_nms...", flush=True)
    try:
        # direct C function
        _nms = torch.ops.torchvision.nms
        def run_nms():
            return _nms(boxes_xyxy, scores, IOU_THRESH)
        _ = run_nms()
        nms_impl = "torch.ops.torchvision.nms"
        print("torch.ops.torchvision.nms OK", flush=True)
    except Exception as e2:
        print(f"also failed ({e2}), using pure-torch greedy NMS", flush=True)
        # Pure-torch greedy NMS (CPU-assisted)
        def box_iou(a, b):
            ax1,ay1 = a[:,0],a[:,1]; ax2,ay2 = a[:,2],a[:,3]
            bx1,by1 = b[:,0],b[:,1]; bx2,by2 = b[:,2],b[:,3]
            ix1 = torch.max(ax1.unsqueeze(1), bx1.unsqueeze(0))
            iy1 = torch.max(ay1.unsqueeze(1), by1.unsqueeze(0))
            ix2 = torch.min(ax2.unsqueeze(1), bx2.unsqueeze(0))
            iy2 = torch.min(ay2.unsqueeze(1), by2.unsqueeze(0))
            inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
            a_area = (ax2-ax1)*(ay2-ay1); b_area = (bx2-bx1)*(by2-by1)
            union = a_area.unsqueeze(1) + b_area.unsqueeze(0) - inter
            return inter / union.clamp(1e-6)
        def run_nms():
            iou = box_iou(boxes_xyxy, boxes_xyxy)
            order = scores.argsort(descending=True)
            keep = []; suppressed = set()
            for idx in order.tolist():
                if idx not in suppressed:
                    keep.append(idx)
                    suppressed.update(
                        (iou[idx] > IOU_THRESH).nonzero(as_tuple=False).flatten().tolist())
            return torch.tensor(keep, device=device)
        _ = run_nms()
        nms_impl = "pure_torch_greedy_loop"

torch.cuda.synchronize()

# warmup
for _ in range(WARMUP):
    _ = run_nms()
    torch.cuda.synchronize()

# measure
lats = []
for _ in range(MEASURE):
    t0 = time.perf_counter()
    keep = run_nms()
    torch.cuda.synchronize()
    lats.append((time.perf_counter() - t0) * 1000.0)

lats_sorted = sorted(lats)
result = {
    "nms_impl": nms_impl,
    "device": str(device),
    "n_boxes_pre_thresh": N_PRE_THRESH,
    "n_boxes_post_thresh": n_post,
    "n_boxes_output": int(keep.shape[0]) if hasattr(keep, 'shape') else len(keep),
    "score_thresh": SCORE_THRESH,
    "iou_thresh": IOU_THRESH,
    "warmup_runs": WARMUP,
    "measure_runs": MEASURE,
    "lat_mean_ms":   round(statistics.mean(lats), 4),
    "lat_median_ms": round(statistics.median(lats), 4),
    "lat_p50_ms":    round(lats_sorted[int(0.50 * MEASURE)], 4),
    "lat_p99_ms":    round(lats_sorted[int(0.99 * MEASURE)], 4),
    "lat_min_ms":    round(lats_sorted[0], 4),
    "lat_max_ms":    round(lats_sorted[-1], 4),
    "source": "orin_nms_bench_v2.py;standalone_no_HEAL;torch1.12+cuda",
    "platform": "orin_agx",
    "power_mode": "MODE_30W",
    "caveat": "2D BEV proxy NMS, not HEAL 3D NMS; representative of post-threshold box count"
}
print(json.dumps(result, indent=2))
