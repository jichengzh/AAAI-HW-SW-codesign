#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect RSU latency trace from native CoDriving perception model forward passes.

§3.8 R-1 implementation. Runs CoDriving perception model forward on an input source
(see --input_source) with CUDA-Event timing on a dedicated GPU (算力隔离: must be
different from CARLA simulation GPU).

=== INPUT SOURCE OPTIONS (--input_source) ===
  sampled   [②候选, default]
            Each call independently samples N_voxels from a realistic per-frame
            distribution (LogNormal fit to CARLA RSU scenes: mu=9.8, sigma=0.4
            → median~18k, p5~12k, p95~27k). This gives real per-call τ variation
            driven by varying batch size, without requiring a live simulation run.

  npy_dir   [①候选, requires --npy_dir <path>]
            Load real lidar frames captured by process C during a prior simulation
            run (collect_sim_lidar.py or --save_lidar in eval_driving_e2e.sh).
            Each call feeds the next real frame tensor, cycling when exhausted.
            → Best option for true scene-density-driven τ variation.
            → Requires a prior sim run with lidar saving enabled.

=== OUTPUT ===
  CSV: call_idx,tau_ms     (warmup calls EXCLUDED from output, §3.8 team-lead §4)
  口径: CUDA-Event forward-only (does NOT include voxelization / IPC / ZMQ)
  Label: latency_ms_source=trace_replay_codriving_4090

=== USAGE ===
  # Option ②: sampled variable N (no prior sim needed):
  CUDA_VISIBLE_DEVICES=<infer_gpu> python scripts/collect_rsu_latency_trace.py \
      --input_source sampled --n_calls 500 --warmup 20 \
      --out results/rsu_latency_trace.csv

  # Option ①: real lidar frames from prior sim (pending GO):
  CUDA_VISIBLE_DEVICES=<infer_gpu> python scripts/collect_rsu_latency_trace.py \
      --input_source npy_dir --npy_dir <path_to_lidar_npy> \
      --n_calls 500 --warmup 20

Design reference: sim_test_design_v1.md §3.8 R-1
"""

import argparse
import csv
import os
import sys
import time

import torch
import numpy as np

# ── locate V2Xverse root ────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_V2X_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _V2X_ROOT)

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_OUT = os.path.join(_V2X_ROOT, 'results', 'rsu_latency_trace.csv')
DEFAULT_PERCEPTION_DIR = os.path.join(_V2X_ROOT, 'checkpoints', 'codriving', 'perception')

# ── voxel grid constants (from checkpoints/codriving/perception/config.yaml) ─
# cav_lidar_range: [-36, -12, -22, 36, 12, 14]
# voxel_size: [0.125, 0.125, 4]  →  grid: x=576, y=192, z=9
# max_voxel_test: 40000,  max_points_per_voxel: 32,  features: 4
VOXEL_GRID = dict(x=576, y=192, z=9)
MAX_POINTS  = 32
FEAT_DIM    = 4
RECORD_LEN  = 2   # ego + 1 RSU

# ── realistic per-frame N_voxels distribution (option ②: sampled) ────────────
# Empirical estimate for V2Xverse RSU scene density:
#   LogNormal(mu=9.8, sigma=0.4) → median≈18k, p5≈12k, p95≈27k
# Clipped to [4000, 39000] to stay within model's max_voxel_test limit.
DIST_MU     = 9.8
DIST_SIGMA  = 0.4
DIST_N_MIN  = 4_000
DIST_N_MAX  = 39_000


# ─────────────────────────────────────────────────────────────────────────────
# Batch builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_voxel_batch(n_voxels_per_agent, device):
    """Construct a CoDriving perception batch dict with `n_voxels_per_agent`
    voxels for each of RECORD_LEN agents (ego + RSU).

    Tensor shapes match what the perception dataloader builds:
      voxel_features:   [total, MAX_POINTS, FEAT_DIM]
      voxel_coords:     [total, 4]  (batch_idx=0, z, y, x)
      voxel_num_points: [total]
      record_len:       [1]
    """
    total = n_voxels_per_agent * RECORD_LEN
    vf = torch.rand(total, MAX_POINTS, FEAT_DIM, dtype=torch.float32, device=device)
    vc = torch.stack([
        torch.zeros(total, dtype=torch.int32, device=device),
        torch.randint(0, VOXEL_GRID['z'], (total,), device=device).int(),
        torch.randint(0, VOXEL_GRID['y'], (total,), device=device).int(),
        torch.randint(0, VOXEL_GRID['x'], (total,), device=device).int(),
    ], dim=1)
    vnp = torch.randint(1, MAX_POINTS + 1, (total,), device=device).int()
    rl  = torch.tensor([RECORD_LEN], dtype=torch.long, device=device)
    return {'voxel_features': vf, 'voxel_coords': vc,
            'voxel_num_points': vnp, 'record_len': rl}


class SampledInputSource:
    """Option ②: each call independently samples N_voxels from LogNormal dist."""

    def __init__(self, rng_seed=42):
        self._rng = np.random.default_rng(rng_seed)

    def next_batch(self, device):
        n = int(np.clip(
            self._rng.lognormal(mean=DIST_MU, sigma=DIST_SIGMA),
            DIST_N_MIN, DIST_N_MAX))
        return _build_voxel_batch(n, device), n

    def describe(self):
        return ("sampled  LogNormal(mu=%.1f,sigma=%.1f)  "
                "clip[%d,%d]" % (DIST_MU, DIST_SIGMA, DIST_N_MIN, DIST_N_MAX))


class NpyDirInputSource:
    """Option ①: load real lidar npy frames, cycle when exhausted.

    Expects `npy_dir` to contain files with names like 0000.npy, 0001.npy ...
    Each file is a pre-voxelized dict with keys matching the batch format,
    OR a raw (N,4) point cloud that gets voxelized on load.

    NOTE: actual voxelization from raw points is NOT implemented here (requires
    the SpVoxelPreprocessor from pnp_infer_action_e2e.py). When raw .npy files
    contain point clouds, pass --input_source sampled and build a proper voxel
    capture pipeline from process C instead. This loader expects pre-built
    batch dicts saved by a helper script (pending §3.8 R-1 GO decision).
    """

    def __init__(self, npy_dir):
        self._dir = npy_dir
        fnames = sorted(f for f in os.listdir(npy_dir) if f.endswith('.npy'))
        if not fnames:
            raise FileNotFoundError("No .npy files in: %s" % npy_dir)
        self._files = [os.path.join(npy_dir, f) for f in fnames]
        self._pos = 0
        self._cycle_count = 0
        print("[trace/npy_dir] Found %d .npy frames in %s" % (len(self._files), npy_dir))

    def next_batch(self, device):
        if self._pos >= len(self._files):
            self._cycle_count += 1
            self._pos = 0
            print("[trace/npy_dir] Cycling frame list (cycle #%d)" % self._cycle_count)
        data = np.load(self._files[self._pos], allow_pickle=True).item()
        self._pos += 1
        batch = {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray)
                    else torch.tensor(v, device=device)
                 for k, v in data.items()}
        n_voxels = batch['voxel_features'].shape[0] // RECORD_LEN
        return batch, n_voxels

    def describe(self):
        return "npy_dir  %s  (%d frames)" % (self._dir, len(self._files))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_perception_model(perception_dir, device='cuda'):
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils

    hypes_path = os.path.join(perception_dir, 'config.yaml')
    if not os.path.exists(hypes_path):
        raise FileNotFoundError("Perception config not found: %s" % hypes_path)

    hypes = yaml_utils.load_yaml(hypes_path, opt=None)
    model = train_utils.create_model(hypes).to(device)

    ckpt = None
    for fname in sorted(os.listdir(perception_dir)):
        if fname.endswith('.pth'):
            ckpt = os.path.join(perception_dir, fname)
    if ckpt is None:
        raise FileNotFoundError("No .pth checkpoint in: %s" % perception_dir)

    print("[trace] Loading checkpoint: %s" % ckpt)
    saved = torch.load(ckpt, map_location=device)
    state = saved.get('model_state_dict', saved)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[trace] missing=%d  unexpected=%d" % (len(missing), len(unexpected)))
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Timed forward passes
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_timed(model, source, n_calls, warmup, device):
    """Run warmup + n_calls forward passes with CUDA-Event timing.

    ★ Warmup calls are NOT included in the returned list (§3.8 team-lead §4).
    Returns list of (call_idx, tau_ms, n_voxels).
    口径: CUDA-Event between model forward entry and device completion only.
    """
    results = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    total = warmup + n_calls
    call_idx = 0

    with torch.no_grad():
        for i in range(total):
            batch, n_vox = source.next_batch(device)

            torch.cuda.synchronize(device)
            start_evt.record()

            _ = model(batch)

            end_evt.record()
            torch.cuda.synchronize(device)
            tau = start_evt.elapsed_time(end_evt)  # ms, CUDA-Event only

            if i < warmup:
                print("[trace] warmup %d/%d  τ=%.2f ms  N_vox=%d" %
                      (i + 1, warmup, tau, n_vox))
            else:
                results.append((call_idx, tau, n_vox))
                call_idx += 1
                if call_idx % 50 == 0 or call_idx == n_calls:
                    print("[trace] %d/%d  τ=%.2f ms  N_vox=%d" %
                          (call_idx, n_calls, tau, n_vox))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Collect RSU latency trace (§3.8 R-1)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_source', choices=['sampled', 'npy_dir'],
                        default='sampled',
                        help='Input source: sampled=variable-N (②) | '
                             'npy_dir=real lidar frames (①)')
    parser.add_argument('--npy_dir', type=str, default=None,
                        help='[npy_dir mode] directory with pre-voxelized .npy frames')
    parser.add_argument('--n_calls', type=int, default=500,
                        help='Timed forward calls to record (default: 500)')
    parser.add_argument('--warmup', type=int, default=20,
                        help='Warmup calls EXCLUDED from output (default: 20)')
    parser.add_argument('--out', type=str, default=DEFAULT_OUT,
                        help='Output CSV path')
    parser.add_argument('--perception_dir', type=str,
                        default=DEFAULT_PERCEPTION_DIR)
    parser.add_argument('--rng_seed', type=int, default=42,
                        help='[sampled mode] RNG seed for reproducibility')
    args = parser.parse_args()

    # ── GPU check ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Set CUDA_VISIBLE_DEVICES to an idle GPU.")
        sys.exit(1)
    device = 'cuda:0'
    gname  = torch.cuda.get_device_name(0)
    gmem   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("[trace] GPU: %s  total=%.1f GB" % (gname, gmem))
    print("[trace] ⚠  Confirm this GPU ≠ CARLA render GPU (算力隔离 §3.1)")

    # ── build input source ──────────────────────────────────────────────────
    if args.input_source == 'sampled':
        source = SampledInputSource(rng_seed=args.rng_seed)
    elif args.input_source == 'npy_dir':
        if args.npy_dir is None:
            print("ERROR: --npy_dir required for input_source=npy_dir")
            sys.exit(1)
        source = NpyDirInputSource(args.npy_dir)
    print("[trace] input_source: %s" % source.describe())

    # ── load model ──────────────────────────────────────────────────────────
    print("[trace] Loading CoDriving perception model ...")
    model = load_perception_model(args.perception_dir, device)

    # ── timed run ───────────────────────────────────────────────────────────
    print("[trace] warmup=%d (excluded from CSV)  timed=%d" %
          (args.warmup, args.n_calls))
    t0 = time.time()
    results = run_forward_timed(model, source, args.n_calls, args.warmup, device)
    wall = time.time() - t0

    tau_arr = np.array([r[1] for r in results])
    nv_arr  = np.array([r[2] for r in results])
    print("\n[trace] === Summary ===")
    print("  n_calls      : %d" % len(results))
    print("  wall_time    : %.1f s" % wall)
    print("  τ_mean/med   : %.2f / %.2f ms" % (tau_arr.mean(), np.median(tau_arr)))
    print("  τ_p25/p75/p95: %.2f / %.2f / %.2f ms" % (
        np.percentile(tau_arr, 25), np.percentile(tau_arr, 75),
        np.percentile(tau_arr, 95)))
    print("  τ_min/max    : %.2f / %.2f ms" % (tau_arr.min(), tau_arr.max()))
    print("  N_vox mean   : %.0f  std=%.0f" % (nv_arr.mean(), nv_arr.std()))
    distinct_deltas = sorted(set(
        min(10, int(__import__('math').ceil(t / 50.0))) for t in tau_arr))
    print("  Δ values     : %s  (=ceil(τ/50), cap 10)" % distinct_deltas)
    print()

    # ── write CSV (warmup EXCLUDED) ─────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['call_idx', 'tau_ms'])
        for call_idx, tau, _ in results:
            writer.writerow([call_idx, '%.4f' % tau])
    print("[trace] Written: %s  (%d rows, warmup excluded)" % (args.out, len(results)))
    print("[trace] 口径: CUDA-Event forward-only, "
          "latency_ms_source=trace_replay_codriving_4090")
    print("[trace] Distinct Δ values in trace: %s" % distinct_deltas)
    print("[trace] ★ Confirm Δ list has ≥2 distinct values before using as trace input")


if __name__ == '__main__':
    main()
