"""
Orin 全链路 e2e 计时脚本骨架 (Task #12 D1+D2)
- D1: PyTorch FP32 全链分模块测 (voxelize→enc→bb→body→NMS), B=1+B=2
- D2: 混合链 PyTorch前处理 + TRT body(base_fp32/p75_int8) + torch NMS
协议: CUDA Event+sync, warmup≥50, p50/mean/p99, 30W MODE_30W
口径: e2e_pytorch_fp32_orin (D1) / e2e_hybrid_trt_orin (D2)
接口: sw-agent 交付 pillar_voxelizer + opencood m1 model modules 后填入
"""
import os, sys, json, time
import numpy as np
import torch

# ============================================================
# 路径配置 (Orin 端)
# ============================================================
DAIR_ROOT = os.path.expanduser("~/m4_8_orin/dair_val/cooperative-vehicle-infrastructure")
CKPT_PATH = os.path.expanduser("~/m4_8_orin/dair_ckpt/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth")
ENGINE_DIR = os.path.expanduser("~/m4_8_orin/p03_build")
CODE_DIR   = os.path.expanduser("~/m4_8_orin/opencood_m1")  # sw-agent 交付目录
VAL_JSON   = os.path.join(DAIR_ROOT, "val.json")
DATA_INFO  = os.path.join(DAIR_ROOT, "cooperative/data_info.json")

# ============================================================
# TV egg for NMS (已验证路径)
# ============================================================
TV_EGG = "/home/jichengzhi/.local/lib/python3.8/site-packages/torchvision-0.13.0a0+da3794e-py3.8-linux-aarch64.egg"
if TV_EGG not in sys.path:
    sys.path.insert(0, TV_EGG)

# ============================================================
# 占位接口 (sw-agent 交付后填入真实实现)
# ============================================================
def load_voxelizer(config):
    """sw-agent 实现: pillar voxelizer (numba/torch), 与 spconv 输出 diff ≤1e-5 已验证"""
    raise NotImplementedError("等待 sw-agent 交付 pillar_voxelizer")

def load_model_modules(config, ckpt_path, device):
    """sw-agent 实现: 加载 encoder_m1, backbone_m1, pyramid_body (from opencood_m1/)"""
    raise NotImplementedError("等待 sw-agent 交付 opencood_m1 模块")

def load_pcd_points(pcd_path):
    """读取 .pcd 点云 -> (N,4) float32 [x,y,z,intensity]"""
    # 简单 binary pcd 读取或调用 open3d
    # 先尝试 numpy binary, 失败则用 pcd_utils
    import struct
    with open(pcd_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('DATA'):
                data_type = line.split()[1]
                break
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1:]
        width = int(header.get('WIDTH', [0])[0])
        if data_type == 'binary':
            n_points = width * int(header.get('HEIGHT', [1])[0])
            fields = header.get('FIELDS', [])
            sizes = [int(s) for s in header.get('SIZE', [])]
            raw = f.read()
            row_size = sum(sizes)
            pts = []
            for i in range(min(n_points, len(raw)//row_size)):
                row = raw[i*row_size:(i+1)*row_size]
                x,y,z,i_ = struct.unpack_from('ffff', row)
                pts.append([x,y,z,i_])
            return np.array(pts, dtype=np.float32)
        else:  # ascii
            pts = []
            for line in f:
                vals = line.decode().strip().split()
                if len(vals) >= 4:
                    pts.append([float(v) for v in vals[:4]])
            return np.array(pts, dtype=np.float32)

# ============================================================
# CUDA Event 计时工具
# ============================================================
class CUDATimer:
    def __init__(self):
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.end_ev   = torch.cuda.Event(enable_timing=True)
    def start(self): self.start_ev.record()
    def stop(self):
        self.end_ev.record()
        torch.cuda.synchronize()
        return self.start_ev.elapsed_time(self.end_ev)  # ms

def percentile(arr, p):
    return float(np.percentile(arr, p))

# ============================================================
# D1: PyTorch FP32 全链计时 (B=1 和 B=2)
# ============================================================
def run_d1(val_ids, data_info, voxelizer, encoder, backbone, body, device,
           n_warmup=50, n_measure=100, batch_size=2):
    """分模块计时: voxelize / encoder / backbone / body / NMS"""
    timer = CUDATimer()
    timings = {k: [] for k in ['voxelize','encoder','backbone','body','nms','e2e']}
    
    for i, vid in enumerate(val_ids[:n_warmup + n_measure]):
        entry = data_info[vid]
        # 读点云
        veh_pcd = load_pcd_points(os.path.join(DAIR_ROOT, entry['vehicle_pointcloud_path']))
        inf_pcd = load_pcd_points(os.path.join(DAIR_ROOT, entry['infrastructure_pointcloud_path']))
        pcd_list = [veh_pcd, inf_pcd] if batch_size == 2 else [veh_pcd]
        
        warmup = (i < n_warmup)
        
        # E2E start
        timer.start(); t_vox_start = time.perf_counter()
        
        # === voxelize ===
        timer.start()
        voxel_features_list = [voxelizer(pcd) for pcd in pcd_list]
        t_vox = timer.stop()
        
        # === encoder ===
        timer.start()
        enc_list = [encoder(vf.to(device)) for vf in voxel_features_list]
        t_enc = timer.stop()
        
        # === backbone ===
        timer.start()
        bb_list = [backbone(enc) for enc in enc_list]
        t_bb = timer.stop()
        
        # === body (fusion+shrink+heads) ===
        timer.start()
        preds = body(bb_list)  # collab2 fusion
        t_body = timer.stop()
        
        # === NMS ===
        timer.start()
        try:
            from torchvision.ops import nms as tv_nms
            nms_fn = tv_nms
        except:
            nms_fn = torch.ops.torchvision.nms
        boxes = preds['boxes']
        scores = preds['scores']
        keep = nms_fn(boxes, scores, iou_threshold=0.5)
        t_nms = timer.stop()
        
        if not warmup:
            timings['voxelize'].append(t_vox)
            timings['encoder'].append(t_enc)
            timings['backbone'].append(t_bb)
            timings['body'].append(t_body)
            timings['nms'].append(t_nms)
            timings['e2e'].append(t_vox + t_enc + t_bb + t_body + t_nms)
    
    return {k: {'p50': percentile(v,50), 'mean': float(np.mean(v)), 'p99': percentile(v,99)}
            for k,v in timings.items()}

# ============================================================
# D2: 混合链 (PyTorch 前处理 + TRT body + torch NMS)
# ============================================================
def run_d2_hybrid(val_ids, data_info, voxelizer, encoder, backbone, trt_engine_path, device,
                  n_warmup=50, n_measure=100, batch_size=2):
    """hybrid: PyTorch vox+enc+bb → TRT body → torch NMS"""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    timer = CUDATimer()
    timings = {k: [] for k in ['preproc','trt_body','nms','e2e']}
    
    for i, vid in enumerate(val_ids[:n_warmup + n_measure]):
        entry = data_info[vid]
        veh_pcd = load_pcd_points(os.path.join(DAIR_ROOT, entry['vehicle_pointcloud_path']))
        inf_pcd = load_pcd_points(os.path.join(DAIR_ROOT, entry['infrastructure_pointcloud_path']))
        pcd_list = [veh_pcd, inf_pcd] if batch_size == 2 else [veh_pcd]
        warmup = (i < n_warmup)
        
        # === preproc (voxelize+enc+bb in PyTorch) ===
        timer.start()
        vf_list = [voxelizer(pcd) for pcd in pcd_list]
        enc_list = [encoder(vf.to(device)) for vf in vf_list]
        bb_out = [backbone(enc) for enc in enc_list]
        # stack for TRT input: (2,64,128,256) per spec
        trt_input = torch.stack([b for b in bb_out], dim=0).contiguous()
        t_preproc = timer.stop()
        
        # === TRT body ===
        timer.start()
        # TRT inference via Python bindings
        # NOTE: exact binding setup filled after seeing sw's output tensor shape
        trt_output = _trt_infer(context, trt_input)
        t_trt = timer.stop()
        
        # === NMS ===
        timer.start()
        try:
            from torchvision.ops import nms as tv_nms
            nms_fn = tv_nms
        except:
            nms_fn = torch.ops.torchvision.nms
        boxes = trt_output['boxes']
        scores = trt_output['scores']
        keep = nms_fn(boxes, scores, iou_threshold=0.5)
        t_nms = timer.stop()
        
        if not warmup:
            timings['preproc'].append(t_preproc)
            timings['trt_body'].append(t_trt)
            timings['nms'].append(t_nms)
            timings['e2e'].append(t_preproc + t_trt + t_nms)
    
    return {k: {'p50': percentile(v,50), 'mean': float(np.mean(v)), 'p99': percentile(v,99)}
            for k,v in timings.items()}

def _trt_infer(context, input_tensor):
    """TRT Python binding 推理 (详细 binding 等 sw 交付后按 ONNX I/O 规格填入)"""
    raise NotImplementedError("待 sw 确认 ONNX input/output 规格后填入 binding")

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=== Orin 全链路 e2e 计时 (Task #12 D1+D2) ===")
    print("等待 sw-agent 交付 opencood_m1 + pillar_voxelizer, 届时填入 load_voxelizer/load_model_modules")
    print("资产检查:")
    print(f"  DAIR_ROOT 存在: {os.path.exists(DAIR_ROOT)}")
    print(f"  CKPT 存在:      {os.path.exists(CKPT_PATH)}")
    print(f"  VAL_JSON 存在:  {os.path.exists(VAL_JSON)}")
    eng_fp32 = os.path.join(ENGINE_DIR, 'base_fp32_orin.engine')
    eng_int8 = os.path.join(ENGINE_DIR, 'p75_int8_orin.engine')
    print(f"  TRT base_fp32:  {os.path.exists(eng_fp32)}")
    print(f"  TRT p75_int8:   {os.path.exists(eng_int8)}")
