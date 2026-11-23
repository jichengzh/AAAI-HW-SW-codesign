import sys, json, time, statistics, torch
TV_EGG = "/home/jichengzhi/.local/lib/python3.8/site-packages/torchvision-0.13.0a0+da3794e-py3.8-linux-aarch64.egg"
if TV_EGG not in sys.path:
    sys.path.insert(0, TV_EGG)
try:
    from torchvision.ops import nms as tv_nms
    nms_fn = tv_nms
    impl = "torchvision.ops.nms"
except:
    nms_fn = torch.ops.torchvision.nms
    impl = "torch.ops.torchvision.nms"

N, IOU = 500, 0.5
WU, M = 300, 500
device = torch.device("cuda")
torch.manual_seed(42)
b = torch.rand(N,4,device=device)
b[:,0]=b[:,0]*100-50; b[:,1]=b[:,1]*100-50; b[:,2:]=b[:,2:]*4+0.5
bx = torch.stack([b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2],1).contiguous()
s = torch.rand(N,device=device)
for _ in range(WU):
    k=nms_fn(bx,s,IOU); torch.cuda.synchronize()
lats=[]
for _ in range(M):
    t0=time.perf_counter()
    k=nms_fn(bx,s,IOU); torch.cuda.synchronize()
    lats.append((time.perf_counter()-t0)*1000)
ls=sorted(lats)
print(json.dumps({"n_in":N,"n_out":len(k),"lat_mean_ms":round(statistics.mean(lats),4),
    "lat_p50_ms":round(ls[int(0.50*M)],4),"lat_p99_ms":round(ls[int(0.99*M)],4),
    "nms_impl":impl,"source":"orin_nms_n500;MODE_30W",
    "caveat":"N=500 realistic post-threshold; 2D BEV proxy for HEAL 3D NMS"},indent=2))
