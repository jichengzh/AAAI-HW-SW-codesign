#!/usr/bin/env python3
"""Accuracy metrics for Orin TRT engines (FP32 reference vs FP16 / INT8).
NaN-aware: frames where either ref or test is non-finite are dropped and counted."""
import os, sys, glob
import numpy as np

SPEC = {
    "outs_planning":      (np.float32, (6, 2)),
    "scores":             (np.float32, (300,)),
    "bboxes_dict_bboxes": (np.float32, (300, 9)),
    "labels":             (np.int32,   (300,)),
    "seg_out":            (np.int32,   (5, 50, 50)),
    "bev_embed":          (np.float32, (2500, 256)),
    "max_obj_id_out":     (np.int32,   (1,)),
}
FLOAT_T = {"outs_planning","scores","bboxes_dict_bboxes","bev_embed"}

def load(d, fid, name):
    dt, shape = SPEC[name]
    fp = os.path.join(d, "dump", "frame%d_%s.bin" % (fid, name))
    if not os.path.exists(fp): return None
    return np.fromfile(fp, dtype=dt).reshape(shape)

def frames(d):
    fs = glob.glob(os.path.join(d, "dump", "frame*_outs_planning.bin"))
    return sorted(int(os.path.basename(f).split("_")[0][5:]) for f in fs)

def finite_pair(r, t):
    return np.isfinite(r.astype(np.float64)).all() and np.isfinite(t.astype(np.float64)).all()

def tmet(ref, tst):
    r = ref.astype(np.float64); t = tst.astype(np.float64); d = t - r
    rn = float(np.linalg.norm(r)); tn = float(np.linalg.norm(t))
    mse = float(np.mean(d*d)); mae = float(np.mean(np.abs(d))); mx = float(np.max(np.abs(d)))
    rel = float(np.linalg.norm(d)/rn) if rn > 1e-12 else float("nan")
    cos = float(np.dot(r,t)/(rn*tn)) if (rn>1e-12 and tn>1e-12) else float("nan")
    return dict(mse=mse, rmse=mse**0.5, mae=mae, maxabs=mx, rel_l2=rel, cos=cos)

def agg(name, fids, rd, td):
    accs=[]; dropped=[]
    for fid in fids:
        r=load(rd,fid,name); t=load(td,fid,name)
        if r is None or t is None: continue
        if name in FLOAT_T and not finite_pair(r,t):
            dropped.append(fid); continue
        accs.append(tmet(r.ravel(), t.ravel()))
    if not accs: return None
    out={}
    for k in accs[0]:
        v=np.array([a[k] for a in accs],dtype=np.float64)
        out[k]=float(np.mean(v)); out[k+"_max"]=float(np.max(v))
    out["n_used"]=len(accs); out["dropped"]=dropped
    return out

def planning(fids, rd, td):
    per=np.zeros(6); n=0; l2=[]; dropped=[]
    for fid in fids:
        r=load(rd,fid,"outs_planning"); t=load(td,fid,"outs_planning")
        if r is None or t is None: continue
        if not finite_pair(r,t): dropped.append(fid); continue
        dl=np.linalg.norm(t.astype(np.float64)-r.astype(np.float64),axis=1)
        per+=dl; l2.append(dl.mean()); n+=1
    if n==0: return None
    return dict(per=(per/n).tolist(), mean=float(np.mean(l2)), mx=float(np.max(l2)), frames=n, dropped=dropped)

def detection(fids, rd, td, thr=0.3, K=10):
    smae=[]; smse=[]; bl1=[]; cr_l=[]; ct_l=[]; ag=[]; tk=[]; dropped=[]
    for fid in fids:
        rs=load(rd,fid,"scores"); ts=load(td,fid,"scores")
        rb=load(rd,fid,"bboxes_dict_bboxes"); tb=load(td,fid,"bboxes_dict_bboxes")
        if rs is None or ts is None: continue
        if not finite_pair(rs,ts): dropped.append(fid); continue
        r=rs.astype(np.float64); t=ts.astype(np.float64)
        smae.append(np.mean(np.abs(t-r))); smse.append(np.mean((t-r)**2))
        m=r>thr
        if m.sum()>0 and rb is not None and tb is not None and finite_pair(rb[m],tb[m]):
            bl1.append(np.mean(np.abs(tb[m].astype(np.float64)-rb[m].astype(np.float64))))
        cr=int((r>thr).sum()); ct=int((t>thr).sum())
        cr_l.append(cr); ct_l.append(ct); ag.append(int(cr==ct))
        rr=np.argsort(-r)[:K]; rt=np.argsort(-t)[:K]
        tk.append(len(set(rr.tolist())&set(rt.tolist()))/float(K))
    if not smae: return None
    return dict(sMAE=float(np.mean(smae)), sMSE=float(np.mean(smse)),
                bL1=float(np.mean(bl1)) if bl1 else float("nan"), thr=thr,
                cr=float(np.mean(cr_l)), ct=float(np.mean(ct_l)),
                agree=float(np.mean(ag)), tk=float(np.mean(tk)), K=K, dropped=dropped)

def seg(fids, rd, td):
    dis=0; tot=0; ex=[]; dropped=[]
    for fid in fids:
        r=load(rd,fid,"seg_out"); t=load(td,fid,"seg_out")
        if r is None or t is None: continue
        if not (np.isfinite(r.astype(np.float64)).all() and np.isfinite(t.astype(np.float64)).all()):
            dropped.append(fid); continue
        rc=r.argmax(0); tc=t.argmax(0)
        dis+=int((rc!=tc).sum()); tot+=rc.size; ex.append(float((r==t).all()))
    if tot==0: return None
    return dict(dis=dis/float(tot), ex=float(np.mean(ex)), dropped=dropped)

def nan_scan(d, fids):
    """Report frames with any non-finite value in the float task outputs."""
    bad={}
    for name in ["outs_planning","scores","bboxes_dict_bboxes","bev_embed"]:
        bl=[]
        for fid in fids:
            a=load(d,fid,name)
            if a is None: continue
            if not np.isfinite(a.astype(np.float64)).all(): bl.append(fid)
        if bl: bad[name]=bl
    return bad

def fmt(x):
    if x is None: return "n/a"
    if isinstance(x,float):
        if np.isnan(x): return "nan"
        if x==0: return "0.00e+00"
        return "%.3e"%x
    return str(x)

def report(rd, td, label, fids):
    print(""); print("="*80)
    print("  %s   (ref=FP32, %d frames, free-run)"%(label,len(fids)))
    print("="*80)
    bad=nan_scan(td, fids)
    if bad:
        print("  [!] NON-FINITE (NaN/Inf) output frames in THIS engine:")
        for k,v in bad.items(): print("      %-20s frames=%s"%(k,v))
    else:
        print("  [ok] all %d frames finite in this engine"%len(fids))
    print("")
    print("[1] Tensor-level deviation vs FP32 (mean over FINITE frames)")
    print("  %-20s %-10s %-10s %-10s %-10s %-9s %-8s %s"%("tensor","MSE","RMSE","MAE","maxabs","relL2","cos","n_used"))
    print("  "+"-"*88)
    for name in ["outs_planning","scores","bboxes_dict_bboxes","bev_embed"]:
        m=agg(name,fids,rd,td)
        if m is None:
            print("  %-20s  (all frames dropped / missing)"%name); continue
        print("  %-20s %-10s %-10s %-10s %-10s %-9s %.6f %d/%d"%(name,fmt(m["mse"]),fmt(m["rmse"]),fmt(m["mae"]),fmt(m["maxabs"]),fmt(m["rel_l2"]),m["cos"],m["n_used"],len(fids)))
        print("  %-20s   worst-frame: MSE=%s maxabs=%s relL2=%s  dropped=%s"%("",fmt(m["mse_max"]),fmt(m["maxabs_max"]),fmt(m["rel_l2_max"]),m["dropped"] if m["dropped"] else "-"))
    print("")
    print("[2] Task-relative metrics vs FP32 (finite frames only)")
    p=planning(fids,rd,td)
    if p:
        print("  planning  mean_traj_L2=%s  max_traj_L2=%s  (frames=%d, dropped=%s)"%(fmt(p["mean"]),fmt(p["mx"]),p["frames"],p["dropped"] if p["dropped"] else "-"))
        print("            per-waypoint L2 (1..6): "+" ".join(fmt(v) for v in p["per"]))
    d=detection(fids,rd,td)
    if d:
        print("  detection score_MAE=%s score_MSE=%s"%(fmt(d["sMAE"]),fmt(d["sMSE"])))
        print("            bbox_L1(conf>%.1f)=%s"%(d["thr"],fmt(d["bL1"])))
        print("            mean#det ref=%.2f test=%.2f  exact-count-agree=%.3f  top%d-overlap=%.3f  dropped=%s"%(d["cr"],d["ct"],d["agree"],d["K"],d["tk"],d["dropped"] if d["dropped"] else "-"))
    s=seg(fids,rd,td)
    if s:
        print("  seg       class_disagree_rate=%s  exact_frame_rate=%.3f  dropped=%s"%(fmt(s["dis"]),s["ex"],s["dropped"] if s["dropped"] else "-"))

def main():
    base=sys.argv[1] if len(sys.argv)>1 else "."
    E="DL4AGX_min/uniad-trt/inference_app/enqueueV2"
    ref=os.path.join(base,E,"acc_fp32"); fp16=os.path.join(base,E,"acc_fp16"); int8=os.path.join(base,E,"acc_int8")
    fids=frames(ref)
    if not fids: print("no FP32 dump frames under %s"%ref); return
    print("frames found (FP32 ref): %d  ids=%d..%d"%(len(fids),fids[0],fids[-1]))
    refbad=nan_scan(ref,fids)
    print("FP32 reference non-finite frames: %s"%(refbad if refbad else "none (clean)"))
    if os.path.isdir(os.path.join(fp16,"dump")): report(ref,fp16,"FP16 vs FP32",fids)
    if os.path.isdir(os.path.join(int8,"dump")): report(ref,int8,"INT8(CNN int8 / attention fp32) vs FP32",fids)
    print(""); print("="*80); print("  DONE")

if __name__=="__main__":
    main()
