#!/usr/bin/env python3
"""Robust detection-deviation metric: greedy match confident boxes by center(x,y),
robust to query-index reordering in free-run. bbox=[x,y,z,w,l,h,yaw,vx,vy]. thr=0.25 (app default)."""
import os, sys, glob
import numpy as np
THR=0.25; MATCH_GATE=2.0  # meters, center distance to count as a match

def load(d,fid,name,dt,shape):
    fp=os.path.join(d,"dump","frame%d_%s.bin"%(fid,name))
    if not os.path.exists(fp): return None
    return np.fromfile(fp,dtype=dt).reshape(shape)

def frames(d):
    fs=glob.glob(os.path.join(d,"dump","frame*_scores.bin"))
    return sorted(int(os.path.basename(f).split("_")[0][5:]) for f in fs)

def conf_boxes(sc,bb,lb):
    m=sc>=THR
    if not np.isfinite(sc[m]).all() or not np.isfinite(bb[m]).all(): return None
    return bb[m].astype(np.float64), sc[m].astype(np.float64), lb[m]

def match(rb,rs,rl,tb,ts,tl):
    """greedy nearest-center matching ref->test."""
    used=np.zeros(len(tb),dtype=bool)
    cd=[]; sd=[]; szd=[]; yawd=[]; lmatch=0; matched=0
    order=np.argsort(-rs)
    for i in order:
        rc=rb[i,:2]
        best=-1; bestd=1e9
        for j in range(len(tb)):
            if used[j]: continue
            dd=np.linalg.norm(tb[j,:2]-rc)
            if dd<bestd: bestd=dd; best=j
        if best>=0 and bestd<=MATCH_GATE:
            used[best]=True; matched+=1
            cd.append(bestd); sd.append(abs(ts[best]-rs[i]))
            szd.append(np.mean(np.abs(tb[best,3:6]-rb[i,3:6])))
            dy=abs(tb[best,6]-rb[i,6]); dy=min(dy,2*np.pi-dy); yawd.append(dy)
            lmatch+=int(tl[best]==rl[i])
        else:
            cd.append(np.nan)  # missed
    return matched,len(rb),cd,sd,szd,yawd,lmatch

def run(rd,td,label,fids):
    tot_ref=0; tot_match=0; cd=[]; sd=[]; szd=[]; yawd=[]; lab_ok=0
    nan_frames=[]
    for fid in fids:
        rs=load(rd,fid,"scores",np.float32,(300,)); ts=load(td,fid,"scores",np.float32,(300,))
        rb=load(rd,fid,"bboxes_dict_bboxes",np.float32,(300,9)); tb=load(td,fid,"bboxes_dict_bboxes",np.float32,(300,9))
        rl=load(rd,fid,"labels",np.int32,(300,)); tl=load(td,fid,"labels",np.int32,(300,))
        if rs is None or ts is None: continue
        R=conf_boxes(rs,rb,rl)
        if R is None: continue
        T=conf_boxes(ts,tb,tl)
        if T is None:
            nan_frames.append(fid)
            # treat all ref boxes as missed
            tot_ref+=int((rs>=THR).sum()); continue
        rbb,rss,rll=R; tbb,tss,tll=T
        m,nr,c_,s_,z_,y_,lo=match(rbb,rss,rll,tbb,tss,tll)
        tot_ref+=nr; tot_match+=m
        cd+= [x for x in c_ if not np.isnan(x)]
        miss = sum(1 for x in c_ if np.isnan(x))
        sd+=s_; szd+=z_; yawd+=y_; lab_ok+=lo
        if miss: pass
    print("\n==== %s : matched detection deviation (thr=%.2f, gate=%.1fm) ===="%(label,THR,MATCH_GATE))
    if nan_frames: print("  [!] test-engine NaN frames (all ref boxes counted missed): %s"%nan_frames)
    recall = tot_match/float(tot_ref) if tot_ref else float("nan")
    print("  ref confident boxes total=%d  matched=%d  recall(match<=%.1fm)=%.4f"%(tot_ref,tot_match,MATCH_GATE,recall))
    if cd:
        cd=np.array(cd)
        print("  matched center-dist(m): mean=%.4f median=%.4f p90=%.4f max=%.4f"%(cd.mean(),np.median(cd),np.percentile(cd,90),cd.max()))
        print("  matched |dscore|      : mean=%.4e max=%.4e"%(np.mean(sd),np.max(sd)))
        print("  matched size L1(w,l,h): mean=%.4e"%np.mean(szd))
        print("  matched |dyaw|(rad)   : mean=%.4e"%np.mean(yawd))
        print("  matched label agree   : %.4f (%d/%d)"%(lab_ok/float(tot_match),lab_ok,tot_match))

def main():
    base=sys.argv[1] if len(sys.argv)>1 else "."
    E="DL4AGX_min/uniad-trt/inference_app/enqueueV2"
    ref=os.path.join(base,E,"acc_fp32")
    fids=frames(ref)
    for name,lab in [("acc_fp16","FP16 vs FP32"),("acc_int8","INT8(cnn) vs FP32")]:
        td=os.path.join(base,E,name)
        if os.path.isdir(os.path.join(td,"dump")): run(ref,td,lab,fids)

if __name__=="__main__": main()
