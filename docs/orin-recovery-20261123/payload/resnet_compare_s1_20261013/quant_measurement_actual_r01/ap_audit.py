"""Independent host audit: actual frozen source files and frame TP/FP VOC AP."""
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from qualification import require,sha,bound_file


def check_actual_files(files):
    require(bool(files),'Missing source files')
    for name,expected in files.items():
        path=Path(name)
        require(path.is_absolute() and path.is_file() and sha(path)==expected,'Actual source file differs: '+name)
    return files


def check_ap(stats,reported,order=None,sort_spec=None):
    require(type(stats['gt']) is int and stats['gt']>0,'AP requires positive GT')
    tp=np.asarray(stats['tp']);fp=np.asarray(stats['fp']);score=np.asarray(stats['score'])
    require(tp.ndim==fp.ndim==score.ndim==1 and len(tp)==len(fp)==len(score),'AP vector shape invalid')
    require(np.isfinite(score).all() and np.isin(tp,[0,1]).all() and np.isin(fp,[0,1]).all() and (tp+fp==1).all(),'AP TP/FP invalid')
    if order is None:order=np.argsort(-score)
    else:
        require(order.dtype==np.int64 and order.shape==score.shape,'Frozen sort dtype/shape differs')
        require(hashlib.sha256(order.tobytes()).hexdigest()==sort_spec['order_sha256'] and hashlib.sha256(score.tobytes()).hexdigest()==sort_spec['score_sha256'],'Frozen sort order/score SHA differs')
        require(np.array_equal(np.sort(order),np.arange(len(score))) and np.all(np.diff(score[order])<=0),'Frozen sort is not a descending-score permutation')
    t=np.cumsum(tp[order]);f=np.cumsum(fp[order])
    recall=np.concatenate(([0.],t/stats['gt'],[1.]));precision=np.concatenate(([0.],np.divide(t,t+f),[0.]))
    require((np.diff(recall)>=0).all(),'AP true positives exceed GT')
    for i in range(len(precision)-2,-1,-1):precision[i]=max(precision[i],precision[i+1])
    value=sum((recall[i]-recall[i-1])*precision[i] for i in range(1,len(recall)) if recall[i]!=recall[i-1])
    require(np.isfinite(reported) and abs(value-reported)<=1e-12,'Reported AP differs from independent TP/FP integration')
    return float(value)


def recompute(rows):
    result={k:dict(tp=[],fp=[],score=[],gt=0) for k in ('0.3','0.5','0.7')}
    for row in rows:
        require(set(row['statistics'])==set(result),'Missing per-frame TP/FP')
        require('predictions' in row and 'ground_truth' in row,'Missing prediction/GT evidence')
        for k,s in row['statistics'].items():
            require(type(s['gt']) is int and s['gt']>=0 and len(s['tp'])==len(s['fp'])==len(s['score']),'Frame TP/FP shape differs')
            for name in ('tp','fp','score'):result[k][name].extend(s[name])
            result[k]['gt']+=s['gt']
    return result


def audit(root,expected_manifest_sha256):
    root=Path(root);e=json.loads((root/'ap_evidence.json').read_text())
    require(e['manifest_sha256']==expected_manifest_sha256,'AP source manifest differs from externally frozen identity')
    manifest_path=bound_file(root,e['manifest_file'],expected_manifest_sha256)
    manifest=json.loads(manifest_path.read_text())
    require(e['source_identity']==manifest['source_identity'] and e['source_files']==manifest['source_files'],'AP source identity differs from frozen manifest')
    source=check_actual_files(e['source_files'])
    require(e['source_identity']['checkpoint_sha256'] in source.values() and e['source_identity']['calibration_sha256'] in source.values(),'Checkpoint/calibration not backed by actual source files')
    eval_path=bound_file(root,e['eval_code_file'],e['eval_code_sha256'])
    require(e['eval_code_sha256']==manifest['eval_code_sha256'],'Evaluation code differs from frozen manifest')
    eval_files=check_actual_files(json.loads(eval_path.read_text()))
    require(e['split_sha256']==manifest['split_sha256'] and e['split_sha256'] in source.values(),'Split not backed by frozen source')
    sorts,sort_identity=freeze_and_verify_sorts(root)
    methods={}
    for method,spec in e['methods'].items():
        frame=bound_file(root,spec['frames_file'],spec['frames_sha256'])
        rows=[json.loads(s) for s in frame.read_text().splitlines()];stats=recompute(rows)
        result_path=root/(method+'_result.json');result=json.loads(result_path.read_text())
        require(result['count']==1789 and result['frames_sha256']==spec['frames_sha256'] and result['result_stats']==stats and result['ap']==spec['ap'],'Result/TPFP/frame evidence disagree')
        values={name:check_ap(stats[k],spec['ap'][name],sorts[method+'_'+k],sort_identity['identity']['points'][method+'_'+k]) for name,k in zip(('ap30','ap50','ap70'),('0.3','0.5','0.7'))}
        methods[method]=dict(result_file=result_path.name,result_sha256=sha(result_path),frames_sha256=spec['frames_sha256'],recomputed_ap=values)
    receipt=dict(status='passed',ap_evidence_sha256=sha(root/'ap_evidence.json'),frozen_manifest_sha256=expected_manifest_sha256,
        sort_identity_file='sort_identity.json',sort_identity_sha256=sha(root/'sort_identity.json'),sort_indices_file='sort_indices.npz',sort_indices_sha256=sha(root/'sort_indices.npz'),source_files=source,eval_code_sha256=e['eval_code_sha256'],eval_files=eval_files,methods=methods,auditor_sha256=sha(__file__))
    return receipt


def freeze_and_verify_sorts(root):
    exporter=Path(__file__).with_name('sort_export.py')
    command=['/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python',str(exporter),str(root)]
    if not (Path(root)/'sort_identity.json').exists():subprocess.run(command,check=True,capture_output=True,text=True)
    subprocess.run(command+['--verify'],check=True,capture_output=True,text=True)
    identity=json.loads((Path(root)/'sort_identity.json').read_text())
    with np.load(Path(root)/'sort_indices.npz',allow_pickle=False) as data:orders={k:data[k] for k in data.files}
    return orders,identity
