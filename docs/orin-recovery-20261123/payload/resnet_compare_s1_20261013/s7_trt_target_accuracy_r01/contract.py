"""Exact TRT engine qualification; never substitutes source/host accuracy."""
import hashlib,json
import numpy as np
from pathlib import Path
SCOPE='independent_Orin_TensorRT_full50000_accuracy_not_host_equivalence'
ACCEPTED_TRANSPORTS={'73be4628be4e86c24419b345f778054f4d145ffb5495507834e93fd4fc5c2f53','9025fcb486a1e6317416595ef3b0a8f94376d1a44a972492cc68e0586074b50b','0bf75bdb3c15b958f36a76a35b663377687b2ebbd72ead8be5d3f46b3f5090b5'}
FINAL='98395d0907d3bee96b59ed4baa9d5aa6d0b693599d33163a3fc7aa048595d2c5'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def check(ok,message):
 if not ok:raise ValueError(message)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def actual_precision(rows):
 check(rows and all(r['precision'] not in ('UNKNOWN','') for r in rows),'Actual precision missing/unknown')
 compute=[r for r in rows if r['inspector'].get('LayerType') not in ('Reformat','Shuffle','Identity','Constant')]
 check(any('FP16' in r['precision'] for r in compute),'No real FP16 compute layers')
 check(all(set(r['precision'].split('+'))<={'FP16','FP32','INT32'} for r in rows),'Unexpected execution precision')
 return {'fp32_sections':[r['name'] for r in rows if 'FP32' in r['precision']],
         'fp16_compute_layers':sum('FP16' in r['precision'] for r in compute),
         'inspector_sha256':digest(rows),'accumulator_precision_guaranteed':False}
def validate_sources(a):
 m=read(a.manifest)
 check(m['transport_id'] in ACCEPTED_TRANSPORTS,'Authenticated original r01 transport required')
 check(m['precision']=='FP16' and m['candidate']['native_low_bit'] is False,'Only original r01 hardware FP16 policy')
 check(m['candidate']['run_id'].endswith('_FP16_r01'),'Reject changed precision-policy rounds')
 body={k:v for k,v in m.items() if k!='transport_id'}
 check(m['transport_id']==digest(body),'Source manifest digest differs')
 check(sha(a.onnx)==m['export_sha256'],'Source ONNX changed')
 root=Path(a.runtime)/'framework/resnet_compare'
 check({p.name:sha(p) for p in root.glob('*.py')}==m['code_sha256'],'Frozen r01 runtime differs')
 return m
def validate_contract(a):
 m=validate_sources(a);c=read(a.contract)
 check(c['engine_sha256']==sha(a.engine),'Exact engine bytes changed')
 check(c['source_manifest_sha256']==sha(a.manifest) and c['onnx_sha256']==sha(a.onnx),'Engine source binding differs')
 check(c['build']['engine_sha256']==c['engine_sha256'] and c['build']['onnx_sha256']==c['onnx_sha256'],'Build binding differs')
 check(c['build']['precision_requested']=='FP16' and c['build']['tf32_enabled'] is False
       and c['build']['workspace_bytes']==1<<30,'Original r01 build policy differs')
 check(c['actual_precision']==actual_precision(c['layer_precision']),'Inspector summary changed')
 check(c['legacy_host_reference']['source_pack_sha256']==m['sample_pack_sha256'],'Historical reference identity')
 legacy=c['legacy_host_reference'];path=Path(a.contract).parent/'legacy_comparison_logits.npz'
 check(sha(path)==legacy['comparison_sha256'],'Legacy comparison evidence changed')
 with np.load(path,allow_pickle=False) as data:logits=data['logits'];host=data['host_logits']
 check(logits.shape==host.shape==(256,1000) and np.isfinite(logits).all() and np.isfinite(host).all(),'Legacy logits contract')
 for i,row in enumerate(m['samples']):check(hashlib.sha256(host[i].tobytes()).hexdigest()==row['source_logits_sha256'],'Legacy host row differs')
 check(legacy['sample_ids']==[r['sample_id'] for r in m['samples']] and legacy['atol']==legacy['rtol']==.01,'Legacy order/tolerance differs')
 check(legacy['failed_samples']==np.flatnonzero(~np.isclose(logits,host,atol=.01,rtol=.01).all(1)).tolist(),'Legacy mismatch evidence differs')
 check(c['prepare_code_sha256']==sha(Path(__file__).with_name('prepare_engine.py')),'Preparation source changed')
 return c

def qualify(result_path,audit_path,expected):
 p=Path(result_path);r=read(p)
 check(r.get('status')=='COMPLETE' and r.get('samples')==50000,'Full50000 incomplete')
 check(r.get('scope')==SCOPE and r.get('final_used_for_selection') is False,'Target scope differs')
 for key,value in expected.items():check(r['target_identity'][key]==value,'Target identity differs '+key)
 for file,key in [('logits.f32','logits_sha256'),('predictions.jsonl','predictions_sha256')]:
  check(sha(p.parent/file)==r[key],'Full output evidence differs')
 check((p.parent/'logits.f32').stat().st_size==50000*1000*4,'Full output size')
 audit=read(audit_path)
 check(audit.get('status')=='PASS' and audit.get('result_sha256')==sha(p) and audit.get('samples')==50000
       and all(audit.get(k) is True for k in ('full_logits_verified','frozen_gt_verified','metrics_recomputed')),'Independent audit incomplete')
 for key in ('top1_correct','top5_correct'):check(audit[key]==r[key],'Independent metrics differ')
 return {'result_sha256':sha(p),'audit_sha256':sha(audit_path),'target_identity':r['target_identity'],
         'qualified_accuracy_only':True,'timing_or_energy_qualified':False}
