"""Root-window-only one-time build/reuse and frozen 256 host-difference evidence."""
import argparse,json,sys,fcntl,faulthandler
from pathlib import Path
import numpy as np
from contract import sha,read,check,validate_sources,actual_precision

def prepare(a):
 manifest=validate_sources(a)
 sys.path.insert(0,a.runtime)
 from framework.resnet_compare.trt_runtime import TensorRTRuntime,build_engine
 from framework.resnet_compare.remote import APPROVED_ROOT
 from framework.resnet_compare.runtime import _check_target
 import torch
 output=Path(a.output);check(not output.exists(),'New preparation directory required')
 check(sha(a.samples)==manifest['sample_pack_sha256'],'Original samples pack differs')
 with np.load(a.samples,allow_pickle=False) as data:inputs=data['inputs'];host=data['source_logits']
 check(inputs.shape==(256,3,224,224) and inputs.dtype==np.float32 and host.shape==(256,1000),'Sample contract')
 for i,row in enumerate(manifest['samples']):
  import hashlib
  check(hashlib.sha256(inputs[i].tobytes()).hexdigest()==row['input_sha256'] and hashlib.sha256(host[i].tobytes()).hexdigest()==row['source_logits_sha256'],'Sample row binding')
 with (Path(APPROVED_ROOT)/'.orin-runtime.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(4);faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
  engine=Path(a.engine)
  if engine.exists():
   check(a.reuse_build is not None,'Existing engine needs authenticated build record')
   build=read(a.reuse_build)
   check(build['engine_sha256']==sha(engine) and build['onnx_sha256']==sha(a.onnx),'Reuse engine/source mismatch')
  else:
   check(a.reuse_build is None,'Cannot rebuild missing historical engine')
   build=build_engine(a.onnx,a.engine,'FP16')
  check(build['precision_requested']=='FP16' and build['tf32_enabled'] is False and build['workspace_bytes']==1<<30,'Original build policy required')
  runtime=TensorRTRuntime(engine);_check_target(runtime,'orin');precision=actual_precision(runtime.layer_precision)
  output.mkdir();logits=[]
  for index,x in enumerate(inputs):
   print(json.dumps({'event':'before_forward','index':index}),file=sys.stderr,flush=True)
   y=runtime.infer(torch.from_numpy(x[None].copy()))['logits'].float().numpy()
   check(y.shape==(1,1000) and np.isfinite(y).all(),'Nonfinite/wrong shape')
   logits.append(y);print(json.dumps({'event':'after_forward','index':index}),file=sys.stderr,flush=True)
  faulthandler.cancel_dump_traceback_later()
 logits=np.concatenate(logits);np.savez(output/'legacy_comparison_logits.npz',logits=logits,host_logits=host)
 close=np.isclose(logits,host,atol=.01,rtol=.01)
 c={'engine_sha256':sha(a.engine),'onnx_sha256':sha(a.onnx),'source_manifest_sha256':sha(a.manifest),
    'build':build,'layer_precision':runtime.layer_precision,'actual_precision':precision,
    'legacy_host_reference':{'source_pack_sha256':sha(a.samples),'reference_kind':'original_FP32_diagnostic_not_target_accuracy',
      'comparison_sha256':sha(output/'legacy_comparison_logits.npz'),'sample_ids':[r['sample_id'] for r in manifest['samples']],
      'atol':.01,'rtol':.01,'failed_samples':np.flatnonzero(~close.all(1)).tolist(),'max_abs':float(abs(logits-host).max()),
      'is_host_equivalence_gate':False},'prepare_code_sha256':sha(__file__),'host_accuracy_reused':False}
 (output/'contract.json').write_text(json.dumps(c,indent=2))
 return c

def main():
 p=argparse.ArgumentParser()
 for name in ('manifest','onnx','engine','runtime','samples','output'):p.add_argument('--'+name,required=True)
 p.add_argument('--reuse-build');prepare(p.parse_args())
if __name__=='__main__':main()
