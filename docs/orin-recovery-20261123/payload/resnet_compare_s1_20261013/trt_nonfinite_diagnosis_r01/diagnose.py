"""Bounded same-engine replay; retain nonfinite raw outputs without qualification."""
import argparse,hashlib,json,sys,fcntl,faulthandler
from pathlib import Path
import numpy as np
ENGINE='68c67e61977d3ed20ccead95fae5da67677f791ab80c07444585577f46ea5c48'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_input(x):
 if x.shape!=(1,3,224,224) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Invalid diagnostic input')
def output_stats(x):
 return {'shape':list(x.shape),'dtype':str(x.dtype),'finite':bool(np.isfinite(x).all()),'nan':int(np.isnan(x).sum()),'positive_inf':int(np.isposinf(x).sum()),'negative_inf':int(np.isneginf(x).sum()),'finite_max_abs':float(np.abs(x[np.isfinite(x)]).max()) if np.isfinite(x).any() else None,'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
def run(a):
 import torch
 if sha(a.engine)!=ENGINE:raise ValueError('Historical engine mismatch')
 sys.path.insert(0,a.runtime)
 from framework.resnet_compare.trt_runtime import TensorRTRuntime
 from framework.resnet_compare.remote import APPROVED_ROOT
 data=np.load(a.inputs,allow_pickle=False);inputs=data['inputs'];indices=data['indices']
 manifest=json.loads(Path(a.manifest).read_text())
 if sha(a.inputs)!=manifest['inputs_sha256']:raise ValueError('Input file differs')
 output=Path(a.output);output.mkdir(exist_ok=False);results=[];raw=[]
 order=list(range(len(indices)))+[list(indices).index(157)]*16+list(range(len(indices)))
 with (Path(APPROVED_ROOT)/'.orin-runtime.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
  faulthandler.enable();faulthandler.dump_traceback_later(30,repeat=True)
  runtime=TensorRTRuntime(a.engine)
  for request,i in enumerate(order):
   x=inputs[i:i+1].copy();check_input(x)
   if hashlib.sha256(x.tobytes()).hexdigest()!=manifest['rows'][i]['input_sha256']:raise ValueError('Row differs')
   print(json.dumps({'before':request,'index':int(indices[i])}),flush=True)
   y=runtime.infer(torch.from_numpy(x))['logits'].float().numpy().copy()
   raw.append(y);row={'request':request,'index':int(indices[i]),'input_sha256':manifest['rows'][i]['input_sha256'],**output_stats(y)};results.append(row)
   np.save(output/('output_%03d.npy'%request),y,allow_pickle=False)
   with (output/'rows.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
  faulthandler.cancel_dump_traceback_later()
 (output/'result.json').write_text(json.dumps({'status':'DIAGNOSTIC_COMPLETE','engine_sha256':sha(a.engine),'script_sha256':sha(__file__),'input_pack_sha256':sha(a.inputs),'torch_version':torch.__version__,'hardware':runtime.hardware,'requests':len(results),'nonfinite_requests':sum(not r['finite'] for r in results)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('engine','runtime','inputs','manifest','output'):p.add_argument('--'+n,required=True)
 run(p.parse_args())
