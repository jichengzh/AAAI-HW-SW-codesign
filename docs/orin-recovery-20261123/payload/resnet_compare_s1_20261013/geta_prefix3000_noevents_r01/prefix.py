"""Fixed real final-prefix diagnostic. Never full50k or accuracy qualification."""
import argparse,hashlib,itertools,json,os,subprocess,sys,time,traceback,fcntl,faulthandler
from pathlib import Path
import numpy as np
from protocol import check,send,receive,serve,encode_chunk,verify_reply,advance,digest,CHUNK
BASE=Path('/data/jichengzhi_v2x/resnet_compare_20261013')
NEW=BASE/'s7_hardware_adaptation_20261026'
COUNT=3000
SCOPE='diagnostic_no_cuda_timing_events_real_final_prefix3000_not_accuracy'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def rows_digest(rows):return digest(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())
def no_events_forward(runtime,x):
 """Blocking CPU return is the completion boundary; zero CUDA timing events."""
 import torch
 with torch.no_grad():
  data=torch.from_numpy(x.copy()).to(device=runtime.device,dtype=torch.float16)
  output=runtime.model(data)
  check(output.dtype==torch.float16,'Diagnostic output must remain half')
  # Tensor.to CPU defaults non_blocking=False: complete the real GPU output copy.
  cpu=output.to('cpu',non_blocking=False)
  return cpu.float().numpy()

def run_prefix(reader,writer,rows,infer,identity,log):
 cursor=itertools.count()
 def observed(x):
  index=next(cursor)
  log({'event':'before_forward','index':index,'sample_id':rows[index]['sample_id'],'monotonic_s':time.monotonic()})
  logits=np.asarray(infer(x),dtype=np.float32)
  check(logits.shape==(1,1000) and np.isfinite(logits).all(),'Nonfinite diagnostic output')
  log({'event':'after_forward','index':index,'sample_id':rows[index]['sample_id'],
       'monotonic_s':time.monotonic(),'finite':True,'logits_sha256':digest(logits.tobytes())})
  return logits
 serve(reader,writer,rows,observed,identity)

def prepare(a):
 import torch
 torch.set_num_threads(2)
 if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
 sys.path.insert(0,a.source)
 from scripts.resnet_frozen_validate import load_final_dataset
 dataset=load_final_dataset(Path(a.data),Path(a.final).parent)
 rows=dataset.rows[:COUNT];out=Path(a.output);out.mkdir(exist_ok=False)
 chunks=[]
 for start in range(0,COUNT,CHUNK):
  subset=rows[start:start+CHUNK]
  items=[dataset[i] for i in range(start,start+len(subset))]
  check([int(v[1]) for v in items]==[r['target'] for r in subset],'Frozen GT mismatch')
  inputs=np.stack([np.asarray(v[0],dtype=np.float32) for v in items])
  header,payload=encode_chunk(start,subset,inputs)
  name=f'chunk_{start:05d}'
  (out/(name+'.zlib')).write_bytes(payload);write(out/(name+'.json'),header)
  chunks.append({'index':start,'count':len(subset),'header':name+'.json','payload':name+'.zlib',
                 'header_sha256':sha(out/(name+'.json')),'payload_sha256':sha(out/(name+'.zlib'))})
  if start%256==0:print(json.dumps({'prepared':start+len(subset),'count':COUNT}),flush=True)
 contract={'scope':SCOPE,'samples':COUNT,'offset':0,'repetitions':1,'batch_size':1,
           'warmup':0,'jit_policy':'default_unchanged','final_file_sha256':sha(a.final),
           'final_sample_manifest_sha256':dataset.sample_manifest_sha256,'prefix_rows_sha256':rows_digest(rows),
           'preprocess_source_sha256':{str(p):sha(p) for p in (Path(a.source)/'scripts/resnet_frozen_validate.py',Path(a.source)/'framework/resnet_compare/data.py')},
           'code_sha256':{'prefix.py':sha(__file__),'protocol.py':sha(Path(__file__).with_name('protocol.py'))}}
 write(out/'contract.json',contract)
 write(out/'manifest.json',{'scope':SCOPE,'contract_sha256':sha(out/'contract.json'),'chunks':chunks,
                            'samples':COUNT,'prefix_rows':rows,'compressed_bytes':sum((out/c['payload']).stat().st_size for c in chunks)})

def target(a):
 import torch
 sys.path.insert(0,a.runtime)
 from framework.resnet_compare.trt_runtime import ScriptRuntime
 from framework.resnet_compare.runtime import _check_target
 from framework.resnet_compare.remote import APPROVED_ROOT
 contract=read(a.contract);audit=read(a.audit);verification=read(a.verification)
 check(contract['scope']==SCOPE and contract['samples']==COUNT and contract['offset']==0
       and contract['repetitions']==1 and contract['warmup']==0,'Fixed diagnostic contract')
 check(contract['code_sha256']=={'prefix.py':sha(__file__),'protocol.py':sha(Path(__file__).with_name('protocol.py'))},'Diagnostic source differs')
 check(sha(a.final)==contract['final_file_sha256'],'Frozen manifest differs')
 rows=read(a.final)['samples'][:COUNT];check(rows_digest(rows)==contract['prefix_rows_sha256'],'Prefix differs')
 check(audit['status']=='PASS_DIAGNOSTIC_METADATA_ONLY' and audit['non_debug_entry_bytes_identical'] is True
       and audit['qualification'] is False and audit['diagnostic_sha256']==sha(a.module),'Diagnostic audit differs')
 check(verification['diagnostic_artifact_sha256']==sha(a.module) and verification['state_sha256']==audit['state_sha256'],'Verified diagnostic model differs')
 output=Path(a.output);output.mkdir(exist_ok=False)
 faulthandler.enable(file=sys.stderr)
 faulthandler.dump_traceback_later(60,repeat=True,file=sys.stderr)
 try:
  with (Path(APPROVED_ROOT)/'.orin-runtime.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   torch.set_num_threads(4)
   check(torch._C._get_graph_executor_optimize(),'Default JIT required')
   runtime=ScriptRuntime(a.module,audit['state_sha256']);_check_target(runtime,'orin')
   # Explicitly exercise source-range decoding, before any model forward.
   graph=str(runtime.model.inlined_graph)
   (output/'target_graph.txt').write_text(graph)
   libdir=Path(torch.__file__).parent/'lib'
   libs={p.name:sha(p) for pattern in ('libtorch_cuda*.so*','libtorch_cpu*.so*','libc10*.so*') for p in libdir.glob(pattern) if p.is_file()}
   identity={'scope':SCOPE,'samples':COUNT,'artifact_sha256':sha(a.module),'state_sha256':audit['state_sha256'],
             'contract_sha256':sha(a.contract),'audit_sha256':sha(a.audit),'torch_version':torch.__version__,
             'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),'library_sha256':libs,
             'runtime_sha256':{p.name:sha(p) for p in sorted((Path(a.runtime)/'framework/resnet_compare').glob('*.py'))},
             'graph_source_range_print_passed':True,'graph_sha256':sha(output/'target_graph.txt'),
             'hardware':runtime.hardware,'default_jit':True,'tf32':False,'batch_size':1,'warmup':0,
             'cuda_timing_events':0,'execution_path':'direct_model_CPU_to_CUDA_half_then_blocking_CPU_return'}
   write(output/'identity.json',identity)
   with (output/'events.jsonl').open('x',buffering=1) as events:
    def log(row):
     text=json.dumps(row);events.write(text+'\n');print(text,file=sys.stderr,flush=True)
    def infer(x):return no_events_forward(runtime,x)
    run_prefix(sys.stdin.buffer,sys.stdout.buffer,rows,infer,identity,log)
   write(output/'result.json',{'status':'DIAGNOSTIC_PREFIX_COMPLETE','scope':SCOPE,'samples':COUNT,'accuracy_qualified':False})
 except BaseException as error:
  write(output/'failure.json',{'status':'DIAGNOSTIC_PARTIAL','scope':SCOPE,'error':repr(error)})
  traceback.print_exc(file=sys.stderr);raise
 finally:faulthandler.cancel_dump_traceback_later()

def host(a):
 cache=Path(a.cache);manifest=read(cache/'manifest.json');contract=read(cache/'contract.json')
 check(manifest['contract_sha256']==sha(cache/'contract.json') and manifest['samples']==COUNT,'Prepared prefix identity')
 check(sum(c['count'] for c in manifest['chunks'])==COUNT,'Prepared prefix length')
 output=Path(a.output);output.mkdir(exist_ok=False)
 command=json.loads(a.command_json)
 check(type(command) is list and all(type(s) is str for s in command),'Command argv')
 with (output/'target.stderr.log').open('x') as errors:
  proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
  try:
   ready,payload=receive(proc.stdout);identity=ready['identity']
   check(ready['type']=='ready' and not payload and identity['scope']==SCOPE
         and identity['contract_sha256']==sha(cache/'contract.json')
         and identity['artifact_sha256']==a.expected_artifact_sha256,'Target handshake identity')
   write(output/'target_identity.json',identity)
   chain='00'*32;count=0
   with (output/'chunk_receipts.jsonl').open('x',buffering=1) as records,(output/'logits.f32').open('xb') as logits:
    for chunk in manifest['chunks']:
     header=read(cache/chunk['header']);payload=(cache/chunk['payload']).read_bytes()
     check(chunk['index']==header['index']==count and sha(cache/chunk['header'])==chunk['header_sha256']
           and digest(payload)==chunk['payload_sha256'],'Prepared chunk changed/order')
     send(proc.stdin,header,payload);reply,raw=receive(proc.stdout);verify_reply(reply,raw,header)
     chain=advance(chain,reply,raw);logits.write(raw);logits.flush()
     records.write(json.dumps({**reply,'chain_sha256':chain})+'\n');count+=header['count']
     print(json.dumps({'diagnostic_samples':count,'count':COUNT}),flush=True)
    send(proc.stdin,{'type':'end','count':count});final,payload=receive(proc.stdout)
    check(not payload and final=={'type':'end','status':'COMPLETE','count':COUNT,'chain_sha256':chain},'Final prefix receipt')
   proc.stdin.close();check(proc.wait(timeout=60)==0,'Target exit failed')
   write(output/'result.json',{'status':'DIAGNOSTIC_PREFIX_COMPLETE','scope':SCOPE,'samples':count,
          'accuracy_qualified':False,'chain_sha256':chain,'logits_sha256':sha(output/'logits.f32')})
  except BaseException as error:
   proc.kill();proc.wait()
   write(output/'failure.json',{'status':'DIAGNOSTIC_PARTIAL','scope':SCOPE,'error':repr(error)});raise

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','host','target'))
 for name in ('module','audit','verification','contract','runtime','cache','command-json','expected-artifact-sha256'):
  p.add_argument('--'+name)
 p.add_argument('--output',required=True)
 p.add_argument('--source',default='/home/jichengzhi/V2X_resnet_compare_s1_20261013')
 p.add_argument('--data',default='/data/jichengzhi_v2x/datasets/imagenet1k_ilsvrc2012')
 p.add_argument('--final',default=str(BASE/'s1_10/data_manifests/final.json'))
 a=p.parse_args();{'prepare':prepare,'host':host,'target':target}[a.mode](a)
if __name__=='__main__':main()
