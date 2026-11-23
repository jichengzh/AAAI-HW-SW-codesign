"""Root-scheduled host/target entry; never changes optimization or precision policy."""
import argparse,json,sys,subprocess,hashlib,fcntl
from pathlib import Path
from stream_accuracy import check,digest,serve,evaluate
from deployment import validate_deployment,no_events_forward
FINAL_DIGEST='98395d0907d3bee96b59ed4baa9d5aa6d0b693599d33163a3fc7aa048595d2c5'
BASE=Path('/data/jichengzhi_v2x/resnet_compare_20261013')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def library_hashes(torch_init):
 root=Path(torch_init).parent/'lib'
 files=sorted({p for pattern in ('libtorch_cuda*.so*','libtorch_cpu*.so*','libc10*.so*') for p in root.glob(pattern) if p.is_file()})
 return {p.name:sha(p) for p in files}
def configure_host_threads():
 import torch
 torch.set_num_threads(2)
 if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
def identity(a):
 export=json.loads(Path(a.export).read_text());final=json.loads(Path(a.final).read_text())
 check(final['count']==len(final['samples'])==50000 and final['sha256']==FINAL_DIGEST,'Frozen final split')
 deployment=validate_deployment(a,export)
 runtime=Path(a.runtime)/'framework/resnet_compare'
 return {**{k:export[k] for k in ('artifact_sha256','adapted_model_state_sha256','source_model_state_sha256','source_checkpoint_sha256')},**deployment,
         'export_metadata_sha256':sha(a.export),'final_manifest_file_sha256':sha(a.final),
         'final_sample_manifest_sha256':FINAL_DIGEST,'precision_policy':'all_fp16','batch_size':1,
         'jit_policy':'default_unchanged','code_sha256':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('stream_accuracy.py'),Path(__file__).with_name('deployment.py'))},
         'runtime_sha256':{p.name:sha(p) for p in sorted(runtime.glob('*.py'))}}
def target(a):
 import torch,numpy as np
 sys.path.insert(0,a.runtime)
 from framework.resnet_compare.trt_runtime import ScriptRuntime
 from framework.resnet_compare.runtime import _check_target
 from framework.resnet_compare.remote import APPROVED_ROOT
 token=identity(a)
 with (Path(APPROVED_ROOT)/'.orin-runtime.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(4)
  runtime=ScriptRuntime(a.module,token['adapted_model_state_sha256']);_check_target(runtime,'orin')
  check(torch._C._get_graph_executor_optimize(),'Default JIT optimization must be active')
  token={**token,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
         'cudnn_version':torch.backends.cudnn.version(),'hardware':runtime.hardware,
         'script_evidence':{k:v for k,v in runtime.script_evidence.items() if k!='graph'},'cudnn_benchmark':torch.backends.cudnn.benchmark,
         'tf32':False,'numpy_version':np.__version__,
         'library_sha256':{'torch_init':sha(torch.__file__),'torch_C':sha(torch._C.__file__),**library_hashes(torch.__file__)}}
  token={**token,'target_source_range_graph_sha256':hashlib.sha256(str(runtime.model.inlined_graph).encode()).hexdigest()}
  rows=json.loads(Path(a.final).read_text())['samples']
  import itertools
  cursor=itertools.count()
  def infer(x):
   index=next(cursor)
   print(json.dumps({'event':'before_forward','index':index,'sample_id':rows[index]['sample_id']}),file=sys.stderr,flush=True)
   logits=no_events_forward(runtime,x)
   print(json.dumps({'event':'after_forward','index':index,'finite':True}),file=sys.stderr,flush=True)
   return logits
  serve(sys.stdin.buffer,sys.stdout.buffer,rows,infer,token)
def host(a):
 configure_host_threads()
 sys.path.insert(0,a.source)
 from scripts.resnet_frozen_validate import load_final_dataset
 dataset=load_final_dataset(Path(a.data),Path(a.final).parent)
 token=identity(a)
 check(dataset.manifest_sha256==token['final_manifest_file_sha256'],'Loader manifest differs')
 command=json.loads(a.command_json)
 check(isinstance(command,list) and command and all(isinstance(x,str) for x in command),'Command argv')
 output=Path(a.output)
 check(not output.exists(),'Output must be new')
 error_path=output.with_suffix('.stderr.log')
 with error_path.open('x') as errors:
  proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
  try:
   result=evaluate(proc.stdout,proc.stdin,dataset,output,token)
   proc.stdin.close();check(proc.wait(timeout=60)==0,'Target exit failure')
   result={**result,'status':'COMPLETE','scope':'independent_Orin_full50000_noevents_metadata_compatible_accuracy_not_host_equivalence',
           'final_used_for_selection':False,'seed':17,'backend':'PyTorch TorchScript',
           'host_source_sha256':{str(p):sha(p) for p in (Path(a.source)/'scripts/resnet_frozen_validate.py',Path(a.source)/'framework/resnet_compare/data.py')},
           'predictions_sha256':sha(output/'predictions.jsonl'),'logits_sha256':sha(output/'logits.f32')}
   (output/'result.json').write_text(json.dumps(result,indent=2))
  except BaseException as error:
   proc.kill();proc.wait()
   output.mkdir(exist_ok=True)
   (output/'failure.json').write_text(json.dumps({'status':'PARTIAL','reason':str(error)}));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('host','target'))
 for name in ('module','source-module','metadata-audit','export','final','runtime'):p.add_argument('--'+name,required=True)
 p.add_argument('--output');p.add_argument('--command-json')
 p.add_argument('--source',default='/home/jichengzhi/V2X_resnet_compare_s1_20261013')
 p.add_argument('--data',default=str(BASE/'../datasets/imagenet1k_ilsvrc2012'))
 a=p.parse_args()
 if a.mode=='host':check(a.output is not None and a.command_json is not None,'Host needs output and command');host(a)
 else:target(a)
if __name__=='__main__':main()
