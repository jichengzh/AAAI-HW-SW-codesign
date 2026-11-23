"""Root-scheduled host/target entry; never changes optimization or precision policy."""
import argparse,json,sys,subprocess,hashlib,fcntl,os,re
from contextlib import contextmanager
from pathlib import Path
from stream_accuracy import check,digest,serve,evaluate
from contract import validate_contract, SCOPE, actual_precision
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
def loaded_trt_libraries(maps='/proc/self/maps'):
 paths={line.split()[-1] for line in Path(maps).read_text().splitlines() if len(line.split())>=6 and line.split()[-1].startswith('/')}
 files=sorted(p for p in paths if any(token in Path(p).name for token in ('libnvinfer','libcudart')))
 check(any('libnvinfer.so' in Path(p).name for p in files),'Loaded TensorRT binary missing')
 check(any('libcudart.so' in Path(p).name for p in files),'Loaded CUDA runtime binary missing')
 return {str(Path(p).resolve()):sha(p) for p in files}
def configure_host_threads():
 import torch
 torch.set_num_threads(2)
 if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
def identity(a):
 contract=validate_contract(a);manifest=json.loads(Path(a.manifest).read_text());final=json.loads(Path(a.final).read_text())
 check(final['count']==len(final['samples'])==50000 and final['sha256']==FINAL_DIGEST,'Frozen final split')
 runtime=Path(a.runtime)/'framework/resnet_compare'
 return {'engine_sha256':contract['engine_sha256'],'contract_sha256':sha(a.contract),
         'source_manifest_sha256':sha(a.manifest),'onnx_sha256':sha(a.onnx),
         'source_model_state_sha256':manifest['candidate']['source_model_state_sha256'],
         'source_checkpoint_sha256':manifest['candidate']['candidate_artifact_sha256'],
         'candidate_id':manifest['candidate']['candidate_id'],'legacy_host_reference':contract['legacy_host_reference'],
         'actual_precision':contract['actual_precision'],'build':contract['build'],
         'final_manifest_file_sha256':sha(a.final),'final_sample_manifest_sha256':FINAL_DIGEST,
         'precision_policy':'original_r01_TRT_FP16_enabled_TF32_disabled','batch_size':1,
         'execution_path':'unchanged_TensorRTRuntime_infer_preallocated_four_events',
         'code_sha256':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('stream_accuracy.py'),Path(__file__).with_name('contract.py'),Path(__file__).with_name('target_wrapper.py'))},
         'runtime_sha256':{p.name:sha(p) for p in sorted(runtime.glob('*.py'))}}

@contextmanager
def protocol_output():
 """Keep binary wire on original fd; redirect Python/native stdout to stderr."""
 sys.stdout.flush()
 saved=os.dup(1)
 try:
  with os.fdopen(os.dup(saved),'wb',buffering=0) as wire:
   os.dup2(2,1)
   yield wire
 finally:
  sys.stdout.flush();os.dup2(saved,1);os.close(saved)

def target(a,wire=None):
 import torch,numpy as np
 sys.path.insert(0,a.runtime)
 from framework.resnet_compare.trt_runtime import TensorRTRuntime
 from framework.resnet_compare.runtime import _check_target
 from framework.resnet_compare.remote import APPROVED_ROOT
 token=identity(a)
 with (Path(APPROVED_ROOT)/'.orin-runtime.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(4)
  import faulthandler,tensorrt as trt
  faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
  runtime=TensorRTRuntime(a.engine);_check_target(runtime,'orin')
  check(actual_precision(runtime.layer_precision)==token['actual_precision'],'Actual engine inspector differs')
  check(trt.__version__==token['build']['tensorrt_version'],'TensorRT build/runtime version differs')
  token={**token,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
         'cudnn_version':torch.backends.cudnn.version(),'hardware':runtime.hardware,
         'tensorrt_version':trt.__version__,'cudnn_benchmark':torch.backends.cudnn.benchmark,
         'tf32':False,'numpy_version':np.__version__,
         'library_sha256':{'torch_init':sha(torch.__file__),'torch_C':sha(torch._C.__file__),**library_hashes(torch.__file__),**loaded_trt_libraries()}}
  rows=json.loads(Path(a.final).read_text())['samples']
  import itertools
  cursor=itertools.count()
  def infer(x):
   index=next(cursor)
   print(json.dumps({'event':'before_forward','index':index,'sample_id':rows[index]['sample_id']}),file=sys.stderr,flush=True)
   logits=runtime.infer(torch.from_numpy(x.copy()))['logits'].float().numpy()
   check(logits.shape==(1,1000) and np.isfinite(logits).all(),'Invalid target logits')
   print(json.dumps({'event':'after_forward','index':index,'finite':True}),file=sys.stderr,flush=True)
   return logits
  serve(sys.stdin.buffer,wire if wire is not None else sys.stdout.buffer,rows,infer,token)
  faulthandler.cancel_dump_traceback_later()
def record_exit(output,error_path,code,command):
 matches=re.findall(r'^REMOTE_FULLVAL_EXIT=(-?\d+)$',error_path.read_text(),flags=re.MULTILINE)
 remote=int(matches[0]) if len(matches)==1 else None
 value={'outer_exit_code':code,'remote_exit_code':remote,'remote_exit_records':len(matches),'target_command':command}
 output.mkdir(exist_ok=True)
 (output/'outer_exit.json').write_text(json.dumps(value,indent=2))
 return value

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
   proc.stdin.close();code=proc.wait(timeout=60)
   exit_evidence=record_exit(output,error_path,code,command)
   check(code==0 and exit_evidence['remote_exit_code']==0,'Target/outer exit failure or missing marker')
   result={**result,'status':'COMPLETE','scope':SCOPE,
           'final_used_for_selection':False,'seed':17,'backend':'TensorRT',
           'exit_evidence':exit_evidence,'outer_exit_sha256':sha(output/'outer_exit.json'),
           'host_source_sha256':{str(p):sha(p) for p in (Path(a.source)/'scripts/resnet_frozen_validate.py',Path(a.source)/'framework/resnet_compare/data.py')},
           'predictions_sha256':sha(output/'predictions.jsonl'),'logits_sha256':sha(output/'logits.f32')}
   (output/'result.json').write_text(json.dumps(result,indent=2))
  except BaseException as error:
   proc.kill();code=proc.wait()
   record_exit(output,error_path,code,command)
   output.mkdir(exist_ok=True)
   (output/'failure.json').write_text(json.dumps({'status':'PARTIAL','reason':str(error)}));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('host','target','qualify'))
 for name in ('contract','manifest','onnx','engine','final','runtime'):p.add_argument('--'+name,required=True)
 p.add_argument('--output');p.add_argument('--command-json');p.add_argument('--result');p.add_argument('--audit')
 p.add_argument('--source',default='/home/jichengzhi/V2X_resnet_compare_s1_20261013')
 p.add_argument('--data',default=str(BASE/'../datasets/imagenet1k_ilsvrc2012'))
 a=p.parse_args()
 if a.mode=='host':check(a.output is not None and a.command_json is not None,'Host needs output and command');host(a)
 elif a.mode=='qualify':
  from contract import qualify
  check(a.result and a.audit and a.output,'Qualification needs result/audit/output')
  result=qualify(a.result,a.audit,identity(a))
  with Path(a.output).open('x') as f:json.dump(result,f,indent=2)
 else:
  with protocol_output() as wire:target(a,wire)
if __name__=='__main__':main()
