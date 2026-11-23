"""Bounded target stream using frozen fullval SDK; no accuracy claim."""
import fcntl,hashlib,json,os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prefix import select_rows
from frozen_stream import serve

CODE='/home/jichengzhi/resnet_compare_s1_20261013/sdk_geta_rebuilt_fullval_r02/code'
LOCK=Path('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock')
def sha(b):return hashlib.sha256(b).hexdigest()
class Recorder:
 def __init__(self,out,rows,infer):self.out=out;self.rows=rows;self.infer=infer;self.index=0
 def __enter__(self):self.file=(self.out/'rows.jsonl').open('x');return self
 def __exit__(self,*args):self.file.close()
 def __call__(self,x):
  i=self.index
  if i>=len(self.rows) or sha(x.tobytes())!=self.rows[i]['input_sha256']:raise ValueError('input identity mismatch')
  y=self.infer(x)
  if y.shape!=(1,1000) or y.dtype!=np.float32 or not np.isfinite(y).all():raise ValueError('output invalid')
  np.save(self.out/('output_%05d.npy'%i),y,allow_pickle=False)
  self.file.write(json.dumps({'index':i,'sample_id':self.rows[i]['sample_id'],'input_sha256':sha(x.tobytes()),'output_sha256':sha(y.tobytes())})+'\n');self.file.flush();os.fsync(self.file.fileno());self.index+=1
  return y

def run(out,manifest,reader,writer):
 from types import SimpleNamespace
 sys.path.insert(0,CODE)
 from sdk_backend import assert_no_torch,library_identity
 from identity import validate,verify_libraries
 from multi_entry import create_runtime
 from stage_logger import bind,record
 args=SimpleNamespace(point='geta',engine='/home/jichengzhi/resnet_compare_s1_20261013/geta_official_rebuild_r01/run_build/candidate_fp16.engine',layers='/home/jichengzhi/resnet_compare_s1_20261013/geta_official_rebuild_r01/run_build/layers.json',manifest='/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_target_accuracy_r01/geta/source/transfer_manifest.json',onnx='/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_target_accuracy_r01/geta/source/candidate.onnx',final='/home/jichengzhi/resnet_compare_s1_20261013/s7_trt_target_accuracy_r01/final.json',output=str(out))
 assert_no_torch();token=validate('geta',args);rows=select_rows(json.loads(Path(args.final).read_text())['samples']);items=json.loads(Path(manifest).read_text())['rows']
 if len(items)!=2432 or any(a['sample_id']!=b['sample_id'] for a,b in zip(rows,items)):raise ValueError('manifest order')
 out=Path(out);out.mkdir(exist_ok=False)
 with LOCK.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runtime=None
  try:
   with bind(out/'init_stages.jsonl'):
    runtime=record('create_runtime',lambda:create_runtime(args));before=record('library_identity',library_identity)
   verify_libraries(before)
   token={**token,'library_sha256':before,'pid':os.getpid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'tensorrt_version':'8.5.2.2','torch_imported':False,'scope':'DIAGNOSTIC_ONLY','count':2432}
   (out/'target_identity.json').write_text(json.dumps(token,indent=2)+'\n');(out/'maps_before.txt').write_text(Path('/proc/self/maps').read_text())
   with Recorder(out,items,runtime.infer) as recorder:serve(reader,writer,rows,recorder,token)
   after=library_identity();verify_libraries(after);assert_no_torch()
   if after!=before:raise ValueError('library drift')
   (out/'maps_after.txt').write_text(Path('/proc/self/maps').read_text())
  finally:
   if runtime is not None:runtime.close()
  (out/'completion.json').write_text(json.dumps({'status':'DIAGNOSTIC_COMPLETE','samples':2432,'library_sha256':after})+'\n')

if __name__=='__main__':
 if len(sys.argv)!=3:raise ValueError('output and input manifest required')
 sys.path.insert(0,CODE)
 from multi_entry import protocol_output
 with protocol_output() as wire:run(Path(sys.argv[1]),Path(sys.argv[2]),sys.stdin.buffer,wire)
