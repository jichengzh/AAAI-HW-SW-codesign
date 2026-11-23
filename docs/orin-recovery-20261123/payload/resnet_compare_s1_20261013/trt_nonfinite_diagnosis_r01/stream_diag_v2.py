"""Instrument the frozen target entry without changing engine/infer or accepting accuracy."""
import argparse,json,sys,subprocess,importlib,hashlib
from pathlib import Path
import numpy as np
from diagnose import output_stats
class PrefixDataset:
 def __init__(self,dataset,count):
  if not 0<count<50000:raise ValueError('Bounded diagnostic only')
  self.dataset=dataset;self.rows=dataset.rows[:count];self.manifest_sha256=dataset.manifest_sha256
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  if not 0<=i<len(self):raise IndexError(i)
  return self.dataset[i]
def capture_buffers(runtime,tensor,output,index):
 directory=output/('failure_%05d'%index);directory.mkdir()
 events=[event.query() for event in runtime.events]
 arrays={'expected_cpu_input':tensor.numpy().copy(),'pinned_cpu_input':runtime.host_input.numpy().copy(),'pinned_cpu_output':runtime.host_output.numpy().copy(),
         'gpu_input':runtime.buffers[runtime.input_index].detach().cpu().numpy().copy(),'gpu_output':runtime.buffers[runtime.output_index].detach().cpu().numpy().copy()}
 for name,array in arrays.items():np.save(directory/(name+'.npy'),array,allow_pickle=False)
 report={'events_complete':events,'capture_changes_synchronization_after_failed_inference':True,
         'cpu_input_equal_expected':bool(np.array_equal(arrays['expected_cpu_input'],arrays['pinned_cpu_input'])),
         'gpu_input_equal_expected':bool(np.array_equal(arrays['expected_cpu_input'],arrays['gpu_input'])),
         'host_gpu_output_equal_including_nan':bool(np.array_equal(arrays['pinned_cpu_output'],arrays['gpu_output'],equal_nan=True)),
         'arrays':{name:output_stats(array) for name,array in arrays.items()}}
 (directory/'buffers.json').write_text(json.dumps(report,indent=2))
def recording_runtime(base,output):
 class Recorded(base):
  def infer(self,tensor):
   index=getattr(self,'diagnostic_index',0)
   result=super().infer(tensor)
   raw=result['logits'].float().numpy().copy()
   np.save(output/('output_%05d.npy'%index),raw,allow_pickle=False)
   row={'index':index,'input_sha256':hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),**output_stats(raw)}
   with (output/'raw_rows.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
   if not np.isfinite(raw).all():capture_buffers(self,tensor,output,index)
   self.diagnostic_index=index+1
   return result
 return Recorded
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('host','target'))
 for n in ('entry','contract','manifest','onnx','engine','final','runtime','output'):p.add_argument('--'+n,required=True)
 p.add_argument('--count',type=int,default=512);p.add_argument('--command-json');a=p.parse_args()
 if not 0<a.count<=1000:raise ValueError('Bounded 1..1000 only')
 sys.path.insert(0,a.entry);original=importlib.import_module('run_accuracy')
 output=Path(a.output);output.mkdir(exist_ok=False)
 base_identity=original.identity
 def identity(args):return {**base_identity(args),'diagnostic_scope':'bounded_stream_prefix_not_accuracy','diagnostic_count':a.count,'diagnostic_code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('stream_diag_v2.py','diagnose.py')}}
 original.identity=identity
 if a.mode=='target':
  sys.path.insert(0,a.runtime)
  from framework.resnet_compare import trt_runtime
  trt_runtime.TensorRTRuntime=recording_runtime(trt_runtime.TensorRTRuntime,output)
  original_serve=original.serve
  original.serve=lambda reader,writer,rows,infer,token:original_serve(reader,writer,rows[:a.count],infer,token)
  with original.protocol_output() as wire:original.target(a,wire)
  (output/'diagnostic_exit.json').write_text(json.dumps({'status':'DIAGNOSTIC_COMPLETE','count':a.count}))
 else:
  original.configure_host_threads();sys.path.insert(0,'/home/jichengzhi/V2X_resnet_compare_s1_20261013')
  from scripts.resnet_frozen_validate import load_final_dataset
  dataset=PrefixDataset(load_final_dataset(Path('/data/jichengzhi_v2x/datasets/imagenet1k_ilsvrc2012'),Path(a.final).parent),a.count)
  command=json.loads(a.command_json)
  with (output/'target.stderr.log').open('x') as errors:
   proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
   try:
    result=original.evaluate(proc.stdout,proc.stdin,dataset,output/'stream_result',identity(a));proc.stdin.close();code=proc.wait(timeout=30)
    if code:raise ValueError('Target exit '+str(code))
    (output/'diagnostic_result.json').write_text(json.dumps({**result,'status':'DIAGNOSTIC_COMPLETE','scope':'bounded_stream_prefix_not_accuracy','target_exit':code},indent=2))
   except BaseException as error:
    if proc.poll() is None:proc.terminate()
    proc.wait(timeout=30);(output/'failure.json').write_text(json.dumps({'status':'DIAGNOSTIC_FAILED','reason':str(error),'target_exit':proc.returncode}));raise
if __name__=='__main__':main()
