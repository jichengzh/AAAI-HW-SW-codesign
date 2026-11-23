"""Bounded by external timeout: existing TensorRT engine deserialization only.

No torch, execution context, inference, CUDA buffer allocation or engine builder.
A COMPLETE result is only a diagnostic milestone, never model accuracy/performance.
"""
import argparse
import faulthandler
import fcntl
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ENGINE_SHA256='68c67e61977d3ed20ccead95fae5da67677f791ab80c07444585577f46ea5c48'
LOCK=Path('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock')

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_maps():
 return Path('/proc/self/maps').read_text()

def loaded_modules():
 return sorted(name for name in sys.modules if name=='torch' or name.startswith('torch.'))

def write(path,value):
 Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def snapshot(output,stage,extra=None):
 maps=read_maps();mapping=output/(stage+'.maps');mapping.write_text(maps)
 modules=loaded_modules()
 forbidden=[line for line in maps.splitlines() if any(token in line.lower() for token in ('/torch/','libtorch','torch_c','libc10'))]
 event={'stage':stage,'pid':os.getpid(),'epoch':time.time(),'monotonic':time.monotonic(),
        'maps_sha256':sha(mapping),'torch_modules':modules,'torch_native_mappings':forbidden,**(extra or {})}
 with (output/'events.jsonl').open('a') as log:
  log.write(json.dumps(event,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 if modules or forbidden:raise ValueError('Torch present in SDK-only process')
 return event

def execute(engine_path,output,lock_path=LOCK):
 engine_path=Path(engine_path);output=Path(output);output.mkdir(exist_ok=False)
 try:
  if platform.machine()!='aarch64':raise ValueError('Orin aarch64 target required')
  if sha(engine_path)!=ENGINE_SHA256:raise ValueError('Original dense engine SHA required')
  write(output/'launch.json',{'pid':os.getpid(),'python':sys.executable,'python_version':sys.version,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'uname':list(os.uname()),'engine_path':str(engine_path),'engine_sha256':ENGINE_SHA256,
        'source_sha256':sha(__file__),'lock_path':str(lock_path),'inference_executed':False})
  with Path(lock_path).open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   faulthandler.enable();faulthandler.dump_traceback_later(30,repeat=True)
   try:
    snapshot(output,'before_import')
    trt=importlib.import_module('tensorrt')
    identity={'tensorrt_version':trt.__version__,'module_path':trt.__file__,'module_sha256':sha(trt.__file__)}
    snapshot(output,'after_import',identity)
    if trt.__version__!='8.5.2.2':raise ValueError('Original TensorRT version required')
    logger=trt.Logger(trt.Logger.WARNING)
    snapshot(output,'before_runtime')
    runtime=trt.Runtime(logger)
    snapshot(output,'after_runtime')
    payload=engine_path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=ENGINE_SHA256:raise ValueError('Original engine changed before deserialization')
    snapshot(output,'before_deserialize')
    engine=runtime.deserialize_cuda_engine(payload)
    if engine is None:raise ValueError('TensorRT returned null engine')
    snapshot(output,'after_deserialize')
    snapshot(output,'before_cleanup')
    del engine
    del runtime
    snapshot(output,'after_cleanup')
    if sha(engine_path)!=ENGINE_SHA256:raise ValueError('Original engine changed during diagnosis')
    result={'status':'SDK_DESERIALIZATION_COMPLETE_NOT_INFERENCE','engine_sha256':ENGINE_SHA256,
            'torch_imported':False,'inference_executed':False,**identity}
    write(output/'result.json',result)
    return result
   finally:faulthandler.cancel_dump_traceback_later()
 except BaseException as error:
  write(output/'failure.json',{'status':'SDK_DIAGNOSTIC_FAILED','error_type':type(error).__name__,'reason':str(error)})
  raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--engine',required=True);parser.add_argument('--output',required=True)
 args=parser.parse_args();print(json.dumps(execute(args.engine,args.output)))

if __name__=='__main__':main()
