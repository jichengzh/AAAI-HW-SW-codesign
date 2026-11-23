"""Observe one fixed official build; invoke only under external timeout 300s/10s.

The child PID is the native binary itself, not a timeout/flock intermediate.
No Torch/TensorRT Python imports. No inference, search, retries or kernel changes.
"""
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/jichengzhi/resnet_compare_s1_20261013')
OUT=ROOT/'trt_official_rebuild_diagnostic_r01'
LOCK=ROOT/'.orin-runtime.lock'
SOURCES={
 'binary':{'path':'/usr/src/tensorrt/bin/trtexec','sha256':'f511762a3c57a08918923586e902424b1abe40320717addafc9b6923c1ceebb4'},
 'onnx':{'path':str(ROOT/'s7_trt_target_accuracy_r01/dense/source/candidate.onnx'),'sha256':'232f666524f07b9d3971381462365774222f0488f6912e144f73c50d36d2fc21'},
 'old_engine':{'path':str(ROOT/'s7_20261026_latency_dense_FP16_r01/73be4628be4e86c24419b345f778054f4d145ffb5495507834e93fd4fc5c2f53/candidate.engine'),'sha256':'68c67e61977d3ed20ccead95fae5da67677f791ab80c07444585577f46ea5c48'}}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def verify_sources():
 actual={name:{'path':row['path'],'sha256':sha(row['path'])} for name,row in SOURCES.items()}
 if actual!=SOURCES:raise ValueError('Immutable source identity changed')
 return actual

def command():
 return [SOURCES['binary']['path'],'--onnx='+SOURCES['onnx']['path'],'--saveEngine='+str(OUT/'candidate_fp16.engine'),
         '--fp16','--noTF32','--memPoolSize=workspace:1024','--profilingVerbosity=detailed',
         '--minTiming=1','--avgTiming=1','--buildOnly','--exportLayerInfo='+str(OUT/'layers.json'),'--verbose']

def spawn(argv,log,lockfd):
 return subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lockfd,))

def read_process_maps(pid):
 try:return Path('/proc') .joinpath(str(pid),'maps').read_text()
 except (FileNotFoundError,PermissionError):return None

def read_process_exe(pid):
 try:return os.readlink('/proc/%s/exe'%pid)
 except (FileNotFoundError,PermissionError):return None

def observe(process):
 snapshots=[];next_capture=0.0
 while process.poll() is None:
  now=time.monotonic()
  if now>=next_capture:
   maps=read_process_maps(process.pid);exe=read_process_exe(process.pid)
   if maps:
    name='maps_%05d.txt'%len(snapshots);(OUT/name).write_text(maps)
    row={'pid':process.pid,'epoch':time.time(),'monotonic':now,'exe':exe,'file':name,'sha256':sha(OUT/name),
         'torch_mappings':[line for line in maps.splitlines() if any(token in line.lower() for token in ('/torch/','libtorch','libc10','torch_c'))],
         'nvinfer_seen':'libnvinfer' in maps}
    snapshots.append(row)
    with (OUT/'maps.jsonl').open('a') as file:file.write(json.dumps(row)+'\n');file.flush();os.fsync(file.fileno())
   next_capture=now+2.0
  time.sleep(.05)
 return snapshots

def execute():
 OUT.mkdir(exist_ok=False);process=None
 try:
  with LOCK.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   before=verify_sources();argv=command()
   write(OUT/'launch.json',{'observer_pid':os.getpid(),'utc_epoch':time.time(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'argv':argv,'sources':before,'observer_sha256':sha(__file__),'external_timeout_required_seconds':300,'external_kill_after_seconds':10,
        'build_only':True,'torch_python_imported':False,'native_same_process_group':True,'lock_inherited_by_native':True})
   with (OUT/'native.log').open('x') as log:
    process=spawn(argv,log,lock.fileno());write(OUT/'native_pid.json',{'pid':process.pid})
    snapshots=observe(process);code=process.wait()
   write(OUT/'native_exit.json',{'native_returncode':code,'pid':process.pid,'epoch':time.time()})
   after=verify_sources()
   if code!=0:raise ValueError('Native build exit '+str(code))
   if not snapshots or not any(row['nvinfer_seen'] for row in snapshots):raise ValueError('Native SDK maps not captured')
   if any(row['torch_mappings'] for row in snapshots):raise ValueError('Torch mapping in official SDK build')
   expected=str(Path(SOURCES['binary']['path']).resolve())
   if not any(row['exe']==expected for row in snapshots):raise ValueError('Direct native executable identity not observed')
   engine=OUT/'candidate_fp16.engine';layers=OUT/'layers.json'
   if engine.stat().st_size==0 or not isinstance(json.loads(layers.read_text()),dict):raise ValueError('Incomplete engine/inspector output')
   result={'status':'BUILD_COMPLETE_PENDING_INSPECTOR_REVIEW','native_returncode':code,'native_pid':process.pid,
           'engine_sha256':sha(engine),'layers_sha256':sha(layers),'sources_before':before,'sources_after':after,
           'maps_snapshots':len(snapshots),'maps_native_sdk_seen':True,'inference_qualified':False,'full50000_qualified':False}
   write(OUT/'result.json',result);return result
 except BaseException as error:
  cleanup=None
  if process is not None and process.poll() is None:
   process.terminate();cleanup=process.wait()
  write(OUT/'failure.json',{'status':'BUILD_DIAGNOSTIC_FAILED','reason':str(error),'cleanup_returncode':cleanup})
  raise

if __name__=='__main__':print(json.dumps(execute()))
