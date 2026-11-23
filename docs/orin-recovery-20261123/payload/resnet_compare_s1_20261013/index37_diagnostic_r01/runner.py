"""One index37 input on unchanged instrumented SDK; no automatic retry."""
import argparse, fcntl, hashlib, json, os, platform, sys
from pathlib import Path
import numpy as np

INPUT_SHA='41fe694f5bbcd30e26aed6eec1ebd9bb8e2cde55c20ed10fb489434711df3bbc'
ENGINE_SHA='2a7110a8ebc458170d77deba53270e25657f04c84b8b8f21400021114c0be7df'
CODE_SHA={
 'sdk_backend.py':'63e7bcf178f167b20d5f9539d0afdb2650b98a1b0d93c4cf855f2c73a0c403f8', 'sdk_entry.py':'b994002ca6b8eb1507de658959e00abd2297c88cc37e9ac1317d37029dade81c', 'sdk_contract.py':'246568da4f81a263e5b51cc0e6fe6daba2fe26010f905b5d048a93f5a5ae382f', 'frozen_stream.py':'6214df738caa17bb1d51e9b08d7c62658c8c5a89ac21a9de4da16a4684ed9b7d', 'stage_logger.py':'b39f544a3d64aea7af7cc272476616a803a16afdd035ba9bd5c536fab8759941'}
LIBS={
 '/usr/lib/aarch64-linux-gnu/libnvinfer.so.8.5.2':'4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca',
 '/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.5.2':'5a30c8e46ba47c3a93eb825ad0a27cbe969fb6f174ea963e4d22dfe33630097a',
 '/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1.1':'151ad1343173e919304e08d4c6801508303111f2b1c7b3737f6b5134590808f4',
 '/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298':'6ca38be1cc6fa0b8ddad0313eea2a4483be27d2e8474437f9f7b7a536b7197a2'}
LOCK='/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_input(x,expected):
 if x.shape!=(1,3,224,224) or x.dtype!=np.float32 or not x.flags.c_contiguous or not np.isfinite(x).all():raise ValueError('Input contract')
 raw=x.tobytes()
 if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Input raw SHA')
 return raw

def check_libraries(libs):
 if libs!=LIBS:raise ValueError('SDK library mismatch')

def execute(a):
 if platform.machine()!='aarch64':raise ValueError('aarch64 required')
 if sha(a.engine)!=ENGINE_SHA:raise ValueError('Engine SHA mismatch')
 for name,expected in CODE_SHA.items():
  if sha(Path(a.code)/name)!=expected:raise ValueError('SDK code SHA mismatch: '+name)
 sys.path.insert(0,str(Path(a.code).resolve()))
 import sdk_entry, sdk_backend, stage_logger
 x=np.load(a.input,allow_pickle=False);validate_input(x,INPUT_SHA)
 out=Path(a.output);out.mkdir(exist_ok=False)
 with Path(LOCK).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  runtime=None
  try:
   with stage_logger.bind(out/'init_stages.jsonl'):
    runtime,identity=stage_logger.record('create_runtime',lambda:sdk_entry.create_runtime(a))
    libs=stage_logger.record('library_identity',sdk_backend.library_identity)
    check_libraries(libs)
   (out/'identity.json').write_text(json.dumps({'engine_sha':ENGINE_SHA,'input_sha':INPUT_SHA,'code_sha':CODE_SHA,'library_sha256':libs,'sdk':identity,'pid':os.getpid()},indent=2)+'\n')
   y=runtime.infer(x)
   with (out/'output.npy').open('xb') as f:np.save(f,y,allow_pickle=False)
   (out/'result.json').write_text(json.dumps({'status':'ONE_REQUEST_PENDING_REVIEW','output_sha':sha(out/'output.npy'),'finite':int(np.isfinite(y).sum()),'top1':int(y.argmax())},indent=2)+'\n')
  finally:
   if runtime is not None:runtime.close()

def main():
 p=argparse.ArgumentParser()
 for n in ('engine','layers','input','code','output'):p.add_argument('--'+n,required=True)
 execute(p.parse_args())
if __name__=='__main__':main()
