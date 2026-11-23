"""Strict one-sample CPU target command and completion contract."""
from pathlib import Path

def check(ok,msg):
    if not ok:raise ValueError(msg)

def command(job):
    root=Path(job['root']);args=job['argv']
    check(job['count']==1 and args[:2]==[job['python'],str(root/'code/cpu_target.py')],'single CPU entry')
    for flag,value in (('--point',job['point']),('--count','1'),('--output',str(root/'run/target'))):
        check(args.count(flag)==1 and args[args.index(flag)+1]==value,'fixed '+flag)
    check(not any(x in args for x in ('--build','--saveEngine')),'build forbidden')
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s','120s']+args

def verify(job,identity,completion,maps_before,maps_after):
    for k in ('point','model_sha256','boot_id','code_sha256','ort_pybind_sha256'):
        check(identity[k]==job[k],k+' mismatch')
    check(identity['execution_provider']=='CPUExecutionProvider' and identity['execution_precision']=='FP32' and identity['count']==1 and identity['scope']=='DIAGNOSTIC_ONLY','CPU mode')
    check(completion=={'status':'STREAM_COMPLETE','count':1,'provider':'CPUExecutionProvider','ort_pybind_sha256':job['ort_pybind_sha256']},'completion')
    check(all(s and not any(x in s for x in ('libcuda.so','libcudart','libnvinfer','libtorch','libc10')) for s in (maps_before,maps_after)),'GPU/Torch maps')
