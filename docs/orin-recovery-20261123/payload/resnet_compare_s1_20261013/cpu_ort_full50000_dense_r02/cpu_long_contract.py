"""Dense CPU ORT full 50000 target with six-hour hard limit."""
from pathlib import Path

def check(ok,msg):
    if not ok:raise ValueError(msg)

def command(job):
    root=Path(job['root']);args=job['argv'];count=job['count']
    check(count==50000 and job['point']=='dense' and args[:2]==[job['python'],str(root/'code/cpu_target.py')],'CPU flow entry/count')
    for flag,value in (('--point',job['point']),('--count',str(count)),('--output',str(root/'run/target'))):
        check(args.count(flag)==1 and args[args.index(flag)+1]==value,'fixed '+flag)
    check(not any(x in args for x in ('--build','--saveEngine')),'build forbidden')
    seconds=21600
    return ['/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(seconds)+'s']+args

def verify(job,identity,completion,maps_before,maps_after):
    for k in ('point','model_sha256','boot_id','code_sha256','ort_pybind_sha256'):
        check(identity[k]==job[k],k+' mismatch')
    count=job['count'];scope='FULL50000' if count==50000 else 'DIAGNOSTIC_ONLY'
    check(identity['execution_provider']=='CPUExecutionProvider' and identity['execution_precision']=='FP32' and identity['count']==count and identity['scope']==scope,'CPU mode')
    check(completion=={'status':'STREAM_COMPLETE','count':count,'provider':'CPUExecutionProvider','ort_pybind_sha256':job['ort_pybind_sha256']},'completion')
    check(all(s and not any(x in s for x in ('libcuda.so','libcudart','libnvinfer','libtorch','libc10')) for s in (maps_before,maps_after)),'GPU/Torch maps')
