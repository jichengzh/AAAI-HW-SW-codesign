"""One CPU ORT model stream; external supervisor controls exclusive scheduling."""
import argparse,fcntl,json,os,platform,sys
from contextlib import contextmanager
from pathlib import Path
from cpu_bootstrap import configure
ORT_ROOT='/home/jichengzhi/resnet_compare_s1_20261013/ort_cpu_20261026_r02/python'
configure(sys.path,sys.modules,platform.machine(),sys.version_info,ORT_ROOT)
from cpu_identity import validate,sha,ORT_PYBIND
from cpu_session import create_session
from cpu_backend import Runtime
from frozen_stream import serve
LOCK=Path('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock')

def maps_without_gpu(path='/proc/self/maps'):
    text=Path(path).read_text()
    if any(s in text for s in ('libcuda.so','libcudart','libnvinfer','libtorch','libc10')):raise ValueError('GPU or Torch library loaded')
    return text

def runtime_environment():
    if platform.machine()!='aarch64' or sys.version_info[:2]!=(3,12):raise ValueError('target architecture/Python')
    sys.path.insert(0,ORT_ROOT)
    import numpy as np,onnxruntime as ort
    from onnxruntime.capi import onnxruntime_pybind11_state as pybind
    if ort.__version__!='1.27.0' or np.__version__!='1.26.4' or sha(pybind.__file__)!=ORT_PYBIND:raise ValueError('ORT/NumPy/pybind identity')
    root=Path(ORT_ROOT).resolve()
    if any(root not in Path(module.__file__).resolve().parents for module in (np,ort,pybind)):raise ValueError('ORT/NumPy package path mismatch')
    maps_without_gpu()
    return ort,{'pybind_path':str(Path(pybind.__file__).resolve()),'python_version':platform.python_version(),'numpy_version':np.__version__}

def target(args,reader,writer):
    if args.count not in (1,512,50000):raise ValueError('unsupported count')
    token=validate(args.point,args);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ort,env=runtime_environment();session=create_session(args.onnx,ort);runtime=Runtime(session)
        before=maps_without_gpu();(out/'maps_before.txt').write_text(before)
        token={**token,**env,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               'scope':'FULL50000' if args.count==50000 else 'DIAGNOSTIC_ONLY','count':args.count}
        (out/'target_identity.json').write_text(json.dumps(token,indent=2)+'\n')
        rows=json.loads(Path(args.final).read_text())['samples'][:args.count]
        if len(rows)!=args.count:raise ValueError('incomplete rows')
        serve(reader,writer,rows,runtime.infer,token)
        runtime.check_provider();after=maps_without_gpu();(out/'maps_after.txt').write_text(after)
        (out/'completion.json').write_text(json.dumps({'status':'STREAM_COMPLETE','count':args.count,
            'provider':'CPUExecutionProvider','ort_pybind_sha256':ORT_PYBIND})+'\n')

@contextmanager
def protocol_output():
    sys.stdout.flush();saved=os.dup(1)
    try:
        with os.fdopen(os.dup(saved),'wb',buffering=0) as wire:
            os.dup2(2,1);yield wire
    finally:
        sys.stdout.flush();os.dup2(saved,1);os.close(saved)

def main():
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('dense','copt','geta'),required=True)
    for n in ('onnx','manifest','final','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--count',type=int,choices=(1,512,50000),required=True)
    a=p.parse_args()
    with protocol_output() as wire:target(a,sys.stdin.buffer,wire)
if __name__=='__main__':main()
