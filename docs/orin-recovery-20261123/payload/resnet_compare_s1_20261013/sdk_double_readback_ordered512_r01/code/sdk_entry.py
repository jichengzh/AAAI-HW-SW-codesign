"""CPU-prepared independent dense SDK target; root schedules all device runs."""
import argparse
import ctypes
import fcntl
import json
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path
from sdk_backend import Cuda, Runtime, assert_no_torch, library_identity, sha
from sdk_contract import ENGINE, validate
from frozen_stream import serve
from stage_logger import record
from d2h_readback import DoubleReadbackCuda
CUDA_PATH=Path('/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298')
CUDA_SHA='6ca38be1cc6fa0b8ddad0313eea2a4483be27d2e8474437f9f7b7a536b7197a2'
LOCK=Path('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock')

def validate_platform():
    if platform.machine()!='aarch64':raise ValueError('SDK target requires aarch64')
    if sha(CUDA_PATH)!=CUDA_SHA:raise ValueError('CUDART bytes changed')

def cuda_library():
    if sha(CUDA_PATH)!=CUDA_SHA:raise ValueError('CUDART bytes changed')
    return ctypes.CDLL(str(CUDA_PATH))

def verify_inspector(engine,trt,path):
    expected=json.loads(Path(path).read_text())['Layers']
    inspector=engine.create_engine_inspector()
    actual=[json.loads(inspector.get_layer_information(i,trt.LayerInformationFormat.JSON))
            for i in range(engine.num_layers)]
    if actual!=expected:raise ValueError('Actual inspector differs from fresh build')
    return actual

def create_runtime(a):
    assert_no_torch();validate_platform()
    if sha(a.engine)!=ENGINE:raise ValueError('Engine bytes changed')
    import tensorrt as trt
    if trt.__version__!='8.5.2.2':raise ValueError('TensorRT version differs')
    logger=trt.Logger(trt.Logger.WARNING)
    owner=record("trt_runtime",lambda:trt.Runtime(logger))
    engine=record("deserialize",lambda:owner.deserialize_cuda_engine(Path(a.engine).read_bytes()))
    if engine is None:raise RuntimeError('Engine deserialization failed')
    record("inspector",lambda:verify_inspector(engine,trt,a.layers))
    lib=record("load_cudart",cuda_library)
    cuda=record("cuda_set_device",lambda:Cuda(lib))
    runtime=Runtime(a.engine,ENGINE,Path(a.output)/'failures',cuda=cuda,engine_loader=lambda _:engine)
    runtime.cuda=DoubleReadbackCuda(runtime.cuda,a.output)
    # Keep TensorRT logger/runtime alive through all context and engine use.
    runtime.sdk_logger=logger
    runtime.sdk_runtime=owner
    return runtime,{'tensorrt_version':trt.__version__,'torch_imported':False}

def target(a,reader,writer):
    assert_no_torch();token=validate(a)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=None
        try:
            runtime,sdk=create_runtime(a)
            before=library_identity()
            token={**token,**sdk,'library_sha256':before,
                   'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
            (out/'target_identity.json').write_text(json.dumps(token,indent=2)+'\n')
            (out/'maps_before.txt').write_text(Path('/proc/self/maps').read_text())
            rows=json.loads(Path(a.final).read_text())['samples']
            serve(reader,writer,rows,runtime.infer,token)
            assert_no_torch()
            after=library_identity()
            (out/'maps_after.txt').write_text(Path('/proc/self/maps').read_text())
            if after!=before:raise RuntimeError('SDK libraries changed during evaluation')
        finally:
            if runtime is not None:runtime.close()
        (out/'completion.json').write_text(json.dumps({'status':'STREAM_AND_CLEANUP_COMPLETE',
            'samples':len(rows),'library_sha256':after})+'\n')

@contextmanager
def protocol_output():
    sys.stdout.flush();saved=os.dup(1)
    try:
        with os.fdopen(os.dup(saved),'wb',buffering=0) as wire:
            os.dup2(2,1)
            yield wire
    finally:
        sys.stdout.flush();os.dup2(saved,1);os.close(saved)

def main():
    p=argparse.ArgumentParser()
    for name in ('engine','manifest','onnx','final','layers','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    with protocol_output() as wire:target(a,sys.stdin.buffer,wire)
if __name__=='__main__':main()
