"""Staged TensorRT initialization probe. It never submits inference."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

STEPS = ('validate_files', 'import_tensorrt', 'create_trt_runtime', 'deserialize_engine',
         'verify_inspector', 'load_cudart', 'cuda_set_device', 'validate_bindings',
         'create_context', 'malloc_input', 'malloc_output', 'create_stream',
         'library_identity', 'cleanup')
ENGINE_SHA = '2a7110a8ebc458170d77deba53270e25657f04c84b8b8f21400021114c0be7df'
BACKEND_SHA = '753037bed518e264decd250ee6246791f71a4524f49ebf06f203907300eb5b54'
ENTRY_SHA = 'b9fea2337b242098bc51572c5f5bcafd2b9d747e88fa467f6b19deeeeceb9307'
CONTRACT_SHA = 'a2553787058e2752b6c5e5c420e44315d60f686e3810200c6cd889613166a18e'
STREAM_SHA = '6214df738caa17bb1d51e9b08d7c62658c8c5a89ac21a9de4da16a4684ed9b7d'
LIBS = {
    '/usr/lib/aarch64-linux-gnu/libnvinfer.so.8.5.2': '4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca',
    '/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.5.2': '5a30c8e46ba47c3a93eb825ad0a27cbe969fb6f174ea963e4d22dfe33630097a',
    '/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1.1': '151ad1343173e919304e08d4c6801508303111f2b1c7b3737f6b5134590808f4',
    '/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298': '6ca38be1cc6fa0b8ddad0313eea2a4483be27d2e8474437f9f7b7a536b7197a2'}
LAYERS_SHA = '6f94d9216855acf2d9e58a2adae7e7ae51e667a0bfc5a4c4d80a8d894dda0f3d'
LOCK = '/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock'


def emit(stream, stage, event, error=None, value=None):
    row = {'stage': stage, 'event': event, 'time_ns': time.time_ns(), 'pid': os.getpid()}
    if value is not None: row['value'] = value
    if error is not None:
        row['error'] = repr(error)
    stream.write(json.dumps(row, sort_keys=True) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def run_steps(call, path):
    with Path(path).open('x') as stream:
        for stage in STEPS:
            emit(stream, stage, 'begin')
            try:
                value = call(stage)
            except BaseException as exc:
                emit(stream, stage, 'error', exc)
                raise
            emit(stream, stage, 'ok', value=value)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


class Probe:
    def __init__(self, engine, layers, code):
        self.engine_path, self.layers, self.code = Path(engine), Path(layers), Path(code)
        self.mem = []
    def call(self, name):
        return getattr(self, name)()
    def validate_files(self):
        if platform.machine() != 'aarch64': raise ValueError('aarch64 required')
        for p, h in ((self.engine_path, ENGINE_SHA), (self.layers, LAYERS_SHA),
                     (self.code/'sdk_backend.py', BACKEND_SHA), (self.code/'sdk_entry.py', ENTRY_SHA),
                     (self.code/'sdk_contract.py', CONTRACT_SHA), (self.code/'frozen_stream.py', STREAM_SHA)):
            if digest(p) != h: raise ValueError('hash mismatch: '+str(p))
        sys.path.insert(0, str(self.code.resolve()))
        import sdk_backend, sdk_entry
        self.backend, self.entry = sdk_backend, sdk_entry
        sdk_backend.assert_no_torch(); sdk_entry.validate_platform()
    def import_tensorrt(self):
        import tensorrt as trt
        if trt.__version__ != '8.5.2.2': raise ValueError('TensorRT version mismatch')
        self.trt = trt
    def create_trt_runtime(self):
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.owner = self.trt.Runtime(self.logger)
    def deserialize_engine(self):
        self.engine = self.owner.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None: raise RuntimeError('deserialize failed')
    def verify_inspector(self): self.entry.verify_inspector(self.engine, self.trt, self.layers)
    def load_cudart(self): self.lib = self.entry.cuda_library()
    def cuda_set_device(self): self.cuda = self.backend.Cuda(self.lib)
    def validate_bindings(self):
        runtime = self.backend.Runtime.__new__(self.backend.Runtime)
        runtime.engine = self.engine
        runtime._bindings()
    def create_context(self):
        self.context = self.engine.create_execution_context()
        if self.context is None: raise RuntimeError('missing execution context')
    def malloc_input(self): self.mem.append(self.cuda.malloc(3*224*224*4))
    def malloc_output(self): self.mem.append(self.cuda.malloc(1000*4))
    def create_stream(self): self.stream = self.cuda.stream()
    def library_identity(self):
        libs = self.backend.library_identity()
        if libs != LIBS: raise ValueError('SDK library identity mismatch')
        return libs
    def cleanup(self):
        self.cuda.sync(self.stream)
        for p in self.mem: self.cuda.free(p)
        self.cuda.destroy(self.stream)
        self.context = None; self.engine = None; self.owner = None; self.logger = None


def main():
    p = argparse.ArgumentParser()
    for n in ('engine', 'layers', 'code', 'events'): p.add_argument('--'+n, required=True)
    a = p.parse_args()
    with open(LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_steps(Probe(a.engine, a.layers, a.code).call, a.events)

if __name__ == '__main__': main()
