"""Fixed batch-one TensorRT bridge using standard CUDA runtime APIs; no kernels."""
import ctypes
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def assert_no_torch(maps="/proc/self/maps"):
    if any(k == 'torch' or k.startswith('torch.') for k in sys.modules):
        raise RuntimeError('Torch must not be imported in SDK target')
    if any(token in Path(maps).read_text() for token in ('libtorch','libc10','/torch/')):
        raise RuntimeError('Torch native library loaded before SDK')

def library_identity(maps='/proc/self/maps'):
    text = Path(maps).read_text()
    if any(token in text for token in ('libtorch', 'libc10', '/torch/')):
        raise RuntimeError('Torch loaded in target maps')
    paths = {line.split()[-1] for line in text.splitlines()
             if len(line.split()) >= 6 and line.split()[-1].startswith('/')}
    libs = sorted(p for p in paths if any(x in Path(p).name
                  for x in ('libnvinfer', 'libcudart', 'libcuda.so')))
    if not all(any(token in p for p in libs) for token in ('libnvinfer', 'libcudart')):
        raise RuntimeError('Missing loaded SDK libraries')
    return {str(Path(p).resolve()): sha(p) for p in libs}

class Cuda:
    def __init__(self, lib):
        self.lib = lib
        specs = {'cudaSetDevice': [ctypes.c_int],
                 'cudaMalloc': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
                 'cudaFree': [ctypes.c_void_p],
                 'cudaStreamCreate': [ctypes.POINTER(ctypes.c_void_p)],
                 'cudaStreamDestroy': [ctypes.c_void_p],
                 'cudaStreamSynchronize': [ctypes.c_void_p],
                 'cudaMemcpyAsync': [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.c_int, ctypes.c_void_p]}
        self.functions = {}
        for name, args in specs.items():
            fn = getattr(lib, name); fn.argtypes = args; fn.restype = ctypes.c_int
            self.functions[name] = fn
        self.call('cudaSetDevice', 0)
    def call(self, name, *args):
        code = self.functions[name](*args)
        if code != 0:
            raise RuntimeError('%s CUDA error %s' % (name, code))
    def malloc(self, size):
        ptr = ctypes.c_void_p(); self.call('cudaMalloc', ctypes.byref(ptr), size)
        if not ptr.value: raise RuntimeError('cudaMalloc null')
        return ptr.value
    def stream(self):
        ptr = ctypes.c_void_p(); self.call('cudaStreamCreate', ctypes.byref(ptr))
        if not ptr.value: raise RuntimeError('cudaStreamCreate null')
        return ptr.value
    def copy(self, dst, src, size, kind, stream):
        self.call('cudaMemcpyAsync', dst, src, size, kind, stream)
    def sync(self, stream): self.call('cudaStreamSynchronize', stream)
    def free(self, ptr): self.call('cudaFree', ptr)
    def destroy(self, stream): self.call('cudaStreamDestroy', stream)

class Runtime:
    def __init__(self, engine_path, engine_sha, failures, *, cuda, engine_loader):
        assert_no_torch()
        if sha(engine_path) != engine_sha: raise ValueError('Engine SHA mismatch')
        self.cuda = cuda; self.failures = Path(failures); self.index = 0
        self.allocations = []; self.stream_handle = None; self.closed = False
        self.inflight = None; self.failed = False
        self.engine = engine_loader(Path(engine_path).read_bytes())
        try:
            self._bindings()
            self.context = self.engine.create_execution_context()
            if self.context is None: raise RuntimeError('Missing execution context')
            self.output = np.full((1,1000), np.nan, dtype=np.float32)
            self.allocations.append(cuda.malloc(3*224*224*4))
            self.allocations.append(cuda.malloc(1000*4))
            self.stream_handle = cuda.stream()
        except BaseException:
            self.close(); raise
    def _bindings(self):
        if self.engine is None or self.engine.num_bindings != 2 or self.engine.has_implicit_batch_dimension:
            raise ValueError('Expected two bindings')
        for i, (name, shape) in enumerate((('input',(1,3,224,224)),('logits',(1,1000)))):
            if (self.engine.get_binding_name(i) != name or
                self.engine.binding_is_input(i) != (i == 0) or
                tuple(self.engine.get_binding_shape(i)) != shape or
                str(self.engine.get_binding_dtype(i)).lower() not in ('float32','datatype.float') or
                str(self.engine.get_location(i)).upper() not in ('DEVICE','TENSORLOCATION.DEVICE') or
                self.engine.is_shape_binding(i) or self.engine.get_binding_vectorized_dim(i) != -1 or
                str(self.engine.get_binding_format(i)).upper() not in ("LINEAR","TENSORFORMAT.LINEAR") or
                self.engine.get_binding_components_per_element(i) != 1 or
                self.engine.get_binding_bytes_per_component(i) != 4):
                raise ValueError('Unexpected binding dtype/name/shape')
    def infer(self, x):
        if self.closed or self.failed: raise RuntimeError('Runtime closed or failed')
        if (x.dtype != np.float32 or x.shape != (1,3,224,224) or
            not x.flags.c_contiguous or not np.isfinite(x).all()):
            raise ValueError('Input dtype/shape/contiguous/finite')
        index = self.index; self.index += 1; stage = 'H2D'; self.inflight = x
        try:
            self.cuda.copy(self.allocations[0], x.ctypes.data, x.nbytes, 1, self.stream_handle)
            stage = 'execute'
            if not self.context.execute_async_v2(bindings=self.allocations, stream_handle=self.stream_handle):
                raise RuntimeError('TensorRT execute returned false')
            stage = 'D2H'
            self.cuda.copy(self.output.ctypes.data, self.allocations[1], self.output.nbytes, 2, self.stream_handle)
            stage = 'synchronize'; self.cuda.sync(self.stream_handle)
            self.inflight = None
            stage = 'finite'
            if not np.isfinite(self.output).all(): raise ValueError('Output nonfinite')
            return self.output.copy()
        except BaseException as error:
            self.failed = True
            out = self.failures / ('failure_%05d' % index); out.mkdir(parents=True, exist_ok=False)
            np.save(out/'input.npy', x, allow_pickle=False)
            # On failed sync/copy these bytes may be stale; never call them completed GPU output.
            np.save(out/'output.npy', self.output, allow_pickle=False)
            (out/'failure.json').write_text(json.dumps({'index':index,'stage':stage,'error':repr(error),
                'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                'output_transfer_completed':stage=='finite'},indent=2)+'\n')
            raise
    def close(self):
        if self.closed: return
        self.closed = True; errors = []
        if self.stream_handle is not None:
            try: self.cuda.sync(self.stream_handle)
            except BaseException as error: errors.append(str(error))
        for ptr in self.allocations:
            try: self.cuda.free(ptr)
            except BaseException as error: errors.append(str(error))
        if self.stream_handle is not None:
            try: self.cuda.destroy(self.stream_handle)
            except BaseException as error: errors.append(str(error))
        self.context = None
        self.engine = None
        self.sdk_runtime = None
        self.sdk_logger = None
        if errors: raise RuntimeError('CUDA cleanup: '+ '; '.join(errors))
        self.inflight = None
