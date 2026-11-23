"""One deterministic FP32 cuBLAS GEMM against an exact CPU reference."""
import math

def make_matrices(n):
    return ([float((r+2*c)%5-2) for c in range(n) for r in range(n)],
            [float((2*r+c)%7-3) for c in range(n) for r in range(n)])

def reference_gemm(n,a,b):
    if n<=0 or len(a)!=n*n or len(b)!=n*n:raise ValueError('Invalid matrix shape')
    return [sum(a[k*n+r]*b[c*n+k] for k in range(n)) for c in range(n) for r in range(n)]

def check_values(actual,expected):
    if len(actual)!=len(expected) or not all(math.isfinite(x) for x in actual):raise ValueError('Invalid output')
    errors=[abs(float(a)-float(b)) for a,b in zip(actual,expected)]
    if any(e!=0.0 for e in errors):raise ValueError('Output differs from exact reference')
    return {'count':len(actual),'max_abs_diff':max(errors,default=0.0)}

def check_code(name,code):
    if code!=0:raise RuntimeError(name+' returned '+str(code))

def run_once(api,n):
    a,b=make_matrices(n)
    expected=reference_gemm(n,a,b)
    actual=api.run(n,a,b)
    return check_values(actual,expected)

class CublasAPI:
    def __init__(self,cuda,blas,emit):
        import ctypes
        self.cuda,self.blas,self.emit=cuda,blas,emit
        i=ctypes.c_int; p=ctypes.c_void_p; size=ctypes.c_size_t
        cuda.cudaSetDevice.argtypes=[i];cuda.cudaSetDevice.restype=i
        cuda.cudaMalloc.argtypes=[ctypes.POINTER(p),size];cuda.cudaMalloc.restype=i
        cuda.cudaMemcpy.argtypes=[p,p,size,i];cuda.cudaMemcpy.restype=i
        cuda.cudaFree.argtypes=[p];cuda.cudaFree.restype=i
        cuda.cudaDeviceSynchronize.argtypes=[];cuda.cudaDeviceSynchronize.restype=i
        blas.cublasCreate_v2.argtypes=[ctypes.POINTER(p)];blas.cublasCreate_v2.restype=i
        blas.cublasDestroy_v2.argtypes=[p];blas.cublasDestroy_v2.restype=i
        blas.cublasSgemm_v2.argtypes=[p,i,i,i,i,i,ctypes.POINTER(ctypes.c_float),p,i,p,i,ctypes.POINTER(ctypes.c_float),p,i];blas.cublasSgemm_v2.restype=i
    def call(self,stage,fn,*args):
        self.emit(stage,'begin');check_code(stage,fn(*args));self.emit(stage,'ok')
    def run(self,n,a,b):
        import ctypes,sys
        count=n*n;size=count*ctypes.sizeof(ctypes.c_float)
        hosta=(ctypes.c_float*count)(*a);hostb=(ctypes.c_float*count)(*b);hostc=(ctypes.c_float*count)()
        buffers=[];handle=ctypes.c_void_p(); created=False
        try:
            self.call('set_device',self.cuda.cudaSetDevice,0)
            for name in ('a','b','c'):
                pointer=ctypes.c_void_p()
                self.call('malloc_'+name,self.cuda.cudaMalloc,ctypes.byref(pointer),size)
                if not pointer.value:raise RuntimeError('cudaMalloc returned null')
                buffers.append(pointer)
            self.call('cudaMemcpyH2D_a',self.cuda.cudaMemcpy,buffers[0],ctypes.cast(hosta,ctypes.c_void_p),size,1)
            self.call('cudaMemcpyH2D_b',self.cuda.cudaMemcpy,buffers[1],ctypes.cast(hostb,ctypes.c_void_p),size,1)
            self.call('create_handle',self.blas.cublasCreate_v2,ctypes.byref(handle));created=True
            if not handle.value:raise RuntimeError('cublasCreate returned null')
            alpha=ctypes.c_float(1);beta=ctypes.c_float(0)
            self.call('sgemm',self.blas.cublasSgemm_v2,handle,0,0,n,n,n,ctypes.byref(alpha),buffers[0],n,buffers[1],n,ctypes.byref(beta),buffers[2],n)
            self.call('synchronize',self.cuda.cudaDeviceSynchronize)
            self.call('cudaMemcpyD2H',self.cuda.cudaMemcpy,ctypes.cast(hostc,ctypes.c_void_p),buffers[2],size,2)
            return list(hostc)
        finally:
            active=sys.exc_info()[0] is not None;errors=[]
            if created:
                rc=self.blas.cublasDestroy_v2(handle)
                if rc:errors.append('cublasDestroy '+str(rc))
            for pointer in reversed(buffers):
                rc=self.cuda.cudaFree(pointer)
                if rc:errors.append('cudaFree '+str(rc))
            if errors:
                self.emit('cleanup_error',';'.join(errors))
                if not active:raise RuntimeError('Cleanup failed: '+','.join(errors))
