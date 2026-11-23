"""Time the unchanged accuracy Runtime; CPU session.run is the forward boundary."""
import math
import time
import numpy as np

class _TimedSession:
    def __init__(self,session):self.session=session;self.calls=0;self.forward=None
    def get_providers(self):return self.session.get_providers()
    def run(self,names,feeds):
        if self.calls:raise ValueError('repeated session execution')
        if names!=['logits'] or set(feeds)!={'input'}:raise ValueError('session contract changed')
        start=time.perf_counter();value=self.session.run(names,feeds);end=time.perf_counter()
        self.forward=(end-start)*1000;self.calls+=1
        return value

class TimedInference:
    def __init__(self,runtime):self.runtime=runtime
    def infer(self,x):
        if (not isinstance(x,np.ndarray) or x.shape!=(1,3,224,224) or x.dtype!=np.float32 or
            not x.flags.c_contiguous or not np.isfinite(x).all()):raise ValueError('invalid input')
        original=self.runtime.session;proxy=_TimedSession(original)
        start_mono=time.monotonic();start=time.perf_counter()
        self.runtime.session=proxy
        try:
            y=self.runtime.infer(x.copy()).copy()
        finally:self.runtime.session=original
        end=time.perf_counter();end_mono=time.monotonic()
        if not isinstance(y,np.ndarray) or y.shape!=(1,1000) or y.dtype!=np.float32 or not np.isfinite(y).all():
            raise ValueError('invalid output')
        forward=proxy.forward;service=(end-start)*1000
        if proxy.calls!=1 or forward is None or not all(math.isfinite(v) and v>0 for v in (forward,service)) or service<forward or end_mono<=start_mono:
            raise ValueError('invalid completed timing')
        return y,forward,service,start_mono,end_mono

def infer(session,x):
    """CPU-only test adapter; formal target uses TimedInference(frozen Runtime)."""
    class Runtime:
        def __init__(self,session):self.session=session
        def infer(self,value):return self.session.run(['logits'],{'input':value})[0]
    return TimedInference(Runtime(session)).infer(x)
