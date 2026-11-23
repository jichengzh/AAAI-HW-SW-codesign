"""Synchronous failure-persistent initialization events; no inference logging."""
import json, os, time
from contextlib import contextmanager

_writer = None

@contextmanager
def bind(path):
    global _writer
    if _writer is not None: raise RuntimeError('stage logger already active')
    with open(path, 'x') as writer:
        _writer = writer
        try: yield
        finally: _writer = None

def emit(stage,event,error=None):
    row={'stage':stage,'event':event,'time_ns':time.time_ns(),'pid':os.getpid()}
    if error is not None: row['error']=repr(error)
    _writer.write(json.dumps(row,sort_keys=True)+'\n')
    _writer.flush();os.fsync(_writer.fileno())

def record(stage, fn):
    if _writer is None: return fn()
    emit(stage,'begin')
    try: result=fn()
    except BaseException as e:
        emit(stage,'error',e)
        raise
    emit(stage,'ok')
    return result
