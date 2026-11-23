"""Dependency-free file-filtered line tracing, not used for reported timings.

Standard trace caches ignores by basename ('runtime') and misses our runtime
after importing a site-packages runtime.py. This collector uses exact suffixes.
"""
import atexit
import json
import runpy
import sys
from pathlib import Path

destination, mode, script = sys.argv[1:4]
counts = {}
def tracer(frame, event, argument):
    filename = frame.f_code.co_filename
    if filename.endswith(('/resnet_compare/runtime.py', '/resnet_compare/trt_runtime.py')):
        if event == 'line':
            key = Path(filename).name
            counts.setdefault(key, set()).add(frame.f_lineno)
        return tracer
    return None
def finish():
    sys.settrace(None)
    Path(destination).write_text(json.dumps({key: sorted(value) for key, value in counts.items()}, indent=2))
atexit.register(finish)
sys.settrace(tracer)
if mode == 'pytest':
    import pytest
    raise SystemExit(pytest.main([script, '-q']))
runpy.run_path(script, run_name='__main__')
