#!/usr/bin/env python3
"""Fixed-allocation CUDA contention with bounded duty windows and owned files."""
import argparse
import json
import math
from pathlib import Path
import signal
import time


def validate_options(duty_cycle, period_ms, matrix_size, duration_s):
    for value, low, high in ((duty_cycle, 0, 1), (period_ms, 1, 1000),
                             (duration_s, .01, 3600)):
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('invalid finite load timing')
    if type(matrix_size) is not int or not 256 <= matrix_size <= 8192 or matrix_size % 256:
        raise ValueError('matrix size must be a multiple of 256 in [256, 8192]')


def run_worker(duty_cycle, period_ms, matrix_size, duration_s, ready_file,
               metrics_jsonl, torch_module=None, clock=time.monotonic, sleep=time.sleep):
    validate_options(duty_cycle, period_ms, matrix_size, duration_s)
    # Reserve both files before loading CUDA; existing evidence is never truncated.
    with Path(metrics_jsonl).open('x') as metrics, Path(ready_file).open('x') as ready:
        if torch_module is None:
            import torch as torch_module
        if not torch_module.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch_module.set_num_threads(1)
        tensors = [torch_module.ones((matrix_size, matrix_size), device='cuda',
                                    dtype=torch_module.float32) for _ in range(3)]
        torch_module.mm(tensors[0], tensors[1], out=tensors[2])
        torch_module.cuda.synchronize()
        stopped = [False]
        def stop(signum, frame):
            stopped[0] = True
        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            ready.write(json.dumps({'ready': True, 'matrix_size': matrix_size}) + '\n')
            ready.flush()
            start = clock()
            while not stopped[0] and clock() - start < duration_s:
                begin, count = clock(), 0
                while not stopped[0] and clock() - begin < period_ms / 1000 * duty_cycle:
                    torch_module.mm(tensors[0], tensors[1], out=tensors[2])
                    torch_module.cuda.synchronize()
                    count += 1
                active = clock() - begin
                sleep(max(0, period_ms / 1000 - active))
                metrics.write(json.dumps({'elapsed_s': clock()-start, 'active_s': active,
                                          'matmuls': count, 'duty_cycle': duty_cycle})+'\n')
                metrics.flush()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in [('duty-cycle',float), ('period-ms',float), ('matrix-size',int), ('duration-s',float)]:
        parser.add_argument('--'+name, type=kind, required=True)
    parser.add_argument('--ready-file', required=True)
    parser.add_argument('--metrics-jsonl', required=True)
    args = parser.parse_args(argv)
    run_worker(args.duty_cycle, args.period_ms, args.matrix_size, args.duration_s,
               args.ready_file, args.metrics_jsonl)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
