"""Target-side three-model SDK stream. Caller schedules the device exclusively."""
import argparse
import ctypes
import fcntl
import json
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path

from sdk_backend import Cuda, Runtime, assert_no_torch, library_identity
from frozen_stream import serve
from stage_logger import bind, record
from identity import (LIBRARIES, expected_layers, point_pins, sha, validate,
                      verify_inspector, verify_libraries)

LOCK = Path('/home/jichengzhi/resnet_compare_s1_20261013/.orin-runtime.lock')
CUDA_PATH = '/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298'


def create_runtime(args):
    if platform.machine() != 'aarch64':
        raise ValueError('SDK target requires aarch64')
    pins = point_pins(args.point)
    if sha(args.engine) != pins['engine']:
        raise ValueError('Engine SHA mismatch')
    if sha(CUDA_PATH) != LIBRARIES[CUDA_PATH]:
        raise ValueError('CUDART SHA mismatch')
    import tensorrt as trt
    if trt.__version__ != '8.5.2.2':
        raise ValueError('TensorRT version differs')
    logger = trt.Logger(trt.Logger.WARNING)
    owner = record('trt_runtime', lambda: trt.Runtime(logger))
    engine = record('deserialize', lambda: owner.deserialize_cuda_engine(Path(args.engine).read_bytes()))
    if engine is None:
        raise RuntimeError('Engine deserialization failed')
    def inspect():
        inspector = engine.create_engine_inspector()
        actual = [json.loads(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
                  for i in range(engine.num_layers)]
        verify_inspector(expected_layers(args.point, args.layers), actual)
    record('inspector', inspect)
    lib = record('load_cudart', lambda: ctypes.CDLL(CUDA_PATH))
    cuda = record('cuda_set_device', lambda: Cuda(lib))
    runtime = Runtime(args.engine, pins['engine'], Path(args.output)/'failures',
                      cuda=cuda, engine_loader=lambda _: engine)
    runtime.sdk_logger, runtime.sdk_runtime = logger, owner
    return runtime


def target(args, reader, writer):
    assert_no_torch()
    token = validate(args.point, args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = None
        try:
            with bind(out/'init_stages.jsonl'):
                runtime = record('create_runtime', lambda: create_runtime(args))
                before = record('library_identity', library_identity)
            verify_libraries(before)
            token = {**token, 'library_sha256': before,
                     'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                     'tensorrt_version': '8.5.2.2', 'torch_imported': False}
            (out/'target_identity.json').write_text(json.dumps(token, indent=2)+'\n')
            (out/'maps_before.txt').write_text(Path('/proc/self/maps').read_text())
            rows = json.loads(Path(args.final).read_text())['samples']
            serve(reader, writer, rows, runtime.infer, token)
            assert_no_torch()
            after = library_identity()
            verify_libraries(after)
            (out/'maps_after.txt').write_text(Path('/proc/self/maps').read_text())
            if before != after:
                raise RuntimeError('SDK libraries changed during evaluation')
        finally:
            if runtime is not None:
                runtime.close()
        (out/'completion.json').write_text(json.dumps({'status': 'STREAM_AND_CLEANUP_COMPLETE',
            'samples': len(rows), 'library_sha256': after})+'\n')


@contextmanager
def protocol_output():
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with os.fdopen(os.dup(saved), 'wb', buffering=0) as wire:
            os.dup2(2, 1)
            yield wire
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', choices=('geta',), required=True)
    for name in ('engine', 'manifest', 'onnx', 'final', 'layers', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    with protocol_output() as wire:
        target(args, sys.stdin.buffer, wire)


if __name__ == '__main__':
    main()
