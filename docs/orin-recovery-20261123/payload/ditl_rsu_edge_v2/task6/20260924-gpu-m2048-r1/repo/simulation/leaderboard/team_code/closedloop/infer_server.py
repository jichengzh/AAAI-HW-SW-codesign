# -*- coding: utf-8 -*-
"""
C-1: Roadside perception inference server (compute-isolation process).

Design: sim_closedloop_arch_v1.md §3.1 / §3.5.
Runs in a SEPARATE process pinned to a dedicated GPU (via CUDA_VISIBLE_DEVICES),
so CARLA rendering (other GPU) never pollutes the perception latency measurement
(user requirement: "推理放另一个GPU,互不抢GPU;推理GPU返回推理时延").

Contract (zmq REQ/REP, msgpack_numpy payloads):
  request : {'cmd': 'infer', 'frame_id': int, 'payload': <opaque numpy/dict>}
            {'cmd': 'ping'}  /  {'cmd': 'shutdown'}
  reply   : {'frame_id': int,
             'latency_ms': float,         # CUDA-Event forward-only (§3.5)
             'latency_source': 'measured_in_carla' | 'injected_from_E7',
             'result': <opaque>}          # detection payload, opaque to transport

IPC = zmq + msgpack_numpy (py3.7 fallback; shared_memory is 3.8+, unavailable).
Serialization (IPC round-trip) is timed separately from forward, keeping the
returned latency_ms a pure forward number (discipline: every number has a kind).

This file is intentionally model-agnostic at the transport layer. The actual
perception callable is injected via `infer_fn(frame_id, payload) -> result`,
and forward timing wraps that call with CUDA events. For the latency-injection
sweep (design §3.4), `inject_latency_ms` overrides the measured value with an
E7 anchor and tags latency_source='injected_from_E7' (model still runs to keep
driving valid; only the reported/aligned latency is the injected archive value).
"""

import time
import argparse

import zmq
import msgpack
import msgpack_numpy as mnp
mnp.patch()


def _cuda_event_timer():
    """Return (start_fn, stop_fn) using torch CUDA events; falls back to wallclock
    if torch/CUDA unavailable (so the server is testable without GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)

            def start():
                torch.cuda.synchronize(); s.record()

            def stop():
                e.record(); torch.cuda.synchronize()
                return s.elapsed_time(e)  # ms
            return start, stop
    except Exception:
        pass

    holder = {}

    def start_wc():
        holder['t'] = time.time()

    def stop_wc():
        return (time.time() - holder['t']) * 1000.0
    return start_wc, stop_wc


class InferServer:
    def __init__(self, infer_fn, port=5557, inject_latency_ms=None):
        """
        infer_fn(frame_id, payload) -> result_payload (must run the real model
                 forward so driving stays valid).
        inject_latency_ms: if set, report this latency (E7 archive) instead of
                 the measured one; latency_source becomes 'injected_from_E7'.
        """
        self.infer_fn = infer_fn
        self.port = port
        self.inject_latency_ms = inject_latency_ms
        self._start, self._stop = _cuda_event_timer()

    def serve_forever(self):
        # ROUTER pairs with the DEALER client and supports async multi-in-flight.
        # Each recv is [identity, msg]; reply must echo identity back.
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.ROUTER)
        sock.bind("tcp://127.0.0.1:%d" % self.port)
        print("[infer_server] listening on tcp://127.0.0.1:%d (inject=%s)"
              % (self.port, self.inject_latency_ms), flush=True)
        try:
            while True:
                ident, raw = sock.recv_multipart()
                req = msgpack.unpackb(raw, raw=False)
                cmd = req.get('cmd')
                if cmd == 'ping':
                    sock.send_multipart([ident, msgpack.packb({'ok': True}, use_bin_type=True)])
                    continue
                if cmd == 'shutdown':
                    sock.send_multipart([ident, msgpack.packb({'ok': True}, use_bin_type=True)])
                    break
                if cmd == 'infer':
                    fid = req['frame_id']
                    payload = req.get('payload')
                    self._start()
                    result = self.infer_fn(fid, payload)
                    measured_ms = self._stop()
                    if self.inject_latency_ms is not None:
                        lat = float(self.inject_latency_ms)
                        src = 'injected_from_E7'
                    else:
                        lat = float(measured_ms)
                        src = 'measured_in_carla'
                    reply = {'frame_id': fid, 'latency_ms': lat,
                             'latency_source': src,
                             'measured_forward_ms': float(measured_ms),
                             'result': result}
                    sock.send_multipart([ident, msgpack.packb(reply, use_bin_type=True)])
                    continue
                sock.send_multipart([ident, msgpack.packb({'error': 'unknown cmd %r' % cmd},
                                                          use_bin_type=True)])
        finally:
            sock.close(0)


class InferClient:
    """Agent-side (process B) client. Non-blocking submit per design C-2:
    submit() sends and returns immediately; poll() retrieves when ready.
    zmq REQ is strictly req/rep, so we use DEALER for async submit/poll."""

    def __init__(self, port=5557, timeout_ms=2000):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect("tcp://127.0.0.1:%d" % port)
        self.timeout_ms = timeout_ms

    def ping(self):
        self.sock.send(msgpack.packb({'cmd': 'ping'}, use_bin_type=True))
        if self.sock.poll(self.timeout_ms):
            self.sock.recv()
            return True
        return False

    def submit(self, frame_id, payload):
        """Non-blocking: fire the request, return immediately (design §3.2)."""
        self.sock.send(msgpack.packb(
            {'cmd': 'infer', 'frame_id': frame_id, 'payload': payload},
            use_bin_type=True))

    def poll(self, timeout_ms=0):
        """Return a reply dict if one is ready within timeout_ms, else None."""
        if self.sock.poll(timeout_ms):
            return msgpack.unpackb(self.sock.recv(), raw=False)
        return None

    def shutdown_server(self):
        self.sock.send(msgpack.packb({'cmd': 'shutdown'}, use_bin_type=True))
        self.sock.poll(self.timeout_ms)


def _demo_infer_fn(frame_id, payload):
    """Tiny GPU conv to exercise real CUDA-Event timing without the full model."""
    try:
        import torch
        x = torch.randn(1, 64, 128, 256, device='cuda')
        w = torch.randn(128, 64, 3, 3, device='cuda')
        y = torch.nn.functional.conv2d(x, w, padding=1)
        return {'frame_id': frame_id, 'out_sum': float(y.sum().item())}
    except Exception as ex:
        return {'frame_id': frame_id, 'cpu_fallback': True, 'err': str(ex)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5557)
    ap.add_argument('--inject-latency-ms', type=float, default=None)
    ap.add_argument('--demo', action='store_true',
                    help='run with a demo GPU conv infer_fn (no real model)')
    args = ap.parse_args()
    fn = _demo_infer_fn if args.demo else _demo_infer_fn
    InferServer(fn, port=args.port,
                inject_latency_ms=args.inject_latency_ms).serve_forever()
