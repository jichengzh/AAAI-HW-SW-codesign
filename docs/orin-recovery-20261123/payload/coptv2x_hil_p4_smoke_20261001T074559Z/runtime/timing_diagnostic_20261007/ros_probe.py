#!/usr/bin/env python3
"""Temporary observation-only launcher for the existing proxy/publisher.

HIL_TIMING_ROLE=proxy|publisher HIL_TIMING_LOG=/path/events.jsonl
HIL_TIMING_ORIGINAL=/path/to/original.py python ros_probe.py <original arguments>

Does not alter FIFO, model inputs, payloads, ROS QoS, or ACK handling. Existing
publisher sideband measures the exact message construction and .publish() call.
All measured timestamps use host CLOCK_MONOTONIC_RAW, not container wall time.
"""

import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time


def raw_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class TimingLog:
    def __init__(self, path, role):
        self.role = role
        self.lock = threading.Lock()
        self.file = open(path, 'a', buffering=1)
        self.warned = False
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def emit(self, event, **fields):
        """An observation failure must not replace a runtime failure/result."""
        try:
            record = dict(fields, event=event, role=self.role,
                clock='CLOCK_MONOTONIC_RAW', mono_raw_ns=raw_ns(),
                boot_id=self.boot_id, pid=os.getpid(), thread=threading.get_ident())
            line = json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n'
            with self.lock:
                self.file.write(line)
        except Exception as error:
            if not self.warned:
                self.warned = True
                print('ROS timing observation unavailable: %s' % type(error).__name__, file=sys.stderr)

    def close(self):
        self.file.close()


class ObservedSocket:
    """Delegate the exact socket; never copy or parse a large result payload."""
    def __init__(self, connection, sink, owner):
        self.connection = connection
        self.sink = sink
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def sendall(self, data, *args, **kwargs):
        awaiting = self.owner.awaiting_ack
        request_id = awaiting[0] if awaiting is not None else None
        self.sink.emit('host_sendall_start', request_id=request_id, framed_bytes=len(data))
        start = raw_ns()
        error_type = None
        try:
            return self.connection.sendall(data, *args, **kwargs)
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            end = raw_ns()
            self.sink.emit('host_sendall', request_id=request_id, framed_bytes=len(data),
                start_ns=start, end_ns=end, duration_ns=end - start, error_type=error_type)


class SocketFacade:
    """Wrap only connections created inside the proxy's host-connect method."""
    def __init__(self, socket_module, context, sink):
        self.original = socket_module
        self.context = context
        self.sink = sink

    def __getattr__(self, name):
        return getattr(self.original, name)

    def create_connection(self, *args, **kwargs):
        connection = self.original.create_connection(*args, **kwargs)
        owner = getattr(self.context, 'connecting_owner', None)
        return ObservedSocket(connection, self.sink, owner) if owner is not None else connection


def _trigger_wrapper(original, sink, event):
    @functools.wraps(original)
    def wrapped(self, stage, trigger, *args, **kwargs):
        sink.emit(event, stage=stage, **trigger)
        return original(self, stage, trigger, *args, **kwargs)
    return wrapped


def install_proxy(module, sink):
    cls = module.CoptV2xHilRosProxy
    context = threading.local()
    module.socket = SocketFacade(module.socket, context, sink)
    connect = cls._connect_host_proxy

    @functools.wraps(connect)
    def connect_host(self, *args, **kwargs):
        context.connecting_owner = self
        try:
            return connect(self, *args, **kwargs)
        finally:
            del context.connecting_owner
    cls._connect_host_proxy = connect_host

    for method, event in (('_schedule_trigger', 'trigger_ingress'),
            ('_record_trigger_received', 'trigger_received'),
            ('_record_trigger_dispatch', 'trigger_dispatch')):
        setattr(cls, method, _trigger_wrapper(getattr(cls, method), sink, event))

    # Capture the ID at the original parser boundary, avoiding a second bytes()
    # conversion or queue inspection inside the ROS result callback.
    parse_id = module.request_id_from_result_wire

    @functools.wraps(parse_id)
    def capture_id(wire, *args, **kwargs):
        request_id = parse_id(wire, *args, **kwargs)
        state = getattr(context, 'result', None)
        if state is not None:
            state['request_id'] = request_id
            state['wire_bytes'] = len(wire)
            state['_wire'] = wire  # Reference only; no additional payload copy.
        return request_id
    module.request_id_from_result_wire = capture_id
    on_result = cls._on_result

    @functools.wraps(on_result)
    def result_callback(self, message):
        start = raw_ns()
        context.result = dict(request_id=None, wire_bytes=None)
        try:
            return on_result(self, message)
        finally:
            end = raw_ns()
            state = context.result
            del context.result
            wire = state.pop('_wire', None)
            if wire is not None:
                # Decode only the small header, after the measured callback ends.
                try:
                    _, size, _ = module.FRAME_PREFIX.unpack(wire[:module.FRAME_PREFIX.size])
                    state['header'] = json.loads(wire[module.FRAME_PREFIX.size:
                        module.FRAME_PREFIX.size + size])
                except Exception:
                    pass
            sink.emit('result_callback', start_ns=start, end_ns=end,
                duration_ns=end - start, failed=self.failed is not None, **state)
    cls._on_result = result_callback

    on_trace = cls._on_trace

    @functools.wraps(on_trace)
    def trace_callback(self, message):
        try:
            sideband = json.loads(message.data)
            if not isinstance(sideband, dict):
                raise ValueError('non-object publisher sideband')
            fields = dict(request_id=sideband.get('request_id'), sideband=sideband)
            for name in ('ros_message', 'ros_publish'):
                start, end = sideband.get(name + '_start'), sideband.get(name + '_end')
                if type(start) is int and type(end) is int:
                    fields[name + '_duration_ns'] = end - start
            sink.emit('publisher_sideband', **fields)
        except Exception as error:
            sink.emit('publisher_sideband_invalid', error_type=type(error).__name__)
        return on_trace(self, message)
    cls._on_trace = trace_callback
    validate_ack = module.validate_ack_line

    @functools.wraps(validate_ack)
    def ack_received(line, expected_request_id):
        received_ns = raw_ns()
        try:
            result = validate_ack(line, expected_request_id)
        except BaseException as error:
            sink.emit('host_ack_invalid', request_id=expected_request_id,
                received_ns=received_ns, error_type=type(error).__name__)
            raise
        sink.emit('host_ack_received', request_id=expected_request_id, received_ns=received_ns)
        return result
    module.validate_ack_line = ack_received


def install_publisher(module, sink):
    if hasattr(module, '_wait_for_result_ack'):
        wait = module._wait_for_result_ack

        @functools.wraps(wait)
        def wait_ack(rclpy, node, timeout_s):
            request_id = node.ack_gate.expected
            sink.emit('publisher_ack_wait_start', request_id=request_id)
            start, error_type = raw_ns(), None
            try:
                return wait(rclpy, node, timeout_s)
            except BaseException as error:
                error_type = type(error).__name__
                raise
            finally:
                end = raw_ns()
                sink.emit('publisher_ack_wait', request_id=request_id, start_ns=start,
                    end_ns=end, duration_ns=end - start, error_type=error_type)
        module._wait_for_result_ack = wait_ack

    if hasattr(module, 'recv_wire_frame'):
        recv = module.recv_wire_frame

        @functools.wraps(recv)
        def recv_handoff(*args, **kwargs):
            start, result, error_type = raw_ns(), None, None
            try:
                result = recv(*args, **kwargs)
                return result
            except BaseException as error:
                error_type = type(error).__name__
                raise
            finally:
                end = raw_ns()
                fields = dict(start_ns=start, end_ns=end, duration_ns=end - start,
                    error_type=error_type, request_id=None)
                if result is not None:
                    try:
                        fields['request_id'] = json.loads(result[0]).get('request_id')
                        fields['payload_bytes'] = len(result[1])
                    except Exception:
                        pass  # Original runtime owns wire validation and rejection.
                sink.emit('local_handoff_recv', **fields)
        module.recv_wire_frame = recv_handoff


def load_original(role, path=None):
    if role not in ('proxy', 'publisher'):
        raise ValueError('HIL_TIMING_ROLE must be proxy or publisher')
    filename = 'coptv2x_hil_ros_proxy.py' if role == 'proxy' else 'ditl_ros2_edge_bridge.py'
    source = Path(path or os.environ.get('HIL_TIMING_ORIGINAL',
        '/home/jichengzhi/V2X/.worktrees/V2Xverse-ditl/scripts/' + filename)).resolve()
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('_hil_timing_original_' + role, str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    role = os.environ['HIL_TIMING_ROLE']
    module = load_original(role)
    sink = TimingLog(os.environ['HIL_TIMING_LOG'], role)
    try:
        (install_proxy if role == 'proxy' else install_publisher)(module, sink)
        sink.emit('probe_started', original_file=module.__file__)
        return module.main(sys.argv[1:] if argv is None else argv)
    finally:
        sink.close()


if __name__ == '__main__':
    sys.exit(main())
