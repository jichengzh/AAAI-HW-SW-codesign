# -*- coding: utf-8 -*-
"""Unit tests for latency_frame_align (design §3.2/§3.3/§3.8). Pure-python, no GPU.

Run: python simulation/leaderboard/team_code/closedloop/test_latency_frame_align.py
"""
import csv
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from latency_frame_align import latency_to_delay_frames, RsuLatencyAligner, LatencyTraceReader, FRAME_MS


def check(cond, msg):
    if not cond:
        print("FAIL:", msg); sys.exit(1)
    print("ok:", msg)


def test_latency_to_delay():
    # 50 ms / frame; ceil; no implicit cap.
    check(latency_to_delay_frames(0) == 0, "L=0 -> Delta=0 (ideal/no-delay arm)")
    check(latency_to_delay_frames(20) == 1, "L=20ms -> ceil(20/50)=1 (p75+INT8 E7)")
    check(latency_to_delay_frames(48) == 1, "L=48ms -> 1 (FP16 base E7)")
    check(latency_to_delay_frames(50) == 1, "L=50ms -> exactly 1 frame")
    check(latency_to_delay_frames(51) == 2, "L=51ms -> 2 (just over one frame, ceil)")
    check(latency_to_delay_frames(108) == 3, "L=108ms -> ceil(108/50)=3 (best-chain e2e)")
    check(latency_to_delay_frames(219) == 5, "L=219ms -> ceil(219/50)=5 (baseline e2e)")
    check(latency_to_delay_frames(800) == 16, "L=800ms -> Delta=16, no 500ms cap")
    try:
        latency_to_delay_frames(800, max_delay_frames=10)
    except ValueError:
        check(True, "explicit max rejects instead of silently clipping")
    else:
        check(False, "explicit max must reject overflow")


def test_zero_delay_passthrough():
    # Delta=0 (ideal arm): result submitted at frame j is available at j, used at j.
    a = RsuLatencyAligner()
    for j in range(5):
        a.submit(j, payload="rsu_%d" % j, latency_ms=0, latency_source="injected_from_E7")
        p, au = a.consume(j)
        check(p == "rsu_%d" % j, "Delta=0: frame %d uses its own RSU result" % j)
        check(au['delta'] == 0 and au['is_zoh_held'] is False, "Delta=0: no zoh at frame %d" % j)
        check(au['zoh_age_frames'] == 0, "Delta=0: zoh_age 0 at frame %d" % j)


def test_injected_delay_lag():
    # Delta=3 (108ms best-chain). result from frame j available at j+3.
    a = RsuLatencyAligner()
    L = 108.0
    # produce RSU results every frame, consume every frame
    used = {}
    for k in range(8):
        a.submit(k, payload="rsu_%d" % k, latency_ms=L, latency_source="injected_from_E7")
        p, au = a.consume(k)
        used[k] = (p, au)
    # at k=0,1,2 nothing available (avail starts at frame 3) -> None + zoh
    check(used[0][0] is None, "k=0: no RSU available yet (Delta=3)")
    check(used[2][0] is None, "k=2: still nothing available (Delta=3)")
    # at k=3 the frame-0 result (avail at 0+3=3) becomes usable
    check(used[3][0] == "rsu_0", "k=3: uses rsu_0 (submitted frame0, Delta3 -> avail frame3)")
    check(used[3][1]['used_rsu_frame_id'] == 0, "k=3: used_rsu_frame_id==0 (3-frame lag)")
    check(used[5][0] == "rsu_2", "k=5: uses rsu_2 (3-frame lag held)")
    check(used[5][1]['delta'] == 3, "k=5: audit delta==3")


def test_zoh_when_no_new_result():
    # RSU only produces at frame 0 (Delta=1 -> avail frame1), then silent.
    a = RsuLatencyAligner()
    a.submit(0, payload="rsu_0", latency_ms=20, latency_source="measured_in_carla")
    # consume several frames; only frame0 result exists, avail at frame1
    p1, au1 = a.consume(1)
    check(p1 == "rsu_0" and au1['is_zoh_held'] is False, "k=1: first use of rsu_0, not zoh")
    p2, au2 = a.consume(2)
    check(p2 == "rsu_0" and au2['is_zoh_held'] is True, "k=2: ZOH holds rsu_0 (no new result)")
    check(au2['zoh_age_frames'] == 1, "k=2: zoh_age 1 frame (avail frame1, now frame2)")
    p4, au4 = a.consume(4)
    check(p4 == "rsu_0" and au4['zoh_age_frames'] == 3, "k=4: zoh_age 3 frames")
    check(au2['latency_ms_source'] == "measured_in_carla", "audit carries latency source")


def test_ego_never_held_is_caller_contract():
    # The aligner is RSU-only by construction; ego is handled by the caller and
    # never enters this buffer. This test documents that the aligner has no ego path.
    a = RsuLatencyAligner()
    # nothing submitted, consume at start -> None (caller keeps ego current itself)
    p, au = a.consume(0)
    check(p is None and au['is_zoh_held'] is True, "empty start: None + zoh flag (caller uses ego-only)")


def _write_trace_csv(path, tau_list):
    """Helper: write a minimal trace CSV for testing."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['call_idx', 'tau_ms'])
        for i, t in enumerate(tau_list):
            w.writerow([i, '%.4f' % t])


def test_trace_reader_basic():
    """LatencyTraceReader reads call_idx order, returns correct tau values."""
    taus = [108.0, 219.0, 50.0, 300.0, 20.0]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
        path = tf.name
    try:
        _write_trace_csv(path, taus)
        reader = LatencyTraceReader(path, cycle=True)
        check(reader.n_calls == 5, "LatencyTraceReader: n_calls=5")
        for i, expected in enumerate(taus):
            val = reader.next()
            check(abs(val - expected) < 1e-3,
                  "LatencyTraceReader: call %d tau=%.3f" % (i, expected))
    finally:
        os.unlink(path)


def test_trace_reader_cycle():
    """LatencyTraceReader cycles back to index 0 when exhausted."""
    taus = [100.0, 200.0, 150.0]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
        path = tf.name
    try:
        _write_trace_csv(path, taus)
        reader = LatencyTraceReader(path, cycle=True)
        # exhaust once
        for _ in range(3):
            reader.next()
        check(reader.cycle_count == 0, "trace_reader: no cycle yet after first pass")
        # next call triggers cycle
        val = reader.next()
        check(reader.cycle_count == 1, "trace_reader: cycle_count=1 after wrap")
        check(abs(val - taus[0]) < 1e-3, "trace_reader: cycled back to taus[0]")
        check(reader.pos == 1, "trace_reader: pos=1 after reading one after wrap")
    finally:
        os.unlink(path)


def test_trace_reader_hold_last():
    """LatencyTraceReader holds last value when cycle=False."""
    taus = [100.0, 200.0]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
        path = tf.name
    try:
        _write_trace_csv(path, taus)
        reader = LatencyTraceReader(path, cycle=False)
        reader.next(); reader.next()  # exhaust
        val = reader.next()  # should hold 200.0
        check(abs(val - 200.0) < 1e-3, "trace_reader hold: returns last value (200.0)")
        val2 = reader.next()
        check(abs(val2 - 200.0) < 1e-3, "trace_reader hold: continues holding last")
    finally:
        os.unlink(path)


def test_trace_reader_latency_source():
    """LatencyTraceReader.LATENCY_SOURCE is 'trace_replay_codriving_4090'."""
    check(LatencyTraceReader.LATENCY_SOURCE == 'trace_replay_codriving_4090',
          "LatencyTraceReader.LATENCY_SOURCE == 'trace_replay_codriving_4090'")


def test_trace_driven_aligner_delta_varies():
    """With trace reader, aligner gets varying Delta = ceil(tau/50) each call."""
    # trace: [20ms->1, 108ms->3, 219ms->5, 50ms->1, 300ms->6]
    taus = [20.0, 108.0, 219.0, 50.0, 300.0]
    expected_deltas = [1, 3, 5, 1, 6]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
        path = tf.name
    try:
        _write_trace_csv(path, taus)
        reader = LatencyTraceReader(path, cycle=True)
        aligner = RsuLatencyAligner()
        for step, (expected_delta) in enumerate(expected_deltas):
            tau = reader.next()
            delta = aligner.submit(step, payload="rsu_%d" % step,
                                   latency_ms=tau,
                                   latency_source=LatencyTraceReader.LATENCY_SOURCE)
            check(delta == expected_delta,
                  "trace-driven aligner step=%d tau=%.0fms -> Delta=%d" %
                  (step, tau, expected_delta))
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_latency_to_delay()
    test_zero_delay_passthrough()
    test_injected_delay_lag()
    test_zoh_when_no_new_result()
    test_ego_never_held_is_caller_contract()
    # ★ R-3 trace-reader tests
    test_trace_reader_basic()
    test_trace_reader_cycle()
    test_trace_reader_hold_last()
    test_trace_reader_latency_source()
    test_trace_driven_aligner_delta_varies()
    print("\nALL TESTS PASSED")
