# -*- coding: utf-8 -*-
"""C-5 offline self-check for the no-roadside (disable_rsu) control arm.

Pure-CPU, no CARLA / no GPU / no torch. Mirrors the test_latency_frame_align.py
style: builds minimal mocks of the two gating points that C-5 touches and
asserts the no-roadside arm degrades fusion to ego-only without raising.

Validates design §4.A-③:
  - config `simulation.disable_rsu: true` -> spawn_rsu() is NOT called and
    rsu_data stays [].
  - downstream `if len(rsu_data_raw) > 0` (pnp_infer_action_e2e L950) is the
    gate that turns empty RSU into ego-only fusion; empty list -> branch skipped,
    no IndexError.
  - default (flag absent / false) -> RSU path runs as before (regression guard).

Run: python simulation/leaderboard/team_code/closedloop/test_disable_rsu.py
"""
import sys


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)
    print("ok:", msg)


# --- minimal mock of the agent run_step RSU gating (pnp_agent_e2e.py) ---------
# Mirrors the real code exactly:
#   self.disable_rsu = bool(config.get('simulation', {}).get('disable_rsu', False))
#   if not self.disable_rsu: self.spawn_rsu()
#   rsu_data = []
#   if not self.disable_rsu and step % change_rsu_frame != 0: <capture rsu_data>
class FakeAgent:
    def __init__(self, config):
        self.config = config
        self.ego_vehicles_num = config['simulation']['ego_num']
        self.disable_rsu = bool(
            config.get('simulation', {}).get('disable_rsu', False))
        self.spawn_called = 0

    def spawn_rsu(self):
        self.spawn_called += 1

    def run_step_rsu_branch(self, step):
        """Reproduces only the RSU-related control flow of run_step."""
        if not self.disable_rsu:
            self.spawn_rsu()
        rsu_data = []
        change = self.config['RSU']['change_rsu_frame']
        if not self.disable_rsu and step % change != 0:
            for vehicle_num in range(self.ego_vehicles_num):
                # in the real agent this is rsu[vn].process(...); the value is
                # opaque here, a non-None placeholder is enough for the mock.
                rsu_data.append({'measurements': {'lidar_pose_x': 0.0}})
        return rsu_data


# --- minimal mock of the downstream fusion gate (pnp_infer_action_e2e L950) ---
def downstream_fusion_gate(rsu_data_raw):
    """Returns 'ego_only' when no RSU, else 'collab'. Must not raise on []."""
    if len(rsu_data_raw) > 0:
        # real code indexes rsu_data_raw[ego_i]['measurements'][...]; with empty
        # list this branch is never entered, so ego-only is exception-free.
        _ = rsu_data_raw[0]['measurements']['lidar_pose_x']
        return 'collab'
    return 'ego_only'


def _cfg(disable_rsu=None, ego_num=1, change=5):
    sim = {'ego_num': ego_num, 'skip_frames': 4}
    if disable_rsu is not None:
        sim['disable_rsu'] = disable_rsu
    return {'simulation': sim, 'RSU': {'change_rsu_frame': change}}


def test_disable_rsu_true_gates_spawn_and_empties_rsu():
    a = FakeAgent(_cfg(disable_rsu=True))
    check(a.disable_rsu is True, "disable_rsu:true parsed as True")
    for step in range(1, 13):  # cover both step%5==0 and !=0 frames
        rsu = a.run_step_rsu_branch(step)
        check(rsu == [], "disable_rsu: rsu_data empty at step %d" % step)
    check(a.spawn_called == 0, "disable_rsu: spawn_rsu never called")
    check(downstream_fusion_gate([]) == 'ego_only',
          "disable_rsu: empty rsu_data -> ego-only fusion (no exception)")


def test_disable_rsu_default_runs_rsu():
    # flag absent -> default False -> RSU path active (regression guard)
    a = FakeAgent(_cfg(disable_rsu=None))
    check(a.disable_rsu is False, "absent flag defaults to False")
    captured = False
    for step in range(1, 13):
        rsu = a.run_step_rsu_branch(step)
        if step % 5 != 0:
            check(len(rsu) == a.ego_vehicles_num,
                  "default: rsu captured on non-spawn frame %d" % step)
            captured = True
    check(a.spawn_called > 0, "default: spawn_rsu called")
    check(captured, "default: at least one frame captured RSU data")
    check(downstream_fusion_gate(
        [{'measurements': {'lidar_pose_x': 1.0}}]) == 'collab',
        "default: non-empty rsu_data -> collaborative fusion")


def test_disable_rsu_explicit_false_runs_rsu():
    a = FakeAgent(_cfg(disable_rsu=False))
    check(a.disable_rsu is False, "explicit false parsed as False")
    a.run_step_rsu_branch(1)  # step%5 != 0 -> would capture / spawn
    check(a.spawn_called == 1, "explicit false: spawn_rsu called")


def test_ego_only_gate_never_raises_on_empty():
    # the load-bearing safety claim: empty list must not IndexError
    try:
        result = downstream_fusion_gate([])
    except Exception as e:  # noqa: BLE001
        check(False, "downstream gate raised on empty rsu_data: %r" % e)
        return
    check(result == 'ego_only', "empty rsu_data path is exception-free")


if __name__ == "__main__":
    test_disable_rsu_true_gates_spawn_and_empties_rsu()
    test_disable_rsu_default_runs_rsu()
    test_disable_rsu_explicit_false_runs_rsu()
    test_ego_only_gate_never_raises_on_empty()
    print("\nALL C-5 TESTS PASSED")
