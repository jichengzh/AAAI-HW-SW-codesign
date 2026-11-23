"""
L1 Trajectory Tracking Controller  (I-1)
=========================================
Reference: sim_ego_rsu_latency_strategy_v1.md §3 I-1
           sim_closedloop_I1_l1_controller_design_v1.md

Provides coordinate transforms and time-indexed interpolation for L1 control.

Coordinate convention (verified against pnp_agent_e2e.py:tick()):
  pos = (gps_x, gps_y) in GPS-metric frame from RoutePlanner
        where gps_x = pos[0], gps_y = pos[1]
  theta = compass heading (radians, IMU)
  rotation alpha = pi/2 - theta  ← matches transform_2d_points (line 799):
                                    r = np.pi/2 - measurements["theta"]
  ★ FIX-C: was wrongly theta + pi/2, which cancels in round-trip but causes
            ~2*theta rotation error when re-projecting at a different heading.

  ego_local → world:  wp_w = R(pi/2 - theta) @ wp_l + [gps_x, gps_y]
  world → ego_local:  wp_l = R(pi/2 - theta)^T @ (wp_w - [gps_x, gps_y])

  Round-trip guarantee: world_to_ego_local(ego_local_to_world(wp, p), p) == wp
                        (proven analytically; verified in V0 test)

DT_PER_WP = 0.2 s: derived from V2X_Controller formula
    desired_speed = mean(np.diff(displacement)[:3]) * max_speed
    i.e. spacing_m * (1/DT_PER_WP) = speed_m_s  =>  DT_PER_WP = 0.2 s

口径: injected_sim_arm / L1 / I-1 (sim_ego_rsu_latency_strategy_v1 §3)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

# ── constants ─────────────────────────────────────────────────────────────────
DT_PER_WP: float = 0.2   # seconds per waypoint (confirmed from controller * 5 factor)
N_WP: int = 10            # number of waypoints from MotionNet planner


# ── coordinate helpers ────────────────────────────────────────────────────────

def _rot(alpha: float) -> np.ndarray:
    """2D rotation matrix R(alpha): local → world."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def ego_local_to_world(
    wp_local: np.ndarray,
    gps_x: float,
    gps_y: float,
    theta: float,
) -> np.ndarray:
    """Convert (N, 2) ego-local waypoints → GPS-metric world frame.

    Parameters
    ----------
    wp_local : (N, 2) float array
        Waypoints in ego-local BEV frame (output of MotionNet planner).
    gps_x, gps_y : float
        Ego position: measurements['gps_x'], measurements['gps_y'].
    theta : float
        IMU compass heading (radians) = measurements['theta'].

    Returns
    -------
    (N, 2) float array in world GPS-metric frame.
    """
    R = _rot(np.pi / 2 - theta)
    return (R @ np.asarray(wp_local, dtype=np.float64).T).T + np.array([gps_x, gps_y])


def world_to_ego_local(
    wp_world: np.ndarray,
    gps_x: float,
    gps_y: float,
    theta: float,
) -> np.ndarray:
    """Convert (N, 2) GPS-metric world waypoints → current ego-local frame.

    Exact inverse of ego_local_to_world.
    """
    R = _rot(np.pi / 2 - theta)
    wp_rel = np.asarray(wp_world, dtype=np.float64) - np.array([gps_x, gps_y])
    return (R.T @ wp_rel.T).T


# ── time-indexed trajectory helpers ──────────────────────────────────────────

def time_index_ref(wp_local: np.ndarray, dt: float) -> np.ndarray:
    """Interpolate reference waypoint at elapsed time *dt* seconds.

    Returns a single (2,) point in ego-local frame.
    dt = 0.0 → wp[0];  dt = 0.2 → wp[1];  dt ≥ 1.8 → wp[9] (clamped).

    Parameters
    ----------
    wp_local : (10, 2)
        Waypoints in current ego-local frame (after world→local transform).
    dt : float
        Seconds elapsed since plan was computed (t_now - t_anchor).
    """
    t_idx = max(dt, 0.0) / DT_PER_WP          # float index
    t_idx = min(t_idx, float(N_WP - 1))        # clamp
    lo = int(t_idx)
    frac = t_idx - lo
    wp_arr = np.asarray(wp_local, dtype=np.float64)
    if lo < N_WP - 1:
        return wp_arr[lo] * (1.0 - frac) + wp_arr[lo + 1] * frac
    return wp_arr[N_WP - 1].copy()


def l1_desired_speed(wp_local: np.ndarray, dt: float, max_speed: float) -> float:
    """Time-indexed desired speed (m/s).

    Uses the 3 waypoint spacings starting from the time-indexed position.
    When the plan is stale (large dt), this naturally picks the tail-end
    spacings — which are smaller (slower) as the vehicle approaches the
    end of the predicted trajectory. This is the "plan age → slower speed"
    mechanism needed for I-2 τ_ego injection.

    Matches V2X_Controller formula:
        desired_speed = mean(diff(displacement)[:3]) * max_speed
    """
    wp_arr = np.asarray(wp_local, dtype=np.float64)
    displacement = np.linalg.norm(wp_arr, ord=2, axis=1)  # (10,)
    diffs = np.diff(displacement)                           # (9,)

    t_idx = max(dt, 0.0) / DT_PER_WP
    t_idx = min(t_idx, float(N_WP - 1))
    lo = int(t_idx)
    start = min(lo, len(diffs) - 3)
    start = max(start, 0)
    speed = float(np.mean(diffs[start:start + 3])) * max_speed
    return max(speed, 0.0)


# ── plan store ────────────────────────────────────────────────────────────────

class L1PlanStore:
    """Stores inference-time plans (world frame) with τ_ego availability delay.

    Updated once per inference frame (every skip_frames ticks) inside
    generate_action_from_model_output().  Read at every CARLA tick in
    run_l1_step() for L1 non-inference frames.

    ★ I-2 τ_ego injection
    ----------------------
    Each plan computed at inference time t_anchor becomes ACTIVE only at
    t_available = t_anchor + tau_ego_s.  Between an inference and that plan's
    availability, the controller keeps tracking the previous (older) active
    plan — physically: "frame k executes the plan derived from frame k−Δ_ego's
    ego perception" (sim_ego_rsu_latency_strategy_v1.md §2/§3 I-2).

    tau_ego_s = 0.0  → t_available = t_anchor → plan active immediately →
                       byte-identical to the I-1 (no-delay) behaviour.
    tau_ego_s > 0    → Δ_ego = ceil(tau_ego_s / tick_dt) frames of staleness.

    select_active(t_now) MUST be called once per tick before reading
    desired_speed / active_t_anchor / get_local_ref (run_l1_step does this).

    Thread safety: CARLA sync mode is single-threaded; no locking needed.
    """

    _MAX_BUFFER = 16   # keep at most this many recent plans (bounds memory)

    def __init__(self, tau_ego_s: float = 0.0) -> None:
        self._tau_ego_s: float = max(float(tau_ego_s), 0.0)
        self._buffer: list = []           # list[dict], ascending t_anchor
        self._active: Optional[dict] = None

    # ── write (inference frame) ──────────────────────────────────────────────

    def update(
        self,
        wp_ego: np.ndarray,
        gps_x: float,
        gps_y: float,
        theta: float,
        t_anchor: float,
        max_speed: float = 5.0,
    ) -> None:
        """Anchor ego-local waypoints to world frame and buffer with τ_ego delay.

        Parameters
        ----------
        wp_ego : (10, 2) float array
            Planner output (ego-local BEV frame).
        gps_x, gps_y : float
            Ego GPS-metric position at inference time.
        theta : float
            Compass heading (radians) at inference time.
        t_anchor : float
            CARLA game time (seconds) at inference time.
        max_speed : float
            Controller max_speed (m/s) from config.  Used to compute
            desired_speed using the L0-proven formula on fresh wp_ego.
        """
        wp_world = ego_local_to_world(wp_ego, gps_x, gps_y, theta)

        # ★ V2: freeze desired_speed from FRESH wp_ego (L0-proven formula, dt=0).
        # wp_ego is fully in front of the vehicle at inference time → monotone
        # Euclidean norms → clean diffs → no geometric spike.  Held until the
        # next inference frame (every skip_frames * tick_dt ≈ 200 ms).
        _wp = np.asarray(wp_ego, dtype=np.float64)
        _disp = np.linalg.norm(_wp, ord=2, axis=1)          # (N_WP,)
        _v = float(np.mean(np.diff(_disp)[:3])) * float(max_speed)

        plan = {
            'waypoints_world': wp_world,       # (10, 2) world frame
            'desired_speed':   max(_v, 0.0),   # ★ V2: frozen scalar (m/s)
            'gps_x': gps_x,
            'gps_y': gps_y,
            'theta': theta,
            't_anchor': t_anchor,
            't_available': t_anchor + self._tau_ego_s,   # ★ I-2 τ_ego delay
        }
        self._buffer.append(plan)
        if len(self._buffer) > self._MAX_BUFFER:
            self._buffer = self._buffer[-self._MAX_BUFFER:]

    # ── activation selection (once per tick) ─────────────────────────────────

    def select_active(self, t_now: float) -> bool:
        """Pick the most recent plan that is available (t_available ≤ t_now).

        Sets self._active.  Returns True iff an active plan exists.
        Plans not yet available (τ_ego still elapsing) are skipped, so the
        controller naturally holds the previous active plan (ZOH on ego path).
        """
        active = None
        for plan in self._buffer:
            if plan['t_available'] <= t_now:
                # buffer is append-ordered by t_anchor; last qualifying wins
                if active is None or plan['t_anchor'] >= active['t_anchor']:
                    active = plan
        if active is not None:
            self._active = active
        # else: keep previous self._active (ZOH); None only before first arrival
        return self._active is not None

    # ── read (every tick, after select_active) ───────────────────────────────

    def get_local_ref(
        self,
        gps_x: float,
        gps_y: float,
        theta: float,
        t_now: float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Convert the ACTIVE plan to current ego-local frame.

        Returns
        -------
        wp_local : (10, 2) or None
            All 10 waypoints in current ego-local frame.
            None if no plan is active yet (sim start / τ_ego not elapsed).
        dt : float
            Seconds elapsed since the active plan was anchored
            (t_now - t_anchor), i.e. includes the τ_ego staleness.
        """
        if self._active is None:
            return None, 0.0
        dt = max(t_now - self._active['t_anchor'], 0.0)
        wp_local = world_to_ego_local(
            self._active['waypoints_world'], gps_x, gps_y, theta
        )
        return wp_local, dt

    @property
    def has_plan(self) -> bool:
        """True once an active plan exists (after select_active)."""
        return self._active is not None

    @property
    def active_t_anchor(self) -> float:
        """t_anchor (CARLA game time) of the active plan; NaN if none."""
        return self._active['t_anchor'] if self._active is not None else float('nan')

    @property
    def desired_speed(self) -> float:
        """Desired speed frozen at the ACTIVE plan's inference frame (m/s).

        With τ_ego>0 this is the speed from an older (Δ_ego-stale) inference,
        so the ego reacts to speed changes Δ_ego frames late.
        Returns 0.0 if no plan is active yet.
        """
        if self._active is None:
            return 0.0
        return self._active.get('desired_speed', 0.0)
