#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen ten-arm manifest for the first Deployment-in-the-Loop trends."""

from __future__ import print_function

from dataclasses import asdict, dataclass
import math
import numbers


MANIFEST_SCHEMA = "ditl.task6-manifest.v1"
ROUTE_ID = 146
SCENARIO_SUFFIX = "_1"
REPEAT = 0
CARLA_SEED = 2000
TRAFFIC_SEED = 2000

CLEAN_PROFILE_ID = "raw_samehost_idle"
COMPUTE_PROFILE_IDS = (
    "raw_samehost_gpu_low",
    "raw_samehost_gpu_medium",
    "raw_samehost_gpu_high",
)
COMM_DROP_RATES = (0.05, 0.10, 0.20)
SPATIAL_DX_M = (0.5, 1.0, 2.0)
SPATIAL_DY_M = 0.0
SPATIAL_DYAW_DEG = 0.0
ARM_IDS = (
    "clean",
    "comm_drop_005", "comm_drop_010", "comm_drop_020",
    "spatial_dx_05", "spatial_dx_10", "spatial_dx_20",
    "compute_gpu_low", "compute_gpu_medium", "compute_gpu_high",
)


@dataclass(frozen=True)
class Task6Arm:
    arm_id: str
    family: str
    severity: str
    route_id: int
    scenario_suffix: str
    repeat: int
    carla_seed: int
    traffic_seed: int
    drop_rate: float
    spatial_dx_m: float
    spatial_dy_m: float
    spatial_dyaw_deg: float
    trace_profile_id: str


def _common_fields():
    return {
        "route_id": ROUTE_ID,
        "scenario_suffix": SCENARIO_SUFFIX,
        "repeat": REPEAT,
        "carla_seed": CARLA_SEED,
        "traffic_seed": TRAFFIC_SEED,
    }


def _arm(arm_id, family, severity, drop_rate, spatial_dx_m, profile_id):
    fields = _common_fields()
    return Task6Arm(
        arm_id=arm_id,
        family=family,
        severity=severity,
        drop_rate=drop_rate,
        spatial_dx_m=spatial_dx_m,
        spatial_dy_m=SPATIAL_DY_M,
        spatial_dyaw_deg=SPATIAL_DYAW_DEG,
        trace_profile_id=profile_id,
        **fields
    )


def build_task6_arms() -> tuple:
    """Return the canonical clean plus nine single-factor Task 6 arms."""
    arms = [
        _arm("clean", "clean", "clean", 0.0, 0.0, CLEAN_PROFILE_ID),
        _arm("comm_drop_005", "communication", "low", COMM_DROP_RATES[0], 0.0, CLEAN_PROFILE_ID),
        _arm("comm_drop_010", "communication", "medium", COMM_DROP_RATES[1], 0.0, CLEAN_PROFILE_ID),
        _arm("comm_drop_020", "communication", "high", COMM_DROP_RATES[2], 0.0, CLEAN_PROFILE_ID),
        _arm("spatial_dx_05", "spatial", "low", 0.0, SPATIAL_DX_M[0], CLEAN_PROFILE_ID),
        _arm("spatial_dx_10", "spatial", "medium", 0.0, SPATIAL_DX_M[1], CLEAN_PROFILE_ID),
        _arm("spatial_dx_20", "spatial", "high", 0.0, SPATIAL_DX_M[2], CLEAN_PROFILE_ID),
        _arm("compute_gpu_low", "compute", "low", 0.0, 0.0, COMPUTE_PROFILE_IDS[0]),
        _arm("compute_gpu_medium", "compute", "medium", 0.0, 0.0, COMPUTE_PROFILE_IDS[1]),
        _arm("compute_gpu_high", "compute", "high", 0.0, 0.0, COMPUTE_PROFILE_IDS[2]),
    ]
    return tuple(arms)


def _reject(message):
    raise ValueError("invalid Task 6 arm manifest: %s" % message)


def _finite_number(value, field, arm_id):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _reject("%s for %s must be a finite number" % (field, arm_id))
    if not math.isfinite(value):
        _reject("%s for %s must be finite" % (field, arm_id))


def _exact_integer(value, expected, field, arm_id):
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _reject("%s for %s must be %r" % (field, arm_id, expected))


def _validate_arm(arm, expected):
    if not isinstance(arm, Task6Arm):
        _reject("all records must be Task6Arm values")
    if arm.arm_id != expected.arm_id:
        _reject("arm order or ID is not canonical")
    for field, expected_value in (
        ("family", expected.family),
        ("severity", expected.severity),
        ("scenario_suffix", expected.scenario_suffix),
        ("trace_profile_id", expected.trace_profile_id),
    ):
        if type(getattr(arm, field)) is not type(expected_value) or getattr(arm, field) != expected_value:
            _reject("unsupported %s for %s" % (field, arm.arm_id))
    for field, expected_value in (
        ("route_id", expected.route_id),
        ("repeat", expected.repeat),
        ("carla_seed", expected.carla_seed),
        ("traffic_seed", expected.traffic_seed),
    ):
        _exact_integer(getattr(arm, field), expected_value, field, arm.arm_id)
    for field in (
        "drop_rate", "spatial_dx_m", "spatial_dy_m", "spatial_dyaw_deg"
    ):
        _finite_number(getattr(arm, field), field, arm.arm_id)
        if getattr(arm, field) != getattr(expected, field):
            _reject("unsupported %s for %s" % (field, arm.arm_id))


def validate_task6_arms(arms) -> tuple:
    """Validate and return a fresh tuple in the frozen canonical order."""
    try:
        candidate = tuple(arm for arm in arms)
    except TypeError:
        _reject("arms must be an iterable of exactly ten records")
    expected = build_task6_arms()
    if len(candidate) != len(expected):
        _reject("exactly ten arms are required")
    seen = set()
    for arm in candidate:
        if isinstance(arm, Task6Arm):
            if arm.arm_id in seen:
                _reject("duplicate arm ID %s" % arm.arm_id)
            seen.add(arm.arm_id)
    if len(seen) != len(expected):
        _reject("arm IDs must be the exact ten-arm set")
    for actual, expected_arm in zip(candidate, expected):
        _validate_arm(actual, expected_arm)
    return tuple(arm for arm in candidate)


def manifest_to_dict(arms) -> dict:
    """Serialize a validated manifest into fresh, JSON-compatible mappings."""
    validated = validate_task6_arms(arms)
    return {
        "schema": MANIFEST_SCHEMA,
        "route": _common_fields(),
        "arms": [asdict(arm) for arm in validated],
    }
