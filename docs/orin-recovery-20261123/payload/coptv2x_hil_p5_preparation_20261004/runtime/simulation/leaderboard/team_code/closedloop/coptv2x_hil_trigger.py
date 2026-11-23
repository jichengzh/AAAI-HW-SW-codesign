from __future__ import print_function

from dataclasses import dataclass
import re
import sys
from threading import Lock

from coptv2x_hil_artifacts import HIL_ARM_IDS


HIL_TRIGGER_SCHEMA_VERSION = "coptv2x.hil-trigger.v1"

_TRIGGER_FIELDS = (
    "schema_version",
    "run_id",
    "route_id",
    "scenario_id",
    "sample_id",
    "sequence_no",
    "source_frame",
    "source_sim_time_ns",
    "arm_id",
)
_IDENTITY_FIELDS = ("run_id", "route_id", "scenario_id", "sample_id")
_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MAX_COUNTER = 2 ** 63 - 1


class HilTriggerError(ValueError):
    """Raised when a HIL trigger violates the fixed wire contract."""


def _require_identity(value, field):
    if type(value) is not str or not _IDENTITY_PATTERN.fullmatch(value):
        raise HilTriggerError(
            "%s must be a bounded non-path identity string" % field
        )
    return value


def _require_counter(value, field):
    if type(value) is not int or value < 0 or value > _MAX_COUNTER:
        raise HilTriggerError(
            "%s must be a non-negative signed 64-bit integer" % field
        )
    return value


@dataclass(frozen=True)
class HilTrigger:
    schema_version: str
    run_id: str
    route_id: str
    scenario_id: str
    sample_id: str
    sequence_no: int
    source_frame: int
    source_sim_time_ns: int
    arm_id: str

    def to_metadata(self):
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "route_id": self.route_id,
            "scenario_id": self.scenario_id,
            "sample_id": self.sample_id,
            "sequence_no": self.sequence_no,
            "source_frame": self.source_frame,
            "source_sim_time_ns": self.source_sim_time_ns,
            "arm_id": self.arm_id,
        }


def parse_hil_trigger(payload):
    """Parse a decoded trigger mapping and reject every non-contract field."""

    if type(payload) is not dict:
        raise HilTriggerError("trigger payload must be a plain object")
    actual_fields = set(payload)
    expected_fields = set(_TRIGGER_FIELDS)
    if actual_fields != expected_fields:
        raise HilTriggerError(
            "trigger field mismatch; missing=%s unknown=%s"
            % (
                sorted(expected_fields - actual_fields),
                sorted(actual_fields - expected_fields, key=repr),
            )
        )
    if payload["schema_version"] != HIL_TRIGGER_SCHEMA_VERSION:
        raise HilTriggerError(
            "schema_version must be %s" % HIL_TRIGGER_SCHEMA_VERSION
        )
    values = {
        field: _require_identity(payload[field], field)
        for field in _IDENTITY_FIELDS
    }
    arm_id = payload["arm_id"]
    if type(arm_id) is not str or arm_id not in HIL_ARM_IDS:
        raise HilTriggerError(
            "unknown arm_id %r; expected one of %s" % (arm_id, HIL_ARM_IDS)
        )
    return HilTrigger(
        schema_version=HIL_TRIGGER_SCHEMA_VERSION,
        run_id=values["run_id"],
        route_id=values["route_id"],
        scenario_id=values["scenario_id"],
        sample_id=values["sample_id"],
        sequence_no=_require_counter(payload["sequence_no"], "sequence_no"),
        source_frame=_require_counter(payload["source_frame"], "source_frame"),
        source_sim_time_ns=_require_counter(
            payload["source_sim_time_ns"], "source_sim_time_ns"
        ),
        arm_id=arm_id,
    )


class HilTriggerAdmission(object):
    """Admit one strictly increasing trigger stream for one frozen run."""

    def __init__(self, run_id, route_id, scenario_id, arm_id):
        self.run_id = _require_identity(run_id, "run_id")
        self.route_id = _require_identity(route_id, "route_id")
        self.scenario_id = _require_identity(scenario_id, "scenario_id")
        if type(arm_id) is not str or arm_id not in HIL_ARM_IDS:
            raise HilTriggerError(
                "unknown arm_id %r; expected one of %s" % (arm_id, HIL_ARM_IDS)
            )
        self.arm_id = arm_id
        self._last_sequence_no = -1
        self._reserved_sequence_nos = set()
        self._lock = Lock()

    def validate(self, payload):
        """Validate active-run identity without consuming sequence state."""

        trigger = parse_hil_trigger(payload)
        expected = (
            ("run_id", self.run_id),
            ("route_id", self.route_id),
            ("scenario_id", self.scenario_id),
            ("arm_id", self.arm_id),
        )
        for field, value in expected:
            if getattr(trigger, field) != value:
                raise HilTriggerError(
                    "%s does not match the active HIL run" % field
                )
        return trigger

    def reserve(self, payload):
        """Reserve sequence state before the local batch is materialized."""

        trigger = self.validate(payload)
        with self._lock:
            sequence_floor = self._last_sequence_no
            if self._reserved_sequence_nos:
                sequence_floor = max(
                    sequence_floor, max(self._reserved_sequence_nos)
                )
            if trigger.sequence_no <= sequence_floor:
                raise HilTriggerError("sequence_no must increase strictly")
            self._reserved_sequence_nos.add(trigger.sequence_no)
        return trigger

    def cancel(self, trigger):
        """Release a reservation when no inference task was launched."""

        with self._lock:
            self._reserved_sequence_nos.discard(trigger.sequence_no)

    def admit_trigger(self, trigger):
        """Atomically consume a previously validated trigger sequence."""

        if type(trigger) is not HilTrigger:
            raise HilTriggerError("admission requires a validated HilTrigger")
        expected = (
            ("run_id", self.run_id),
            ("route_id", self.route_id),
            ("scenario_id", self.scenario_id),
            ("arm_id", self.arm_id),
        )
        with self._lock:
            for field, value in expected:
                if getattr(trigger, field) != value:
                    raise HilTriggerError(
                        "%s does not match the active HIL run" % field
                    )
            if trigger.sequence_no not in self._reserved_sequence_nos:
                raise HilTriggerError("sequence_no was not reserved")
            self._reserved_sequence_nos.remove(trigger.sequence_no)
            if trigger.sequence_no <= self._last_sequence_no:
                raise HilTriggerError("sequence_no must increase strictly")
            self._last_sequence_no = trigger.sequence_no
        return trigger

    def admit(self, payload):
        trigger = self.reserve(payload)
        try:
            return self.admit_trigger(trigger)
        except Exception:
            self.cancel(trigger)
            raise


__all__ = (
    "HIL_TRIGGER_SCHEMA_VERSION",
    "HilTrigger",
    "HilTriggerAdmission",
    "HilTriggerError",
    "parse_hil_trigger",
)

# CARLA loads some team-code modules as top-level modules while the agent
# imports this module through ``team_code.closedloop``. Keep one class
# identity across both supported import paths.
_module = sys.modules[__name__]
sys.modules.setdefault("coptv2x_hil_trigger", _module)
sys.modules.setdefault("team_code.closedloop.coptv2x_hil_trigger", _module)
