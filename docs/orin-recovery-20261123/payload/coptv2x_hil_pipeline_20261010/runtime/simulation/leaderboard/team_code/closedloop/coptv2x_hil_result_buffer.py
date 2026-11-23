from __future__ import print_function

import copy
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

_CLOSEDLOOP_DIR = str(Path(__file__).resolve().parent)
if _CLOSEDLOOP_DIR not in sys.path:
    sys.path.insert(0, _CLOSEDLOOP_DIR)

from coptv2x_hil_backend import (
    CODRIVING_BATCH_ORDER,
    COPTV2X_PAYLOAD_SEMANTICS,
    HilFeatureResult,
    hash_feature_payload,
    validate_feature_payload_contract,
)
from coptv2x_hil_trigger import HilTrigger, parse_hil_trigger
from ditl_edge_protocol import verified_feature_payload_sha256


HIL_AUDIT_SCHEMA_VERSION = "coptv2x.hil-fusion-audit.v1"
_RSU_PAYLOAD_SEMANTICS = "codriving_rsu_multiscale_features.v1"


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason: str
    sequence_no: Optional[int]
    evicted_sequence_no: Optional[int] = None


@dataclass(frozen=True)
class FusionSelection:
    result: Optional[HilFeatureResult]
    audit: Mapping[str, object]


@dataclass(frozen=True)
class HilResultEnvelope:
    feature_result: HilFeatureResult
    batch_order: tuple


@dataclass(frozen=True)
class _ArrivedResult:
    result: HilFeatureResult
    arrival_ns: int


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError("%s must be a positive integer" % label)
    return value


def _counter(value, label):
    if type(value) is not int or value < 0 or value > 2 ** 63 - 1:
        raise ValueError("%s must be a non-negative signed 64-bit integer" % label)
    return value


def _immutable_bytes_owner(value):
    """Prove storage is an immutable bytes owner, not just a readonly view.

    NumPy's readonly flag alone is insufficient: another view can still mutate
    an ndarray or bytearray base. Restrict reuse to contiguous readonly arrays
    whose complete ndarray/memoryview base chain terminates at actual bytes.
    """
    if (not isinstance(value, np.ndarray) or value.flags.writeable
            or not value.flags.c_contiguous):
        return None
    owner = value
    while isinstance(owner, (np.ndarray, memoryview)):
        if isinstance(owner, memoryview):
            if not owner.readonly:
                return None
            owner = owner.obj
        else:
            owner = owner.base
    return owner if type(owner) is bytes else None


def _frozen_feature_result(result, reuse_immutable=False, checksum_dedup=False):
    outputs = {}
    for name, value in result.outputs.items():
        if (reuse_immutable and result.payload_semantics == _RSU_PAYLOAD_SEMANTICS
                and _immutable_bytes_owner(value) is not None):
            copied = value
        else:
            copied = np.array(value, copy=True, order="C")
            copied.setflags(write=False)
        outputs[str(name)] = copied
    verified = (
        checksum_dedup and reuse_immutable
        and result.payload_semantics == _RSU_PAYLOAD_SEMANTICS
        and verified_feature_payload_sha256(result.outputs) is not None
    )
    return HilFeatureResult(
        outputs=result.outputs if verified else MappingProxyType(outputs),
        artifact_identity=_freeze(dict(result.artifact_identity)),
        trigger_identity=_freeze(dict(result.trigger_identity)),
        payload_semantics=str(result.payload_semantics),
        payload_sha256=str(result.payload_sha256),
        remote_input_tensor_bytes=int(result.remote_input_tensor_bytes),
        timestamps_ns=_freeze(dict(result.timestamps_ns)),
        durations_ns=_freeze(dict(result.durations_ns)),
        model=_freeze(dict(result.model)),
    )


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value):
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


class CoptV2xHilResultBuffer(object):
    """Bounded, identity-bound result cache with monotonic fusion selection."""

    drop_policy = "evict_lowest_sequence"

    def __init__(
        self,
        run_id,
        route_id,
        scenario_id,
        arm_id,
        capacity,
        contract_validator=validate_feature_payload_contract,
        trigger_capacity=None,
        timing_mode="async",
        fixed_comm_delay_ms=100,
        checksum_dedup=False,
    ):
        if timing_mode not in ("async", "serial_fixed_comm"):
            raise ValueError("unsupported HIL timing_mode %r" % timing_mode)
        if type(fixed_comm_delay_ms) not in (int, float) or fixed_comm_delay_ms != 100:
            raise ValueError("fixed_comm_delay_ms must be exactly 100")
        self.timing_mode = timing_mode
        if checksum_dedup and timing_mode != "serial_fixed_comm":
            raise ValueError("checksum_dedup requires serial_fixed_comm")
        self.checksum_dedup = bool(checksum_dedup)
        self.fixed_comm_delay_ms = 100
        self.run_id = str(run_id)
        self.route_id = str(route_id)
        self.scenario_id = str(scenario_id)
        self.arm_id = str(arm_id)
        self.capacity = _positive_integer(capacity, "capacity")
        if trigger_capacity is None:
            trigger_capacity = max(self.capacity * 4, 4096)
        self.trigger_capacity = _positive_integer(
            trigger_capacity, "trigger_capacity"
        )
        if not callable(contract_validator):
            raise TypeError("contract_validator must be callable")
        if timing_mode == "serial_fixed_comm" and contract_validator is validate_feature_payload_contract:
            from coptv2x_hil_backend import create_feature_payload_validator
            self._contract_validator = create_feature_payload_validator()
        else:
            self._contract_validator = contract_validator
        self._triggers = OrderedDict()
        self._results = {}
        self._retired_sequence_floor = -1
        self._last_registered_sequence = -1
        self._last_registered_source_frame = -1
        self._selected = None
        self._rejection_count = 0
        self._last_reported_rejection_count = 0
        self._last_rejection_reason = None
        self._lock = Lock()

    @property
    def size(self):
        with self._lock:
            return len(self._results)

    def register_trigger(self, trigger, published_ns):
        if type(trigger) is not HilTrigger:
            raise TypeError("trigger must be a validated HilTrigger")
        _counter(published_ns, "published_ns")
        self._validate_active_trigger(trigger)
        identity = trigger.to_metadata()
        with self._lock:
            if trigger.sequence_no <= self._last_registered_sequence:
                raise ValueError("trigger sequence_no must increase strictly")
            if trigger.source_frame < self._last_registered_source_frame:
                raise ValueError("trigger source_frame must not regress")
            self._triggers[trigger.sequence_no] = MappingProxyType(
                copy.deepcopy(identity)
            )
            self._last_registered_sequence = trigger.sequence_no
            self._last_registered_source_frame = trigger.source_frame
            while len(self._triggers) > self.trigger_capacity:
                sequence_no, _ = self._triggers.popitem(last=False)
                self._retired_sequence_floor = max(
                    self._retired_sequence_floor, sequence_no
                )

    def cancel_trigger(self, trigger):
        if type(trigger) is not HilTrigger:
            return
        with self._lock:
            self._triggers.pop(trigger.sequence_no, None)

    def _validate_active_trigger(self, trigger):
        for field, expected in (
            ("run_id", self.run_id),
            ("route_id", self.route_id),
            ("scenario_id", self.scenario_id),
            ("arm_id", self.arm_id),
        ):
            if getattr(trigger, field) != expected:
                raise ValueError("%s does not match active HIL identity" % field)

    def _reject_locked(self, reason, sequence_no=None):
        self._rejection_count += 1
        self._last_rejection_reason = str(reason)
        return AdmissionDecision(False, str(reason), sequence_no)

    def record_rejection(self, reason, sequence_no=None):
        """Record a local queue/transport rejection for the next frame audit."""
        with self._lock:
            return self._reject_locked(reason, sequence_no)

    def _admission_reason_locked(self, envelope, result, trigger):
        sequence_no = trigger.sequence_no
        for field, expected in (
            ("run_id", self.run_id),
            ("route_id", self.route_id),
            ("scenario_id", self.scenario_id),
            ("arm_id", self.arm_id),
        ):
            if getattr(trigger, field) != expected:
                prefix = "cross_" if field in ("run_id", "arm_id") else "identity_"
                return prefix + field
        if sequence_no <= self._retired_sequence_floor:
            return "retired_sequence"
        registered = self._triggers.get(sequence_no)
        if registered is None:
            if (
                trigger.source_frame > self._last_registered_source_frame
                or sequence_no > self._last_registered_sequence
            ):
                return "future_or_unknown_trigger"
            return "identity_unknown_trigger"
        if dict(registered) != trigger.to_metadata():
            return "identity_conflict"
        if sequence_no in self._results:
            return "duplicate_sequence"
        try:
            batch_order = tuple(envelope.batch_order)
        except TypeError:
            return "batch_order_mismatch"
        expected_order = {
            COPTV2X_PAYLOAD_SEMANTICS: CODRIVING_BATCH_ORDER,
            _RSU_PAYLOAD_SEMANTICS: ("rsu",),
        }.get(result.payload_semantics)
        if expected_order is None:
            return "payload_contract_semantics"
        if batch_order != expected_order:
            return "batch_order_mismatch"
        if self.timing_mode == "serial_fixed_comm":
            try:
                _counter(result.durations_ns.get("local_processing"), "local_processing")
            except (AttributeError, ValueError):
                return "local_processing_invalid"
        if (
            type(result.remote_input_tensor_bytes) is not int
            or result.remote_input_tensor_bytes != 0
        ):
            return "remote_input_nonzero"
        try:
            artifact_arm = result.artifact_identity.get("arm_id")
        except AttributeError:
            return "payload_contract_invalid"
        if artifact_arm != self.arm_id:
            return "cross_arm_id_artifact"
        return None

    def accept(self, envelope, arrival_ns):
        _counter(arrival_ns, "arrival_ns")
        if type(envelope) is not HilResultEnvelope:
            with self._lock:
                return self._reject_locked("invalid_result_envelope")
        result = envelope.feature_result
        if type(result) is not HilFeatureResult:
            with self._lock:
                return self._reject_locked("invalid_feature_result")
        try:
            trigger = parse_hil_trigger(dict(result.trigger_identity))
        except (TypeError, ValueError):
            with self._lock:
                return self._reject_locked("identity_invalid")
        sequence_no = trigger.sequence_no
        with self._lock:
            reason = self._admission_reason_locked(envelope, result, trigger)
            if reason is not None:
                return self._reject_locked(reason, sequence_no)

        try:
            frozen = _frozen_feature_result(
                result, reuse_immutable=self.timing_mode == "serial_fixed_comm",
                checksum_dedup=self.checksum_dedup,
            )
            frozen_trigger = parse_hil_trigger(dict(frozen.trigger_identity))
            # Preserve the registry's public JSON shape at the validator
            # boundary (for example, widths is a list, not our frozen tuple).
            artifact_identity = _thaw(frozen.artifact_identity)
            if frozen.payload_semantics == COPTV2X_PAYLOAD_SEMANTICS:
                # Keep legacy two-argument contract validators compatible.
                self._contract_validator(artifact_identity, frozen.outputs)
            else:
                self._contract_validator(
                    artifact_identity, frozen.outputs,
                    payload_semantics=frozen.payload_semantics,
                )
        except Exception:
            return self.record_rejection("payload_contract_invalid", sequence_no)
        try:
            actual_hash = (
                verified_feature_payload_sha256(frozen.outputs)
                if self.checksum_dedup else None
            )
            if actual_hash is None:
                actual_hash = hash_feature_payload(frozen.outputs)
        except Exception:
            return self.record_rejection("payload_hash_invalid", sequence_no)
        if frozen.payload_sha256 != actual_hash:
            return self.record_rejection("payload_hash_mismatch", sequence_no)

        with self._lock:
            reason = self._admission_reason_locked(
                envelope, frozen, frozen_trigger
            )
            if reason is not None:
                return self._reject_locked(reason, sequence_no)
            evicted = None
            if len(self._results) >= self.capacity:
                evicted = min(tuple(self._results) + (sequence_no,))
                if evicted == sequence_no:
                    self._retired_sequence_floor = max(
                        self._retired_sequence_floor, sequence_no
                    )
                    decision = self._reject_locked(
                        "buffer_overflow_evict_incoming", sequence_no
                    )
                    return AdmissionDecision(
                        decision.accepted,
                        decision.reason,
                        decision.sequence_no,
                        evicted_sequence_no=sequence_no,
                    )
                del self._results[evicted]
                self._retired_sequence_floor = max(
                    self._retired_sequence_floor, evicted
                )
            self._results[sequence_no] = _ArrivedResult(frozen, arrival_ns)
            return AdmissionDecision(
                True,
                "accepted" if evicted is None else "accepted_evict_lowest_sequence",
                sequence_no,
                evicted_sequence_no=evicted,
            )

    def select_for_fusion(
        self, current_frame, current_sim_time_ns, trigger, selection_ns
    ):
        _counter(current_frame, "current_frame")
        _counter(current_sim_time_ns, "current_sim_time_ns")
        _counter(selection_ns, "selection_ns")
        if type(trigger) is not HilTrigger:
            raise TypeError("trigger must be a validated HilTrigger")
        self._validate_active_trigger(trigger)
        with self._lock:
            if self.timing_mode == "serial_fixed_comm":
                return self._select_serial_locked(
                    current_frame, current_sim_time_ns, trigger
                )
            eligible = [
                (sequence_no, arrived)
                for sequence_no, arrived in self._results.items()
                if int(arrived.result.trigger_identity["source_frame"]) <= current_frame
            ]
            candidate = max(eligible, default=None, key=lambda item: item[0])
            previous_sequence = (
                -1
                if self._selected is None
                else int(self._selected.result.trigger_identity["sequence_no"])
            )
            if candidate is not None and candidate[0] > previous_sequence:
                self._selected = candidate[1]
                reason = "new"
            elif self._rejection_count > self._last_reported_rejection_count:
                reason = "rejected"
            elif self._selected is not None:
                reason = "zoh"
            else:
                reason = "empty"
            self._last_reported_rejection_count = self._rejection_count
            selected = self._selected
            audit = self._audit(
                current_frame,
                current_sim_time_ns,
                trigger,
                selection_ns,
                selected,
                reason,
            )
            return FusionSelection(
                None if selected is None else selected.result,
                _freeze(audit),
            )

    def get_exact_result(self, trigger):
        """Return only the admitted result bound to this exact trigger identity."""
        if type(trigger) is not HilTrigger:
            raise TypeError("trigger must be a validated HilTrigger")
        self._validate_active_trigger(trigger)
        with self._lock:
            arrived = self._results.get(trigger.sequence_no)
            if arrived is None or dict(arrived.result.trigger_identity) != trigger.to_metadata():
                return None
            return arrived.result

    def _select_serial_locked(self, current_frame, current_sim_time_ns, trigger):
        arrived = self._results.get(trigger.sequence_no)
        if arrived is not None and dict(arrived.result.trigger_identity) != trigger.to_metadata():
            arrived = None
        local_ns = None if arrived is None else arrived.result.durations_ns["local_processing"]
        ready_ns = None if local_ns is None else trigger.source_sim_time_ns + local_ns + 100000000
        ready = (arrived is not None and current_frame >= trigger.source_frame
                 and current_sim_time_ns >= ready_ns)
        selected = arrived if ready else None
        reason = "new" if ready else "pending"
        audit = self._audit(current_frame, current_sim_time_ns, trigger, 0, selected, reason)
        # Experimental timing is modeled simulation timing only. Physical wait
        # and transport arrival clocks are deliberately absent from this audit.
        audit.pop("selection_ns")
        audit.pop("selected_result_arrival_ns")
        audit.update({
            "timing_mode": self.timing_mode,
            "fixed_comm_delay_ms": self.fixed_comm_delay_ms,
            "local_processing_ns": local_ns,
            "sim_ready_ns": ready_ns,
            "source_sim_time_ns": trigger.source_sim_time_ns,
            "drop_policy": "serial_exact_no_overlap",
        })
        return FusionSelection(None if selected is None else selected.result, _freeze(audit))

    def _audit(
        self,
        current_frame,
        current_sim_time_ns,
        trigger,
        selection_ns,
        selected,
        reason,
    ):
        if selected is None:
            selected_identity = {}
            artifact_identity = None
            payload_sha256 = None
            artifact_sha256 = None
            remote_input_tensor_bytes = 0
            arrival_ns = None
            aoi_frames = None
            aoi_ms = None
        else:
            selected_identity = dict(selected.result.trigger_identity)
            artifact_identity = _thaw(selected.result.artifact_identity)
            payload_sha256 = selected.result.payload_sha256
            artifact_sha256 = artifact_identity.get("artifact_sha256")
            remote_input_tensor_bytes = selected.result.remote_input_tensor_bytes
            arrival_ns = selected.arrival_ns
            aoi_frames = current_frame - selected_identity["source_frame"]
            aoi_ms = max(
                0.0,
                (current_sim_time_ns - selected_identity["source_sim_time_ns"])
                / 1000000.0,
            )
        return {
            "schema_version": HIL_AUDIT_SCHEMA_VERSION,
            "current_frame": current_frame,
            "current_sim_time_ns": current_sim_time_ns,
            "selection_ns": selection_ns,
            "trigger_identity": trigger.to_metadata(),
            "result_identity": (None if selected is None else selected_identity),
            "batch_order": (["rsu"] if selected is not None and
                            selected.result.payload_semantics == _RSU_PAYLOAD_SEMANTICS
                            else ["ego", "rsu"]),
            "trigger_sequence_no": trigger.sequence_no,
            "trigger_source_frame": trigger.source_frame,
            "selected_result_trigger_identity": (
                None if selected is None else selected_identity
            ),
            "selected_result_sequence_no": selected_identity.get("sequence_no"),
            "selected_result_source_frame": selected_identity.get("source_frame"),
            "selected_result_arrival_ns": arrival_ns,
            "rsu_aoi_frames": aoi_frames,
            "aoi_frames": aoi_frames,
            "rsu_aoi_ms": aoi_ms,
            "selection_reason": reason,
            "rejection_reason": (
                self._last_rejection_reason if reason == "rejected" else None
            ),
            "arm_id": self.arm_id,
            "artifact_identity": artifact_identity,
            "artifact_sha256": artifact_sha256,
            "payload_sha256": payload_sha256,
            "remote_input_tensor_bytes": remote_input_tensor_bytes,
            "drop_policy": self.drop_policy,
            "buffer_capacity": self.capacity,
            "buffer_size": len(self._results),
        }


__all__ = (
    "AdmissionDecision",
    "CoptV2xHilResultBuffer",
    "FusionSelection",
    "HIL_AUDIT_SCHEMA_VERSION",
    "HilResultEnvelope",
)


# The agent imports this module by its top-level name while package-oriented
# callers may use ``team_code.closedloop``. Preserve one exact class identity
# because result-envelope validation intentionally uses ``type(...) is``.
_module = sys.modules[__name__]
sys.modules.setdefault("coptv2x_hil_result_buffer", _module)
sys.modules.setdefault("team_code.closedloop.coptv2x_hil_result_buffer", _module)
