"""Deterministic, order-independent message impairment decisions."""
from __future__ import print_function

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from typing import Dict, Optional, Tuple

try:
    from .ditl_replay_config import BurstConfig, ImpairmentConfig
except ImportError:
    from ditl_replay_config import BurstConfig, ImpairmentConfig


RNG_VERSION = "hmac-sha256-53bit-v1"
_UNIFORM_DENOMINATOR = float(1 << 53)


@dataclass(frozen=True)
class MessageIdentity:
    route_id: str
    stream_id: str
    logical_message_id: str
    sequence_id: int


@dataclass(frozen=True)
class DeliveryPlan:
    copy_index: int
    effective_delay_ns: int
    reorder_delay_ns: int


@dataclass(frozen=True)
class PolicyDecision:
    dropped: bool
    drop_reason: Optional[str]
    delay_clamped: bool
    loss_uniform: float
    duplicate_uniform: float
    burst_state: str
    deliveries: Tuple[DeliveryPlan, ...]


@dataclass(frozen=True)
class _PolicyPreview:
    decision: PolicyDecision
    burst_key: Optional[Tuple[str, str]]
    burst_value: Optional[Tuple[str, int]]


def _canonical_component(value):
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, byteorder="big") + encoded


def _valid_rate(rate):
    return (
        not isinstance(rate, bool)
        and isinstance(rate, (int, float))
        and math.isfinite(float(rate))
        and 0.0 <= float(rate) <= 1.0
    )


def _valid_nonnegative_integer(value):
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_config(config):
    if not isinstance(config, ImpairmentConfig):
        raise TypeError("config must be an ImpairmentConfig")
    for name in ("extra_delay_ns", "jitter_ns", "reorder_extra_ns"):
        if not _valid_nonnegative_integer(getattr(config, name)):
            raise ValueError("%s must be a non-negative integer" % name)
    for name in ("drop_rate", "reorder_rate", "duplicate_rate"):
        if not _valid_rate(getattr(config, name)):
            raise ValueError("%s must be in [0, 1]" % name)
    if not isinstance(config.burst, BurstConfig):
        raise TypeError("burst must be a BurstConfig")
    if not isinstance(config.burst.enabled, bool):
        raise ValueError("burst.enabled must be a boolean")
    for name in ("enter_rate", "exit_rate"):
        if not _valid_rate(getattr(config.burst, name)):
            raise ValueError("burst.%s must be in [0, 1]" % name)


def _canonical_config(config):
    value = {
        "extra_delay_ns": config.extra_delay_ns,
        "jitter_ns": config.jitter_ns,
        "drop_rate": float(config.drop_rate),
        "burst": {
            "enabled": config.burst.enabled,
            "enter_rate": float(config.burst.enter_rate),
            "exit_rate": float(config.burst.exit_rate),
        },
        "reorder_rate": float(config.reorder_rate),
        "reorder_extra_ns": config.reorder_extra_ns,
        "duplicate_rate": float(config.duplicate_rate),
    }
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _rate_selected(rate, uniform):
    return rate >= 1.0 or (rate > 0.0 and uniform < rate)


class DeterministicImpairmentPolicy:
    """Produce stable decisions while retaining only per-stream burst state."""

    def __init__(self, seed: int, config: ImpairmentConfig):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be a signed integer")
        _validate_config(config)
        self._seed_key = str(seed).encode("ascii")
        self._config = config
        self.config_hash = hashlib.sha256(
            _canonical_config(config)
        ).hexdigest()
        self._burst_streams = {}  # type: Dict[Tuple[str, str], Tuple[str, int]]

    def _uniform(self, identity, impairment_name, copy_index=0):
        self._validate_identity(identity)
        if not isinstance(impairment_name, str) or not impairment_name:
            raise ValueError("impairment_name must be a non-empty string")
        if not _valid_nonnegative_integer(copy_index):
            raise ValueError("copy_index must be a non-negative integer")
        components = (
            identity.route_id,
            identity.stream_id,
            identity.logical_message_id,
            str(identity.sequence_id),
            impairment_name,
            str(copy_index),
            RNG_VERSION,
        )
        message = b"".join(_canonical_component(item) for item in components)
        digest = hmac.new(self._seed_key, message, hashlib.sha256).digest()
        first_53_bits = int.from_bytes(digest[:7], byteorder="big") >> 3
        return first_53_bits / _UNIFORM_DENOMINATOR

    @staticmethod
    def _validate_identity(identity):
        if not isinstance(identity, MessageIdentity):
            raise TypeError("identity must be a MessageIdentity")
        for name in ("route_id", "stream_id", "logical_message_id"):
            value = getattr(identity, name)
            if not isinstance(value, str) or not value:
                raise ValueError("%s must be a non-empty string" % name)
        if not _valid_nonnegative_integer(identity.sequence_id):
            raise ValueError("sequence_id must be a non-negative integer")

    def _next_burst_state(self, identity):
        if not self._config.burst.enabled:
            return "disabled", False, None
        key = (identity.route_id, identity.stream_id)
        current, last_sequence = self._burst_streams.get(key, ("good", -1))
        if identity.sequence_id <= last_sequence:
            raise ValueError(
                "burst sequence_id must be strictly increasing per stream"
            )
        if current == "good":
            uniform = self._uniform(identity, "burst_enter", 0)
            if _rate_selected(self._config.burst.enter_rate, uniform):
                current = "bad"
        else:
            uniform = self._uniform(identity, "burst_exit", 0)
            if _rate_selected(self._config.burst.exit_rate, uniform):
                current = "good"
        return current, current == "bad", (key, identity.sequence_id)

    def _preview_decision(self, identity, base_delay_ns):
        self._validate_identity(identity)
        if not _valid_nonnegative_integer(base_delay_ns):
            raise ValueError("base_delay_ns must be a non-negative integer")

        jitter_uniform = self._uniform(identity, "jitter", 0)
        jitter_offset = int(round(
            (2.0 * jitter_uniform - 1.0) * self._config.jitter_ns
        ))
        raw_delay = (
            base_delay_ns + self._config.extra_delay_ns + jitter_offset
        )
        delay_clamped = raw_delay < 0
        message_delay = max(0, raw_delay)

        loss_uniform = self._uniform(identity, "loss", 0)
        bernoulli_drop = _rate_selected(
            self._config.drop_rate, loss_uniform
        )
        burst_state, burst_drop, pending_burst = self._next_burst_state(
            identity
        )
        duplicate_uniform = self._uniform(identity, "duplicate", 0)
        duplicate = _rate_selected(
            self._config.duplicate_rate, duplicate_uniform
        )

        drop_reason = None
        if bernoulli_drop and burst_drop:
            drop_reason = "bernoulli_and_burst"
        elif bernoulli_drop:
            drop_reason = "bernoulli"
        elif burst_drop:
            drop_reason = "burst"

        deliveries = []
        if drop_reason is None:
            copy_count = 2 if duplicate else 1
            for copy_index in range(copy_count):
                reorder_uniform = self._uniform(
                    identity, "reorder", copy_index
                )
                reordered = _rate_selected(
                    self._config.reorder_rate, reorder_uniform
                )
                reorder_delay = (
                    self._config.reorder_extra_ns if reordered else 0
                )
                deliveries.append(DeliveryPlan(
                    copy_index=copy_index,
                    effective_delay_ns=message_delay,
                    reorder_delay_ns=reorder_delay,
                ))

        decision = PolicyDecision(
            dropped=drop_reason is not None,
            drop_reason=drop_reason,
            delay_clamped=delay_clamped,
            loss_uniform=loss_uniform,
            duplicate_uniform=duplicate_uniform,
            burst_state=burst_state,
            deliveries=tuple(deliveries),
        )
        if pending_burst is None:
            return _PolicyPreview(decision, None, None)
        key, sequence_id = pending_burst
        return _PolicyPreview(
            decision, key, (burst_state, sequence_id)
        )

    def _commit_preview(self, preview):
        if preview.burst_key is not None:
            self._burst_streams[preview.burst_key] = preview.burst_value

    def decide(self, identity: MessageIdentity, base_delay_ns: int) -> PolicyDecision:
        preview = self._preview_decision(identity, base_delay_ns)
        self._commit_preview(preview)
        return preview.decision
