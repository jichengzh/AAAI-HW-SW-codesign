from __future__ import print_function

from dataclasses import FrozenInstanceError
import os
import sys
import unittest
from unittest import mock


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ditl_impairment_policy import (  # noqa: E402
    RNG_VERSION,
    DeliveryPlan,
    DeterministicImpairmentPolicy,
    MessageIdentity,
    PolicyDecision,
)
from ditl_replay_config import BurstConfig, ImpairmentConfig  # noqa: E402


def make_config(**overrides):
    values = {
        "extra_delay_ns": 0,
        "jitter_ns": 0,
        "drop_rate": 0.0,
        "burst": BurstConfig(False, 0.0, 1.0),
        "reorder_rate": 0.0,
        "reorder_extra_ns": 0,
        "duplicate_rate": 0.0,
    }
    values.update(overrides)
    return ImpairmentConfig(**values)


def make_policy(seed=20260923, **overrides):
    return DeterministicImpairmentPolicy(seed, make_config(**overrides))


def identity(number=7, route="route-1", stream="rsu-1"):
    return MessageIdentity(route, stream, "message-%d" % number, number)


class DeterministicImpairmentPolicyTest(unittest.TestCase):
    def test_public_models_are_frozen_and_rng_version_is_pinned(self):
        self.assertEqual("hmac-sha256-53bit-v1", RNG_VERSION)
        message = identity()
        delivery = DeliveryPlan(0, 85, 0)
        decision = PolicyDecision(
            False, None, False, 0.25, 0.75, "disabled", (delivery,)
        )
        with self.assertRaises(FrozenInstanceError):
            message.sequence_id = 8
        with self.assertRaises(FrozenInstanceError):
            delivery.effective_delay_ns = 86
        with self.assertRaises(FrozenInstanceError):
            decision.dropped = True

    def test_frozen_uniform_and_config_hash_vectors_cover_signed_seed(self):
        positive = make_policy(seed=20260923)
        negative = make_policy(seed=-20260923)

        self.assertAlmostEqual(
            0.19428189810337382,
            positive._uniform(identity(), "loss", 0),
        )
        self.assertAlmostEqual(
            0.869553088604487,
            negative._uniform(identity(), "loss", 0),
        )
        self.assertEqual(
            "126efc272f80e58403de65711173b7cc9ad90a80479b31359972d0ebc8c93e94",
            positive.config_hash,
        )

    def test_identical_inputs_produce_identical_byte_stable_decisions(self):
        config = make_config(
            extra_delay_ns=10,
            jitter_ns=7,
            drop_rate=0.2,
            reorder_rate=0.5,
            reorder_extra_ns=11,
            duplicate_rate=1.0,
        )
        first = DeterministicImpairmentPolicy(41, config).decide(
            identity(), 85
        )
        second = DeterministicImpairmentPolicy(41, config).decide(
            identity(), 85
        )

        self.assertEqual(first, second)
        self.assertEqual(
            PolicyDecision(
                dropped=False,
                drop_reason=None,
                delay_clamped=False,
                loss_uniform=0.5983253408953352,
                duplicate_uniform=0.9132185874224704,
                burst_state="disabled",
                deliveries=(
                    DeliveryPlan(0, 93, 0),
                    DeliveryPlan(1, 93, 11),
                ),
            ),
            first,
        )

    def test_changed_seed_changes_uniform_vector_and_not_config_hash(self):
        first = make_policy(seed=1)
        second = make_policy(seed=2)
        first_vector = tuple(
            first._uniform(identity(i), "loss", 0) for i in range(100)
        )
        second_vector = tuple(
            second._uniform(identity(i), "loss", 0) for i in range(100)
        )
        self.assertNotEqual(first_vector, second_vector)
        self.assertEqual(first.config_hash, second.config_hash)

    def test_config_hash_changes_with_configuration_and_is_canonical(self):
        base = make_policy(seed=1)
        equal = DeterministicImpairmentPolicy(1, make_config())
        changed = make_policy(seed=1, extra_delay_ns=1)
        self.assertEqual(base.config_hash, equal.config_hash)
        self.assertNotEqual(base.config_hash, changed.config_hash)
        self.assertRegex(base.config_hash, r"^[0-9a-f]{64}$")

    def test_loss_decision_is_independent_of_optional_duplicate_decision(self):
        message = identity()
        base = make_policy(duplicate_rate=0.0).decide(message, 85000000)
        duplicate = make_policy(duplicate_rate=1.0).decide(
            message, 85000000
        )
        self.assertEqual(base.dropped, duplicate.dropped)
        self.assertEqual(base.loss_uniform, duplicate.loss_uniform)

    def test_non_burst_decisions_are_independent_of_call_order(self):
        config = make_config(
            jitter_ns=25,
            drop_rate=0.45,
            duplicate_rate=0.5,
            reorder_rate=0.5,
            reorder_extra_ns=13,
        )
        forward_policy = DeterministicImpairmentPolicy(17, config)
        reverse_policy = DeterministicImpairmentPolicy(17, config)
        messages = tuple(identity(i) for i in range(20))
        forward = {
            item: forward_policy.decide(item, 50) for item in messages
        }
        reverse = {
            item: reverse_policy.decide(item, 50)
            for item in reversed(messages)
        }
        self.assertEqual(forward, reverse)

    def test_rate_boundaries_are_exact_even_at_uniform_boundaries(self):
        with mock.patch.object(
                DeterministicImpairmentPolicy, "_uniform", return_value=0.0):
            keep = make_policy(
                drop_rate=0.0,
                duplicate_rate=0.0,
                reorder_rate=0.0,
            ).decide(identity(), 10)
        self.assertFalse(keep.dropped)
        self.assertEqual((DeliveryPlan(0, 10, 0),), keep.deliveries)

        with mock.patch.object(
                DeterministicImpairmentPolicy,
                "_uniform",
                return_value=0.9999999999999999):
            impaired = make_policy(
                drop_rate=1.0,
                duplicate_rate=1.0,
                reorder_rate=1.0,
                reorder_extra_ns=5,
            ).decide(identity(), 10)
        self.assertTrue(impaired.dropped)
        self.assertEqual("bernoulli", impaired.drop_reason)
        self.assertEqual((), impaired.deliveries)

        with mock.patch.object(
                DeterministicImpairmentPolicy,
                "_uniform",
                return_value=0.9999999999999999):
            duplicated = make_policy(
                drop_rate=0.0,
                duplicate_rate=1.0,
                reorder_rate=1.0,
                reorder_extra_ns=5,
            ).decide(identity(), 10)
        self.assertEqual(
            (DeliveryPlan(0, 10, 5), DeliveryPlan(1, 10, 5)),
            duplicated.deliveries,
        )

    def test_negative_jitter_clamps_effective_delay_to_zero(self):
        policy = make_policy(extra_delay_ns=0, jitter_ns=100)
        with mock.patch.object(policy, "_uniform", return_value=0.0):
            decision = policy.decide(identity(1), 50)
        self.assertEqual(0, decision.deliveries[0].effective_delay_ns)
        self.assertTrue(decision.delay_clamped)

    def test_duplicate_expands_one_extra_copy_with_per_copy_reorder(self):
        policy = make_policy(
            duplicate_rate=1.0,
            reorder_rate=0.5,
            reorder_extra_ns=20,
        )

        def uniforms(_identity, name, copy_index=0):
            values = {
                ("loss", 0): 0.9,
                ("jitter", 0): 0.5,
                ("duplicate", 0): 0.1,
                ("reorder", 0): 0.1,
                ("reorder", 1): 0.9,
            }
            return values[(name, copy_index)]

        with mock.patch.object(policy, "_uniform", side_effect=uniforms):
            decision = policy.decide(identity(), 50)
        self.assertEqual(
            (DeliveryPlan(0, 50, 20), DeliveryPlan(1, 50, 0)),
            decision.deliveries,
        )

    def test_burst_entry_recovery_and_logical_or_drop_reasons(self):
        policy = make_policy(
            drop_rate=0.5,
            burst=BurstConfig(True, 0.5, 0.5),
        )
        table = {
            (1, "loss"): 0.9,
            (1, "burst_enter"): 0.1,
            (2, "loss"): 0.1,
            (2, "burst_exit"): 0.9,
            (3, "loss"): 0.9,
            (3, "burst_exit"): 0.1,
        }

        def uniforms(message, name, copy_index=0):
            if name in ("duplicate", "jitter", "reorder"):
                return 0.9 if name != "jitter" else 0.5
            return table[(message.sequence_id, name)]

        with mock.patch.object(policy, "_uniform", side_effect=uniforms):
            entered = policy.decide(identity(1), 10)
            held = policy.decide(identity(2), 10)
            recovered = policy.decide(identity(3), 10)

        self.assertEqual((True, "burst", "bad"), (
            entered.dropped, entered.drop_reason, entered.burst_state
        ))
        self.assertEqual((True, "bernoulli_and_burst", "bad"), (
            held.dropped, held.drop_reason, held.burst_state
        ))
        self.assertEqual((False, None, "good"), (
            recovered.dropped, recovered.drop_reason, recovered.burst_state
        ))

    def test_burst_state_is_isolated_per_route_and_stream(self):
        policy = make_policy(burst=BurstConfig(True, 1.0, 0.0))
        first_a = policy.decide(identity(1, route="a", stream="rsu"), 10)
        first_b = policy.decide(identity(1, route="b", stream="rsu"), 10)
        first_c = policy.decide(identity(1, route="a", stream="cav"), 10)
        self.assertEqual("bad", first_a.burst_state)
        self.assertEqual("bad", first_b.burst_state)
        self.assertEqual("bad", first_c.burst_state)

    def test_burst_requires_strictly_increasing_sequence_per_stream(self):
        policy = make_policy(burst=BurstConfig(True, 0.0, 1.0))
        policy.decide(identity(2), 10)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            policy.decide(identity(2), 10)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            policy.decide(identity(1), 10)

    def test_policy_never_mutates_identity_or_config(self):
        burst = BurstConfig(False, 0.1, 0.8)
        config = make_config(burst=burst, duplicate_rate=1.0)
        message = identity()
        before = (message, config)
        make_policy(burst=burst, duplicate_rate=1.0).decide(message, 5)
        self.assertEqual(before, (message, config))

    def test_invalid_policy_inputs_fail_before_hmac_or_state_changes(self):
        policy = make_policy(burst=BurstConfig(True, 1.0, 0.0))
        invalid_identities = (
            MessageIdentity("", "rsu", "message", 1),
            MessageIdentity("route", "", "message", 1),
            MessageIdentity("route", "rsu", "", 1),
            MessageIdentity("route", "rsu", "message", -1),
            MessageIdentity("route", "rsu", "message", True),
        )
        for message in invalid_identities:
            with self.subTest(message=message):
                with self.assertRaises((TypeError, ValueError)):
                    policy.decide(message, 10)
        for value in (-1, True, 1.5):
            with self.subTest(base_delay_ns=value):
                with self.assertRaises((TypeError, ValueError)):
                    policy.decide(identity(), value)

    def test_constructor_rejects_invalid_seed_and_direct_config_values(self):
        for seed in (True, 1.5, "1"):
            with self.subTest(seed=seed):
                with self.assertRaises(TypeError):
                    DeterministicImpairmentPolicy(seed, make_config())
        with self.assertRaises(TypeError):
            DeterministicImpairmentPolicy(1, object())
        with self.assertRaisesRegex(TypeError, "burst must be a BurstConfig"):
            DeterministicImpairmentPolicy(1, make_config(burst=None))

        invalid = (
            {"extra_delay_ns": -1},
            {"jitter_ns": True},
            {"reorder_extra_ns": 1.5},
            {"drop_rate": -0.1},
            {"reorder_rate": float("nan")},
            {"duplicate_rate": 1.1},
            {"burst": BurstConfig("yes", 0.0, 1.0)},
            {"burst": BurstConfig(True, -0.1, 1.0)},
            {"burst": BurstConfig(True, 0.0, float("inf"))},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    DeterministicImpairmentPolicy(1, make_config(**overrides))


if __name__ == "__main__":
    unittest.main()
