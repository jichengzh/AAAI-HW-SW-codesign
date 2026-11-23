"""Serial timing is source-simulation time, never the physical transport wait."""
import os
import sys
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
from coptv2x_hil_client import AsyncHilClient, run_hil_prefusion_forward
from coptv2x_hil_result_buffer import CoptV2xHilResultBuffer, HilResultEnvelope
from coptv2x_hil_backend import hash_feature_payload
from test_coptv2x_hil_client import (
    ARM_ID, FakeTransport, _trigger, _small_result, _small_contract_validator,
)

RSU_SEMANTICS = "codriving_rsu_multiscale_features.v1"


def result(trigger, local_ns=120000000, semantics=None, order=None):
    value = _small_result(trigger)
    outputs = value.outputs
    if semantics == RSU_SEMANTICS:
        outputs = {name: tensor[1:2] for name, tensor in outputs.items()}
    return HilResultEnvelope(replace(
        value, outputs=outputs, payload_sha256=hash_feature_payload(outputs),
        payload_semantics=semantics or value.payload_semantics,
        durations_ns={} if local_ns is None else {"local_processing": local_ns},
    ), order or (("rsu",) if semantics == RSU_SEMANTICS else ("ego", "rsu")))


def buffer(validator=_small_contract_validator):
    return CoptV2xHilResultBuffer(
        "run-live", "Town05-route-01", "scenario-01", ARM_ID,
        capacity=4, contract_validator=validator, timing_mode="serial_fixed_comm",
        fixed_comm_delay_ms=100,
    )


class SerialBufferTests(unittest.TestCase):
    def test_gate_uses_elapsed_local_plus_fixed_comm_not_arrival_or_rounded_frames(self):
        store = buffer()
        trigger = _trigger(0, source_frame=0)
        store.register_trigger(trigger, 1)
        self.assertTrue(store.accept(result(trigger), 900000000000).accepted)
        self.assertIsNotNone(store.get_exact_result(trigger))
        early = store.select_for_fusion(4, 200000000, trigger, 999999999999)
        self.assertIsNone(early.result)
        ready = store.select_for_fusion(5, 250000000, trigger, 999999999999)
        self.assertEqual(0, ready.result.trigger_identity["sequence_no"])
        self.assertEqual(220000000, ready.audit["sim_ready_ns"])
        self.assertEqual(250.0, ready.audit["rsu_aoi_ms"])
        self.assertEqual(100, ready.audit["fixed_comm_delay_ms"])
        self.assertNotIn("selection_ns", ready.audit)
        self.assertNotIn("selected_result_arrival_ns", ready.audit)

    def test_selection_and_getter_are_exact_even_when_newer_result_arrives_first(self):
        store = buffer()
        first, newer = _trigger(0), _trigger(1)
        for trigger in (first, newer):
            store.register_trigger(trigger, 1)
        store.accept(result(newer, 0), 1)
        self.assertIsNone(store.get_exact_result(first))
        store.accept(result(first, 0), 900000000000)
        selected = store.select_for_fusion(10, 500000000, first, 1)
        self.assertEqual(0, selected.result.trigger_identity["sequence_no"])

    def test_local_duration_missing_negative_bool_and_float_are_rejected(self):
        for local in (None, -1, True, 1.5):
            with self.subTest(local=local):
                store, trigger = buffer(), _trigger(0)
                store.register_trigger(trigger, 1)
                decision = store.accept(result(trigger, local), 1)
                self.assertFalse(decision.accepted)
                self.assertEqual("local_processing_invalid", decision.reason)

    def test_rsu_only_requires_matching_semantics_batch_order_and_contract(self):
        seen = []
        def validator(identity, outputs, *, payload_semantics):
            seen.append(payload_semantics)
            self.assertEqual(RSU_SEMANTICS, payload_semantics)
            self.assertTrue(all(t.shape[0] == 1 for t in outputs.values()))
        store, trigger = buffer(validator), _trigger(0)
        store.register_trigger(trigger, 1)
        self.assertFalse(store.accept(result(trigger, semantics=RSU_SEMANTICS,
                                            order=("ego", "rsu")), 1).accepted)
        self.assertTrue(store.accept(result(trigger, semantics=RSU_SEMANTICS), 1).accepted)
        self.assertEqual([RSU_SEMANTICS], seen)
        calls = []
        run_hil_prefusion_forward(lambda data, **kwargs: calls.append(kwargs), {},
                                 store.get_exact_result(trigger).outputs,
                                 payload_semantics=RSU_SEMANTICS)
        self.assertTrue(all(t.shape[0] == 1 for t in calls[0]["hil_remote_multiscale"]))
        with self.assertRaises(ValueError):
            run_hil_prefusion_forward(lambda *a, **kw: None, {},
                                     store.get_exact_result(trigger).outputs)


class SerialClientTests(unittest.TestCase):
    def client(self, transport=None):
        client = AsyncHilClient(transport or FakeTransport(), buffer(), result_timeout_s=.3)
        client.start()
        self.addCleanup(client.shutdown)
        return client

    def test_wait_does_not_return_when_sender_idle_or_another_result_is_received(self):
        client = self.client()
        first, newer = _trigger(0), _trigger(1)
        client.publish_trigger(first)
        client.publish_trigger(newer)
        finished = threading.Event()
        answers = []
        def wait():
            answers.append(client.wait_for_result(first))
            finished.set()
        waiter = threading.Thread(target=wait)
        waiter.start()
        client.transport.deliver(result(newer))
        self.assertFalse(finished.wait(.04))
        client.transport.deliver(result(first))
        self.assertTrue(finished.wait(.3))
        waiter.join()
        self.assertEqual(0, answers[0].trigger_identity["sequence_no"])

    def test_timeout_rejection_sender_failure_and_transport_fatal_propagate(self):
        client = self.client()
        trigger = _trigger(0)
        client.publish_trigger(trigger)
        with self.assertRaises(TimeoutError):
            client.wait_for_result(trigger, timeout_s=.01)
        client.transport.deliver(result(trigger, -1))
        with self.assertRaisesRegex(RuntimeError, "local_processing_invalid"):
            client.wait_for_result(trigger)
        fatal = self.client()
        fatal.publish_trigger(trigger)
        fatal.transport.last_error = ValueError("fatal channel")
        with self.assertRaisesRegex(RuntimeError, "fatal channel"):
            fatal.wait_for_result(trigger)
        class FailingTransport(FakeTransport):
            def publish_trigger(self, payload):
                raise OSError("sender down")
        failed = self.client(FailingTransport())
        failed.publish_trigger(trigger)
        with self.assertRaisesRegex(RuntimeError, "sender down"):
            failed.wait_for_result(trigger)

    def test_serial_does_not_sample_physical_publish_arrival_or_selection_clock(self):
        def forbidden_wall_clock():
            self.fail("serial experimental timing must not sample transport wall clock")
        client = AsyncHilClient(FakeTransport(), buffer(), clock_ns=forbidden_wall_clock)
        client.start()
        self.addCleanup(client.shutdown)
        trigger = _trigger(0)
        self.assertTrue(client.publish_trigger(trigger).accepted)
        client.transport.deliver(result(trigger))
        client.wait_for_result(trigger, .3)
        selected = client.select_for_fusion(5, 250000000, trigger)
        self.assertIsNotNone(selected.result)

    def test_unidentified_malformed_result_propagates_instead_of_waiting_for_timeout(self):
        client = self.client()
        trigger = _trigger(0)
        client.publish_trigger(trigger)
        client.transport.deliver(None)
        with self.assertRaisesRegex(RuntimeError, "invalid_result_envelope"):
            client.wait_for_result(trigger)

    def test_serial_cycle_waits_once_holds_then_consumes_without_overlapping_trigger(self):
        import coptv2x_hil_serial_clock
        client = self.client()
        clock = coptv2x_hil_serial_clock.SerialHilClock(client)
        trigger = _trigger(0)
        def deliver():
            client.transport.publish_entered.wait(.3)
            time.sleep(.04)  # Physical waiting cannot enter the modeled clock.
            client.transport.deliver(result(trigger))
        sender = threading.Thread(target=deliver)
        sender.start()
        source = [{"pose": [1, 2]}]
        clock.begin(trigger, [], source)
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            clock.begin(_trigger(1), [], source)
        source[0]["pose"][0] = 99
        self.assertTrue(clock.pending)
        for frame, now in ((1, 50000000), (2, 100000000), (4, 200000000)):
            self.assertIsNone(clock.consume_if_ready(frame, now))
        self.assertEqual(1, len(client.transport.published))
        selected, collaboration, rsu = clock.consume_if_ready(5, 250000000)
        self.assertEqual(0, selected.result.trigger_identity["sequence_no"])
        self.assertEqual([{"pose": [1, 2]}], rsu)
        self.assertFalse(clock.pending)
        self.assertEqual(1, len(client.transport.published))
        sender.join()

    def test_serial_rejects_raw_rsu_delay_and_replay_instead_of_double_delaying(self):
        from coptv2x_hil_serial_clock import SerialHilClock
        client = self.client()
        for config in ({"rsu_fifo_delay_ms": 100}, {"latency_inject_ms": 100},
                       {"comm_latency": 2}, {"latency_trace": "trace.csv"},
                       {"latency_aware": True}, {"ditl_replay": {}}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "conflicts"):
                SerialHilClock(client, simulation_config=config)

    def test_real_inference_boundary_holds_then_pairs_current_ego_and_old_rsu(self):
        from test_dual_fifo_wiring import INFER, compile_method, method
        from coptv2x_hil_serial_clock import SerialHilClock
        prepare = compile_method(method(INFER, "_prepare_hil_frame"))
        client = self.client()
        config = dict(_trigger(0).to_metadata())
        state = SimpleNamespace(
            _coptv2x_hil_client=client, _coptv2x_hil_sequence=0,
            _coptv2x_hil_config=config, _serial_hil_clock=SerialHilClock(client),
        )
        # The inference boundary creates its own exact sample identity.
        def receive(payload):
            from coptv2x_hil_trigger import parse_hil_trigger
            client.transport.published.append(payload)
            client.transport.deliver(result(parse_hil_trigger(payload)))
        client.transport.publish_trigger = receive
        hold, _, _, _, _ = prepare(state, [{"ego_frame": 0}], [{"rsu_frame": 0}], 0, 0.)
        self.assertTrue(hold)
        hold, _, _, _, _ = prepare(state, [{"ego_frame": 4}], [{"rsu_frame": 4}], 4, .2)
        self.assertTrue(hold)
        hold, trigger, selection, cars, rsu = prepare(
            state, [{"ego_frame": 5}], [{"rsu_frame": 5}], 5, .25)
        self.assertFalse(hold)
        self.assertEqual([{"ego_frame": 5}], cars)
        self.assertEqual([{"rsu_frame": 0}], rsu)
        self.assertEqual(0, trigger.source_frame)
        self.assertEqual(0, selection.result.trigger_identity["source_frame"])
        self.assertEqual(1, len(client.transport.published))
        self.assertEqual(1, state._coptv2x_hil_sequence)

    def test_real_agent_polls_each_tick_pending_and_does_not_use_l1_or_new_trigger(self):
        from test_dual_fifo_wiring import AGENT, INFER, compile_method, method, Control
        from coptv2x_hil_serial_clock import SerialHilClock
        from action_fifo_delay import EgoOutputFifoDelay
        prepare = compile_method(method(INFER, "_prepare_hil_frame"))
        provider = SimpleNamespace(get_hero_actor=lambda **kwargs: object())
        run = compile_method(method(AGENT, "run_step"), CarlaDataProvider=provider)
        client = self.client()
        def receive(payload):
            from coptv2x_hil_trigger import parse_hil_trigger
            client.transport.published.append(payload)
            client.transport.deliver(result(parse_hil_trigger(payload)))
        client.transport.publish_trigger = receive
        clock = SerialHilClock(client)
        trigger = _trigger(0, source_frame=4)
        clock.begin(trigger, [], [{"rsu_frame": 4}])
        old, new = [Control(.2)], [Control(.7)]
        inference = SimpleNamespace(
            _coptv2x_hil_client=client, _coptv2x_hil_sequence=1,
            _coptv2x_hil_config=trigger.to_metadata(), _serial_hil_clock=clock,
            _l1_plan_stores=[object()], prev_control=old,
        )
        fused = []
        def infer_frame(**kwargs):
            hold, _, selection, cars, rsu = prepare(inference, **kwargs)
            if hold:
                return old
            fused.append((kwargs["step"], cars, rsu))
            inference.prev_control = new
            return new
        inference.get_action_from_list_inter = infer_frame
        def l1(*args):
            self.fail("L1 must not override previous control during pending hold")
        inference.run_l1_step = l1
        # Use the production property's behavior on an object with real state.
        class InferProxy:
            @property
            def serial_hil_pending(self):
                return clock.pending
            def __getattr__(self, name):
                return getattr(inference, name)
        state = SimpleNamespace(
            initialized=True, step=4, disable_rsu=True, ego_vehicles_num=1,
            config={"simulation": {"l1_control": True, "skip_frames": 4},
                    "planning": {"core_method": "MotionNet"}},
            _ego_output_fifo=EgoOutputFifoDelay(0), infer=InferProxy(),
        )
        state.tick = lambda *args: {"ego_frame": state.step}
        for now in (.25, .30, .35, .40):
            self.assertEqual(.2, run(state, {}, now)[0].throttle)
        self.assertEqual(.7, run(state, {}, .45)[0].throttle)
        self.assertEqual([(9, [{"ego_frame": 9}], [{"rsu_frame": 4}])], fused)
        self.assertEqual(1, len(client.transport.published))

    def test_serial_startup_consumed_result_waits_for_next_configured_inference_cycle(self):
        from test_dual_fifo_wiring import AGENT, compile_method, method, Control
        from action_fifo_delay import EgoOutputFifoDelay
        provider = SimpleNamespace(get_hero_actor=lambda **kwargs: object())
        run = compile_method(method(AGENT, "run_step"), CarlaDataProvider=provider)
        def premature_inference(**kwargs):
            self.fail("no new inference cycle at startup frame3 between scheduled frames0/4")
        state = SimpleNamespace(
            initialized=True, step=2, disable_rsu=True, ego_vehicles_num=1,
            config={"simulation": {"skip_frames": 4},
                    "planning": {"core_method": "MotionNet"}},
            _ego_output_fifo=EgoOutputFifoDelay(0), tick=lambda *args: {},
            infer=SimpleNamespace(serial_hil_pending=False, _serial_hil_clock=object(),
                                  prev_control=[Control(.7)],
                                  get_action_from_list_inter=premature_inference),
        )
        self.assertEqual(.7, run(state, {}, .15)[0].throttle)


if __name__ == "__main__":
    unittest.main()
