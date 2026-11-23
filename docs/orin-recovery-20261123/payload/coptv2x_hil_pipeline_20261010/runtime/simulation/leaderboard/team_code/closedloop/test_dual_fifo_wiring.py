"""CARLA/GPU-free checks of the actual agent and inference boundaries.

Compile selected production methods/prefixes from AST: heavyweight imports and
model execution are excluded, while the FIFO implementations remain real.
"""

import ast
import copy
import csv
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

if __package__:
    from .action_fifo_delay import EgoOutputFifoDelay
    from .dual_fifo_config import resolve_dual_fifo_config
    from .latency_frame_align import (
        LatencyTraceReader, RsuLatencyAligner, apply_delayed_collaboration,
    )
else:  # Direct discovery from the closedloop directory.
    from action_fifo_delay import EgoOutputFifoDelay
    from dual_fifo_config import resolve_dual_fifo_config
    from latency_frame_align import (
        LatencyTraceReader, RsuLatencyAligner, apply_delayed_collaboration,
    )


TEAM_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEAM_CODE.parent))
sys.path.insert(0, str(TEAM_CODE / "closedloop"))
AGENT = TEAM_CODE / "pnp_agent_e2e.py"
INFER = TEAM_CODE / "pnp_infer_action_e2e.py"
RSU_FIELDS = [
    "consume_frame", "submitted_rsu_source_frame", "submitted_rsu_ready_frame",
    "used_rsu_source_frame", "used_rsu_ready_frame", "configured_delay_ms",
    "delay_frames", "information_age_frames", "is_zoh_held",
    "is_ego_only_startup", "source_sim_time_s", "consume_sim_time_s",
    "step", "delta", "used_rsu_frame_id",
    "latency_ms_source", "zoh_age_frames",
]


def tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def method(path, name):
    class_name = "PnP_Agent" if path == AGENT else "PnP_infer"
    return next(node for cls in tree(path).body if isinstance(cls, ast.ClassDef)
                and cls.name == class_name
                for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return ""


def calls(node, name):
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)
            and dotted(child.func) == name]


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        dotted(target) == name for target in node.targets)


def compile_method(node, body=None, **namespace):
    node = copy.deepcopy(node)
    node.decorator_list = []
    if body is not None:
        node.body = copy.deepcopy(body)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, "<production-boundary>", "exec"), namespace)
    return namespace[node.name]


class Control:
    def __init__(self, throttle=0.0):
        self.throttle = throttle
        self.steer = 0.0
        self.brake = 0.0


class DualFifoWiringTest(unittest.TestCase):
    def test_planning_adapter_reads_shrink_contract_from_hypes_mapping(self):
        node = next(item for item in tree(INFER).body
                    if isinstance(item, ast.FunctionDef)
                    and item.name == "_planner_feature_adapter_config")
        select = compile_method(node)
        contract = {"dim": [128], "input_dim": 384}
        self.assertEqual(contract, select({"model": {"args": {"shrink_header": contract}}}))
        with self.assertRaisesRegex(ValueError, "lack shrink_header"):
            select({"model": {"args": {}}})

    def test_planning_adapter_selects_only_pretrained_shrink_conv_weights(self):
        node = next(item for item in tree(INFER).body
                    if isinstance(item, ast.FunctionDef)
                    and item.name == "_planning_feature_adapter_weights")
        select = compile_method(node)
        sentinel = object()
        selected = select({
            "shrink_conv.layers.0.double_conv.0.weight": sentinel,
            "cls_head.weight": object(),
        })
        self.assertEqual({"layers.0.double_conv.0.weight": sentinel}, selected)

    def test_hil_planning_feature_adapter_center_crops_oversized_grid(self):
        node = next(item for item in tree(INFER).body
                    if isinstance(item, ast.FunctionDef)
                    and item.name == "_fit_planning_feature_shape")
        fit = compile_method(node)
        values = __import__("numpy").arange(1 * 2 * 192 * 128).reshape(1, 2, 192, 128)
        actual = fit(values, 192, 96)
        self.assertEqual((1, 2, 192, 96), actual.shape)
        self.assertTrue((actual == values[:, :, :, 16:112]).all())
        unchanged = fit(values[:, :, :, 16:112], 192, 96)
        self.assertTrue((unchanged == values[:, :, :, 16:112]).all())
        with self.assertRaisesRegex(ValueError, "smaller than planner grid"):
            fit(values[:, :, :, :64], 192, 96)

    def inference_prefix(self):
        node = method(INFER, "get_action_from_list_inter")
        end = next(i for i, stmt in enumerate(node.body)
                   if calls(stmt, "self.check_data"))
        # Return the exact inputs about to enter production preprocessing.
        result = ast.parse("return car_data_raw, rsu_data_raw").body[0]
        run = compile_method(node, node.body[:end] + [result],
                             apply_delayed_collaboration=apply_delayed_collaboration)
        prepare = compile_method(method(INFER, "_prepare_hil_frame"))
        def invoke(state, *args, **kwargs):
            state._prepare_hil_frame = lambda *a, **kw: prepare(state, *a, **kw)
            return run(state, *args, **kwargs)
        return invoke

    def inference_state(self, delay=100, audit=False):
        return SimpleNamespace(
            config={"simulation": {}}, _lat_aligner=RsuLatencyAligner(),
            _trace_reader=None, _latency_inject_ms=delay,
            _latency_source_tag="injected_from_E7", _latency_audit_enabled=audit,
            _lat_audit_rows=[], _dump_latency_audit=lambda: None,
        )

    def test_sources_remain_syntactically_valid(self):
        for path in (AGENT, INFER):
            compile(tree(path), str(path), "exec")

    def test_every_control_exit_is_directly_through_same_final_fifo(self):
        exits = [node for node in ast.walk(method(AGENT, "run_step"))
                 if isinstance(node, ast.Return)]
        self.assertEqual(len(exits), 3)
        for node in exits:
            self.assertIsInstance(node.value, ast.Call)
            self.assertEqual(dotted(node.value.func), "self._ego_output_fifo.release")
            self.assertEqual(dotted(node.value.args[1]), "self.step")

    def test_both_components_resolve_canonical_configuration(self):
        for path, name in ((AGENT, "setup"), (INFER, "__init__")):
            imports = [node for node in tree(path).body
                       if isinstance(node, ast.ImportFrom)
                       and node.module == "team_code.closedloop.dual_fifo_config"]
            self.assertTrue(any(alias.name == "resolve_dual_fifo_config"
                                for node in imports for alias in node.names))
            found = calls(method(path, name), "resolve_dual_fifo_config")
            self.assertEqual(len(found), 1, str(path))
            self.assertEqual(dotted(found[0].args[0]), "sim_cfg")

    def test_delayed_selection_precedes_all_collation_and_fusion(self):
        node = method(INFER, "get_action_from_list_inter")
        selection = calls(node, "apply_delayed_collaboration")
        # The legacy branch keeps its existing application boundary; the
        # mutually exclusive event branch delegates to _apply_ditl_replay().
        self.assertEqual(len(selection), 1)
        for target in ("self.check_data", "self.collate_batch_infer_perception",
                       "self.perception_dataloader.__getitem__",
                       "self.perception_dataloader.collate_batch_test",
                       "inference_utils.inference_intermediate_fusion_multiclass"):
            downstream = calls(node, target)
            self.assertTrue(downstream, target)
            self.assertTrue(all(item.lineno < call.lineno
                                for item in selection for call in downstream))

    def test_hil_missing_rsu_slot_holds_control_before_collation(self):
        class HilClient(object):
            def __init__(self):
                self.published = []
                self.selections = []

            def publish_trigger(self, trigger):
                self.published.append(trigger)

            def select_for_fusion(self, frame, sim_time_ns, trigger):
                self.selections.append((frame, sim_time_ns, trigger))

        client = HilClient()
        control = Control(throttle=0.25)
        state = SimpleNamespace(
            _coptv2x_hil_client=client,
            _coptv2x_hil_config={
                "run_id": "run-live",
                "route_id": "route-1",
                "scenario_id": "scenario-1",
                "arm_id": "joint_shcosearch",
            },
            _coptv2x_hil_sequence=7,
            _lat_aligner=None,
            _trace_reader=None,
            config={"simulation": {}},
            prev_control=control,
        )

        result = self.inference_prefix()(
            state,
            car_data_raw=[{"ego": True}],
            rsu_data_raw=[None],
            step=7,
            timestamp=1.25,
        )

        self.assertIs(result, control)
        self.assertEqual([], client.published)
        self.assertEqual(1, len(client.selections))
        self.assertEqual((7, 1250000000), client.selections[0][:2])
        self.assertEqual(7, client.selections[0][2].source_frame)
        self.assertEqual(7, client.selections[0][2].sequence_no)
        self.assertEqual(7, state._coptv2x_hil_sequence)

    def test_capture_prepass_holds_control_until_route_has_ego_and_rsu(self):
        control = Control(throttle=0.2)
        state = SimpleNamespace(
            _coptv2x_hil_client=None,
            _coptv2x_route_capture=object(),
            config={"simulation": {}},
            prev_control=control,
            _lat_aligner=None,
            _trace_reader=None,
        )
        result = self.inference_prefix()(
            state,
            car_data_raw=[{"ego": True}],
            rsu_data_raw=[None],
            step=1,
            timestamp=0.05,
        )
        self.assertIs(result, control)

    def test_rsu_configuration_accepts_aliases_and_rejects_trace_conflicts(self):
        node = method(INFER, "__init__")
        start = next(i for i, stmt in enumerate(node.body)
                     if assigns(stmt, "self._lat_aligner"))
        end = next(i for i, stmt in enumerate(node.body)
                   if assigns(stmt, "self._l1_plan_stores"))
        initialize = compile_method(
            node, node.body[start:end], resolve_dual_fifo_config=resolve_dual_fifo_config,
            RsuLatencyAligner=RsuLatencyAligner, LatencyTraceReader=LatencyTraceReader)
        for key in ("rsu_fifo_delay_ms", "latency_inject_ms"):
            with self.subTest(key=key):
                state = SimpleNamespace(config={"simulation": {key: 100}})
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    initialize(state, None, 1, None, None, None, None)
                self.assertEqual([item.category for item in seen],
                                 [FutureWarning] if key == "latency_inject_ms" else [])
                self.assertIsNotNone(state._lat_aligner)
                self.assertEqual(state._latency_inject_ms, 100)
                state.config["simulation"]["latency_trace"] = "must-not-be-opened.csv"
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    with self.assertRaises(ValueError):
                        initialize(state, None, 1, None, None, None, None)
                self.assertEqual([item.category for item in seen],
                                 [FutureWarning] if key == "latency_inject_ms" else [])
        state = SimpleNamespace(config={"simulation": {"latency_aware": True}})
        initialize(state, None, 1, None, None, None, None)
        self.assertIsNotNone(state._lat_aligner)
        self.assertIsNone(state._latency_inject_ms)
        self.assertEqual(state._latency_source_tag, "measured_in_carla")
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory, "trace.csv")
            trace.write_text("call_idx,tau_ms\n0,100\n1,0\n")
            state = SimpleNamespace(config={"simulation": {"latency_trace": str(trace)}})
            initialize(state, None, 1, None, None, None, None)
            run = self.inference_prefix()
            self.assertEqual(run(state, ["ego-0"], ["rsu-0"], 0, 0), (["ego-0"], []))
            self.assertEqual(run(state, ["ego-1"], ["rsu-1"], 1, .05),
                             (["ego-1"], ["rsu-1"]))
            self.assertEqual(state._latency_source_tag, "trace_replay_codriving_4090")

    def test_startup_and_missing_slots_never_leak_current_collaboration(self):
        run = self.inference_prefix()
        self.assertEqual(run(self.inference_state(), ["ego"], ["rsu"], 0, 0),
                         (["ego"], []))
        state = self.inference_state()
        current_cars = ["ego-0", "cav-0"]
        current_rsus = ["rsu-0"]
        self.assertEqual(run(state, current_cars, current_rsus, 0, 0),
                         (["ego-0", None], []))
        self.assertEqual(run(state, ["ego-1", "cav-1"], ["rsu-1"], 1, .05),
                         (["ego-1", None], []))
        self.assertEqual(run(state, ["ego-2", "cav-2", "new-cav"], ["rsu-2"], 2, .1),
                         (["ego-2", "cav-0", None], ["rsu-0"]))
        self.assertEqual(current_cars, ["ego-0", "cav-0"])
        self.assertEqual(current_rsus, ["rsu-0"])

    def test_hil_frame_without_local_rsu_fails_closed_before_batch2_model(self):
        class HilClient:
            def __init__(self):
                self.selected = []

            def publish_trigger(self, trigger):
                return SimpleNamespace(accepted=True, reason="queued")

            def select_for_fusion(self, frame, sim_time_ns, trigger):
                self.selected.append((frame, sim_time_ns, trigger.sequence_no))
                return SimpleNamespace(result=None)

        client = HilClient()
        state = SimpleNamespace(
            config={"simulation": {}},
            _coptv2x_hil_client=client,
            _coptv2x_hil_sequence=0,
            _coptv2x_hil_config={
                "run_id": "run",
                "route_id": "route",
                "scenario_id": "scenario",
                "arm_id": "joint_shcosearch",
            },
            _lat_aligner=None,
            prev_control=["held-control"],
        )
        run = self.inference_prefix()

        with mock.patch.object(sys, "path", [str(TEAM_CODE.parent)] + sys.path):
            result = run(state, ["ego"], [], 0, 0.0)

        self.assertEqual(["held-control"], result)
        self.assertEqual([(0, 0, 0)], client.selected)

    def test_zero_delay_and_disabled_aligner_preserve_current_inputs(self):
        run = self.inference_prefix()
        state = self.inference_state(delay=0)
        self.assertEqual(run(state, ["ego", "cav"], ["rsu"], 0, 0),
                         (["ego", "cav"], ["rsu"]))
        state._lat_aligner = None
        self.assertEqual(run(state, ["ego", "cav"], ["rsu"], 1, .05),
                         (["ego", "cav"], ["rsu"]))

    def test_rsu_audit_is_opt_in_and_writes_fixed_v2_columns(self):
        run = self.inference_prefix()
        state = self.inference_state()
        run(state, ["ego"], ["rsu"], 0, 0)
        self.assertEqual(state._lat_audit_rows, [])
        with tempfile.TemporaryDirectory() as directory:
            state.save_path = directory
            Path(directory, "meta").mkdir()
            dump = compile_method(method(INFER, "_dump_latency_audit"), os=os)
            dump(state)
            path = Path(directory, "meta", "latency_align_audit.csv")
            self.assertFalse(path.exists())
            state._latency_audit_enabled = True
            state._dump_latency_audit = lambda: dump(state)
            run(state, ["ego"], ["rsu"], 1, .05)
            with path.open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], RSU_FIELDS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1][0:3], ["1", "1", "3"])

    def test_inference_l1_and_fallback_share_one_temporal_queue(self):
        self.assertTrue(calls(method(AGENT, "run_step"), "self._ego_output_fifo.release"))
        provider = SimpleNamespace(get_hero_actor=lambda **kwargs: object())
        run = compile_method(method(AGENT, "run_step"), CarlaDataProvider=provider)
        generated = [Control(.2)]
        l1 = [Control(.6)]
        fallback = [Control(.9)]
        state = SimpleNamespace(
            initialized=True, step=3, disable_rsu=True, ego_vehicles_num=1,
            config={"simulation": {"l1_control": True, "skip_frames": 4},
                    "planning": {"core_method": "MotionNet"}},
            tick=lambda *args: {"sensor": "current"},
            _ego_output_fifo=EgoOutputFifoDelay(100),
            infer=SimpleNamespace(
                _l1_plan_stores=[object()], prev_control=fallback,
                get_action_from_list_inter=lambda **kwargs: generated,
                run_l1_step=lambda *args: l1),
        )
        self.assertEqual(run(state, {}, .2)[0].throttle, 0)
        self.assertEqual(run(state, {}, .25)[0].throttle, 0)
        state.infer._l1_plan_stores = None
        self.assertEqual(run(state, {}, .3)[0].throttle, .2)
        self.assertEqual(run(state, {}, .35)[0].throttle, .6)
        self.assertEqual(run(state, {}, .4)[0].throttle, .9)
        self.assertEqual([row["apply_frame"] for row in state._ego_output_fifo.audit_rows],
                         [4, 5, 6, 7, 8])

    def test_agent_initialization_selects_canonical_and_legacy_audit_files(self):
        node = method(AGENT, "setup")
        initialization = calls(node, "EgoOutputFifoDelay")
        self.assertEqual(len(initialization), 1)
        # Execute the contiguous, model-free FIFO setup after inference creation.
        start = next(i for i, stmt in enumerate(node.body)
                     if assigns(stmt, "ego_delay_ms"))
        initialize = compile_method(node, node.body[start:],
                                    EgoOutputFifoDelay=EgoOutputFifoDelay, os=os)
        for simulation, filename in (
            ({"ego_output_fifo_delay_ms": 100, "latency_audit": True},
             "ego_output_fifo_audit.csv"),
            ({"action_fifo_delay_ms": 100, "action_fifo_audit": True},
             "action_fifo_audit.csv"),
            ({"ego_output_fifo_delay_ms": 100}, None),
        ):
            with self.subTest(simulation=simulation), tempfile.TemporaryDirectory() as directory:
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    config = resolve_dual_fifo_config(simulation)
                self.assertEqual([item.category for item in seen],
                                 [FutureWarning] if "action_fifo_delay_ms" in simulation else [])
                state = SimpleNamespace(_dual_fifo_config=config,
                                        infer=SimpleNamespace(save_path=directory))
                initialize(state, None, 1)
                self.assertEqual(state._ego_output_fifo.delay_frames, 2)
                state._ego_output_fifo.release([Control(.7)], 0)
                destroy = compile_method(method(AGENT, "destroy"))
                state.perception_model = state.planning_model = None
                destroy(state)
                found = list(Path(directory).rglob("*.csv"))
                self.assertEqual([path.name for path in found], [filename] if filename else [])
                if filename:
                    with found[0].open() as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
