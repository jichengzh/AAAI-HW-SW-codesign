"""CPU-only tests for the opt-in DitL boundary in ``PnP_infer``."""

import ast
from collections import OrderedDict
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

from opencood.utils.ditl_spatial_offset import (
    SpatialOffsetConfig, apply_rsu_spatial_offset,
)
from opencood.utils.pose_utils import add_noise_data_dict_asymmetric
from opencood.utils.transformation_utils import (
    get_pairwise_transformation_asymmetric,
)

if __package__:
    from .ditl_audit import DitlAuditError
    from .ditl_event_scheduler import (
        DitlEventScheduler, DitlReplayEngine, ReplaySubmission,
    )
    from .ditl_impairment_policy import DeterministicImpairmentPolicy
    from .ditl_replay_config import BurstConfig, ImpairmentConfig
    from .ditl_replay_schema import ReplaySample
    from .ditl_trace_library import ReplaySampleCursor
    from .latency_frame_align import apply_delayed_collaboration
else:  # Direct discovery from the closedloop directory.
    from ditl_audit import DitlAuditError
    from ditl_event_scheduler import (
        DitlEventScheduler, DitlReplayEngine, ReplaySubmission,
    )
    from ditl_impairment_policy import DeterministicImpairmentPolicy
    from ditl_replay_config import BurstConfig, ImpairmentConfig
    from ditl_replay_schema import ReplaySample
    from ditl_trace_library import ReplaySampleCursor
    from latency_frame_align import apply_delayed_collaboration


INFER = Path(__file__).resolve().parent.parent / "pnp_infer_action_e2e.py"
AGENT = Path(__file__).resolve().parent.parent / "pnp_agent_e2e.py"
DATASET = (Path(__file__).resolve().parents[4] / "opencood" / "data_utils" /
           "datasets" / "intermediate_multiclass_fusion_dataset.py")


def tree(path=INFER):
    return ast.parse(path.read_text(), filename=str(path))


def method(name, path=INFER, class_name="PnP_infer"):
    return next(
        node
        for cls in tree(path).body
        if isinstance(cls, ast.ClassDef) and cls.name == class_name
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def nested_method(name, path=DATASET):
    return next(
        node for node in ast.walk(tree(path))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return ""


def calls(node, name):
    return [
        child for child in ast.walk(node)
        if isinstance(child, ast.Call) and dotted(child.func) == name
    ]


def string_literals(node):
    return {
        child.s if isinstance(child, ast.Str) else child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Str) or (
            isinstance(child, ast.Constant) and isinstance(child.value, str)
        )
    }


def compile_method(node, body=None, **namespace):
    copied = copy.deepcopy(node)
    copied.decorator_list = []
    if body is not None:
        copied.body = copy.deepcopy(body)
    module = ast.fix_missing_locations(ast.Module(body=[copied], type_ignores=[]))
    exec(compile(module, "<ditl-production-boundary>", "exec"), namespace)
    return namespace[copied.name]


def compile_dataset_spatial_prefix():
    item = nested_method("__getitem_single_class__")
    pairwise_call = calls(item, "get_pairwise_transformation_asymmetric")[0]
    end = next(
        index for index, statement in enumerate(item.body)
        if pairwise_call in list(ast.walk(statement))
    )
    returned = ast.Return(value=ast.Tuple(
        elts=[ast.Name(id=name, ctx=ast.Load()) for name in (
            "base_data_dict", "pairwise_t_matrix",
            "lidar_pose_clean_list", "spatial_audit",
        )],
        ctx=ast.Load(),
    ))
    return compile_method(
        item, item.body[:end + 1] + [returned],
        OrderedDict=OrderedDict,
        add_noise_data_dict_asymmetric=add_noise_data_dict_asymmetric,
        apply_rsu_spatial_offset=apply_rsu_spatial_offset,
        get_pairwise_transformation_asymmetric=(
            get_pairwise_transformation_asymmetric
        ),
        math=math,
    )


def dataset_base():
    return OrderedDict((
        ("car_0", {
            "ego": True,
            "params": {
                "lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "map_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "lidar_pose_clean": [0.0] * 6,
            },
            "scene_dict": None,
            "frame_id": None,
            "det_data": np.zeros((1, 7)),
            "detmap_pose": np.zeros(3),
        }),
        ("rsu_0", {
            "ego": False,
            "params": {
                "lidar_pose": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "map_pose": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "lidar_pose_clean": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "detmap_pose": np.ones(3),
        }),
    ))


def run_dataset_four_pass(extra_source, apply_spy):
    current_tpe = [None]
    reference_poses = []

    def retrieve_base_data(idx=None, tpe=None, extra_source=None,
                           data_dir=None):
        current_tpe[0] = tpe
        return copy.deepcopy(dataset_base())

    def get_item_single_car(selected, ego, tpe, cav_id, online_eval):
        pose_x = selected["params"]["lidar_pose"][0]
        reference_poses.append((tpe, cav_id, tuple(
            selected["params"]["lidar_pose"]
        )))
        box = np.array([[pose_x, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
        return {
            "object_bbx_center": box,
            "object_ids": [cav_id],
            "single_label_dict": {},
            "single_object_bbx_center": box.copy(),
            "single_object_bbx_mask": np.ones(1),
        }

    def tracked_apply(base_data, config):
        return apply_spy(current_tpe[0], base_data, config)

    single = compile_method(
        nested_method("__getitem_single_class__"),
        OrderedDict=OrderedDict,
        add_noise_data_dict_asymmetric=add_noise_data_dict_asymmetric,
        apply_rsu_spatial_offset=tracked_apply,
        get_pairwise_transformation_asymmetric=(
            get_pairwise_transformation_asymmetric
        ),
        math=math,
        np=np,
        torch=torch,
    )
    outer = compile_method(
        nested_method("__getitem__"), np=np, torch=torch,
    )
    state = SimpleNamespace(
        retrieve_base_data=retrieve_base_data,
        get_item_single_car=get_item_single_car,
        params={
            "noise_setting": {"add_noise": False},
            "comm_range": 1000.0,
            "postprocess": {"max_num": 8},
        },
        visualize=False,
        kd_flag=False,
        box_align=False,
        max_cav=2,
        proj_first=False,
        heterogeneous=False,
        load_lidar_file=False,
        load_camera_file=False,
        anchor_box=np.zeros((1, 7)),
        supervise_single=True,
    )
    setattr(state, "__getitem_single_class__", single.__get__(state))
    return outer(state, None, extra_source), tuple(reference_poses)


class FakeEngine:
    def __init__(self, payload=None):
        self.payload = payload
        self.submissions = []
        self.advances = []

    def submit(self, submission):
        self.submissions.append(submission)
        return "submission-outcome"

    def advance(self, frame, sim_time_ns):
        self.advances.append((frame, sim_time_ns))
        payload = self.payload
        selection = SimpleNamespace(
            payload=payload,
            materialize_payload=lambda: copy.deepcopy(payload),
        )
        return SimpleNamespace(streams={"rsu-1": selection})


def real_engine():
    sample = ReplaySample(
        sample_id="sample-0",
        ordinal=0,
        source_trace_id="capture-0",
        source_sequence_id=0,
        base_delay_ns=0,
        stage_ns={
            "acquisition_preprocess": 0,
            "gpu_inference": 0,
            "packing_postprocess": 0,
            "transport_receive": 0,
        },
        raw_payload_bytes=128,
        wire_payload_bytes=128,
        capture_condition_id="condition-0",
    )
    impairment = ImpairmentConfig(
        extra_delay_ns=0,
        jitter_ns=0,
        drop_rate=0.0,
        burst=BurstConfig(False, 0.0, 1.0),
        reorder_rate=0.0,
        reorder_extra_ns=0,
        duplicate_rate=0.0,
    )
    return DitlReplayEngine(
        ReplaySampleCursor((sample,), "cycle"),
        DeterministicImpairmentPolicy(20260925, impairment),
        DitlEventScheduler(4, None),
        ("rsu-1",),
        None,
        None,
    )


class DitlReplayWiringTest(unittest.TestCase):
    def apply_method(self):
        return compile_method(
            method("_apply_ditl_replay"),
            ReplaySubmission=ReplaySubmission,
            apply_delayed_collaboration=apply_delayed_collaboration,
            build_audit_record=lambda tick, submission, stream, context: (
                tick, submission, stream, context
            ),
            math=math,
        )

    @staticmethod
    def state(payload=None, writer=None):
        return SimpleNamespace(
            _ditl_replay_engine=FakeEngine(payload),
            _ditl_route_id="route-a",
            _ditl_audit_context=object(),
            _ditl_audit_writer=writer or mock.Mock(),
            _ditl_replay_submission_type=ReplaySubmission,
            _ditl_build_audit_record=lambda tick, submission, stream, context: (
                tick, submission, stream, context
            ),
        )

    def test_replay_imports_are_lazy_and_validation_precedes_legacy_aligner(self):
        parsed = tree()
        forbidden = {
            "team_code.closedloop.ditl_replay_config",
            "team_code.closedloop.ditl_trace_library",
            "team_code.closedloop.ditl_event_scheduler",
            "team_code.closedloop.ditl_impairment_policy",
            "team_code.closedloop.ditl_audit",
        }
        self.assertFalse(any(
            isinstance(node, ast.ImportFrom) and node.module in forbidden
            for node in parsed.body
        ))
        init = method("__init__")
        resolve = calls(init, "resolve_ditl_replay_config")
        aligner = calls(init, "RsuLatencyAligner")
        self.assertEqual(len(resolve), 1)
        self.assertTrue(aligner)
        self.assertLess(resolve[0].lineno, min(call.lineno for call in aligner))
        parent = next(node for node in ast.walk(init)
                      if isinstance(node, ast.If) and resolve[0] in list(ast.walk(node)))
        self.assertIn("ditl_replay", string_literals(parent.test))

    def test_application_is_before_both_check_data_calls_and_mutually_exclusive(self):
        action = method("get_action_from_list_inter")
        replay = calls(action, "self._apply_ditl_replay")
        checks = calls(action, "self.check_data")
        self.assertEqual(len(replay), 1)
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(replay[0].lineno < item.lineno for item in checks))
        replay_if = next(node for node in ast.walk(action)
                         if isinstance(node, ast.If) and replay[0] in list(ast.walk(node)))
        self.assertTrue(replay_if.orelse)

    def test_startup_drop_or_expiry_keeps_ego_and_clears_collaboration(self):
        state = self.state(payload=None)
        cars = [{"ego": 1}, {"current": 2}]
        rsus = [{"current_rsu": 3}]
        delayed_cars, delayed_rsus = self.apply_method()(state, cars, rsus, 4, .2)
        self.assertEqual(delayed_cars, [{"ego": 1}, None])
        self.assertEqual(delayed_rsus, [])
        self.assertEqual(cars, [{"ego": 1}, {"current": 2}])
        self.assertEqual(rsus, [{"current_rsu": 3}])

    def test_selected_and_zoh_payload_replace_only_collaboration(self):
        payload = {"car_data": [None, {"held": 2}],
                   "rsu_data": [{"held_rsu": 3}]}
        payload_before = copy.deepcopy(payload)
        cars = [{"ego": 1}, {"current": 2}]
        rsus = [{"current_rsu": 3}]
        cars_before = copy.deepcopy(cars)
        rsus_before = copy.deepcopy(rsus)
        state = self.state(payload=payload)
        result = self.apply_method()(
            state, cars, rsus, 5, .25
        )
        self.assertEqual(result, ([{"ego": 1}, {"held": 2}], [{"held_rsu": 3}]))
        self.assertEqual(payload, payload_before)
        self.assertEqual(cars, cars_before)
        self.assertEqual(rsus, rsus_before)

    def test_real_selection_payload_reaches_check_data_as_independent_native_data(self):
        engine = real_engine()
        state = self.state()
        state._ditl_replay_engine = engine
        image = Image.new("L", (1, 1), color=7)
        array = np.array([1, 2], dtype=np.int64)
        cars = [{"ego": 1}, {"nested": {"source": "car"}}]
        rsus = [{
            "measurements": {"command": 2},
            "nested": {"items": [1, 2], "shape": (1, 2)},
            "array": array,
            "image": image,
        }]

        delayed_cars, delayed_rsus = self.apply_method()(
            state, cars, rsus, 0, 0.0
        )
        preprocess_inputs = []

        def preprocess(value, car=True):
            preprocess_inputs.append(value)
            value["measurements"]["command"] = -1
            value["array"][0] = 99
            value["image"].putpixel((0, 0), 9)
            return value

        state.preprocess_data = preprocess
        check_data = compile_method(method("check_data"), copy=copy)
        processed, mask = check_data(state, delayed_rsus, car=False)

        self.assertEqual([1], mask)
        self.assertEqual(1, len(preprocess_inputs))
        self.assertIsInstance(delayed_rsus[0], dict)
        self.assertIsInstance(delayed_rsus[0]["nested"], dict)
        self.assertIsInstance(delayed_rsus[0]["nested"]["items"], tuple)
        self.assertIsInstance(delayed_rsus[0]["nested"]["shape"], tuple)
        self.assertEqual(-1, processed[0]["measurements"]["command"])
        self.assertEqual(99, processed[0]["array"][0])
        self.assertEqual(9, processed[0]["image"].getpixel((0, 0)))
        self.assertEqual({"ego": 1}, delayed_cars[0])
        self.assertEqual(2, rsus[0]["measurements"]["command"])
        self.assertEqual(1, array[0])
        self.assertEqual(7, image.getpixel((0, 0)))

        delayed_rsus[0]["measurements"]["command"] = 8
        delayed_rsus[0]["array"][0] = 8
        delayed_rsus[0]["image"].putpixel((0, 0), 8)

        held = engine.advance(1, 1).streams["rsu-1"]
        self.assertTrue(held.is_zoh)
        self.assertEqual(2, held.payload["rsu_data"][0]["measurements"]["command"])
        self.assertEqual(1, held.payload["rsu_data"][0]["array"][0])
        self.assertEqual(7, held.payload["rsu_data"][0]["image"].getpixel((0, 0)))

        first_hold = held.materialize_payload()
        second_hold = held.materialize_payload()
        first_hold["rsu_data"][0]["measurements"]["command"] = 6
        first_hold["rsu_data"][0]["array"][0] = 6
        first_hold["rsu_data"][0]["image"].putpixel((0, 0), 6)
        self.assertEqual(
            2, second_hold["rsu_data"][0]["measurements"]["command"]
        )
        self.assertEqual(1, second_hold["rsu_data"][0]["array"][0])
        self.assertTrue(second_hold["rsu_data"][0]["array"].flags.writeable)
        self.assertEqual(
            7, second_hold["rsu_data"][0]["image"].getpixel((0, 0))
        )

    def test_tensor_payload_remains_rejected_before_replay_state(self):
        with self.assertRaisesRegex(TypeError, "unsupported payload"):
            ReplaySubmission(
                route_id="route-a",
                stream_id="rsu-1",
                source_id="rsu-1",
                source_frame=0,
                source_sim_time_ns=0,
                sequence_id=0,
                payload={"tensor": torch.tensor([1.0])},
            )

    def test_step_identity_and_rounded_nanosecond_conversion(self):
        state = self.state(payload=None)
        self.apply_method()(state, ["ego", "other"], ["rsu"], 7, .1234567896)
        submitted = state._ditl_replay_engine.submissions[0]
        self.assertEqual(submitted.source_frame, 7)
        self.assertEqual(submitted.sequence_id, 7)
        self.assertEqual(submitted.source_sim_time_ns, 123456790)
        self.assertEqual(state._ditl_replay_engine.advances, [(7, 123456790)])
        self.assertEqual(list(submitted.payload["car_data"]), [None, "other"])
        for invalid in (float("nan"), float("inf"), -1.0, 1e308):
            with self.subTest(timestamp=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    self.apply_method()(state, ["ego"], [], 8, invalid)

    def test_audit_failure_aborts_real_path_before_fusion(self):
        writer = mock.Mock()
        writer.append.side_effect = DitlAuditError("disk full")
        state = self.state(payload={"car_data": [None, "held"], "rsu_data": []},
                           writer=writer)
        state.config = {"simulation": {"ditl_replay": {}}}
        state._lat_aligner = None
        state.check_data = mock.Mock(side_effect=AssertionError("fusion reached"))
        state._apply_ditl_replay = self.apply_method().__get__(state)
        action = method("get_action_from_list_inter")
        end = next(i for i, stmt in enumerate(action.body)
                   if calls(stmt, "self.check_data"))
        prefix = compile_method(action, action.body[:end + 1])
        with self.assertRaisesRegex(DitlAuditError, "disk full"):
            prefix(state, ["ego", "current"], ["rsu"], 4, .2)
        state.check_data.assert_not_called()

    def test_audit_path_follows_meta_and_destroy_closes_writer(self):
        init = method("__init__")
        writers = calls(init, "DitlAuditWriter")
        self.assertEqual(len(writers), 1)
        self.assertIn("meta", string_literals(writers[0]))
        self.assertIn("ditl_event_audit.jsonl", string_literals(writers[0]))
        destroy = method("destroy")
        closes = calls(destroy, "self._ditl_audit_writer.close")
        self.assertEqual(len(closes), 1)
        writer = mock.Mock()
        close = compile_method(destroy)
        close(SimpleNamespace(_ditl_audit_writer=writer))
        writer.close.assert_called_once_with()
        writer.close.side_effect = DitlAuditError("close failed /audit/path")
        with self.assertRaisesRegex(DitlAuditError, "/audit/path"):
            close(SimpleNamespace(_ditl_audit_writer=writer))

    def test_spatial_config_and_audit_are_opt_in_and_ordered_after_replay(self):
        init = method("__init__")
        resolves = calls(init, "resolve_spatial_offset")
        writers = calls(init, "SpatialAuditWriter")
        self.assertEqual(len(resolves), 1)
        self.assertEqual(len(writers), 1)
        self.assertIn("meta", string_literals(writers[0]))
        self.assertIn("ditl_spatial_audit.jsonl", string_literals(writers[0]))

        action = method("get_action_from_list_inter")
        replay = calls(action, "self._apply_ditl_replay")
        dataset_calls = calls(action, "self.perception_dataloader.__getitem__")
        builds = calls(action, "self._ditl_build_spatial_audit_record")
        appends = calls(action, "self._ditl_spatial_audit_writer.append")
        self.assertEqual((len(replay), len(dataset_calls), len(builds),
                          len(appends)), (1, 1, 1, 1))
        self.assertLess(replay[0].lineno, dataset_calls[0].lineno)
        self.assertLess(dataset_calls[0].lineno, builds[0].lineno)
        self.assertLess(builds[0].lineno, appends[0].lineno)
        spatial_keys = [
            node for node in ast.walk(action)
            if isinstance(node, ast.Subscript) and
            "ditl_spatial_offset" in string_literals(node)
        ]
        self.assertTrue(spatial_keys)
        audit_if = next(
            node for node in ast.walk(action)
            if isinstance(node, ast.If) and appends[0] in list(ast.walk(node))
        )
        self.assertIn("_ditl_spatial_offset", {
            child.attr for child in ast.walk(audit_if.test)
            if isinstance(child, ast.Attribute)
        })

    def test_dataset_offsets_after_retrieval_and_restores_clean_before_pairwise(self):
        item = nested_method("__getitem_single_class__")
        retrieves = calls(item, "self.retrieve_base_data")
        offsets = calls(item, "apply_rsu_spatial_offset")
        noise = calls(item, "add_noise_data_dict_asymmetric")
        pairwise = calls(item, "get_pairwise_transformation_asymmetric")
        self.assertEqual((len(retrieves), len(offsets), len(noise), len(pairwise)),
                         (3, 1, 1, 1))
        self.assertLess(max(call.lineno for call in retrieves), offsets[0].lineno)
        self.assertLess(offsets[0].lineno, noise[0].lineno)
        self.assertLess(noise[0].lineno, pairwise[0].lineno)
        clean_assignments = [
            node for node in ast.walk(item)
            if isinstance(node, ast.Assign) and
            any("lidar_pose_clean" in string_literals(target)
                for target in node.targets)
        ]
        self.assertTrue(any(
            noise[0].lineno < assignment.lineno < pairwise[0].lineno
            for assignment in clean_assignments
        ))

    def test_dataset_pairwise_levels_preserve_clean_pose(self):
        prefix = compile_dataset_spatial_prefix()
        previous_pairwise_x = None
        for dx_m in (0.0, 0.5, 1.0, 2.0):
            base = OrderedDict((
                ("car_0", {
                    "ego": True,
                    "params": {
                        "lidar_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "map_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "lidar_pose_clean": [0.0] * 6,
                    },
                }),
                ("rsu_0", {
                    "ego": False,
                    "params": {
                        "lidar_pose": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "map_pose": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "lidar_pose_clean": [10.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0],
                    },
                }),
            ))
            state = SimpleNamespace(
                retrieve_base_data=lambda **kwargs: base,
                params={
                    "noise_setting": {"add_noise": False},
                    "comm_range": 1000.0,
                },
                visualize=False,
                kd_flag=False,
                box_align=False,
                max_cav=2,
                proj_first=False,
            )
            transformed, pairwise, clean_poses, audit = prefix(
                state, None, "all",
                {"ditl_spatial_offset": SpatialOffsetConfig(dx_m, 0, 0)},
                None,
            )
            self.assertEqual(clean_poses[1],
                             [10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.assertEqual(
                transformed["rsu_0"]["params"]["lidar_pose_clean"],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            self.assertEqual(len(audit), 1)
            pairwise_x = pairwise[1, 0, 0, 3]
            self.assertAlmostEqual(pairwise_x, 10.0 + dx_m)
            if previous_pairwise_x is not None:
                self.assertGreater(pairwise_x, previous_pairwise_x)
            previous_pairwise_x = pairwise_x

    def test_dataset_four_pass_applies_spatial_offset_only_once_for_all(self):
        applied_tpes = []

        def spy(tpe, base_data, config):
            applied_tpes.append(tpe)
            return apply_rsu_spatial_offset(base_data, config)

        run_dataset_four_pass(
            {"ditl_spatial_offset": SpatialOffsetConfig(1.0, 0.0, 0.0)},
            spy,
        )
        self.assertEqual(applied_tpes, ["all"])

    def test_class_gt_and_reference_poses_match_disabled_run(self):
        disabled, disabled_references = run_dataset_four_pass(
            {"car_data": [], "rsu_data": []},
            lambda tpe, base, config: apply_rsu_spatial_offset(base, config),
        )
        enabled, enabled_references = run_dataset_four_pass(
            {
                "car_data": [],
                "rsu_data": [],
                "ditl_spatial_offset": SpatialOffsetConfig(2.0, 0.0, 0.0),
            },
            lambda tpe, base, config: apply_rsu_spatial_offset(base, config),
        )
        disabled_class_refs = tuple(
            item for item in disabled_references if item[0] != "all"
        )
        enabled_class_refs = tuple(
            item for item in enabled_references if item[0] != "all"
        )
        self.assertEqual(enabled_class_refs, disabled_class_refs)
        np.testing.assert_array_equal(
            enabled["ego"]["object_bbx_center"],
            disabled["ego"]["object_bbx_center"],
        )
        torch.testing.assert_close(
            enabled["ego"]["single_object_bbx_center_torch"],
            disabled["ego"]["single_object_bbx_center_torch"],
        )

    def test_destroy_attempts_both_independent_audit_closes(self):
        events = []
        event_writer = mock.Mock()
        spatial_writer = mock.Mock()
        # Use explicit functions so call order and primary failure are observable.
        def close_event():
            events.append("event")
            raise RuntimeError("event close failed")

        def close_spatial():
            events.append("spatial")

        event_writer.close.side_effect = close_event
        spatial_writer.close.side_effect = close_spatial
        close = compile_method(method("destroy"))
        state = SimpleNamespace(
            _ditl_audit_writer=event_writer,
            _ditl_spatial_audit_writer=spatial_writer,
        )
        with self.assertRaisesRegex(RuntimeError, "event close failed"):
            close(state)
        self.assertEqual(events, ["event", "spatial"])
        self.assertIsNone(state._ditl_audit_writer)
        self.assertIsNone(state._ditl_spatial_audit_writer)

        # Disabled mode owns no spatial writer and must remain a no-op.
        close(SimpleNamespace())

    def test_agent_destroy_surfaces_audit_failure_after_other_cleanup(self):
        events = []

        class Infer:
            def destroy(self):
                events.append("infer")
                raise DitlAuditError("audit persistence failed")

        class Fifo:
            def close(self):
                events.append("fifo")
                raise RuntimeError("secondary fifo failure")

        state = SimpleNamespace(
            infer=Infer(), _ego_output_fifo=Fifo(),
            perception_model=object(), planning_model=object(),
        )
        destroy = compile_method(method("destroy", AGENT, "PnP_Agent"))
        with self.assertRaisesRegex(DitlAuditError, "audit persistence failed"):
            destroy(state)
        self.assertEqual(events, ["infer", "fifo"])
        self.assertFalse(hasattr(state, "perception_model"))
        self.assertFalse(hasattr(state, "planning_model"))

    def test_agent_destroy_tolerates_absent_infer_in_legacy_lifecycle(self):
        fifo = mock.Mock()
        state = SimpleNamespace(
            _ego_output_fifo=fifo,
            perception_model=object(), planning_model=object(),
        )
        destroy = compile_method(method("destroy", AGENT, "PnP_Agent"))
        destroy(state)
        fifo.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
