"""Retained-channel planner projections preserve the pretrained mapping."""

import ast
import copy
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import yaml

TEAM_CODE = Path(__file__).resolve().parent.parent
REPO = TEAM_CODE.parents[2]
sys.path.insert(0, str(TEAM_CODE.parent))
sys.path.insert(0, str(REPO))
from opencood.models.sub_modules.downsample_conv import DownsampleConv


def projection_module():
    try:
        return importlib.import_module("team_code.closedloop.coptv2x_hil_planner_adapter")
    except ImportError:
        raise AssertionError("retained-channel pretrained projection is not implemented")


def production_loader():
    names = {"_planning_feature_adapter_weights", "_planner_feature_adapter_config",
             "_load_planning_feature_adapter"}
    tree = ast.parse((TEAM_CODE / "pnp_infer_action_e2e.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes), "<production-planner-loader>", "exec"), namespace)
    return namespace["_load_planning_feature_adapter"]


class RetainedPlannerProjectionTest(unittest.TestCase):
    def test_indices_preserve_prefixes_in_deblock_concatenation_order(self):
        project = projection_module()
        self.assertEqual([0, 1, 4, 8, 9, 10],
                         project.retained_projection_indices([4, 4, 4], [2, 1, 3]))
        for widths in ([], [0, 1, 1], [5, 1, 1], [1, 1], [True, 1, 1]):
            with self.assertRaises(ValueError):
                project.retained_projection_indices([4, 4, 4], widths)

    def test_projection_matches_original_with_removed_channels_zeroed(self):
        project = projection_module()
        torch.manual_seed(9)
        config = {"input_dim": 384, "dim": [128], "kernal_size": [1],
                  "stride": [1], "padding": [0]}
        original = DownsampleConv(config).eval()
        original_weights = {key: value.clone() for key, value in original.state_dict().items()}
        for retained, literal_indices in (
                ([128, 128, 128], list(range(384))),
                ([64, 64, 64], list(range(64)) + list(range(128, 192)) + list(range(256, 320))),
                ([80, 80, 80], list(range(80)) + list(range(128, 208)) + list(range(256, 336))),
                ([32, 32, 32], list(range(32)) + list(range(128, 160)) + list(range(256, 288)))):
            adapted_config, adapted_weights = project.slice_pretrained_projection(
                config, original_weights, [128, 128, 128], retained)
            adapter = DownsampleConv(adapted_config).eval()
            adapter.load_state_dict(adapted_weights, strict=True)
            reduced = torch.randn(1, sum(retained), 3, 4)
            expanded = torch.zeros(1, 384, 3, 4)
            expanded[:, literal_indices] = reduced
            with torch.no_grad():
                self.assertTrue(torch.allclose(adapter(reduced), original(expanded),
                                              atol=1e-6, rtol=1e-5))
            for key in original_weights:
                self.assertTrue(torch.equal(original.state_dict()[key], original_weights[key]))
                if key != "layers.0.double_conv.0.weight":
                    self.assertTrue(torch.equal(adapted_weights[key], original_weights[key]))
            adapted_config["dim"][0] = 1
            adapted_weights["layers.0.double_conv.0.bias"].zero_()
            self.assertEqual(128, config["dim"][0])
            self.assertTrue(torch.equal(original.state_dict()["layers.0.double_conv.0.bias"],
                                       original_weights["layers.0.double_conv.0.bias"]))

    def test_projection_rejects_inconsistent_source_kernel_contract(self):
        project = projection_module()
        with self.assertRaises(ValueError):
            project.slice_pretrained_projection({"input_dim": 8}, {}, [4, 4, 4], [2, 2, 2])
        for weights in ({}, {"layers.0.double_conv.0.weight": torch.zeros(8, 6, 1, 1)},
                        {"layers.0.double_conv.0.weight": torch.zeros(8, 12)}):
            with self.assertRaises(ValueError):
                project.slice_pretrained_projection({"input_dim": 12}, weights,
                                                    [4, 4, 4], [2, 2, 2])

    def test_loader_constructs_the_retained_width_projection_from_frozen_weights(self):
        config = {"input_dim": 12, "dim": [128], "kernal_size": [1],
                  "stride": [1], "padding": [0]}
        original = DownsampleConv(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            hypes = Path(directory) / "config.yaml"
            checkpoint = Path(directory) / "weights.pth"
            hypes.write_text(yaml.safe_dump({"model": {"args": {"shrink_header": config}}}))
            torch.save({"shrink_conv." + key: value
                        for key, value in original.state_dict().items()}, str(checkpoint))
            loaded = production_loader()({
                "feature_adapter_hypes": str(hypes),
                "feature_adapter_checkpoint": str(checkpoint),
                "feature_adapter_source_widths": [4, 4, 4],
                "feature_adapter_retained_widths": [2, 1, 3],
            }, "cpu")
            reduced = torch.randn(1, 6, 3, 4)
            expanded = torch.zeros(1, 12, 3, 4)
            expanded[:, [0, 1, 4, 8, 9, 10]] = reduced
            with torch.no_grad():
                actual = loaded(reduced)
                self.assertTrue(torch.allclose(actual, original(expanded), atol=1e-6, rtol=1e-5))
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))

    def test_real_planning_boundary_applies_projection_to_compressed_channels(self):
        tree = ast.parse((TEAM_CODE / "pnp_infer_action_e2e.py").read_text())
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and any(isinstance(child, ast.Attribute)
                              and child.attr == "_planning_feature_adapter"
                              for child in ast.walk(node.test)))
        function = ast.parse("def apply(self, feature):\n    return feature\n").body[0]
        function.body = [copy.deepcopy(branch), function.body[0]]
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function])),
                     "<production-planner-boundary>", "exec"), namespace)
        config = {"input_dim": 96, "dim": [128], "kernal_size": [1],
                  "stride": [1], "padding": [0]}
        feature = torch.randn(1, 96, 3, 4)
        projected = namespace["apply"](
            SimpleNamespace(_planning_feature_adapter=DownsampleConv(config).eval()), feature)
        self.assertEqual((1, 128, 3, 4), tuple(projected.shape))


if __name__ == "__main__":
    unittest.main()
