from __future__ import annotations

import unittest

import numpy as np

from tools.orin_deploy.lane_c_backbone_evidence import (
    compute_output_metrics,
    prepare_heldout_batches,
)


class LaneCBackboneEvidenceTest(unittest.TestCase):
    def test_prepare_heldout_batches_excludes_calibration_scenes(self) -> None:
        scene_lens = [1, 2, 1, 2]
        features = np.arange(6 * 2, dtype=np.float32).reshape(6, 1, 1, 2)
        batches, audit = prepare_heldout_batches(
            features, scene_lens, calibration_scene_count=2
        )
        self.assertEqual(list(batches.shape), [1, 2, 1, 1, 2])
        np.testing.assert_array_equal(batches.reshape(2, 2), features[3:5].reshape(2, 2))
        self.assertEqual(audit["excluded_agent_instances"], 3)
        self.assertEqual(audit["heldout_agent_instances_used"], 2)
        self.assertEqual(audit["heldout_agent_instances_dropped"], 1)

    def test_compute_output_metrics_reports_exact_match(self) -> None:
        reference = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        metrics = compute_output_metrics(reference, reference.copy())
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertEqual(metrics["nrmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["candidate_below_reference_min_fraction"], 0.0)
        self.assertEqual(metrics["candidate_above_reference_max_fraction"], 0.0)
        self.assertEqual(
            metrics["saturation_clipping_semantics"],
            "output_distribution_proxy_not_internal_quantizer_counter",
        )


if __name__ == "__main__":
    unittest.main()
