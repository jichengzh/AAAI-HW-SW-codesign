from __future__ import print_function

import copy
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ditl_replay_config import (  # noqa: E402
    BurstConfig,
    DitlReplayConfig,
    DitlReplayConfigError,
    ImpairmentConfig,
    resolve_ditl_replay_config,
)


def valid_simulation_config():
    return {
        "skip_frames": 4,
        "ditl_replay": {
            "library_root": "/var/lib/ditl",
            "catalog": "/var/lib/ditl/catalog.json",
            "bundle": "/var/lib/ditl/v1/bundles/feature-1",
            "exhaustion": "error",
            "seed": 20260923,
            "ttl_ms": None,
            "deadline_ms": 125.25,
            "max_queue_events": 1024,
            "impairment": {
                "extra_delay_ms": 1.5,
                "jitter_ms": 0.25,
                "drop_rate": 0.1,
                "burst": {
                    "enabled": False,
                    "enter_rate": 0.2,
                    "exit_rate": 0.8,
                },
                "reorder_rate": 0.3,
                "reorder_extra_ms": 2.75,
                "duplicate_rate": 0.4,
            },
        },
    }


class DitlReplayConfigTest(unittest.TestCase):
    def test_replay_config_is_opt_in_and_input_is_unchanged(self):
        simulation = {"skip_frames": 5, "ego_output_fifo_delay_ms": 200}
        original = copy.deepcopy(simulation)
        self.assertIsNone(resolve_ditl_replay_config(simulation))
        self.assertEqual(original, simulation)

    def test_complete_valid_mapping_is_copied_converted_and_frozen(self):
        simulation = valid_simulation_config()
        original = copy.deepcopy(simulation)

        config = resolve_ditl_replay_config(simulation)

        self.assertIsInstance(config, DitlReplayConfig)
        self.assertEqual(config.library_root, Path("/var/lib/ditl"))
        self.assertEqual(config.catalog, Path("/var/lib/ditl/catalog.json"))
        self.assertEqual(
            config.bundle, Path("/var/lib/ditl/v1/bundles/feature-1")
        )
        self.assertEqual(config.exhaustion, "error")
        self.assertEqual(config.seed, 20260923)
        self.assertIsNone(config.ttl_ns)
        self.assertEqual(config.deadline_ns, 125250000)
        self.assertEqual(config.max_queue_events, 1024)
        self.assertEqual(
            config.impairment,
            ImpairmentConfig(
                extra_delay_ns=1500000,
                jitter_ns=250000,
                drop_rate=0.1,
                burst=BurstConfig(False, 0.2, 0.8),
                reorder_rate=0.3,
                reorder_extra_ns=2750000,
                duplicate_rate=0.4,
            ),
        )
        self.assertEqual(simulation, original)
        with self.assertRaises(FrozenInstanceError):
            config.seed = 7
        with self.assertRaises(FrozenInstanceError):
            config.impairment.drop_rate = 0.5
        with self.assertRaises(FrozenInstanceError):
            config.impairment.burst.enabled = True

    def test_cycle_and_null_optional_times_are_accepted(self):
        simulation = valid_simulation_config()
        replay = simulation["ditl_replay"]
        replay["exhaustion"] = "cycle"
        replay["ttl_ms"] = None
        replay["deadline_ms"] = None
        config = resolve_ditl_replay_config(simulation)
        self.assertEqual(config.exhaustion, "cycle")
        self.assertIsNone(config.ttl_ns)
        self.assertIsNone(config.deadline_ns)

    def test_ego_output_fifo_may_coexist(self):
        simulation = valid_simulation_config()
        simulation["ego_output_fifo_delay_ms"] = 200
        self.assertIsNotNone(resolve_ditl_replay_config(simulation))

    def test_replay_config_rejects_each_legacy_rsu_latency_mode(self):
        for key, value in (
            ("comm_latency", 4),
            ("rsu_fifo_delay_ms", 200),
            ("latency_inject_ms", 200),
            ("latency_trace", "/x.csv"),
        ):
            with self.subTest(key=key):
                simulation = valid_simulation_config()
                simulation[key] = value
                with self.assertRaisesRegex(DitlReplayConfigError, key):
                    resolve_ditl_replay_config(simulation)

    def test_unknown_keys_are_rejected_at_every_replay_nesting_level(self):
        cases = (
            (("ditl_replay",), "unknown_replay"),
            (("ditl_replay", "impairment"), "unknown_impairment"),
            (("ditl_replay", "impairment", "burst"), "unknown_burst"),
        )
        for path, key in cases:
            with self.subTest(path=path, key=key):
                simulation = valid_simulation_config()
                target = simulation
                for component in path:
                    target = target[component]
                target[key] = 1
                with self.assertRaisesRegex(DitlReplayConfigError, key):
                    resolve_ditl_replay_config(simulation)

        simulation = valid_simulation_config()
        simulation["ditl_replay"][7] = "numeric YAML key"
        simulation["ditl_replay"]["another_unknown"] = True
        with self.assertRaises(DitlReplayConfigError):
            resolve_ditl_replay_config(simulation)

    def test_every_replay_field_is_required(self):
        required = tuple(valid_simulation_config()["ditl_replay"])
        for key in required:
            with self.subTest(key=key):
                simulation = valid_simulation_config()
                del simulation["ditl_replay"][key]
                with self.assertRaisesRegex(DitlReplayConfigError, key):
                    resolve_ditl_replay_config(simulation)

    def test_every_impairment_and_burst_field_is_required(self):
        valid = valid_simulation_config()
        impairment_keys = tuple(valid["ditl_replay"]["impairment"])
        burst_keys = tuple(
            valid["ditl_replay"]["impairment"]["burst"]
        )
        for key in impairment_keys:
            with self.subTest(level="impairment", key=key):
                simulation = valid_simulation_config()
                del simulation["ditl_replay"]["impairment"][key]
                with self.assertRaisesRegex(DitlReplayConfigError, key):
                    resolve_ditl_replay_config(simulation)
        for key in burst_keys:
            with self.subTest(level="burst", key=key):
                simulation = valid_simulation_config()
                del simulation["ditl_replay"]["impairment"]["burst"][key]
                with self.assertRaisesRegex(DitlReplayConfigError, key):
                    resolve_ditl_replay_config(simulation)

    def test_replay_and_nested_sections_must_be_mappings(self):
        for path, value in (
            (("ditl_replay",), []),
            (("ditl_replay", "impairment"), None),
            (("ditl_replay", "impairment", "burst"), "disabled"),
        ):
            with self.subTest(path=path):
                simulation = valid_simulation_config()
                target = simulation
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                with self.assertRaises(DitlReplayConfigError):
                    resolve_ditl_replay_config(simulation)

    def test_all_paths_must_be_absolute(self):
        for key, value in (
            ("library_root", "relative/library"),
            ("catalog", "relative/catalog.json"),
            ("bundle", "relative/v1/bundles/feature-1"),
        ):
            with self.subTest(key=key):
                simulation = valid_simulation_config()
                simulation["ditl_replay"][key] = value
                with self.assertRaisesRegex(DitlReplayConfigError, key):
                    resolve_ditl_replay_config(simulation)

    def test_catalog_must_be_exactly_below_library_root(self):
        for value in (
            "/other/catalog.json",
            "/var/lib/ditl/nested/catalog.json",
            "/var/lib/ditl/../ditl/catalog.json",
        ):
            with self.subTest(value=value):
                simulation = valid_simulation_config()
                simulation["ditl_replay"]["catalog"] = value
                with self.assertRaisesRegex(DitlReplayConfigError, "catalog"):
                    resolve_ditl_replay_config(simulation)

    def test_bundle_must_be_direct_child_of_v1_bundles(self):
        for value in (
            "/other/v1/bundles/feature-1",
            "/var/lib/ditl/feature-1",
            "/var/lib/ditl/v1/bundles/nested/feature-1",
            "/var/lib/ditl/v1/bundles/../bundles/feature-1",
        ):
            with self.subTest(value=value):
                simulation = valid_simulation_config()
                simulation["ditl_replay"]["bundle"] = value
                with self.assertRaisesRegex(DitlReplayConfigError, "bundle"):
                    resolve_ditl_replay_config(simulation)

    def test_exhaustion_accepts_only_error_or_cycle(self):
        for value in ("stop", "ERROR", "", None, True):
            with self.subTest(value=value):
                simulation = valid_simulation_config()
                simulation["ditl_replay"]["exhaustion"] = value
                with self.assertRaisesRegex(DitlReplayConfigError, "exhaustion"):
                    resolve_ditl_replay_config(simulation)

    def test_rates_must_be_finite_numbers_in_closed_unit_interval(self):
        fields = (
            ("drop_rate",),
            ("reorder_rate",),
            ("duplicate_rate",),
            ("burst", "enter_rate"),
            ("burst", "exit_rate"),
        )
        for path in fields:
            for value in (-0.01, 1.01, float("nan"), float("inf"), "0.5"):
                with self.subTest(path=path, value=value):
                    simulation = valid_simulation_config()
                    target = simulation["ditl_replay"]["impairment"]
                    for component in path[:-1]:
                        target = target[component]
                    target[path[-1]] = value
                    with self.assertRaisesRegex(
                        DitlReplayConfigError, path[-1]
                    ):
                        resolve_ditl_replay_config(simulation)
        simulation = valid_simulation_config()
        impairment = simulation["ditl_replay"]["impairment"]
        impairment["drop_rate"] = 0
        impairment["reorder_rate"] = 1
        self.assertIsNotNone(resolve_ditl_replay_config(simulation))

    def test_times_must_be_finite_nonnegative_numbers(self):
        fields = (
            ("ditl_replay", "ttl_ms"),
            ("ditl_replay", "deadline_ms"),
            ("ditl_replay", "impairment", "extra_delay_ms"),
            ("ditl_replay", "impairment", "jitter_ms"),
            ("ditl_replay", "impairment", "reorder_extra_ms"),
        )
        for path in fields:
            for value in (-0.001, float("nan"), -float("inf"), "1"):
                with self.subTest(path=path, value=value):
                    simulation = valid_simulation_config()
                    target = simulation
                    for component in path[:-1]:
                        target = target[component]
                    target[path[-1]] = value
                    with self.assertRaisesRegex(
                        DitlReplayConfigError, path[-1]
                    ):
                        resolve_ditl_replay_config(simulation)

    def test_booleans_are_never_accepted_as_numbers(self):
        paths = (
            ("ditl_replay", "seed"),
            ("ditl_replay", "ttl_ms"),
            ("ditl_replay", "deadline_ms"),
            ("ditl_replay", "max_queue_events"),
            ("ditl_replay", "impairment", "extra_delay_ms"),
            ("ditl_replay", "impairment", "jitter_ms"),
            ("ditl_replay", "impairment", "drop_rate"),
            ("ditl_replay", "impairment", "burst", "enter_rate"),
            ("ditl_replay", "impairment", "burst", "exit_rate"),
            ("ditl_replay", "impairment", "reorder_rate"),
            ("ditl_replay", "impairment", "reorder_extra_ms"),
            ("ditl_replay", "impairment", "duplicate_rate"),
        )
        for path in paths:
            with self.subTest(path=path):
                simulation = valid_simulation_config()
                target = simulation
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = True
                with self.assertRaisesRegex(
                    DitlReplayConfigError, path[-1]
                ):
                    resolve_ditl_replay_config(simulation)

    def test_burst_enabled_requires_a_real_boolean(self):
        for value in (0, 1, "false", None):
            with self.subTest(value=value):
                simulation = valid_simulation_config()
                burst = simulation["ditl_replay"]["impairment"]["burst"]
                burst["enabled"] = value
                with self.assertRaisesRegex(DitlReplayConfigError, "enabled"):
                    resolve_ditl_replay_config(simulation)

    def test_seed_requires_an_integer(self):
        for value in (1.0, "1", None, float("inf")):
            with self.subTest(value=value):
                simulation = valid_simulation_config()
                simulation["ditl_replay"]["seed"] = value
                with self.assertRaisesRegex(DitlReplayConfigError, "seed"):
                    resolve_ditl_replay_config(simulation)

    def test_queue_limit_requires_a_positive_integer(self):
        for value in (0, -1, 1.0, 1.5, "10", None):
            with self.subTest(value=value):
                simulation = valid_simulation_config()
                simulation["ditl_replay"]["max_queue_events"] = value
                with self.assertRaisesRegex(
                    DitlReplayConfigError, "max_queue_events"
                ):
                    resolve_ditl_replay_config(simulation)

    def test_parser_does_not_touch_the_filesystem(self):
        with mock.patch.object(
            Path, "exists", side_effect=AssertionError("filesystem access")
        ), mock.patch.object(
            Path, "open", side_effect=AssertionError("filesystem access")
        ), mock.patch.object(
            Path, "resolve", side_effect=AssertionError("filesystem access")
        ):
            self.assertIsNotNone(
                resolve_ditl_replay_config(valid_simulation_config())
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
