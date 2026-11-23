import os
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dual_fifo_config import DualFifoConfig, resolve_dual_fifo_config


class DualFifoConfigTest(unittest.TestCase):
    def test_legacy_keys_emit_one_actionable_future_warning(self):
        aliases = {
            "latency_inject_ms": (200, "rsu_fifo_delay_ms"),
            "action_fifo_delay_ms": (500, "ego_output_fifo_delay_ms"),
            "action_fifo_audit": (True, "latency_audit"),
        }
        for keys in ((key,) for key in aliases):
            with self.subTest(keys=keys), warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                resolve_dual_fifo_config({key: aliases[key][0] for key in keys})
                self.assertEqual(len(seen), 1)
                self.assertIs(seen[0].category, FutureWarning)
                self.assertEqual(seen[0].filename, __file__)
                message = str(seen[0].message)
                for legacy, (_, canonical) in aliases.items():
                    if legacy in keys:
                        self.assertIn("%s -> %s" % (legacy, canonical), message)
                    else:
                        self.assertNotIn(legacy, message)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            resolve_dual_fifo_config({key: value for key, (value, _) in aliases.items()})
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0].category, FutureWarning)
        for legacy, (_, canonical) in aliases.items():
            self.assertIn("%s -> %s" % (legacy, canonical), str(seen[0].message))

    def test_canonical_keys_emit_no_warning(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            resolve_dual_fifo_config({
                "rsu_fifo_delay_ms": 200,
                "ego_output_fifo_delay_ms": 500,
                "latency_audit": True,
            })
        self.assertEqual(seen, [])

    def test_canonical_fields_are_loaded_without_mutating_input(self):
        source = {
            "rsu_fifo_delay_ms": 200,
            "ego_output_fifo_delay_ms": 500,
            "latency_audit": True,
            "tau_ego_ms": 0,
        }
        before = dict(source)
        config = resolve_dual_fifo_config(source)
        self.assertEqual(config.rsu_fifo_delay_ms, 200.0)
        self.assertEqual(config.ego_output_fifo_delay_ms, 500.0)
        self.assertTrue(config.latency_audit)
        self.assertTrue(config.uses_new_contract)
        self.assertEqual(source, before)

    def test_legacy_fields_remain_supported(self):
        with self.assertWarnsRegex(FutureWarning, "legacy dual-FIFO keys"):
            config = resolve_dual_fifo_config({
                "latency_inject_ms": 200,
                "action_fifo_delay_ms": 500,
                "action_fifo_audit": True,
            })
        self.assertEqual(config.rsu_fifo_delay_ms, 200.0)
        self.assertEqual(config.ego_output_fifo_delay_ms, 500.0)
        self.assertTrue(config.latency_audit)
        self.assertFalse(config.uses_new_contract)

    def test_equal_aliases_are_accepted_and_conflicts_rejected(self):
        with self.assertWarnsRegex(FutureWarning, "legacy dual-FIFO keys"):
            config = resolve_dual_fifo_config({
                "rsu_fifo_delay_ms": 200,
                "latency_inject_ms": 200,
                "ego_output_fifo_delay_ms": 500,
                "action_fifo_delay_ms": 500,
                "tau_ego_ms": 0,
            })
        self.assertEqual(config.rsu_fifo_delay_ms, 200.0)
        with self.assertWarnsRegex(FutureWarning, "latency_inject_ms -> rsu_fifo_delay_ms"), \
                self.assertRaisesRegex(ValueError, "rsu_fifo_delay_ms"):
            resolve_dual_fifo_config({
                "rsu_fifo_delay_ms": 200,
                "latency_inject_ms": 500,
            })
        with self.assertWarnsRegex(FutureWarning, "action_fifo_delay_ms -> ego_output_fifo_delay_ms"), \
                self.assertRaisesRegex(ValueError, "ego_output_fifo_delay_ms"):
            resolve_dual_fifo_config({
                "ego_output_fifo_delay_ms": 200,
                "action_fifo_delay_ms": 500,
            })

    def test_equal_audit_aliases_are_accepted_and_conflicts_rejected(self):
        with self.assertWarnsRegex(FutureWarning, "action_fifo_audit -> latency_audit"):
            config = resolve_dual_fifo_config({
                "latency_audit": True,
                "action_fifo_audit": True,
                "tau_ego_ms": 0,
            })
        self.assertTrue(config.latency_audit)
        self.assertTrue(config.uses_new_contract)
        with self.assertWarnsRegex(FutureWarning, "action_fifo_audit -> latency_audit"), \
                self.assertRaisesRegex(ValueError, "latency_audit"):
            resolve_dual_fifo_config({
                "latency_audit": False,
                "action_fifo_audit": True,
                "tau_ego_ms": 0,
            })

    def test_new_contract_rejects_nonzero_tau_ego(self):
        with self.assertRaisesRegex(ValueError, "tau_ego_ms must be 0"):
            resolve_dual_fifo_config({
                "rsu_fifo_delay_ms": 0,
                "ego_output_fifo_delay_ms": 200,
                "tau_ego_ms": 200,
            })

    def test_legacy_plan_experiments_remain_loadable(self):
        config = resolve_dual_fifo_config({"tau_ego_ms": 200})
        self.assertFalse(config.uses_new_contract)

    def test_negative_and_nonfinite_delays_are_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_dual_fifo_config({"rsu_fifo_delay_ms": value})
                with self.assertRaises(ValueError):
                    resolve_dual_fifo_config({"ego_output_fifo_delay_ms": value})


if __name__ == "__main__":
    unittest.main(verbosity=2)
