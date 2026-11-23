"""Read-only INA3221 VIN_SYS_5V0 rail integration for completed Orin requests.

This measures the named 5 V rail only. It is not GPU-only or whole-board energy.
Only the three user-authorized sysfs nodes are opened; no privilege or service.
"""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time


RAIL_ROOT = Path("/sys/class/hwmon/hwmon3")
RAIL_LABEL = "VIN_SYS_5V0"
SENSOR_ID = "ina3221:/sys/class/hwmon/hwmon3:VIN_SYS_5V0"


def integrate_samples(samples):
    """Trapezoidal integral of actual sampled power (monotonic seconds, watts)."""
    if len(samples) < 2:
        raise ValueError("Sensor integral needs at least two actual samples")
    total = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        if (not all(math.isfinite(value) for value in (t0, p0, t1, p1))
                or t1 <= t0 or p0 <= 0 or p1 <= 0):
            raise ValueError("Power samples must be finite positive and strictly ordered")
        total += (p0 + p1) * 0.5 * (t1 - t0)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid integrated rail energy")
    return total


class RailReader:
    def __init__(self, root=RAIL_ROOT, poll_interval_s=0.002):
        self.root = Path(root)
        if not 0.001 <= poll_interval_s <= 0.02:
            raise ValueError("Sensor polling interval must be bounded")
        self.poll_interval_s = poll_interval_s
        self._active = False
        self._samples = []
        self._error = None
        self._stop = None
        self._thread = None

    def _power(self):
        millivolts = int((self.root / "in3_input").read_text().strip())
        milliamps = int((self.root / "curr3_input").read_text().strip())
        if not 1000 <= millivolts <= 10000 or not 0 < milliamps <= 100000:
            raise ValueError("VIN_SYS_5V0 voltage/current out of plausible positive range")
        return millivolts * milliamps / 1_000_000.0

    def probe(self):
        try:
            label = (self.root / "in3_label").read_text().strip()
            if label != RAIL_LABEL:
                return {"available": False, "reason": "Authorized rail label differs"}
            watts = self._power()
            return {"available": True, "reason": "Read-only VIN_SYS_5V0 rail sampled; not GPU-only/whole-board",
                    "sensor_id": SENSOR_ID, "observed_w": watts,
                    "rail_label": RAIL_LABEL, "unit": "joule_rail_integral"}
        except (OSError, ValueError) as exc:
            return {"available": False, "reason": str(exc)}

    def _sample(self):
        stamp = time.monotonic()
        self._samples.append((stamp, self._power()))

    def _poll(self):
        try:
            while not self._stop.wait(self.poll_interval_s):
                self._sample()
        except (OSError, ValueError) as exc:
            self._error = exc

    def start_request(self):
        if self._active:
            raise RuntimeError("Power reader request already active")
        self._samples = []
        self._error = None
        self._stop = threading.Event()
        self._sample()
        self._active = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def finish_request(self):
        if not self._active:
            raise RuntimeError("Power reader request was not started")
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._active = False
        if self._thread.is_alive():
            raise TimeoutError("Power sampling thread did not exit")
        if self._error is not None:
            raise RuntimeError("Power sampling failed") from self._error
        self._sample()
        energy = integrate_samples(self._samples)
        powers = [value for _, value in self._samples]
        return {"method": "sensor_integral", "sensor_id": SENSOR_ID,
                "rail_label": RAIL_LABEL, "scope": "VIN_SYS_5V0_rail_not_GPU_or_whole_board",
                "voltage_unit": "millivolt", "current_unit": "milliamp",
                "power_formula": "mV_times_mA_div_1e6_watts",
                "integration": "trapezoidal_actual_sensor_samples",
                "sample_interval_requested_s": self.poll_interval_s,
                "sample_count": len(self._samples),
                "start_monotonic_s": self._samples[0][0],
                "end_monotonic_s": self._samples[-1][0],
                "sampled_power_min_w": min(powers),
                "sampled_power_max_w": max(powers),
                "energy_j": energy}


def open_reader():
    return RailReader()
