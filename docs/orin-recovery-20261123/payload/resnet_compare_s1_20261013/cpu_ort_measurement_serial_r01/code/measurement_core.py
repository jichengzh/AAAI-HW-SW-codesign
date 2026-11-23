"""CPU ORT classification measurement core; requires independently audited orchestration."""
import hashlib
import math
import json
import os
from pathlib import Path
import time
import numpy as np

POINTS = ('dense', 'copt', 'geta')
SENSOR = 'ina3221:/sys/class/hwmon/hwmon3:VIN_SYS_5V0'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def _validate_inputs(inputs, ids, references):
    check(isinstance(inputs, np.ndarray) and inputs.shape == (256, 3, 224, 224)
          and inputs.dtype == np.float32 and inputs.flags.c_contiguous and np.isfinite(inputs).all(), 'invalid input')
    check(len(ids) == 256 and len(set(ids)) == 256, 'invalid input IDs')
    for point in POINTS:
        ref = references[point]
        check(isinstance(ref, np.ndarray) and ref.shape == (256, 1000) and
              ref.dtype == np.float32 and np.isfinite(ref).all(), 'invalid reference')


def _validate_energy(value):
    check(value.get('method') == 'sensor_integral' and value.get('sensor_id') == SENSOR and
          value.get('rail_label') == 'VIN_SYS_5V0' and value.get('scope') == 'VIN_SYS_5V0_rail_not_GPU_or_whole_board' and
          value.get('integration') == 'trapezoidal_actual_sensor_samples' and
          type(value.get('sample_count')) is int and value['sample_count'] >= 2 and
          all(type(value.get(k)) in (int, float) and math.isfinite(value[k]) for k in
              ('energy_j', 'start_monotonic_s', 'end_monotonic_s')) and
          value['energy_j'] > 0 and value['end_monotonic_s'] > value['start_monotonic_s'], 'invalid rail evidence')
    samples = value.get('raw_power_samples')
    check(isinstance(samples, list) and len(samples) == value['sample_count'], 'invalid raw rail samples')
    total = 0.0
    for pair in samples:
        check(isinstance(pair, (list, tuple)) and len(pair) == 2 and
              all(type(v) in (int, float) and math.isfinite(v) for v in pair) and pair[1] > 0,
              'invalid raw rail sample')
    for first, second in zip(samples, samples[1:]):
        check(second[0] > first[0], 'invalid raw rail sample order')
        total += (first[1] + second[1]) * 0.5 * (second[0] - first[0])
    check(samples[0][0] == value['start_monotonic_s'] and samples[-1][0] == value['end_monotonic_s']
          and math.isclose(total, value['energy_j'], rel_tol=1e-12, abs_tol=1e-12), 'invalid raw rail integral')


def validate_energy_row(row):
    energy=row['energy']
    _validate_energy(energy)
    check(energy['start_monotonic_s'] <= row['service_start_monotonic_s'] < row['service_end_monotonic_s'] <=
          energy['end_monotonic_s'], 'rail samples must enclose service')


def _request(index):
    if index < 256:
        return 'numerical', index
    if index < 306:
        return 'warmup', index - 256
    return 'timing', (index - 306) % 200


def _summary(rows):
    selected = rows[306:]
    result = {}
    for field in ('forward_ms', 'service_ms'):
        values = [r[field] for r in selected]
        result[field] = {'mean': float(np.mean(values)), **{'p%d' % p: float(np.percentile(values, p)) for p in (50, 95, 99)}}
    result['energy_j'] = float(np.mean([r['energy']['energy_j'] for r in selected])) if selected[0]['energy'] else None
    return result


def measure_all(runtimes, inputs, ids, references, identities, identity_check, qualification_check, reader_factory, emit, *, latency_gate):
    """Run once; caller must prove verifier functions and retain emitted raw rows.

    This core does not establish 50k qualification, capture references, load models,
    lock Orin, check libraries, or provide crash-safe output; external orchestration
    must supply and independently audit these before formal measurement.
    """
    check(set(runtimes) == set(POINTS) and set(references) == set(POINTS) and
          set(identities) == set(POINTS), 'expected three points')
    _validate_inputs(inputs, ids, references)
    for point in POINTS:
        check(qualification_check(point) is True, 'full accuracy qualification failed')
        check(identity_check(point) == identities[point], 'runtime identity mismatch')
        digest = identities[point].get('graph_sha256', '')
        check(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'invalid graph identity')
    output = {}
    for phase in ('latency', 'energy'):
        if phase == 'energy':
            check(latency_gate(output['latency']) is True, 'latency evidence qualification failed')
        output[phase] = {}
        for point in POINTS:
            check(identity_check(point) == identities[point], 'runtime identity changed')
            reader = reader_factory() if phase == 'energy' else None
            if reader:
                probe = reader.probe()
                check(probe.get('available') is True and probe.get('sensor_id') == SENSOR, 'rail unavailable')
            rows = []
            for index in range(906):
                kind, sample = _request(index)
                if reader:
                    reader.start_request()
                try:
                    logits, forward_ms, service_ms, service_start, service_end = runtimes[point].infer(inputs[sample:sample+1])
                finally:
                    energy = reader.finish_request() if reader else None
                check(isinstance(logits, np.ndarray) and logits.shape == (1,1000) and logits.dtype == np.float32
                      and np.isfinite(logits).all(), 'invalid SDK logits')
                check(np.allclose(logits[0], references[point][sample], atol=.01, rtol=.01), 'target replay mismatch')
                row = {'index': index, 'kind': kind, 'sample_id': ids[sample], 'phase': phase, 'point': point,
                       'forward_ms': forward_ms, 'service_ms': service_ms, 'energy': energy,
                       'service_start_monotonic_s':service_start, 'service_end_monotonic_s':service_end,
                       'logits_sha256': hashlib.sha256(logits.tobytes()).hexdigest()}
                if reader:
                    check(isinstance(energy,dict), 'missing rail evidence')
                    validate_energy_row(row)
                emit(row, logits.copy())
                rows.append(row)
            check(identity_check(point) == identities[point], 'runtime identity changed')
            output[phase][point] = {'budget': {'warmup':50, 'numerical':256, 'windows':3, 'requests_per_window':200},
                                    'identity': identities[point], 'summary': _summary(rows)}
    return output


class DurableLog:
    """Append exact logits and JSON rows in a fresh run; partial failures remain visible."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.streams = {}
        self.counts = {}

    def emit(self, row, logits):
        key = (row['point'], row['phase'])
        check(key[0] in POINTS and key[1] in ('latency', 'energy'), 'invalid log destination')
        check(row['index'] == self.counts.get(key, 0), 'out of order log row')
        data = logits.tobytes()
        check(logits.shape == (1, 1000) and logits.dtype == np.float32 and
              hashlib.sha256(data).hexdigest() == row['logits_sha256'], 'logits hash differs')
        if key not in self.streams:
            directory = self.root / key[0] / key[1]
            directory.mkdir(parents=True, exist_ok=False)
            self.streams[key] = ((directory / 'logits.f32').open('xb'),
                                 (directory / 'requests.jsonl').open('x', encoding='utf-8'))
        raw, records = self.streams[key]
        raw.write(data); raw.flush(); os.fsync(raw.fileno())
        records.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
        records.flush(); os.fsync(records.fileno())
        self.counts[key] = row['index'] + 1

    def close(self):
        streams, self.streams = self.streams, {}
        for pair in streams.values():
            for stream in pair:
                stream.close()


def _audit_phase(root, ids, references, expected, phase):
    root = Path(root)
    check(set(expected) == set(POINTS), 'incomplete phase points')
    for point in POINTS:
        directory = root / point / phase
        lines = (directory / 'requests.jsonl').read_text().splitlines()
        data = (directory / 'logits.f32').read_bytes()
        check(len(lines) == 906 and len(data) == 906 * 1000 * 4, 'incomplete phase raw')
        rows = []
        for index, line in enumerate(lines):
            row = json.loads(line)
            kind, sample = _request(index)
            check(row['index'] == index and row['kind'] == kind and row['sample_id'] == ids[sample]
                  and row['phase'] == phase and row['point'] == point, 'invalid phase order')
            if phase == 'energy':
                validate_energy_row(row)
            else:
                check(row['energy'] is None, 'unexpected latency energy')
            raw = data[index * 4000:(index + 1) * 4000]
            check(hashlib.sha256(raw).hexdigest() == row['logits_sha256'], 'logits hash differs')
            logits = np.frombuffer(raw, dtype=np.float32)
            check(np.isfinite(logits).all() and np.allclose(logits, references[point][sample], atol=.01, rtol=.01),
                  'logits replay mismatch')
            check(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] > 0
                      for k in ('forward_ms', 'service_ms')), 'invalid phase timing')
            rows.append(row)
        check(expected[point]['summary'] == _summary(rows), 'phase summary differs')
    return True


def audit_latency(root, ids, references, expected):
    """Recompute every persisted latency record before permitting energy phase."""
    return _audit_phase(root, ids, references, expected, 'latency')


def audit_energy(root, ids, references, expected):
    """Recompute every persisted energy record; independent review remains required."""
    return _audit_phase(root, ids, references, expected, 'energy')
