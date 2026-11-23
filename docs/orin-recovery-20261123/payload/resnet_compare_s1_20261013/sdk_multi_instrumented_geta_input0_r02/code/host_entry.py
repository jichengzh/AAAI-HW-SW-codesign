"""Fail-closed host transport; completion is not an independent accuracy audit."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from frozen_stream import evaluate


class RawReader:
    """Preserve every protocol byte consumed, including rejected replies."""
    def __init__(self, reader, raw):
        self.reader, self.raw = reader, raw

    def read(self, size=-1):
        data = self.reader.read(size)
        self.raw.write(data)
        self.raw.flush()
        return data


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _settle(process):
    """Bound local SSH teardown without asserting remote process termination."""
    try:
        process.stdin.close()
    except (OSError, ValueError):
        pass
    for action in (None, process.terminate, process.kill):
        try:
            if action is not None:
                action()
            return process.wait(timeout=120 if action is None else 30)
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            continue
    return None


def _validate(dataset, expected_identity, command, source_hashes):
    _check(len(dataset) == 50000 and len(dataset.rows) == 50000, 'Full 50000 dataset required')
    identity = expected_identity.get('manifest_identity')
    _check(bool(identity) and getattr(dataset, 'manifest_identity', None) == identity,
           'Fixed manifest_identity mismatch')
    _check(isinstance(command, (list, tuple)) and bool(command)
           and all(isinstance(x, str) and x and '\x00' not in x for x in command), 'Command argv required')
    _check(isinstance(source_hashes, dict) and bool(source_hashes)
           and all(isinstance(k, str) and k and isinstance(v, str)
                   and re.fullmatch('[0-9a-f]{64}', v) for k, v in source_hashes.items()),
           'Source SHA256 identities required')


def run_host(dataset, expected_identity, command, output, source_hashes):
    """Run exactly one full stream; preserve failure evidence and never retry.

    Caller supplies verified source hashes and a dataset with a fixed manifest
    identity. Returned COMPLETE means authenticated transport completion only.
    Independent prediction/accuracy and source-identity audits remain required.
    """
    _validate(dataset, expected_identity, command, source_hashes)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    process = None
    result = None
    error = None
    local_exit = None
    with (output / 'target.stderr.log').open('xb') as stderr, (output / 'target.stdout.raw').open('xb') as raw:
        try:
            process = subprocess.Popen(list(command), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=stderr, shell=False)
            result = evaluate(RawReader(process.stdout, raw), process.stdin,
                              dataset, output / 'stream', expected_identity)
            _check(result.get('samples') == 50000, 'Full 50000 result required')
        except BaseException as exc:
            error = exc
        finally:
            if process is not None:
                local_exit = _settle(process)
                process.stdout.close()
    stderr_text = (output / 'target.stderr.log').read_text(errors='replace')
    markers = re.findall(r'^REMOTE_SDK_FULLVAL_EXIT=(-?\d+)$', stderr_text, re.MULTILINE)
    remote_exit = int(markers[0]) if len(markers) == 1 else None
    if error is None and (local_exit != 0 or remote_exit != 0 or len(markers) != 1):
        error = ValueError('SSH/unique remote exit qualification failed')
    artifacts = ('target.stdout.raw', 'target.stderr.log', 'stream/logits.f32', 'stream/predictions.jsonl')
    receipt = {
        'status': 'FAILED' if error else 'COMPLETE', 'accuracy': result,
        'source_hashes': dict(source_hashes), 'expected_identity': dict(expected_identity),
        'command': list(command), 'ssh_exit': local_exit, 'remote_exit': remote_exit,
        'remote_marker_count': len(markers), 'cleanup_required': remote_exit is None,
        'error': None if error is None else {'type': type(error).__name__, 'message': str(error)},
        'artifact_sha256': {name: _hash(output / name) for name in artifacts if (output / name).is_file()},
    }
    (output / 'result.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    if error is not None:
        raise error
    return receipt
