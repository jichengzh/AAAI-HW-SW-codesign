"""Deterministic, immutable deployment trace bundles on Linux filesystems."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import ctypes
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Mapping, Optional, Tuple

try:
    from .ditl_replay_schema import (
        NORMALIZER_NAME, NORMALIZER_VERSION, REPLAY_SAMPLE_SCHEMA_VERSION,
        STAGE_NAMES, ReplaySample, normalize_raw_trace, parse_replay_sample, replay_sample_to_dict,
    )
except ImportError:
    from ditl_replay_schema import (
        NORMALIZER_NAME, NORMALIZER_VERSION, REPLAY_SAMPLE_SCHEMA_VERSION,
        STAGE_NAMES, ReplaySample, normalize_raw_trace, parse_replay_sample, replay_sample_to_dict,
    )

LIBRARY_SCHEMA_VERSION = 'ditl.feature-library.v1'
MAX_SOURCE_FILE_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024  # Includes the line terminator, if present.
MAX_TRACE_RECORDS = 100000
MAX_JSON_DEPTH = 64  # The outermost object or array counts as depth one.
SOURCE_NAMES = (
    'trace.jsonl', 'command_manifest.txt', 'device_snapshot.txt',
    'model_sha256.txt', 'source_sha256sums.txt',
)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_BASENAME = re.compile(r'^[A-Za-z0-9_.-]+$')
_UTC_TIMESTAMP = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$'
)


class BundleBuildError(ValueError):
    """Raised when an immutable bundle cannot be constructed safely."""


@dataclass(frozen=True)
class SampleSelection:
    sample: ReplaySample
    cycle_index: int


class ReplaySampleCursor(object):
    """Read verified replay samples in order under an explicit exhaustion rule."""

    def __init__(self, samples: Tuple[ReplaySample, ...], exhaustion: str):
        if type(samples) is not tuple:
            raise TypeError("samples must be a tuple")
        if not samples:
            raise ValueError("samples must not be empty")
        if any(type(item) is not ReplaySample for item in samples):
            raise TypeError("samples must contain ReplaySample values")
        if exhaustion not in ("error", "cycle"):
            raise ValueError("exhaustion must be either error or cycle")
        self._samples = tuple(samples)
        self._exhaustion = exhaustion
        self._position = 0

    def _peek(self):
        if self._position >= len(self._samples) and self._exhaustion == "error":
            raise RuntimeError("replay sample cursor is exhausted")
        cycle_index, index = divmod(self._position, len(self._samples))
        return SampleSelection(self._samples[index], cycle_index)

    def _commit_peek(self):
        self._position += 1

    def next(self) -> SampleSelection:
        selected = self._peek()
        self._commit_peek()
        return selected


@dataclass(frozen=True)
class CaptureCondition:
    profile_id: str
    workload_label: str
    ros_placement: str
    link_type: str
    qos: str
    publish_frequency_hz: Optional[float]
    payload_semantics: str
    codec: str
    raw_payload_bytes: int
    wire_payload_bytes: int
    clock_domain: str
    clock_comparable: bool
    source_commit: str
    model_id: str
    engine_sha256: str
    device_identity: str
    jetpack_l4t: str
    power_mode: str
    load_label: str
    capture_time_utc: str
    capture_time_source: str


@dataclass(frozen=True)
class BuildRequest:
    source_dir: Path
    library_root: Path
    input_root: Path
    condition: CaptureCondition
    exhaustion_policies: Tuple[str, ...] = ('error', 'cycle')


@dataclass(frozen=True)
class BundleBuildResult:
    feature_id: str
    bundle_path: Path
    manifest_sha256: str
    checksums_sha256: str
    sample_count: int


_CONDITION_FIELDS = frozenset(field.name for field in fields(CaptureCondition))
MANIFEST_FIELDS = _CONDITION_FIELDS | frozenset((
    'library_schema_version', 'feature_id', 'capture_condition_id', 'capture_run_id',
    'source_trace_schema_version', 'source_trace_sha256', 'normalizer_name',
    'normalizer_version', 'normalizer_config_sha256', 'eligibility_criteria',
    'total_count', 'warmup_count', 'rejected_count', 'accepted_count',
    'exhaustion_policies', 'replay_sample_schema_version', 'selected_delay_field',
    'source_artifact_path',
))


def _require(valid, field, message):
    if not valid:
        raise BundleBuildError('%s: %s' % (field, message))


def _basename(name):
    _require(isinstance(name, str) and _BASENAME.fullmatch(name) is not None
             and name not in ('.', '..'), 'basename', 'invalid path component')
    return name


def parse_capture_condition(record: Mapping[str, Any]) -> CaptureCondition:
    """Validate every required condition field without discarding unknown keys."""
    _require(isinstance(record, Mapping), 'condition', 'must be an object')
    missing = _CONDITION_FIELDS - set(record)
    unknown = set(record) - _CONDITION_FIELDS
    _require(not missing, 'condition', 'missing fields: %s' % sorted(missing))
    _require(not unknown, 'condition', 'unknown fields: %s' % list(unknown))
    numeric = {'publish_frequency_hz', 'raw_payload_bytes', 'wire_payload_bytes',
               'clock_comparable'}
    for field in _CONDITION_FIELDS - numeric:
        value = record[field]
        _require(isinstance(value, str) and bool(value.strip()) and '\x00' not in value,
                 field, 'must be a non-empty string without NUL')
    _require(_BASENAME.fullmatch(record['profile_id']) is not None and
             record['profile_id'] not in ('.', '..'), 'profile_id', 'invalid basename')
    for field in ('raw_payload_bytes', 'wire_payload_bytes'):
        value = record[field]
        _require(type(value) is int and value >= 0, field, 'must be an integer >= 0')
    _require(type(record['clock_comparable']) is bool, 'clock_comparable', 'must be boolean')
    _require(_SHA256.fullmatch(record['engine_sha256']) is not None,
             'engine_sha256', 'must be 64 lowercase hexadecimal characters')
    _validate_frequency(record)
    stamp = record['capture_time_utc']
    _require(_UTC_TIMESTAMP.fullmatch(stamp) is not None, 'capture_time_utc',
             'must be an RFC 3339 UTC timestamp')
    try:
        datetime.strptime(stamp[:19], '%Y-%m-%dT%H:%M:%S')
    except ValueError as error:
        raise BundleBuildError('capture_time_utc: %s' % error)
    return CaptureCondition(**dict(record))


def _validate_frequency(record):
    value = record['publish_frequency_hz']
    if value is None:
        _require(record['workload_label'] == 'fixed_input_tensor_replay',
                 'publish_frequency_hz', 'null is only valid for fixed_input_tensor_replay')
        return
    valid = type(value) in (int, float) and value > 0
    try:
        valid = valid and math.isfinite(value)
    except OverflowError:
        valid = False
    _require(valid, 'publish_frequency_hz', 'must be finite and positive')


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _absolute(path):
    path = Path(path)
    _require('..' not in path.parts, 'path', 'parent traversal is forbidden')
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except RuntimeError as error:
        raise BundleBuildError('path: %s' % error) from error
    _require(absolute == resolved, 'path', 'symlink components are forbidden')
    return absolute


@contextmanager
def _directory(path):
    """Open each component without following links, including intermediate ones."""
    path = _absolute(path)
    fd = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for name in path.parts[1:]:
            next_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _child_directory(parent_fd, name):
    fd = os.open(_basename(name), _DIRECTORY_FLAGS, dir_fd=parent_fd)
    try:
        yield fd
    finally:
        os.close(fd)


def _read_file(directory_fd, name):
    fd = os.open(_basename(name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                 dir_fd=directory_fd)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode), name, 'must be a regular file')
        _require(info.st_size <= MAX_SOURCE_FILE_BYTES, name, 'file size exceeds byte limit')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(MAX_SOURCE_FILE_BYTES + 1)
        _require(len(data) <= MAX_SOURCE_FILE_BYTES, name, 'file size exceeds byte limit')
        return data
    finally:
        os.close(fd)


def _source_bytes(source_fd):
    _require(set(os.listdir(source_fd)) == set(SOURCE_NAMES), 'source_dir',
             'must contain exactly the five required source files')
    return {name: _read_file(source_fd, name) for name in SOURCE_NAMES}


def _write_file(directory_fd, name, data):
    fd = os.open(_basename(name), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def _normalization_config(policies):
    return {
        'measurement_scope': 'rsu_to_fusion', 'codec': 'none',
        'clock_comparable': True, 'headline_eligible': True, 'outcome': 'consumed',
        'remote_input_tensor_bytes': 0, 'exclude_warmup': True,
        'stages': list(STAGE_NAMES), 'selected_delay_field': 'base_delay_ns',
        'exhaustion_policies': list(policies),
    }


def _validate_trace_condition(records, normalized, condition):
    for key in ('model_id', 'engine_sha256', 'payload_semantics'):
        _require(normalized.identity[key] == getattr(condition, key), key,
                 'capture condition does not match trace identity')
    _require(condition.clock_comparable is True, 'clock_comparable', 'must be true')
    _require(condition.codec == 'none', 'codec', 'must be none')
    _require(normalized.accepted_count > 0, 'trace', 'must contain measured samples')
    for record in records:
        codec = record.get('codec', {})
        for field in ('raw_payload_bytes', 'wire_payload_bytes'):
            _require(codec.get(field) == getattr(condition, field), field,
                     'capture condition does not match trace')
        _require(codec.get('name') == condition.codec, 'codec', 'must match every record')
        for role in ('sender', 'receiver'):
            clock = record.get('clock', {}).get(role, {})
            _require(clock.get('domain') == condition.clock_domain, 'clock_domain',
                     'capture condition does not match trace')


def _statistics(values):
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered) - 1) * q
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {
        'count': len(ordered), 'min': ordered[0], 'max': ordered[-1],
        'mean': sum(ordered) / len(ordered), 'p50': percentile(0.50),
        'p95': percentile(0.95), 'p99': percentile(0.99),
    }


def _summary(samples):
    return {
        'base_delay_ns': _statistics([s.base_delay_ns for s in samples]),
        'stage_ns': {name: _statistics([s.stage_ns[name] for s in samples])
                     for name in STAGE_NAMES},
        'raw_payload_bytes': _statistics([s.raw_payload_bytes for s in samples]),
        'wire_payload_bytes': _statistics([s.wire_payload_bytes for s in samples]),
    }


def _prepare_files(request, source):
    condition = asdict(parse_capture_condition(asdict(request.condition)))
    condition_id = _sha256(_canonical(condition))
    records = _jsonl_records(source['trace.jsonl'])
    _require(all(isinstance(r, dict) for r in records), 'trace', 'records must be objects')
    normalized = normalize_raw_trace(records, condition_id)
    _validate_trace_condition(records, normalized, request.condition)
    config = _normalization_config(request.exhaustion_policies)
    source_hash = _sha256(source['trace.jsonl'])
    config_hash = _sha256(_canonical(config))
    feature_id = _sha256(_canonical({
        'capture_condition': condition, 'source_trace_sha256': source_hash,
        'normalizer_version': NORMALIZER_VERSION, 'normalizer_config': config,
    }))
    report = {
        'normalizer_name': NORMALIZER_NAME, 'normalizer_version': NORMALIZER_VERSION,
        'normalizer_config_sha256': config_hash, 'capture_condition_id': condition_id,
        'total_count': normalized.total_count, 'warmup_count': normalized.warmup_count,
        'rejected_count': normalized.rejected_count, 'accepted_count': normalized.accepted_count,
        'rejection_reasons': {'warmup': normalized.warmup_count},
    }
    manifest = dict(condition, **{key: value for key, value in report.items()
                                  if key != 'rejection_reasons'})
    manifest = dict(manifest, library_schema_version=LIBRARY_SCHEMA_VERSION,
                    feature_id=feature_id, capture_run_id=normalized.identity['run_id'],
                    source_trace_schema_version='ditl.trace.v2', source_trace_sha256=source_hash,
                    eligibility_criteria=config, exhaustion_policies=list(request.exhaustion_policies),
                    replay_sample_schema_version=REPLAY_SAMPLE_SCHEMA_VERSION,
                    selected_delay_field='base_delay_ns', source_artifact_path='raw/trace.jsonl')
    files = {
        'manifest.json': _canonical(manifest), 'raw/trace.jsonl': source['trace.jsonl'],
        'normalized/replay_samples.jsonl': b''.join(_canonical(replay_sample_to_dict(s))
                                                   for s in normalized.samples),
        'stats/summary.json': _canonical(_summary(normalized.samples)),
        'stats/normalization_report.json': _canonical(report),
        **{'provenance/' + name: source[name] for name in SOURCE_NAMES if name != 'trace.jsonl'},
    }
    checksums = ''.join('%s  %s\n' % (_sha256(files[name]), name) for name in sorted(files))
    return feature_id, dict(files, SHA256SUMS=checksums.encode('utf-8')), normalized.accepted_count


def _write_staging(staging_fd, files):
    for directory in ('raw', 'normalized', 'stats', 'provenance'):
        os.mkdir(directory, mode=0o700, dir_fd=staging_fd)
        with _child_directory(staging_fd, directory) as child_fd:
            for name in sorted(files):
                if name.startswith(directory + '/'):
                    _write_file(child_fd, name.split('/')[1], files[name])
            os.fsync(child_fd)
    for name in sorted(files):
        if '/' not in name:
            _write_file(staging_fd, name, files[name])
    os.fsync(staging_fd)


def _verify_existing(parent_fd, feature_id, files):
    with _child_directory(parent_fd, feature_id) as bundle_fd:
        expected_top = {name.split('/')[0] for name in files}
        _require(set(os.listdir(bundle_fd)) == expected_top, 'existing bundle', 'file set mismatch')
        for name in sorted(expected_top):
            if name in files:
                actual = _read_file(bundle_fd, name)
                _require(actual == files[name], 'existing bundle', 'bytes mismatch: ' + name)
            else:
                with _child_directory(bundle_fd, name) as child_fd:
                    expected = {path.split('/')[1] for path in files if path.startswith(name + '/')}
                    _require(set(os.listdir(child_fd)) == expected, 'existing bundle', 'file set mismatch')
                    for child in sorted(expected):
                        _require(_read_file(child_fd, child) == files[name + '/' + child],
                                 'existing bundle', 'bytes mismatch: ' + name + '/' + child)


def _atomic_publish(parent_fd, staging_name, final_name):
    """One atomic no-replace rename; unsupported platforms fail closed."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    _require(rename is not None, 'publish', 'renameat2 is required for atomic no-replace publication')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    result = rename(parent_fd, os.fsencode(_basename(staging_name)),
                    parent_fd, os.fsencode(_basename(final_name)), 1)
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _same_inode(first, second):
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _owns_file(parent_fd, name, owned):
    try:
        return owned is not None and _same_inode(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False), owned)
    except FileNotFoundError:
        return False


def _remove_staging(parent_fd, name, owned=None):
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if owned is not None and not _same_inode(info, owned):
        return
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return
    with _child_directory(parent_fd, name) as fd:
        if not _same_inode(os.fstat(fd), info):
            return
        for child in os.listdir(fd):
            _remove_staging(fd, child)
    if _owns_file(parent_fd, name, info):
        os.rmdir(name, dir_fd=parent_fd)


def _validate_request(request):
    _require(isinstance(request, BuildRequest), 'request', 'must be BuildRequest')
    _require(isinstance(request.condition, CaptureCondition), 'condition', 'must be CaptureCondition')
    parse_capture_condition(asdict(request.condition))
    _require(type(request.exhaustion_policies) is tuple and request.exhaustion_policies in
             (('error',), ('error', 'cycle')), 'exhaustion_policies',
             'must be (error,) or (error, cycle)')
    input_root = _absolute(request.input_root)
    source_path = _absolute(request.source_dir)
    _require(source_path == input_root or input_root in source_path.parents,
             'source_dir', 'must be contained in input_root')
    library_root = _absolute(request.library_root)
    return input_root, source_path, library_root


def _verify_directory_binding(path, original_fd):
    with _directory(path) as current_fd:
        original = os.fstat(original_fd)
        current = os.fstat(current_fd)
        _require((original.st_dev, original.st_ino) == (current.st_dev, current.st_ino),
                 'path', 'directory changed or was replaced during construction')


def _construct(request):
    input_root, source_path, library_root = _validate_request(request)
    with ExitStack() as stack:
        input_fd = stack.enter_context(_directory(input_root))
        source_fd = stack.enter_context(_directory(source_path))
        library_fd = stack.enter_context(_directory(library_root))
        parent_fd = stack.enter_context(_directory(library_root / 'v1/bundles'))
        source = _source_bytes(source_fd)
        initial_hashes = {name: _sha256(data) for name, data in source.items()}
        feature_id, files, count = _prepare_files(request, source)
        staging_name = '.%s.tmp-%d-%s' % (feature_id, os.getpid(), secrets.token_hex(16))
        os.mkdir(_basename(staging_name), mode=0o700, dir_fd=parent_fd)
        owned = None
        try:
            with _child_directory(parent_fd, staging_name) as staging_fd:
                owned = os.fstat(staging_fd)
                _write_staging(staging_fd, files)
            current_hashes = {name: _sha256(data) for name, data in _source_bytes(source_fd).items()}
            _require(current_hashes == initial_hashes, 'source', 'changed during construction')
            for path, fd in ((input_root, input_fd), (source_path, source_fd),
                             (library_root, library_fd), (library_root / 'v1/bundles', parent_fd)):
                _verify_directory_binding(path, fd)
            if feature_id in os.listdir(parent_fd):
                _verify_existing(parent_fd, feature_id, files)
            else:
                try:
                    _atomic_publish(parent_fd, staging_name, feature_id)
                except FileExistsError:
                    _verify_existing(parent_fd, feature_id, files)
            os.fsync(parent_fd)
            for path, fd in ((library_root, library_fd), (library_root / 'v1/bundles', parent_fd)):
                _verify_directory_binding(path, fd)
        except BaseException:
            if owned is not None:
                _remove_staging(parent_fd, feature_id, owned)
            raise
        finally:
            _remove_staging(parent_fd, staging_name, owned)
        return BundleBuildResult(feature_id, library_root / 'v1/bundles' / feature_id,
                                 _sha256(files['manifest.json']), _sha256(files['SHA256SUMS']), count)


def build_feature_bundle(request: BuildRequest) -> BundleBuildResult:
    """Build or verify one immutable content-addressed bundle."""
    try:
        return _construct(request)
    except BundleBuildError:
        raise
    except (OSError, ValueError, TypeError, OverflowError) as error:
        raise BundleBuildError('bundle construction failed: %s' % error) from error


class BundleVerificationError(ValueError):
    """Raised before any samples are returned from an invalid bundle."""


def _freeze(value):
    from types import MappingProxyType
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class VerifiedFeatureBundle:
    feature_id: str
    profile_id: str
    bundle_path: Path
    manifest: Mapping[str, object]
    samples: Tuple[ReplaySample, ...]
    checksums_sha256: str

    def __post_init__(self):
        object.__setattr__(self, 'manifest', _freeze(self.manifest))
        object.__setattr__(self, 'samples', tuple(self.samples))


def _strict_json(data):
    def pairs(items):
        result = dict(items)
        _require(len(result) == len(items), 'JSON', 'duplicate fields are forbidden')
        return result
    def invalid_constant(value):
        raise BundleBuildError('JSON: non-finite number ' + value)
    def finite_float(value):
        result = float(value)
        _require(math.isfinite(result), 'JSON', 'non-finite number ' + value)
        return result
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    _check_json_depth(text)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant,
                      parse_float=finite_float)


def _check_json_depth(text):
    depth, quoted, escaped = 0, False, False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in '{[':
            depth += 1
            _require(depth <= MAX_JSON_DEPTH, 'JSON', 'nesting depth exceeds limit')
        elif character in '}]':
            depth -= 1


def _jsonl_records(data):
    records = []
    with io.BytesIO(data) as stream:
        for line in iter(lambda: stream.readline(MAX_JSONL_LINE_BYTES + 1), b''):
            _require(len(line) <= MAX_JSONL_LINE_BYTES, 'JSONL', 'line exceeds byte limit')
            _require(len(records) < MAX_TRACE_RECORDS, 'JSONL', 'records exceed limit')
            records.append(_strict_json(line))
    return records


_BUNDLE_LAYOUT = {
    'raw': ('trace.jsonl',), 'normalized': ('replay_samples.jsonl',),
    'stats': ('summary.json', 'normalization_report.json'),
    'provenance': tuple(name for name in SOURCE_NAMES if name != 'trace.jsonl'),
}


def _bundle_bytes(bundle_fd):
    top_files = ('manifest.json', 'SHA256SUMS')
    _require(set(os.listdir(bundle_fd)) == set(top_files) | set(_BUNDLE_LAYOUT),
             'bundle', 'missing or extra files')
    files = {name: _read_file(bundle_fd, name) for name in top_files}
    for name, children in _BUNDLE_LAYOUT.items():
        with _child_directory(bundle_fd, name) as child_fd:
            _require(set(os.listdir(child_fd)) == set(children), name, 'missing or extra files')
            files = dict(files, **{name + '/' + child: _read_file(child_fd, child)
                                   for child in children})
    return files


def _verify_checksums(files):
    expected = ''.join('%s  %s\n' % (_sha256(files[name]), name)
                       for name in sorted(set(files) - {'SHA256SUMS'})).encode('utf-8')
    _require(files['SHA256SUMS'] == expected, 'SHA256SUMS', 'file set or digest mismatch')


def _catalog_location(catalog_path, root):
    path = Path(catalog_path).absolute()
    _require('..' not in path.parts, 'catalog path', 'parent traversal is forbidden')
    parent = _absolute(path.parent)
    _require(parent == root or root in parent.parents, 'catalog path', 'outside library_root')
    _basename(path.name)
    _require(path.name != 'catalog.lock', 'catalog path', 'cannot replace catalog.lock')
    return path


def _parse_catalog(data):
    catalog = _strict_json(data)
    _require(isinstance(catalog, dict) and set(catalog) == {'library_schema_version', 'features'},
             'catalog', 'invalid or unknown fields')
    _require(catalog['library_schema_version'] == LIBRARY_SCHEMA_VERSION,
             'catalog', 'unsupported schema version')
    _require(isinstance(catalog['features'], dict), 'catalog.features', 'must be an object')
    for feature_id, entry in catalog['features'].items():
        _require(_SHA256.fullmatch(feature_id) is not None, 'catalog.feature_id', 'invalid digest')
        _require(isinstance(entry, dict) and set(entry) ==
                 {'checksums_sha256', 'profile_id', 'relative_path'}, 'catalog entry', 'invalid fields')
        _require(isinstance(entry['checksums_sha256'], str) and
                 _SHA256.fullmatch(entry['checksums_sha256']) is not None,
                 'catalog.checksums_sha256', 'invalid digest')
        _basename(entry['profile_id'])
        _require(entry['relative_path'] == 'v1/bundles/' + feature_id,
                 'catalog path', 'must identify the canonical bundle path')
    return catalog


def _manifest_samples(files, feature_id, root):
    manifest = _strict_json(files['manifest.json'])
    _require(isinstance(manifest, dict) and set(manifest) == MANIFEST_FIELDS,
             'manifest', 'missing or unknown fields')
    condition = parse_capture_condition({key: manifest[key] for key in _CONDITION_FIELDS})
    policies = manifest['exhaustion_policies']
    _require(policies in (['error'], ['error', 'cycle']), 'exhaustion_policies', 'invalid policies')
    source = {name: files[('raw/' if name == 'trace.jsonl' else 'provenance/') + name]
              for name in SOURCE_NAMES}
    request = BuildRequest(root, root, root, condition, tuple(policies))
    expected_id, expected, _ = _prepare_files(request, source)
    _require(expected_id == feature_id, 'feature_id', 'content identity mismatch')
    _require(files['manifest.json'] == expected['manifest.json'], 'manifest', 'contract mismatch')
    samples = tuple(parse_replay_sample(record)
                    for record in _jsonl_records(files['normalized/replay_samples.jsonl']))
    _require(len(samples) == manifest['accepted_count'], 'samples', 'count mismatch')
    for name in ('normalized/replay_samples.jsonl', 'stats/summary.json',
                 'stats/normalization_report.json'):
        _require(files[name] == expected[name], name, 'normalization or manifest linkage mismatch')
    return manifest, samples


def _verify_bundle(bundle_path, library_root, catalog_path):
    root = _absolute(library_root)
    path = _absolute(bundle_path)
    feature_id = path.name
    _require(_SHA256.fullmatch(feature_id) is not None, 'feature_id', 'invalid digest')
    _require(path == root / 'v1/bundles' / feature_id, 'catalog path',
             'bundle must use its canonical path below library_root')
    with ExitStack() as stack:
        root_fd = stack.enter_context(_directory(root))
        parent_fd = stack.enter_context(_directory(path.parent))
        bundle_fd = stack.enter_context(_child_directory(parent_fd, feature_id))
        files = _bundle_bytes(bundle_fd)
        digest = _sha256(files['SHA256SUMS'])
        catalog_data = None
        if catalog_path is not None:
            catalog = _catalog_location(catalog_path, root)
            catalog_fd = stack.enter_context(_directory(catalog.parent))
            catalog_data = _read_file(catalog_fd, catalog.name)
            entry = _parse_catalog(catalog_data)['features'].get(feature_id)
            _require(entry is not None, 'catalog', 'feature is not registered')
            _require(entry['checksums_sha256'] == digest, 'catalog', 'SHA256SUMS digest mismatch')
        _verify_checksums(files)
        manifest, samples = _manifest_samples(files, feature_id, root)
        if catalog_data is not None:
            _require(entry['profile_id'] == manifest['profile_id'], 'catalog', 'profile_id mismatch')
            _require(_read_file(catalog_fd, catalog.name) == catalog_data, 'catalog', 'changed during read')
            _verify_directory_binding(catalog.parent, catalog_fd)
        _verify_existing(parent_fd, feature_id, files)
        for location, fd in ((root, root_fd), (path.parent, parent_fd), (path, bundle_fd)):
            _verify_directory_binding(location, fd)
        return VerifiedFeatureBundle(feature_id, manifest['profile_id'], path, manifest, samples, digest)


def verify_feature_bundle(bundle_path, library_root, catalog_path=None) -> VerifiedFeatureBundle:
    """Validate the entire confined snapshot before returning immutable samples."""
    try:
        return _verify_bundle(bundle_path, library_root, catalog_path)
    except BundleVerificationError:
        raise
    except (OSError, ValueError, TypeError, OverflowError, KeyError) as error:
        raise BundleVerificationError('bundle verification failed: %s' % error) from error


@contextmanager
def _catalog_lock(parent_fd):
    import fcntl
    fd = os.open('catalog.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                 0o600, dir_fd=parent_fd)
    try:
        _require(stat.S_ISREG(os.fstat(fd).st_mode), 'catalog.lock', 'must be a regular file')
        fcntl.flock(fd, fcntl.LOCK_EX)
        current = os.stat('catalog.lock', dir_fd=parent_fd, follow_symlinks=False)
        held = os.fstat(fd)
        _require((current.st_dev, current.st_ino) == (held.st_dev, held.st_ino),
                 'catalog.lock', 'lock file was replaced')
        yield
    finally:
        os.close(fd)  # Closing releases the advisory lock, including on failure.


def _replace_catalog(parent_fd, path, old, data):
    name = path.name
    suffix = '%d-%s' % (os.getpid(), secrets.token_hex(16))
    temporary, backup = '.catalog.tmp-' + suffix, '.catalog.backup-' + suffix
    owned = None
    try:
        _write_file(parent_fd, temporary, data)
        owned = os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
        if old is not None:
            _require(_read_file(parent_fd, name) == old, 'catalog', 'changed while locked')
            _write_file(parent_fd, backup, old)
        else:
            _require(name not in os.listdir(parent_fd), 'catalog', 'created while locked')
        _verify_directory_binding(path.parent, parent_fd)
        os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
        _verify_directory_binding(path.parent, parent_fd)
    except BaseException as error:
        try:
            if _owns_file(parent_fd, name, owned):
                if old is None:
                    os.unlink(name, dir_fd=parent_fd)
                else:
                    os.replace(backup, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                # Restore visible old bytes even when fsync keeps failing.
                os.fsync(parent_fd)
        except BaseException as rollback_error:
            raise error from rollback_error
        raise
    finally:
        for leftover in (temporary, backup):
            try:
                os.unlink(leftover, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def register_bundle(bundle, catalog_path, library_root) -> None:
    """Reverify and register a complete bundle under an atomic catalog lock."""
    try:
        _require(isinstance(bundle, (BundleBuildResult, VerifiedFeatureBundle)),
                 'bundle', 'must be a built or verified bundle')
        root = _absolute(library_root)
        path = _catalog_location(catalog_path, root)
        with _directory(path.parent) as parent_fd, _catalog_lock(parent_fd):
            verified = verify_feature_bundle(bundle.bundle_path, library_root)
            _require(bundle.feature_id == verified.feature_id and
                     bundle.checksums_sha256 == verified.checksums_sha256,
                     'bundle', 'result does not match verified content')
            try:
                old = _read_file(parent_fd, path.name)
            except FileNotFoundError:
                old = None
            catalog = (_parse_catalog(old) if old is not None else
                       {'library_schema_version': LIBRARY_SCHEMA_VERSION, 'features': {}})
            entry = {'checksums_sha256': verified.checksums_sha256,
                     'profile_id': verified.profile_id,
                     'relative_path': 'v1/bundles/' + verified.feature_id}
            previous = catalog['features'].get(verified.feature_id)
            _require(previous is None or previous == entry, 'catalog', 'existing entry conflicts')
            updated = dict(catalog, features=dict(catalog['features'], **{verified.feature_id: entry}))
            _verify_directory_binding(path.parent, parent_fd)
            _replace_catalog(parent_fd, path, old, _canonical(updated))
    except BundleVerificationError:
        raise
    except (OSError, ValueError, TypeError, OverflowError, KeyError) as error:
        raise BundleVerificationError('catalog registration failed: %s' % error) from error
