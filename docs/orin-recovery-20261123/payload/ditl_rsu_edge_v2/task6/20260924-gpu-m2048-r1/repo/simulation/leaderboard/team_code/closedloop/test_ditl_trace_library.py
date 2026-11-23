"""Filesystem contract tests for immutable deployment trace bundles."""
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import ditl_trace_library as library
from ditl_trace_library import (
    BuildRequest, BundleBuildError, CaptureCondition, build_feature_bundle,
    parse_capture_condition,
)

FIXTURE = Path(__file__).parent / 'fixtures/ditl_library/source'
FILES = {
    'manifest.json', 'raw/trace.jsonl', 'normalized/replay_samples.jsonl',
    'stats/summary.json', 'stats/normalization_report.json',
    'provenance/command_manifest.txt', 'provenance/device_snapshot.txt',
    'provenance/model_sha256.txt', 'provenance/source_sha256sums.txt', 'SHA256SUMS',
}


def condition_record():
    return dict(
        profile_id='raw_samehost_idle', workload_label='fixed_input_tensor_replay',
        ros_placement='same_host', link_type='loopback', qos='reliable',
        publish_frequency_hz=None, payload_semantics='dense_head_debug', codec='none',
        raw_payload_bytes=2621440, wire_payload_bytes=2621440,
        clock_domain='monotonic_raw', clock_comparable=True, source_commit='a' * 40,
        model_id='rsu-edge', engine_sha256='a' * 64, device_identity='orin-agx-01',
        jetpack_l4t='L4T 35.4.1', power_mode='MAXN', load_label='idle',
        capture_time_utc='2026-09-22T12:34:56Z', capture_time_source='reviewed log',
    )


def snapshot_bytes(path):
    return {p.relative_to(path).as_posix(): p.read_bytes()
            for p in path.rglob('*') if p.is_file()}


def refresh_checksums(source):
    names = sorted(p.name for p in source.iterdir() if p.name != 'source_sha256sums.txt')
    (source / 'source_sha256sums.txt').write_text(''.join(
        '%s  %s\n' % (hashlib.sha256((source / name).read_bytes()).hexdigest(), name)
        for name in names), encoding='utf-8')


class CaptureConditionTests(unittest.TestCase):
    def test_valid_condition_is_frozen_and_supports_finite_frequency(self):
        result = parse_capture_condition(condition_record())
        self.assertIsInstance(result, CaptureCondition)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.profile_id = 'changed'
        record = dict(condition_record(), workload_label='live', publish_frequency_hz=10)
        self.assertEqual(10, parse_capture_condition(record).publish_frequency_hz)
        record['capture_time_utc'] = '2026-09-22T12:34:56.123456+00:00'
        self.assertEqual(record['capture_time_utc'], parse_capture_condition(record).capture_time_utc)

    def test_every_required_field_is_required(self):
        for field in condition_record():
            with self.subTest(field=field):
                record = condition_record()
                del record[field]
                with self.assertRaisesRegex(BundleBuildError, field):
                    parse_capture_condition(record)

    def test_unknown_fields_and_non_objects_are_rejected(self):
        for record in (dict(condition_record(), typo=True), None, [], 'condition'):
            with self.subTest(record=record):
                with self.assertRaises(BundleBuildError):
                    parse_capture_condition(record)

    def test_each_string_rejects_blank_and_non_string(self):
        for field, value in condition_record().items():
            if isinstance(value, str):
                for invalid in ('', '   ', None, True, 1, 'bad\x00text'):
                    with self.subTest(field=field, invalid=invalid):
                        with self.assertRaisesRegex(BundleBuildError, field):
                            parse_capture_condition(dict(condition_record(), **{field: invalid}))

    def test_rates_counts_checksum_timestamp_and_boolean_are_strict(self):
        invalids = {
            'publish_frequency_hz': [True, False, 0, -1, math.inf, math.nan, '10', []],
            'raw_payload_bytes': [True, -1, 1.5, math.inf, None],
            'wire_payload_bytes': [False, -1, 1.5, math.nan, '1'],
            'engine_sha256': ['A' * 64, 'a' * 63, 'g' * 64, '../engine'],
            'capture_time_utc': ['2026-02-30T00:00:00Z', '2026-09-22',
                                 '2026-09-22T00:00:00+08:00', '2026-09-22T00:00:00',
                                 '2026-09-22T25:00:00Z'],
            'clock_comparable': [0, 1, 'true', None],
            'profile_id': ['../escape', 'a/b', '.', '..', '/absolute'],
        }
        for field, values in invalids.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(BundleBuildError, field):
                        parse_capture_condition(dict(condition_record(), **{field: value}))
        with self.assertRaisesRegex(BundleBuildError, 'publish_frequency_hz'):
            parse_capture_condition(dict(condition_record(), workload_label='live'))


class BundleTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/var/tmp/codriving-ditl')
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='bundle-test-', dir=str(scratch))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.input_root = self.base / 'inputs'
        self.source = self.input_root / 'source'
        shutil.copytree(str(FIXTURE), str(self.source))
        self.root = self.make_root('library')

    def make_root(self, name):
        root = self.base / name
        (root / 'v1/bundles').mkdir(parents=True)
        return root

    def request(self, **changes):
        values = dict(source_dir=self.source, library_root=self.root,
                      input_root=self.input_root,
                      condition=parse_capture_condition(condition_record()))
        values.update(changes)
        return BuildRequest(**values)

    def assert_clean(self):
        self.assertEqual([], list((self.root / 'v1/bundles').iterdir()))

    def test_identical_inputs_build_byte_identical_bundles(self):
        first = build_feature_bundle(self.request())
        second = build_feature_bundle(self.request(library_root=self.make_root('other')))
        self.assertEqual(first.feature_id, second.feature_id)
        self.assertEqual(snapshot_bytes(first.bundle_path), snapshot_bytes(second.bundle_path))
        self.assertEqual(FILES, set(snapshot_bytes(first.bundle_path)))
        self.assertEqual(self.source.joinpath('trace.jsonl').read_bytes(),
                         first.bundle_path.joinpath('raw/trace.jsonl').read_bytes())
        self.assertEqual(3, first.sample_count)
        self.assertEqual(64, len(first.feature_id))
        self.assertEqual(first.feature_id, first.bundle_path.name)
        for name, digest in [('manifest.json', first.manifest_sha256),
                             ('SHA256SUMS', first.checksums_sha256)]:
            self.assertEqual(hashlib.sha256((first.bundle_path / name).read_bytes()).hexdigest(), digest)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.sample_count = 10

    def test_relocated_source_builds_byte_identical_bundle(self):
        first = build_feature_bundle(self.request())
        other_input = self.base / 'relocated-input'
        other_source = other_input / 'different-source-name'
        shutil.copytree(str(self.source), str(other_source))
        second = build_feature_bundle(self.request(
            input_root=other_input, source_dir=other_source,
            library_root=self.make_root('relocated-library')))
        self.assertEqual(first.feature_id, second.feature_id)
        self.assertEqual(snapshot_bytes(first.bundle_path), snapshot_bytes(second.bundle_path))
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.checksums_sha256, second.checksums_sha256)
        for content in snapshot_bytes(second.bundle_path).values():
            self.assertNotIn(str(self.source).encode('utf-8'), content)
            self.assertNotIn(str(other_source).encode('utf-8'), content)

    def test_relocated_source_reuses_existing_bundle_without_replacement(self):
        first = build_feature_bundle(self.request())
        inode = first.bundle_path.stat().st_ino
        other_input = self.base / 'relocated-input'
        other_source = other_input / 'different-source-name'
        shutil.copytree(str(self.source), str(other_source))
        second = build_feature_bundle(self.request(input_root=other_input, source_dir=other_source))
        self.assertEqual(first, second)
        self.assertEqual(inode, second.bundle_path.stat().st_ino)
        self.assertEqual([first.bundle_path], list((self.root / 'v1/bundles').iterdir()))

    def test_manifest_report_summaries_and_samples_follow_contract(self):
        result = build_feature_bundle(self.request())
        data = snapshot_bytes(result.bundle_path)
        manifest = json.loads(data['manifest.json'])
        self.assertEqual('ditl.feature-library.v1', manifest['library_schema_version'])
        self.assertEqual('rsu-local-run', manifest['capture_run_id'])
        self.assertEqual('base_delay_ns', manifest['selected_delay_field'])
        self.assertEqual('ditl.trace.v2', manifest['source_trace_schema_version'])
        self.assertEqual('ditl.replay-sample.v1', manifest['replay_sample_schema_version'])
        self.assertEqual(['error', 'cycle'], manifest['exhaustion_policies'])
        self.assertEqual('raw/trace.jsonl', manifest['source_artifact_path'])
        self.assertNotIn('checksums_sha256', manifest)
        self.assertEqual(set(library.MANIFEST_FIELDS), set(manifest))
        for field, value in condition_record().items():
            self.assertEqual(value, manifest[field])
        report = json.loads(data['stats/normalization_report.json'])
        self.assertEqual((4, 1, 1, 3), tuple(report[k] for k in
                         ['total_count', 'warmup_count', 'rejected_count', 'accepted_count']))
        self.assertEqual({'warmup': 1}, report['rejection_reasons'])
        condition_bytes = (json.dumps(condition_record(), sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False) + '\n').encode('utf-8')
        condition_hash = hashlib.sha256(condition_bytes).hexdigest()
        self.assertEqual(condition_hash, report['capture_condition_id'])
        samples = [json.loads(line) for line in data['normalized/replay_samples.jsonl'].splitlines()]
        self.assertEqual([1, 2, 3], [s['source_sequence_id'] for s in samples])
        self.assertTrue(all(s['capture_condition_id'] == condition_hash for s in samples))
        summary = json.loads(data['stats/summary.json'])
        self.assertEqual({'count': 3, 'min': 34000000, 'max': 54000000,
                          'mean': 44000000, 'p50': 44000000, 'p95': 53000000,
                          'p99': 53800000}, summary['base_delay_ns'])
        self.assertEqual(24000000, summary['stage_ns']['transport_receive']['min'])
        self.assertEqual(3000000, summary['stage_ns']['acquisition_preprocess']['p99'])
        self.assertEqual(2621440, summary['raw_payload_bytes']['p50'])
        self.assertEqual(2621440, summary['wire_payload_bytes']['mean'])
        for name in ['manifest.json', 'stats/summary.json', 'stats/normalization_report.json',
                     'normalized/replay_samples.jsonl']:
            for line in data[name].splitlines(keepends=True):
                self.assertEqual((json.dumps(json.loads(line), sort_keys=True,
                                 separators=(',', ':'), ensure_ascii=False) + '\n').encode('utf-8'), line)
        checksum_lines = data['SHA256SUMS'].decode('utf-8').splitlines()
        self.assertEqual(sorted(FILES - {'SHA256SUMS'}), [line[66:] for line in checksum_lines])
        for line in checksum_lines:
            self.assertEqual(hashlib.sha256(data[line[66:]]).hexdigest(), line[:64])

    def test_condition_change_changes_feature_and_sample_identity(self):
        first = build_feature_bundle(self.request())
        changed = parse_capture_condition(dict(condition_record(), load_label='loaded'))
        second = build_feature_bundle(self.request(condition=changed))
        self.assertNotEqual(first.feature_id, second.feature_id)
        self.assertNotEqual((first.bundle_path / 'normalized/replay_samples.jsonl').read_bytes(),
                            (second.bundle_path / 'normalized/replay_samples.jsonl').read_bytes())

    def test_trace_byte_change_changes_feature_identity(self):
        first = build_feature_bundle(self.request())
        trace = self.source / 'trace.jsonl'
        trace.write_bytes(trace.read_bytes().replace(b'"frame_id":1', b'"frame_id":99'))
        refresh_checksums(self.source)
        second = build_feature_bundle(self.request())
        self.assertNotEqual(first.feature_id, second.feature_id)

    def test_identical_existing_bundle_returns_without_replacing_inode(self):
        first = build_feature_bundle(self.request())
        inode = first.bundle_path.stat().st_ino
        second = build_feature_bundle(self.request())
        self.assertEqual(first, second)
        self.assertEqual(inode, second.bundle_path.stat().st_ino)
        self.assertFalse(list((self.root / 'v1/bundles').glob('.*.tmp-*')))

    def test_existing_tampered_missing_extra_or_empty_bundle_is_not_overwritten(self):
        for mutation in ['tamper', 'missing', 'extra', 'empty']:
            with self.subTest(mutation=mutation):
                root = self.make_root(mutation)
                request = self.request(library_root=root)
                bundle = build_feature_bundle(request).bundle_path
                if mutation == 'tamper':
                    (bundle / 'manifest.json').write_text('tampered')
                elif mutation == 'missing':
                    (bundle / 'manifest.json').unlink()
                elif mutation == 'extra':
                    (bundle / 'extra').write_text('extra')
                else:
                    shutil.rmtree(str(bundle))
                    bundle.mkdir()
                before = snapshot_bytes(bundle)
                with self.assertRaises(BundleBuildError):
                    build_feature_bundle(request)
                self.assertEqual(before, snapshot_bytes(bundle))

    def test_failure_at_publish_cleans_staging(self):
        with mock.patch.object(library, '_atomic_publish', side_effect=OSError('publish failure')):
            with self.assertRaisesRegex(BundleBuildError, 'publish failure'):
                build_feature_bundle(self.request())
        self.assert_clean()

    def test_failure_after_each_staged_file_write_cleans_staging(self):
        original = library._write_file
        for failed_write in range(1, 11):
            calls = [0]
            def fail_after_write(*args, **kwargs):
                original(*args, **kwargs)
                calls[0] += 1
                if calls[0] == failed_write:
                    raise OSError('injected file write failure')
            with self.subTest(failed_write=failed_write):
                with mock.patch.object(library, '_write_file', side_effect=fail_after_write):
                    with self.assertRaisesRegex(BundleBuildError, 'file write failure'):
                        build_feature_bundle(self.request())
                self.assert_clean()

    def test_staging_is_private_sibling_and_files_and_directories_are_fsynced(self):
        original = library._atomic_publish
        seen_sync = []
        real_fsync = os.fsync
        def sync(fd):
            seen_sync.append(stat.S_IFMT(os.fstat(fd).st_mode))
            real_fsync(fd)
        def publish(parent_fd, staging_name, final_name):
            parent = self.root / 'v1/bundles'
            staging = parent / staging_name
            self.assertTrue(staging_name.startswith('.' + final_name + '.tmp-'))
            self.assertEqual(0o700, stat.S_IMODE(staging.stat().st_mode))
            self.assertEqual(FILES, set(snapshot_bytes(staging)))
            self.assertEqual(10, seen_sync.count(stat.S_IFREG))
            self.assertGreaterEqual(seen_sync.count(stat.S_IFDIR), 5)
            return original(parent_fd, staging_name, final_name)
        with mock.patch.object(library, '_atomic_publish', side_effect=publish), \
                mock.patch.object(library.os, 'fsync', side_effect=sync):
            build_feature_bundle(self.request())

    def test_source_modified_during_normalization_is_rejected(self):
        normalize = library.normalize_raw_trace
        def modify(*args, **kwargs):
            result = normalize(*args, **kwargs)
            with (self.source / 'trace.jsonl').open('ab') as handle:
                handle.write(b'\n')
            return result
        with mock.patch.object(library, 'normalize_raw_trace', side_effect=modify):
            with self.assertRaisesRegex(BundleBuildError, 'changed'):
                build_feature_bundle(self.request())
        self.assert_clean()

    def test_directory_replaced_during_build_fails_closed(self):
        for target_name in ('source', 'bundles'):
            with self.subTest(target=target_name):
                target = self.source if target_name == 'source' else self.root / 'v1/bundles'
                moved = target.with_name(target.name + '-moved')
                normalize = library.normalize_raw_trace
                def replace_directory(*args, **kwargs):
                    result = normalize(*args, **kwargs)
                    target.rename(moved)
                    target.mkdir()
                    if target_name == 'source':
                        for path in moved.iterdir():
                            shutil.copyfile(str(path), str(target / path.name))
                    return result
                try:
                    with mock.patch.object(library, 'normalize_raw_trace', side_effect=replace_directory):
                        with self.assertRaisesRegex(BundleBuildError, 'changed|replaced'):
                            build_feature_bundle(self.request())
                    if target_name == 'bundles':
                        self.assertEqual([], list(moved.iterdir()))
                    self.assert_clean()
                finally:
                    shutil.rmtree(str(target))
                    moved.rename(target)

    def test_symlink_loop_is_reported_as_bundle_error(self):
        loop = self.base / 'loop'
        loop.symlink_to(loop)
        with self.assertRaises(BundleBuildError):
            build_feature_bundle(self.request(input_root=loop))
        self.assert_clean()

    def test_source_symlink_at_file_directory_and_ancestor_is_rejected(self):
        for level in ['file', 'source', 'input_root']:
            with self.subTest(level=level):
                original = {'file': self.source / 'trace.jsonl', 'source': self.source,
                            'input_root': self.input_root}[level]
                moved = original.with_name(original.name + '-real')
                original.rename(moved)
                original.symlink_to(moved, target_is_directory=level != 'file')
                try:
                    with self.assertRaises(BundleBuildError):
                        build_feature_bundle(self.request())
                finally:
                    original.unlink()
                    moved.rename(original)
        self.assert_clean()

    def test_destination_symlink_at_root_parent_or_final_is_rejected(self):
        for level in ['root', 'v1', 'bundles', 'final']:
            with self.subTest(level=level):
                request = self.request(library_root=self.make_root('dest-' + level))
                root = request.library_root
                target = {'root': root, 'v1': root / 'v1', 'bundles': root / 'v1/bundles'}
                if level == 'final':
                    target[level] = build_feature_bundle(request).bundle_path
                original = target[level]
                moved = original.with_name(original.name + '-real')
                original.rename(moved)
                original.symlink_to(moved, target_is_directory=True)
                with self.assertRaises(BundleBuildError):
                    build_feature_bundle(request)

    def test_nested_existing_bundle_symlink_is_rejected(self):
        request = self.request()
        result = build_feature_bundle(request)
        original = result.bundle_path / 'manifest.json'
        moved = self.base / 'manifest-copy'
        original.rename(moved)
        original.symlink_to(moved)
        with self.assertRaises(BundleBuildError):
            build_feature_bundle(request)

    def test_dotdot_outside_missing_roots_and_unsupported_files_fail_closed(self):
        for changes in [dict(source_dir=self.source / '..' / 'source'),
                        dict(library_root=self.root / '..' / 'library'),
                        dict(source_dir=self.base),
                        dict(input_root=self.base / 'missing'),
                        dict(library_root=self.base / 'missing')]:
            with self.subTest(changes=changes):
                with self.assertRaises(BundleBuildError):
                    build_feature_bundle(self.request(**changes))
        (self.source / 'untrusted.pkl').write_bytes(b'not executable')
        with self.assertRaises(BundleBuildError):
            build_feature_bundle(self.request())
        self.assert_clean()

    def test_missing_required_source_and_non_regular_source_are_rejected(self):
        original = self.source / 'command_manifest.txt'
        saved = original.read_bytes()
        original.unlink()
        with self.assertRaises(BundleBuildError):
            build_feature_bundle(self.request())
        os.mkfifo(str(original))
        with self.assertRaises(BundleBuildError):
            build_feature_bundle(self.request())
        original.unlink()
        original.write_bytes(saved)
        self.assert_clean()

    def test_missing_bundles_parent_is_not_created_implicitly(self):
        shutil.rmtree(str(self.root / 'v1'))
        with self.assertRaises(BundleBuildError):
            build_feature_bundle(self.request())
        self.assertFalse((self.root / 'v1').exists())

    def test_source_checksum_provenance_is_copied_without_interpreting_paths(self):
        provenance = b'opaque historical checksums /outside/path; $(never execute)\n'
        (self.source / 'source_sha256sums.txt').write_bytes(provenance)
        result = build_feature_bundle(self.request())
        self.assertEqual(provenance, (result.bundle_path /
                         'provenance/source_sha256sums.txt').read_bytes())

    def test_invalid_json_empty_trace_no_measured_records_and_identity_mismatch_rejected(self):
        trace = self.source / 'trace.jsonl'
        original = trace.read_bytes()
        records = [json.loads(line) for line in original.splitlines()]
        invalids = [b'', b'{broken}\n', b'{}\n', b'null\n',
                    (json.dumps(dict(records[0], warmup=True)) + '\n').encode('utf-8')]
        for payload in invalids:
            with self.subTest(payload=payload[:40]):
                trace.write_bytes(payload)
                refresh_checksums(self.source)
                with self.assertRaises(BundleBuildError):
                    build_feature_bundle(self.request())
                self.assert_clean()
        trace.write_bytes(original)
        refresh_checksums(self.source)
        for field, value in [('model_id', 'wrong'), ('engine_sha256', 'b' * 64),
                             ('payload_semantics', 'wrong'), ('raw_payload_bytes', 1),
                             ('wire_payload_bytes', 1), ('clock_comparable', False),
                             ('codec', 'lz4'), ('clock_domain', 'ptp')]:
            with self.subTest(field=field):
                condition = parse_capture_condition(dict(condition_record(), **{field: value}))
                with self.assertRaises(BundleBuildError):
                    build_feature_bundle(self.request(condition=condition))
        self.assert_clean()

    def test_direct_dataclass_cannot_bypass_validation_and_policies_are_strict(self):
        with self.assertRaises(BundleBuildError):
            build_feature_bundle(self.request(condition=CaptureCondition(**dict(
                condition_record(), raw_payload_bytes=True))))
        for policies in [(), ('cycle',), ('error', 'oops'), ('error', 'error'), ['error']]:
            with self.subTest(policies=policies):
                with self.assertRaises(BundleBuildError):
                    build_feature_bundle(self.request(exhaustion_policies=policies))
        request = self.request(exhaustion_policies=('error',))
        result = build_feature_bundle(request)
        self.assertEqual(['error'], json.loads((result.bundle_path / 'manifest.json').read_text())['exhaustion_policies'])

    def test_policy_change_changes_feature_id(self):
        first = build_feature_bundle(self.request())
        second = build_feature_bundle(self.request(exhaustion_policies=('error',)))
        self.assertNotEqual(first.feature_id, second.feature_id)

    def test_concurrent_identical_bundle_is_verified_and_reused(self):
        real_publish = library._atomic_publish
        def race(parent_fd, staging_name, final_name):
            staging = self.root / 'v1/bundles' / staging_name
            destination = self.root / 'v1/bundles' / final_name
            shutil.copytree(str(staging), str(destination))
            return real_publish(parent_fd, staging_name, final_name)
        with mock.patch.object(library, '_atomic_publish', side_effect=race):
            result = build_feature_bundle(self.request())
        self.assertEqual(FILES, set(snapshot_bytes(result.bundle_path)))
        self.assertEqual([result.bundle_path], list((self.root / 'v1/bundles').iterdir()))

    def test_concurrent_destination_creation_is_never_overwritten(self):
        real_publish = library._atomic_publish
        def race(parent_fd, staging_name, final_name):
            destination = self.root / 'v1/bundles' / final_name
            destination.mkdir()
            return real_publish(parent_fd, staging_name, final_name)
        with mock.patch.object(library, '_atomic_publish', side_effect=race):
            with self.assertRaises(BundleBuildError):
                build_feature_bundle(self.request())
        directories = list((self.root / 'v1/bundles').iterdir())
        self.assertEqual(1, len(directories))
        self.assertEqual([], list(directories[0].iterdir()))
        self.assertFalse(directories[0].name.startswith('.'))


def resign_bundle(bundle_path):
    """Re-sign mutations so semantic tests get beyond digest verification."""
    names = sorted(FILES - {'SHA256SUMS'})
    (bundle_path / 'SHA256SUMS').write_text(''.join(
        '%s  %s\n' % (hashlib.sha256((bundle_path / name).read_bytes()).hexdigest(), name)
        for name in names), encoding='utf-8')


class VerificationTests(unittest.TestCase):
    setUp = BundleTests.setUp
    make_root = BundleTests.make_root
    request = BundleTests.request

    def api(self):
        self.assertTrue(hasattr(library, 'verify_feature_bundle'), 'verifier is missing')
        self.assertTrue(hasattr(library, 'register_bundle'), 'registration is missing')
        return library.verify_feature_bundle

    def test_valid_bundle_has_deeply_immutable_samples_and_manifest(self):
        verify = self.api()
        built = build_feature_bundle(self.request())
        verified = verify(built.bundle_path, self.root)
        self.assertEqual(built.feature_id, verified.feature_id)
        self.assertEqual('raw_samehost_idle', verified.profile_id)
        self.assertEqual(3, len(verified.samples))
        self.assertEqual((0, 1, 2), tuple(s.ordinal for s in verified.samples))
        self.assertEqual(built.checksums_sha256, verified.checksums_sha256)
        with self.assertRaises(TypeError):
            verified.manifest['profile_id'] = 'changed'
        with self.assertRaises(TypeError):
            verified.manifest['eligibility_criteria']['codec'] = 'changed'
        with self.assertRaises(TypeError):
            verified.samples[0].stage_ns['gpu_inference'] = 0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            verified.feature_id = 'changed'

    def test_every_protected_file_byte_change_and_missing_file_fails(self):
        verify = self.api()
        built = build_feature_bundle(self.request())
        for name in sorted(FILES):
            with self.subTest(name=name):
                target = built.bundle_path / name
                original = target.read_bytes()
                target.write_bytes(original + b'x')
                with self.assertRaises(library.BundleVerificationError):
                    verify(built.bundle_path, self.root)
                target.unlink()
                with self.assertRaises(library.BundleVerificationError):
                    verify(built.bundle_path, self.root)
                target.write_bytes(original)

    def test_extra_files_directories_symlinks_and_fifo_fail_closed(self):
        verify = self.api()
        built = build_feature_bundle(self.request())
        for directory in ('', 'raw', 'stats', 'provenance', 'normalized'):
            for kind in ('file', 'directory', 'symlink', 'fifo'):
                with self.subTest(directory=directory, kind=kind):
                    extra = built.bundle_path / directory / 'extra'
                    if kind == 'file':
                        extra.write_bytes(b'x')
                    elif kind == 'directory':
                        extra.mkdir()
                    elif kind == 'symlink':
                        extra.symlink_to(self.source)
                    else:
                        os.mkfifo(str(extra))
                    with self.assertRaises(library.BundleVerificationError):
                        verify(built.bundle_path, self.root)
                    extra.rmdir() if kind == 'directory' else extra.unlink()
        target = built.bundle_path / 'manifest.json'
        saved = target.read_bytes()
        target.unlink()
        target.symlink_to(self.source / 'trace.jsonl')
        with self.assertRaises(library.BundleVerificationError):
            verify(built.bundle_path, self.root)
        target.unlink()
        os.mkfifo(str(target))
        with self.assertRaises(library.BundleVerificationError):
            verify(built.bundle_path, self.root)
        target.unlink()
        target.write_bytes(saved)
        alias = self.base / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(library.BundleVerificationError):
            verify(alias / 'v1/bundles' / built.feature_id, alias)

    def test_resigned_manifest_and_sample_semantic_mutations_fail(self):
        verify = self.api()
        built = build_feature_bundle(self.request())
        original = snapshot_bytes(built.bundle_path)
        manifest_mutations = [
            ('accepted_count', 99), ('accepted_count', True), ('unknown', 1),
            ('feature_id', 'f' * 64), ('capture_condition_id', 'e' * 64),
            ('source_artifact_path', '/outside/trace.jsonl'),
            ('normalizer_version', '99'), ('capture_run_id', 'other'),
            ('exhaustion_policies', ['cycle']), ('profile_id', 'other'),
        ]
        sample_mutations = [
            ('ordinal', 9), ('unknown', 1), ('source_sequence_id', 99),
            ('source_trace_id', 'other'), ('capture_condition_id', 'e' * 64),
            ('base_delay_ns', 0), ('sample_id', 'f' * 64),
        ]
        for kind, mutations in [('manifest', manifest_mutations), ('sample', sample_mutations)]:
            for key, value in mutations:
                with self.subTest(kind=kind, key=key, value=value):
                    if kind == 'manifest':
                        record = json.loads(original['manifest.json'])
                        record[key] = value
                        (built.bundle_path / 'manifest.json').write_text(json.dumps(record) + '\n')
                    else:
                        records = [json.loads(line) for line in original['normalized/replay_samples.jsonl'].splitlines()]
                        records[0][key] = value
                        (built.bundle_path / 'normalized/replay_samples.jsonl').write_text(
                            ''.join(json.dumps(r) + '\n' for r in records))
                    resign_bundle(built.bundle_path)
                    with self.assertRaises(library.BundleVerificationError):
                        verify(built.bundle_path, self.root)
                    for name, data in original.items():
                        (built.bundle_path / name).write_bytes(data)
        lines = original['normalized/replay_samples.jsonl'].splitlines(keepends=True)
        (built.bundle_path / 'normalized/replay_samples.jsonl').write_bytes(b''.join(reversed(lines)))
        resign_bundle(built.bundle_path)
        with self.assertRaises(library.BundleVerificationError):
            verify(built.bundle_path, self.root)

    def test_bad_checksum_lists_and_duplicate_json_keys_fail(self):
        verify = self.api()
        built = build_feature_bundle(self.request())
        checksums = built.bundle_path / 'SHA256SUMS'
        saved = checksums.read_bytes()
        for invalid in (b'', saved + saved, saved.replace(b'manifest.json', b'../manifest.json'),
                        saved.replace(b'manifest.json', b'/manifest.json'), saved.split(b'\n', 1)[1]):
            checksums.write_bytes(invalid)
            with self.assertRaises(library.BundleVerificationError):
                verify(built.bundle_path, self.root)
        manifest = built.bundle_path / 'manifest.json'
        manifest.write_bytes(manifest.read_bytes().replace(b'{', b'{"accepted_count":3,', 1))
        resign_bundle(built.bundle_path)
        with self.assertRaises(library.BundleVerificationError):
            verify(built.bundle_path, self.root)

    def test_catalog_digest_path_and_profile_are_bound(self):
        verify = self.api()
        built = build_feature_bundle(self.request())
        catalog = self.root / 'catalog.json'
        library.register_bundle(built, catalog, self.root)
        verified = verify(built.bundle_path, self.root, catalog)
        self.assertEqual(built.feature_id, verified.feature_id)
        original = catalog.read_bytes()
        for field, value in [('relative_path', '../outside'), ('profile_id', 'other'),
                             ('checksums_sha256', 'a' * 64), ('unknown', 1)]:
            record = json.loads(original)
            record['features'][built.feature_id][field] = value
            catalog.write_text(json.dumps(record))
            with self.assertRaises(library.BundleVerificationError):
                verify(built.bundle_path, self.root, catalog)
        catalog.write_bytes(original)
        replacement = self.base / built.feature_id
        shutil.copytree(str(built.bundle_path), str(replacement))
        with self.assertRaisesRegex(library.BundleVerificationError, 'catalog path'):
            verify(replacement, self.root, catalog)
        checksum_file = built.bundle_path / 'SHA256SUMS'
        checksum_file.write_bytes(checksum_file.read_bytes() + b'\n')
        with self.assertRaises(library.BundleVerificationError):
            verify(built.bundle_path, self.root, catalog)

    def test_catalog_unknown_fields_missing_entries_and_symlinks_fail(self):
        verify = self.api()
        built = build_feature_bundle(self.request())
        catalog = self.root / 'catalog.json'
        for record in ({}, {'library_schema_version': 'wrong', 'features': {}},
                       {'library_schema_version': 'ditl.feature-library.v1', 'features': {}},
                       {'library_schema_version': 'ditl.feature-library.v1', 'features': {}, 'extra': 1}):
            catalog.write_text(json.dumps(record))
            with self.assertRaises(library.BundleVerificationError):
                verify(built.bundle_path, self.root, catalog)
        catalog.unlink()
        catalog.symlink_to(self.source / 'trace.jsonl')
        with self.assertRaises(library.BundleVerificationError):
            library.register_bundle(built, catalog, self.root)

    def test_bundle_directory_substitution_during_read_is_rejected(self):
        verify = self.api()
        built = build_feature_bundle(self.request())
        real_read = library._read_file
        replaced = []
        def substitute(fd, name):
            data = real_read(fd, name)
            if name == 'manifest.json' and not replaced:
                old = built.bundle_path.with_name('moved')
                built.bundle_path.rename(old)
                shutil.copytree(str(old), str(built.bundle_path))
                replaced.append(True)
            return data
        with mock.patch.object(library, '_read_file', side_effect=substitute):
            with self.assertRaises(library.BundleVerificationError):
                verify(built.bundle_path, self.root)

    def test_registration_failure_preserves_old_catalog_and_cleans_temporaries(self):
        self.api()
        import fcntl
        first = build_feature_bundle(self.request())
        second = build_feature_bundle(self.request(condition=parse_capture_condition(
            dict(condition_record(), profile_id='raw_samehost_busy'))))
        catalog = self.root / 'catalog.json'
        library.register_bundle(first, catalog, self.root)
        old = catalog.read_bytes()
        original_write = library._write_file
        original_fsync = os.fsync
        original_replace = os.replace
        def fail_write(fd, name, data):
            if '.tmp-' in name:
                original_write(fd, name, data[:3])
                raise OSError('injected temporary write failure')
            return original_write(fd, name, data)
        def fail_fsync(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError('injected file fsync failure')
            return original_fsync(fd)
        def fail_directory_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('injected parent fsync failure')
            return original_fsync(fd)
        def fail_replace(src, dst, **kwargs):
            if '.tmp-' in str(src):
                raise OSError('injected atomic replace failure')
            return original_replace(src, dst, **kwargs)
        failures = [(fcntl, 'flock', mock.Mock(side_effect=OSError('injected lock failure'))),
                    (library, '_write_file', fail_write), (os, 'fsync', fail_fsync),
                    (os, 'replace', fail_replace), (os, 'fsync', fail_directory_fsync)]
        for owner, method, failure in failures:
            with self.subTest(method=method, failure=str(failure)):
                with mock.patch.object(owner, method, side_effect=failure):
                    with self.assertRaises(library.BundleVerificationError):
                        library.register_bundle(second, catalog, self.root)
                self.assertEqual(old, catalog.read_bytes())
                self.assertEqual({'catalog.json', 'catalog.lock', 'v1'}, set(os.listdir(str(self.root))))
                library.verify_feature_bundle(first.bundle_path, self.root, catalog)

    def test_first_registration_parent_fsync_failure_removes_catalog(self):
        self.api()
        built = build_feature_bundle(self.request())
        catalog = self.root / 'catalog.json'
        original = os.fsync
        def fail(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('injected directory fsync failure')
            return original(fd)
        with mock.patch.object(os, 'fsync', side_effect=fail):
            with self.assertRaises(library.BundleVerificationError):
                library.register_bundle(built, catalog, self.root)
        self.assertFalse(catalog.exists())
        self.assertEqual({'catalog.lock', 'v1'}, set(os.listdir(str(self.root))))

    def test_two_process_registrations_preserve_both_complete_entries(self):
        self.api()
        import subprocess
        first = build_feature_bundle(self.request())
        second = build_feature_bundle(self.request(condition=parse_capture_condition(
            dict(condition_record(), profile_id='raw_samehost_busy'))))
        catalog = self.root / 'catalog.json'
        code = ('import sys; from pathlib import Path; '
                'from ditl_trace_library import verify_feature_bundle, register_bundle; '
                'root=Path(sys.argv[1]); bundle=verify_feature_bundle(Path(sys.argv[2]),root); '
                'register_bundle(bundle,root/"catalog.json",root)')
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent))
        processes = [subprocess.Popen([sys.executable, '-c', code, str(self.root), str(b.bundle_path)],
                                     env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                     for b in (first, second)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(0, process.returncode, (stdout, stderr))
        record = json.loads(catalog.read_bytes())
        self.assertEqual({first.feature_id, second.feature_id}, set(record['features']))
        before = catalog.read_bytes()
        library.register_bundle(first, catalog, self.root)
        self.assertEqual(before, catalog.read_bytes())
        for built in (first, second):
            library.verify_feature_bundle(built.bundle_path, self.root, catalog)
        self.assertEqual({'catalog.json', 'catalog.lock', 'v1'}, set(os.listdir(str(self.root))))


if __name__ == '__main__':
    unittest.main()
