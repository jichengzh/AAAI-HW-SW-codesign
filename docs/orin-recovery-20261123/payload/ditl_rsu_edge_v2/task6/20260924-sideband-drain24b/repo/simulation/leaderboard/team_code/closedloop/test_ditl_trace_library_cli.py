"""CPU-only CLI integration tests using real source bundles and processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import test_ditl_trace_library as fixture_tests

REPO = Path(__file__).resolve().parents[4]


class CatalogCancellationTests(unittest.TestCase):
    setUp = fixture_tests.BundleTests.setUp
    make_root = fixture_tests.BundleTests.make_root
    request = fixture_tests.BundleTests.request

    def test_catalog_cancellation_restores_previous_bytes_and_propagates(self):
        library = fixture_tests.library
        built = library.build_feature_bundle(self.request())
        catalog = self.root / 'catalog.json'
        empty = b'{"library_schema_version":"ditl.feature-library.v1","features":{}}\n'
        root_inode, fsync = self.root.stat().st_ino, os.fsync
        for previous in (None, empty):
            for error_type in (KeyboardInterrupt, SystemExit):
                with self.subTest(previous=previous, error=error_type):
                    if previous is not None:
                        catalog.write_bytes(previous)
                    failure = error_type('injected cancellation')
                    def cancel(fd):
                        if os.fstat(fd).st_ino == root_inode:
                            raise failure
                        fsync(fd)
                    with mock.patch.object(os, 'fsync', side_effect=cancel):
                        with self.assertRaises(error_type) as raised:
                            library.register_bundle(built, catalog, self.root)
                    self.assertIs(failure, raised.exception)
                    if previous is None:
                        self.assertFalse(catalog.exists())
                    else:
                        self.assertEqual(previous, catalog.read_bytes())
                        catalog.unlink()
                    self.assertEqual({'v1', 'catalog.lock'}, set(os.listdir(str(self.root))))

    def test_catalog_cancellation_preserves_concurrent_replacement(self):
        library = fixture_tests.library
        built = library.build_feature_bundle(self.request())
        catalog = self.root / 'catalog.json'
        concurrent = b'{"library_schema_version":"ditl.feature-library.v1","features":{}}\n'
        catalog.write_bytes(concurrent)
        root_inode, fsync = self.root.stat().st_ino, os.fsync
        failure = KeyboardInterrupt('injected cancellation after concurrent replacement')
        def replace_then_cancel(fd):
            if os.fstat(fd).st_ino == root_inode:
                replacement = self.root / 'replacement'
                replacement.write_bytes(concurrent + b' ')
                replacement.replace(catalog)
                raise failure
            fsync(fd)
        with mock.patch.object(os, 'fsync', side_effect=replace_then_cancel):
            with self.assertRaises(KeyboardInterrupt) as raised:
                library.register_bundle(built, catalog, self.root)
        self.assertIs(failure, raised.exception)
        self.assertEqual(concurrent + b' ', catalog.read_bytes())
        self.assertEqual({'v1', 'catalog.lock', 'catalog.json'}, set(os.listdir(str(self.root))))


class TraceLibraryCliTests(unittest.TestCase):
    setUp = fixture_tests.BundleTests.setUp
    make_root = fixture_tests.BundleTests.make_root

    def run_cli(self, name, *args):
        return subprocess.run([sys.executable, str(REPO / 'scripts' / name)] + list(args),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=20)

    def build(self, *extra):
        condition = self.base / 'condition.json'
        condition.write_text(json.dumps(fixture_tests.condition_record()))
        return self.run_cli('ditl_build_feature_bundle.py', '--source', str(self.source),
                            '--input-root', str(self.input_root), '--library-root', str(self.root),
                            '--condition', str(condition), *extra)

    def test_build_register_result_json_and_all_verify_selectors(self):
        output = self.base / 'result.json'
        built = self.build('--register', '--result-json', str(output))
        self.assertEqual(0, built.returncode, built.stderr)
        record = json.loads(output.read_text())
        self.assertEqual(record, json.loads(built.stdout))
        self.assertEqual(3, record['sample_count'])
        selectors = [('--bundle-path', record['bundle_path']),
                     ('--feature-id', record['feature_id']), ('--profile', 'raw_samehost_idle')]
        for selector, value in selectors:
            verified = self.run_cli('ditl_verify_feature_bundle.py', '--library-root', str(self.root),
                                    '--catalog', str(self.root / 'catalog.json'), selector, value)
            self.assertEqual(0, verified.returncode, verified.stderr)
            summary = json.loads(verified.stdout)
            self.assertEqual(record['feature_id'], summary['feature_id'])
            self.assertEqual(3, summary['sample_count'])
            self.assertEqual(json.dumps(summary, sort_keys=True, separators=(',', ':')) + '\n',
                             verified.stdout)
        (Path(record['bundle_path']) / 'raw/trace.jsonl').write_bytes(b'tampered')
        failed = self.run_cli('ditl_verify_feature_bundle.py', '--library-root', str(self.root),
                              '--feature-id', record['feature_id'])
        self.assertNotEqual(0, failed.returncode)
        self.assertEqual('', failed.stdout)
        self.assertNotIn('Traceback', failed.stderr)

    def test_missing_conflicting_invalid_selectors_and_bad_condition_fail(self):
        for selectors in ([], ['--feature-id', '../escape'],
                          ['--feature-id', 'a' * 64, '--profile', 'profile'],
                          ['--profile', 'unregistered']):
            result = self.run_cli('ditl_verify_feature_bundle.py', '--library-root', str(self.root),
                                  *selectors)
            self.assertNotEqual(0, result.returncode)
            self.assertEqual('', result.stdout)
            self.assertNotIn('Traceback', result.stderr)
        invalid = self.base / 'invalid.json'
        invalid.write_text('{}')
        result = self.run_cli('ditl_build_feature_bundle.py', '--source', str(self.source),
                              '--input-root', str(self.input_root), '--library-root', str(self.root),
                              '--condition', str(invalid))
        self.assertNotEqual(0, result.returncode)
        self.assertEqual('', result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_profile_selection_requires_unique_catalog_match(self):
        built = self.build('--register')
        self.assertEqual(0, built.returncode, built.stderr)
        catalog = self.root / 'catalog.json'
        record = json.loads(catalog.read_text())
        feature = next(iter(record['features'].values()))
        record['features']['f' * 64] = dict(feature, relative_path='v1/bundles/' + 'f' * 64)
        catalog.write_text(json.dumps(record))
        result = self.run_cli('ditl_verify_feature_bundle.py', '--library-root', str(self.root),
                              '--profile', 'raw_samehost_idle')
        self.assertNotEqual(0, result.returncode)
        self.assertEqual('', result.stdout)
        self.assertIn('unique', result.stderr)

    def test_condition_and_result_parent_symlinks_fail_closed(self):
        condition = self.base / 'condition.json'
        condition.write_text(json.dumps(fixture_tests.condition_record()))
        alias = self.base / 'condition-link.json'
        alias.symlink_to(condition)
        result = self.run_cli('ditl_build_feature_bundle.py', '--source', str(self.source),
                              '--input-root', str(self.input_root), '--library-root', str(self.root),
                              '--condition', str(alias))
        self.assertNotEqual(0, result.returncode)
        self.assertEqual('', result.stdout)
        output_dir = self.base / 'outputs'
        output_dir.mkdir()
        output_alias = self.base / 'outputs-link'
        output_alias.symlink_to(output_dir, target_is_directory=True)
        result = self.build('--result-json', str(output_alias / 'result.json'))
        self.assertNotEqual(0, result.returncode)
        self.assertEqual([], list(output_dir.iterdir()))

    def test_help_and_imports_do_not_load_heavy_runtime_modules(self):
        for name in ('ditl_build_feature_bundle.py', 'ditl_verify_feature_bundle.py'):
            result = self.run_cli(name, '--help')
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn('usage:', result.stdout)
            code = '''import runpy, sys
class RejectHeavy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'tensorrt', 'carla', 'rospy', 'rclpy'):
            raise RuntimeError('forbidden heavy dependency: ' + fullname)
sys.meta_path.insert(0, RejectHeavy())
runpy.run_path(sys.argv[1], run_name='dependency_check')
'''
            result = subprocess.run([sys.executable, '-c', code, str(REPO / 'scripts' / name)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, timeout=10)
            self.assertEqual(0, result.returncode, result.stderr)


class CatalogBoundaryTests(unittest.TestCase):
    setUp = fixture_tests.BundleTests.setUp
    make_root = fixture_tests.BundleTests.make_root
    request = fixture_tests.BundleTests.request

    def test_bundle_changed_while_waiting_for_catalog_lock_is_not_registered(self):
        import fcntl
        from unittest import mock
        library = fixture_tests.library
        built = library.build_feature_bundle(self.request())
        catalog = self.root / 'catalog.json'
        original = fcntl.flock
        def change_bundle(fd, operation):
            original(fd, operation)
            (built.bundle_path / 'raw/trace.jsonl').write_bytes(b'changed during lock wait')
        with mock.patch.object(fcntl, 'flock', side_effect=change_bundle):
            with self.assertRaises(library.BundleVerificationError):
                library.register_bundle(built, catalog, self.root)
        self.assertFalse(catalog.exists())

    def test_catalog_parent_replacement_during_write_is_rejected(self):
        from unittest import mock
        library = fixture_tests.library
        built = library.build_feature_bundle(self.request())
        catalog = self.root / 'catalog.json'
        library.register_bundle(built, catalog, self.root)
        old = catalog.read_bytes()
        original = library._write_file
        moved = self.base / 'moved-root'
        def substitute(fd, name, data):
            original(fd, name, data)
            if '.tmp-' in name:
                self.root.rename(moved)
                self.root.mkdir()
                catalog.write_bytes(old)
        with mock.patch.object(library, '_write_file', side_effect=substitute):
            with self.assertRaises(library.BundleVerificationError):
                library.register_bundle(built, catalog, self.root)
        self.assertEqual(old, catalog.read_bytes())
        self.assertEqual(old, (moved / 'catalog.json').read_bytes())
        self.assertEqual({'catalog.json', 'catalog.lock', 'v1'}, set(os.listdir(str(moved))))

    def test_lock_symlink_and_catalog_conflict_are_rejected_without_changes(self):
        library = fixture_tests.library
        built = library.build_feature_bundle(self.request())
        catalog = self.root / 'catalog.json'
        lock = self.root / 'catalog.lock'
        lock.symlink_to(self.source / 'trace.jsonl')
        with self.assertRaises(library.BundleVerificationError):
            library.register_bundle(built, catalog, self.root)
        self.assertFalse(catalog.exists())
        lock.unlink()
        library.register_bundle(built, catalog, self.root)
        record = json.loads(catalog.read_text())
        record['features'][built.feature_id]['checksums_sha256'] = 'f' * 64
        catalog.write_text(json.dumps(record))
        old = catalog.read_bytes()
        with self.assertRaises(library.BundleVerificationError):
            library.register_bundle(built, catalog, self.root)
        self.assertEqual(old, catalog.read_bytes())


class ResultPublicationTests(unittest.TestCase):
    make_root = fixture_tests.BundleTests.make_root

    def setUp(self):
        import runpy
        fixture_tests.BundleTests.setUp(self)
        self.output_dir = self.base / 'outputs'
        self.output_dir.mkdir()
        self.output = self.output_dir / 'result.json'
        condition = self.base / 'condition.json'
        condition.write_text(json.dumps(fixture_tests.condition_record()))
        self.main = runpy.run_path(str(REPO / 'scripts/ditl_build_feature_bundle.py'))['main']
        self.arguments = ['--source', str(self.source), '--input-root', str(self.input_root),
                          '--library-root', str(self.root), '--condition', str(condition),
                          '--result-json', str(self.output)]

    def invoke(self):
        import contextlib
        import io
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                status = self.main(self.arguments)
            except SystemExit as error:
                status = error.code
        return status, stdout.getvalue(), stderr.getvalue()

    def result_descriptor(self, fd):
        return Path(os.readlink('/proc/self/fd/%d' % fd)).parent == self.output_dir

    def patch_result_writer(self, fail=False, fail_flush=False):
        from unittest import mock
        original = os.fdopen
        class ShortWriter:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                self.stream.__enter__()
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def write(self, data):
                count = self.stream.write(data[:7])
                if fail:
                    raise OSError('injected partial result write failure')
                return count
            def flush(self):
                self.stream.flush()
                if fail_flush:
                    raise OSError('injected result flush failure')
        def open_stream(fd, mode, *args, **kwargs):
            stream = original(fd, mode, *args, **kwargs)
            return ShortWriter(stream) if mode == 'wb' and self.result_descriptor(fd) else stream
        return mock.patch.object(os, 'fdopen', side_effect=open_stream)

    def assert_failed_without_result(self):
        status, stdout, stderr = self.invoke()
        self.assertNotEqual(0, status)
        self.assertEqual('', stdout)
        self.assertIn('error:', stderr)
        self.assertEqual([], list(self.output_dir.iterdir()))

    def test_temp_fstat_failure_closes_descriptor_and_removes_every_candidate(self):
        import errno
        from unittest import mock
        original = os.fstat
        descriptors = []
        def fail(fd):
            if self.result_descriptor(fd):
                descriptors.append(fd)
                raise OSError('injected temporary fstat failure')
            return original(fd)
        with mock.patch.object(os, 'fstat', side_effect=fail):
            status, stdout, stderr = self.invoke()
        self.assertNotEqual(0, status)
        self.assertEqual('', stdout)
        self.assertIn('injected temporary fstat failure', stderr)
        self.assertEqual(1, len(descriptors))
        with self.subTest(check='descriptor closed'):
            try:
                with self.assertRaises(OSError) as caught:
                    original(descriptors[0])
                self.assertEqual(errno.EBADF, caught.exception.errno)
            finally:
                # Avoid leaking the reproducer's fd during the expected RED run.
                if Path('/proc/self/fd/%d' % descriptors[0]).exists():
                    os.close(descriptors[0])
        with self.subTest(check='no files left behind'):
            self.assertEqual([], list(self.output_dir.iterdir()))

    def test_flush_failure_removes_candidates_and_preserves_existing_target(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.output.write_bytes(b'previous result remains unchanged')
                with self.patch_result_writer(fail_flush=True):
                    status, stdout, stderr = self.invoke()
                self.assertNotEqual(0, status)
                self.assertEqual('', stdout)
                self.assertIn('injected result flush failure', stderr)
                expected = [self.output] if existing else []
                self.assertEqual(expected, list(self.output_dir.iterdir()))
                if existing:
                    self.assertEqual(b'previous result remains unchanged', self.output.read_bytes())

    def test_partial_write_failure_leaves_no_final_result_or_temporary(self):
        with self.patch_result_writer(fail=True):
            self.assert_failed_without_result()

    def test_short_writes_are_completed_before_publishing(self):
        with self.patch_result_writer():
            status, stdout, stderr = self.invoke()
        self.assertEqual(0, status, stderr)
        self.assertEqual(stdout.encode('utf-8'), self.output.read_bytes())
        self.assertEqual([self.output], list(self.output_dir.iterdir()))

    def test_result_file_fsync_failure_leaves_no_final_result_or_temporary(self):
        from unittest import mock
        original = os.fsync
        def fail(fd):
            if self.result_descriptor(fd):
                raise OSError('injected result file fsync failure')
            return original(fd)
        with mock.patch.object(os, 'fsync', side_effect=fail):
            self.assert_failed_without_result()

    def test_result_parent_fsync_failure_rolls_back_published_result(self):
        from unittest import mock
        original = os.fsync
        def fail(fd):
            if Path(os.readlink('/proc/self/fd/%d' % fd)) == self.output_dir:
                raise OSError('injected result parent fsync failure')
            return original(fd)
        with mock.patch.object(os, 'fsync', side_effect=fail):
            self.assert_failed_without_result()

    def test_existing_result_is_preserved_byte_for_byte_without_temporary_residue(self):
        original = b'existing user-owned result\n'
        self.output.write_bytes(original)
        status, stdout, stderr = self.invoke()
        self.assertNotEqual(0, status)
        self.assertEqual('', stdout)
        self.assertIn('error:', stderr)
        self.assertEqual(original, self.output.read_bytes())
        self.assertEqual([self.output], list(self.output_dir.iterdir()))


if __name__ == '__main__':
    unittest.main()
