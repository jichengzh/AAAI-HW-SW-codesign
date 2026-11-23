"""Publication rollback and bounded strict-ingest regressions."""
import json
import os
import shutil
import unittest
from unittest import mock

from . import test_ditl_trace_library as fixtures

library = fixtures.library
BundleBuildError = library.BundleBuildError
build_feature_bundle = library.build_feature_bundle
FILES = fixtures.FILES
snapshot_bytes = fixtures.snapshot_bytes


class HardeningTests(unittest.TestCase):
    setUp = fixtures.BundleTests.setUp
    make_root = fixtures.BundleTests.make_root
    request = fixtures.BundleTests.request
    assert_clean = fixtures.BundleTests.assert_clean

    def test_publish_boundary_directory_replacement_rolls_back_owned_bundle(self):
        for target_name in ('root', 'bundles'):
            with self.subTest(target=target_name):
                target = self.root if target_name == 'root' else self.root / 'v1/bundles'
                moved = self.base / ('moved-' + target_name)
                publish = library._atomic_publish
                def replace(parent_fd, staging_name, final_name):
                    target.rename(moved)
                    target.mkdir()
                    if target_name == 'root':
                        (target / 'v1/bundles').mkdir(parents=True)
                    publish(parent_fd, staging_name, final_name)
                try:
                    with mock.patch.object(library, '_atomic_publish', side_effect=replace):
                        with self.assertRaisesRegex(BundleBuildError, 'changed|replaced'):
                            build_feature_bundle(self.request())
                    escaped_parent = moved / 'v1/bundles' if target_name == 'root' else moved
                    self.assertEqual([], list(escaped_parent.iterdir()))
                    self.assert_clean()
                finally:
                    shutil.rmtree(str(target))
                    moved.rename(target)
                    for bundle in (self.root / 'v1/bundles').iterdir():
                        shutil.rmtree(str(bundle))

    def test_publish_parent_fsync_failure_removes_only_owned_bundle(self):
        for concurrent in (False, True):
            with self.subTest(concurrent=concurrent):
                publish, fsync = library._atomic_publish, os.fsync
                parent = self.root / 'v1/bundles'
                parent_inode = parent.stat().st_ino
                def race(fd, staging_name, final_name):
                    if concurrent:
                        shutil.copytree(str(parent / staging_name), str(parent / final_name))
                    publish(fd, staging_name, final_name)
                def fail(fd):
                    if os.fstat(fd).st_ino == parent_inode:
                        raise OSError('injected parent fsync failure')
                    fsync(fd)
                with mock.patch.object(library, '_atomic_publish', side_effect=race), \
                        mock.patch.object(os, 'fsync', side_effect=fail):
                    with self.assertRaisesRegex(BundleBuildError, 'parent fsync failure'):
                        build_feature_bundle(self.request())
                remaining = list(parent.iterdir())
                if concurrent:
                    self.assertEqual(1, len(remaining))
                    self.assertEqual(FILES, set(snapshot_bytes(remaining[0])))
                else:
                    self.assert_clean()

    def test_publish_rollback_preserves_replacement_inode(self):
        publish, fsync = library._atomic_publish, os.fsync
        parent = self.root / 'v1/bundles'
        parent_inode = parent.stat().st_ino
        def replace(fd, staging_name, final_name):
            publish(fd, staging_name, final_name)
            (parent / final_name).rename(self.base / 'owned-bundle')
            (parent / final_name).mkdir()
            (parent / final_name / 'concurrent').write_bytes(b'keep')
        def fail(fd):
            if os.fstat(fd).st_ino == parent_inode:
                raise OSError('injected parent fsync failure')
            fsync(fd)
        with mock.patch.object(library, '_atomic_publish', side_effect=replace), \
                mock.patch.object(os, 'fsync', side_effect=fail):
            with self.assertRaises(BundleBuildError):
                build_feature_bundle(self.request())
        remaining = list(parent.iterdir())
        self.assertEqual(1, len(remaining))
        self.assertEqual({'concurrent': b'keep'}, snapshot_bytes(remaining[0]))

    def test_raw_duplicate_keys_and_nonfinite_extensions_fail_build_and_verify(self):
        trace = self.source / 'trace.jsonl'
        original = trace.read_bytes()
        extensions = (b'"extra":0,"extra":1,', b'"extra":{"a":0,"a":1},',
                      b'"extra":NaN,', b'"extra":Infinity,', b'"extra":-Infinity,',
                      b'"extra":1e999,', b'"extra":{"nested":[1e999]},')
        for extension in extensions:
            with self.subTest(extension=extension):
                trace.write_bytes(original.replace(b'{', b'{' + extension, 1))
                # Emulate a legacy permissive importer to create a hash-consistent
                # invalid bundle; only verification below uses the real loader.
                with mock.patch.object(library, '_strict_json', side_effect=json.loads):
                    legacy = build_feature_bundle(self.request())
                with self.subTest(operation='verify'):
                    with self.assertRaises(library.BundleVerificationError):
                        library.verify_feature_bundle(legacy.bundle_path, self.root)
                with self.subTest(operation='build'):
                    with self.assertRaises(BundleBuildError):
                        build_feature_bundle(self.request())
                shutil.rmtree(str(legacy.bundle_path))

    def test_required_source_file_size_boundary_and_overflow(self):
        for name in library.SOURCE_NAMES:
            with self.subTest(name=name):
                original = (self.source / name).read_bytes()
                # Scale the limit for each file so the same ingest path is exercised
                # without generating five 64 MiB fixture copies per test run.
                limit = max(len(p.read_bytes()) for p in self.source.iterdir())
                padding = b' ' * (limit - len(original))
                (self.source / name).write_bytes(padding + original)
                try:
                    with mock.patch.object(library, 'MAX_SOURCE_FILE_BYTES', limit, create=True):
                        accepted = build_feature_bundle(self.request())
                        library.verify_feature_bundle(accepted.bundle_path, self.root)
                        shutil.rmtree(str(accepted.bundle_path))
                        (self.source / name).write_bytes(b' ' + padding + original)
                        with self.assertRaisesRegex(BundleBuildError, 'size|bytes|limit'):
                            build_feature_bundle(self.request())
                finally:
                    for bundle in (self.root / 'v1/bundles').iterdir():
                        shutil.rmtree(str(bundle))
                    (self.source / name).write_bytes(original)

    def test_oversized_source_is_rejected_before_reading(self):
        target = self.source / 'device_snapshot.txt'
        with target.open('wb') as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
        fdopen = os.fdopen
        target_inode = target.stat().st_ino
        def guard(fd, *args, **kwargs):
            if os.fstat(fd).st_ino == target_inode:
                self.fail('oversized source reached the read boundary')
            return fdopen(fd, *args, **kwargs)
        with mock.patch.object(os, 'fdopen', side_effect=guard):
            with self.assertRaisesRegex(BundleBuildError, 'size|bytes|limit'):
                build_feature_bundle(self.request())
        self.assert_clean()

    def test_source_file_accepts_exactly_sixty_four_mib(self):
        target = self.source / 'device_snapshot.txt'
        with target.open('wb') as stream:
            stream.truncate(64 * 1024 * 1024)
        built = build_feature_bundle(self.request())
        self.assertEqual(64 * 1024 * 1024,
                         (built.bundle_path / 'provenance/device_snapshot.txt').stat().st_size)
        library.verify_feature_bundle(built.bundle_path, self.root)

    def test_raw_parser_accepts_record_limit_and_rejects_the_next_record(self):
        self.assertTrue(hasattr(library, '_jsonl_records'), 'bounded raw parser is missing')
        self.assertEqual(100000, len(library._jsonl_records(b'{}\n' * 100000)))
        with self.assertRaisesRegex(BundleBuildError, 'records|limit'):
            library._jsonl_records(b'{}\n' * 100001)

    def test_growing_source_read_is_bounded_and_rejected(self):
        target = self.source / 'device_snapshot.txt'
        fdopen = os.fdopen
        limit = max(p.stat().st_size for p in self.source.iterdir())
        target_inode = target.stat().st_ino
        def grow(fd, *args, **kwargs):
            if os.fstat(fd).st_ino == target_inode:
                target.write_bytes(b'x' * (limit + 1))
            return fdopen(fd, *args, **kwargs)
        with mock.patch.object(library, 'MAX_SOURCE_FILE_BYTES', limit, create=True), \
                mock.patch.object(os, 'fdopen', side_effect=grow):
            with self.assertRaisesRegex(BundleBuildError, 'size|bytes|limit'):
                build_feature_bundle(self.request())
        self.assert_clean()

    def test_jsonl_line_boundary_accepts_eight_mib_and_rejects_one_more_byte(self):
        trace = self.source / 'trace.jsonl'
        first, rest = trace.read_bytes().split(b'\n', 1)
        line = first + b' ' * (8 * 1024 * 1024 - len(first) - 1) + b'\n'
        trace.write_bytes(line + rest)
        accepted = build_feature_bundle(self.request())
        library.verify_feature_bundle(accepted.bundle_path, self.root)
        trace.write_bytes(b' ' + line + rest)
        with self.assertRaisesRegex(BundleBuildError, 'line|bytes|limit'):
            build_feature_bundle(self.request())

    def test_normalized_jsonl_overlong_line_is_rejected_before_parsing(self):
        built = build_feature_bundle(self.request())
        target = built.bundle_path / 'normalized/replay_samples.jsonl'
        first, rest = target.read_bytes().split(b'\n', 1)
        target.write_bytes(first + b' ' * (8 * 1024 * 1024 - len(first)) + b'\n' + rest)
        fixtures.resign_bundle(built.bundle_path)
        with self.assertRaisesRegex(library.BundleVerificationError, 'line.*limit'):
            library.verify_feature_bundle(built.bundle_path, self.root)

    def test_record_limit_counts_warmup_and_measured_records(self):
        with mock.patch.object(library, 'MAX_TRACE_RECORDS', 4, create=True):
            built = build_feature_bundle(self.request())
            library.verify_feature_bundle(built.bundle_path, self.root)
        with mock.patch.object(library, 'MAX_TRACE_RECORDS', 3, create=True):
            with self.assertRaisesRegex(BundleBuildError, 'records|limit'):
                build_feature_bundle(self.request())
            with self.assertRaisesRegex(library.BundleVerificationError, 'records|limit'):
                library.verify_feature_bundle(built.bundle_path, self.root)

    def test_json_depth_boundary_ignores_escaped_strings_and_rejects_excess(self):
        trace = self.source / 'trace.jsonl'
        original = trace.read_bytes()
        for depth in (64, 65, 2000):
            with self.subTest(depth=depth):
                extension = b'"extra":' + b'[' * (depth - 1) + b'"[\\\"{"' + b']' * (depth - 1) + b','
                trace.write_bytes(original.replace(b'{', b'{' + extension, 1))
                if depth == 64:
                    built = build_feature_bundle(self.request())
                    library.verify_feature_bundle(built.bundle_path, self.root)
                else:
                    try:
                        with self.assertRaisesRegex(BundleBuildError, 'depth|nesting'):
                            build_feature_bundle(self.request())
                    except RecursionError:
                        self.fail('nesting must fail with BundleBuildError before JSON recursion')


if __name__ == '__main__':
    unittest.main()
