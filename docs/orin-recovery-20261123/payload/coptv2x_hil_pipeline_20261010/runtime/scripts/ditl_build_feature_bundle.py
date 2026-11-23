#!/usr/bin/env python3
"""Build a deterministic CPU-only deployment trace feature bundle."""
import argparse
from dataclasses import asdict
import os
import secrets
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'simulation/leaderboard/team_code/closedloop'))
from ditl_trace_library import (
    BuildRequest, BundleBuildError, BundleVerificationError, _canonical,
    _atomic_publish, _basename, _directory, _read_file, _strict_json,
    _verify_directory_binding,
    build_feature_bundle, parse_capture_condition, register_bundle,
)


def _write_result_bytes(fd, payload):
    with os.fdopen(fd, 'wb', closefd=False) as stream:
        offset = 0
        while offset < len(payload):
            written = stream.write(payload[offset:])
            if written is None or written <= 0:
                raise OSError('result JSON write made no progress')
            offset += written
        stream.flush()
        os.fsync(fd)


def _remove_owned_result(parent_fd, name, identity):
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    # None is only used for our exclusively created temporary if fstat failed.
    if identity is None or (current.st_dev, current.st_ino) == identity:
        os.unlink(name, dir_fd=parent_fd)


def _publish_result_json(output, payload):
    output = output.absolute()
    _basename(output.name)
    temporary = '.ditl-result.tmp-' + secrets.token_hex(16)
    with _directory(output.parent) as parent_fd:
        identity, published = None, False
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            _write_result_bytes(fd, payload)
            closing_fd, fd = fd, None
            os.close(closing_fd)
            _verify_directory_binding(output.parent, parent_fd)
            _atomic_publish(parent_fd, temporary, output.name)
            published = True
            os.fsync(parent_fd)
            _verify_directory_binding(output.parent, parent_fd)
        except BaseException:
            if published:
                _remove_owned_result(parent_fd, output.name, identity)
                os.fsync(parent_fd)
            raise
        finally:
            try:
                if fd is not None:
                    os.close(fd)
            finally:
                _remove_owned_result(parent_fd, temporary, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', '--source-dir', dest='source', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--condition', '--condition-json', dest='condition', type=Path, required=True,
                        help='Reviewed capture condition JSON file')
    parser.add_argument('--register', action='store_true', help='Register in library_root/catalog.json')
    parser.add_argument('--result-json', type=Path, help='Write result to a new JSON file')
    args = parser.parse_args(argv)
    try:
        with _directory(args.condition.absolute().parent) as condition_fd:
            condition = parse_capture_condition(_strict_json(
                _read_file(condition_fd, args.condition.name)))
        result = build_feature_bundle(BuildRequest(args.source, args.library_root,
                                                   args.input_root, condition))
        if args.register:
            register_bundle(result, args.library_root / 'catalog.json', args.library_root)
        summary = dict(asdict(result), bundle_path=str(result.bundle_path))
        payload = _canonical(summary)
        if args.result_json is not None:
            _publish_result_json(args.result_json, payload)
        sys.stdout.write(payload.decode('utf-8'))
        return 0
    except (BundleBuildError, BundleVerificationError, OSError, ValueError, TypeError) as error:
        parser.exit(1, 'error: %s\n' % error)


if __name__ == '__main__':
    sys.exit(main())
