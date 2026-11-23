#!/usr/bin/env python3
"""Verify a complete deployment trace bundle before exposing any samples."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'simulation/leaderboard/team_code/closedloop'))
from ditl_trace_library import (
    BundleBuildError, BundleVerificationError, _SHA256, _absolute, _canonical,
    _catalog_location, _directory, _parse_catalog, _read_file,
    verify_feature_bundle,
)


def _select_bundle(args):
    root = _absolute(args.library_root)
    catalog = args.catalog
    if args.profile is not None:
        catalog = _catalog_location(catalog or root / 'catalog.json', root)
        with _directory(catalog.parent) as fd:
            entries = _parse_catalog(_read_file(fd, catalog.name))['features']
        matches = [feature_id for feature_id, entry in entries.items()
                   if entry['profile_id'] == args.profile]
        if len(matches) != 1:
            raise BundleVerificationError('profile must have a unique registered feature')
        return root / 'v1/bundles' / matches[0], root, catalog
    if args.feature_id is not None:
        if _SHA256.fullmatch(args.feature_id) is None:
            raise BundleVerificationError('feature_id must be a lowercase SHA-256 digest')
        return root / 'v1/bundles' / args.feature_id, root, catalog
    return args.bundle_path, root, catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--catalog', type=Path)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--bundle-path', type=Path)
    selector.add_argument('--feature-id')
    selector.add_argument('--profile', '--profile-id', '--registered-profile', dest='profile')
    args = parser.parse_args(argv)
    try:
        bundle, root, catalog = _select_bundle(args)
        verified = verify_feature_bundle(bundle, root, catalog)
        if args.profile is not None and verified.profile_id != args.profile:
            raise BundleVerificationError('catalog profile changed during verification')
        summary = {'feature_id': verified.feature_id, 'profile_id': verified.profile_id,
                   'bundle_path': str(verified.bundle_path),
                   'sample_count': len(verified.samples),
                   'checksums_sha256': verified.checksums_sha256}
        sys.stdout.write(_canonical(summary).decode('utf-8'))
        return 0
    except (BundleBuildError, BundleVerificationError, OSError, ValueError, TypeError) as error:
        parser.exit(1, 'error: %s\n' % error)


if __name__ == '__main__':
    sys.exit(main())
