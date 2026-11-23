#!/usr/bin/env python3
"""Read-only recovery checks; run with Python >=3.8. Does not restore or execute payload."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def relative_path(relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('Expected nonempty relative path')
    name = Path(relative)
    if name.is_absolute() or '..' in name.parts or name.as_posix() != relative:
        raise ValueError('Unsafe relative path: ' + relative)
    return name


def check_file(root, relative, size, expected):
    name = relative_path(relative)
    path = root / name
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError('Path escapes root: ' + str(path))
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing regular file: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != size or digest.hexdigest() != expected:
        raise ValueError('Size/SHA256 mismatch: ' + str(path))


def four_records(manifest, key):
    rows = manifest.get(key)
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError('Copt manifest requires exactly four ' + key)
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid Copt record in ' + key)
        name = relative_path(row['relative_path'])
        if name in seen:
            raise ValueError('Duplicate Copt path: ' + str(name))
        seen.add(name)
        if type(row['bytes']) is not int or row['bytes'] <= 0:
            raise ValueError('Invalid Copt file size: ' + str(name))
        if not isinstance(row['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid Copt SHA256: ' + str(name))
    return rows


def check_copt_backup(repo, home):
    manifest = json.loads((repo / 'manifests/copt_engine_backup_manifest.json').read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Invalid Copt engine manifest')
    engines = four_records(manifest, 'files')
    aliases = four_records(manifest, 'checkpoint_aliases')
    lane = 'lane_c_orin_stage6_five_config_20260724/'
    count = 0
    for row in engines:
        relative = row['relative_path']
        if not relative.startswith(lane + 'codriving/') or not relative.endswith('/backbone.engine'):
            raise ValueError('Invalid Copt engine path: ' + relative)
        if row['published_relative_path'] != 'engine_payload/' + relative:
            raise ValueError('Copt published path does not match: ' + relative)
        check_file(repo, row['published_relative_path'], row['bytes'], row['sha256'])
        count += 1
        if home:
            check_file(home, relative, row['bytes'], row['sha256'])
            count += 1
    sources = set()
    for row in aliases:
        source = row['source_relative_path']
        relative_path(source)
        prefix = lane + 'models/codriving/'
        if (not source.startswith(prefix) or not source.endswith('.pth') or source in sources
                or row['relative_path'] != lane + 'source/codriving/' + source[len(prefix):]):
            raise ValueError('Invalid Copt checkpoint alias: ' + row['relative_path'])
        sources.add(source)
        if home:
            check_file(home, row['relative_path'], row['bytes'], row['sha256'])
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--offline', type=Path, help='Also check every offline archive part')
    parser.add_argument('--home', type=Path, help='Also check restored model/wheel files at the home root')
    args = parser.parse_args()
    count = check_copt_backup(args.repo, args.home)
    source = json.loads((args.repo / 'manifests/source_capture_manifest.json').read_text())
    for row in source['files']:
        check_file(args.repo, row['published_relative_path'], row['published_bytes'], row['published_sha256'])
        count += 1
        if args.home:
            check_file(args.home, row['relative_path'], row['published_bytes'], row['published_sha256'])
            count += 1
    results = json.loads((args.repo / 'manifests/key_results_manifest.json').read_text())
    for row in results['files']:
        check_file(args.repo / 'key_results', row['relative_path'], row['bytes'], row['sha256'])
        count += 1
    manifests = [args.repo / 'manifests' / name for name in (
        'offline_assets_manifest.json', 'fcooper_assets_manifest.json',
        'supplemental_assets_manifest.json', 'dataset_metadata_assets_manifest.json')]

    if args.offline or args.home:
        missing = [path.name for path in manifests if not path.is_file()]
        if missing:
            raise ValueError('Required asset manifests missing: ' + ', '.join(missing))
        for path in manifests:
            manifest = json.loads(path.read_text())
            if not manifest.get('verified'):
                raise ValueError('Unverified asset manifest: ' + str(path))
            if args.offline:
                for row in manifest['archive_parts']:
                    check_file(args.offline, row['name'], row['bytes'], row['sha256'])
                    count += 1
            if args.home:
                for row in manifest['files']:
                    check_file(args.home, row['relative_path'], row['bytes'], row['sha256'])
                    count += 1
        docker = json.loads((args.repo / 'manifests/docker_images_manifest.json').read_text())
        if not docker.get('verified'):
            raise ValueError('Docker image backup is unverified')
        if args.offline:
            check_file(args.offline, docker['name'], docker['bytes'], docker['sha256'])
            count += 1
    print('PASS: %d files verified; payload was not executed.' % count)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print('FAIL: ' + str(error), file=sys.stderr)
        sys.exit(1)
