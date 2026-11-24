"""Read-only recovery must verify the frozen Copt engines and checkpoint aliases."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


CHECKER = (Path(__file__).resolve().parents[2] /
           'docs/orin-recovery-20261123/tools/verify_recovery.py')
LANE = 'lane_c_orin_stage6_five_config_20260724'
WIDTHS = ('016x032x064', '032x064x096', '032x080x128', '064x128x256')
ASSET_MANIFESTS = ('offline_assets_manifest.json', 'fcooper_assets_manifest.json',
                   'supplemental_assets_manifest.json', 'dataset_metadata_assets_manifest.json',
                   'restore_audit_assets_manifest.json')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_payload(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def recovery(tmp_path):
    repo, home, offline = (tmp_path / name for name in ('repo', 'home', 'offline'))
    for root in (repo, home, offline):
        root.mkdir()
    write_json(repo / 'manifests/source_capture_manifest.json', {'files': []})
    write_json(repo / 'manifests/key_results_manifest.json', {'files': []})
    for name in ASSET_MANIFESTS:
        write_json(repo / 'manifests' / name,
                   {'verified': True, 'files': [], 'archive_parts': []})
    docker = write_payload(offline, 'docker.tar', b'docker archive fixture')
    write_json(repo / 'manifests/docker_images_manifest.json',
               dict(docker, verified=True, name='docker.tar'))
    engines, aliases, models = [], [], []
    for index, width in enumerate(WIDTHS):
        relative = LANE + '/codriving/' + width + '/backbone.engine'
        published = 'engine_payload/' + relative
        payload = ('engine fixture %d' % index).encode()
        metadata = write_payload(repo, published, payload)
        write_payload(home, relative, payload)
        engines.append(dict(metadata, relative_path=relative,
                            published_relative_path=published,
                            download_url='https://example.invalid/' + published))
        model = LANE + '/models/codriving/' + width + '/checkpoint.pth'
        alias = LANE + '/source/codriving/' + width + '/checkpoint.pth'
        payload = ('checkpoint fixture %d' % index).encode()
        metadata = write_payload(home, model, payload)
        write_payload(home, alias, payload)
        aliases.append(dict(metadata, source_relative_path=model, relative_path=alias))
        models.append(dict(metadata, relative_path=model))
    write_json(repo / 'manifests/offline_assets_manifest.json',
               {'verified': True, 'archive_parts': [], 'files': models})
    manifest = {'files': engines, 'checkpoint_aliases': aliases}
    manifest_path = repo / 'manifests/copt_engine_backup_manifest.json'
    write_json(manifest_path, manifest)
    return {'repo': repo, 'home': home, 'offline': offline,
            'manifest': manifest, 'manifest_path': manifest_path}


def verify(recovery, mode='default'):
    command = [sys.executable, str(CHECKER), '--repo', str(recovery['repo'])]
    if mode != 'default':
        command.extend(['--' + mode, str(recovery[mode])])
    return subprocess.run(command, capture_output=True, text=True, check=False)


def assert_failure(result):
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'FAIL:' in result.stderr
    assert 'PASS:' not in result.stdout


@pytest.mark.parametrize('mode', ['default', 'offline', 'home'])
def test_complete_layout_passes_without_modifying_files(recovery, mode):
    before = {path: path.read_bytes() for root in ('repo', 'home', 'offline')
              for path in recovery[root].rglob('*') if path.is_file()}
    result = verify(recovery, mode)
    assert result.returncode == 0, result.stderr
    assert 'PASS:' in result.stdout
    assert 'payload was not executed' in result.stdout
    after = {path: path.read_bytes() for root in ('repo', 'home', 'offline')
             for path in recovery[root].rglob('*') if path.is_file()}
    assert after == before


@pytest.mark.parametrize('mode', ['default', 'offline', 'home'])
def test_engine_manifest_is_mandatory(recovery, mode):
    recovery['manifest_path'].unlink()
    assert_failure(verify(recovery, mode))


@pytest.mark.parametrize('mode', ['default', 'offline', 'home'])
@pytest.mark.parametrize('damage', ['missing', 'same_size_corruption'])
def test_repo_engine_integrity_is_required_in_every_mode(recovery, mode, damage):
    path = recovery['repo'] / recovery['manifest']['files'][0]['published_relative_path']
    if damage == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'X' * path.stat().st_size)
    assert_failure(verify(recovery, mode))


@pytest.mark.parametrize('category', ['files', 'checkpoint_aliases'])
@pytest.mark.parametrize('damage', ['missing', 'empty', 'short', 'extra', 'duplicate'])
def test_manifest_requires_four_distinct_engines_and_aliases(recovery, category, damage):
    manifest = recovery['manifest']
    rows = manifest[category]
    if damage == 'missing':
        manifest = {key: value for key, value in manifest.items() if key != category}
    else:
        replacements = {'empty': [], 'short': rows[:3], 'extra': rows + [rows[0]],
                        'duplicate': [rows[0], rows[0], rows[2], rows[3]]}
        manifest = dict(manifest, **{category: replacements[damage]})
    write_json(recovery['manifest_path'], manifest)
    assert_failure(verify(recovery))


@pytest.mark.parametrize('category', ['files', 'checkpoint_aliases'])
@pytest.mark.parametrize('damage', ['missing', 'same_size_corruption'])
def test_home_requires_engines_and_aliases_even_when_models_exist(recovery, category, damage):
    row = recovery['manifest'][category][0]
    path = recovery['home'] / row['relative_path']
    if damage == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'X' * path.stat().st_size)
    for alias in recovery['manifest']['checkpoint_aliases']:
        assert (recovery['home'] / alias['source_relative_path']).is_file()
    assert_failure(verify(recovery, 'home'))


@pytest.mark.parametrize('relative', ['../outside.engine', '/tmp/outside.engine', ''])
def test_engine_paths_must_be_safe_relative_paths(recovery, relative):
    manifest = recovery['manifest']
    rows = [dict(manifest['files'][0], relative_path=relative,
                 published_relative_path='engine_payload/' + relative)] + manifest['files'][1:]
    write_json(recovery['manifest_path'], dict(manifest, files=rows))
    assert_failure(verify(recovery))


def test_published_engine_path_must_match_restored_path(recovery):
    manifest = recovery['manifest']
    rows = [dict(manifest['files'][0], published_relative_path=
                 manifest['files'][1]['published_relative_path'])] + manifest['files'][1:]
    write_json(recovery['manifest_path'], dict(manifest, files=rows))
    assert_failure(verify(recovery))


def test_checkpoint_alias_cannot_be_replaced_by_models_path(recovery):
    manifest = recovery['manifest']
    row = manifest['checkpoint_aliases'][0]
    aliases = [dict(row, relative_path=row['source_relative_path'])] + manifest['checkpoint_aliases'][1:]
    write_json(recovery['manifest_path'], dict(manifest, checkpoint_aliases=aliases))
    assert_failure(verify(recovery, 'home'))


@pytest.mark.parametrize('mode', ['default', 'home'])
@pytest.mark.parametrize('kind', ['leaf', 'parent_escape'])
def test_symlink_payloads_cannot_satisfy_recovery(recovery, tmp_path, mode, kind):
    row = recovery['manifest']['files'][0]
    root = recovery['repo'] if mode == 'default' else recovery['home']
    relative = row['published_relative_path'] if mode == 'default' else row['relative_path']
    path = root / relative
    outside = tmp_path / 'outside'
    outside.mkdir()
    if kind == 'leaf':
        target = outside / 'engine'
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    else:
        shutil.copytree(path.parent, outside / 'engines')
        shutil.rmtree(path.parent)
        path.parent.symlink_to(outside / 'engines', target_is_directory=True)
    assert_failure(verify(recovery, mode))


@pytest.mark.parametrize('category', ['source', 'results', 'offline_part', 'model', 'docker'])
def test_existing_payload_checks_still_reject_corruption(recovery, category):
    payload = b'original captured artifact'
    if category == 'source':
        relative = 'runtime/worker.py'
        metadata = write_payload(recovery['repo'], 'source/' + relative, payload)
        row = {'relative_path': relative, 'published_relative_path': 'source/' + relative,
               'published_bytes': metadata['bytes'], 'published_sha256': metadata['sha256']}
        write_json(recovery['repo'] / 'manifests/source_capture_manifest.json', {'files': [row]})
        path, mode = recovery['repo'] / ('source/' + relative), 'default'
    elif category == 'results':
        metadata = write_payload(recovery['repo'], 'key_results/run/result.json', payload)
        write_json(recovery['repo'] / 'manifests/key_results_manifest.json',
                   {'files': [dict(metadata, relative_path='run/result.json')]})
        path, mode = recovery['repo'] / 'key_results/run/result.json', 'default'
    elif category == 'offline_part':
        metadata = write_payload(recovery['offline'], 'models.tar.part00', payload)
        write_json(recovery['repo'] / 'manifests/fcooper_assets_manifest.json',
                   {'verified': True, 'files': [], 'archive_parts': [
                       dict(metadata, name='models.tar.part00')]})
        path, mode = recovery['offline'] / 'models.tar.part00', 'offline'
    elif category == 'model':
        path = recovery['home'] / recovery['manifest']['checkpoint_aliases'][0]['source_relative_path']
        mode = 'home'
    else:
        path, mode = recovery['offline'] / 'docker.tar', 'offline'
    path.write_bytes(b'X' * path.stat().st_size)
    assert_failure(verify(recovery, mode))


@pytest.mark.parametrize('name', ASSET_MANIFESTS + ('docker_images_manifest.json',))
@pytest.mark.parametrize('damage', ['missing', 'unverified'])
def test_existing_asset_manifests_remain_mandatory_and_verified(recovery, name, damage):
    path = recovery['repo'] / 'manifests' / name
    if damage == 'missing':
        path.unlink()
    else:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        write_json(path, dict(manifest, verified=False))
    assert_failure(verify(recovery, 'offline'))


@pytest.mark.parametrize('category', ['files', 'checkpoint_aliases'])
@pytest.mark.parametrize('field,value', [('bytes', 0), ('bytes', True), ('sha256', 'bad')])
def test_manifest_metadata_is_required_even_for_aliases_without_home(recovery, category, field, value):
    manifest = recovery['manifest']
    rows = [dict(manifest[category][0], **{field: value})] + manifest[category][1:]
    write_json(recovery['manifest_path'], dict(manifest, **{category: rows}))
    assert_failure(verify(recovery))
