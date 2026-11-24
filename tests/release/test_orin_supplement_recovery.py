"""The private UniAD/chassis supplement is required for asset recovery checks."""

import json

import pytest

from test_orin_engine_recovery import (CHECKER, assert_failure, recovery, verify,
                                       write_json, write_payload)


SUPPLEMENT = 'restore_audit_assets_manifest.json'
ARCHIVE = 'restore-audit-supplement.tar.gz.part000'
SDK = ('ROS1/wheeltec_robot/src/ros_astra_camera-main/include/'
       'openni2_redist/arm64/libOpenNI2.so')
INPUT = 'liruozhan/toOrinV2/uniad_trt_input_119plus/'


@pytest.fixture
def supplement(recovery):
    files = []
    for relative, kind in (
        (SDK, 'astra_arm64_runtime_sdk'),
        (INPUT + 'info.txt', 'uniad_default_input_info'),
        (INPUT + 'timestamp/120.bin', 'uniad_default_input_metadata'),
    ):
        metadata = write_payload(recovery['home'], relative, relative.encode())
        files.append(dict(metadata, relative_path=relative, kind=kind))
    part = write_payload(recovery['offline'], ARCHIVE, b'supplement archive fixture')
    write_json(recovery['repo'] / 'manifests' / SUPPLEMENT,
               {'verified': True, 'files': files, 'archive_parts': [dict(part, name=ARCHIVE)]})
    return recovery


@pytest.mark.parametrize('mode', ['offline', 'home'])
def test_supplement_manifest_is_required_for_asset_checks(supplement, mode):
    (supplement['repo'] / 'manifests' / SUPPLEMENT).unlink()
    assert_failure(verify(supplement, mode))


@pytest.mark.parametrize('mode', ['offline', 'home'])
def test_supplement_manifest_must_be_verified(supplement, mode):
    path = supplement['repo'] / 'manifests' / SUPPLEMENT
    manifest = json.loads(path.read_text())
    write_json(path, dict(manifest, verified=False))
    assert_failure(verify(supplement, mode))


@pytest.mark.parametrize('damage', ['missing', 'same_size_corruption'])
def test_supplement_archive_integrity_is_required(supplement, damage):
    path = supplement['offline'] / ARCHIVE
    if damage == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'X' * path.stat().st_size)
    assert_failure(verify(supplement, 'offline'))


@pytest.mark.parametrize('relative', [SDK, INPUT + 'info.txt', INPUT + 'timestamp/120.bin'])
def test_restored_sdk_and_default_121_frame_inputs_are_required(supplement, relative):
    (supplement['home'] / relative).unlink()
    assert_failure(verify(supplement, 'home'))


@pytest.mark.parametrize('mode', ['default', 'offline', 'home'])
def test_complete_supplement_passes_read_only(supplement, mode):
    before = {path: path.read_bytes() for root in ('repo', 'home', 'offline')
              for path in supplement[root].rglob('*') if path.is_file()}
    result = verify(supplement, mode)
    assert result.returncode == 0, result.stderr
    after = {path: path.read_bytes() for root in ('repo', 'home', 'offline')
             for path in supplement[root].rglob('*') if path.is_file()}
    assert before == after


def test_default_check_does_not_require_private_supplement(supplement):
    (supplement['repo'] / 'manifests' / SUPPLEMENT).unlink()
    (supplement['offline'] / ARCHIVE).unlink()
    (supplement['home'] / SDK).unlink()
    result = verify(supplement)
    assert result.returncode == 0, result.stderr


def test_published_supplement_manifest_is_complete_and_minimal():
    manifest = json.loads((CHECKER.parents[1] / 'manifests' / SUPPLEMENT).read_text())
    assert manifest['verified'] is True
    assert len(manifest['files']) == 1581
    assert len({row['relative_path'] for row in manifest['files']}) == 1581
    for row in manifest['files']:
        assert set(row) == {'relative_path', 'bytes', 'sha256', 'kind'}
        assert not row['relative_path'].startswith('/')
    assert len(manifest['archive_parts']) == 1
    assert manifest['archive_parts'][0]['name'] == ARCHIVE
