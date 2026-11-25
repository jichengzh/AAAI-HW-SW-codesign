"""The published checksum table must describe the files a Git checkout supplies."""

import hashlib
from pathlib import Path
import re
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPOSITORY_ROOT / 'docs/orin-recovery-20261123'


@pytest.fixture(scope='module')
def publication_records():
    if not (REPOSITORY_ROOT / '.git').exists():
        pytest.skip('Git publication audit requires checkout metadata; source ZIPs omit it')
    metadata = subprocess.run(['git', '-C', str(REPOSITORY_ROOT), 'rev-parse', '--show-toplevel'],
                              capture_output=True, text=True, check=False)
    assert metadata.returncode == 0, metadata.stderr
    assert Path(metadata.stdout.strip()).resolve() == REPOSITORY_ROOT, metadata.stdout
    records = {}
    for line in (PACKAGE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, separator, relative = line.partition('  ')
        assert separator and re.fullmatch('[0-9a-f]{64}', digest), line
        path = Path(relative)
        assert relative and not path.is_absolute() and '..' not in path.parts, relative
        assert path.as_posix() == relative, relative
        assert relative not in records, 'Duplicate checksum entry: ' + relative
        records[relative] = digest
    return records


def test_checksum_table_matches_exactly_the_tracked_package(publication_records):
    tracked = subprocess.run(
        ['git', '-C', str(REPOSITORY_ROOT), 'ls-files', '-z', '--',
         PACKAGE.relative_to(REPOSITORY_ROOT).as_posix() + '/'],
        capture_output=True, check=True,
    )
    expected = {
        (REPOSITORY_ROOT / name.decode('utf-8')).relative_to(PACKAGE).as_posix()
        for name in tracked.stdout.split(b'\0') if name
    } - {'SHA256SUMS'}
    actual = set(publication_records)
    assert actual == expected, {
        'untracked_entries': sorted(actual - expected),
        'missing_tracked_entries': sorted(expected - actual),
    }


def test_checksum_table_excludes_python_cache(publication_records):
    forbidden = [name for name in publication_records
                 if '__pycache__' in Path(name).parts or Path(name).suffix in ('.pyc', '.pyo')]
    assert not forbidden, forbidden


def test_every_published_checksum_matches_file_bytes(publication_records):
    for relative, expected in publication_records.items():
        path = PACKAGE / relative
        assert path.is_file() and not path.is_symlink(), relative
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected, relative
