#!/usr/bin/env python3
"""Generated archive preparation; do not touch live source attempts or artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent
WORKTREE = Path('/home/jichengzhi/V2X/.worktrees/V2Xverse-ditl')
BUNDLE = ROOT / 'final-bundle'
REMOTE = '/home/jichengzhi/coptv2x_hil_p4_recovery_20261003T054210Z'
DOCROOT = Path('/home/jichengzhi/V2X/multi_agent/v2x_rts')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

receipt = json.loads((ROOT / 'evidence/original-resume-receipt.json').read_text())
assert receipt['status'] == 'PASS', receipt['status']
snapshot = ROOT / 'evidence/code-snapshot.tgz'
assert sha(snapshot) == '8503a08c8fe664190ee72161914a262a98b35ceb20ada91a4f344378d026c747'
checked, differences = [], []
with tarfile.open(snapshot, 'r:gz') as archive:
    for member in archive.getmembers():
        if not member.isfile():
            continue
        actual = WORKTREE / member.name
        archived = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        if not actual.is_file() or sha(actual) != archived:
            differences.append(member.name)
        checked.append(member.name)
assert not differences, differences
diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=WORKTREE)
assert hashlib.sha256(diff).hexdigest() == sha(ROOT / 'evidence/worktree.patch')
write(ROOT / 'evidence/code-snapshot-final-verification.json', {
    'status': 'PASS', 'snapshot_sha256': sha(snapshot), 'regular_members_checked': len(checked),
    'changed_members': differences, 'current_git_diff_sha256': hashlib.sha256(diff).hexdigest(),
    'existing_worktree_patch_unchanged': True})
BUNDLE.mkdir(exist_ok=True)
for source in sorted(ROOT.iterdir()):
    if source.name in ('final-bundle', 'closeout-receipts'):
        continue
    target = BUNDLE / source.name
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True,
                        ignore=lambda parent, names: [n for n in names if Path(parent, n).is_symlink()])
    else:
        shutil.copy2(source, target)
mapping = json.loads((ROOT / 'evidence/frozen-artifact-remote-verification.json').read_text())
for row in mapping['symlinks']:
    link = BUNDLE / row['bundle_relative_path']
    if not link.is_symlink():
        assert not link.exists(), str(link)
        link.symlink_to(row['verified_orin_artifact'])
    assert str(link.readlink()) == row['verified_orin_artifact'], str(link)
references = {'schema': 'coptv2x.p4-final-bundle-references.v1',
              'bundle_remote_path': REMOTE + '/final-bundle',
              'external_frozen_artifacts': mapping['records'],
              'checkpoint_symlinks': mapping['symlinks'],
              'input_cache_remote_root': '/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z/input',
              'input_manifest_bundle_path': 'evidence/orin-local-input-manifest.json',
              'input_cache_verified_file_count': 2225,
              'no_input_cache_recopied': True, 'source_attempts_unchanged': True,
              'server_dependencies_bundle_manifest': 'evidence/server-dependency-assets.json'}
write(BUNDLE / 'evidence/final-bundle-reference-map.json', references)
docs = BUNDLE / 'docs'
docs.mkdir(exist_ok=True)
for source in [DOCROOT / 'progress/CoptV2X在线HIL_P4实验进度_20261003.md',
               DOCROOT / 'result/CoptV2X在线HIL_P4最终报告_20261003.md',
               DOCROOT / 'result/CoptV2X在线HIL_P4独立复审_20261003.md',
               Path('/home/jichengzhi/V2X/.superpowers/sdd/2026-10-01-coptv2x-p4-smoke/progress.md')]:
    shutil.copy2(source, docs / (('SDD-' if source.name == 'progress.md' else '') + source.name))
files = sorted(p for p in BUNDLE.rglob('*') if p.is_file() and not p.is_symlink()
               and p.name != 'BUNDLE.SHA256SUMS')
sums = BUNDLE / 'BUNDLE.SHA256SUMS'
sums.write_text(''.join(sha(p) + '  ' + p.relative_to(BUNDLE).as_posix() + '\n' for p in files))
print(json.dumps({'bundle': str(BUNDLE), 'regular_files': len(files),
                  'bytes': sum(p.stat().st_size for p in files), 'sha256sums_sha256': sha(sums),
                  'symlinks_remapped': len(mapping['symlinks']), 'code_members_checked': len(checked)}))
