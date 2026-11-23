#!/usr/bin/env python3
"""Evidence-only wrapper: execute captured original CLI without changing its argv."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / 'p4-five-arms-20261003T054210Z'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def snapshot():
    paths = sorted(str(p.relative_to(COHORT)) for p in (COHORT / 'runs').glob('*/attempt-*') if p.is_dir())
    protected = {}
    for p in sorted((COHORT / 'runs').rglob('*')):
        if p.is_file() and ('evaluator' in p.name or 'episode' in p.name or 'service' in p.name):
            protected[str(p.relative_to(COHORT))] = {'sha256': sha(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
    state = json.loads((COHORT / 'state.json').read_text())
    return {'state_sha256': sha(COHORT / 'state.json'), 'attempt_dirs': paths,
            'attempt_counts': {k: len(v) for k, v in state['attempts'].items()},
            'evaluator_hook_evidence': protected}

inv = json.loads((ROOT / 'evidence/original-runner-invocation.json').read_text())
before = snapshot()
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
env = dict(os.environ, **inv['environment'])
t0 = time.monotonic()
with (ROOT / 'evidence/original-resume-cli.log').open('w') as log:
    process = subprocess.run(inv['argv'], cwd=inv['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT)
after = snapshot()
receipt = {'schema': 'coptv2x.p4-original-resume-receipt.v1', 'started_utc': started,
           'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'duration_seconds': time.monotonic() - t0, 'argv': inv['argv'], 'cwd': inv['cwd'],
           'captured_environment': inv['environment'], 'exit_code': process.returncode,
           'before': before, 'after': after, 'no_new_attempt_dirs': before['attempt_dirs'] == after['attempt_dirs'],
           'state_sha256_unchanged': before['state_sha256'] == after['state_sha256'],
           'no_evaluator_hooks_or_evaluator_log_changes': before['evaluator_hook_evidence'] == after['evaluator_hook_evidence']}
receipt['status'] = 'PASS' if process.returncode == 0 and before == after else 'FAIL'
(ROOT / 'evidence/original-resume-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: receipt[k] for k in ['status', 'exit_code', 'duration_seconds', 'no_new_attempt_dirs', 'state_sha256_unchanged', 'no_evaluator_hooks_or_evaluator_log_changes']}), flush=True)
raise SystemExit(0 if receipt['status'] == 'PASS' else 1)
