#!/usr/bin/env python3
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shlex

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / 'final-bundle'
REMOTE = '/home/jichengzhi/coptv2x_hil_p4_recovery_20261003T054210Z'
ROUND = sys.argv[1]
env = dict(os.environ, SSH_AUTH_SOCK='/tmp/ssh-XXXXXXJ80vcM/agent.4066676')
ssh = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=8', 'jichengzhi@172.16.62.222']
t0 = time.monotonic()
with (ROOT / ('closeout-receipts/rsync-' + ROUND + '.log')).open('w') as log:
    sync = subprocess.run(['rsync', '-a', '--stats', '-e', 'ssh -oBatchMode=yes -oConnectTimeout=8',
                           str(BUNDLE) + '/', 'jichengzhi@172.16.62.222:' + REMOTE + '/final-bundle/'],
                          env=env, stdout=log, stderr=subprocess.STDOUT)
if sync.returncode:
    raise SystemExit(sync.returncode)
checks = subprocess.run(ssh + ['cd ' + REMOTE + '/final-bundle && sha256sum -c BUNDLE.SHA256SUMS'],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
path = ROOT / ('closeout-receipts/remote-sha256-' + ROUND + '.log')
path.write_text(checks.stdout)
rows = (BUNDLE / 'BUNDLE.SHA256SUMS').read_text().splitlines()
oks = [line for line in checks.stdout.splitlines() if line.endswith(': OK')]
digest = hashlib.sha256((BUNDLE / 'BUNDLE.SHA256SUMS').read_bytes()).hexdigest()
count_code = "from pathlib import Path; p=Path(%r); print(sum(x.is_file() and not x.is_symlink() for x in p.rglob('*')))" % (REMOTE + '/final-bundle')
counts = subprocess.run(ssh + ['python3 -c ' + shlex.quote(count_code)], env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
remote_count = int(counts.stdout.strip()) if counts.returncode == 0 else None
receipt = {'schema': 'coptv2x.p4-final-bundle-remote-receipt.v1', 'round': ROUND,
           'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'duration_seconds': time.monotonic() - t0, 'rsync_exit_code': sync.returncode,
           'remote_sha256sum_exit_code': checks.returncode, 'expected_regular_file_rows': len(rows),
           'remote_ok_rows': len(oks), 'sha256sums_sha256': digest,
           'remote_regular_files_including_sums': remote_count,
           'remote_all_regular_files_covered': remote_count == len(rows) + 1,
           'bundle_remote_path': REMOTE + '/final-bundle',
           'remote_stdout_sha256': hashlib.sha256(checks.stdout.encode()).hexdigest(),
           'no_delete_option_used': True,
           'status': 'PASS' if checks.returncode == 0 and len(rows) == len(oks) and remote_count == len(rows) + 1 else 'FAIL'}
(ROOT / ('closeout-receipts/remote-verification-' + ROUND + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if receipt['status'] == 'PASS' else 1)
