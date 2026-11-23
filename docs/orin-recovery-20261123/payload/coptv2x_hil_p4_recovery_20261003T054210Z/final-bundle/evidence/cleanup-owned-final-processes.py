#!/usr/bin/env python3
"""Stop only the verified original owned CARLA group and reverse tunnel."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent

def read(pid):
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return None
    stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
    return {'pid': pid, 'uid': proc.stat().st_uid, 'state': stat[0], 'pgid': int(stat[2]),
            'start_ticks': int(stat[19]), 'argv': (proc / 'cmdline').read_bytes().decode().split('\0')[:-1]}

before = {str(pid): read(pid) for pid in [2851974, 2851981, 2784297]}
assert all(before.values()), before
assert all(p['uid'] == os.getuid() for p in before.values())
shell, carla, tunnel = [before[str(pid)] for pid in [2851974, 2851981, 2784297]]
assert shell['pgid'] == carla['pgid'] == 2851974
assert shell['argv'] == ['/bin/sh', 'external_paths/carla_root/CarlaUE4.sh', '--world-port=56800', '-prefer-nvidia', '-opengl', '-RenderOffScreen']
assert carla['argv'][0] == '/home/jichengzhi/V2Xverse/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
assert '--world-port=56800' in carla['argv']
assert tunnel['pgid'] == 2784297 and tunnel['argv'][0] == 'ssh'
assert '127.0.0.1:56811:127.0.0.1:56810' in tunnel['argv'] and tunnel['argv'][-1] == 'jichengzhi@172.16.62.222'
assert json.loads((ROOT / 'evidence/original-resume-receipt.json').read_text())['status'] == 'PASS'
assert read(2860810) is None, 'Original runner unexpectedly live'
for pid in [2851974, 2851981, 2784297]:
    assert read(pid) == before[str(pid)], 'Identity changed before signal'
os.killpg(2851974, signal.SIGTERM)
os.kill(2784297, signal.SIGTERM)
for unused in range(20):
    if all(read(pid) is None or read(pid)['state'] == 'Z' for pid in [2851974, 2851981, 2784297]):
        break
    time.sleep(0.5)
after = {str(pid): read(pid) for pid in [2851974, 2851981, 2784297]}
valid = all(p is None or p['state'] == 'Z' for p in after.values())
receipt = {'schema': 'coptv2x.p4-owned-process-cleanup.v1', 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'before': before, 'after': after, 'signals': [{'pgid': 2851974, 'signal': 'SIGTERM'}, {'pid': 2784297, 'signal': 'SIGTERM'}],
           'no_shared_ros_container_stopped': True, 'no_other_jobs_signalled': True, 'status': 'PASS' if valid else 'FAIL'}
(ROOT / 'evidence/owned-process-cleanup.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if valid else 1)
