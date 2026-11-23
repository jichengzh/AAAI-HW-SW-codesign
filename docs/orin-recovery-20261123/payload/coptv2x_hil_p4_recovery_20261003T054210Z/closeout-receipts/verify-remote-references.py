#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent
mapping = json.loads((ROOT / 'evidence/frozen-artifact-remote-verification.json').read_text())
inputs = json.loads((ROOT / 'evidence/orin-local-input-manifest.json').read_text())
payload = {'records': mapping['records'], 'symlinks': mapping['symlinks'],
           'input_paths': [row['inputs']['spatial_features']['path'] for row in inputs['samples']]}
code = '''import hashlib,json,os,sys
from pathlib import Path
data=json.loads(sys.stdin.read())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
base=Path('/home/jichengzhi/coptv2x_hil_p4_recovery_20261003T054210Z/final-bundle')
unique={r['orin_path']:r['sha256'] for r in data['records']}
records=[{'path':p,'sha256':sha(p),'expected_sha256':s} for p,s in sorted(unique.items())]
links=[{'path':r['bundle_relative_path'],'target':os.readlink(base/r['bundle_relative_path']),
        'expected_target':r['verified_orin_artifact'],'sha256':sha(base/r['bundle_relative_path']),
        'expected_sha256':r['sha256']} for r in data['symlinks']]
cache=Path('/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z/input')
present=sum((cache/p).is_file() for p in data['input_paths'])
valid=all(r['sha256']==r['expected_sha256'] for r in records) and all(r['target']==r['expected_target'] and r['sha256']==r['expected_sha256'] for r in links) and present==2225
print(json.dumps({'status':'PASS' if valid else 'FAIL','unique_frozen_files':len(records),
                 'checkpoint_symlinks':len(links),'input_cache_existing_files':present,
                 'input_cache_no_rehash':True,'records':records,'links':links}))
sys.exit(0 if valid else 1)
'''
env = dict(os.environ, SSH_AUTH_SOCK='/tmp/ssh-XXXXXXJ80vcM/agent.4066676')
result = subprocess.run(['ssh', '-oBatchMode=yes', '-oConnectTimeout=8', 'jichengzhi@172.16.62.222',
                         'python3 -c ' + shlex.quote(code)], input=json.dumps(payload),
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
(ROOT / 'closeout-receipts/remote-reference-verification.stderr.log').write_text(result.stderr)
(ROOT / 'closeout-receipts/remote-reference-verification.json').write_text(result.stdout)
print('exit_code=', result.returncode, flush=True)
if result.returncode == 0:
    data = json.loads(result.stdout)
    print(json.dumps({k: data[k] for k in ['status', 'unique_frozen_files', 'checkpoint_symlinks', 'input_cache_existing_files']}))
raise SystemExit(result.returncode)
