"""Diagnostic only: full graphs, no performance acceptance or energy."""
import gc,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
import runner as r
m=json.loads((root/'manifest.json').read_text())
assert r.sha(root/'manifest.json')=='3edf343cfcbc0a8835f95bbf16e3a5fed7f171d25da3786c2db4c2daafa8984f'
assert r.sha(root/'runner.py')==m['runner_sha256']
assert r.sha(root/'reference_identity.json')==m['reference_identity_sha256']
assert r.sha(root/'wheel_inventory.json')==m['wheel_inventory_sha256']
r.verify_identity(m,json.loads((root/'reference_identity.json').read_text()))
out=root/'complete_graph_probe_r01';out.mkdir(exist_ok=False)
results={}
with r.serial_target(root):
 for p in m['points']:
  key=p['key'];assert r.sha(root/p['graph'])==p['graph_sha256']
  feed=r.load_pack(root/p['inputs'],p['input_spec']);refs=r.load_pack(root/p['reference'],p['reference_spec'])
  print('begin',key,flush=True);start=time.monotonic()
  session=r.session(root/p['graph'],out/('profile_'+key))
  assert set(feed)=={v.name for v in session.get_inputs()}
  assert set(refs)=={v.name for v in session.get_outputs()}
  row=r.request(session,feed);arrays=row.pop('outputs')
  assert len(arrays)==len(session.get_outputs())
  differences={}
  for spec,value in zip(session.get_outputs(),arrays):
   reference=refs[spec.name]
   if value.shape!=reference.shape or value.dtype!=reference.dtype or not r.np.isfinite(value).all():
    differences[spec.name]={'shape':list(value.shape),'dtype':str(value.dtype),'finite':bool(r.np.isfinite(value).all()),'allclose':False,'contract_mismatch':True};continue
   delta=r.np.abs(value-reference)
   differences[spec.name]={'shape':list(value.shape),'dtype':str(value.dtype),'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),'allclose':bool(r.np.allclose(value,reference,rtol=1e-4,atol=1e-5)),'outside_count':int(r.np.count_nonzero(delta>1e-5+1e-4*r.np.abs(reference)))}
  r.np.savez(out/(key+'_outputs.npz'),**{v.name:a for v,a in zip(session.get_outputs(),arrays)})
  profile=session.end_profiling();execution=r.verify_profile(json.loads(Path(profile).read_text()),key.startswith('integer'))
  results[key]={'differences':differences,'execution':execution,'diagnostic_first_request_ms':row['service_ms'],'elapsed_including_session_s':time.monotonic()-start,'versions':session._measurement_versions,'graph_sha256':p['graph_sha256']}
  r.write(out/'results.json',results);print(key,results[key],flush=True)
  del session,feed,refs,arrays;gc.collect()
