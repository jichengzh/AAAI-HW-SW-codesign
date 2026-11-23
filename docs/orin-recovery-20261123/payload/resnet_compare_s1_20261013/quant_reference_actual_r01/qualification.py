"""Fail-closed qualification for target-replay measurements after complete target AP."""
import hashlib
import json
import math
from pathlib import Path


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def bound_file(root,name,expected):
    root=Path(root).resolve();path=(root/name).resolve()
    require(root in path.parents,'Evidence path escapes bundle')
    require(path.is_file() and sha(path)==expected,'Evidence file SHA mismatch')
    return path


def valid_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_target_ap(root,binding,graphs,options):
    require(bool(binding),'Complete full target AP evidence is required')
    path=bound_file(root,binding['file'],binding['sha256'])
    evidence=json.loads(path.read_text());base=path.parent
    require(evidence['status']=='complete' and evidence['count']==1789,'Incomplete full target AP')
    require(evidence['target_options']==options,'AP runtime differs from measurement')
    require(evidence['graph_sha256']==graphs,'AP graphs differ from measurement')
    validate_audit(base,binding,evidence,options)
    source=evidence['source_identity']
    require(bool(source.get('source_commit')) and valid_hash(source.get('checkpoint_sha256')) and
            valid_hash(source.get('calibration_sha256')),'Missing source/checkpoint/calibration identity')
    require(valid_hash(evidence.get('runtime_binary_sha256')) and valid_hash(evidence.get('eval_code_sha256')),'Missing runtime/eval binary identity')
    split=json.loads(bound_file(base,evidence['split_file'],evidence['split_sha256']).read_text())
    require(len(split)==1789 and len({str(x) for x in split})==1789,'Split incomplete or duplicated')
    require(set(evidence['methods'])=={'control','integer'},'Matched detection control required')
    paired={}
    for method,spec in evidence['methods'].items():
        require(set(spec['ap'])=={'ap30','ap50','ap70'} and all(math.isfinite(v) and 0<=v<=1 for v in spec['ap'].values()),'Invalid complete target AP values')
        trace=bound_file(base,spec['frames_file'],spec['frames_sha256'])
        rows=[json.loads(line) for line in trace.read_text().splitlines()]
        require(len(rows)==1789,'Incomplete full target AP trace')
        previous=None
        for i,row in enumerate(rows):
            require(row['index']==i and row['sample_id']==split[i] and row['completed'] is True,'AP frame omitted/repeated/reordered')
            require(row['agents'] in (1,2) and row['graph_sha256']==graphs[method+'_a'+str(row['agents'])],'AP frame graph differs')
            for name in ('input_sha256','initial_state_sha256','final_state_sha256'):
                require(valid_hash(row[name]),'Missing AP input/state identity')
            require(previous is None or row['initial_state_sha256']==previous,'AP EMA continuity broken')
            previous=row['final_state_sha256']
        paired[method]=[(r['sample_id'],r['agents'],r['input_sha256']) for r in rows]
    require(paired['control']==paired['integer'],'Control/integer AP data or random inputs differ')
    return dict(status='qualified_target_replay',count=1789,ap={k:v['ap'] for k,v in evidence['methods'].items()},
                runtime_binary_sha256=evidence['runtime_binary_sha256'],eval_code_sha256=evidence['eval_code_sha256'],source_identity=source,split_sha256=evidence['split_sha256'],evidence_sha256=binding['sha256'],host_equivalence=False)


def validate_target_references(root,manifest,ap,options):
    binding=manifest['target_reference_identity']
    path=bound_file(root,binding['file'],binding['sha256']);base=path.parent
    identity=json.loads(path.read_text())
    require(identity['target_options']==options,'Target reference runtime differs')
    require(identity['runtime_binary_sha256']==ap['runtime_binary_sha256'],'Reference/AP runtime binary differs')
    probe_path=bound_file(base,identity['probe_results']['file'],identity['probe_results']['sha256'])
    probe=json.loads(probe_path.read_text())
    require(set(identity['points'])==set(probe)=={p['key'] for p in manifest['points']},'Target reference point set differs')
    for p in manifest['points']:
        entry=identity['points'][p['key']]
        require(all(entry[k]==v for k,v in dict(graph_sha256=p['graph_sha256'],input_sha256=p['input_spec']['sha256'],reference_sha256=p['target_reference_spec']['sha256']).items()),'Target reference graph/input/output mismatch')
        require(probe[p['key']]['graph_sha256']==p['graph_sha256'],'Probe graph mismatch')
        versions=probe[p['key']]['versions']
        require(versions['onnxruntime']==options['version'] and versions['numpy']==options['numpy_version'] and versions['python'].startswith(options['python_major_minor']+'.'),'Probe runtime differs')
        require(len(probe[p['key']]['differences'])==6,'Probe must preserve all six host differences')
    return dict(kind='independent_target_probe_fixed_input_replay',host_equivalence=False,identity_sha256=binding['sha256'])


def validate_audit(root,binding,evidence,options):
    import ap_audit
    base=Path(root)
    receipt=json.loads(bound_file(base,binding['audit_file'],binding['audit_sha256']).read_text())
    require(receipt['status']=='passed' and receipt['ap_evidence_sha256']==binding['sha256'],'Missing/patched independent AP audit')
    require(receipt['frozen_manifest_sha256']==binding['expected_manifest_sha256']==evidence['manifest_sha256'],'Unanchored AP source manifest')
    manifest=json.loads(bound_file(base,evidence['manifest_file'],evidence['manifest_sha256']).read_text())
    require(evidence['source_identity']==manifest['source_identity'] and evidence['source_files']==manifest['source_files']==receipt['source_files'],'Source differs from independently audited frozen files')
    require(evidence['target_options']==manifest['target_options']==options and evidence['runtime_binary_sha256']==manifest['runtime_binary_sha256'],'AP runtime differs from frozen manifest')
    require(evidence['graph_sha256']=={k:v['sha256'] for k,v in manifest['graphs'].items()} and evidence['split_sha256']==manifest['split_sha256'],'Graphs/split differ from frozen AP manifest')
    eval_files=json.loads(bound_file(base,evidence['eval_code_file'],evidence['eval_code_sha256']).read_text())
    require(evidence['eval_code_sha256']==manifest['eval_code_sha256']==receipt['eval_code_sha256'] and eval_files==receipt['eval_files'],'Evaluation source audit differs')
    sort_identity=json.loads(bound_file(base,receipt['sort_identity_file'],receipt['sort_identity_sha256']).read_text())
    indices_path=bound_file(base,receipt['sort_indices_file'],receipt['sort_indices_sha256'])
    require(sort_identity['identity']['exporter_sha256']==sha(Path(__file__).with_name('sort_export.py')),'Frozen sort exporter code differs')
    require(sort_identity['indices_sha256']==receipt['sort_indices_sha256'] and sort_identity['identity']['ap_evidence_sha256']==binding['sha256'],'Frozen sorts do not bind AP evidence')
    require(sort_identity['identity']['numpy']=='1.22.4' and sort_identity['identity']['binary_sha256']=='674f8b0f4c7dec1adb4232dcdde3ec3a54fae75ea3c0984822489271b3226c40','Original AP sorting environment differs')
    with ap_audit.np.load(indices_path,allow_pickle=False) as pack:sorts={k:pack[k] for k in pack.files}
    require(set(sorts)=={m+'_'+k for m in ('control','integer') for k in ('0.3','0.5','0.7')},'Frozen sort set incomplete')
    for method,spec in evidence['methods'].items():
        audit=receipt['methods'][method]
        require(audit['frames_sha256']==spec['frames_sha256'],'Audited frames differ')
        result=json.loads(bound_file(base,audit['result_file'],audit['result_sha256']).read_text())
        rows=[json.loads(s) for s in bound_file(base,spec['frames_file'],spec['frames_sha256']).read_text().splitlines()]
        require(rows and rows[0]['initial_state_sha256']==manifest['initial_state_sha256'][method],'AP initial EMA differs from frozen source state')
        stats=ap_audit.recompute(rows)
        require(result['result_stats']==stats and result['ap']==spec['ap'] and result['frames_sha256']==spec['frames_sha256'] and result['count']==1789,'AP result does not bind complete frame TP/FP')
        for name,key in zip(('ap30','ap50','ap70'),('0.3','0.5','0.7')):
            sort_spec=sort_identity['identity']['points'][method+'_'+key]
            require(sort_spec['frames_sha256']==spec['frames_sha256'],'Frozen sort frame trace differs')
            value=ap_audit.check_ap(stats[key],spec['ap'][name],sorts[method+'_'+key],sort_spec)
            require(abs(value-audit['recomputed_ap'][name])<=1e-12,'Audited AP recomputation differs')
