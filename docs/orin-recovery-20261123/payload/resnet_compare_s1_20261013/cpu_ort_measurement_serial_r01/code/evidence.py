"""Fail-closed accuracy and 256-sample replay evidence for SDK measurements."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

POINTS=('dense','copt','geta')

def check(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def valid_hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def qualification(point, token, result_path, audit_path, expected_audit_sha, expected_auditor_sha):
    check(point in POINTS and token.get('point')==point, 'point mismatch')
    check(valid_hash(expected_audit_sha) and sha(audit_path)==expected_audit_sha, 'audit SHA mismatch')
    audit=json.loads(Path(audit_path).read_text())
    result=json.loads(Path(result_path).read_text())
    check(valid_hash(expected_auditor_sha) and audit.get('audit_code_sha256')==expected_auditor_sha, 'auditor mismatch')
    check(audit.get('artifact_sha256',{}).get('result.json')==sha(result_path), 'result SHA mismatch')
    check(audit.get('status')=='PASS_CPU_ORT_FULL50000_INDEPENDENT_AUDIT' and audit.get('samples')==50000,
          'full audited 50000 required')
    check(result.get('status')=='COMPLETE' and result.get('ssh_exit')==result.get('remote_exit')==0 and
          result.get('remote_marker_count')==1 and result.get('cleanup_required') is False and
          result.get('accuracy',{}).get('samples')==50000, 'full target completion required')
    for key in ('point','model_sha256','source_sha256','final_sample_manifest_sha256','code_sha256','execution_provider','execution_precision','ort_version','ort_pybind_sha256','session_config'):
        check(audit.get(key)==token.get(key), 'audit identity mismatch '+key)
        check(result['accuracy']['target_identity'].get(key)==token.get(key), 'result identity mismatch '+key)
    check(audit.get('execution_provider')=='CPUExecutionProvider' and audit.get('execution_precision')=='FP32' and audit.get('ort_version')=='1.27.0' and valid_hash(audit.get('ort_pybind_sha256')), 'invalid actual CPU backend')
    check(audit.get('precision_scope')=='FP32_GRAPH_TENSORS_AND_IO; ORT_KERNEL_INTERNAL_ARITHMETIC_NOT_INDEPENDENTLY_PROVEN', 'precision scope mismatch')
    graph=audit.get('graph_precision_evidence',{})
    expected={'onnx_sha256':token['model_sha256'],'onnx_version':'1.19.1','ir_version':7,'opset':13,'nodes':122,
              'operators':{'Conv':53,'Relu':49,'MaxPool':1,'Add':16,'GlobalAveragePool':1,'Flatten':1,'Gemm':1},
              'initializers':108,'all_initializers_float32':True,'all_intermediate_tensor_types_float32':True,
              'input_shape':[1,3,224,224],'output_shape':[1,1000],
              'scope':'STATIC_GRAPH_TENSOR_TYPES_AND_IO_ONLY; ORT_KERNEL_INTERNAL_ARITHMETIC_NOT_INDEPENDENTLY_PROVEN'}
    check(graph==expected,'graph precision evidence mismatch')
    check(isinstance(audit.get('boot_id'),str) and bool(audit['boot_id']), 'boot absent')
    check(valid_hash(audit.get('chain_sha256')), 'chain absent')
    one,five=audit.get('top1_correct'),audit.get('top5_correct')
    check(type(one) is int and type(five) is int and 0<=one<=five<=50000, 'accuracy counts invalid')
    return {'point':point,'model_sha256':token['model_sha256'],'audit_sha256':expected_audit_sha,
            'result_sha256':sha(result_path),'audit_code_sha256':expected_auditor_sha,
            'ort_pybind_sha256':audit['ort_pybind_sha256'],'session_config':audit['session_config'],'boot_id':audit['boot_id']}

def reference_rows(inputs, ids, references, rows):
    check(isinstance(inputs,np.ndarray) and inputs.shape==(256,3,224,224) and inputs.dtype==np.float32 and
          inputs.flags.c_contiguous and np.isfinite(inputs).all(), 'invalid inputs')
    check(len(ids)==256 and len(set(ids))==256, 'invalid sample ids')
    check(set(references)==set(POINTS) and set(rows)==set(POINTS), 'missing points')
    for point in POINTS:
        ref=references[point]; data=rows[point]
        check(ref.shape==(256,1000) and ref.dtype==np.float32 and np.isfinite(ref).all() and len(data)==256,
              'invalid references')
        for index, row in enumerate(data):
            check(row.get('index')==index and row.get('sample_id')==ids[index], 'sample identity mismatch')
            check(row.get('input_sha256')==hashlib.sha256(inputs[index].tobytes()).hexdigest(), 'input SHA mismatch')
            check(row.get('logits_sha256')==hashlib.sha256(ref[index].tobytes()).hexdigest(), 'logits SHA mismatch')
    return True
