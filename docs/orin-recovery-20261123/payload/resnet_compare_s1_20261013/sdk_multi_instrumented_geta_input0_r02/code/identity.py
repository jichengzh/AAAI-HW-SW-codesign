"""Frozen three-point identity checks for the TensorRT SDK path.

This module only checks bytes and metadata; it does not initialize the GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path

CODE_ROOT = Path(__file__).parent
BACKEND_SHA = '63e7bcf178f167b20d5f9539d0afdb2650b98a1b0d93c4cf855f2c73a0c403f8'
STREAM_SHA = '6214df738caa17bb1d51e9b08d7c62658c8c5a89ac21a9de4da16a4684ed9b7d'
FINAL_ROWS = '98395d0907d3bee96b59ed4baa9d5aa6d0b693599d33163a3fc7aa048595d2c5'
FINAL = '34117fa447b6cc4236b59e3f2c804f2a69fd01f221b25bcc7cf0983400e077a0'
POINTS = {
    'dense': {'engine': '2a7110a8ebc458170d77deba53270e25657f04c84b8b8f21400021114c0be7df',
              'onnx': '232f666524f07b9d3971381462365774222f0488f6912e144f73c50d36d2fc21',
              'manifest': 'bded4e74c656424bf38cce70b2375ff61107ff4e2eb5e0817d84a1aa006427b2',
              'layers': '6f94d9216855acf2d9e58a2adae7e7ae51e667a0bfc5a4c4d80a8d894dda0f3d',
              'final': FINAL, 'layer_format': 'layers'},
    'copt': {'engine': '2aca0cb77d7b962825c63c5d7bba69175b857ee2a3e07542df6046dfab5c8aee',
             'onnx': '55b4b63df303d8702dbe8d0357508266000b50168240098780c9a1009efdf6ad',
             'manifest': 'd66f3318631727964d3f0e0893dbf6d270c46117a59fab440a289aab0b25279d',
             'layers': '6e39bfc9642f59b47371e63fdbf66bf1184bf64bb8070bbaf76244e47e77ab2d',
             'final': FINAL, 'layer_format': 'prepared'},
    'geta': {'engine': '59f030a76d8565ba888ca354d666b73337309f7e4c8628ae1d6720ed31118690',
             'onnx': '38f01f97bf8d83154fedffbbd83f212fcb3437bd882b25e763723dfd1e4e7121',
             'manifest': '31887a0d906f1e9be813c4df245678ec90290feb829950d4e324c88f678fce44',
             'layers': '76f8d96599a679b91bab87255989a4a8d2f6d2dac892343d02a952535c43062e',
             'final': FINAL, 'layer_format': 'prepared'},
}
LIBRARIES = {
    '/usr/lib/aarch64-linux-gnu/libnvinfer.so.8.5.2': '4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca',
    '/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.5.2': '5a30c8e46ba47c3a93eb825ad0a27cbe969fb6f174ea963e4d22dfe33630097a',
    '/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298': '6ca38be1cc6fa0b8ddad0313eea2a4483be27d2e8474437f9f7b7a536b7197a2',
    '/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1.1': '151ad1343173e919304e08d4c6801508303111f2b1c7b3737f6b5134590808f4',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_identity():
    names = ('identity.py', 'multi_entry.py', 'multi_host.py', 'host_entry.py',
             'sdk_wrapper.py', 'sdk_backend.py', 'frozen_stream.py', 'sdk_entry.py', 'sdk_contract.py', 'stage_logger.py')
    values = {name: sha(CODE_ROOT/name) for name in names}
    if values['sdk_backend.py'] != BACKEND_SHA:
        raise ValueError('SDK backend changed')
    fixed = {'sdk_entry.py':'b994002ca6b8eb1507de658959e00abd2297c88cc37e9ac1317d37029dade81c', 'sdk_contract.py':'246568da4f81a263e5b51cc0e6fe6daba2fe26010f905b5d048a93f5a5ae382f', 'stage_logger.py':'b39f544a3d64aea7af7cc272476616a803a16afdd035ba9bd5c536fab8759941'}
    if any(values[n] != h for n,h in fixed.items()):
        raise ValueError('Instrumented SDK source changed')
    if values['frozen_stream.py'] != STREAM_SHA:
        raise ValueError('Frozen stream changed')
    return values


def point_pins(point):
    if point not in POINTS:
        raise ValueError('Unknown point: ' + str(point))
    return POINTS[point]


def expected_layers(point, path):
    pins = point_pins(point)
    if sha(path) != pins['layers']:
        raise ValueError('layers SHA mismatch')
    doc = json.loads(Path(path).read_text())
    if pins['layer_format'] == 'layers':
        return doc['Layers']
    if pins['layer_format'] == 'prepared':
        return [row['inspector'] for row in doc['layer_precision']]
    raise ValueError('Unknown layer format')


def verify_inspector(expected, actual):
    if actual != expected:
        raise ValueError('Actual inspector differs from pinned layers')


def verify_libraries(actual):
    if any(actual.get(path) != value for path, value in LIBRARIES.items()):
        raise ValueError('Actual libraries missing or changed')


def validate(point, args):
    pins = point_pins(point)
    for name in ('engine', 'manifest', 'onnx', 'final', 'layers'):
        if sha(getattr(args, name)) != pins[name]:
            raise ValueError(name + ' SHA mismatch')
    final = json.loads(Path(args.final).read_text())
    if (final['count'] != 50000 or len(final['samples']) != 50000 or
            final['sha256'] != FINAL_ROWS):
        raise ValueError('Frozen 50000 split differs')
    manifest = json.loads(Path(args.manifest).read_text())
    return {'point': point, 'manifest_identity': pins['final'], 'candidate_id': manifest['candidate']['candidate_id'],
            'engine_sha256': pins['engine'], 'source_sha256': {k: pins[k] for k in
            ('manifest', 'onnx', 'final', 'layers')}, 'final_sample_manifest_sha256': FINAL_ROWS,
            'execution_path': 'TensorRT8.5_CUDART_standard_memory_stream_SDK',
            'code_sha256': code_identity(), 'selection_used': False, 'timing_or_energy_qualified': False}


def main():
    parser = argparse.ArgumentParser(description='Offline three-point SDK identity preflight')
    parser.add_argument('--point', required=True, choices=sorted(POINTS))
    for name in ('engine', 'manifest', 'onnx', 'final', 'layers'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.point, args), sort_keys=True))


if __name__ == '__main__':
    main()
