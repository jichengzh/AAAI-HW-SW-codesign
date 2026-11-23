"""Diagnostic only: frozen Orin outputs, no search/training/control changes."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--native-repo-root', required=True)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runtime = Path(args.runtime)
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / 'simulation/leaderboard/team_code/closedloop'))
    import numpy as np
    from coptv2x_hil_artifacts import HIL_ARM_IDS
    from coptv2x_hil_backend import CoptV2xHilBackend, hash_feature_payload
    from coptv2x_hil_trigger import parse_hil_trigger
    from ditl_edge_protocol import encode_tensor_bundle, decode_tensor_bundle

    root = Path(args.input_root)
    manifest_path = root / 'local_input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sample = manifest['samples'][0]
    spec = sample['inputs']['spatial_features']
    input_path = root / spec['path']
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Frozen input SHA mismatch')
    inputs = {'spatial_features': np.load(str(input_path), allow_pickle=False)}
    report = {'scope': 'one recorded sample per frozen arm, codec diagnosis only',
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'input_sha256': spec['sha256'], 'arms': []}
    for arm in HIL_ARM_IDS:
        backend = CoptV2xHilBackend(
            arm, artifact_root=args.artifact_root,
            config_dir=runtime / 'simulation/leaderboard/team_code/agent_config/coptv2x_hil',
            native_repo_root=args.native_repo_root)
        backend.load()
        trigger = parse_hil_trigger({
            'schema_version': 'coptv2x.hil-trigger.v1', 'run_id': 'p5-codec-diagnostic',
            'route_id': manifest['route_id'], 'scenario_id': manifest['scenario_id'],
            'sample_id': sample['sample_id'], 'source_frame': sample['source_frame'],
            'source_sim_time_ns': sample['source_frame'] * 50000000,
            'sequence_no': 0, 'arm_id': arm})
        result = backend.infer_hil_validated(inputs, trigger, batch_order=('ego', 'rsu'))
        arrays = dict(result.outputs)
        reference_hash = hash_feature_payload(arrays)
        row = {'arm_id': arm, 'payload_sha256': reference_hash,
               'frozen_artifact_sha256': dict(result.artifact_identity['artifact_sha256']),
               'inference_durations_ns': dict(result.durations_ns), 'codecs': []}
        for codec in ('none', 'lz4', 'zlib'):
            start = time.monotonic_ns()
            metadata, payload = encode_tensor_bundle(arrays, compression=codec)
            encoded_ns = time.monotonic_ns() - start
            start = time.monotonic_ns()
            decoded = decode_tensor_bundle(metadata, payload)
            decoded_ns = time.monotonic_ns() - start
            actual_hash = hash_feature_payload(decoded)
            if actual_hash != reference_hash:
                raise ValueError('Lossless output verification failed')
            for name in arrays:
                if (arrays[name].dtype != decoded[name].dtype or
                        arrays[name].shape != decoded[name].shape or
                        arrays[name].tobytes() != decoded[name].tobytes()):
                    raise ValueError('Tensor bitwise identity mismatch')
            row['codecs'].append({'compression': codec, 'encode_ns': encoded_ns,
                                  'decode_ns': decoded_ns,
                                  'raw_bytes': metadata['raw_payload_bytes'],
                                  'wire_bytes': len(payload), 'bitwise_equal': True})
            del decoded, payload
        report['arms'].append(row)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
        del backend, result, arrays
        gc.collect()
        import torch
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
