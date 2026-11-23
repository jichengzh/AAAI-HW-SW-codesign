"""Real frozen Orin worker correctness smoke; NOT a throughput/network benchmark."""

import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT = Path('/home/jichengzhi/coptv2x_hil_pipeline_20261010')
RUNTIME = ROOT / 'runtime'
OLD = Path('/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z')
sys.path.insert(0, str(RUNTIME / 'simulation/leaderboard/team_code/closedloop'))
from coptv2x_hil_backend import create_feature_payload_validator, hash_feature_payload
from ditl_edge_protocol import ProtocolLimits, decode_wire_frame, recv_wire_frame
from coptv2x_hil_trigger import HIL_TRIGGER_SCHEMA_VERSION


def run():
    manifest_path = ROOT / 'evidence/worker_smoke_manifest_20.json'
    manifest = json.loads(manifest_path.read_text())
    samples = manifest['samples']
    assert len(samples) == 20
    validator = create_feature_payload_validator(RUNTIME / 'simulation/leaderboard/team_code/agent_config/coptv2x_hil')
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    listener.settimeout(180)
    result_port = listener.getsockname()[1]
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        worker_port = reservation.getsockname()[1]
    rows, errors = [], []

    def collect():
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(180)
                limits = ProtocolLimits(max_payload_bytes=512 * 1024 ** 2, max_raw_payload_bytes=512 * 1024 ** 2)
                for index, sample in enumerate(samples):
                    header_bytes, payload, _ = recv_wire_frame(connection, limits)
                    header, outputs = decode_wire_frame(header_bytes, payload, limits, readonly=True)
                    validator(header['artifact_identity'], outputs, payload_semantics=header['payload_semantics'])
                    assert header['sample_id'] == sample['sample_id']
                    assert header['source_frame'] == sample['source_frame']
                    assert header['sequence_no'] == index
                    assert header['request_id'] == 'real-pipeline-worker-smoke:%d' % index
                    assert hash_feature_payload(outputs) == header['payload_sha256']
                    assert header['tensor_bundle']['compression'] == 'none'
                    assert header['batch_order'] == ['rsu']
                    rows.append({'sequence_no': index, 'source_frame': sample['source_frame'],
                                 'sample_id': sample['sample_id'], 'payload_sha256': header['payload_sha256'],
                                 'payload_bytes': len(payload)})
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    command = [sys.executable, str(RUNTIME / 'scripts/coptv2x_hil_orin_worker.py'),
               '--data-root', str(OLD / 'input'), '--manifest', str(manifest_path),
               '--artifact-root', '/home/jichengzhi/lane_c_orin_stage6_five_config_20260724',
               '--native-repo-root', str(OLD / 'native/V2Xverse'),
               '--worker-port', str(worker_port), '--result-port', str(result_port),
               '--result-count', '20', '--max-samples', '20', '--pipeline-depth', '2',
               '--serial-optimized', '--startup-arm-id', 'joint_shcosearch',
               '--runtime-warmup-count', '1', '--preload-static-inputs', '--wire-compression', 'none']
    process = None
    try:
        with (ROOT / 'evidence/worker_real_smoke.stdout').open('w') as stdout, (ROOT / 'evidence/worker_real_smoke.stderr').open('w') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('worker exited before ready: %s' % process.returncode)
                try:
                    connection = socket.create_connection(('127.0.0.1', worker_port), timeout=1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('worker readiness timeout')
                    time.sleep(.1)
            acknowledgments = []
            for index, sample in enumerate(samples):
                trigger = {'schema_version': HIL_TRIGGER_SCHEMA_VERSION, 'run_id': 'real-pipeline-worker-smoke',
                           'route_id': manifest['route_id'], 'scenario_id': manifest['scenario_id'],
                           'sample_id': sample['sample_id'], 'source_frame': sample['source_frame'],
                           'sequence_no': index, 'source_sim_time_ns': sample['source_frame'] * 50000000,
                           'arm_id': 'joint_shcosearch'}
                if index:
                    connection = socket.create_connection(('127.0.0.1', worker_port), timeout=30)
                with connection:
                    connection.settimeout(30)
                    connection.sendall((json.dumps(trigger) + '\n').encode())
                    reply = connection.makefile('rb').readline(2048).decode().strip()
                expected = 'ACCEPTED real-pipeline-worker-smoke:%d' % index
                assert reply == expected, (reply, expected)
                acknowledgments.append(reply)
            exit_code = process.wait(timeout=180)
            thread.join(timeout=180)
            assert not thread.is_alive(), 'collector failed to drain'
            if errors:
                raise errors[0]
            assert exit_code == 0 and len(rows) == 20
        return {'passed': True, 'completed': len(rows), 'admission_replies': acknowledgments,
                'results': rows, 'scope': 'real_orin_worker_loopback_no_ros_no_carla_no_network_timing',
                'local_handoff_ack_required': False, 'final_ros_ack_tested_here': False}
    finally:
        listener.close()
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
