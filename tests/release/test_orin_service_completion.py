"""Finite services must drain the final publisher ACK before owned cleanup."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SCRIPTS = (Path(__file__).resolve().parents[2] /
           'docs/orin-recovery-20261123/payload/coptv2x_hil_pipeline_20261010/runtime/scripts')
SUPERVISOR = SCRIPTS / 'coptv2x_hil_orin_services.py'
CHILD = '''import pathlib, sys, time
root, role, status, immediate = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
(root / (role + '.ready')).touch()
while immediate != 'yes' and not (root / (role + '.release')).exists():
    time.sleep(0.01)
if role == 'publisher' and status == 0:
    (root / 'publisher.ack').touch()
(root / (role + '.exiting')).touch()
sys.exit(status)
'''


def wait_for(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail('timed out waiting for service test event')


@contextmanager
def stack(tmp_path, finite=True, statuses=None, immediate_worker=False, drain=3):
    child_script = tmp_path / 'child.py'
    child_script.write_text(CHILD)
    services = [
        {'name': role, 'argv': [sys.executable, str(child_script), str(tmp_path), role,
                               str((statuses or {}).get(role, 0)),
                               'yes' if role == 'worker' and immediate_worker else 'no'],
         'ready_file': str(tmp_path / (role + '.ready'))}
        for role in ('publisher', 'worker', 'proxy')
    ]
    config = {'state_dir': str(tmp_path), 'ready_timeout_s': 3, 'services': services}
    if finite:
        config['completion_policy'] = {'mode': 'publisher_ack', 'worker': 'worker',
                                       'publisher': 'publisher', 'drain_timeout_s': drain}
    config_path = tmp_path / 'services.json'
    config_path.write_text(json.dumps(config))
    with (tmp_path / 'supervisor.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(SUPERVISOR), '--mode', 'serve',
                                    '--config', str(config_path)], stdout=log, stderr=log)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=20)


def release(tmp_path, role):
    (tmp_path / (role + '.release')).touch()
    wait_for(lambda: (tmp_path / (role + '.exiting')).exists())


def assert_cleaned(tmp_path):
    assert json.loads((tmp_path / 'stopped.json').read_text())['cleaned'] is True
    for path in tmp_path.glob('*.owner.json'):
        if path.name != 'supervisor.owner.json':
            pid = json.loads(path.read_text())['pid']
            assert not Path('/proc/%d' % pid).exists()


def test_worker_exit_zero_waits_for_publisher_ack_before_completion(tmp_path):
    with stack(tmp_path) as process:
        wait_for(lambda: (tmp_path / 'ready.json').exists())
        release(tmp_path, 'worker')
        time.sleep(0.5)
        assert process.poll() is None
        assert not (tmp_path / 'failure.json').exists()
        assert not (tmp_path / 'completed.json').exists()
        assert not (tmp_path / 'publisher.ack').exists()
        release(tmp_path, 'publisher')
        assert process.wait(timeout=15) == 0
    assert json.loads((tmp_path / 'completed.json').read_text())['completed'] is True
    assert_cleaned(tmp_path)


def test_publisher_exit_alone_cannot_complete_before_worker(tmp_path):
    with stack(tmp_path) as process:
        wait_for(lambda: (tmp_path / 'ready.json').exists())
        release(tmp_path, 'publisher')
        time.sleep(0.5)
        assert process.poll() is None
        assert not (tmp_path / 'completed.json').exists()
        release(tmp_path, 'worker')
        assert process.wait(timeout=15) == 0
    assert (tmp_path / 'completed.json').exists()


def test_worker_can_finish_after_its_readiness_before_supervisor_observes_it(tmp_path):
    with stack(tmp_path, immediate_worker=True) as process:
        wait_for(lambda: (tmp_path / 'ready.json').exists())
        release(tmp_path, 'publisher')
        assert process.wait(timeout=15) == 0
    assert (tmp_path / 'completed.json').exists()


@pytest.mark.parametrize('role,status', [('worker', 7), ('publisher', 9), ('proxy', 0)])
def test_unexpected_service_exit_fails_and_cleans_every_child(tmp_path, role, status):
    with stack(tmp_path, statuses={role: status}) as process:
        wait_for(lambda: (tmp_path / 'ready.json').exists())
        release(tmp_path, role)
        assert process.wait(timeout=20) != 0
    assert role in json.loads((tmp_path / 'failure.json').read_text())['error']
    assert not (tmp_path / 'completed.json').exists()
    assert_cleaned(tmp_path)


def test_missing_ack_hits_drain_deadline_without_success(tmp_path):
    with stack(tmp_path, drain=0.3) as process:
        wait_for(lambda: (tmp_path / 'ready.json').exists())
        release(tmp_path, 'worker')
        assert process.wait(timeout=20) != 0
    assert 'completion' in json.loads((tmp_path / 'failure.json').read_text())['error']
    assert not (tmp_path / 'publisher.ack').exists()
    assert not (tmp_path / 'completed.json').exists()
    assert_cleaned(tmp_path)


def test_stop_during_ack_drain_cleans_without_claiming_completion(tmp_path):
    with stack(tmp_path) as process:
        wait_for(lambda: (tmp_path / 'ready.json').exists())
        release(tmp_path, 'worker')
        time.sleep(0.3)
        process.terminate()
        assert process.wait(timeout=20) == 0
    assert not (tmp_path / 'completed.json').exists()
    assert not (tmp_path / 'failure.json').exists()
    assert_cleaned(tmp_path)


def test_configuration_without_finite_policy_keeps_legacy_exit_failure(tmp_path):
    with stack(tmp_path, finite=False) as process:
        wait_for(lambda: (tmp_path / 'ready.json').exists())
        release(tmp_path, 'worker')
        assert process.wait(timeout=20) != 0
    assert (tmp_path / 'failure.json').exists()
    assert not (tmp_path / 'completed.json').exists()


def test_episode_config_selects_bounded_publisher_ack_completion():
    spec = importlib.util.spec_from_file_location('recovery_episode_services',
                                               SCRIPTS / 'coptv2x_hil_episode_services.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(runtime='/tmp/runtime', container='local-test',
                           result_count=1, result_port=55022, worker_port=56812,
                           forward_port=56811, orin_python=sys.executable,
                           artifact_root='/tmp/artifacts', ready_timeout_s=3,
                           completion_drain_timeout_s=4)
    config = module.service_configuration(args, {'token': 'fresh', 'arm_id': 'compression_only'})
    assert config['completion_policy'] == {'mode': 'publisher_ack', 'worker': 'worker',
                                           'publisher': 'publisher', 'drain_timeout_s': 4}
