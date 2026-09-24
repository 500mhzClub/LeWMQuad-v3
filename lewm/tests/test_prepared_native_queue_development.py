"""Scheduling, process identity, negative results and incomplete evidence gates."""
from copy import deepcopy
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_go2_prepared_native_queue_v1 as q


def stat(state='S', start=None):
    return '2534319 (python with ) name) '+ ' '.join(
        [state]+['0']*18+[str(q.OWNER_START_TICKS if start is None else start)]+['0']*10)


def test_original_owner_identity_survives_spaces_and_parentheses():
    assert q.owner_state(stat(), q.BOOT_ID)
    assert not q.owner_state(stat('Z'), q.BOOT_ID)
    assert not q.owner_state(None, q.BOOT_ID)
    with pytest.raises(ValueError, match='PID reused'):
        q.owner_state(stat(start=q.OWNER_START_TICKS+1), q.BOOT_ID)
    with pytest.raises(ValueError, match='boot'):
        q.owner_state(None, 'different-boot')


@pytest.mark.parametrize(('command', 'expected'), [
    (['python', 'scripts/run_go2_supervised_rollout_mazes_v1.py'], True),
    (['python', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main()'], True),
    (['python', 'scripts/run_go2_direct_flow_maze03_pilot_v1.py', '--preflight-only'], False),
    (['python', 'scripts/read_go2_example.py'], False),
])
def test_competing_native_and_preflight_distinction(command, expected):
    assert q.competing_command(command) is expected


def test_wait_does_not_admit_live_owner_or_orphan_worker(monkeypatch):
    live = iter([True, False, False])
    others = iter([[], [dict(pid=25)], []])
    sleeps, events = [], []
    monkeypatch.setattr(q, 'owner_live', lambda: next(live))
    monkeypatch.setattr(q, 'competitors', lambda: next(others))
    monkeypatch.setattr(q.time, 'sleep', sleeps.append)
    q.wait_for_idle(lambda status, **kw: events.append((status, kw)), original=True)
    assert sleeps == [30, 30]
    assert events[0][1]['original_supervisor_live'] is True
    assert events[1][1]['competing_processes'] == [dict(pid=25)]


def evidence(spec):
    records, audits, prefixes = [], [], []
    for name, index in spec['cases']:
        prefix = dict(physical_and_public_prefix_exact=True,
            all_preintervention_requested_commands_exact=True,
            complete_candidate_decisions_match_prospective_prefix=True)
        audit = dict(layout_index=index, raw_sensor_reconstruction_pass=True,
            raw_command_audit_pass=True, raw_model_command_replay_pass=True,
            model_state_unchanged=True, verified_round_trip=False,
            native_evaluation={'arrivals': 0}, strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[], renderer_capture_audit={'pass': True})
        record = dict(case=name, layout_index=index, status=spec['worker_status'],
            model_state_unchanged=True, prefix_comparison=deepcopy(prefix), artifact_sha256={},
            **{k:audit[k] for k in ('verified_round_trip', 'native_evaluation',
                'strict_physical_visibility_pass', 'hard_measurement_failed_frames', 'renderer_capture_audit')})
        records.append(record); audits.append(audit); prefixes.append(prefix)
    result = dict(status=spec['result_status'], conditions=records, measured_round_trip_successes=0)
    launch = dict(native_scene_workers=1, output_root=str(q.BASE/spec['output']))
    launch['planned_cases' if spec == q.SUPERVISED else 'planned_case'] = (
        [list(c) for c in spec['cases']] if spec == q.SUPERVISED else list(spec['cases'][0]))
    key = 'learned_cohort_result_sha256' if spec['arguments'][0].startswith('--learned') else 'prospective_prefix_result_sha256'
    launch[key] = spec['arguments'][1]
    return result, launch, audits, prefixes, deepcopy(records)


@pytest.mark.parametrize('spec', [q.SUPERVISED, *q.JOBS])
def test_scientific_negative_with_original_raw_audit_advances(spec):
    report = q.admit_outcomes(spec, *evidence(spec))
    assert report['measured_round_trip_successes'] == 0
    assert report['scientific_success_required'] is False


@pytest.mark.parametrize('fault', ['audit', 'prefix', 'worker', 'terminal', 'missing_case',
    'order', 'source_argument', 'count', 'audit_outcome', 'model', 'concurrency'])
def test_incomplete_or_inconsistent_native_evidence_stops(fault):
    spec = q.SUPERVISED
    result, launch, audits, prefixes, terminals = evidence(spec)
    if fault == 'audit': audits[0]['raw_sensor_reconstruction_pass'] = False
    elif fault == 'prefix': prefixes[0]['physical_and_public_prefix_exact'] = False
    elif fault == 'worker': result['conditions'][0]['failure'] = 'retained failure'
    elif fault == 'terminal': terminals[0]['status'] = 'FAILED'
    elif fault == 'missing_case': result['conditions'].pop()
    elif fault == 'order': result['conditions'].reverse()
    elif fault == 'source_argument': launch['learned_cohort_result_sha256'] = 'altered'
    elif fault == 'count': result['measured_round_trip_successes'] = 1
    elif fault == 'audit_outcome': audits[0]['verified_round_trip'] = True
    elif fault == 'model': result['conditions'][0]['model_state_unchanged'] = False
    elif fault == 'concurrency': launch['native_scene_workers'] = 2
    with pytest.raises(ValueError):
        q.admit_outcomes(spec, result, launch, audits, prefixes, terminals)


def test_fixed_sequence_waits_authenticates_and_preserves_negative(monkeypatch, tmp_path):
    monkeypatch.setattr(q, 'OUTPUT', tmp_path)
    trace = []
    monkeypatch.setattr(q, 'wait_for_idle', lambda event, **kw: trace.append(('wait', kw)))
    def authenticate(spec, sources, sha):
        trace.append(('authenticate', spec['stem'], sha))
        return dict(measured_round_trip_successes=0)
    def execute(spec, sources, event):
        trace.append(('execute', spec['stem']))
        return dict(measured_round_trip_successes=0)
    completed = q.run_sequence({}, lambda *a, **k: None, authenticate=authenticate, execute=execute)
    assert len(completed) == 4
    assert trace == [('wait', {'original': True}),
        ('authenticate', q.SUPERVISED['stem'], q.SUPERVISED_LAUNCH_SHA),
        *[('execute', j['stem']) for j in q.JOBS]]


@pytest.mark.parametrize('failed_position', [-1, 0, 1, 2])
def test_queue_failure_never_retries_or_starts_later_case(monkeypatch, tmp_path, failed_position):
    monkeypatch.setattr(q, 'OUTPUT', tmp_path)
    monkeypatch.setattr(q, 'wait_for_idle', lambda *a, **k: None)
    calls = []
    def authenticate(*args):
        if failed_position == -1: raise ValueError('supervised failed')
        return dict(measured_round_trip_successes=0)
    def execute(spec, *args):
        calls.append(spec['stem'])
        if len(calls)-1 == failed_position: raise ValueError('native failed')
        return dict(measured_round_trip_successes=0)
    with pytest.raises(ValueError):
        q.run_sequence({}, lambda *a, **k: None, authenticate=authenticate, execute=execute)
    assert calls == [j['stem'] for j in q.JOBS[:failed_position+1]]


@pytest.mark.parametrize('exit_code', [0, 1])
def test_child_original_args_environment_timeout_and_failed_exit(monkeypatch, tmp_path, exit_code):
    monkeypatch.setattr(q, 'BASE', tmp_path)
    monkeypatch.setattr(q, 'OUTPUT', tmp_path)
    monkeypatch.setattr(q, 'validate_root', lambda p, **kw: p)
    monkeypatch.setattr(q, 'wait_for_idle', lambda *a, **k: None)
    monkeypatch.setattr(q, 'competitors', lambda: [])
    monkeypatch.setattr(q, 'verify', lambda *a: None)
    monkeypatch.setattr(q, 'hardware', lambda: dict(memory_available_bytes=40*q.GIB, artifact_free_bytes=60*q.GIB))
    monkeypatch.setenv('PYTHONOPTIMIZE', '2')
    authenticated, launched = [], []
    monkeypatch.setattr(q, 'authenticate_completed', lambda *args: authenticated.append(args) or {'pass': True})
    outcomes = iter([subprocess.TimeoutExpired('native', 30), exit_code])
    def wait(timeout):
        assert timeout == 30
        value = next(outcomes)
        if isinstance(value, Exception): raise value
        return value
    def popen(command, **kw):
        launched.append((command, kw))
        return SimpleNamespace(pid=123, wait=wait)
    if exit_code:
        with pytest.raises(ValueError, match='no automatic retry'):
            q.execute_one(q.JOBS[1], {}, lambda *a, **k: None, popen=popen)
        assert not authenticated
    else:
        assert q.execute_one(q.JOBS[1], {}, lambda *a, **k: None, popen=popen) == {'pass': True}
        assert len(authenticated) == 1
    assert len(launched) == 1
    command, settings = launched[0]
    assert command == [str(q.PYTHON), q.JOBS[1]['runner'], '--prefix-result-sha256',
        '3cbd24abad8a6c70565648977ce8482df37b4c90b8a2e28c728799910cf402b5']
    assert settings['cwd'] == q.ROOT and 'PYTHONOPTIMIZE' not in settings['env']
    assert {k:settings['env'][k] for k in q.ENVIRONMENT} == q.ENVIRONMENT


def test_preexisting_output_never_launches(monkeypatch, tmp_path):
    monkeypatch.setattr(q, 'BASE', tmp_path)
    monkeypatch.setattr(q, 'validate_root', lambda p, **kw: p)
    (tmp_path/q.JOBS[0]['output']).mkdir()
    with pytest.raises(ValueError, match='no skip, retry or resume'):
        q.execute_one(q.JOBS[0], {}, lambda *a, **k: None,
            popen=lambda *a, **k: pytest.fail('must not launch'))


@pytest.mark.parametrize('fault', [None, 'changed_output', 'missing_audit_binding',
    'foreign_source', 'wrong_launch', 'missing_worker_artifact', 'original_failure', 'verifier_failure'])
def test_completed_authentication_checks_actual_files_and_original_verifier(monkeypatch, tmp_path, fault):
    from scripts import navigation_artifact_root_development as custody
    monkeypatch.setattr(custody, 'BASE', tmp_path)
    monkeypatch.setattr(q, 'BASE', tmp_path)
    monkeypatch.setattr(q, 'verify', lambda sources: None)
    spec = q.JOBS[2]
    root = tmp_path/spec['output']; root.mkdir()
    result, launch, audits, prefixes, terminals = evidence(spec)
    sources = {'scripts/frozen_example.py': 'a'*64}
    launch['source_sha256'] = deepcopy(sources)
    result['source_sha256'] = deepcopy(sources)
    name = spec['cases'][0][0]
    (root/name).mkdir()
    (root/name/'evidence.bin').write_bytes(b'original evidence')
    worker_bindings = {name+'/evidence.bin': q.digest(root/name/'evidence.bin')}
    for suffix, value in [('_audit.json', audits[0]), ('_prefix_comparison.json', prefixes[0])]:
        q.write_json(root/(name+suffix), value)
        worker_bindings[name+suffix] = q.digest(root/(name+suffix))
    result['conditions'][0]['artifact_sha256'] = deepcopy(worker_bindings)
    terminals[0]['artifact_sha256'] = deepcopy(worker_bindings)
    q.write_json(root/(name+'_worker_terminal.json'), terminals[0])
    q.write_json(root/'launch.json', launch)
    launch_sha = q.digest(root/'launch.json')
    bindings = {**worker_bindings, 'launch.json': launch_sha,
        name+'_worker_terminal.json': q.digest(root/(name+'_worker_terminal.json'))}
    result['artifact_sha256'] = bindings
    calls = []
    def original_verifier(actual):
        calls.append(actual)
        if fault == 'verifier_failure': raise ValueError('original input verifier failed')
    monkeypatch.setattr(q.importlib, 'import_module', lambda module: SimpleNamespace(verify_inputs=original_verifier))
    if fault == 'changed_output': (root/name/'evidence.bin').write_bytes(b'changed evidence')
    elif fault == 'missing_audit_binding': del bindings[name+'_audit.json']
    elif fault == 'foreign_source': sources['scripts/frozen_example.py'] = 'b'*64
    elif fault == 'wrong_launch': launch_sha = 'f'*64
    elif fault == 'missing_worker_artifact': del bindings[name+'/evidence.bin']
    elif fault == 'original_failure': q.write_json(root/'failure.json', {'reason': 'retained'})
    q.write_json(root/'result.json', result)
    if fault:
        with pytest.raises(ValueError):
            q.authenticate_completed(spec, sources, launch_sha)
    else:
        report = q.authenticate_completed(spec, sources, launch_sha)
        assert report['result_sha256'] == q.digest(root/'result.json')
        assert report['original_verifier_reexecuted'] is True and len(calls) == 1
        assert report['measured_round_trip_successes'] == 0
