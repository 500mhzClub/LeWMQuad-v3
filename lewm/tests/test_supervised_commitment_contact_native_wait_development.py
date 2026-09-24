"""One future native launch only after the observed original queue completes."""
from types import SimpleNamespace
import subprocess
import pytest
from scripts import await_go2_supervised_commitment_contact_native_v1 as waiter
from scripts.run_go2_prepared_native_queue_v1 import competing_command


def stat(state='S', ticks=None):
    fields = [state]+['0']*18+[str(waiter.OWNER_START_TICKS if ticks is None else ticks)]
    return '2551088 (python) '+' '.join(fields)


def test_original_live_owner_identity_and_observer_name():
    assert waiter.owner_state(stat(), waiter.BOOT_ID)
    assert not waiter.owner_state(None, waiter.BOOT_ID)
    assert not waiter.owner_state(stat('Z'), waiter.BOOT_ID)
    with pytest.raises(ValueError): waiter.owner_state(stat(ticks=1), waiter.BOOT_ID)
    with pytest.raises(ValueError): waiter.owner_state(None, 'different boot')
    assert not competing_command([str(waiter.PYTHON), waiter.SOURCE])
    assert competing_command([str(waiter.PYTHON), waiter.native.SOURCE])


@pytest.mark.parametrize('fault', [None, 'queue_failure', 'missing_result', 'timeout', 'busy'])
def test_waiter_never_treats_live_queue_or_other_owner_as_idle(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(waiter, 'QUEUE', tmp_path)
    if fault == 'queue_failure': (tmp_path/'failure.json').write_text('{}')
    live = iter([True, False, False]); busy = iter([[], [{'pid':1}] if fault == 'busy' else [], []])
    monkeypatch.setattr(waiter, 'owner_live', lambda:next(live))
    monkeypatch.setattr(waiter, 'competitors', lambda:next(busy))
    def path(root, name):
        assert root == tmp_path and name == 'result.json'
        if fault == 'missing_result': raise ValueError('no completed result')
        return tmp_path/name
    monkeypatch.setattr(waiter, 'artifact_path', path); monkeypatch.setattr(waiter, 'digest', lambda p:'r'*64)
    checks = []; sleeps = []; events = []
    def verify(sha, sources):
        checks.append(sha); assert sources == {'frozen':'sources'}
        return {'queue_result_sha256':sha}
    monkeypatch.setattr(waiter, 'verify_queue_completion', verify)
    clocks = iter([0, waiter.WAIT_SECONDS if fault == 'timeout' else 1, 2, 3])
    if fault in ('queue_failure', 'missing_result', 'timeout'):
        with pytest.raises(ValueError): waiter.wait_for_queue({'frozen':'sources'}, lambda *a,**k:events.append((a,k)), sleep=sleeps.append, clock=lambda:next(clocks))
        assert not checks and not any(a == ('ORIGINAL_QUEUE_COMPLETED',) for a,k in events)
    else:
        receipt = waiter.wait_for_queue({'frozen':'sources'}, lambda *a,**k:events.append((a,k)), sleep=sleeps.append, clock=lambda:next(clocks))
        assert receipt == {'queue_result_sha256':'r'*64}
        assert sleeps == [30]*(2 if fault == 'busy' else 1)
        assert events[-1][0] == ('ORIGINAL_QUEUE_COMPLETED',)


@pytest.mark.parametrize('fault', [None, 'nonzero', 'source', 'queue', 'memory', 'storage', 'busy', 'existing'])
def test_exact_child_arguments_and_no_restart_on_observation_timeout(monkeypatch, tmp_path, fault):
    output = tmp_path/'waiter'; output.mkdir(); native_output = tmp_path/'native'
    monkeypatch.setattr(waiter, 'OUTPUT', output); monkeypatch.setattr(waiter.native, 'OUTPUT', native_output)
    if fault == 'existing': native_output.mkdir()
    monkeypatch.setattr(waiter, 'validate_root', lambda *a,**k:None)
    sources = {'fixed':'sources'}; receipt = {'queue_result_sha256':'q'*64}; calls = []; events = []
    def verify(value):
        assert value is sources
        if fault == 'source': raise ValueError('source changed')
    monkeypatch.setattr(waiter, 'verify', verify)
    monkeypatch.setattr(waiter, 'verify_queue_completion', lambda *a:dict(receipt, changed=True) if fault == 'queue' else dict(receipt))
    monkeypatch.setattr(waiter, 'hardware', lambda:dict(memory_available_bytes=(1 if fault == 'memory' else 64)*1024**3,
        artifact_free_bytes=(1 if fault == 'storage' else 100)*1024**3))
    def idle():
        if fault == 'busy': raise ValueError('native owner live')
    monkeypatch.setattr(waiter, 'require_native_idle', idle)
    monkeypatch.setattr(waiter, 'authenticate_native', lambda *a:dict(queue_result_sha256='q'*64, measured_round_trip_successes=0))
    waits = []
    def popen(command, **kwargs):
        calls.append(command)
        assert command == [str(waiter.PYTHON), waiter.native.SOURCE, '--prefix-result-sha256', waiter.native.PREFIX_SHA,
            '--queue-result-sha256', 'q'*64]
        assert kwargs['env']['PYTHONHASHSEED'] == '0' and kwargs['env']['PYTHONPATH'] == '.:lewm_genesis:lewm_worlds'
        assert 'PYTHONOPTIMIZE' not in kwargs['env'] and kwargs['cwd'] == waiter.ROOT
        def wait(timeout):
            assert timeout == 30; waits.append(True)
            if len(waits) == 1: raise subprocess.TimeoutExpired(command, timeout)
            return 1 if fault == 'nonzero' else 0
        return SimpleNamespace(pid=456, wait=wait)
    if fault is None:
        result = waiter.execute(sources, receipt, lambda *a,**k:events.append(a[0]), popen=popen)
        assert result['measured_round_trip_successes'] == 0
        assert len(calls) == 1 and len(waits) == 2 and events == ['NATIVE_CHILD_STARTED', 'NATIVE_CHILD_LIVE', 'NATIVE_CHILD_EXITED']
    else:
        with pytest.raises(ValueError): waiter.execute(sources, receipt, lambda *a,**k:events.append(a[0]), popen=popen)
        assert len(calls) == (1 if fault == 'nonzero' else 0)


@pytest.mark.parametrize('fault', [None, 'audit', 'incomplete_command', 'worker', 'source', 'missing_binding', 'outcome', 'model'])
def test_completed_native_authentication_includes_negative_outcomes_and_all_bound_evidence(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(waiter.native, 'OUTPUT', tmp_path)
    name = waiter.native.CASE[0]; sources = {'source':'sha'}
    prefix = dict(physical_and_public_prefix_exact=True, all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True, candidate_intervention_command_completed=True,
        common_prefix_frames=4, first_intervention_frame=3, physical_prefix_samples=900, raw_model_forecast_comparisons=1)
    audit = dict(raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True, raw_model_command_replay_pass=True,
        model_state_unchanged=True, verified_round_trip=False, native_evaluation={}, strict_physical_visibility_pass=False,
        hard_measurement_failed_frames=[], renderer_capture_audit={})
    worker = dict(case=name, layout_index=1, status='SUPERVISED_COMMITMENT_CONTACT_MAZE01_COLLECTED_AND_RAW_AUDITED',
        model_state_unchanged=True, model_state_sha256=waiter.native.SUPERVISED_STATE, prefix_comparison=prefix,
        artifact_sha256={name+'/raw':'raw_sha'}, **{k:audit[k] for k in ('verified_round_trip', 'native_evaluation',
            'strict_physical_visibility_pass', 'hard_measurement_failed_frames', 'renderer_capture_audit')})
    launch = dict(source_sha256=dict(sources), queue_result_sha256='q'*64)
    bindings = {n:'sha' for n in ('launch.json', name+'_audit.json', name+'_prefix_comparison.json', name+'_worker_terminal.json', name+'_worker.log')}
    bindings.update(worker['artifact_sha256'])
    result = dict(status='SUPERVISED_COMMITMENT_CONTACT_MAZE01_PILOT_V1_COMPLETE', conditions=[worker],
        source_sha256=dict(sources), artifact_sha256=bindings, measured_round_trip_successes=0, queue_result_sha256='q'*64)
    if fault == 'audit': audit['raw_model_command_replay_pass'] = False
    elif fault == 'incomplete_command': prefix['candidate_intervention_command_completed'] = False
    elif fault == 'worker': worker['status'] = 'FAILED'
    elif fault == 'source': result['source_sha256'] = {'other':'sha'}
    elif fault == 'missing_binding': bindings.pop(name+'_audit.json')
    elif fault == 'outcome': result['measured_round_trip_successes'] = 1
    elif fault == 'model': worker['model_state_sha256'] = '0'*64
    files = {'result.json':result, 'launch.json':launch, name+'_audit.json':audit,
        name+'_prefix_comparison.json':prefix, name+'_worker_terminal.json':worker}
    monkeypatch.setattr(waiter, 'artifact_path', lambda p,n:p/n); monkeypatch.setattr(waiter, 'digest', lambda p:'r'*64)
    monkeypatch.setattr(waiter, 'read_json', lambda p,n:files[n])
    monkeypatch.setattr(waiter, 'verify_artifacts', lambda *a:None); monkeypatch.setattr(waiter, 'verify', lambda *a:None)
    calls = []; monkeypatch.setattr(waiter.native, 'verify_inputs', lambda value:calls.append(value))
    if fault is None:
        report = waiter.authenticate_native(sources)
        assert report['measured_round_trip_successes'] == 0 and report['all_raw_audits_pass']
        assert calls == [launch] and not report['scientific_success_required']
    else:
        with pytest.raises(ValueError): waiter.authenticate_native(sources)
