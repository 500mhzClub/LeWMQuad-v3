"""Exact original owners and one deferred child; no real waiting or simulation."""
from copy import deepcopy

import pytest

from scripts import await_go2_chained_anchor_maze02_native_v1 as waiter


def identities():
    return dict(raw=waiter.RAW_RESULT_SHA, flow='b'*64)


def test_fixed_roster_contains_exact_raw_owner_and_original_final_waiter():
    assert [s[0] for s in waiter.PREREQUISITES] == ['raw', 'flow']
    assert [s[1]['pid'] for s in waiter.PREREQUISITES] == [2840884, 2827789]
    assert waiter.PREREQUISITES[0][3] == '8eba2f8dfea706109f8cec4fcf55206f36fa3b9f588c0bb2c344b492e95269cd'
    assert waiter.PREREQUISITES[1][3] == '3fc8e765b6edc16120b418e6dc8cedf1da47eb134d959f98450b2adeaa2c6c72'


def test_command_forwards_every_exact_completion_once():
    ids = identities(); cmd = waiter.command_for(ids)
    assert cmd[:3] == [waiter.sys.executable, '-B', waiter.native.SOURCE]
    assert dict(zip(cmd[3::2], cmd[4::2], strict=True)) == dict(
        zip(['--controller-completion-sha256', '--flow-wait-result-sha256'],
            [waiter.native.inputs.COMPLETION_SHA, ids['flow']], strict=True))


@pytest.mark.parametrize('fault', ['missing', 'extra', 'invalid', 'non_string'])
def test_incomplete_or_invented_command_roster_rejected(fault):
    ids = identities()
    if fault == 'missing': ids.pop('flow')
    elif fault == 'extra': ids['replacement'] = 'a'*64
    elif fault == 'invalid': ids['flow'] = 'not-a-hash'
    else: ids['flow'] = True
    with pytest.raises(ValueError): waiter.command_for(ids)


def loop_fixture(monkeypatch):
    g = waiter.wait_for_inputs.__globals__; state = dict(tick=0); events = []; checked = []
    ids = identities()
    def owner(owner):
        return state['tick'] == 0 or (state['tick'] == 1 and owner['pid'] == 2827789)
    def complete(spec, sources): checked.append((state['tick'], spec[0])); return ids[spec[0]]
    monkeypatch.setitem(g, 'owner_live', owner)
    monkeypatch.setitem(g, 'completion_identity', complete)
    monkeypatch.setitem(g, 'verify', lambda sources:None)
    def sleep(seconds): assert seconds == 30; state['tick'] += 1
    return g, state, events, checked, ids, sleep


def test_no_identity_is_inferred_while_its_original_owner_is_live(monkeypatch):
    _, state, events, checked, ids, sleep = loop_fixture(monkeypatch)
    result = waiter.wait_for_inputs({}, lambda status, **kw:events.append(kw), sleep=sleep, clock=lambda:state['tick']*30)
    assert result == ids and len(events) == 2
    assert not any(tick == 0 for tick, key in checked)
    assert (1, 'flow') not in checked and (2, 'flow') in checked
    assert events[0]['completed_result_sha256'] == {}


def test_changed_completed_identity_is_not_accepted_on_later_poll(monkeypatch):
    g, state, events, checked, ids, sleep = loop_fixture(monkeypatch)
    def changed(spec, sources): return 'f'*64 if state['tick'] == 2 and spec[0] == 'raw' else ids[spec[0]]
    monkeypatch.setitem(g, 'completion_identity', changed)
    with pytest.raises(ValueError, match='identity changed'):
        waiter.wait_for_inputs({}, lambda *a, **kw:None, sleep=sleep, clock=lambda:state['tick']*30)


def test_owner_ending_without_valid_completion_is_terminal(monkeypatch):
    g, state, events, checked, ids, sleep = loop_fixture(monkeypatch)
    def failed(*a): raise ValueError('original prerequisite failed')
    monkeypatch.setitem(g, 'completion_identity', failed)
    with pytest.raises(ValueError, match='prerequisite failed'):
        waiter.wait_for_inputs({}, lambda *a, **kw:None, sleep=sleep, clock=lambda:state['tick']*30)
    assert state['tick'] == 1


def test_live_owner_timeout_does_not_restart_any_job(monkeypatch):
    g, state, events, checked, ids, sleep = loop_fixture(monkeypatch)
    monkeypatch.setitem(g, 'owner_live', lambda owner:True)
    monkeypatch.setitem(g, 'WAIT_SECONDS', 30)
    with pytest.raises(ValueError, match='expired'):
        waiter.wait_for_inputs({}, lambda *a, **kw:None, sleep=sleep, clock=lambda:state['tick']*30)
    assert checked == [] and state['tick'] == 1


def test_existing_waiter_or_child_prevents_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(waiter, 'OUTPUT', tmp_path)
    monkeypatch.setattr(waiter, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(waiter, 'prepared_sources', lambda:pytest.fail('no source work for an existing attempt'))
    with pytest.raises(ValueError, match='exclusive'): waiter.main()


def test_completion_failure_file_is_not_bypassed(tmp_path, monkeypatch):
    monkeypatch.setattr(waiter.native, 'OUTPUT', tmp_path)
    (tmp_path/'failure.json').write_text('{}')
    monkeypatch.setattr(waiter, 'digest', lambda *a:pytest.fail('must retain failure before reading result'))
    with pytest.raises(ValueError, match='failure must be preserved'): waiter.authenticate_completed({}, identities())


def completion_fixture(tmp_path, monkeypatch, key='raw'):
    spec = next(s for s in waiter.PREREQUISITES if s[0] == key)
    spec = (spec[0], spec[1], tmp_path, spec[3], spec[4])
    sources = {'fixed.py':'fixed-sha'}
    result = dict(status=spec[4], source_sha256=sources.copy(),
        artifact_sha256={'launch.json':spec[3]}, automatic_retry=False,
        report=__import__('lewm.tests.test_chained_anchor_native_prefix_development',fromlist=['population']).population()[-1])
    monkeypatch.setattr(waiter, 'RAW_RESULT_SHA', 'a'*64)
    launch = dict(source_sha256=sources.copy())
    (tmp_path/'result.json').write_text('{}')
    monkeypatch.setattr(waiter, 'digest', lambda path:'a'*64)
    verified = []
    monkeypatch.setattr(waiter, 'verify_artifacts', lambda root, ids:verified.append((root, ids)))
    monkeypatch.setattr(waiter, 'read_json', lambda root, name:result if name == 'result.json' else launch)
    return spec, sources, result, launch, verified


@pytest.mark.parametrize('key', ['raw', 'flow'])
def test_light_completion_binds_exact_launch_and_result_without_full_admission(tmp_path, monkeypatch, key):
    spec, sources, result, launch, verified = completion_fixture(tmp_path, monkeypatch, key)
    monkeypatch.setattr(waiter.native.inputs, 'admit', lambda *a:pytest.fail('no full admission while waiting'))
    assert waiter.completion_identity(spec, sources) == 'a'*64
    assert verified == [(tmp_path, {'launch.json':spec[3], 'result.json':'a'*64})]


@pytest.mark.parametrize('fault', ['missing', 'failure', 'failure_symlink', 'status', 'launch', 'source',
    'outside_closure', 'boundary', 'native_claim', 'retry'])
def test_incomplete_or_changed_original_completion_rejected(tmp_path, monkeypatch, fault):
    spec, sources, result, launch, verified = completion_fixture(tmp_path, monkeypatch,
        'flow' if fault == 'retry' else 'raw')
    if fault == 'missing': (tmp_path/'result.json').unlink()
    if fault == 'failure': (tmp_path/'failure.json').write_text('{}')
    if fault == 'failure_symlink': (tmp_path/'failure.json').symlink_to(tmp_path/'absent')
    if fault == 'status': result['status'] = 'RUNNING'
    if fault == 'launch': result['artifact_sha256']['launch.json'] = 'b'*64
    if fault == 'source': launch['source_sha256']['fixed.py'] = 'changed'
    if fault == 'outside_closure': sources.clear()
    if fault == 'boundary': result['report']['frames'] = 561
    if fault == 'native_claim': result['report']['native_execution'] = True
    if fault == 'retry': result['automatic_retry'] = True
    with pytest.raises(ValueError): waiter.completion_identity(spec, sources)


def terminal_fixture(tmp_path, monkeypatch):
    root = tmp_path; name = waiter.native.CASE[0]; ids = identities()
    collection = {'observations':562}; readout = {'verified_round_trip':False}; prefix = {'actual':True}
    bindings = {n:'hash-'+n for n in [name+'/result.json', name+'/physics_trace.npz',
        name+'_worker_terminal.json', name+'_worker.log', name+'_audit.json', name+'_prefix_comparison.json',
        name+'_readout.json', 'launch.json', 'resource_monitor.jsonl']}
    record = dict(collection=collection, readout=readout, prefix_comparison=prefix,
        worker_log_sha256=bindings[name+'_worker.log'], artifact_sha256={name+'/physics_trace.npz':bindings[name+'/physics_trace.npz']},
        verified_round_trip=False)
    launch = dict(source_sha256={'src':'sha'}, input_admission=dict(controller_completion_sha256=waiter.native.inputs.COMPLETION_SHA,
        flow_wait_result_sha256=ids['flow'], prefix_report={'raw':True}))
    result = dict(status='CHAINED_ANCHOR_MAZE02_PILOT_V1_COMPLETE',
        artifact_sha256=bindings, source_sha256={'src':'sha'}, conditions=[record], measured_round_trip_successes=0,
        controller_completion_sha256=waiter.native.inputs.COMPLETION_SHA, prospective_prefix_result_sha256=ids['raw'], flow_wait_result_sha256=ids['flow'])
    records = {'result.json':result, 'launch.json':launch, name+'_worker_terminal.json':deepcopy(record),
        name+'/result.json':deepcopy(collection), name+'_readout.json':deepcopy(readout),
        name+'_prefix_comparison.json':deepcopy(prefix), name+'_audit.json':{'audit':True}}
    monkeypatch.setattr(waiter.native, 'OUTPUT', root)
    monkeypatch.setattr(waiter, 'digest', lambda p:'a'*64)
    monkeypatch.setattr(waiter, 'read_json', lambda r, n:records[n])
    monkeypatch.setattr(waiter, 'verify_artifacts', lambda *a:None)
    monkeypatch.setattr(waiter, 'verify', lambda *a:None)
    monkeypatch.setattr(waiter.native, 'artifacts', lambda *a:['result.json','physics_trace.npz'])
    calls = []
    monkeypatch.setattr(waiter.native, 'verify_inputs', lambda value:calls.append('inputs'))
    monkeypatch.setattr(waiter.native, 'require_worker', lambda *a:calls.append('worker'))
    def readout_fn(*a): calls.append('readout'); return deepcopy(readout)
    def prefix_fn(*a): calls.append('prefix'); return deepcopy(prefix)
    monkeypatch.setattr(waiter.native, 'case_readout', readout_fn)
    monkeypatch.setattr(waiter.native, 'compare', prefix_fn)
    class Physics:
        def __enter__(self): return {'physics_contact':[False]}
        def __exit__(self, *a): pass
    monkeypatch.setattr(waiter.np, 'load', lambda *a, **k:Physics())
    return ids, records, calls


def test_completed_scientific_failure_reconstructs_actual_evidence(tmp_path, monkeypatch):
    ids, records, calls = terminal_fixture(tmp_path, monkeypatch)
    report = waiter.authenticate_completed({'src':'sha'}, ids)
    assert calls == ['inputs','worker','readout','prefix']
    assert report['measured_round_trip_successes'] == 0 and report['scientific_success_required'] is False
    assert report['actual_physical_prefix_reconstructed'] is True


@pytest.mark.parametrize('fault', ['raw', 'flow', 'admission', 'missing_raw', 'worker', 'collection',
    'readout', 'prefix', 'log', 'outcome', 'actual_readout', 'actual_prefix'])
def test_changed_native_evidence_cannot_be_certified(tmp_path, monkeypatch, fault):
    ids, records, calls = terminal_fixture(tmp_path, monkeypatch)
    result = records['result.json']; name = waiter.native.CASE[0]
    if fault == 'raw': result['controller_completion_sha256'] = 'b'*64
    if fault == 'flow': result['flow_wait_result_sha256'] = 'c'*64
    if fault == 'admission': records['launch.json']['input_admission']['controller_completion_sha256'] = 'b'*64
    if fault == 'missing_raw': result['artifact_sha256'].pop(name+'/physics_trace.npz')
    if fault == 'worker': records[name+'_worker_terminal.json']['changed'] = True
    if fault == 'collection': records[name+'/result.json']['changed'] = True
    if fault == 'readout': records[name+'_readout.json']['changed'] = True
    if fault == 'prefix': records[name+'_prefix_comparison.json']['changed'] = True
    if fault == 'log': result['artifact_sha256'][name+'_worker.log'] = 'changed'
    if fault == 'outcome': result['measured_round_trip_successes'] = 1
    if fault == 'actual_readout': monkeypatch.setattr(waiter.native, 'case_readout', lambda *a:{'changed':True})
    if fault == 'actual_prefix': monkeypatch.setattr(waiter.native, 'compare', lambda *a:{'changed':True})
    with pytest.raises(ValueError): waiter.authenticate_completed({'src':'sha'}, ids)


@pytest.mark.parametrize('exit_code', [0, 1])
def test_waiter_launches_one_child_and_never_retries_failure(tmp_path, monkeypatch, exit_code):
    output = tmp_path/'waiter'; child_root = tmp_path/'child'; ids = identities(); calls = []
    monkeypatch.setattr(waiter, 'OUTPUT', output)
    monkeypatch.setattr(waiter.native, 'OUTPUT', child_root)
    monkeypatch.setattr(waiter, 'validate_root', lambda *a, **k:None)
    monkeypatch.setattr(waiter, 'prepared_sources', lambda:{})
    monkeypatch.setattr(waiter, 'verify', lambda *a:None)
    monkeypatch.setattr(waiter, 'verify_artifacts', lambda *a:None)
    monkeypatch.setattr(waiter, 'owner_live', lambda *a:True)
    monkeypatch.setattr(waiter, 'create_output', lambda root:root.mkdir())
    monkeypatch.setattr(waiter.native, 'hardware', lambda:{})
    monkeypatch.setattr(waiter.native, 'cohort_resources', lambda *a:None)
    def wait_for(*a): calls.append('wait'); return ids
    monkeypatch.setattr(waiter, 'wait_for_inputs', wait_for)
    monkeypatch.setattr(waiter.native.inputs, 'owners_ended', lambda:calls.append('ended'))
    monkeypatch.setattr(waiter.native, 'require_native_idle', lambda:calls.append('idle'))
    def completed(*a): calls.append('authenticate'); return {'verified':True}
    monkeypatch.setattr(waiter, 'authenticate_completed', completed)
    class Child:
        pid = 1234
        returncode = exit_code
        def poll(self): return self.returncode
    def popen(command, **kwargs):
        calls.append(('child', command))
        assert kwargs['stdout'].name == str(output/'native_stdout.log')
        return Child()
    monkeypatch.setattr(waiter.subprocess, 'Popen', popen)
    if exit_code:
        with pytest.raises(ValueError, match='no retry'): waiter.main()
        assert (output/'failure.json').is_file() and not (output/'result.json').exists()
        assert calls == ['wait', 'ended', 'idle', ('child', waiter.command_for(ids))]
    else:
        waiter.main()
        assert (output/'result.json').is_file() and not (output/'failure.json').exists()
        assert calls == ['wait', 'ended', 'idle', ('child', waiter.command_for(ids)), 'authenticate']
    assert sum(isinstance(call, tuple) and call[0] == 'child' for call in calls) == 1


def test_another_completed_raw_result_cannot_replace_the_verified_replay(tmp_path,monkeypatch):
    spec,sources,*_=completion_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(waiter,'RAW_RESULT_SHA','b'*64)
    with pytest.raises(ValueError,match='exact verified completed'):
        waiter.completion_identity(spec,sources)
