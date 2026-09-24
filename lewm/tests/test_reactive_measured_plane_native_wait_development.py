"""Wait for the same owner and distinguish early negatives from intervention."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import await_go2_reactive_measured_plane_native_v1 as waiter
from lewm.tests.test_reactive_measured_plane_native_launcher_development import fixture


def test_live_owner_is_observed_until_end_before_result_read(monkeypatch, tmp_path):
    states = iter([True, True, False]); observations, reads, sleeps, events = [], [], [], []
    monkeypatch.setattr(waiter.native.inputs.nominal_wait, 'OUTPUT', tmp_path)
    def live(owner):
        assert owner == waiter.native.inputs.NOMINAL_WAIT_OWNER
        value = next(states); observations.append(value); return value
    def digest(path):
        assert observations[-1] is False
        reads.append(path.name); return 'completed'
    monkeypatch.setattr(waiter.run, 'owner_live', live)
    checks = []
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda root, ids: checks.append(ids))
    monkeypatch.setattr(waiter.run, 'digest', digest)
    monkeypatch.setattr(waiter.time, 'sleep', sleeps.append)
    assert waiter.wait_for_nominal(lambda status, **kw: events.append(status)) == 'completed'
    assert sleeps == [30, 30] and reads == ['result.json']
    assert checks == [{'launch.json': waiter.native.inputs.NOMINAL_WAIT_LAUNCH_SHA}]*3
    assert events == ['EXACT_NOMINAL_WAITER_LIVE']*2


def test_transient_owner_observation_retries_same_owner_without_early_result(monkeypatch, tmp_path):
    states = iter([PermissionError('transient observation'), True, False]); owners, sleeps, events = [], [], []
    monkeypatch.setattr(waiter.native.inputs.nominal_wait, 'OUTPUT', tmp_path)
    def live(owner):
        owners.append(deepcopy(owner)); value = next(states)
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr(waiter.run, 'owner_live', live)
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda *args: None)
    def digest(path):
        assert len(owners) == 3; return 'completed'
    monkeypatch.setattr(waiter.run, 'digest', digest)
    monkeypatch.setattr(waiter.time, 'sleep', sleeps.append)
    assert waiter.wait_for_nominal(lambda status, **kw: events.append(status)) == 'completed'
    assert owners == [waiter.native.inputs.NOMINAL_WAIT_OWNER]*3 and sleeps == [30, 30]
    assert events == ['NOMINAL_OWNER_OBSERVATION_RETRY', 'EXACT_NOMINAL_WAITER_LIVE']


def test_ended_failed_owner_is_not_admitted(monkeypatch, tmp_path):
    (tmp_path/'failure.json').write_text('{}')
    monkeypatch.setattr(waiter.native.inputs.nominal_wait, 'OUTPUT', tmp_path)
    monkeypatch.setattr(waiter.run, 'owner_live', lambda owner: False)
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda *args: None)
    def digest(path): raise AssertionError('failed attempt must not read completion')
    monkeypatch.setattr(waiter.run, 'digest', digest)
    with pytest.raises(ValueError, match='preserve failed nominal'): waiter.wait_for_nominal(lambda *args: None)


@pytest.mark.parametrize('fault', [None, 'roster', 'monitor', 'launch', 'readout', 'prefix', 'model', 'parent_live'])
def test_complete_child_requires_raw_receipts_and_labels_early_negative_honestly(tmp_path, monkeypatch, fault):
    native = waiter.native; name = native.CASE[0]; record, audit = fixture()
    directory = tmp_path/name; directory.mkdir()
    np.savez(directory/'physics_trace.npz', physics_contact=np.zeros(950, dtype=int))
    monkeypatch.setattr(native, 'OUTPUT', tmp_path)
    sources = {'source': 'frozen'}
    admission = dict(nominal_wait_result_sha256='nominal-wait', learned_result_sha256='learned',
        nominal_result_sha256='nominal', prefix_report={})
    launch = dict(owner={'pid': 1}, source_sha256=sources, input_admission=admission,
        boot_id=waiter.run.Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    ids = {name+s: 'hash' for s in ('_worker_terminal.json', '_worker.log', '_audit.json', '_prefix_comparison.json', '_readout.json')}
    ids[name+'/result.json'] = 'hash'; ids['launch.json'] = 'hash'; ids['resource_monitor.jsonl'] = 'hash'
    record.update(artifact_sha256={name+'/result.json': 'hash'}, worker_log_sha256='hash', readout={'evaluator_only': True})
    result = dict(status='REACTIVE_MEASURED_PLANE_MAZE02_V1_COMPLETE', source_sha256=sources,
        nominal_wait_result_sha256='nominal-wait', learned_result_sha256='learned', nominal_result_sha256='nominal',
        reactive_prefix_result_sha256=native.prefix.RESULT_SHA, conditions=[record], automatic_retry=False,
        high_level_world_model_loaded=False, fully_nonpredictive_controller=True,
        reactive_is_whole_method_comparison=True, isolated_prediction_ranking_ablation=False,
        measured_round_trip_successes=0, artifact_sha256=ids)
    documents = {'launch.json': launch, 'result.json': result, name+'_worker_terminal.json': record,
        name+'_audit.json': audit, name+'_readout.json': record['readout'],
        name+'_prefix_comparison.json': record['prefix_comparison']}
    if fault == 'roster': del ids[name+'_audit.json']
    elif fault == 'monitor': del ids['resource_monitor.jsonl']
    elif fault == 'launch': ids['launch.json'] = 'changed'
    elif fault == 'readout': documents[name+'_readout.json'] = {'changed': True}
    elif fault == 'prefix': documents[name+'_prefix_comparison.json'] = {'changed': True}
    elif fault == 'model': result['high_level_world_model_loaded'] = True
    monkeypatch.setattr(waiter.run, 'read_json', lambda root, key:
        deepcopy(record['collection'] if root == directory and key == 'result.json' else documents[key]))
    monkeypatch.setattr(waiter.run, 'owner_live', lambda owner: fault == 'parent_live')
    monkeypatch.setattr(waiter.run, 'digest', lambda path: 'hash' if path.name == 'launch.json' else 'result-sha')
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda *args: None)
    monkeypatch.setattr(waiter.run, 'verify', lambda *args: None)
    monkeypatch.setattr(native, 'verify_inputs', lambda launch: None)
    monkeypatch.setattr(native.pipeline, 'artifacts', lambda *args: ['result.json'])
    monkeypatch.setattr(native.original, 'case_readout', lambda *args: {'evaluator_only': True})
    if fault is not None:
        with pytest.raises(ValueError): waiter.completed_child(sources, 'nominal-wait')
    else:
        report = waiter.completed_child(sources, 'nominal-wait')
        assert report['complete_native_worker_and_artifact_roster_verified']
        assert report['physical_prefix_accounting_reconstructed']
        assert not report['actual_physical_prefix_reconstructed']
        assert not report['scientific_success_required']
        assert report['reactive_measured_round_trip_successes'] == 0
