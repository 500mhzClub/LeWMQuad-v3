"""No premature native completion or undersized CPU dispatch; exact final proof."""
from copy import deepcopy

import pytest

from scripts import await_go2_measured_plane_full_history_timing_v1 as waiter
from lewm.tests.test_measured_plane_full_history_timing_development import population


def test_native_is_observed_until_ended_before_reading_result(monkeypatch, tmp_path):
    states = iter([True, True, False]); observed, reads, checks, sleeps = [], [], [], []
    monkeypatch.setattr(waiter.job.native, 'OUTPUT', tmp_path)
    def live(owner):
        assert owner == waiter.job.inputs.LEARNED_OWNER
        value = next(states); observed.append(value); return value
    def digest(path):
        assert observed[-1] is False; reads.append(path.name); return 'native-result'
    monkeypatch.setattr(waiter.run, 'owner_live', live)
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda root, ids: checks.append(ids))
    monkeypatch.setattr(waiter.run, 'digest', digest); monkeypatch.setattr(waiter.time, 'sleep', sleeps.append)
    assert waiter.wait_for_native(lambda *args, **kw: None) == 'native-result'
    assert reads == ['result.json'] and sleeps == [30, 30]
    assert checks == [{'launch.json': waiter.job.inputs.LEARNED_LAUNCH_SHA}]*3


def test_transient_native_observation_preserves_same_owner_and_does_not_infer_end(monkeypatch, tmp_path):
    states = iter([PermissionError('temporary'), True, False]); owners, events = [], []
    monkeypatch.setattr(waiter.job.native, 'OUTPUT', tmp_path)
    def live(owner):
        owners.append(deepcopy(owner)); value = next(states)
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr(waiter.run, 'owner_live', live)
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda *args: None)
    def digest(path):
        assert len(owners) == 3; return 'native-result'
    monkeypatch.setattr(waiter.run, 'digest', digest); monkeypatch.setattr(waiter.time, 'sleep', lambda seconds: None)
    assert waiter.wait_for_native(lambda status, **kw: events.append(status)) == 'native-result'
    assert owners == [waiter.job.inputs.LEARNED_OWNER]*3
    assert events == ['NATIVE_OWNER_OBSERVATION_RETRY', 'EXACT_LEARNED_NATIVE_OWNER_LIVE']


def test_ended_native_execution_failure_does_not_dispatch(monkeypatch, tmp_path):
    (tmp_path/'failure.json').write_text('{}')
    monkeypatch.setattr(waiter.job.native, 'OUTPUT', tmp_path)
    monkeypatch.setattr(waiter.run, 'owner_live', lambda owner: False)
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda *args: None)
    def digest(path): raise AssertionError('failed native result must not be read')
    monkeypatch.setattr(waiter.run, 'digest', digest)
    with pytest.raises(ValueError, match='preserve failed native'): waiter.wait_for_native(lambda *args: None)


def test_resource_wait_reobserves_errors_and_both_reserve_thresholds(monkeypatch):
    gib = 1024**3
    observations = iter([OSError('temporary'), dict(memory_available_bytes=63*gib, artifact_free_bytes=44*gib),
        dict(memory_available_bytes=65*gib, artifact_free_bytes=42*gib),
        dict(memory_available_bytes=64*gib, artifact_free_bytes=43*gib)])
    events, sleeps = [], []
    def hardware():
        value = next(observations)
        if isinstance(value, Exception): raise value
        return value
    monkeypatch.setattr(waiter.run, 'hardware', hardware); monkeypatch.setattr(waiter.time, 'sleep', sleeps.append)
    result = waiter.wait_for_resources(lambda status, **kw: events.append(status))
    assert result == dict(memory_available_bytes=64*gib, artifact_free_bytes=43*gib)
    assert events == ['RESOURCE_OBSERVATION_RETRY', 'FULL_REPLAY_RESOURCE_WAIT', 'FULL_REPLAY_RESOURCE_WAIT']
    assert sleeps == [30, 30, 30]


@pytest.mark.parametrize('fault', [None, 'owner', 'source', 'roster', 'report', 'population', 'scope', 'input'])
def test_ended_replay_requires_complete_population_bindings_and_reconstructed_output(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(waiter.job, 'OUTPUT', tmp_path)
    sources = {'source': 'frozen'}; rows, states = population(4)
    report = waiter.job.comparison.summarize(rows, states, frames=4, model_sha='model', input_result_sha='native')
    admission = dict(frames=4, learned_result_sha256='native')
    launch = dict(owner={'pid': 1}, source_sha256=sources, input_admission=admission,
        boot_id=waiter.run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        state_frames=waiter.job.comparison.state_frames(4), actual_complete_population_required=True,
        automatic_retry=False)
    ids = dict.fromkeys(('launch.json', 'comparison.jsonl', 'state_checks.json', 'resource_monitor.jsonl', 'report.json'), 'hash')
    result = dict(status='MEASURED_PLANE_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE', source_sha256=sources,
        complete_output_and_public_packets_rechecked=True, original_raw_inputs_reauthenticated_before_and_after=True,
        native_execution=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False,
        artifact_sha256=ids, report=report)
    documents = {'launch.json': launch, 'result.json': result, 'report.json': deepcopy(report)}
    if fault == 'source': result['source_sha256'] = {'changed': 'source'}
    elif fault == 'roster': del ids['state_checks.json']
    elif fault == 'report': documents['report.json']['frames'] = 3
    elif fault == 'population': launch['state_frames'] = [0]
    elif fault == 'scope': result['real_time_qualified'] = True
    elif fault == 'input': launch['input_admission'] = dict(frames=4, learned_result_sha256='changed')
    monkeypatch.setattr(waiter.run, 'owner_live', lambda owner: fault == 'owner')
    monkeypatch.setattr(waiter.run, 'read_json', lambda root, name: deepcopy(documents[name]))
    monkeypatch.setattr(waiter.run, 'digest', lambda path: 'result-sha')
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda *args: None)
    monkeypatch.setattr(waiter.run, 'verify', lambda *args: None)
    monkeypatch.setattr(waiter.job, 'admit', lambda sha, bindings: deepcopy(admission))
    checked = []
    def check(value, admitted):
        assert value == report and admitted == admission; checked.append(True)
    monkeypatch.setattr(waiter.job, 'check_output', check)
    if fault is not None:
        with pytest.raises(ValueError): waiter.completed_child(sources, 'native')
    else:
        proof = waiter.completed_child(sources, 'native')
        assert checked == [True]
        assert proof['complete_rows_states_and_public_packets_reconstructed']
        assert proof['all_observation_time_reduction_percent'] == 50.
        assert not proof['controller_replay_reexecuted'] and not proof['real_time_qualified']
