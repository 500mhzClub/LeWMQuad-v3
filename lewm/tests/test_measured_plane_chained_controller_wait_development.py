"""Exact predecessor polling, resource waiting and completed replay evidence."""
from copy import deepcopy

import pytest

from scripts import await_go2_measured_plane_chained_controller_prefix_v1 as waiter


@pytest.mark.parametrize('transient',[False,True])
def test_predecessor_is_polled_until_ended_without_inventing_completion(monkeypatch,tmp_path,transient):
    states=iter(([PermissionError('temporary')] if transient else [])+[True,False])
    owners=[];events=[];reads=[];sleeps=[]
    monkeypatch.setattr(waiter.job.timing_waiter,'OUTPUT',tmp_path)
    def live(owner):
        owners.append(deepcopy(owner));value=next(states)
        if isinstance(value,Exception):raise value
        return value
    monkeypatch.setattr(waiter.run,'owner_live',live)
    monkeypatch.setattr(waiter.run,'verify_artifacts',lambda *args:None)
    def digest(path):
        assert len(owners)==(3 if transient else 2)
        reads.append(path);return 'sha'
    monkeypatch.setattr(waiter.run,'digest',digest)
    monkeypatch.setattr(waiter.time,'sleep',sleeps.append)
    assert waiter.wait_for_predecessor(lambda status,**kw:events.append(status))==('sha','sha')
    assert owners==[waiter.job.CPU_OWNER]*len(owners) and len(reads)==2
    assert sleeps==[30]*(2 if transient else 1)
    assert events==(['CPU_OWNER_OBSERVATION_RETRY'] if transient else [])+['EXACT_RESERVED_TIMING_OWNER_LIVE']


def test_ended_failed_predecessor_does_not_dispatch(monkeypatch,tmp_path):
    (tmp_path/'failure.json').write_text('{}')
    monkeypatch.setattr(waiter.job.timing_waiter,'OUTPUT',tmp_path)
    monkeypatch.setattr(waiter.run,'owner_live',lambda owner:False)
    monkeypatch.setattr(waiter.run,'verify_artifacts',lambda *args:None)
    def forbidden(path):pytest.fail('failed predecessor result accessed')
    monkeypatch.setattr(waiter.run,'digest',forbidden)
    with pytest.raises(ValueError,match='preserve preceding'):waiter.wait_for_predecessor(lambda *args:None)


def test_resource_wait_rechecks_both_thresholds_and_transient_errors(monkeypatch):
    gib=1024**3
    observations=iter([OSError('temporary'),dict(memory_available_bytes=63*gib,artifact_free_bytes=44*gib),
        dict(memory_available_bytes=65*gib,artifact_free_bytes=42*gib),
        dict(memory_available_bytes=64*gib,artifact_free_bytes=43*gib)])
    events=[];sleeps=[]
    def hardware():
        value=next(observations)
        if isinstance(value,Exception):raise value
        return value
    monkeypatch.setattr(waiter.run,'hardware',hardware);monkeypatch.setattr(waiter.time,'sleep',sleeps.append)
    result=waiter.wait_for_resources(lambda status,**kw:events.append(status))
    assert result==dict(memory_available_bytes=64*gib,artifact_free_bytes=43*gib)
    assert events==['RESOURCE_OBSERVATION_RETRY','PAIRED_CONTROLLER_RESOURCE_WAIT','PAIRED_CONTROLLER_RESOURCE_WAIT']
    assert sleeps==[30,30,30]


@pytest.mark.parametrize('fault',[None,'owner','source','roster','report','scope','input','boundary'])
def test_complete_causal_child_and_reconstruction_required(monkeypatch,tmp_path,fault):
    monkeypatch.setattr(waiter.job,'OUTPUT',tmp_path)
    sources={'source':'frozen'};admission=dict(frames=3124,learned_result_sha256='native')
    report=dict(frames=5,boundary_comparison=dict(stop=True,stop_reason='FIRST_CHANGED_REQUEST_OR_TERMINAL'),
        actual_model_forward_calls=[2,2])
    launch=dict(owner={'pid':1},source_sha256=sources,input_admission=admission,
        boot_id=waiter.run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        stop_at_first_changed_command_or_terminal=True,automatic_retry=False)
    ids=dict.fromkeys(('launch.json','context_decisions.jsonl.gz','resource_monitor.jsonl','report.json'),'hash')
    result=dict(status='MEASURED_PLANE_CHAINED_CONTROLLER_PREFIX_V1_COMPLETE',source_sha256=sources,
        complete_output_and_consumed_public_packets_rechecked=True,
        original_raw_inputs_reauthenticated_before_and_after=True,
        native_execution=False,navigation_qualified=False,goal_achieved=False,artifact_sha256=ids,report=report)
    documents={'launch.json':launch,'result.json':result,'report.json':deepcopy(report)}
    if fault=='source':result['source_sha256']={'changed':'source'}
    elif fault=='roster':ids.pop('context_decisions.jsonl.gz')
    elif fault=='report':documents['report.json']['frames']=4
    elif fault=='scope':result['navigation_qualified']=True
    elif fault=='input':launch['input_admission']={'frames':4}
    elif fault=='boundary':launch['stop_at_first_changed_command_or_terminal']=False
    monkeypatch.setattr(waiter.run,'owner_live',lambda owner:fault=='owner')
    monkeypatch.setattr(waiter.run,'read_json',lambda root,name:deepcopy(documents[name]))
    monkeypatch.setattr(waiter.run,'digest',lambda path:'result-sha')
    monkeypatch.setattr(waiter.run,'verify_artifacts',lambda *args:None)
    monkeypatch.setattr(waiter.run,'verify',lambda *args:None)
    monkeypatch.setattr(waiter.job,'admit',lambda native,cpu,bindings:deepcopy(admission))
    checked=[]
    def check(value,admitted):
        assert value==report and admitted==admission;checked.append(True)
    monkeypatch.setattr(waiter.job,'check_output',check)
    if fault is not None:
        with pytest.raises(ValueError):waiter.completed_child(sources,'native','cpu')
    else:
        proof=waiter.completed_child(sources,'native','cpu')
        assert checked==[True] and proof['complete_consumed_rows_and_public_packets_reconstructed']
        assert not proof['controller_replay_reexecuted'] and not proof['changed_command_executed']
        assert proof['frames']==5
