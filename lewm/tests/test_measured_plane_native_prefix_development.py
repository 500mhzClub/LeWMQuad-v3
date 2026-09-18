"""Physical prefix identity and actual execution of a different boundary action."""
from copy import deepcopy
import numpy as np
import pytest
from scripts import measured_plane_native_prefix_development as prefix
from lewm.tests.test_measured_plane_controller_prefix_comparison_development import example
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


def population():
    saved,actual,old_tape,new_tape = [],[],[],[]
    for frame in range(prefix.FRAMES):
        old,new,_,observer = example()
        old['tick'] = new['tick'] = frame
        if frame == prefix.INTERVENTION:
            new['requested_command'] = [0.,0.,-.45]; new['new_selection']['action'] = 'right_turn'
        _,a = endpoint(frame,old); row,b = endpoint(frame,new)
        old_tape.append(a); new_tape.append(b); actual.append(row)
        check = prefix.job.compare(old,new,old,observer,frame=frame)
        saved.append(dict(tick=frame,decision=new,public_packet_sha256='actual'))
    report = dict(frames=123,raw_forecast_comparisons=120,model_state_sha256=prefix.job.MODEL_SHA,
        model_states_unchanged=True,following_changed_command_outcome_consumed=False,
        changed_command_executed=False,native_execution=False,navigation_recovered=False,goal_achieved=False,
        boundary_comparison=check,boundary_original=old,boundary_candidate=new)
    return report,saved,actual,old_tape,new_tape


def setup(tmp_path,monkeypatch):
    report,saved,actual,old_tape,new_tape = population()
    prior,current = tmp_path/'prior',tmp_path/'current'
    for p in (prior,current):
        p.mkdir(); np.savez(p/'physics_trace.npz',position=np.zeros((6900,3)),clock=np.arange(6900))
    monkeypatch.setattr(prefix,'artifact_path',lambda root,name:root/name)
    monkeypatch.setattr(prefix.run,'read_json',lambda root,name:deepcopy(old_tape if root == prior else new_tape))
    monkeypatch.setattr(prefix.run.pipeline,'read_rows',lambda root:iter(actual if root == current else saved))
    # A generator supplies the required close method of the real stream.
    def stream(root): yield from (actual if root == current else saved)
    monkeypatch.setattr(prefix.run.pipeline,'read_rows',stream)
    monkeypatch.setattr(prefix,'public_packets',lambda root:iter(['actual']*123))
    return prior,current,report,actual,new_tape


def test_new_command_executes_after_exact_shared_physics(tmp_path,monkeypatch):
    prior,current,report,_,_ = setup(tmp_path,monkeypatch)
    position=np.zeros((6900,3));position[6850:,0]=.003
    np.savez(current/'physics_trace.npz',position=position,clock=np.arange(6900))
    result=prefix.compare(prior,current,report)
    assert result['physical_prefix_samples'] == 6850
    assert result['candidate_intervention_command_completed'] and result['intervention_command_changed']
    assert not result['following_physical_outcomes_compared'] and not result['navigation_verified']


@pytest.mark.parametrize('fault',['physics','earlier_command','boundary_command','incomplete','short_physics',
    'decision','public','missing','overclaim'])
def test_divergence_or_unexecuted_command_cannot_pass(tmp_path,monkeypatch,fault):
    prior,current,report,actual,tape = setup(tmp_path,monkeypatch)
    if fault == 'physics':
        position=np.zeros((6900,3));position[6849,0]=.003
        np.savez(current/'physics_trace.npz',position=position,clock=np.arange(6900))
    elif fault == 'earlier_command': tape[121]['requested_command']=[.2,0.,0.]
    elif fault == 'boundary_command': tape[122]['requested_command']=[0.,0.,0.]
    elif fault == 'incomplete': tape[122]['completed']=False
    elif fault == 'short_physics': np.savez(current/'physics_trace.npz',position=np.zeros((6899,3)))
    elif fault == 'decision': actual[50]['decision']=dict(actual[50]['decision'],failure='changed')
    elif fault == 'public': monkeypatch.setattr(prefix,'public_packets',lambda root:iter([str(root)]*123))
    elif fault == 'missing': actual.pop()
    else: report['navigation_recovered']=True
    with pytest.raises(ValueError): prefix.compare(prior,current,report)
