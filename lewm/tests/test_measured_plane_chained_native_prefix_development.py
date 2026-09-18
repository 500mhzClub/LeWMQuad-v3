"""Dynamic actual intervention, complete prefixes and post-boundary isolation."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import measured_plane_chained_native_prefix_development as prefix
from lewm.tests.test_measured_plane_chained_controller_prefix_development import decision,candidate
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


def fixture_report(intervention=4,terminal_only=False):
    old=decision(intervention);new=candidate(old)
    calls=[1,1]
    if terminal_only:
        old.update(terminal='SENSOR_OR_MODEL_FAILURE',failure='original failure',new_selection=None)
        calls[0]=0
    else:
        new['new_selection']['action']='forward';new['requested_command']=[.2,0.,0.]
    check=prefix.replay.comparison.compare(old,new,old,frame=intervention,maximum_frames=3124,model_calls=calls)
    return prefix.replay.report_for(intervention+1,[intervention-2,intervention-2],
        intervention-2,None,check,old,new)


@pytest.mark.parametrize('intervention',[3,17,122,3113])
def test_boundary_is_derived_from_actual_report(intervention):
    report=fixture_report(intervention);bound=prefix.boundary(report)
    assert bound==dict(frames=intervention+1,intervention=intervention,
        physics_samples=750+50*intervention,command_changed=True,terminal_changed=False)


@pytest.mark.parametrize('fault',['frames','model','native','candidate_failure','no_change','report_changed'])
def test_invalid_or_nonadmitted_boundary_cannot_authorize_prefix(fault):
    report=fixture_report()
    if fault=='frames':report['frames']=True
    elif fault=='model':report['model_state_sha256']='different'
    elif fault=='native':report['native_execution']=True
    elif fault=='candidate_failure':
        report['boundary_candidate'].update(terminal='SENSOR_OR_MODEL_FAILURE',failure='candidate failure',
            requested_command=[0.,0.,0.],new_selection=None)
    elif fault=='no_change':
        report['boundary_candidate']['requested_command']=[0.,0.,0.]
        report['boundary_candidate']['new_selection']['action']='hold'
    else:report['boundary_comparison']['requested_command_changed']=False
    with pytest.raises(ValueError):prefix.boundary(report)


def setup(tmp_path,monkeypatch,intervention=4,terminal_only=False):
    prior=tmp_path/'prior';current=tmp_path/'current';saved=tmp_path/'saved'
    for p in (prior,current,saved):p.mkdir()
    report=fixture_report(intervention,terminal_only);count=intervention+1;samples=750+50*intervention
    tapes=[[],[]];old_rows=[];new_rows=[];replay_rows=[]
    for frame in range(count):
        old=decision(frame);new=candidate(old)
        if frame==intervention:old,new=deepcopy(report['boundary_original']),deepcopy(report['boundary_candidate'])
        old_row,old_cmd=endpoint(frame,old);new_row,new_cmd=endpoint(frame,new)
        old_rows.append(old_row);new_rows.append(new_row);tapes[0].append(old_cmd);tapes[1].append(new_cmd)
        calls=[int(bool(row['new_selection'])) for row in (old,new)]
        check=prefix.replay.comparison.compare(old,new,old,frame=frame,maximum_frames=3124,model_calls=calls)
        replay_rows.append(dict(tick=frame,original=old,decision=new,comparison=check,
            actual_model_forward_calls=calls,public_packet_sha256=f'packet-{frame}'))
    for path,tape in zip((prior,current),tapes,strict=True):
        raw=dict(timestamp_s=np.arange(samples+100)*.002,requested_command=np.zeros((samples+100,3)),
            arbitrary_physical_channel=np.zeros((samples+100,7)))
        raw['requested_command'][samples:samples+50]=tape[-1]['requested_command']
        # Outcomes after the actual boundary interval need not match, and are
        # excluded from the common-prefix fingerprint.
        raw['arbitrary_physical_channel'][samples:]=0 if path==prior else 1
        np.savez(path/'physics_trace.npz',**raw)
        (path/'command_tape.json').write_text(json.dumps(tape))
    monkeypatch.setattr(prefix.replay,'OUTPUT',saved)
    monkeypatch.setattr(prefix,'artifact_path',lambda root,name:root/name)
    monkeypatch.setattr(prefix.run,'read_json',lambda root,name:json.loads((root/name).read_text()))
    documents={prior:deepcopy(old_rows),current:deepcopy(new_rows),saved:deepcopy(replay_rows)};consumed=[]
    def rows(root):
        for row in documents[root]:yield deepcopy(row)
        if root!=saved:pytest.fail('following actual controller observation consumed')
    def packets(root,frames):
        assert frames==count
        for frame in range(frames):consumed.append((root,frame));yield f'packet-{frame}'
    monkeypatch.setattr(prefix.run.pipeline,'read_rows',rows)
    monkeypatch.setattr(prefix,'public_packets',packets)
    return prior,current,saved,report,documents,consumed


@pytest.mark.parametrize('intervention',[3,17,122])
def test_full_physical_public_and_controller_prefix_matches_at_dynamic_boundary(tmp_path,monkeypatch,intervention):
    prior,current,_,report,_,consumed=setup(tmp_path,monkeypatch,intervention)
    result=prefix.compare(prior,current,report)
    assert result['first_changed_decision_frame']==intervention
    assert result['common_prefix_frames']==intervention+1
    assert result['physical_prefix_samples']==750+50*intervention
    assert result['candidate_intervention_command_completed'] is True
    assert result['intervention_command_changed'] is True
    assert result['following_physical_outcomes_compared'] is False
    assert result['navigation_verified'] is False
    assert consumed==[(path,frame) for frame in range(intervention+1) for path in (prior,current)]


def test_terminal_only_change_is_not_misreported_as_command_change(tmp_path,monkeypatch):
    prior,current,_,report,_,_=setup(tmp_path,monkeypatch,terminal_only=True)
    result=prefix.compare(prior,current,report)
    assert result['intervention_terminal_changed'] is True
    assert result['intervention_command_changed'] is False
    assert result['original_intervention_command']==result['candidate_intervention_command']==[0.,0.,0.]


@pytest.mark.parametrize('counts,reason',[
    ((3,3,3,900),'INTERVENTION_OBSERVATION_NOT_REACHED'),
    ((5,4,4,950),'INTERVENTION_COMMAND_NOT_ISSUED'),
    ((5,5,4,970),'INTERVENTION_COMMAND_INCOMPLETE'),
    ((5,5,5,970),'INSUFFICIENT_PHYSICAL_SAMPLES'),
    ((5,5,5,1000),None)])
def test_early_negative_or_count_metadata_never_claims_reconstructed_prefix(counts,reason):
    collection=dict(zip(('decisions','command_ticks','completed_ticks','physics_samples'),counts,strict=True))
    receipt=prefix.prefix_availability(collection,fixture_report())
    assert receipt['unavailable_reason']==reason
    assert receipt['boundary_interval_available_by_collection_counts']==(reason is None)
    assert receipt['full_raw_prefix_compare_still_required']==(reason is None)
    assert receipt['actual_physical_prefix_reconstructed'] is False
    assert receipt['complete_candidate_decisions_reconstructed'] is False


@pytest.mark.parametrize('counts',[(True,0,0,750),(5,4,5,1000),(6,4,4,1000)])
def test_invalid_collection_accounting_rejected(counts):
    collection=dict(zip(('decisions','command_ticks','completed_ticks','physics_samples'),counts,strict=True))
    with pytest.raises(ValueError):prefix.prefix_availability(collection,fixture_report())


@pytest.mark.parametrize('fault',['physics','boundary_physics','early_command','incomplete_command','baseline_decision',
    'candidate_decision','packet','clock','extra_saved_row','missing_saved_row','early_stop'])
def test_corrupted_or_incomplete_physical_prefix_is_rejected(tmp_path,monkeypatch,fault):
    prior,current,saved,report,documents,_=setup(tmp_path,monkeypatch)
    samples=prefix.boundary(report)['physics_samples']
    if fault in ('physics','boundary_physics'):
        path=current/'physics_trace.npz'
        with np.load(path,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
        if fault=='physics':raw['arbitrary_physical_channel'][samples-1,0]=1
        else:raw['requested_command'][samples+49,0]=.1
        np.savez(path,**raw)
    elif fault in ('early_command','incomplete_command'):
        path=current/'command_tape.json';tape=json.loads(path.read_text())
        if fault=='early_command':tape[3]['requested_command']=[.2,0.,0.]
        else:tape[-1]['completed']=False
        path.write_text(json.dumps(tape))
    elif fault=='baseline_decision':documents[prior][3]['decision']['evidence']['floor']='corrupted'
    elif fault=='candidate_decision':documents[current][3]['decision']['evidence']['floor']='corrupted'
    elif fault=='packet':documents[saved][3]['public_packet_sha256']='changed'
    elif fault=='clock':documents[current][3]['pre_sample_index']+=1
    elif fault=='extra_saved_row':documents[saved].append(deepcopy(documents[saved][-1])|dict(tick=5))
    elif fault=='missing_saved_row':documents[saved].pop()
    else:documents[saved][3]['comparison']['stop']=True
    with pytest.raises((ValueError,AssertionError)):prefix.compare(prior,current,report)
