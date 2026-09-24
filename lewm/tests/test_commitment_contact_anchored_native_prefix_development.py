"""Complete four-observation physical intervention, with no later equivalence claim."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_executed_waypoint_score_development import fixture as score_fixture
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.commitment_contact_anchored_controller_development import ordinary_commitment_contact, CONTROLLER, FLAG
from lewm.commitment_contact_anchored_prefix_development import PrefixComparison
from scripts import commitment_contact_anchored_native_prefix_development as prefix


def fixture():
    s, receipt = score_fixture()
    s['prediction'][1][-1][4] = 10.
    for p in s['prediction'][4]: p[0] = .003
    receipt.update(frame=3, measured_ns=1_800_000_000)
    original_selection = score_waypoint_execution(s, receipt)
    candidate_selection = ordinary_commitment_contact(original_selection)
    assert original_selection['action'] == 'left_turn' and candidate_selection['action'] == 'forward'
    originals = []; prospective = []; tape = []; comparator = PrefixComparison()
    for frame in range(4):
        old = dict(tick=frame, controller='residual_anchored_continuation_controller_v1',
            model_condition='supervised_rollout', input_variant='full', memory_variant='persistent',
            terminal=None, failure=None, new_selection=None if frame<3 else deepcopy(original_selection),
            selected_action=None if frame<3 else original_selection['action'],
            requested_command=[0.,0.,0.] if frame<3 else original_selection['requested_command'],
            evidence=dict(xy=[.2,.3]), mission_receipt=dict(arrivals=[]))
        new = deepcopy(old); new.update(controller=CONTROLLER, **{FLAG:True})
        if frame == 3:
            new.update(new_selection=deepcopy(candidate_selection), selected_action=candidate_selection['action'],
                requested_command=candidate_selection['requested_command'])
        check = comparator.compare(old,new,old['requested_command'],frame=frame)
        originals.append(dict(tick=frame, observation_index=frame, pre_sample_index=749+50*frame, decision=old))
        prospective.append(dict(tick=frame, decision=new, comparison=check,
            original_requested_command=old['requested_command'], original_complete_decision_reconstructed=True,
            public_input_arrays_unchanged=True, complete_retained_observed_state_equal=True,
            selected_pending_forecasts_checked=True, public_input_sha256='packet_'+str(frame)))
        tape.append(dict(tick=frame, completed=True, requested_command=old['requested_command'],
            pre_sample_index=749+50*frame, post_sample_index=799+50*frame))
    report = dict(frames=4, first_changed_command_frame=3, raw_model_forecast_comparisons=1,
        original_requested_command=[0.,0.,.45], candidate_requested_command=[.2,0.,0.])
    return report, originals, prospective, tape, candidate_selection


def test_complete_saved_comparisons_reconstruct_the_fixed_boundary():
    result = prefix.reconstruct(*fixture())
    assert result == dict(frames=4, first_intervention_frame=3, raw_model_forecast_comparisons=1,
        complete_saved_comparisons_reconstructed=True)


@pytest.mark.parametrize('fault', ['truncated','extra','counter','bool_counter','boundary','command',
    'comparison','incomplete','observed_state','forecast','expected_selection','pending_unchecked'])
def test_incomplete_or_changed_raw_prefix_is_rejected(fault):
    r,old,saved,tape,expected = fixture(); expected = deepcopy(expected)
    if fault == 'truncated': saved.pop()
    elif fault == 'extra': saved.append(deepcopy(saved[-1]))
    elif fault == 'counter': r['raw_model_forecast_comparisons'] = 0
    elif fault == 'bool_counter': r['raw_model_forecast_comparisons'] = True
    elif fault == 'boundary': r['first_changed_command_frame'] = 2
    elif fault == 'command': tape[3]['requested_command'] = [0.,0.,0.]
    elif fault == 'comparison': saved[3]['comparison']['stop'] = False
    elif fault == 'incomplete': tape[3]['completed'] = False
    elif fault == 'observed_state': saved[3]['decision']['evidence']['xy'][0] += 1
    elif fault == 'forecast': saved[3]['decision']['new_selection']['prediction'][0][0][0] += 1
    elif fault == 'expected_selection': expected['requested_command'] = [0.,0.,-.45]
    else: saved[3]['selected_pending_forecasts_checked'] = False
    with pytest.raises(ValueError): prefix.reconstruct(r,old,saved,tape,expected)


@pytest.mark.parametrize('fault', ['incomplete','post','pre','earlier','boundary','short'])
def test_all_prior_commands_and_changed_command_must_complete(fault):
    r,_,_,old,_ = fixture(); new = deepcopy(old); new[-1]['requested_command'] = r['candidate_requested_command']
    if fault == 'incomplete': new[-1]['completed'] = False
    elif fault == 'post': new[-1]['post_sample_index'] -= 1
    elif fault == 'pre': new[-1]['pre_sample_index'] -= 1
    elif fault == 'earlier': new[2]['requested_command'] = [0.,0.,.45]
    elif fault == 'boundary': new[-1]['requested_command'] = [0.,0.,.45]
    else: new.pop()
    with pytest.raises(ValueError): prefix.executed_boundary([old,new],r)


def physical_fixture(monkeypatch,tmp_path,fault=None):
    r,old,saved,tape,expected = fixture()
    prior=tmp_path/'prior';current=tmp_path/'current';root=tmp_path/'prefix'
    prior.mkdir();current.mkdir()
    actual=deepcopy(old)
    for i,row in enumerate(actual): row['decision']=deepcopy(saved[i]['decision'])
    new_tape=deepcopy(tape);new_tape[-1]['requested_command']=r['candidate_requested_command']
    raw=np.arange(950,dtype=np.float64);other=raw.copy();other[900:]+=100
    if fault=='physics':other[899]+=1
    np.savez(prior/'physics_trace.npz',sample=raw)
    np.savez(current/'physics_trace.npz',sample=other)
    if fault=='decision':actual[-1]['decision']['evidence']['xy'][0]+=1
    if fault=='completion':new_tape[-1]['completed']=False
    if fault=='public':saved[-1]['public_input_sha256']='changed'
    monkeypatch.setattr(prefix,'artifact_path',lambda parent,name:parent/name)
    monkeypatch.setattr(prefix,'read_rows',lambda p:iter(deepcopy(old if p==prior else actual if p==current else saved)))
    monkeypatch.setattr(prefix,'read_json',lambda p,n:deepcopy(tape if p==prior else new_tape))
    monkeypatch.setattr(prefix,'public_packets',lambda p,n:iter('packet_'+str(i) for i in range(n)))
    monkeypatch.setattr(prefix.replay,'saved_inputs',lambda:({}, {'expected_candidate_selection':expected}, []))
    return prior,current,root,r


def test_physical_comparison_covers_900_samples_and_completed_forward_command(monkeypatch,tmp_path):
    result=prefix.compare(*physical_fixture(monkeypatch,tmp_path))
    assert result['physical_prefix_samples']==900 and result['common_prefix_frames']==4
    assert result['candidate_intervention_command_completed']
    assert result['following_physical_outcomes_compared'] is False


@pytest.mark.parametrize('fault',['physics','decision','completion','public'])
def test_changed_physical_or_public_prefix_is_rejected(monkeypatch,tmp_path,fault):
    with pytest.raises(ValueError):prefix.compare(*physical_fixture(monkeypatch,tmp_path,fault))
