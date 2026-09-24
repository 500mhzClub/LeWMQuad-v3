"""Selection responds to predicted motion/contact without reading native labels."""
import numpy as np
import pytest
import torch
from lewm.geometry_progress_predictive_selection_development import score_candidates,candidate_inputs
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands


def predictions():
    p=np.zeros((6,8,5),np.float32);p[:,:,3]=1.;p[:,:,4]=-10.
    p[1,:,0]=.4;p[2:4,:,0]=.3;p[2,:,1]=.1;p[3,:,1]=-.1
    return p


def score(p,goal=(1.2,0.),cost=1.2):
    return score_candidates(p,goal_body_xy_m=goal,contact_penalty_m=cost)


def test_mirrored_contact_forecasts_reverse_choice_with_same_goal_and_candidate_bank():
    p=predictions();p[1,:,4]=10.;p[3,:,4]=10.
    left=score(p);p[2,:,4]=10.;p[3,:,4]=-10.;right=score(p)
    assert left['action']=='left_arc' and right['action']=='right_arc'
    assert left['requested_command']==candidate_commands('left_arc')[0]
    assert len(left['candidates'])==6 and not left['contact_probability_calibrated']


def test_zero_contact_cost_changes_ranking_and_goal_behind_prefers_hold():
    p=predictions();p[1,:,4]=10.
    assert score(p)['action']=='left_arc' and score(p,cost=0.)['action']=='forward'
    assert score(p,goal=(-1.2,0.))['action']=='hold'


def test_ties_deterministically_hold_and_extreme_finite_logits_are_stable():
    p=np.zeros((6,8,5),np.float64);p[:,:,4]=1000.
    assert score(p)['action']=='hold'
    p[:,:,4]=-1000.;assert score(p)['candidates'][0]['predicted_contact_score']==0.


@pytest.mark.parametrize('fault',['missing_candidate','nan','noncumulative','bad_goal','negative_cost'])
def test_incomplete_or_invalid_forecasts_fail_closed(fault):
    p=predictions();kwargs={}
    if fault=='missing_candidate':p=p[:-1]
    if fault=='nan':p[0,0,0]=np.nan
    if fault=='noncumulative':p[0,-1,4]=-11.
    if fault=='bad_goal':kwargs['goal']=(np.inf,0.)
    if fault=='negative_cost':kwargs['cost']=-1.
    with pytest.raises(ValueError):score(p,**kwargs)


def test_all_candidates_get_the_same_history_and_no_target_channel():
    history={k:torch.zeros(s) for k,s in {'rgb':(4,3,96,128),'body':(4,20,63),'control':(4,15,7)}.items()}
    inputs=candidate_inputs(history)
    for k,v in inputs['observation_history'].items():
        assert torch.equal(v[0],history[k]) and torch.equal(v[0],v[5])
    assert inputs['known_action_blocks'].shape==(6,8,5,3)
    assert inputs['known_action_valid'].all()
    with pytest.raises(ValueError):candidate_inputs(history|{'future_images':history['rgb']})


def test_real_model_inference_has_exact_forecasts_and_does_not_train():
    from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA
    from lewm.geometry_progress_predictive_selection_development import select
    from lewm.pulse_timed_training_runner_development import state_digest
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026090921);model=CumulativePulseRGBBodyJEPA(32).eval()
    history={k:torch.zeros(s) for k,s in {'rgb':(4,3,96,128),'body':(4,20,63),'control':(4,15,7)}.items()}
    before=state_digest(model.state_dict())
    r=select(model,history,head='rollout_outcomes',input_variant='full',goal_body_xy_m=(1.2,0.),contact_penalty_m=1.2)
    assert r['action'] in ACTIONS and np.asarray(r['prediction']).shape==(6,8,5)
    assert r['selection_wall_ms']>=0. and state_digest(model.state_dict())==before
    assert all(p.grad is None for p in model.parameters()) and not model.training
    assert not r['model_checkpoint_admitted_by_this_helper']
