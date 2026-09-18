from copy import deepcopy
from types import SimpleNamespace
import math
import numpy as np
import pytest
from lewm.tests.test_hold_reorientation_development import selection
from lewm.sustained_hold_reorientation_development import SustainedHoldReorientation, MAX_TURN_COMMANDS
from lewm.sustained_hold_reorientation_controller_development import (
    SustainedHoldReorientationSelector, SustainedHoldReorientationController)
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationSelector
from lewm.hold_reorientation_controller_development import HoldReorientationController


def bank(frame, **changes):
    s = selection(frame)
    s['candidates'][4]['utility_m'] = -.05
    p = np.asarray(s['prediction']); p[:,:,3] = 1.
    for i,direction in ((4,1),(5,-1)):
        angle = direction*np.arange(1,9)*.04
        p[i,:,2] = np.sin(angle); p[i,:,3] = np.cos(angle)
    s['prediction'] = p.tolist()
    return s | changes


def choose(state, frame, s=None, yaw=0.):
    return state.reconsider(bank(frame) if s is None else s, frame=frame,
        now_ns=1_500_000_000+frame*100_000_000,
        observed_heading_map=[math.cos(yaw),math.sin(yaw)])


def active():
    state = SustainedHoldReorientation()
    for f in range(3,13): assert choose(state,f)['action']=='hold'
    first = choose(state,13)
    assert first['action']=='left_turn'
    assert first['sustained_hold_reorientation']['commands']==1
    return state


def test_continuation_is_fresh_lower_ranked_turn_and_preserves_original_evidence():
    state = active(); original = bank(14); before = deepcopy(original)
    r = choose(state,14,original,yaw=.02)
    assert r['action']=='left_turn' and original==before
    receipt = r.pop('sustained_hold_reorientation')
    assert receipt['observed_turn_progress_rad']==pytest.approx(.02)
    assert receipt['target_observed_turn_rad']==pytest.approx(.32)
    assert receipt['commands']==2 and not receipt['starting']
    for k in ('action','action_index','requested_command'):r[k]=original[k]
    assert r==original


def test_eight_command_cap_does_not_infer_progress_from_command_count():
    state=active()
    for f in range(14,21):
        r=choose(state,f,yaw=0.)
        assert r['action']=='left_turn'
        assert r['sustained_hold_reorientation']['observed_turn_progress_rad']==0.
    assert state.active['commands']==MAX_TURN_COMMANDS
    assert choose(state,21)['action']=='hold' and state.active is None
    assert choose(state,22)['action']=='hold'


def test_measured_target_stops_early_and_reverse_motion_does_not_count_as_progress():
    state=active()
    assert choose(state,14,yaw=-.1)['action']=='left_turn'
    assert choose(state,15,yaw=.321)['action']=='hold' and state.active is None


def test_right_turn_uses_signed_observation_and_original_predicted_target():
    state=SustainedHoldReorientation()
    for f in range(3,14):
        s=bank(f);s['candidates'][4]['utility_m']=-.2;r=choose(state,f,s)
    assert r['action']=='right_turn'
    s=bank(14);s['candidates'][4]['utility_m']=-.2
    assert choose(state,14,s,yaw=.4)['action']=='right_turn'
    s=bank(15);s['candidates'][4]['utility_m']=-.2
    assert choose(state,15,s,yaw=-.33)['action']=='hold'


@pytest.mark.parametrize('gate',['phase','surface','path'])
def test_new_veto_immediately_cancels_continuation_without_switching_direction(gate):
    state=active();s=bank(14)
    if gate=='phase':s['phase_allowed_actions'].remove('left_turn')
    if gate=='surface':s['surface_checks'][4]['possible_intersection']=True
    if gate=='path':s['nominal_path_checks'][4]['all_predicted_segments_nominally_clear']=False
    assert choose(state,14,s) is s and state.active is None


@pytest.mark.parametrize('changes',[dict(action='forward'),dict(action=None),dict(mode='VIEW_ACQUISITION'),
    dict(intermediate_target_is_mission_goal=True),dict(nominal_clearance_reentry=True),
    dict(residual_anchored_continuation={}),dict(view_budget_exhausted=True)])
def test_current_original_nonhold_or_special_case_cancels_recovery(changes):
    state=active();s=bank(14,**changes)
    assert choose(state,14,s) is s and state.active is None


def test_gap_and_goal_change_discard_recovery_without_resetting_global_clock():
    state=active();assert choose(state,15)['action']=='hold' and state.active is None
    state=active();state.reset_goal();assert state.last_frame==13
    assert choose(state,14)['action']=='hold' and state.active is None


@pytest.mark.parametrize('value',[[0.,0.],[float('nan'),1.],[1.,0.,0.]])
def test_invalid_observed_heading_cannot_drive_recovery(value):
    with pytest.raises(ValueError,match='heading'):
        active().reconsider(bank(14),frame=14,now_ns=2_900_000_000,observed_heading_map=value)


def test_wrong_clock_stale_score_and_false_hold_optimum_rejected_during_continuation():
    with pytest.raises(ValueError,match='clock'):choose(active(),13)
    s=bank(14);s['causal_score_residual_receipt']['frame']=13
    with pytest.raises(ValueError,match='current original'):choose(active(),14,s)
    s=bank(14);s['candidates'][1]['utility_m']=.1
    with pytest.raises(ValueError,match='hold must win'):choose(active(),14,s)


def test_opposite_predicted_target_preserves_original_single_turn():
    state=SustainedHoldReorientation()
    for f in range(3,13):choose(state,f)
    s=bank(13);s['prediction'][4][-1][2]=-.1
    r=choose(state,13,s)
    assert r['action']=='left_turn' and 'hold_reorientation' in r
    assert 'sustained_hold_reorientation' not in r and state.active is None


def test_selector_uses_current_observed_map_heading_and_entire_original_chain(monkeypatch):
    selector=SustainedHoldReorientationSelector(residual=SimpleNamespace(frame=14),
        condition='jepa',variant='full',goal_initial_body_xy_m=[3.,2.])
    selector.hold_reorientation=active();calls=[]
    def original(self,model,history,mapper,geometry,*,now_ns):
        calls.append((model,history,geometry,now_ns));return bank(14)
    monkeypatch.setattr(ResidualAnchoredContinuationSelector,'choose',original)
    angle=.33;rotation=np.array([[math.cos(angle),-math.sin(angle),0.],[math.sin(angle),math.cos(angle),0.],[0.,0.,1.]])
    mapper=SimpleNamespace(failed=False,map_from_initial=np.eye(3),surface=SimpleNamespace(
        failed=False,last_ns=2_900_000_000,route=[None]*15,rotation=rotation))
    assert selector.choose('model','history',mapper,'geometry',now_ns=2_900_000_000)['action']=='hold'
    assert calls==[('model','history','geometry',2_900_000_000)]
    mapper.surface.failed=True
    with pytest.raises(ValueError,match='same admitted'):
        selector.choose('model','history',mapper,'geometry',now_ns=2_900_000_000)
    assert SustainedHoldReorientationController.observe is HoldReorientationController.observe
    assert SustainedHoldReorientationController.advance is HoldReorientationController.advance


def test_same_goal_retains_recovery_new_goal_clears_only_recovery_state():
    residual=object();selector=SustainedHoldReorientationSelector(residual=residual,
        condition='jepa',variant='full',goal_initial_body_xy_m=[3.,2.])
    selector.hold_reorientation=active();selector.set_goal([3.,2.]);assert selector.hold_reorientation.active
    selector.set_goal([0.,0.]);assert selector.hold_reorientation.active is None
    assert selector.residual is residual and selector.hold_reorientation.last_frame==13
