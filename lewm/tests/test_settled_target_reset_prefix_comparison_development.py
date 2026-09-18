from copy import deepcopy
import pytest
from lewm.settled_target_reset_prefix_comparison_development import compare_current
from lewm.tests.test_settled_boundary_prefix_comparison_development import pair
from lewm.mission_target_waypoint_selection_development import MissionTargetWaypointSelector


def transition():
    old, new = pair()
    for d in (old, new):
        d.update(planner_mode='WAYPOINT', new_selection=None)
    previous = deepcopy(new)
    previous['mission_receipt'].update(frame=9, measured_ns=2_400_000_000)
    old.update(planner_mode='NEW', goal_initial_body_xy_m=[0., 0.])
    old['mission_receipt'].update(phase='RETURN', active_goal_initial_body_xy_m=[0., 0.],
        phase_transition='OUTBOUND_TO_RETURN', arrival_confirmed_this_frame=True, arrivals=[{'frame':10}])
    return old, new, previous


def test_target_setter_retains_mode_for_same_goal_and_resets_on_change():
    selector = MissionTargetWaypointSelector(condition='jepa', variant='full', goal_initial_body_xy_m=[.2, 0.])
    selector.mode='WAYPOINT'; selector.scan_sign=1; selector.scan_index=2; selector.scan_target=.5
    selector.set_goal([.2, 0.])
    assert (selector.mode, selector.scan_sign, selector.scan_index, selector.scan_target)==('WAYPOINT',1,2,.5)
    selector.set_goal([0., 0.])
    assert (selector.mode, selector.scan_sign, selector.scan_index, selector.scan_target)==('NEW',None,0,None)


def test_delayed_target_reset_is_checked_without_mutating_decisions():
    old,new,previous=transition(); before=deepcopy((old,new,previous))
    r=compare_current(old,new,previous_candidate=previous)
    assert r['delayed_target_reset_mode_difference'] and not r['requested_command_changed']
    assert r['mission_behavior_differences']==['phase','active_goal_initial_body_xy_m']
    assert (old,new,previous)==before


def test_ordinary_counter_changes_do_not_allow_mode_changes():
    old,new=pair(); old['planner_mode']=new['planner_mode']='WAYPOINT'
    assert not compare_current(old,new)['delayed_target_reset_mode_difference']
    new['planner_mode']='VIEW_ACQUISITION'
    with pytest.raises(ValueError):compare_current(old,new)


@pytest.mark.parametrize('fault', ['previous','frame','clock','phase','transition','arrival','goal','top_goal',
    'mode','previous_mode','hold','previous_hold','selection','command','previous_command','terminal',
    'map','pose','forecast'])
def test_unrelated_or_unwitnessed_changes_remain_rejected(fault):
    old,new,previous=transition()
    if fault=='previous':previous=None
    elif fault=='frame':previous['mission_receipt']['frame']-=1
    elif fault=='clock':previous['mission_receipt']['measured_ns']-=1
    elif fault=='phase':old['mission_receipt']['phase']='OUTBOUND'
    elif fault=='transition':old['mission_receipt']['phase_transition']=None
    elif fault=='arrival':new['mission_receipt']['arrivals']=[{}]
    elif fault=='goal':previous['mission_receipt']['active_goal_initial_body_xy_m']=[.3,0.]
    elif fault=='top_goal':new['goal_initial_body_xy_m']=[.3,0.]
    elif fault=='mode':old['planner_mode']='VIEW_ACQUISITION'
    elif fault=='previous_mode':previous['planner_mode']='VIEW_ACQUISITION'
    elif fault=='hold':new['mission_receipt']['hold_required']=False
    elif fault=='previous_hold':previous['mission_receipt']['hold_required']=False
    elif fault=='selection':new['new_selection']={}
    elif fault=='command':new['requested_command']=[.2,0.,0.]
    elif fault=='previous_command':previous['requested_command']=[.2,0.,0.]
    elif fault=='terminal':new['terminal']='FAIL'
    elif fault=='map':new['memory_receipt']['cells']+=1
    elif fault=='pose':new['evidence']['pose']+=.1
    else:new['unexpected_prediction']=[.1]
    with pytest.raises(ValueError):compare_current(old,new,previous_candidate=previous)
