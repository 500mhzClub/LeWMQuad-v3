from copy import deepcopy
import pytest
from lewm.settled_boundary_prefix_comparison_development import compare_current


def pair():
    mission=dict(frame=10,measured_ns=2_500_000_000,phase='OUTBOUND',active_goal_initial_body_xy_m=[.2,0.],
        hold_required=True,terminal=None,quiet_intervals=4,phase_transition=None,
        arrival_confirmed_this_frame=False,arrivals=[])
    old=dict(controller='later_floor_resolution_round_trip_controller_v1',terminal=None,failure=None,
        quiet_intervals=4,goal_initial_body_xy_m=[.2,0.],mission_receipt=mission,requested_command=[0.,0.,0.],
        evidence={'pose':0.},memory_receipt={'cells':10},new_selection={'prediction':[1.,2.]})
    new=deepcopy(old);new.update(controller='settled_boundary_round_trip_controller_v1',
        measured_settling_required_for_arrival=True,measured_quiet_boundary_required_before_dwell=True)
    new['mission_receipt'].update(measured_settling_required=True,observed_settling=dict(current_frame=10,
        measured_ns=2_500_000_000,continuous_speed_bound=False,native_state_used=False,
        first_quiet_observation_starts_dwell=True))
    return old,new


def test_only_counter_difference_is_allowed_without_behavior_change():
    old,new=pair();new['quiet_intervals']=2;new['mission_receipt']['quiet_intervals']=2
    before=deepcopy(new);r=compare_current(old,new)
    assert r['quiet_counter_changed'] and not r['mission_behavior_differences'] and new==before


@pytest.mark.parametrize('field',['evidence','memory_receipt','new_selection','requested_command'])
def test_any_undeclared_observer_map_model_or_command_change_fails(field):
    old,new=pair();new[field]=None
    with pytest.raises(ValueError,match='outside declared'):compare_current(old,new)


def test_earlier_candidate_arrival_is_rejected():
    old,new=pair();new['mission_receipt']['arrivals']=[{}]
    with pytest.raises(ValueError,match='earlier arrival'):compare_current(old,new)


def test_original_return_transition_can_be_delayed_without_altering_actual_command():
    old,new=pair();old['mission_receipt'].update(phase='RETURN',active_goal_initial_body_xy_m=[0.,0.],
        phase_transition='OUTBOUND_TO_RETURN',arrivals=[{'frame':10}],arrival_confirmed_this_frame=True)
    old['goal_initial_body_xy_m']=[0.,0.]
    r=compare_current(old,new)
    assert r['mission_behavior_differences']==['phase','active_goal_initial_body_xy_m']
    assert not r['requested_command_changed']
