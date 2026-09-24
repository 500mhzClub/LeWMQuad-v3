from copy import deepcopy
import pytest
from lewm.observed_round_trip_mission_development import ObservedRoundTripMission


def mission(budget=40, required=True):
    return ObservedRoundTripMission(dict(goal_initial_body_xy_m=[.2, 0.],
        return_initial_body_xy_m=[0., 0.], require_return_after_goal=required), navigation_ticks=budget)


def step(m, tick, x, previous=(0., 0., 0.)):
    return m.advance([x, 0.], frame=tick, now_ns=1_500_000_000+tick*100_000_000,
        previous_requested_command=previous)


def test_two_dwell_states_and_transition_zero_require_actual_return_observations():
    m = mission(budget=22)
    for i in range(3): assert step(m, i, 0.)['hold_required']
    assert step(m, 3, .2, (.2, 0., 0.))['quiet_intervals'] == 0
    for i in range(4, 14): r = step(m, i, .2)
    assert r['phase_transition'] == 'OUTBOUND_TO_RETURN' and r['hold_required']
    assert r['active_goal_initial_body_xy_m'] == [0., 0.] and r['quiet_intervals'] == 0
    assert len(r['arrivals']) == 1 and r['terminal'] is None
    assert not r['observed_map_reset_required'] and not r['controller_state_reset_required']
    assert not step(m, 14, .2)['hold_required']
    assert step(m, 15, 0., (.2, 0., 0.))['quiet_intervals'] == 0
    for i in range(16, 26): r = step(m, i, 0.)
    assert r['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE' and len(r['arrivals']) == 2
    assert not r['verified_round_trip']
    assert [a['frame'] for a in r['arrivals']] == [13, 25]


def test_outbound_arrival_does_not_reset_global_budget_or_finish_round_trip():
    m = mission(budget=10)
    for i in range(3): step(m, i, 0.)
    for i in range(3, 14): r = step(m, i, .2)
    assert r['phase'] == 'RETURN' and len(r['arrivals']) == 1
    assert r['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED' and r['hold_required']


def test_leaving_four_centimetres_or_nonzero_previous_request_resets_dwell():
    m = mission(required=False)
    for i in range(3): step(m, i, 0.)
    for i in range(3, 8): step(m, i, .2)
    assert step(m, 8, .245)['quiet_intervals'] == 0
    assert step(m, 9, .2)['quiet_intervals'] == 0
    assert step(m, 10, .2, (.2, 0., 0.))['quiet_intervals'] == 0
    for i in range(11, 21): r = step(m, i, .2)
    assert r['terminal'] == 'OBSERVED_GOAL_CANDIDATE' and len(r['arrivals']) == 1


def test_bad_clock_latches_hold_and_no_future_frame_resumes_mission():
    m = mission(); step(m, 0, 0.)
    r = step(m, 2, .2)
    assert r['terminal'] == 'SENSOR_OR_MISSION_FAILURE' and r['hold_required']
    assert step(m, 3, .2) == r
    r['arrivals'].append({'invented': True})
    assert not step(m, 4, .2)['arrivals']


def test_no_topology_payload_alias_or_implicit_budget():
    payload = dict(goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True)
    m = ObservedRoundTripMission(payload, navigation_ticks=40); payload['goal_initial_body_xy_m'][0] = 4.
    assert m.target().tolist() == [.2, 0.]
    for value in (True, 0, 4001, 1.5):
        with pytest.raises(ValueError): ObservedRoundTripMission(payload, navigation_ticks=value)
    bad = deepcopy(payload); bad['wall_boxes'] = []
    with pytest.raises(ValueError): ObservedRoundTripMission(bad, navigation_ticks=40)


def test_tighter_observed_target_requires_closer_pose_before_dwell():
    payload=dict(goal_initial_body_xy_m=[.2,0.],return_initial_body_xy_m=[0.,0.],require_return_after_goal=False)
    m=ObservedRoundTripMission(payload,navigation_ticks=60,arrival_radius_m=.02)
    for frame in range(20):row=step(m,frame,.17)
    assert not row['arrivals'] and row['observed_arrival_radius_m']==.02
    for frame in range(20,31):row=step(m,frame,.19)
    assert row['terminal']=='OBSERVED_GOAL_CANDIDATE'
