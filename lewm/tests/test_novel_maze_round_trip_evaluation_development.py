from copy import deepcopy
import numpy as np
import pytest
from lewm.novel_maze_round_trip_scene_development import START, PITCH_M, graph, evaluator_route, public_mission
from lewm.novel_maze_round_trip_evaluation_development import evaluate, traversed_edges


@pytest.fixture
def native_round_trip():
    route = np.asarray(evaluator_route(graph(0)), float)*PITCH_M
    pieces = [np.tile(route[0], (750, 1))]
    outbound = np.concatenate([np.linspace(a, b, 2601)[1:] for a, b in zip(route, route[1:])])
    pieces += [outbound, np.tile(route[-1], (550, 1))]
    outbound_end = sum(map(len, pieces))-1
    reverse = route[::-1]
    pieces += [np.concatenate([np.linspace(a, b, 2601)[1:] for a, b in zip(reverse, reverse[1:])]),
        np.tile(route[0], (550, 1))]
    return_end = sum(map(len, pieces))-1
    pieces.append(np.tile(route[0], (500, 1)))
    xy = np.concatenate(pieces); n = len(xy)
    poses = np.zeros((n, 7)); poses[:, :2] = xy; poses[:, 2] = .3; poses[:, 6] = 1.
    twists = np.zeros((n, 6)); twists[1:, :2] = np.diff(xy, axis=0)/.002
    requests = np.zeros((n, 3)); requests[np.linalg.norm(twists[:, :2], axis=1) > 0, 0] = .2
    raw = dict(base_pose_world=poses, base_twist_world=twists, requested_command=requests,
        timestamp_s=np.arange(1, n+1)*.002, physics_contact=np.zeros(n, bool))
    mission = public_mission(0); arrivals = []
    for phase, end, target in [('OUTBOUND', outbound_end, mission['goal_initial_body_xy_m']),
            ('RETURN', return_end, mission['return_initial_body_xy_m'])]:
        assert (end-749) % 50 == 0
        frame = (end-749)//50
        arrivals.append(dict(phase=phase, frame=frame, measured_ns=1_500_000_000+frame*100_000_000,
            target_initial_body_xy_m=target, quiet_intervals=10, native_verified=False))
    receipt = dict(arrivals=arrivals, terminal='OBSERVED_ROUND_TRIP_CANDIDATE')
    collection = dict(schedule_terminal=receipt['terminal'], terminal_zero_ticks=10,
        physical_stop=None, acquisition_stop=None)
    return raw, receipt, collection


def test_two_real_native_windows_and_reverse_edges_still_need_full_audit(native_round_trip):
    raw, receipt, collection = native_round_trip
    r = evaluate(raw, receipt, collection, layout_index=0)
    assert r['native_round_trip_candidate_pass'] and r['physically_retraced_outbound_route']
    assert [len(r[k]['crossings']) for k in ('outbound_traversal', 'return_traversal')] == [6, 6]
    assert not r['verified_round_trip'] and r['requires_raw_sensor_command_and_visibility_audit']


@pytest.mark.parametrize('failure', ['outbound_nonzero', 'outbound_speed', 'contact', 'budget_terminal', 'missing_drain'])
def test_terminal_proximity_cannot_override_failed_dwell_or_execution(native_round_trip, failure):
    raw, receipt, collection = native_round_trip
    end = 749+50*receipt['arrivals'][0]['frame']
    if failure == 'outbound_nonzero': raw['requested_command'][end-250, 0] = .2
    elif failure == 'outbound_speed': raw['base_twist_world'][end-250, 0] = .051
    elif failure == 'contact': raw['physics_contact'][1000] = True
    elif failure == 'budget_terminal': collection['schedule_terminal'] = 'MISSION_TICK_BUDGET_EXHAUSTED'
    else: collection['terminal_zero_ticks'] = 9
    r = evaluate(raw, receipt, collection, layout_index=0)
    assert r['terminal_native_quiet_pass'] and not r['native_round_trip_candidate_pass']


def test_teleport_and_closed_wall_crossing_do_not_prove_backtracking():
    start = np.asarray(START)*PITCH_M
    blocked = start+[0., PITCH_M]
    r = traversed_edges(np.linspace(start, blocked, 2601), 0, first_sample=749)
    assert r['native_step_bound_pass'] and len(r['invalid_crossings']) == 1
    route = np.asarray(evaluator_route(graph(0)), float)*PITCH_M
    r = traversed_edges(route, 0, first_sample=749)
    assert not r['native_step_bound_pass'] and not r['invalid_crossings']


def test_bad_clock_arrival_order_or_incomplete_window_rejected(native_round_trip):
    raw, receipt, collection = native_round_trip
    wrong = deepcopy(receipt); wrong['arrivals'].reverse()
    with pytest.raises(ValueError): evaluate(raw, wrong, collection, layout_index=0)
    wrong = deepcopy(receipt); wrong['arrivals'][1]['frame'] = 3999
    wrong['arrivals'][1]['measured_ns'] = 1_500_000_000+3999*100_000_000
    with pytest.raises(ValueError): evaluate(raw, wrong, collection, layout_index=0)
    raw['timestamp_s'][1000] += .001
    with pytest.raises(ValueError): evaluate(raw, receipt, collection, layout_index=0)
