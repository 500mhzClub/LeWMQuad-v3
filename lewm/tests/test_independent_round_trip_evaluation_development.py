"""Synthetic native-shaped traces; no controller or simulator execution."""
from copy import deepcopy
import inspect
import ast

import numpy as np
import pytest

from lewm.independent_round_trip_layouts_development import specification, public_mission
from lewm.independent_round_trip_evaluation_development import evaluate, traversed_edges
from lewm.novel_maze_round_trip_scene_development import PITCH_M


def trace(index):
    route = np.asarray(specification(index)['evaluation_layout']['shortest_outbound_route'], float)*PITCH_M
    pieces = [np.tile(route[0], (750, 1))]
    pieces += [np.concatenate([np.linspace(a, b, 2601)[1:] for a, b in zip(route, route[1:])]),
        np.tile(route[-1], (550, 1))]
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
    mission = public_mission(index); arrivals = []
    for phase, end, key in [('OUTBOUND', outbound_end, 'goal_initial_body_xy_m'),
            ('RETURN', return_end, 'return_initial_body_xy_m')]:
        assert (end-749) % 50 == 0
        frame = (end-749)//50
        arrivals.append(dict(phase=phase, frame=frame, measured_ns=1_500_000_000+frame*100_000_000,
            target_initial_body_xy_m=mission[key], quiet_intervals=10, native_verified=False))
    receipt = dict(arrivals=arrivals, terminal='OBSERVED_ROUND_TRIP_CANDIDATE')
    collection = dict(schedule_terminal=receipt['terminal'], terminal_zero_ticks=10,
        physical_stop=None, acquisition_stop=None)
    return raw, receipt, collection


@pytest.mark.parametrize('index', range(8))
def test_all_eight_complete_routes_and_dwell_require_separate_raw_audit(index):
    result = evaluate(*trace(index), layout_index=index)
    edges = len(specification(index)['evaluation_layout']['shortest_outbound_route'])-1
    assert result['native_round_trip_candidate_pass'] and result['physically_retraced_outbound_route']
    assert [len(result[k]['crossings']) for k in ('outbound_traversal', 'return_traversal')] == [edges, edges]
    assert not result['verified_round_trip'] and not result['independent_navigation_claim']
    assert result['requires_raw_sensor_command_and_visibility_audit']


@pytest.mark.parametrize('failure', ['contact', 'dwell_command', 'dwell_speed', 'drain',
    'physical_stop', 'acquisition_stop', 'budget_terminal', 'no_arrivals', 'goal_only'])
def test_terminal_home_proximity_does_not_override_failure(failure):
    raw, receipt, collection = trace(7)
    end = 749+50*receipt['arrivals'][0]['frame']
    if failure == 'contact': raw['physics_contact'][1000] = True
    elif failure == 'dwell_command': raw['requested_command'][end-250, 0] = .2
    elif failure == 'dwell_speed': raw['base_twist_world'][end-250, 0] = .051
    elif failure == 'drain': collection['terminal_zero_ticks'] = 9
    elif failure in ('physical_stop', 'acquisition_stop'): collection[failure] = 'STOP'
    elif failure == 'budget_terminal': collection['schedule_terminal'] = 'MISSION_TICK_BUDGET_EXHAUSTED'
    elif failure == 'no_arrivals': receipt['arrivals'] = []
    else: receipt['arrivals'] = receipt['arrivals'][:1]
    result = evaluate(raw, receipt, collection, layout_index=7)
    assert result['terminal_native_quiet_pass'] and not result['native_round_trip_candidate_pass']


@pytest.mark.parametrize('failure', ['clock', 'order', 'target', 'window', 'layout'])
def test_wrong_clock_or_arrival_identity_is_rejected(failure):
    raw, receipt, collection = trace(7); index = 7
    if failure == 'clock': raw['timestamp_s'][1000] += .001
    elif failure == 'order': receipt['arrivals'].reverse()
    elif failure == 'target': receipt['arrivals'][0]['target_initial_body_xy_m'][0] += .1
    elif failure == 'window': receipt['arrivals'][0]['frame'] = 9
    else: index = 8
    with pytest.raises(ValueError): evaluate(raw, receipt, collection, layout_index=index)


def test_graph_binding_rejects_closed_edges_and_teleports():
    spec = specification(7)
    start = np.asarray(spec['geometry']['spawn_se2_world'][:2])
    result = traversed_edges(np.linspace(start, start+[0., PITCH_M], 2601), 7, first_sample=749)
    assert result['native_step_bound_pass'] and len(result['invalid_crossings']) == 1
    route = np.asarray(spec['evaluation_layout']['shortest_outbound_route'])*PITCH_M
    result = traversed_edges(route, 7, first_sample=749)
    assert not result['native_step_bound_pass'] and not result['invalid_crossings']
    # Every new graph is distinct: its complete edge set must not be accepted
    # under a different inventory index, even if their goal coordinates agree.
    for wrong in range(7):
        rejected = []
        for a, b in spec['evaluation_layout']['edges']:
            xy = np.linspace(np.asarray(a)*PITCH_M, np.asarray(b)*PITCH_M, 2601)
            rejected.extend(traversed_edges(xy, wrong, first_sample=749)['invalid_crossings'])
        assert rejected


def test_numerical_success_and_rejection_contract_is_unchanged():
    from lewm import independent_round_trip_evaluation_development as new
    from lewm import novel_maze_round_trip_evaluation_development as old
    for name in ('evaluate', 'traversed_edges', 'loop_erased'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(new, name)))) == ast.dump(
            ast.parse(inspect.getsource(getattr(old, name))))
