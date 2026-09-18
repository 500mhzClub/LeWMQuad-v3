from copy import deepcopy
import pytest
from scripts.diagnose_go2_adapter_hold_prefix_v1 import ACTIONS, compact, summarize, canonical


def row(frame, *, required=False, action='hold'):
    values = {'hold': .1, 'left_turn': -.1, 'right_turn': .2}
    selection = dict(action=action, mode='WAYPOINT', prediction=[],
        waypoint_map_xy_m=[1., 0.], goal_body_xy_m=[.2, 0.],
        proposal=dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER', route_cells=[[0, 0], [1, 0]]),
        candidates=[dict(action=a, utility_m=values.get(a, -.2)) for a in ACTIONS],
        surface_checks=[dict(possible_intersection=False) for a in ACTIONS],
        nominal_path_checks=[dict(all_predicted_segments_nominally_clear=a in ('hold', 'left_turn')) for a in ACTIONS],
        phase_allowed_actions=list(ACTIONS))
    return dict(tick=frame, observation_index=frame, pre_sample_index=749+50*frame,
        decision=dict(tick=frame, new_selection=selection, requested_command=[0., 0., 0.], terminal=None,
            mission_receipt=dict(hold_required=required, arrivals=[]), observed_goal_distance_m=3.,
            evidence=dict(current_pose=dict(position_initial_body_m=[.01*frame, 0., 0.]))))


def test_distinguishes_lower_utility_legal_turn_from_higher_utility_vetoed_turn():
    result = compact(row(3))
    assert result['discretionary_hold']
    assert result['raw_feasible_moving_actions'] == ['left_turn']
    assert result['higher_utility_path_vetoed_actions'] == ['right_turn']
    assert result['raw_best_eligible_action'] == 'hold'


@pytest.mark.parametrize('frame,required,terminal', [(0, False, None), (3, True, None), (3, False, 'DONE')])
def test_warmup_settling_and_terminal_holds_are_not_stagnation(frame, required, terminal):
    data = row(frame, required=required); data['decision']['terminal'] = terminal
    assert compact(data)['discretionary_hold'] is False


def test_contiguous_observation_spans_and_end_boundary_count_correctly():
    rows = [compact(row(i, required=i == 5)) for i in range(8)]
    report = summarize(rows)
    assert report['discretionary_hold_observations'] == 4
    assert [(r['first_frame'], r['last_frame'], r['observations']) for r in report['longest_discretionary_hold_runs']] == [(3, 4, 2), (6, 7, 2)]
    assert report['longest_discretionary_hold_runs'][0]['observation_span_s'] == .1
    assert report['hold_with_raw_feasible_movement'] == 4
    assert report['native_command_completion_audited'] is False


def test_recovery_metadata_is_retained_without_reclassifying_raw_gate_evidence():
    data = row(3); data['decision']['new_selection']['residual_anchored_continuation'] = {'eligible_actions': ['right_turn']}
    result = compact(data)
    assert result['applied_feasibility_recoveries'] == ['residual_anchored_continuation']
    assert result['raw_feasible_moving_actions'] == ['left_turn']


def test_no_target_view_holds_do_not_invent_a_target_distance():
    data = [row(i) for i in range(5)]
    for item in data: item['decision']['new_selection'].update(waypoint_map_xy_m=None, goal_body_xy_m=None)
    report = summarize([compact(item) for item in data])
    assert report['longest_discretionary_hold_runs'][0]['target_distance_min_m'] is None


def test_incomplete_candidate_bank_and_out_of_order_prefix_rejected():
    data = row(3); data['decision']['new_selection']['candidates'].pop()
    with pytest.raises(ValueError): compact(data)
    with pytest.raises(ValueError): summarize([compact(row(1))])


def test_canonical_prefix_identity_ignores_dictionary_order_but_detects_changed_receipts():
    data = row(0); reordered = dict(reversed(list(data.items())))
    assert canonical(data) == canonical(reordered)
    changed = deepcopy(data); changed['decision']['observed_goal_distance_m'] += .001
    assert canonical(data) != canonical(changed)
