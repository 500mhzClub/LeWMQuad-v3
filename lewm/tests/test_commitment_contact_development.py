from copy import deepcopy
import pytest
from lewm.tests.test_executed_waypoint_score_development import fixture
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.commitment_contact_score_development import score_commitment_contact
from lewm.commitment_contact_prefix_development import PrefixComparison


def selection():
    s, receipt = fixture()
    s['prediction'][1][-1][4] = 10.
    return score_waypoint_execution(s, receipt)


def test_actual_commitment_contact_changes_ranking_without_changing_forecasts_or_constraints():
    old = selection(); saved = deepcopy(old); new = score_commitment_contact(old)
    assert old['action'] == 'hold' and new['action'] == 'forward' and old == saved
    assert new['scored_contact_horizon_ns'] == new['scored_pose_horizon_ns'] == 100_000_000
    assert new['path_constraint_horizon_ns'] == 800_000_000
    assert new['candidates'][1]['commitment_contact_score'] < .00005
    assert new['candidates'][1]['full_plan_contact_score'] > .9999
    for key in ('prediction', 'nominal_path_checks', 'surface_checks', 'phase_allowed_actions',
            'causal_score_residual_receipt', 'original_waypoint_candidates'):
        assert new[key] == old[key]
    assert new['original_full_contact_candidates'] == old['candidates']
    assert not new['contact_scores_calibrated']


@pytest.mark.parametrize('veto', ['later_path', 'surface', 'phase', 'first_contact', 'all_paths'])
def test_horizon_change_cannot_bypass_geometric_or_immediate_contact_cost(veto):
    s, receipt = fixture()
    if veto == 'later_path': s['nominal_path_checks'][1]['all_predicted_segments_nominally_clear'] = False
    elif veto == 'surface': s['surface_checks'][1]['possible_intersection'] = True
    elif veto == 'phase': s['phase_allowed_actions'] = ['hold', 'left_turn', 'right_turn']
    elif veto == 'first_contact':
        for step in s['prediction'][1]: step[4] = 10.
    else:
        for path in s['nominal_path_checks']: path['all_predicted_segments_nominally_clear'] = False
    result = score_commitment_contact(score_waypoint_execution(s, receipt))
    assert result['action'] != 'forward'
    if veto == 'all_paths': assert result['action'] is None and result['requested_command'] == [0., 0., 0.]


def test_pass_through_scopes_and_original_arithmetic_when_contacts_are_equal():
    s, receipt = fixture(); old = score_waypoint_execution(s, receipt)
    new = score_commitment_contact(old)
    assert [c['utility_m'] for c in new['candidates']] == [c['utility_m'] for c in old['candidates']]
    assert new['action'] == old['action']
    for patch in ({'intermediate_target_is_mission_goal': True}, {'mode': 'VIEW_ACQUISITION'},
            {'nominal_clearance_reentry': True}):
        value = old | patch
        assert score_commitment_contact(value) is value
    assert score_commitment_contact(None) is None
    assert score_commitment_contact({}) == {}


@pytest.mark.parametrize('fault', ['future', 'native', 'nonfinite', 'decreasing_contact', 'score', 'horizon'])
def test_malformed_forecasts_receipts_and_inherited_scores_fail_closed(fault):
    old = selection()
    if fault == 'future': old['causal_score_residual_receipt']['residual_source_ticks'] = [100]
    elif fault == 'native': old['causal_score_residual_receipt']['native_outcomes_used'] = True
    elif fault == 'nonfinite': old['prediction'][0][0][0] = float('nan')
    elif fault == 'decreasing_contact': old['prediction'][1][0][4] = 11.
    elif fault == 'score': old['candidates'][0]['utility_m'] += .1
    else: old['scored_contact_horizon_ns'] = 100_000_000
    with pytest.raises(ValueError): score_commitment_contact(old)


def decisions(frame=0):
    old = dict(tick=frame, controller='measured_floor_transport_round_trip_controller_v1',
        model_condition='supervised_rollout', input_variant='full', memory_variant='persistent',
        new_selection=selection(), selected_action='hold', requested_command=[0., 0., 0.],
        terminal=None, failure=None, evidence={'observed': frame}, causal_residual_receipt={'pending_forecast_tick':frame})
    new = deepcopy(old); new.update(controller='commitment_contact_round_trip_controller_v1',
        commitment_contact_policy_enabled=True, new_selection=score_commitment_contact(old['new_selection']),
        selected_action='forward', requested_command=[.22, 0., 0.])
    from lewm.geometry_progress_pilot_development import candidate_commands
    new['requested_command'] = candidate_commands('forward')[0]
    return old, new


@pytest.mark.parametrize('fault', [None, 'evidence', 'residual', 'prediction', 'constraint', 'wrong_command', 'order', 'objective'])
def test_whole_decision_comparison_rejects_unassigned_changes_and_latches_stop(fault):
    old, new = decisions(); comp = PrefixComparison()
    if fault == 'evidence': new['evidence']['observed'] = -1
    elif fault == 'residual': new['causal_residual_receipt']['pending_forecast_tick'] = -1
    elif fault == 'prediction': new['new_selection']['prediction'][0][0][0] += .001
    elif fault == 'constraint': new['new_selection']['surface_checks'][0]['possible_intersection'] = True
    elif fault == 'wrong_command': new['requested_command'] = [1., 0., 0.]
    elif fault == 'objective': new['model_condition'] = 'jepa'
    if fault is not None:
        with pytest.raises(ValueError): comp.compare(old, new, old['requested_command'], frame=1 if fault == 'order' else 0)
    else:
        result = comp.compare(old, new, old['requested_command'], frame=0)
        assert result['stop'] and result['requested_command_changed'] and comp.first_command_difference == 0
        with pytest.raises(ValueError): comp.compare(*decisions(1), old['requested_command'], frame=1)


def test_warmup_exact_and_either_terminal_stops():
    old, new = decisions()
    for decision in (old, new):
        decision.update(new_selection=None, selected_action=None, requested_command=[0., 0., 0.])
    comp = PrefixComparison()
    assert not comp.compare(old, new, old['requested_command'], frame=0)['stop']
    for decision in (old, new): decision.update(tick=1, terminal='MISSION_TICK_BUDGET_EXHAUSTED')
    assert comp.compare(old, new, old['requested_command'], frame=1)['stop']


def test_original_observer_mission_memory_execution_and_failure_latch_inherited():
    from lewm.commitment_contact_controller_development import CommitmentContactController
    from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
    assert CommitmentContactController.observe is MeasuredFloorTransportController.observe
    assert CommitmentContactController.advance is MeasuredFloorTransportController.advance
    args = dict(public_mission=dict(goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.],
        require_return_after_goal=True), navigation_ticks=40, condition='supervised_rollout', variant='full', persistent=True)
    original = MeasuredFloorTransportController(object(), object(), **args)
    candidate = CommitmentContactController(object(), object(), **args)
    for key in ('motion', 'mission', 'registration', 'mapper', 'memory', 'residual'):
        assert type(getattr(candidate, key)) is type(getattr(original, key))
    assert candidate.residual.snapshot() == original.residual.snapshot()
    result = candidate.observe({}, {}, {}, now_ns=1)
    assert result['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and result['requested_command'] == [0., 0., 0.]
    before = candidate.residual.snapshot()
    assert candidate.advance({}, {}, now_ns=2)['terminal'] == result['terminal']
    assert candidate.residual.snapshot() == before
