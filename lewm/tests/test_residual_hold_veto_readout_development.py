"""Saved-score reconstruction and the limits of changing only one future point."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_residual_hold_feasibility_development import fixture
from lewm.residual_hold_veto_readout_development import classify_hold, summarize_hold_vetoes
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan


@pytest.mark.parametrize('kind', ['first_only', 'late', 'surface', 'utility'])
def test_original_geometry_and_utility_distinguish_hold_causes(kind):
    selection, receipt, mapper = fixture(late_collision=kind == 'late', expensive=kind == 'utility')
    if kind == 'surface':
        for row in selection['surface_checks'][1:]: row['possible_intersection'] = True
    before = deepcopy(selection); result = classify_hold(selection, frame=receipt['frame'])
    expected = {'first_only':'at_least_one_better_nonhold_blocked_only_in_first_two_segments',
        'late':'every_better_nonhold_has_a_first_point_invariant_veto',
        'surface':'every_better_nonhold_has_a_first_point_invariant_veto',
        'utility':'no_strictly_better_allowed_nonhold'}
    assert result['reason'] == expected[kind] and selection == before
    assert result['corrected_path_feasibility_not_recomputed']
    assert result['alternative_physical_outcomes_not_observed']


def test_first_point_correction_cannot_change_original_later_segment_receipts():
    selection, receipt, mapper = fixture(late_collision=True)
    corrected = deepcopy(selection); p = np.asarray(corrected['prediction'])
    p[:, 0, :2] -= receipt['correction_xy_m']; corrected['prediction'] = p.tolist()
    position = mapper.surface.position; rotation = np.eye(3)
    corrected = plan(constrain(corrected, position, rotation, mapper.occupied), position, rotation, mapper.occupied)
    for old, new in zip(selection['nominal_path_checks'], corrected['nominal_path_checks'], strict=True):
        assert old['segments'][2:] == new['segments'][2:]
    report = classify_hold(selection, frame=receipt['frame'])
    assert all(a['unchanged_suffix_veto'] for a in report['better_allowed_nonholds'])


@pytest.mark.parametrize('fault', ['score', 'progress', 'contact', 'bias', 'future', 'path_summary',
    'radius', 'segment_order', 'short_path', 'hold_veto', 'wrong_frame', 'zero_yaw'])
def test_forged_original_scoring_or_veto_evidence_rejected(fault):
    selection, receipt, mapper = fixture(); frame = receipt['frame']
    if fault == 'score': selection['candidates'][0]['utility_m'] += .001
    elif fault == 'progress': selection['candidates'][1]['executed_waypoint_distance_progress_m'] += .001
    elif fault == 'contact': selection['candidates'][1]['full_plan_contact_score'] += .001
    elif fault == 'bias': selection['causal_score_residual_receipt']['correction_xy_m'][0] += .001
    elif fault == 'future': selection['causal_score_residual_receipt']['residuals'][-1]['available_tick'] = frame+1
    elif fault == 'path_summary': selection['nominal_path_checks'][1]['all_predicted_segments_nominally_clear'] = True
    elif fault == 'radius': selection['nominal_path_checks'][1]['segments'][0]['radius_m'] = .4
    elif fault == 'segment_order': selection['nominal_path_checks'][1]['segments'].reverse()
    elif fault == 'short_path': selection['nominal_path_checks'][1]['segments'].pop()
    elif fault == 'hold_veto': selection['surface_checks'][0]['possible_intersection'] = True
    elif fault == 'wrong_frame': frame += 1
    elif fault == 'zero_yaw': selection['prediction'][0][0][2:4] = [0., 0.]
    with pytest.raises(ValueError): classify_hold(selection, frame=frame)


@pytest.mark.parametrize('fault', [None, 'truncated', 'order', 'command', 'incomplete_command'])
def test_complete_population_and_actual_commands_required(fault):
    selection, receipt, mapper = fixture(); frame = receipt['frame']
    rows = [dict(tick=i, observation_index=i, pre_sample_index=749+50*i,
        decision=dict(new_selection=selection if i == frame else None,
            requested_command=[0., 0., 0.], terminal='BUDGET_EXHAUSTED' if i == frame+1 else None))
        for i in range(frame+2)]
    tape = [dict(requested_command=[0., 0., 0.], completed=True) for _ in range(frame+1)]
    if fault == 'truncated': rows.pop()
    elif fault == 'order': rows[-1]['tick'] += 1
    elif fault == 'command': tape[frame]['requested_command'] = [1., 0., 0.]
    elif fault == 'incomplete_command': tape[frame]['completed'] = False
    if fault is None:
        result = summarize_hold_vetoes(iter(rows), tape)
        assert result['selected_holds'] == 1 and result['observations'] == frame+2
        assert sum(result['hold_reason_counts'].values()) == 1 and result['every_hold_utility_reconstructed']
        assert not result['model_loaded'] and not result['alternative_physical_outcomes_inferred']
    else:
        with pytest.raises(ValueError): summarize_hold_vetoes(iter(rows), tape)
