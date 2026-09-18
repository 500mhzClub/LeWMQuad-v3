from copy import deepcopy
import pytest
from lewm.tests.test_reached_frontier_prefix_comparison_development import fixture as decision_fixture
from lewm.reached_frontier_prefix_comparison_development import compare_step
from scripts.reached_frontier_native_prefix_development import boundary, reconstruct, executed_boundary


def fixture():
    originals = []; saved = []; tape = []; seen = False
    for frame in range(6):
        old, new = decision_fixture(reached=frame == 4)
        old['tick'] = new['tick'] = frame
        for row in (old, new):
            row['causal_residual_receipt']['pending_forecast_tick'] = frame if frame >= 3 else None
            if frame < 3: row.update(requested_command=[0., 0., 0.], new_selection=None)
        transition = new['last_frontier_transition_receipt']
        transition.update(frame=frame, measured_ns=1_500_000_000+100_000_000*frame)
        if frame >= 4: new['new_selection']['changed_frontier_target'] = [10, -2]
        if frame == 5: new['requested_command'] = [0., 0., 0.]
        check = compare_step(old, new, old['requested_command'], frame=frame, frontier_previously_reached=seen)
        seen |= check['frontier_reached_this_frame']
        originals.append(dict(tick=frame, decision=old))
        saved.append(dict(tick=frame, decision=new, comparison=check,
            original_requested_command=old['requested_command'], original_complete_decision_reconstructed=True,
            public_input_arrays_unchanged=True, complete_retained_contact_state_equal=True))
        tape.append(dict(tick=frame, completed=True, requested_command=old['requested_command'],
            pre_sample_index=749+50*frame, post_sample_index=799+50*frame))
    last = saved[-1]['decision']
    report = dict(frames=6, first_request_or_terminal_difference=5, first_reached_frontier_frame=4,
        first_normalized_decision_difference=4, final_terminal=None,
        final_requested_command=last['requested_command'], prior_requested_command=originals[-1]['decision']['requested_command'],
        stop_reason='first_changed_request_or_terminal', raw_model_forecast_comparisons=3,
        final_frontier_transition_receipt=last['last_frontier_transition_receipt'], changed_selection=last['new_selection'])
    return report, originals, saved, tape


def test_reconstructs_delayed_command_change_and_unchanged_forecasts():
    report = reconstruct(*fixture())
    assert report['first_reached_frontier_frame'] == 4
    assert report['first_normalized_decision_difference'] == 4
    assert report['first_intervention_frame'] == 5
    assert report['raw_model_forecast_comparisons'] == 3


@pytest.mark.parametrize('fault', ['truncated', 'extra', 'counter', 'late_boundary', 'missing_reach',
    'changed_original_request', 'altered_comparison', 'uncompleted', 'observed_state', 'forecast'])
def test_incomplete_or_changed_replay_is_rejected(fault):
    report, old, saved, tape = fixture()
    if fault == 'truncated': saved.pop()
    if fault == 'extra': saved.append(deepcopy(saved[-1]))
    if fault == 'counter': report['raw_model_forecast_comparisons'] = 2
    if fault == 'late_boundary': report['first_request_or_terminal_difference'] = 6
    if fault == 'missing_reach': report['first_reached_frontier_frame'] = None
    if fault == 'changed_original_request': tape[3]['requested_command'] = [0., 0., 0.]
    if fault == 'altered_comparison': saved[4]['comparison']['normalized_complete_decision_exact'] = True
    if fault == 'uncompleted': tape[5]['completed'] = False
    if fault == 'observed_state': saved[5]['decision']['memory_receipt'] = None
    if fault == 'forecast': saved[5]['decision']['new_selection']['prediction'][0][0][0] += .001
    with pytest.raises(ValueError): reconstruct(report, old, saved, tape)


def test_nonterminal_hold_is_a_valid_measured_intervention_not_filtered_for_movement():
    report, _, _, original = fixture(); new = deepcopy(original)
    new[-1]['requested_command'] = report['final_requested_command']
    assert boundary(report) == (6, 5)
    executed_boundary([original, new], report)


@pytest.mark.parametrize('fault', ['partial', 'post_index', 'pre_index', 'earlier_command', 'boundary_command', 'short'])
def test_native_intervention_must_actually_complete_on_same_prefix(fault):
    report, _, _, original = fixture(); new = deepcopy(original)
    new[-1]['requested_command'] = report['final_requested_command']
    if fault == 'partial': new[-1]['completed'] = False
    if fault == 'post_index': new[-1]['post_sample_index'] -= 1
    if fault == 'pre_index': new[-1]['pre_sample_index'] -= 1
    if fault == 'earlier_command': new[3]['requested_command'] = [0., 0., 0.]
    if fault == 'boundary_command': new[-1]['requested_command'] = [.16, 0., .45]
    if fault == 'short': new.pop()
    with pytest.raises(ValueError): executed_boundary([original, new], report)
