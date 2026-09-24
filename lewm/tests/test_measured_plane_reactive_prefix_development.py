"""Reject false reactive provenance, changed shared evidence and causal overrun."""
from copy import deepcopy

import pytest

from scripts import replay_go2_measured_plane_reactive_prefix_v1 as job
from lewm.geometry_progress_pilot_development import candidate_commands


def fixture(frame=3, changed=False):
    reference = dict(controller='measured_plane_residual_continuation_controller_v1',
        measured_plane_constrained_estimator=True, tick=frame, terminal=None, failure=None,
        requested_command=[0., 0., 0.], original_visual_evidence={'frame': frame},
        evidence={'floor': frame}, memory_receipt={'frame': frame},
        mission_receipt={'hold_required': frame < 3}, observed_goal_distance_m=2.,
        auxiliary_floor_partition_receipt={'frame': frame},
        new_selection=None if frame < 3 else dict(prediction=[['synthetic']], action='hold'))
    candidate = deepcopy(reference)
    candidate.update(controller='measured_plane_reactive_controller_v1',
        fully_nonpredictive_controller=True, reactive_is_whole_method_comparison=True,
        **dict.fromkeys(job.FALSE_FLAGS, False))
    if frame >= 3:
        candidate['new_selection'] = dict(action='forward' if changed else 'hold',
            requested_command=list(candidate_commands('forward')[0]) if changed else [0., 0., 0.],
            current_geometry_checked=True, learned_model_used=False,
            candidate_future_outcomes_evaluated=False, predictive_surface_or_path_gates_applied=False,
            command_integrated_pose_used=False, native_state_used=False)
        candidate['requested_command'] = candidate['new_selection']['requested_command'].copy()
    return deepcopy(reference), candidate, reference


@pytest.mark.parametrize('frame', [0, 3, 121, 122])
def test_common_prefix_stops_at_fixed_end_without_navigation_claim(frame):
    baseline, reactive, reference = fixture(frame)
    check = job.compare(baseline, reactive, reference, frame=frame)
    assert check['stop'] == (frame == 122)
    assert check['reactive_is_whole_method_comparison']
    assert not check['future_constraint_gates_matched']
    assert not check['isolated_prediction_ranking_ablation']
    assert not check['navigation_qualified']


def test_first_changed_request_and_terminal_stop_before_following_observation():
    baseline, reactive, reference = fixture(changed=True)
    check = job.compare(baseline, reactive, reference, frame=3)
    assert check['stop'] and check['requested_command_changed']
    assert not check['changed_command_executed']
    assert not check['following_changed_command_observation_consumed']
    reactive.update(terminal='VIEW_BUDGET_EXHAUSTED', requested_command=[0., 0., 0.])
    check = job.compare(baseline, reactive, reference, frame=3)
    assert check['stop'] and check['terminal_boundary']


@pytest.mark.parametrize('key', job.SHARED_KEYS)
def test_complete_shared_receipts_cannot_be_selectively_ignored(key):
    baseline, reactive, reference = fixture()
    reactive[key] = {'changed': True}
    with pytest.raises(ValueError, match='same complete measured'):
        job.compare(baseline, reactive, reference, frame=3)


@pytest.mark.parametrize('fault', ['baseline', 'reactive_model', 'forecast', 'integrated_pose',
    'native_state', 'future_gate', 'current_geometry', 'false_action', 'false_request',
    'missing_selection', 'missing_hold', 'terminal_moves', 'tick', 'frame_bool', 'past_end'])
def test_false_provenance_requests_and_missing_controls_are_rejected(fault):
    baseline, reactive, reference = fixture(); frame = 3
    selection = reactive['new_selection']
    if fault == 'baseline': baseline['new_selection']['prediction'] = [['changed']]
    elif fault == 'reactive_model': reactive['learned_model_used'] = True
    elif fault == 'forecast': selection['prediction'] = [['invented']]
    elif fault == 'integrated_pose': selection['command_integrated_pose_used'] = True
    elif fault == 'native_state': selection['native_state_used'] = True
    elif fault == 'future_gate': selection['predictive_surface_or_path_gates_applied'] = True
    elif fault == 'current_geometry': selection['current_geometry_checked'] = False
    elif fault == 'false_action': selection['action'] = 'forward'
    elif fault == 'false_request': reactive['requested_command'] = [.2, 0., 0.]
    elif fault == 'missing_selection': reactive['new_selection'] = None
    elif fault == 'missing_hold':
        baseline, reactive, reference = fixture(0); frame = 0
        reactive['mission_receipt']['hold_required'] = False
    elif fault == 'terminal_moves': reactive.update(terminal='VIEW_BUDGET_EXHAUSTED', requested_command=[.2, 0., 0.])
    elif fault == 'tick': reactive['tick'] = 4
    elif fault == 'frame_bool': frame = True
    elif fault == 'past_end': frame = 123
    with pytest.raises(ValueError): job.compare(baseline, reactive, reference, frame=frame)


@pytest.mark.parametrize('fault', [None, 'extra', 'short', 'input', 'receipt', 'report', 'forward_guard'])
def test_saved_population_requires_complete_boundary_and_recomputed_report(monkeypatch, fault):
    rows, references = [], []
    for frame in range(4):
        baseline, candidate, reference = fixture(frame, changed=frame == 3)
        check = job.compare(baseline, candidate, reference, frame=frame)
        references.append(dict(tick=frame, decision=reference, public_packet_sha256=str(frame)))
        rows.append(dict(tick=frame, baseline=baseline, decision=candidate, comparison=check,
            public_packet_sha256=str(frame), public_inputs_unchanged=True, reactive_model_forward_blocked=True))
    report = job.result_report(4, 1, check, baseline, candidate)
    if fault == 'extra': rows.append(deepcopy(rows[-1]))
    elif fault == 'short': rows.pop()
    elif fault == 'input': rows[-1]['public_packet_sha256'] = 'changed'
    elif fault == 'receipt': rows[-1]['comparison']['changed_command_executed'] = True
    elif fault == 'report': report['actual_model_forward_calls'][1] = 1
    elif fault == 'forward_guard': rows[-1]['reactive_model_forward_blocked'] = False
    def read(root):
        yield from rows if root == job.OUTPUT else references
    monkeypatch.setattr(job.run.pipeline, 'read_rows', read)
    if fault is None: job.check_output(report)
    else:
        with pytest.raises(ValueError): job.check_output(report)
