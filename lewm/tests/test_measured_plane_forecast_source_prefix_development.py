"""Exact source provenance, full original reproduction and causal stop checks."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import replay_go2_measured_plane_forecast_source_prefix_v1 as job
from lewm.geometry_progress_pilot_development import candidate_commands


def fixture(frame=3, changed=False):
    reference = dict(controller='measured_plane_residual_continuation_controller_v1',
        measured_plane_constrained_estimator=True, tick=frame, terminal=None, failure=None,
        requested_command=[0., 0., 0.], original_visual_evidence={'frame': frame},
        evidence={'floor': frame}, memory_receipt={'frame': frame}, mission_receipt={'hold_required': False},
        new_selection=None if frame < 3 else dict(prediction=[['synthetic']], action='hold',
            model_prediction_corrected=True, translation_bias_training_only=True))
    arms = []
    for mode in job.MODES:
        row = deepcopy(reference)
        row.update(controller='measured_plane_forecast_source_controller_v1', assigned_forecast_source=mode,
            shared_observed_residual_correction_retained=True, fully_nonpredictive_controller=False)
        learned = mode == job.MODES[0]
        if frame >= 3:
            row['new_selection'].update(model_prediction_corrected=learned, translation_bias_training_only=learned,
                forecast_provenance=dict(forecast_source=mode, frozen_model_forward_called=learned,
                    learned_forecasts_used=learned, nominal_requested_twist_forecasts_used=not learned))
        arms.append(row)
    if changed:
        arms[1]['new_selection']['action'] = 'forward'
        arms[1]['requested_command'] = list(candidate_commands('forward')[0])
    return *arms, reference


def test_learned_normalization_keeps_nested_scientific_evidence_and_inputs():
    baseline, _, reference = fixture()
    before = deepcopy(baseline)
    assert job.learned_reference(baseline) == reference
    assert baseline == before
    baseline['new_selection']['prediction'] = [['changed']]
    assert job.learned_reference(baseline) != reference


@pytest.mark.parametrize('frame', [0, 3, 121, 122])
def test_fixed_shared_prefix_stops_only_at_its_complete_end(frame):
    baseline, nominal, reference = fixture(frame)
    result = job.compare(baseline, nominal, reference, frame=frame)
    assert result['stop'] == (frame == 122)
    assert result['both_arms_predictive'] and not result['isolated_planning_on_off_comparison']


def test_first_changed_request_and_terminal_are_explicit_boundaries():
    baseline, nominal, reference = fixture(changed=True)
    result = job.compare(baseline, nominal, reference, frame=3)
    assert result['stop'] and result['requested_command_changed']
    assert not result['changed_command_executed']
    assert not result['following_changed_command_observation_consumed']
    nominal.update(terminal='SENSOR_OR_MODEL_FAILURE', failure='synthetic failure', requested_command=[0., 0., 0.])
    result = job.compare(baseline, nominal, reference, frame=3)
    assert result['stop'] and result['terminal_boundary']


@pytest.mark.parametrize('fault', ['original', 'visual', 'floor', 'memory', 'mission', 'mode',
    'model_called', 'learned', 'bias', 'command', 'tick', 'missing_prediction', 'past_end'])
def test_changed_evidence_or_false_source_claim_is_rejected(fault):
    baseline, nominal, reference = fixture()
    frame = 3
    if fault == 'original': baseline['failure'] = 'changed'
    elif fault in ('visual', 'floor', 'memory', 'mission'):
        key = dict(visual='original_visual_evidence', floor='evidence', memory='memory_receipt', mission='mission_receipt')[fault]
        nominal[key]['changed'] = True
    elif fault == 'mode': nominal['fully_nonpredictive_controller'] = True
    elif fault == 'model_called': nominal['new_selection']['forecast_provenance']['frozen_model_forward_called'] = True
    elif fault == 'learned': nominal['new_selection']['forecast_provenance']['learned_forecasts_used'] = True
    elif fault == 'bias': nominal['new_selection']['translation_bias_training_only'] = True
    elif fault == 'command': nominal['requested_command'] = [1., 0., 0.]
    elif fault == 'tick': nominal['tick'] = 4
    elif fault == 'missing_prediction': nominal['new_selection'] = None
    elif fault == 'past_end': frame = 123
    with pytest.raises(ValueError): job.compare(baseline, nominal, reference, frame=frame)


@pytest.mark.parametrize('fault', [None, 'extra', 'short', 'input', 'receipt', 'report'])
def test_complete_saved_population_is_reconstructed_and_cannot_cross_boundary(monkeypatch, fault):
    rows, references = [], []
    for frame in range(4):
        baseline, nominal, reference = fixture(frame, changed=frame == 3)
        check = job.compare(baseline, nominal, reference, frame=frame)
        references.append(dict(tick=frame, decision=reference, public_packet_sha256=str(frame)))
        rows.append(dict(tick=frame, baseline=baseline, decision=nominal, comparison=check,
            public_packet_sha256=str(frame), public_inputs_unchanged=True))
    report = job.result_report(4, 1, check, baseline, nominal)
    if fault == 'extra': rows.append(deepcopy(rows[-1]))
    elif fault == 'short': rows.pop()
    elif fault == 'input': rows[-1]['public_packet_sha256'] = 'changed'
    elif fault == 'receipt': rows[-1]['comparison']['changed_command_executed'] = True
    elif fault == 'report': report['actual_model_forward_calls'][1] = 1
    def read(root):
        yield from rows if root == job.OUTPUT else references
    monkeypatch.setattr(job.run.pipeline, 'read_rows', read)
    if fault is None: job.check_output(report)
    else:
        with pytest.raises(ValueError): job.check_output(report)
