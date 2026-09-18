"""Matched physical budgets, model absence and exact prospective intervention."""
from copy import deepcopy
import inspect

import numpy as np
import pytest

from scripts import reactive_measured_plane_extended_maze_development as pipeline
from scripts import reactive_measured_plane_native_prefix_development as prefix
from lewm.tests.test_measured_plane_reactive_prefix_development import fixture
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


def test_private_physics_and_full_raw_audit_keep_code_and_extended_dependencies():
    assert pipeline.collect.__code__ is pipeline.episode.collect.__code__
    assert pipeline.audit.__code__ is pipeline.raw_audit.audit.__code__
    assert pipeline.audit_commands.__code__ is pipeline.commands.audit_commands.__code__
    for fn, old in ((pipeline.collect, pipeline.episode.collect), (pipeline.audit, pipeline.raw_audit.audit)):
        assert fn.__globals__ is not old.__globals__
        assert fn.__globals__['NAVIGATION_TICKS'] == 4000
        assert fn.__globals__['ReactiveFloorTransportController'] is pipeline.MeasuredPlaneReactiveController
        assert old.__globals__['ReactiveFloorTransportController'] is not pipeline.MeasuredPlaneReactiveController
        assert 'model' not in inspect.signature(fn).parameters
    collect, audit = pipeline.collect.__globals__, pipeline.audit.__globals__
    assert collect['MAX_OBSERVATIONS'] == pipeline.MAX_COMMAND_TICKS+1 == 4014
    assert collect['writer'] is pipeline.extended.writer
    assert collect['RendererWitnessDualCameraMazeSession'] is pipeline.extended.ExtendedBudgetRendererSession
    assert collect['COLLECTION_ALLOWANCE_BYTES'] == 14*1024**3
    assert audit['MAX_COMMAND_TICKS'] == 4013
    assert audit['IntentReturnRGBDReplay'] is pipeline.ExtendedBudgetRGBDReplay
    assert audit['audit_commands'].__globals__['NAVIGATION_TICKS'] == 4000
    assert audit['read_rows'] is pipeline.read_rows and audit['packet'] is pipeline.rgb_packet
    assert audit['audit_sensors'] is pipeline.extended.audit_sensors
    assert audit['renderer_audit'] is pipeline.extended.renderer_audit


def test_definition_retains_same_physical_budget_and_explicit_method_scope():
    definition = pipeline.definition(); learned = pipeline.learned.definition()
    for key in ('navigation_ticks', 'max_command_ticks', 'max_observations', 'collection_allowance_bytes',
            'renderer_capture_witnesses_enabled', 'physics_paused_during_compute', 'native_pose_input'):
        assert definition[key] == learned[key]
    assert definition['fully_nonpredictive_controller'] and definition['reactive_is_whole_method_comparison']
    for key in ('high_level_world_model_loaded', 'candidate_future_outcomes_evaluated',
            'predictive_surface_or_path_gates_applied', 'learned_residual_used',
            'isolated_prediction_ranking_ablation', 'future_constraint_gates_matched',
            'navigation_qualified', 'real_time_qualified', 'hardware_qualified', 'goal_achieved'):
        assert definition[key] is False


def population():
    saved, baseline, reactive, old_tape, new_tape = [], [], [], [], []
    for frame in range(4):
        old, new, _ = fixture(frame, changed=frame == 3)
        if frame == 3:
            old['new_selection']['action'] = 'left_arc'; old['requested_command'] = [.16, 0., .45]
        a, ta = endpoint(frame, deepcopy(old)); b, tb = endpoint(frame, deepcopy(new))
        baseline.append(a); reactive.append(b); old_tape.append(ta); new_tape.append(tb)
        check = prefix.replay.compare(old, new, old, frame=frame)
        saved.append(dict(tick=frame, baseline=old, decision=new, comparison=check, public_packet_sha256='public'))
    report = prefix.replay.result_report(4, 1, check, old, new)
    return report, saved, baseline, reactive, old_tape, new_tape


def setup(tmp_path, monkeypatch):
    report, saved, baseline, reactive, old_tape, new_tape = population()
    prior, current = tmp_path/'prior', tmp_path/'current'
    for directory in (prior, current):
        directory.mkdir(); np.savez(directory/'physics_trace.npz', pose=np.zeros((950, 3)), clock=np.arange(950))
    monkeypatch.setitem(prefix._compare.__globals__, 'artifact_path', lambda root, name: root/name)
    monkeypatch.setattr(prefix.run, 'read_json', lambda root, name: deepcopy(old_tape if root == prior else new_tape))
    def rows(root):
        yield from baseline if root == prior else reactive if root == current else saved
    monkeypatch.setattr(prefix.run.pipeline, 'read_rows', rows)
    monkeypatch.setitem(prefix._compare.__globals__, 'packets', lambda root: iter(['public']*4))
    return prior, current, report, baseline, reactive, old_tape, new_tape


def test_full_actual_prefix_and_first_reactive_command_with_no_outcome_inference(tmp_path, monkeypatch):
    prior, current, report, *_ = setup(tmp_path, monkeypatch)
    pose = np.zeros((950, 3)); pose[900:, 0] = .02
    np.savez(current/'physics_trace.npz', pose=pose, clock=np.arange(950))
    result = prefix.compare(prior, current, report)
    assert result['physical_prefix_samples'] == 900 and result['common_prefix_frames'] == 4
    assert result['candidate_intervention_command_completed']
    assert result['complete_baseline_decisions_match_prospective_prefix']
    assert result['fully_nonpredictive_candidate'] and result['reactive_is_whole_method_comparison']
    assert not result['both_arms_predictive'] and not result['future_constraint_gates_matched']
    assert not result['following_physical_outcomes_compared'] and not result['navigation_verified']


@pytest.mark.parametrize('fault', ['physics', 'before', 'new_command', 'old_command', 'incomplete',
    'short_physics', 'old_decision', 'new_decision', 'public', 'missing', 'model_calls', 'scope', 'residual'])
def test_unexecuted_or_unmatched_reactive_intervention_is_rejected(tmp_path, monkeypatch, fault):
    prior, current, report, baseline, reactive, old_tape, new_tape = setup(tmp_path, monkeypatch)
    if fault == 'physics':
        pose = np.zeros((950, 3)); pose[899, 0] = .02
        np.savez(current/'physics_trace.npz', pose=pose, clock=np.arange(950))
    elif fault == 'before': new_tape[2]['requested_command'] = [.2, 0., 0.]
    elif fault == 'new_command': new_tape[3]['requested_command'] = [0., 0., 0.]
    elif fault == 'old_command': old_tape[3]['requested_command'] = [0., 0., 0.]
    elif fault == 'incomplete': new_tape[3]['completed'] = False
    elif fault == 'short_physics': np.savez(current/'physics_trace.npz', pose=np.zeros((949, 3)))
    elif fault == 'old_decision': baseline[3]['decision']['failure'] = 'changed'
    elif fault == 'new_decision': reactive[3]['decision']['failure'] = 'changed'
    elif fault == 'public': monkeypatch.setitem(prefix._compare.__globals__, 'packets', lambda root: iter([str(root)]*4))
    elif fault == 'missing': reactive.pop()
    elif fault == 'model_calls': report['actual_model_forward_calls'][1] = 1
    elif fault == 'scope': report['fully_nonpredictive_arm'] = False
    elif fault == 'residual': report['reactive_residual_instantiated'] = True
    with pytest.raises(ValueError): prefix.compare(prior, current, report)
