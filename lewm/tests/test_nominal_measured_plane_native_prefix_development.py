"""Both actual controllers and all preintervention physics must match."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import nominal_measured_plane_native_prefix_development as prefix
from lewm.tests.test_measured_plane_forecast_source_prefix_development import fixture
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


def population():
    saved, baseline, nominal, old_tape, new_tape = [], [], [], [], []
    for frame in range(4):
        old, new, reference = fixture(frame)
        if frame == 3:
            old['new_selection']['action'] = 'left_arc'; old['requested_command'] = [.16, 0., .45]
            old['new_selection']['phase_admissible_candidates'] = 6
            new['new_selection']['action'] = 'forward'; new['requested_command'] = [.2, 0., 0.]
            new['new_selection']['phase_admissible_candidates'] = 6
            reference = prefix.replay.learned_reference(old)
        a, ta = endpoint(frame, reference); b, tb = endpoint(frame, new)
        baseline.append(a); nominal.append(b); old_tape.append(ta); new_tape.append(tb)
        check = prefix.replay.compare(old, new, reference, frame=frame)
        saved.append(dict(tick=frame, baseline=old, decision=new, comparison=check, public_packet_sha256='public'))
    report = prefix.replay.result_report(4, 1, check, old, new)
    return report, saved, baseline, nominal, old_tape, new_tape


def setup(tmp_path, monkeypatch):
    report, saved, baseline, nominal, old_tape, new_tape = population()
    prior, current = tmp_path/'prior', tmp_path/'current'
    for directory in (prior, current):
        directory.mkdir(); np.savez(directory/'physics_trace.npz', pose=np.zeros((950, 3)), clock=np.arange(950))
    monkeypatch.setattr(prefix, 'artifact_path', lambda root, name: root/name)
    monkeypatch.setattr(prefix.run, 'read_json', lambda root, name: deepcopy(old_tape if root == prior else new_tape))
    def rows(root):
        yield from baseline if root == prior else nominal if root == current else saved
    monkeypatch.setattr(prefix.run.pipeline, 'read_rows', rows)
    monkeypatch.setattr(prefix, 'packets', lambda root: iter(['public']*4))
    return prior, current, report, baseline, nominal, old_tape, new_tape


def test_actual_new_command_follows_identical_full_physics_prefix(tmp_path, monkeypatch):
    prior, current, report, *_ = setup(tmp_path, monkeypatch)
    pose = np.zeros((950, 3)); pose[900:, 0] = .02
    np.savez(current/'physics_trace.npz', pose=pose, clock=np.arange(950))
    result = prefix.compare(prior, current, report)
    assert result['physical_prefix_samples'] == 900 and result['common_prefix_frames'] == 4
    assert result['candidate_intervention_command_completed']
    assert result['complete_baseline_decisions_match_prospective_prefix']
    assert not result['following_physical_outcomes_compared'] and not result['navigation_verified']


@pytest.mark.parametrize('fault', ['physics', 'before', 'new_command', 'old_command', 'incomplete',
    'short_physics', 'old_decision', 'new_decision', 'public', 'missing', 'model_calls'])
def test_unexecuted_or_unmatched_intervention_is_rejected(tmp_path, monkeypatch, fault):
    prior, current, report, baseline, nominal, old_tape, new_tape = setup(tmp_path, monkeypatch)
    if fault == 'physics':
        pose = np.zeros((950, 3)); pose[899, 0] = .02
        np.savez(current/'physics_trace.npz', pose=pose, clock=np.arange(950))
    elif fault == 'before': new_tape[2]['requested_command'] = [.2, 0., 0.]
    elif fault == 'new_command': new_tape[3]['requested_command'] = [0., 0., 0.]
    elif fault == 'old_command': old_tape[3]['requested_command'] = [0., 0., 0.]
    elif fault == 'incomplete': new_tape[3]['completed'] = False
    elif fault == 'short_physics': np.savez(current/'physics_trace.npz', pose=np.zeros((949, 3)))
    elif fault == 'old_decision': baseline[3]['decision']['failure'] = 'changed'
    elif fault == 'new_decision': nominal[3]['decision']['failure'] = 'changed'
    elif fault == 'public': monkeypatch.setattr(prefix, 'packets', lambda root: iter([str(root)]*4))
    elif fault == 'missing': nominal.pop()
    elif fault == 'model_calls': report['actual_model_forward_calls'][1] = 1
    with pytest.raises(ValueError): prefix.compare(prior, current, report)
