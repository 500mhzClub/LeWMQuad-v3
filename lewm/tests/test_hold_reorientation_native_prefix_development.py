"""Synthetic physical-prefix admission; no real navigation outcome is asserted."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_hold_reorientation_development import selection
from lewm.hold_reorientation_development import HoldReorientation
from lewm.hold_reorientation_prefix_comparison_development import compare_step
from scripts import hold_reorientation_native_prefix_development as prefix


def fixture():
    originals = []; prospective = []; tape = []; state = HoldReorientation()
    for frame in range(406):
        s = selection(frame) if frame >= 3 else None
        if s is not None:
            s['nominal_path_checks'][5]['all_predicted_segments_nominally_clear'] = False
            if frame < 395: s.update(action='left_arc', action_index=2, requested_command=[.16, 0., .45])
        old = dict(tick=frame, controller='residual_anchored_continuation_controller_v1',
            terminal=None, failure=None, new_selection=s,
            selected_action=None if s is None else s['action'],
            requested_command=[0., 0., 0.] if s is None else s['requested_command'],
            evidence=dict(xy=[.2, .3]), mission_receipt=dict(arrivals=[]))
        new = deepcopy(old); new.update(controller='hold_reorientation_controller_v1', hold_reorientation_enabled=True)
        if s is not None:
            new['new_selection'] = state.reconsider(deepcopy(s), frame=frame, now_ns=1_500_000_000+100_000_000*frame)
            new.update(selected_action=new['new_selection']['action'], requested_command=new['new_selection']['requested_command'])
        check = compare_step(old, new, old['requested_command'], frame=frame, expected_selection=new['new_selection'])
        originals.append(dict(tick=frame, observation_index=frame, pre_sample_index=749+50*frame, decision=old))
        prospective.append(dict(tick=frame, decision=new, comparison=check,
            original_requested_command=old['requested_command'], original_complete_decision_reconstructed=True,
            public_input_arrays_unchanged=True, complete_retained_contact_state_equal=True,
            public_input_sha256='packet_'+str(frame)))
        tape.append(dict(tick=frame, completed=True, requested_command=old['requested_command'],
            pre_sample_index=749+50*frame, post_sample_index=799+50*frame))
    report = dict(frames=406, first_changed_command_frame=405, raw_model_forecast_comparisons=403,
        original_requested_command=[0., 0., 0.], candidate_requested_command=[0., 0., .45])
    return report, originals, prospective, tape, prospective[-1]['decision']['new_selection']


def test_complete_comparisons_reconstruct_through_fixed_boundary():
    result = prefix.reconstruct(*fixture())
    assert result['frames'] == 406 and result['raw_model_forecast_comparisons'] == 403
    assert result['first_intervention_frame'] == 405


@pytest.mark.parametrize('fault', ['truncated', 'extra', 'counter', 'boundary', 'command',
    'comparison', 'incomplete', 'observed_state', 'forecast', 'expected_selection'])
def test_partial_or_changed_raw_evidence_cannot_be_admitted(fault):
    r, old, saved, tape, expected = fixture(); expected = deepcopy(expected)
    if fault == 'truncated': saved.pop()
    if fault == 'extra': saved.append(deepcopy(saved[-1]))
    if fault == 'counter': r['raw_model_forecast_comparisons'] -= 1
    if fault == 'boundary': r['first_changed_command_frame'] -= 1
    if fault == 'command': tape[3]['requested_command'] = [0., 0., 0.]
    if fault == 'comparison': saved[3]['comparison']['normalized_complete_decision_exact'] = False
    if fault == 'incomplete': tape[405]['completed'] = False
    if fault == 'observed_state': saved[405]['decision']['evidence']['xy'][0] += 1
    if fault == 'forecast': saved[405]['decision']['new_selection']['prediction'][0][0][0] += 1
    if fault == 'expected_selection': expected['requested_command'] = [0., 0., -.45]
    with pytest.raises(ValueError): prefix.reconstruct(r, old, saved, tape, expected)


@pytest.mark.parametrize('fault', ['incomplete', 'post', 'pre', 'earlier', 'boundary', 'short'])
def test_intervention_command_must_complete_after_identical_prior_commands(fault):
    r, _, _, old, _ = fixture(); new = deepcopy(old); new[-1]['requested_command'] = r['candidate_requested_command']
    if fault == 'incomplete': new[-1]['completed'] = False
    if fault == 'post': new[-1]['post_sample_index'] -= 1
    if fault == 'pre': new[-1]['pre_sample_index'] -= 1
    if fault == 'earlier': new[3]['requested_command'] = [0., 0., 0.]
    if fault == 'boundary': new[-1]['requested_command'] = [0., 0., -.45]
    if fault == 'short': new.pop()
    with pytest.raises(ValueError): prefix.executed_boundary([old, new], r)


def physical_fixture(monkeypatch, tmp_path, fault=None):
    r, old, saved, tape, expected = fixture(); prior = tmp_path/'prior'; current = tmp_path/'current'
    prior.mkdir(); current.mkdir(); root = tmp_path/'prefix'
    actual = deepcopy(old)
    for i,row in enumerate(actual): row['decision'] = deepcopy(saved[i]['decision'])
    new_tape = deepcopy(tape); new_tape[-1]['requested_command'] = r['candidate_requested_command']
    raw = np.arange(21050, dtype=np.float64); other = raw.copy()
    other[21000:] += 100  # Measured outcomes after the changed request must be allowed to differ.
    if fault == 'physics': other[20999] += 1
    np.savez(prior/'physics_trace.npz', sample=raw); np.savez(current/'physics_trace.npz', sample=other)
    if fault == 'decision': actual[-1]['decision']['evidence']['xy'][0] += 1
    if fault == 'completion': new_tape[-1]['completed'] = False
    if fault == 'public': saved[-1]['public_input_sha256'] = 'different'
    def packets(directory, frames):
        for i in range(frames): yield 'packet_'+str(i)
    monkeypatch.setattr(prefix, 'artifact_path', lambda parent, name: parent/name)
    monkeypatch.setattr(prefix, 'read_rows', lambda p: iter(deepcopy(old if p == prior else actual if p == current else saved)))
    monkeypatch.setattr(prefix, 'read_json', lambda p, n: deepcopy(tape if p == prior else new_tape))
    monkeypatch.setattr(prefix, 'public_packets', packets)
    monkeypatch.setattr(prefix.replay, 'saved_inputs', lambda: ({}, {'candidate_selection': expected}, []))
    return prior, current, root, r


def test_physical_comparison_covers_21000_samples_and_completed_boundary(monkeypatch, tmp_path):
    result = prefix.compare(*physical_fixture(monkeypatch, tmp_path))
    assert result['physical_prefix_samples'] == 21000
    assert result['common_prefix_frames'] == 406 and result['candidate_intervention_command_completed']
    assert result['following_physical_outcomes_compared'] is False


@pytest.mark.parametrize('fault', ['physics', 'decision', 'completion', 'public'])
def test_altered_native_or_public_prefix_is_rejected(monkeypatch, tmp_path, fault):
    args = physical_fixture(monkeypatch, tmp_path, fault)
    with pytest.raises(ValueError): prefix.compare(*args)
