"""A budget control must expose, rather than normalize away, earlier drift."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import extended_budget_anchored_prefix_development as prefix


def row(frame, budget):
    terminal = 'MISSION_TICK_BUDGET_EXHAUSTED' if budget == 3000 and frame == 3003 else None
    return dict(tick=frame, observation_index=frame, pre_sample_index=749+50*frame,
        decision=dict(tick=frame, requested_command=[0., 0., 0.], terminal=terminal,
            shared_navigation_budget_ticks=budget,
            mission_receipt=dict(global_navigation_ticks=budget, phase='RETURN', arrivals=[]),
            new_selection=None if frame < 3 else dict(prediction=[[.1, .2]], view_budget_exhausted=False),
            evidence=dict(position=[.1, .2])))


def tape():
    return [dict(tick=i, requested_command=[0., 0., 0.], completed=True,
        pre_sample_index=749+50*i, post_sample_index=799+50*i,
        phase=1 if i < 3 else 2, role='synthetic') for i in range(3003)]


def test_normalization_changes_only_two_paths_without_aliasing():
    before = row(3, 4000)['decision']; saved = deepcopy(before)
    result = prefix.normalize(before, budget=4000)
    assert result == row(3, 3000)['decision'] and before == saved
    result['new_selection']['prediction'][0][0] += 1
    assert before == saved
    assert result['new_selection']['view_budget_exhausted'] is False


@pytest.mark.parametrize('fault', ['missing_mission', 'missing_shared', 'float', 'wrong_mission', 'wrong_shared', 'bool'])
def test_exact_typed_budget_fields_required(fault):
    d = row(0, 4000)['decision']
    if fault == 'missing_mission': d.pop('mission_receipt')
    elif fault == 'missing_shared': d.pop('shared_navigation_budget_ticks')
    elif fault == 'float': d['mission_receipt']['global_navigation_ticks'] = 4000.
    elif fault == 'wrong_mission': d['mission_receipt']['global_navigation_ticks'] = 3000
    elif fault == 'wrong_shared': d['shared_navigation_budget_ticks'] = 3999
    else: d['shared_navigation_budget_ticks'] = True
    with pytest.raises(ValueError): prefix.normalize(d, budget=4000)


def test_complete_prefix_exposes_only_original_budget_intervention():
    c = prefix.BudgetPrefixComparison()
    for i in range(3004): c.observe(row(i, 3000), row(i, 4000))
    r = c.report()
    assert r['all_preboundary_decisions_exact']
    assert r['first_normalized_decision_difference'] == r['first_requested_command_or_terminal_difference'] == 3003
    assert r['boundary']['candidate_continues'] and r['equal_preboundary_forecast_decisions'] == 3000
    assert not r['verified_round_trip'] and not r['following_observations_compared']
    with pytest.raises(ValueError): c.observe(row(3004, 3000), row(3004, 4000))


@pytest.mark.parametrize('fault', ['forecast', 'pose', 'command', 'unrelated_budget', 'terminal'])
def test_earlier_differences_are_retained_as_negative_results(fault):
    c = prefix.BudgetPrefixComparison()
    for i in range(3004):
        a, b = row(i, 3000), row(i, 4000)
        if i == 17:
            d = b['decision']
            if fault == 'forecast': d['new_selection']['prediction'][0][0] += .01
            elif fault == 'pose': d['evidence']['position'][0] += .01
            elif fault == 'command': d['requested_command'][0] = .2
            elif fault == 'unrelated_budget': d['new_selection']['view_budget_exhausted'] = True
            else: d['terminal'] = 'TRACKING_FAILURE'
        c.observe(a, b)
    r = c.report()
    assert not r['all_preboundary_decisions_exact'] and r['first_normalized_decision_difference'] == 17
    assert r['first_requested_command_or_terminal_difference'] == (17 if fault in ('command', 'terminal') else 3003)


def test_population_chronology_and_original_terminal_boundary_required():
    c = prefix.BudgetPrefixComparison()
    with pytest.raises(ValueError): c.report()
    with pytest.raises(ValueError): c.observe(row(1, 3000), row(1, 4000))
    a, b = row(0, 3000), row(0, 4000); b['decision']['tick'] = False
    with pytest.raises(ValueError): c.observe(a, b)
    for i in range(3003): c.observe(row(i, 3000), row(i, 4000))
    with pytest.raises(ValueError): c.report()
    a = row(3003, 3000); a['decision']['terminal'] = 'OTHER_FAILURE'
    with pytest.raises(ValueError): c.observe(a, row(3003, 4000))


@pytest.mark.parametrize('fault', ['missing', 'incomplete', 'endpoint', 'command', 'phase'])
def test_command_population_and_drift(fault):
    a = tape(); b = deepcopy(a)
    if fault == 'missing': b.pop()
    elif fault == 'incomplete': b[3002]['completed'] = False
    elif fault == 'endpoint': b[3002]['post_sample_index'] -= 1
    elif fault == 'command': b[3002]['requested_command'][0] = .2
    else: b[3002]['phase'] = 3
    if fault in ('missing', 'incomplete', 'endpoint'):
        with pytest.raises(ValueError): prefix.command_prefix([a, b])
    else:
        r = prefix.command_prefix([a, b])
        assert not r['all_preboundary_commands_exact'] and r['first_complete_command_difference'] == 3002


@pytest.mark.parametrize('fault', [None, 'physics', 'public', 'decision', 'unbound', 'after_hash'])
def test_physical_public_decision_and_file_binding_checks_are_joined(tmp_path, monkeypatch, fault):
    prior, current = tmp_path/'prior', tmp_path/'current'
    for p in (prior, current):
        p.mkdir(); raw = np.arange(151400, dtype=np.float64)
        if p == current:
            raw[prefix.PHYSICS_SAMPLES:] += 100  # Executed outcomes after intervention may differ.
            if fault == 'physics': raw[prefix.PHYSICS_SAMPLES-1] += .1
        np.savez_compressed(p/'physics_trace.npz', timestamp_s=raw)
        (p/'command_tape.json').write_text(json.dumps(tape()))
    checks = []
    def verify(root, ids):
        checks.append((root, ids))
        if fault == 'after_hash' and len(checks) == 4: raise ValueError('changed input bytes')
    monkeypatch.setattr(prefix, 'verify_artifacts', verify)
    monkeypatch.setattr(prefix, 'artifact_path', lambda root, name: root/name)
    def rows(p):
        for i in range(3004):
            value = row(i, 3000 if p == prior else 4000)
            if fault == 'decision' and p == current and i == 100: value['decision']['evidence']['position'][0] += 1
            yield value
        raise AssertionError('must not consume observation 3004')
    monkeypatch.setattr(prefix, 'original_rows', rows)
    monkeypatch.setattr(prefix.extended, 'read_rows', rows)
    def packets(p, frames):
        assert frames == 3004
        for i in range(frames): yield str(i)+(str(p) if fault == 'public' and i == 3003 else '')
    monkeypatch.setattr(prefix, 'public_packets', packets)
    names = {'physics_trace.npz', 'command_tape.json', 'context_decisions.jsonl.gz',
        'policy_observations.json', 'policy_histories.npz', 'depth_observations.json',
        'fast_gyro_histories.npz', 'auxiliary_camera_audit.json'}
    names |= {f'{kind}_{i:04d}.{suffix}' for i in range(3004) for kind, suffix in
        [('rgb', 'png'), ('depth', 'npz'), ('auxiliary_rgb', 'png'), ('auxiliary_depth', 'npz')]}
    ids = [{p.name+'/'+n:'a'*64 for n in names} for p in (prior, current)]
    if fault == 'unbound': ids[1].pop('current/auxiliary_depth_3003.npz')
    if fault in ('unbound', 'after_hash'):
        with pytest.raises(ValueError): prefix.compare(prior, current, prior_bindings=ids[0], current_bindings=ids[1])
        return
    r = prefix.compare(prior, current, prior_bindings=ids[0], current_bindings=ids[1])
    assert r['budget_only_preboundary_execution_supported'] is (fault is None)
    assert len(checks) == 4
    assert r['physical_prefix_samples'] == 150900
    if fault == 'public': assert r['first_public_packet_difference'] == 3003
    if fault == 'decision': assert r['first_normalized_decision_difference'] == 100
    if fault == 'physics': assert not r['physical_prefix_exact']
    assert not r['launch_and_model_admission_performed'] and not r['full_raw_sensor_audit_replaced']
