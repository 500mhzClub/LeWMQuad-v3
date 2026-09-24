"""Physical-prefix integration and counterexamples using synthetic data only."""
from copy import deepcopy
import numpy as np
import pytest

from lewm.tests.test_chained_anchor_controller_completion_development import fixture
from scripts import chained_anchor_native_prefix_development as prefix


def population():
    old, saved, visual, tape = [], [], [], []
    for frame in range(854):
        a, b, c, d = fixture(frame)
        if frame < 3:
            a['decision']['new_selection'] = b['decision']['new_selection'] = None
            b['comparison'] = prefix.replay.compare(a['decision'], b['decision'], d['requested_command'],
                c['candidate'], frame=frame, boundary=853)
        if frame == 853:
            for decision in (a['decision'], b['decision']):
                decision['new_selection']['action'] = 'right_turn'
                decision['requested_command'] = [0., 0., -.45]
                decision['selected_action'] = 'right_turn'
            b['original_requested_command'] = d['requested_command'] = [0., 0., -.45]
        old.append(a); saved.append(b); visual.append(c); tape.append(d)
    report = prefix.completed.reconstruct_report(854,853,850,saved[-1])
    return old, saved, visual, tape, report


def test_full_saved_population_reconstructs_and_preserves_limits():
    old, saved, visual, tape, report = population()
    assert prefix.reconstruct(report, iter(old), iter(saved), tape, iter(visual)) == 850
    assert prefix.PHYSICS_SAMPLES == 43400
    for groups in [(old[:-1],saved,visual),(old,saved[:-1],visual),(old,saved,visual[:-1]),([],[],[])]:
        with pytest.raises(ValueError):
            prefix.reconstruct(report,iter(groups[0]),iter(groups[1]),tape,iter(groups[2]))


@pytest.mark.parametrize('key,value', [('navigation_qualified',True),('new_command_executed',True),
    ('frames',855),('model_state_sha256','other'),('boundary_terminal','FAIL'),
    ('boundary_selected_action','left_turn'),('boundary_requested_command',[0.,0.,.45])])
def test_different_or_overclaimed_report_rejected(key,value):
    report = population()[-1]; report[key] = value
    with pytest.raises(ValueError): prefix.boundary(report)


def test_same_command_is_not_described_as_controller_failure_recovery():
    report = population()[-1]
    report['boundary_comparison']['navigation_recovered'] = True
    with pytest.raises(ValueError): prefix.boundary(report)


@pytest.mark.parametrize('kind', ['earlier_request','boundary_request','incomplete','endpoint','missing'])
def test_execution_tape_changes_rejected(kind):
    *_, tape, report = population()
    other = deepcopy(tape)
    if kind == 'earlier_request': other[400]['requested_command'] = [.2,0.,0.]
    elif kind == 'boundary_request': other[853]['requested_command'] = [0.,0.,0.]
    elif kind == 'incomplete': other[853]['completed'] = False
    elif kind == 'endpoint': other[853]['post_sample_index'] -= 1
    else: other.pop()
    with pytest.raises(ValueError): prefix.executed_boundary([tape,other],report)


def setup_comparison(tmp_path, monkeypatch):
    old,saved,visual,tape,report = population()
    prior,current,root = (tmp_path/n for n in ('prior','current','prefix'))
    for directory in (prior,current):
        directory.mkdir()
        np.savez(directory/'physics_trace.npz', position=np.zeros((43450,3)), clock=np.arange(43450))
    actual = [dict(a,decision=deepcopy(b['decision'])) for a,b in zip(old,saved,strict=True)]
    populations = {prior:old,current:actual,root:saved,prefix.replay.observer.OUTPUT:visual}
    monkeypatch.setattr(prefix,'read_rows',lambda p: (row for row in populations[p]))
    monkeypatch.setattr(prefix,'read_json',lambda p,n: deepcopy(tape))
    monkeypatch.setattr(prefix,'artifact_path',lambda p,n: p/n)
    monkeypatch.setattr(prefix,'public_packets',lambda p: iter(['actual']*854))
    return prior,current,root,report,populations


def test_complete_physical_public_and_decision_prefix(tmp_path,monkeypatch):
    prior,current,root,report,_ = setup_comparison(tmp_path,monkeypatch)
    result = prefix.compare(prior,current,root,report)
    assert result['physical_prefix_samples'] == 43400
    assert result['boundary_command_samples_present'] == 50
    assert result['common_prefix_frames'] == 854
    assert result['candidate_intervention_command_completed'] is True
    assert result['intervention_command_changed'] is False
    assert result['navigation_verified'] is False


@pytest.mark.parametrize('kind',['physics','short_physics','decision','public','missing_observation'])
def test_fresh_simulation_divergence_is_rejected(tmp_path,monkeypatch,kind):
    prior,current,root,report,rows = setup_comparison(tmp_path,monkeypatch)
    if kind in ('physics','short_physics'):
        n = 43449 if kind == 'short_physics' else 43450
        position = np.zeros((n,3))
        if kind == 'physics': position[43399,0] = .001
        np.savez(current/'physics_trace.npz',position=position,clock=np.arange(n))
    elif kind == 'decision': rows[current][400]['decision']['new_selection']['prediction']['x'] = [2]
    elif kind == 'public':
        monkeypatch.setattr(prefix,'public_packets',lambda p: iter(['other' if p == current else 'actual']*854))
    else: rows[current].pop()
    with pytest.raises(ValueError): prefix.compare(prior,current,root,report)


def test_following_physics_is_not_claimed_to_match(tmp_path,monkeypatch):
    prior,current,root,report,_ = setup_comparison(tmp_path,monkeypatch)
    position = np.zeros((43450,3)); position[43400:,0] = .001
    np.savez(current/'physics_trace.npz',position=position,clock=np.arange(43450))
    result = prefix.compare(prior,current,root,report)
    assert result['following_physical_outcomes_compared'] is False
