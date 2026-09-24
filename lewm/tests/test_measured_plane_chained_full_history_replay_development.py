"""The chained adapter retains the full paired loop and closed-output checks."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import measured_plane_chained_full_history_replay_development as job
from scripts import measured_plane_chained_full_history_timing_development as accounting
from lewm.tests import test_measured_plane_full_history_replay_development as original_test


def setup(tmp_path, monkeypatch, fault=None):
    admission, consumed, models, output = original_test.setup(tmp_path, monkeypatch, fault)
    previous = original_test.job
    monkeypatch.setattr(job, 'native', SimpleNamespace(OUTPUT=previous.native.OUTPUT,
        CASE=previous.original.CASE, assigned_model=previous.original.assigned_model))
    monkeypatch.setattr(job, 'MeasuredPlaneChainedAnchorController', previous.MeasuredPlaneResidualController)
    monkeypatch.setattr(job, 'MeasuredPlaneChainedSinglePassController', previous.MeasuredPlaneSinglePassController)
    # Synthetic controller objects only: real chained state/decision behavior is
    # covered by the separate image-to-action composition tests.
    view = SimpleNamespace(**vars(accounting))
    view.normalize = previous.comparison.normalize
    view.observed_state = previous.comparison.observed_state
    monkeypatch.setattr(job, 'comparison', view)
    return admission, consumed, models, output


def test_exact_original_function_bodies_are_privately_bound_to_chained_inputs(tmp_path):
    originals = (job.previous.replay, job.previous.check_output)
    snapshots = [dict(f.__globals__) for f in originals]
    replay, check = job.adapters(tmp_path)
    for copied, original, snapshot in zip((replay, check), originals, snapshots, strict=True):
        assert copied.__code__ is original.__code__
        assert copied.__globals__ is not original.__globals__
        assert original.__globals__.keys() == snapshot.keys()
        assert all(original.__globals__[key] is value for key, value in snapshot.items())
        assert copied.__globals__['native'] is job.native
        assert copied.__globals__['original'].CASE == job.native.CASE
        assert copied.__globals__['original'].assigned_model is job.native.assigned_model
        assert copied.__globals__['comparison'] is accounting
        assert copied.__globals__['OUTPUT'] == tmp_path
    assert replay.__globals__['MeasuredPlaneResidualController'] is job.MeasuredPlaneChainedAnchorController
    assert replay.__globals__['MeasuredPlaneSinglePassController'] is job.MeasuredPlaneChainedSinglePassController
    assert not list(tmp_path.iterdir())


def test_complete_synthetic_history_and_terminal_are_replayed_and_reconstructed(tmp_path, monkeypatch):
    admission, consumed, models, output = setup(tmp_path, monkeypatch)
    report = job.replay(admission, output=output)
    assert consumed == list(range(5))
    assert report['baseline'] == 'MeasuredPlaneChainedAnchorController'
    assert report['candidate'] == 'MeasuredPlaneChainedSinglePassController'
    assert report['actual_model_forward_calls'] == [1, 1]
    assert [s['frame'] for s in report['observed_state_checks']] == [0, 3, 4]
    assert all(not m._forward_hooks for m in models)
    job.check_output(report, admission, output=output)
    assert consumed == list(range(5))*2
    assert (output/'resource_monitor.jsonl').is_file()
    assert not (output/'result.json').exists()


@pytest.mark.parametrize('fault', ['decision', 'input', 'calls', 'state', 'model'])
def test_changed_evidence_stops_the_pair_and_releases_model_hooks(tmp_path, monkeypatch, fault):
    admission, consumed, models, output = setup(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError): job.replay(admission, output=output)
    assert consumed == (list(range(5)) if fault == 'model' else list(range(4)))
    assert all(not m._forward_hooks for m in models)
    assert not (output/'result.json').exists()
    if fault == 'decision':
        mismatch = json.loads((output/'decision_mismatch.json').read_text())
        assert mismatch['frame'] == 3
        assert mismatch['candidate']['nested']['evidence'] == 'changed'


@pytest.mark.parametrize('fault', ['missing', 'extra', 'packet', 'original', 'clock', 'report', 'state'])
def test_checker_rejects_modified_saved_evidence(tmp_path, monkeypatch, fault):
    admission, _, _, output = setup(tmp_path, monkeypatch)
    report = job.replay(admission, output=output)
    path = output/'comparison.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == 'missing': rows.pop()
    elif fault == 'extra': rows.append(deepcopy(rows[-1]))
    elif fault == 'packet': rows[3]['public_packet_sha256'] = 'changed'
    elif fault == 'original': rows[3]['original_decision_sha256'] = 'changed'
    elif fault == 'clock': rows[3]['observation_now_ns'] += 1
    elif fault == 'report': report['timing']['all_observations']['candidate_total_s'] = 0
    else:
        states = json.loads((output/'state_checks.json').read_text()); states.pop()
        (output/'state_checks.json').write_text(json.dumps(states))
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError): job.check_output(report, admission, output=output)
