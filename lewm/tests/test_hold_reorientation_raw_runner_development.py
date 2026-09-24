"""Synthetic wiring tests; no claim of real raw-controller reconstruction."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
from lewm.tests.test_hold_reorientation_prefix_comparison_development import pair
from scripts import replay_go2_hold_reorientation_maze02_prefix_v1 as runner


def fixture(monkeypatch, tmp_path, fault=None):
    count = 14
    pairs = [pair(i, i == count-1) for i in range(count)]
    rows = [dict(tick=i, decision=a) for i, (a,b) in enumerate(pairs)]
    comparisons = [dict(frame=i, original_row_sha256=runner.saved.identity(row),
        candidate_selection_sha256=runner.saved.identity(pairs[i][1]['new_selection']))
        for i,row in enumerate(rows)]
    boundary = dict(candidate_selection=pairs[-1][1]['new_selection'],
        original_requested_command=[0., 0., 0.], candidate_requested_command=[0., 0., -.45])
    tape = [dict(tick=i, completed=True, pre_sample_index=749+50*i,
        post_sample_index=799+50*i, requested_command=[0., 0., 0.]) for i in range(count)]
    if fault == 'incomplete': tape[-1]['completed'] = False
    if fault == 'clock': tape[-1]['post_sample_index'] += 1
    if fault == 'saved_row': comparisons[3]['original_row_sha256'] = 'bad'
    seen = []
    def stream(directory):
        yield from deepcopy(rows)
        raise AssertionError('must never consume the observation after divergence')
    class Reader:
        def __init__(self, directory): pass
        def packet(self, frame):
            if frame >= count: raise AssertionError('post-intervention raw packet consumed')
            seen.append(frame)
            return dict(frame=frame), {}, {}, 1_500_000_000+frame*100_000_000
    class Controller:
        def __init__(self, *args, candidate=False, **kwargs):
            self.candidate = candidate
            self.residual = SimpleNamespace(pending=None)
            self.mapper = SimpleNamespace(floor={(0,0)}, occupied=set())
            self.memory = dict(observed=['same'])
        def observe(self, p, *args, **kwargs):
            frame = p['frame']; result = deepcopy(pairs[frame][int(self.candidate)])
            if fault == 'raw_original' and not self.candidate and frame == 3:
                result['evidence']['observed_xy'] = [99., 99.]
            if fault == 'contact' and self.candidate and frame == 3: self.memory = {'changed': True}
            if fault == 'map' and self.candidate and frame == 3: self.mapper.floor.add((1,1))
            if fault == 'mutation' and self.candidate and frame == 3: p['changed'] = True
            if fault == 'pending' and self.candidate and frame == 3: self.residual.pending = {'changed': True}
            if fault == 'early_command' and self.candidate and frame == 3:
                result['requested_command'] = [0., 0., .45]
            return result
    monkeypatch.setattr(runner, 'FRAMES', count)
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'saved_inputs', lambda: ({}, deepcopy(boundary), deepcopy(comparisons)))
    monkeypatch.setattr(runner, 'read_rows', stream)
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', Reader)
    monkeypatch.setattr(runner, 'packet', lambda *a, **k: ({}, {}))
    monkeypatch.setattr(runner, 'public_acquisition', lambda x: x)
    monkeypatch.setattr(runner, 'read_json', lambda root, name: deepcopy(tape) if name == 'command_tape.json'
        else [{}]*count if name == 'auxiliary_camera_audit.json' else {})
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda x: object())
    monkeypatch.setattr(runner, 'ResidualAnchoredContinuationController', Controller)
    monkeypatch.setattr(runner, 'HoldReorientationController', lambda *a, **k: Controller(*a, candidate=True, **k))
    monkeypatch.setattr(runner.original, 'assigned_model', lambda *a: SimpleNamespace(state_dict=lambda: {}, parameters=lambda: []))
    monkeypatch.setattr(runner, 'state_digest', lambda x: runner.MODEL_SHA)
    return seen


def test_complete_synthetic_pipeline_stops_before_first_unexecuted_observation(monkeypatch, tmp_path):
    seen = fixture(monkeypatch, tmp_path)
    result = runner.replay()
    assert seen == list(range(14))
    assert result['frames'] == 14 and result['raw_model_forecast_comparisons'] == 11
    assert result['first_changed_command_frame'] == 13
    assert result['candidate_requested_command'] == [0., 0., -.45]
    assert result['native_execution'] is False


@pytest.mark.parametrize('fault', ['incomplete', 'clock', 'saved_row', 'raw_original',
    'contact', 'map', 'mutation', 'pending', 'early_command'])
def test_changed_inputs_incomplete_commands_and_state_divergence_stop_replay(monkeypatch, tmp_path, fault):
    fixture(monkeypatch, tmp_path, fault)
    with pytest.raises(ValueError): runner.replay()


@pytest.mark.parametrize('memory,storage', [(31, 100), (64, 40)])
def test_resource_shortfall_rejected(memory, storage):
    with pytest.raises(ValueError): runner.resources_for(dict(memory_available_bytes=memory*1024**3,
        artifact_free_bytes=storage*1024**3))
