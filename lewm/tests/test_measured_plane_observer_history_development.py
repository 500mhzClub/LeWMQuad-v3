"""Original evidence reproduction and operational gates for the history probe."""
from copy import deepcopy
import pytest

from scripts import replay_go2_measured_plane_observer_history_v1 as runner


def example(frame=0):
    visual = dict(identity=[0,0,0], nested=dict(position=[.1,.2,.3]), terminal_failure=None)
    floor = dict(original_visual_evidence=deepcopy(visual), correction=dict(height=.001))
    row = dict(tick=frame, decision=dict(original_visual_evidence=deepcopy(visual),
        evidence=deepcopy(floor), failure=None))
    return row, visual, floor


def test_original_evidence_comparison_accepts_only_json_tuple_normalization():
    row, visual, floor = example()
    visual['identity'] = (0,0,0)
    runner.compare_original(row, visual, floor, None, 0)


@pytest.mark.parametrize('fault', ['visual', 'floor', 'premature_floor_failure', 'clock', 'lost_field'])
def test_any_undeclared_original_difference_rejects(fault):
    row, visual, floor = example()
    error = None
    if fault == 'visual': visual['nested']['position'][0] += 1e-9
    elif fault == 'floor': floor['correction']['height'] += 1e-9
    elif fault == 'premature_floor_failure': error = 'original rejection'
    elif fault == 'clock': row['tick'] = 1
    elif fault == 'lost_field': del visual['terminal_failure']
    with pytest.raises(ValueError): runner.compare_original(row, visual, floor, error, 0)


def test_terminal_boundary_requires_the_exact_original_floor_rejection():
    frame = runner.FRAMES-1
    row, visual, floor = example(frame)
    row['decision']['failure'] = runner.diagnosis.FAILURE
    runner.compare_original(row, visual, None, runner.diagnosis.FAILURE, frame)
    for wrong_floor, wrong_error in ((floor, runner.diagnosis.FAILURE), (None, None), (None, 'other failure')):
        with pytest.raises(ValueError): runner.compare_original(row, visual, wrong_floor, wrong_error, frame)


def test_floor_failure_remains_an_explicit_negative_result():
    class Rejection:
        def observe(self, *args, **kwargs): raise ValueError('fixed original gate')
    assert runner.floor_observe(Rejection(), None, None, None, dict(terminal_failure=None), 1) == (None, 'fixed original gate')
    assert runner.floor_observe(None, None, None, None, dict(terminal_failure={'cause':'image'}), 1) == (None, 'visual observer terminal')


@pytest.mark.parametrize('resource', ['memory_available_bytes', 'artifact_free_bytes'])
def test_resource_gate_rejects_one_byte_below_requirement(monkeypatch, resource):
    limits = dict(memory_available_bytes=40*1024**3, artifact_free_bytes=40*1024**3+runner.MAX_OUTPUT_BYTES)
    monkeypatch.setattr(runner, 'hardware', lambda: dict(limits))
    assert runner.resources() == limits
    limits[resource] -= 1
    with pytest.raises(ValueError): runner.resources()
