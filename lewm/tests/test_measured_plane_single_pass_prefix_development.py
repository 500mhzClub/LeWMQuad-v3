"""Complete prefix accounting and rejection of causal/input/decision failures."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import replay_go2_measured_plane_single_pass_prefix_v1 as job


def population():
    rows = [dict(frame=i, execution_order=[0, 1] if i % 2 == 0 else [1, 0],
        complete_reference_decision_equal=True, complete_normalized_candidate_equal=True,
        public_inputs_unchanged=True, forecast_compared=i >= 3,
        reference_decision_sha256='a'*64, baseline_decision_sha256='a'*64,
        baseline_controller_s=.5, candidate_controller_s=.4) for i in range(job.FRAMES)]
    states = [dict(frame=i, state_sha256='b'*64, retained_observed_state_equal=True) for i in job.STATE_FRAMES]
    return rows, states


def test_complete_serialized_population_and_timing_scope():
    rows, states = json.loads(json.dumps(population()))
    report = job.summarize(rows, states)
    assert report['frames'] == 123 and report['raw_model_forecast_comparisons'] == 120
    assert report['timing']['baseline_total_s'] == 60
    assert report['timing']['candidate_total_s'] == 48
    assert not report['following_changed_command_observation_consumed']
    assert not report['real_time_qualified'] and not report['navigation_qualified']


@pytest.mark.parametrize('fault', ['short', 'order', 'execution_order', 'decision', 'input',
    'forecast', 'hash', 'nan', 'zero', 'bool', 'state_short', 'state_false'])
def test_complete_population_rejects_faults(fault):
    rows, states = population()
    if fault == 'short': rows.pop()
    elif fault == 'order': rows[-1]['frame'] -= 1
    elif fault == 'execution_order': rows[-1]['execution_order'].reverse()
    elif fault == 'decision': rows[-1]['complete_normalized_candidate_equal'] = False
    elif fault == 'input': rows[-1]['public_inputs_unchanged'] = False
    elif fault == 'forecast': rows[-1]['forecast_compared'] = False
    elif fault == 'hash': rows[-1]['baseline_decision_sha256'] = 'c'*64
    elif fault == 'nan': rows[-1]['candidate_controller_s'] = float('nan')
    elif fault == 'zero': rows[-1]['candidate_controller_s'] = 0
    elif fault == 'bool': rows[-1]['candidate_controller_s'] = True
    elif fault == 'state_short': states.pop()
    elif fault == 'state_false': states[-1]['retained_observed_state_equal'] = False
    with pytest.raises(ValueError): job.summarize(rows, states)


@pytest.mark.parametrize('fault', [None, 'extra', 'decision', 'input', 'state'])
def test_paired_loop_complete_reference_and_physical_boundary(monkeypatch, tmp_path, fault):
    real = job.run
    references, accessed = [], []
    def packet(i):
        accessed.append(i)
        return {'frame': i}, {}, {}, 1_500_000_000+i*100_000_000
    for i in range(job.FRAMES+(fault == 'extra')):
        decision = dict(terminal=None, requested_command=[0., 0., -.45] if i == 122 else [0., 0., 0.],
            new_selection={'prediction': 'synthetic'} if i >= 3 else None)
        public = ({'frame': i}, {}, {}, {}, {})
        references.append(dict(tick=i, decision=decision,
            original={'requested_command': [0., 0., 0.]}, public_packet_sha256=real.fingerprint(public)))
    class Controller:
        def __init__(self, model, geometry, **kwargs): self.n = -1
        def observe(self, p, *args, **kwargs):
            self.n = p['frame']
            return deepcopy(references[self.n]['decision'])
    class Candidate(Controller):
        def observe(self, p, *args, **kwargs):
            result = super().observe(p, *args, **kwargs)
            if self.n == 122:
                if fault == 'decision': result['requested_command'] = [1., 0., 0.]
                elif fault == 'input': p['mutated'] = True
                elif fault == 'state': self.n += 1
            return result
    model_factory = lambda *args: torch.nn.Linear(1, 1).eval()
    fake_native = SimpleNamespace(OUTPUT=tmp_path, CASE=('case', 2, 'no_rgb', 'direct'), assigned_model=model_factory)
    tape = [dict(requested_command=[0., 0., 0.]) for _ in references]
    original = SimpleNamespace(native=fake_native, MODEL_SHA='fixture', state_digest=lambda s: 'fixture',
        public_mission=lambda i: {}, ArticulatedCollisionGeometry=lambda path: None, URDF=None,
        command_endpoint=lambda row, command, frame: None)
    pipeline = SimpleNamespace(ExtendedBudgetRGBDReplay=lambda directory: SimpleNamespace(packet=packet),
        rgb_packet=lambda *args, **kwargs: ({}, {}), read_rows=lambda directory: iter(references))
    # list_iterator is intentionally wrapped in a closable iterator like the
    # actual compressed stream, so boundary failures close it too.
    def stream(directory):
        yield from references
    pipeline.read_rows = stream
    monkeypatch.setattr(job, 'OUTPUT', tmp_path)
    monkeypatch.setattr(job, 'job', original)
    monkeypatch.setattr(original, 'OUTPUT', tmp_path, raising=False)
    monkeypatch.setattr(job, 'run', SimpleNamespace(pipeline=pipeline,
        read_json=lambda root, name: tape if name == 'command_tape.json' else [{}]*len(references),
        fingerprint=real.fingerprint, canonical=real.canonical, public_acquisition=lambda x: x,
        write_json=lambda path, value: path.write_text(json.dumps(value))))
    monkeypatch.setattr(job, 'MeasuredPlaneResidualController', Controller)
    monkeypatch.setattr(job, 'MeasuredPlaneSinglePassController', Candidate)
    monkeypatch.setattr(job, 'normalize', lambda d: d)
    monkeypatch.setattr(job, 'state', lambda c: {'observed_frame': c.n})
    if fault is None:
        report = job.replay()
        assert report['frames'] == 123 and accessed == list(range(123))
        job.check_output(report)
        path = tmp_path/'comparison.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['public_packet_sha256'] = 'changed'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with pytest.raises(ValueError): job.check_output(report)
    else:
        with pytest.raises(ValueError): job.replay()
        assert 123 not in accessed
