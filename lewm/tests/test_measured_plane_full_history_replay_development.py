"""Full replay orchestration, strict stop on mismatch and closed-output checks."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from scripts import replay_go2_measured_plane_single_pass_full_history_v1 as job
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('weight', torch.zeros(1)); self.eval()

    def forward(self):
        return self.weight


def setup(tmp_path, monkeypatch, fault=None):
    source, output = tmp_path/'source', tmp_path/'output'
    output.mkdir(); source.mkdir(); directory = source/job.original.CASE[0]; directory.mkdir()
    monkeypatch.setattr(job.native, 'OUTPUT', source); monkeypatch.setattr(job, 'OUTPUT', output)
    model_sha = job.original.state_digest(Model().state_dict())
    monkeypatch.setattr(job.inputs.job, 'MODEL_SHA', model_sha)
    models = []
    def model():
        value = Model(); models.append(value); return value
    monkeypatch.setattr(job.original, 'assigned_model', model)
    monkeypatch.setattr(job.original, 'ArticulatedCollisionGeometry', lambda path: 'geometry')
    references, tape = [], []
    for frame in range(5):
        decision = dict(controller='synthetic baseline', requested_command=[0., 0., 0.],
            terminal='SENSOR_OR_MODEL_FAILURE' if frame == 4 else None,
            failure='original preserved failure' if frame == 4 else None,
            new_selection={'prediction': [['complete synthetic forecast']]} if frame == 3 else None,
            nested={'evidence': frame})
        row, command = endpoint(frame, decision); references.append(row)
        if frame < 4: tape.append(command)
    controllers = []
    class Controller:
        def __init__(self, model, geometry, **kwargs):
            self.model = model; self.arm = len(controllers); self.frame = -1; self.hidden = 0
            controllers.append(self)
        def observe(self, policy, depth, fast, **kwargs):
            frame = policy['frame']; self.frame = frame
            if frame == 3 and not (self.arm == 1 and fault == 'calls'): self.model()
            result = deepcopy(references[frame]['decision'])
            if self.arm == 1 and frame == 3:
                if fault == 'decision': result['nested']['evidence'] = 'changed'
                elif fault == 'input': policy['changed'] = True
                elif fault == 'state': self.hidden = 1
                elif fault == 'model': self.model.weight.add_(1)
            return result
    monkeypatch.setattr(job, 'MeasuredPlaneResidualController', Controller)
    monkeypatch.setattr(job, 'MeasuredPlaneSinglePassController', Controller)
    monkeypatch.setattr(job.comparison, 'normalize', lambda decision: deepcopy(decision))
    monkeypatch.setattr(job.comparison, 'observed_state', lambda c: {'frame': c.frame, 'hidden': c.hidden})
    def read_json(root, name):
        if root == directory and name == 'command_tape.json': return deepcopy(tape)
        return json.loads((root/name).read_text())
    monkeypatch.setattr(job.run, 'read_json', read_json)
    def rows(root):
        assert root == directory
        yield from deepcopy(references)
    monkeypatch.setattr(job.run.pipeline, 'read_rows', rows)
    consumed = []
    def packets(root, count):
        assert root == directory and count == 5
        for frame in range(count):
            consumed.append(frame)
            yield ({'frame': frame}, {'image': np.array([frame])}, {}, {}, {}), 1_600_000_000+frame*100_000_000
    monkeypatch.setattr(job, 'packets', packets)
    return dict(frames=5, learned_result_sha256='actual-completed-original'), consumed, models, output


def test_complete_history_including_original_failure_and_final_observation_is_verified(tmp_path, monkeypatch):
    admission, consumed, models, output = setup(tmp_path, monkeypatch)
    report = job.replay(admission)
    assert consumed == list(range(5))
    assert report['frames'] == 5 and report['actual_model_forward_calls'] == [1, 1]
    assert [s['frame'] for s in report['observed_state_checks']] == [0, 3, 4]
    assert all(not m._forward_hooks for m in models)
    job.check_output(report, admission)
    assert consumed == list(range(5))*2
    assert (output/'resource_monitor.jsonl').is_file()


@pytest.mark.parametrize('fault', ['decision', 'input', 'calls', 'state', 'model'])
def test_mismatch_stops_before_following_observation_and_releases_hooks(tmp_path, monkeypatch, fault):
    admission, consumed, models, output = setup(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError): job.replay(admission)
    assert consumed == (list(range(5)) if fault == 'model' else list(range(4)))
    assert all(not m._forward_hooks for m in models)
    assert not (output/'result.json').exists()
    if fault == 'decision':
        mismatch = json.loads((output/'decision_mismatch.json').read_text())
        assert mismatch['frame'] == 3
        assert mismatch['original']['nested']['evidence'] == 3
        assert mismatch['candidate']['nested']['evidence'] == 'changed'


@pytest.mark.parametrize('fault', ['missing', 'extra', 'packet', 'original', 'clock', 'report', 'state'])
def test_complete_saved_population_and_reconstructed_raw_packets_cannot_be_forged(tmp_path, monkeypatch, fault):
    admission, _, _, output = setup(tmp_path, monkeypatch)
    report = job.replay(admission)
    path = output/'comparison.jsonl'; rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == 'missing': rows.pop()
    elif fault == 'extra': rows.append(deepcopy(rows[-1]))
    elif fault == 'packet': rows[3]['public_packet_sha256'] = 'changed'
    elif fault == 'original': rows[3]['original_decision_sha256'] = 'changed'
    elif fault == 'clock': rows[3]['observation_now_ns'] += 1
    elif fault == 'report': report['timing']['all_observations']['candidate_total_s'] = 0
    elif fault == 'state':
        states = json.loads((output/'state_checks.json').read_text()); states.pop()
        (output/'state_checks.json').write_text(json.dumps(states))
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError): job.check_output(report, admission)
