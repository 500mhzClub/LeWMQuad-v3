"""Synthetic matched scheduling and causal streamed inference, not model evidence."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer
from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.independent_pulse_study_runner_development import train_schedule, predict_heads
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan, validate_timed_plan
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.tests.test_independent_pulse_evaluation_development import fixture


class SyntheticStream:
    def __init__(self):
        inv, w, t, r = fixture()
        self.evaluation = IndependentPulseEvaluation(inv, PulseTimedDataset(w, t, r))
        self.calls = []

    def inference_batch(self, indices, *, role):
        self.calls.append(('inference', list(indices), role)); n = len(indices)
        w = self.evaluation.dataset.windows
        plans = [pulse_brake_plan(tuple(w[i]['command']), w[i]['pulse_ticks']) for i in indices]
        history = dict(rgb=torch.zeros(n, 4, 3, 96, 128), body=torch.zeros(n, 4, 20, 63), control=torch.zeros(n, 4, 15, 7))
        for j, i in enumerate(indices): history['rgb'][j].fill_(i / 100.)
        return dict(observation_history=history, known_action_blocks=torch.stack([p[0] for p in plans]),
            known_action_valid=torch.stack([p[1] for p in plans]))

    def training_batch(self, indices):
        inputs = self.inference_batch(indices, role='train'); self.calls[-1] = ('training', list(indices), 'train')
        active, offsets = validate_timed_plan(inputs['known_action_blocks'], inputs['known_action_valid'], len(indices))
        d = self.evaluation.dataset
        targets = {k: torch.stack([d._targets[i][k].clone() for i in indices])
            for k in ('motion', 'motion_valid', 'contact', 'contact_valid')}
        targets.update(target_offsets_ns=offsets, future_valid=active.clone(),
            future_observations={k: v[:, :1].expand(-1, 8, *v.shape[2:]).clone()
                for k, v in inputs['observation_history'].items()})
        return dict(inputs=inputs, targets=targets)


@pytest.mark.parametrize('condition', ['direct', 'supervised_rollout', 'jepa'])
def test_one_fresh_matched_synthetic_step_preserves_exact_exposure_and_accounting(condition):
    s = SyntheticStream(); trainer = CumulativePulseTrainer(condition, seed=37, latent_dim=8)
    schedule = s.evaluation.dataset.schedule('train', updates=1, batch_size=2, seed=71)
    records = []; before = trainer.initial_sha256
    result = train_schedule(trainer, s, schedule, on_update=records.append)
    assert result['updates'] == 1 and trainer.updates == 1 and len(records) == 1
    assert result['initial_sha256'] == before and result['model_sha256'] != before
    assert records[0]['sample_indices'] == schedule['batches'][0] == result['training_draw_indices']
    assert s.calls == [('training', schedule['batches'][0], 'train')]
    assert not result['inference_performed'] and not result['checkpoint_selection_performed']


@pytest.mark.parametrize('fault', ['role', 'draw', 'digest', 'oversize', 'actions', 'identity'])
def test_schedule_changes_latch_before_any_step_or_materialization(fault):
    s = SyntheticStream(); trainer = CumulativePulseTrainer('jepa', seed=7, latent_dim=8)
    schedule = s.evaluation.dataset.schedule('train', updates=1, batch_size=2, seed=71)
    if fault == 'role': schedule['role'] = 'selection'
    elif fault == 'draw': schedule['batches'][0][0] = int(s.evaluation.arrays('selection')['indices'][0])
    elif fault == 'digest': schedule['schedule_sha256'] = '0' * 64
    elif fault == 'oversize': schedule['batch_size'] = 17
    elif fault == 'actions': schedule = s.evaluation.dataset.schedule('train', updates=1, batch_size=2, seed=71, require_all_actions=False)
    else: schedule['episode_roles'] = {}
    with pytest.raises(ValueError): train_schedule(trainer, s, schedule, on_update=lambda r: None)
    assert trainer.failed and trainer.updates == 0 and not s.calls
    assert state_digest(trainer.model.state_dict()) == trainer.initial_sha256


def test_callback_failure_preserves_completed_update_and_prevents_resume():
    s = SyntheticStream(); trainer = CumulativePulseTrainer('direct', seed=7, latent_dim=8)
    schedule = s.evaluation.dataset.schedule('train', updates=2, batch_size=2, seed=71)
    def fail(record): raise OSError('synthetic accounting failure')
    with pytest.raises(OSError): train_schedule(trainer, s, schedule, on_update=fail)
    assert trainer.failed and trainer.updates == 1 and len(s.calls) == 1
    with pytest.raises(ValueError, match='fresh'): train_schedule(trainer, s, schedule, on_update=lambda r: None)


def test_external_schedule_and_record_mutation_cannot_change_future_draws():
    s = SyntheticStream(); trainer = CumulativePulseTrainer('direct', seed=7, latent_dim=8)
    schedule = s.evaluation.dataset.schedule('train', updates=2, batch_size=2, seed=71)
    original = deepcopy(schedule)
    def callback(record):
        record['update'] = 999; record['sample_indices'].clear(); schedule['batches'][1] = [999]
    result = train_schedule(trainer, s, schedule, on_update=callback)
    assert [ids for _, ids, _ in s.calls] == original['batches']
    assert result['training_draw_indices'] == [i for ids in original['batches'] for i in ids]


def test_missing_whole_training_layout_cannot_be_silently_omitted():
    s = SyntheticStream(); view = s.evaluation; d = view.dataset
    omitted = d.episode_roles[d.windows[0]['condition']]['layout_id']
    keep = [i for i, w in enumerate(d.windows) if d.episode_roles[w['condition']]['layout_id'] != omitted]
    inv, windows, targets, roles = fixture(); names = {windows[i]['condition'] for i in keep}
    s.evaluation = IndependentPulseEvaluation(inv, PulseTimedDataset([windows[i] for i in keep],
        [targets[i] for i in keep], {k: v for k, v in roles.items() if k in names}))
    schedule = s.evaluation.dataset.schedule('train', updates=1, batch_size=2, seed=71)
    trainer = CumulativePulseTrainer('jepa', seed=7, latent_dim=8)
    with pytest.raises(ValueError, match='absent'): train_schedule(trainer, s, schedule, on_update=lambda r: None)
    assert trainer.failed and trainer.updates == 0 and not s.calls


@pytest.mark.parametrize('condition', ['direct', 'supervised_rollout', 'jepa'])
def test_streamed_predictions_have_exact_indices_clocks_primary_head_and_no_updates(condition):
    s = SyntheticStream(); trainer = CumulativePulseTrainer(condition, seed=7, latent_dim=8)
    before = state_digest(trainer.model.state_dict()); trainer.model.train()
    result = predict_heads(trainer, s, role='development_eval', batch_size=2)
    ids = s.evaluation.arrays('development_eval')['indices']
    assert [i for kind, block, role in s.calls for i in block] == ids.tolist()
    assert all(kind == 'inference' and role == 'development_eval' and len(block) <= 2 for kind, block, role in s.calls)
    assert result['primary_head'] == ('direct_outcomes' if condition == 'direct' else 'rollout_outcomes')
    assert len(result['heads']) == (1 if condition == 'direct' else 2)
    assert trainer.updates == 0 and trainer.model.training and state_digest(trainer.model.state_dict()) == before
    for entry in result['heads'].values():
        np.testing.assert_array_equal(entry['indices'], ids)
        assert entry['prediction'].shape == (len(ids), 8, 5)
        assert np.all(np.diff(entry['prediction'][:, :5, 4], axis=1) >= 0)
    scored = s.evaluation.compare(result['heads'], role='development_eval')
    assert scored['all_requested_heads_comparable'] and scored['independent_layout_units'] == 3


def test_batch_partition_does_not_change_row_binding_or_scores_beyond_float_roundoff():
    s = SyntheticStream(); trainer = CumulativePulseTrainer('jepa', seed=7, latent_dim=8)
    a = predict_heads(trainer, s, role='selection', batch_size=1)
    b = predict_heads(trainer, s, role='selection', batch_size=3)
    for key in a['heads']:
        np.testing.assert_array_equal(a['heads'][key]['indices'], b['heads'][key]['indices'])
        np.testing.assert_allclose(a['heads'][key]['prediction'], b['heads'][key]['prediction'], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('fault', ['target_input', 'plan', 'mask', 'clock', 'nonfinite', 'shape', 'state_mutation'])
def test_inference_contract_failure_returns_no_partial_scores_and_latches(monkeypatch, fault):
    s = SyntheticStream(); trainer = CumulativePulseTrainer('jepa', seed=7, latent_dim=8); trainer.model.eval()
    original_inputs = s.inference_batch; forward = trainer.model.forward
    def inputs(ids, *, role):
        data = original_inputs(ids, role=role)
        if fault == 'target_input': data['targets'] = object()
        elif fault == 'plan': data['known_action_blocks'][0, 0, 0, 0] += .1
        return data
    monkeypatch.setattr(s, 'inference_batch', inputs)
    def broken(**data):
        result = forward(**data)
        if fault == 'mask': result['prediction_valid'][0, 0] = False
        elif fault == 'clock': result['target_offsets_ns'][0, 4] += 1
        elif fault == 'nonfinite': result['rollout_outcomes'][0, 0, 0] = float('nan')
        elif fault == 'shape': result['rollout_outcomes'] = result['rollout_outcomes'][:, :7]
        elif fault == 'state_mutation': next(trainer.model.parameters()).add_(.001)
        return result
    monkeypatch.setattr(trainer.model, 'forward', broken)
    with pytest.raises(ValueError): predict_heads(trainer, s, role='selection', batch_size=2)
    assert trainer.failed and not trainer.model.training


@pytest.mark.parametrize('size', [True, 0, 17])
def test_inference_batch_bound_rejects_before_any_read(size):
    s = SyntheticStream(); trainer = CumulativePulseTrainer('direct', seed=7, latent_dim=8)
    with pytest.raises(ValueError): predict_heads(trainer, s, role='selection', batch_size=size)
    assert not s.calls
