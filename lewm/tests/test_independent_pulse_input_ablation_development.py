"""Synthetic information-dependency checks; no recorded fits or navigation."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer, training_loss
from lewm.independent_pulse_input_ablation_development import (
    VARIANTS, transform_inputs, transform_training_batch)
from lewm.independent_pulse_study_runner_development import train_schedule, predict_heads
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan
from lewm.tests.test_independent_pulse_study_runner_development import SyntheticStream


def equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    else: assert a == b


def batch():
    data = SyntheticStream().training_batch([0, 1])
    for source in [data['inputs']['observation_history'], data['targets']['future_observations']]:
        for key, value in source.items():
            # Distinct row, packet and internal sensor history values.
            value.copy_(torch.arange(value.numel()).reshape_as(value).remainder(97) / 97.)
            for packet in range(value.shape[1]): value[:, packet].add_(packet / 100.)
    return data


@pytest.mark.parametrize('variant', VARIANTS)
def test_treatment_preserves_input_source_labels_clocks_censoring_and_shape(variant):
    raw = batch(); original = deepcopy(raw)
    result = transform_training_batch(raw, input_variant=variant)
    equal(raw, original)
    equal(result['inputs'], transform_inputs(raw['inputs'], input_variant=variant))
    for key in raw['targets']:
        if key != 'future_observations': equal(result['targets'][key], raw['targets'][key])
    for key in ('body', 'control'):
        equal(result['targets']['future_observations'][key], raw['targets']['future_observations'][key])
    equal(result['inputs']['known_action_valid'], raw['inputs']['known_action_valid'])
    old = validate_timed_plan(raw['inputs']['known_action_blocks'], raw['inputs']['known_action_valid'], 2)
    new = validate_timed_plan(result['inputs']['known_action_blocks'], result['inputs']['known_action_valid'], 2)
    for a, b in zip(old, new, strict=True): equal(a, b)
    if variant == 'full': equal(raw, result)
    elif variant == 'no_rgb':
        assert not result['inputs']['observation_history']['rgb'].any()
        assert not result['targets']['future_observations']['rgb'].any()
        for key in ('body', 'control'):
            equal(result['inputs']['observation_history'][key], raw['inputs']['observation_history'][key])
    elif variant == 'latest_packet_only':
        for key, value in result['inputs']['observation_history'].items():
            for packet in range(4): equal(value[:, packet], raw['inputs']['observation_history'][key][:, -1])
        equal(result['targets'], raw['targets'])
    else:
        assert not result['inputs']['known_action_blocks'].any()
        equal(result['inputs']['observation_history'], raw['inputs']['observation_history'])
        equal(result['targets'], raw['targets'])


@pytest.mark.parametrize('condition', ['direct', 'supervised_rollout', 'jepa'])
@pytest.mark.parametrize('variant', VARIANTS)
def test_all_objectives_train_and_predict_with_identical_named_treatment(monkeypatch, condition, variant):
    s = SyntheticStream(); trainer = CumulativePulseTrainer(condition, seed=17, latent_dim=8)
    schedule = s.evaluation.dataset.schedule('train', updates=1, batch_size=2, seed=71)
    expected = transform_training_batch(s.training_batch(schedule['batches'][0]), input_variant=variant)
    step = trainer.step; training_seen = []
    def capture(data):
        equal(data, expected); training_seen.append(True)
        return step(data)
    monkeypatch.setattr(trainer, 'step', capture)
    records = []
    fit = train_schedule(trainer, s, schedule, on_update=records.append, input_variant=variant)
    assert training_seen == [True] and trainer.updates == 1 and not trainer.failed
    assert fit['input_variant'] == records[0]['input_variant'] == variant
    assert all(torch.isfinite(p.grad).all() for p in trainer.parameters)
    forward = trainer.model.forward; seen = []
    def capture_forward(**inputs):
        if variant == 'no_rgb': assert not inputs['observation_history']['rgb'].any()
        elif variant == 'latest_packet_only':
            for value in inputs['observation_history'].values():
                equal(value, value[:, -1:].expand_as(value))
        elif variant == 'no_candidate_command': assert not inputs['known_action_blocks'].any()
        seen.append(True)
        return forward(**inputs)
    monkeypatch.setattr(trainer.model, 'forward', capture_forward)
    prediction = predict_heads(trainer, s, role='development_eval', batch_size=2, input_variant=variant)
    assert prediction['input_variant'] == variant and seen
    assert prediction['model_sha256'] == fit['model_sha256']
    assert trainer.updates == 1
    for head in prediction['heads'].values():
        np.testing.assert_array_equal(head['indices'], s.evaluation.arrays('development_eval')['indices'])
    assert s.evaluation.compare(prediction['heads'], role='development_eval')['all_requested_heads_comparable']


@pytest.mark.parametrize('condition', ['direct', 'supervised_rollout', 'jepa'])
@pytest.mark.parametrize('variant', ['no_rgb', 'latest_packet_only', 'no_candidate_command'])
def test_removed_information_cannot_change_loss_or_parameter_gradients(condition, variant):
    raw = batch(); changed = deepcopy(raw)
    if variant == 'no_rgb':
        changed['inputs']['observation_history']['rgb'].add_(.25)
        changed['targets']['future_observations']['rgb'].add_(.5)
    elif variant == 'latest_packet_only':
        for value in changed['inputs']['observation_history'].values(): value[:, :3].add_(.25)
    else:
        changed['inputs']['known_action_blocks'].neg_()
    trainer = CumulativePulseTrainer(condition, seed=17, latent_dim=8)
    records = []
    for data in (raw, changed):
        trainer.optimizer.zero_grad(set_to_none=True)
        loss, parts = training_loss(trainer.model, transform_training_batch(data, input_variant=variant), condition)
        loss.backward()
        records.append((loss.detach().clone(), parts, [p.grad.clone() for p in trainer.parameters]))
    equal(records[0][0], records[1][0]); equal(records[0][1], records[1][1])
    for a, b in zip(records[0][2], records[1][2], strict=True): equal(a, b)
    assert all(p.grad is None for p in trainer.model.target_encoder.parameters())
    # Positive control: the invariance is caused by the transform, not a model
    # which already ignores this information on the synthetic example.
    original_loss, _ = training_loss(trainer.model, raw, condition)
    changed_loss, _ = training_loss(trainer.model, changed, condition)
    assert original_loss.item() != changed_loss.item()


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('phase', ['training', 'inference'])
def test_original_plan_corruption_is_rejected_even_when_values_will_be_removed(monkeypatch, variant, phase):
    s = SyntheticStream(); trainer = CumulativePulseTrainer('jepa', seed=17, latent_dim=8)
    original = s.inference_batch
    def broken(ids, *, role):
        inputs = original(ids, role=role)
        inputs['known_action_blocks'][0, 0, 0, 0] += .1
        return inputs
    monkeypatch.setattr(s, 'inference_batch', broken)
    with pytest.raises(ValueError, match='exact prescribed'):
        if phase == 'training':
            schedule = s.evaluation.dataset.schedule('train', updates=1, batch_size=2, seed=71)
            train_schedule(trainer, s, schedule, on_update=lambda r: None, input_variant=variant)
        else: predict_heads(trainer, s, role='selection', batch_size=2, input_variant=variant)
    assert trainer.failed and trainer.updates == 0


@pytest.mark.parametrize('variant', [None, True, 'no_history', 'no_action', 'no_rgb_typo'])
@pytest.mark.parametrize('phase', ['training', 'inference'])
def test_unknown_variant_rejected_before_materialization(variant, phase):
    s = SyntheticStream(); trainer = CumulativePulseTrainer('direct', seed=17, latent_dim=8)
    with pytest.raises(ValueError, match='variant'):
        if phase == 'training':
            schedule = s.evaluation.dataset.schedule('train', updates=1, batch_size=2, seed=71)
            train_schedule(trainer, s, schedule, on_update=lambda r: None, input_variant=variant)
        else: predict_heads(trainer, s, role='selection', batch_size=2, input_variant=variant)
    assert trainer.failed and trainer.updates == 0 and not s.calls


@pytest.mark.parametrize('field', ['past_rgb', 'old_body', 'future_rgb', 'action'])
def test_ablation_cannot_hide_nonfinite_available_data(field):
    data = batch()
    if field == 'past_rgb': data['inputs']['observation_history']['rgb'][0, 0, 0, 0, 0] = float('nan')
    elif field == 'old_body': data['inputs']['observation_history']['body'][0, 0, 0, 0] = float('nan')
    elif field == 'future_rgb': data['targets']['future_observations']['rgb'][0, 0, 0, 0, 0] = float('nan')
    else: data['inputs']['known_action_blocks'][0, 0, 0, 0] = float('nan')
    variant = {'past_rgb': 'no_rgb', 'old_body': 'latest_packet_only', 'future_rgb': 'no_rgb', 'action': 'no_candidate_command'}[field]
    with pytest.raises(ValueError): transform_training_batch(data, input_variant=variant)


def test_unavailable_future_nan_padding_stays_unavailable_without_becoming_supervision():
    data = batch(); valid = data['targets']['future_valid']; valid[0, 0] = False
    for value in data['targets']['future_observations'].values(): value[~valid] = float('nan')
    for variant in VARIANTS:
        result = transform_training_batch(data, input_variant=variant)
        equal(result['targets']['future_valid'], valid)
        assert not result['targets']['future_valid'][0, 0]
        if variant == 'no_rgb': assert not result['targets']['future_observations']['rgb'].any()
