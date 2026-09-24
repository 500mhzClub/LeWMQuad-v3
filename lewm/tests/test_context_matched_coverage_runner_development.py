import copy

import numpy as np
import pytest
import torch

from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA
from lewm.temporal_rgb_body_learning_development import active_parameters, training_loss
from lewm.tests.test_temporal_rgb_body_jepa_development import batch
from scripts.run_go2_context_matched_coverage_learning_development_v1 import (
    predictions, source_bindings, comparisons, SEEDS, DATA_CONDITIONS, CONDITIONS)


def example():
    data = batch()
    data['metadata'] = [{'layout_id': layout, 'offset_ns': 1_000_000_000,
                         'action_index': 2, 'prefix_action_index': 1,
                         'coverage_source': 'switch', 'data_role': 'train'} for layout in ('a', 'b')]
    return data


@pytest.mark.parametrize('condition', CONDITIONS)
def test_actual_model_replay_and_nonmutating_past_action_matched_shuffle(condition):
    data = example()
    saved = copy.deepcopy(data)
    model = TemporalRGBBodyJEPA(16).eval()
    p, z, eligible = predictions(model, data, condition)
    with torch.no_grad():
        full = model(data['observation_history'], data['known_action_blocks'], data['known_action_valid'])
    for head, values in p.items():
        np.testing.assert_allclose(values, full[head + '_outcomes'].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(z, full['latent'].numpy(), rtol=0, atol=1e-6)
    assert eligible.all()
    shuffled, _, _ = predictions(model, data, condition, 'rgb_shuffle')
    assert not np.array_equal(shuffled['direct'], p['direct'])
    for key in data['observation_history']:
        assert torch.equal(data['observation_history'][key], saved['observation_history'][key])


@pytest.mark.parametrize('condition', CONDITIONS)
def test_all_objectives_accept_new_switch_metadata_without_prediction_label_leak(condition):
    data = example()
    model = TemporalRGBBodyJEPA(16)
    loss, _ = training_loss(model, data, condition)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in active_parameters(model, condition))
    model.eval()
    original, _, _ = predictions(model, data, condition)
    data['targets']['motion'][:] = 999
    data['targets']['contact'][:] = 1
    changed, _, _ = predictions(model, data, condition)
    for head in original:
        assert np.array_equal(original[head], changed[head])


def test_recursive_source_closure_includes_new_and_frozen_learning_and_fixture_paths():
    source = source_bindings()
    for path in ('lewm/context_matched_coverage_development.py',
                 'lewm/coverage_prediction_metrics_development.py',
                 'lewm/moving_prefix_learning_data_development.py',
                 'lewm/temporal_rgb_body_learning_development.py',
                 'lewm/causal_sensor_state.py',
                 'lewm/tests/test_temporal_prediction_metrics_development.py'):
        assert path in source and len(source[path]) == 64
    assert all('sealed' not in path.split('/') for path in source)


def test_comparisons_keep_all_fixed_models_seed_and_layout_pairs():
    runs = []
    strata = ('moving_switch', 'moving_continuation', 'old_later', 'old_initial')
    horizons = ('first_half_second', 'three_seconds', 'all_known')
    for seed in SEEDS:
        for data in DATA_CONDITIONS:
            for condition in CONDITIONS:
                value = (2 if data == 'expanded' else 0) + CONDITIONS.index(condition)
                rows = [{'layout_id': f'layout-{i}', 'position_error_m': value,
                         'contact_brier': value, 'regret': value, 'contact': value,
                         'realized_cost': value} for i in range(8)]
                heads = ('direct',) if condition == 'direct' else ('direct', 'rollout')
                runs.append({'seed': seed, 'data_condition': data, 'condition': condition,
                             'validation': {'intact': {head: {s: {h: {'layouts': rows} for h in horizons} for s in strata} for head in heads}},
                             'moving_decisions': {head: {h: {'layouts': rows} for h in horizons[:2]} for head in heads}})
    result = comparisons(runs)
    assert len(result) == 15
    delta = result['expanded/jepa/rollout_minus_coverage_limited/jepa/rollout']
    assert delta['moving_switch/first_half_second/contact_brier']['mean_delta'] == 2
    assert delta['choice/three_seconds/regret']['per_seed_mean_delta'] == [2, 2, 2]
    with pytest.raises(ValueError, match='all fixed models'):
        comparisons(runs[:-1])
