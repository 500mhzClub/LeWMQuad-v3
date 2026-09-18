"""The nominal branch cannot consume model forecasts or invent their provenance."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from lewm import forecast_source_selection_development as source
from lewm.training_translation_bias_development import TrainingTranslationBiasModel
from lewm.tests.test_requested_twist_forecast_bank_development import inputs
from lewm.requested_twist_forecast_bank_development import forecast_bank


def history():
    return {key: torch.ones(shape, dtype=torch.float32) for key, shape in {
        'rgb': (4, 3, 96, 128), 'body': (4, 20, 63), 'control': (4, 15, 7)}.items()}


class FixedModel(TrainingTranslationBiasModel):
    """Synthetic interface fixture, never an admitted trained checkpoint."""
    def __init__(self, prediction=None):
        nn.Module.__init__(self)
        self.corrected_heads = ('rollout_outcomes',)
        if prediction is None:
            prediction = forecast_bank(*inputs())['nominal_outcomes']
        self.register_buffer('fixed_prediction', torch.as_tensor(prediction).clone().float())
        self.register_buffer('rollout_outcomes_xy_bias', torch.full((8, 2), .123))
        self.calls = []
        self.eval()

    def forward(self, **kwargs):
        self.calls.append({key: value.clone() for key, value in kwargs['observation_history'].items()})
        return dict(rollout_outcomes=self.fixed_prediction.clone(),
            prediction_valid=torch.ones((6, 8), dtype=torch.bool),
            target_offsets_ns=torch.arange(1, 9, dtype=torch.int64).mul(100_000_000).expand(6, 8))


def options():
    return dict(head='rollout_outcomes', input_variant='full', goal_body_xy_m=[1., 0.], contact_penalty_m=1.2)


@pytest.mark.parametrize('variant', ['full', 'no_rgb'])
def test_learned_mode_is_the_original_call_and_cost_with_added_provenance(variant):
    model = FixedModel(); h = history(); kwargs = options() | {'input_variant': variant}
    expected = source.learned_select(model, h, **kwargs)
    actual = source.select(model, h, forecast_source='frozen_world_model', **kwargs)
    provenance = actual.pop('forecast_provenance')
    expected.pop('selection_wall_ms'); actual.pop('selection_wall_ms')
    assert actual == expected and len(model.calls) == 2
    assert provenance['learned_forecasts_used'] and provenance['frozen_model_forward_called']
    assert provenance['observation_history_supplied_to_forecaster']
    assert not provenance['model_checkpoint_admitted_by_this_helper']
    assert bool(model.calls[-1]['rgb'].any()) == (variant == 'full')


@pytest.mark.parametrize('variant', ['full', 'no_rgb'])
def test_nominal_mode_ignores_model_outputs_bias_and_history_values(variant):
    model = FixedModel(); h = history(); before = {k: v.clone() for k, v in model.state_dict().items()}
    result = source.select(model, h, forecast_source='nominal_requested_twist',
        **(options() | {'input_variant': variant}))
    expected = forecast_bank(*inputs())['nominal_outcomes'].numpy()
    np.testing.assert_array_equal(result['prediction'], expected)
    assert not model.calls and all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert result['head'] == 'nominal_requested_twist'
    assert not result['model_prediction_corrected'] and not result['translation_bias_training_only']
    assert result['translation_bias_xy_m'] is None
    assert result['forecast_provenance']['assigned_frozen_model_head'] == 'rollout_outcomes'
    assert not result['forecast_provenance']['observation_history_supplied_to_forecaster']
    assert result['forecast_provenance']['nominal_forecast_assumptions']['requested_velocity_tracking_assumed']
    for value in h.values(): value.mul_(.25)
    model.fixed_prediction.fill_(999.)
    later = source.select(model, h, forecast_source='nominal_requested_twist', **options())
    assert later['prediction'] == result['prediction'] and not model.calls


def test_intervention_can_change_selected_action_under_the_same_cost_and_actions():
    model = FixedModel()
    model.fixed_prediction[1:, :, :2] = -.1
    learned = source.select(model, history(), forecast_source='frozen_world_model', **options())
    nominal = source.select(model, history(), forecast_source='nominal_requested_twist', **options())
    assert learned['action'] == 'hold' and nominal['action'] == 'forward'
    for key in ('score_contract', 'contact_penalty_m', 'goal_body_xy_m', 'target_offsets_ns'):
        assert learned[key] == nominal[key]
    assert [c['action'] for c in learned['candidates']] == [c['action'] for c in nominal['candidates']]


@pytest.mark.parametrize('mode', source.SOURCES)
@pytest.mark.parametrize('fault', ['model', 'head', 'variant', 'extra_history', 'nan', 'dtype'])
def test_both_modes_retain_input_and_assigned_model_contracts(mode, fault):
    model = FixedModel(); h = history(); kwargs = options()
    if fault == 'model': model = object()
    elif fault == 'head': kwargs['head'] = 'missing'
    elif fault == 'variant': kwargs['input_variant'] = 'no_candidate_command'
    elif fault == 'extra_history': h['future_rgb'] = h['rgb'].clone()
    elif fault == 'nan': h['rgb'][0, 0, 0, 0] = float('nan')
    elif fault == 'dtype': h['rgb'] = h['rgb'].double()
    with pytest.raises(ValueError): source.select(model, h, forecast_source=mode, **kwargs)


@pytest.mark.parametrize('value', [None, True, 'reactive', 'rollout_off', []])
def test_ambiguous_source_names_rejected(value):
    with pytest.raises(ValueError): source.require_source(value)


def test_actual_small_neural_model_is_unchanged_by_both_modes():
    from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
    from lewm.training_translation_bias_development import fit_translation_bias
    from lewm.tests.test_training_translation_bias_development import fixture
    from lewm.tests.test_observation_horizon_goal_selection_development import history as real_history
    rows, arrays, schedule = fixture()
    correction = fit_translation_bias(rows, arrays, schedule, head='direct_outcomes')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026091132)
        model = TrainingTranslationBiasModel(ObservationHorizonRGBBodyJEPA(8).eval(),
            {'direct_outcomes': correction})
    before = deepcopy(model.state_dict()); h = real_history(); h_before = {k: v.clone() for k, v in h.items()}
    for mode in source.SOURCES:
        result = source.select(model, h, forecast_source=mode, **(options() | {'head': 'direct_outcomes'}))
        assert len(result['prediction']) == 6
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(torch.equal(v, h_before[k]) for k, v in h.items())
    assert all(parameter.grad is None for parameter in model.parameters())
