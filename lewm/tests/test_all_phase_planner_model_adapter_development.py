import numpy as np
import pytest
import torch

from lewm.all_phase_translation_bias_development import AllPhaseTranslationBiasModel, fit_translation_bias
from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel
from lewm.training_translation_bias_development import TrainingTranslationBiasModel
from lewm.training_bias_predictive_selection_development import select
from lewm.observation_horizon_predictive_selection_development import candidate_inputs
from lewm.observation_horizon_input_ablation_development import transform_inputs
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.tests.test_all_phase_translation_bias_development import data
from lewm.tests.test_observation_horizon_goal_selection_development import history


def model(heads):
    rows, arrays, schedule = data(); records = {}
    for head in heads:
        arrays[head] = arrays['direct_outcomes'].copy()
        records[head] = fit_translation_bias(rows, arrays, schedule, head=head)
    return AllPhaseTranslationBiasModel(ObservationHorizonRGBBodyJEPA(8).eval(), records)


@pytest.mark.parametrize('heads', [('direct_outcomes',), ('direct_outcomes', 'rollout_outcomes')])
@pytest.mark.parametrize('variant', ['full', 'no_rgb'])
def test_real_existing_selector_accepts_adapter_and_preserves_all_forecast_tensors(heads, variant):
    source = model(heads); adapted = AllPhasePlannerModel(source); observations = history()
    kwargs = dict(head=heads[-1], input_variant=variant, goal_body_xy_m=[1., 0.], contact_penalty_m=1.2)
    with pytest.raises(ValueError, match='wrapper'): select(source, observations, **kwargs)
    assert isinstance(adapted, TrainingTranslationBiasModel)
    assert AllPhasePlannerModel.forward is AllPhaseTranslationBiasModel.forward
    before = state_digest(source.state_dict()); inputs = transform_inputs(candidate_inputs(observations), input_variant=variant)
    with torch.inference_mode(): expected = source(**inputs); actual = adapted(**inputs)
    assert set(expected) == set(actual)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    result = select(adapted, observations, **kwargs)
    np.testing.assert_array_equal(result['prediction'], expected[heads[-1]].numpy())
    assert result['translation_bias_training_only'] and result['model_prediction_corrected']
    assert state_digest(source.state_dict()) == state_digest(adapted.state_dict()) == before
    for key, value in source.state_dict().items():
        assert value.data_ptr() != adapted.state_dict()[key].data_ptr()
    assert all(parameter.grad is None for parameter in adapted.parameters())
    with pytest.raises(ValueError): adapted.train()


def test_censored_padding_and_target_clocks_are_identical():
    source = model(('direct_outcomes', 'rollout_outcomes')); adapted = AllPhasePlannerModel(source)
    inputs = candidate_inputs(history())
    inputs['known_action_valid'][:, -3:] = False
    inputs['known_action_blocks'][~inputs['known_action_valid']] = 0
    with torch.inference_mode(): expected = source(**inputs); actual = adapted(**inputs)
    for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_nonzero_unknown_padding_still_rejects_in_both_interfaces():
    source = model(('direct_outcomes',)); adapted = AllPhasePlannerModel(source)
    inputs = candidate_inputs(history()); inputs['known_action_valid'][:, -3:] = False
    for wrapper in (source, adapted):
        with pytest.raises(ValueError, match='padding'): wrapper(**inputs)


@pytest.mark.parametrize('fault', ['base_mode', 'wrapper_mode', 'gradient', 'nan_bias', 'wrong_type'])
def test_invalid_source_model_is_rejected(fault):
    source = model(('direct_outcomes',))
    if fault == 'base_mode': source.base.train()
    elif fault == 'wrapper_mode': source.training = True
    elif fault == 'gradient': next(source.parameters()).grad = torch.zeros_like(next(source.parameters()))
    elif fault == 'nan_bias': source.direct_outcomes_xy_bias[0, 0] = float('nan')
    else: source = object()
    with pytest.raises(ValueError): AllPhasePlannerModel(source)
