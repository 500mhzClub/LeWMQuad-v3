"""Explicit learned/nominal forecast intervention at the shared candidate input.

Both modes retain the original cost and six eight-step action plans. Nominal
forecasts assume perfect requested-velocity tracking and supply a constant
contact logit. They are not learned forecasts or executed-motion evidence.
Checkpoint admission and matched physical execution remain the caller's work.
"""
import time

import torch

from lewm.training_translation_bias_development import TrainingTranslationBiasModel
from lewm.training_bias_predictive_selection_development import select as learned_select
from lewm.observation_horizon_predictive_selection_development import candidate_inputs, score_candidates
from lewm.requested_twist_forecast_bank_development import forecast_bank

SOURCES = ('frozen_world_model', 'nominal_requested_twist')


def require_source(source):
    if type(source) is not str or source not in SOURCES:
        raise ValueError('explicit frozen-model or nominal-requested-twist forecast source required')
    return source


@torch.inference_mode()
def select(model, observation_history, *, head, input_variant, goal_body_xy_m,
        contact_penalty_m, forecast_source):
    source = require_source(forecast_source)
    if (not isinstance(model, TrainingTranslationBiasModel) or model.training
            or head not in model.corrected_heads or input_variant not in ('full', 'no_rgb')):
        raise ValueError('same evaluation-only corrected model and supported assigned input variant required')
    start = time.perf_counter_ns()
    if source == 'frozen_world_model':
        result = learned_select(model, observation_history, head=head, input_variant=input_variant,
            goal_body_xy_m=goal_body_xy_m, contact_penalty_m=contact_penalty_m)
        provenance = dict(schema='forecast_source_selection_development.v1',
            forecast_source=source, assigned_frozen_model_head=head,
            frozen_model_forward_called=True, learned_forecasts_used=True,
            nominal_requested_twist_forecasts_used=False,
            observation_history_supplied_to_forecaster=True,
            model_checkpoint_admitted_by_this_helper=False,
            native_execution=False, navigation_qualified=False)
    else:
        inputs = candidate_inputs(observation_history)
        bank = forecast_bank(inputs['known_action_blocks'], inputs['known_action_valid'])
        prediction = bank['nominal_outcomes'].numpy()
        result = score_candidates(prediction, goal_body_xy_m=goal_body_xy_m,
            contact_penalty_m=contact_penalty_m) | dict(
            first_prediction_horizon_ns=100_000_000,
            target_offsets_ns=bank['target_offsets_ns'][0].tolist(),
            head='nominal_requested_twist', input_variant=input_variant,
            prediction=prediction.tolist(), selection_wall_ms=(time.perf_counter_ns()-start)/1e6,
            model_checkpoint_admitted_by_this_helper=False, model_prediction_corrected=False,
            translation_bias_training_only=False, translation_bias_xy_m=None)
        provenance = dict(schema='forecast_source_selection_development.v1',
            forecast_source=source, assigned_frozen_model_head=head,
            frozen_model_forward_called=False, learned_forecasts_used=False,
            nominal_requested_twist_forecasts_used=True,
            observation_history_supplied_to_forecaster=False,
            model_checkpoint_admitted_by_this_helper=False,
            nominal_forecast_assumptions=bank['provenance'],
            native_execution=False, navigation_qualified=False)
    return result | {'forecast_provenance': provenance}
