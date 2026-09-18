"""Matched RGB removal with the actual one-command block contract."""
import torch
from lewm.observation_horizon_plan_development import validate_plan
from lewm.independent_pulse_input_ablation_development import _observations


def transform_inputs(inputs,*,input_variant):
    if input_variant not in ('full','no_rgb'):raise ValueError('fixed full/no-RGB short-horizon treatment required')
    if not isinstance(inputs,dict) or set(inputs)!={'observation_history','known_action_blocks','known_action_valid'}:
        raise ValueError('exact policy-only inference fields required')
    n=_observations(inputs['observation_history'],4)
    validate_plan(inputs['known_action_blocks'],inputs['known_action_valid'],n)
    history=dict(inputs['observation_history'])
    if input_variant=='no_rgb':history['rgb']=torch.zeros_like(history['rgb'])
    return dict(inputs,observation_history=history)


def transform_training_batch(batch,*,input_variant):
    if not isinstance(batch,dict) or set(batch)!={'inputs','targets'}:
        raise ValueError('explicit input/target separation required')
    targets=batch['targets']
    if not isinstance(targets,dict) or set(targets)!={'motion','motion_valid','contact','contact_valid',
            'future_observations','future_valid','target_offsets_ns'}:
        raise ValueError('exact physical and future target fields required')
    if not isinstance(targets['future_valid'],torch.Tensor):raise ValueError('explicit future availability required')
    n=_observations(targets['future_observations'],8,observed=targets['future_valid'])
    inputs=transform_inputs(batch['inputs'],input_variant=input_variant)
    if n!=len(inputs['observation_history']['rgb']):raise ValueError('matched input and target batch sizes required')
    targets=dict(targets)
    if input_variant=='no_rgb':
        future=dict(targets['future_observations']);future['rgb']=torch.zeros_like(future['rgb'])
        targets['future_observations']=future
    return dict(inputs=inputs,targets=targets)
