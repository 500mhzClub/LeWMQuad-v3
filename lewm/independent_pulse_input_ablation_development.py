"""Matched information removals, not sensor realism or navigation evidence.

Transforms never mutate supplied tensors. Unchanged tensors remain shared and
must be treated as read-only. The caller authenticates original data and plans
BEFORE removing information. Labels, availability and clocks are never changed.
"""
import torch
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan

VARIANTS = ('full', 'no_rgb', 'latest_packet_only', 'no_candidate_command')
SHAPES = {'rgb': (3, 96, 128), 'body': (20, 63), 'control': (15, 7)}


def validate_variant(input_variant):
    if type(input_variant) is not str or input_variant not in VARIANTS:
        raise ValueError('explicit supported input variant required')


def _observations(observations, packets, *, observed=None):
    if not isinstance(observations, dict) or set(observations) != set(SHAPES):
        raise ValueError('exact RGB/body/control observations required')
    n = None
    for key, shape in SHAPES.items():
        value = observations[key]
        if not isinstance(value, torch.Tensor) or value.ndim != len(shape) + 2:
            raise ValueError('exact observation tensor shape required')
        n = len(value) if n is None else n
        if value.shape != (n, packets, *shape) or not 1 <= n <= 16:
            raise ValueError('bounded matched observation shapes required')
        if value.dtype != torch.float32 or value.device.type != 'cpu':
            raise ValueError('CPU float32 observations required')
        if observed is not None and (not isinstance(observed, torch.Tensor)
                or observed.dtype != torch.bool or observed.device.type != 'cpu'
                or observed.shape != (n, packets)):
            raise ValueError('exact future availability mask required')
        if not torch.isfinite(value if observed is None else value[observed]).all():
            raise ValueError('finite available observations required before ablation')
    return n


def transform_inputs(inputs, *, input_variant):
    """Apply the identical causal treatment at training and inference.

    latest_packet_only repeats packet 4, including its 20 body samples and 15
    control samples, over the four existing packet positions. It is NOT a
    memoryless sensor ablation. no_candidate_command zeros prospective values
    only; plan validity/duration and past applied controls remain informative.
    """
    validate_variant(input_variant)
    if not isinstance(inputs, dict) or set(inputs) != {
            'observation_history', 'known_action_blocks', 'known_action_valid'}:
        raise ValueError('policy-only inference fields required')
    n = _observations(inputs['observation_history'], 4)
    validate_timed_plan(inputs['known_action_blocks'], inputs['known_action_valid'], n)
    result = dict(inputs)
    history = dict(inputs['observation_history'])
    if input_variant == 'no_rgb':
        history['rgb'] = torch.zeros_like(history['rgb'])
    elif input_variant == 'latest_packet_only':
        history = {key: value[:, -1:].expand_as(value).clone() for key, value in history.items()}
    elif input_variant == 'no_candidate_command':
        result['known_action_blocks'] = torch.zeros_like(inputs['known_action_blocks'])
    result['observation_history'] = history
    return result


def transform_training_batch(batch, *, input_variant):
    """Remove RGB from BOTH online and teacher future observations for no_rgb.

    Future images are target-only; the other variants preserve these targets.
    Physical outcomes, target times and every censoring mask remain untouched.
    This is a retrained missing-information control, not test-time corruption.
    """
    validate_variant(input_variant)
    if not isinstance(batch, dict) or set(batch) != {'inputs', 'targets'}:
        raise ValueError('explicit input/target separation required')
    targets = batch['targets']
    if not isinstance(targets, dict) or set(targets) != {
            'motion', 'motion_valid', 'contact', 'contact_valid',
            'future_observations', 'future_valid', 'target_offsets_ns'}:
        raise ValueError('exact physical and future target fields required')
    if not isinstance(targets['future_valid'], torch.Tensor):
        raise ValueError('exact future availability mask required')
    n = _observations(targets['future_observations'], 8, observed=targets['future_valid'])
    inputs = transform_inputs(batch['inputs'], input_variant=input_variant)
    if n != len(inputs['observation_history']['rgb']):
        raise ValueError('matched input and target batch sizes required')
    targets = dict(targets)
    if input_variant == 'no_rgb':
        future = dict(targets['future_observations'])
        future['rgb'] = torch.zeros_like(future['rgb'])
        targets['future_observations'] = future
    return dict(inputs=inputs, targets=targets)
