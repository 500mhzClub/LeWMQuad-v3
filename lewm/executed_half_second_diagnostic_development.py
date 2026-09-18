"""Post-outcome scoring of actually completed commands; no policy or training."""
import math
import numpy as np
import torch
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.geometry_progress_predictive_selection_development import candidate_inputs
from lewm.independent_pulse_input_ablation_development import transform_inputs
from lewm.physical_execution_development import rotation_xyzw


@torch.inference_mode()
def predict(model, history, *, condition, variant):
    if model.training or condition not in ('direct', 'supervised_rollout', 'jepa'):
        raise ValueError('explicit evaluation-only fitted condition required')
    inputs = transform_inputs(candidate_inputs(history), input_variant=variant)
    output = model(**inputs)
    clock = torch.arange(1, 9, dtype=torch.int64).mul(500_000_000).expand(6, 8)
    if (not torch.equal(output['target_offsets_ns'], clock)
            or output['prediction_valid'].dtype != torch.bool
            or output['prediction_valid'].shape != (6, 8) or not output['prediction_valid'].all()):
        raise ValueError('complete exact prediction clock required')
    head = 'direct_outcomes' if condition == 'direct' else 'rollout_outcomes'
    result = output[head].cpu().numpy().copy()
    if result.shape != (6, 8, 5) or not np.isfinite(result).all():
        raise ValueError('complete finite predictions required')
    return result


def executed_outcome(raw, tape, *, tick, action):
    """No future arrays leave this evaluator-only function for model inference."""
    if type(tick) is not int or tick < 0 or action not in ACTIONS:
        raise ValueError('explicit actual decision required')
    if tick + 5 > len(tape):
        return dict(eligible=False, reason='five command intervals unavailable')
    plan = candidate_commands(action)[:5]
    for offset, item in enumerate(tape[tick:tick+5]):
        if item['tick'] != tick+offset:
            raise ValueError('contiguous exact command tape required')
        if not item['completed']:
            return dict(eligible=False, reason='command interval incomplete')
        if item['requested_command'] != list(plan[offset]):
            return dict(eligible=False, reason='executed command differs from predicted plan')
        if (item['pre_sample_index'] != 749+50*(tick+offset)
                or item['post_sample_index'] != item['pre_sample_index']+50):
            raise ValueError('exact native half-second interval required')
    a, b = tape[tick]['pre_sample_index'], tape[tick+4]['post_sample_index']
    if b >= len(raw['base_pose_world']):
        raise ValueError('complete physical outcome missing')
    if not np.isclose(raw['timestamp_s'][b]-raw['timestamp_s'][a], .5, rtol=0, atol=1e-12):
        raise ValueError('physical horizon clock mismatch')
    expected = np.repeat(np.asarray(plan, np.float64), 50, axis=0)
    np.testing.assert_array_equal(raw['requested_command'][a+1:b+1], expected)
    pose = raw['base_pose_world']; rotation = rotation_xyzw(pose[a, 3:])
    position = rotation.T @ (pose[b, :3]-pose[a, :3])
    relative = rotation.T @ rotation_xyzw(pose[b, 3:])
    return dict(eligible=True, start_sample=a, end_sample=b, xy_m=position[:2].tolist(),
        yaw_rad=math.atan2(relative[1, 0], relative[0, 0]),
        contact=bool(raw['physics_contact'][a+1:b+1].any()), evaluator_only=True)


def error(prediction, outcome):
    p = np.asarray(prediction, float)
    if p.shape != (5,) or not np.isfinite(p).all() or not outcome['eligible']:
        raise ValueError('finite prediction and eligible actual outcome required')
    yaw = math.atan2(p[2], p[3]) if np.linalg.norm(p[2:4]) > 1e-8 else None
    difference = None if yaw is None else yaw-outcome['yaw_rad']
    probability = float(np.exp(-np.logaddexp(0., -p[4])))
    return dict(predicted_xy_m=p[:2].tolist(), predicted_yaw_rad=yaw,
        predicted_contact_score=probability,
        xy_error_m=float(np.linalg.norm(p[:2]-outcome['xy_m'])),
        yaw_error_rad=None if difference is None else abs(math.atan2(math.sin(difference), math.cos(difference))),
        contact_brier=(probability-int(outcome['contact']))**2)
