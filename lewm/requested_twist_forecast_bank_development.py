"""A nominal command-following reference, never a learned or measured forecast.

Accept only the existing six ordered eight-step candidate plans. Integrate each
requested planar twist exactly over its 100-ms interval, assuming perfect
velocity tracking. This neither models the executor nor updates robot pose.
The constant -30 contact logit is the existing no-event reference convention;
it supplies no learned risk discrimination or calibrated contact probability.
No model, observation, checkpoint, private geometry or future data is accepted.
"""
import math

import numpy as np
import torch

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.observation_horizon_plan_development import plan

CONTACT_LOGIT = -30.0
INTERVAL_S = .1


def forecast_bank(known_action_blocks, known_action_valid):
    expected_blocks, expected_valid = zip(*(plan(action) for action in ACTIONS), strict=True)
    if (not isinstance(known_action_blocks, torch.Tensor)
            or not isinstance(known_action_valid, torch.Tensor)
            or known_action_blocks.device.type != 'cpu' or known_action_valid.device.type != 'cpu'
            or known_action_blocks.dtype != torch.float32 or known_action_valid.dtype != torch.bool
            or known_action_blocks.requires_grad or known_action_valid.requires_grad
            or not torch.equal(known_action_blocks, torch.stack(expected_blocks))
            or not torch.equal(known_action_valid, torch.stack(expected_valid))):
        raise ValueError('exact original ordered six-candidate CPU command bank required')
    commands = known_action_blocks[:, :, 0].numpy().astype(np.float64)
    commands *= np.array([.3, 1., .5], dtype=np.float64)
    outcomes = np.zeros((6, 8, 5), dtype=np.float32)
    for index in range(6):
        x = y = yaw = 0.
        for horizon in range(8):
            vx, vy, wz = commands[index, horizon]
            if vy != 0.:
                raise ValueError('original zero-lateral action bank required')
            angle = float(wz)*INTERVAL_S
            if wz == 0.:
                dx, dy = float(vx)*INTERVAL_S, 0.
            else:
                dx = float(vx/wz)*math.sin(angle)
                dy = float(vx/wz)*(1.-math.cos(angle))
            x, y = (x+math.cos(yaw)*dx-math.sin(yaw)*dy,
                    y+math.sin(yaw)*dx+math.cos(yaw)*dy)
            yaw += angle
            outcomes[index, horizon] = (x, y, math.sin(yaw), math.cos(yaw), CONTACT_LOGIT)
    if not np.isfinite(outcomes).all():
        raise ValueError('finite nominal candidate outcomes required')
    return dict(nominal_outcomes=torch.from_numpy(outcomes),
        prediction_valid=torch.ones((6, 8), dtype=torch.bool),
        target_offsets_ns=torch.arange(1, 9, dtype=torch.int64).mul(100_000_000).expand(6, 8).clone(),
        provenance=dict(schema='requested_twist_forecast_bank_development.v1',
            actions=list(ACTIONS), motion_source='exact_planar_integration_of_requested_twists',
            requested_velocity_tracking_assumed=True, executed_command_limiter_modelled=False,
            measured_robot_pose_updated=False, executed_motion_established=False,
            learned_world_model_forward_called=False, translation_bias_applied=False,
            online_residual_update_performed=False, observation_input_used=False,
            future_sensor_input_used=False, native_state_used=False,
            contact_source='constant_no_event_reference_logit', contact_logit=CONTACT_LOGIT,
            contact_probability_calibrated=False, native_execution=False,
            navigation_qualified=False, real_time_qualified=False))
