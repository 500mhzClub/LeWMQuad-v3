import math
import numpy as np
import pytest
import torch

from lewm import requested_twist_forecast_bank_development as nominal
from lewm.observation_horizon_plan_development import plan
from lewm.observation_horizon_predictive_selection_development import score_candidates


def inputs():
    blocks,valid=zip(*(plan(a) for a in nominal.ACTIONS),strict=True)
    return torch.stack(blocks),torch.stack(valid)


def test_exact_straight_turn_and_circular_arc_solutions():
    blocks,valid=inputs();before=blocks.clone();before_valid=valid.clone()
    result=nominal.forecast_bank(blocks,valid)
    p=result['nominal_outcomes'].numpy();t=np.arange(1,9)*.1
    speed=float(blocks[1,0,0,0])*.3
    arc_speed=float(blocks[2,0,0,0])*.3
    yaw_rate=float(blocks[2,0,0,2])*.5
    np.testing.assert_array_equal(p[0,:,:2],np.zeros((8,2)))
    np.testing.assert_allclose(p[1,:,0],speed*t,rtol=1e-7,atol=1e-8)
    np.testing.assert_array_equal(p[1,:,1:3],np.zeros((8,2)))
    np.testing.assert_allclose(p[2,:,0],arc_speed/yaw_rate*np.sin(yaw_rate*t),rtol=1e-7,atol=1e-8)
    np.testing.assert_allclose(p[2,:,1],arc_speed/yaw_rate*(1-np.cos(yaw_rate*t)),rtol=1e-7,atol=1e-8)
    np.testing.assert_array_equal(p[4,:,:2],np.zeros((8,2)))
    np.testing.assert_allclose(p[4,:,2],np.sin(yaw_rate*t),rtol=1e-7,atol=1e-8)
    np.testing.assert_allclose(p[4,:,3],np.cos(yaw_rate*t),rtol=1e-7,atol=1e-8)
    np.testing.assert_allclose(p[:,:,2]**2+p[:,:,3]**2,1.,atol=1e-7)
    assert torch.equal(blocks,before) and torch.equal(valid,before_valid)
    assert result['prediction_valid'].all()
    assert result['target_offsets_ns'].tolist()==[list(range(100_000_000,800_000_001,100_000_000))]*6
    assert result['nominal_outcomes'].dtype==torch.float32


def test_mirrored_candidates_and_constant_contact_do_not_invent_scene_risk():
    p=nominal.forecast_bank(*inputs())['nominal_outcomes'].numpy()
    for left,right in ((2,3),(4,5)):
        np.testing.assert_array_equal(p[left,:,0],p[right,:,0])
        np.testing.assert_array_equal(p[left,:,1:3],-p[right,:,1:3])
        np.testing.assert_array_equal(p[left,:,3:],p[right,:,3:])
    np.testing.assert_array_equal(p[:,:,4],np.full((6,8),-30.,dtype=np.float32))


def test_existing_cost_interface_accepts_nominal_bank_without_changed_cost():
    result=nominal.forecast_bank(*inputs())
    selection=score_candidates(result['nominal_outcomes'].numpy(),goal_body_xy_m=[1.2,0.],contact_penalty_m=1.2)
    assert selection['action']=='forward'
    assert len(selection['candidates'])==6
    assert len({r['predicted_contact_score'] for r in selection['candidates']})==1
    assert selection['navigation_qualified'] is False


def test_nominal_source_never_claims_measured_or_learned_motion():
    result=nominal.forecast_bank(*inputs());p=result['provenance']
    assert 'direct_outcomes' not in result and 'rollout_outcomes' not in result
    assert p['requested_velocity_tracking_assumed'] is True
    for field in ('learned_world_model_forward_called','translation_bias_applied',
            'online_residual_update_performed','measured_robot_pose_updated',
            'executed_motion_established','executed_command_limiter_modelled',
            'future_sensor_input_used','native_state_used','navigation_qualified'):
        assert p[field] is False


@pytest.mark.parametrize('fault',['order','unknown','nan','float64','gradient','lateral','scaled','short','array'])
def test_other_command_banks_or_mutable_training_inputs_rejected(fault):
    blocks,valid=inputs()
    if fault=='order':blocks=blocks.flip(0)
    elif fault=='unknown':valid[-1,-1]=False
    elif fault=='nan':blocks[0,0,0,0]=float('nan')
    elif fault=='float64':blocks=blocks.double()
    elif fault=='gradient':blocks.requires_grad_()
    elif fault=='lateral':blocks[0,0,0,1]=.1
    elif fault=='scaled':blocks[1,0,0,0]+=.01
    elif fault=='short':blocks=blocks[:,:7];valid=valid[:,:7]
    else:blocks=blocks.numpy()
    with pytest.raises(ValueError,match='exact original ordered'):nominal.forecast_bank(blocks,valid)


def test_repeat_calls_do_not_share_output_or_mutate_inputs():
    args=inputs();first=nominal.forecast_bank(*args);second=nominal.forecast_bank(*args)
    first['nominal_outcomes'].zero_();first['target_offsets_ns'].zero_()
    assert second['nominal_outcomes'][1,-1,0] > .15
    assert second['target_offsets_ns'][0,-1]==800_000_000
