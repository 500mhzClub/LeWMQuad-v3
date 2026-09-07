import torch
import pytest

from lewm.rgb_body_jepa_reference_development import RGBBodyJEPAReference,variance_covariance_penalty,masked_outcome_loss
from lewm.rgb_body_tensor_interface_development import observation_tensors
from lewm.tests.test_simulated_body_observation_development import packet


def fixture():
    torch.manual_seed(7)
    observation={k:torch.stack([v,v+.001]) for k,v in observation_tensors(packet()).items()}
    return RGBBodyJEPAReference(),observation,torch.zeros(2,8,5,3)


def test_tensor_interface_shapes_and_declared_input_boundary():
    model,observation,plan=fixture()
    assert observation['body'].shape==(2,20,63)
    assert observation['control'].shape==(2,15,7)
    result=model(observation,plan)
    assert result['future_latents'].shape==(2,8,128)
    assert result['direct_outcomes'].shape==result['rollout_outcomes'].shape==(2,8,5)
    with pytest.raises(ValueError,match='undeclared'): model.encoder(observation | {'world_pose':torch.zeros(2,7)})


def test_future_action_changes_cannot_affect_earlier_horizons():
    model,observation,plan=fixture()
    original=model(observation,plan)
    plan[:,4:,:,2]=.5
    changed=model(observation,plan)
    for name in ('future_latents','direct_outcomes','rollout_outcomes'):
        assert torch.equal(original[name][:,:4],changed[name][:,:4])
        assert not torch.equal(original[name][:,4:],changed[name][:,4:])


def test_target_is_stop_gradient_and_ema_updates_only_when_called():
    model,observation,plan=fixture()
    before=[p.clone() for p in model.target_encoder.parameters()]
    assert not model.target(observation).requires_grad
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    loss=model(observation,plan)['direct_outcomes'].square().mean()
    loss.backward(); optimizer.step()
    assert all(torch.equal(a,b) for a,b in zip(before,model.target_encoder.parameters()))
    model.update_target(0.)
    assert all(torch.equal(a,b) for a,b in zip(model.encoder.parameters(),model.target_encoder.parameters()))


def test_temporal_order_changes_body_representation():
    model,observation,_=fixture()
    observation['body']=torch.randn_like(observation['body'])
    original=model.encoder(observation)
    shuffled=model.encoder(observation | {'body':observation['body'].flip(1)})
    assert not torch.allclose(original,shuffled)


def test_collapse_has_variance_penalty():
    variance,covariance=variance_covariance_penalty(torch.zeros(8,16))
    assert variance>.9 and covariance==0


def test_direct_inference_never_executes_the_world_predictor():
    model,observation,plan=fixture()
    def forbidden(*args,**kwargs): raise AssertionError('world rollout executed')
    model.transition.forward=forbidden
    assert model.direct_prediction(observation,plan).shape==(2,8,5)


def test_censored_nan_targets_are_not_used_in_loss():
    prediction=torch.zeros(2,8,5,requires_grad=True)
    motion=torch.full((2,8,3),float('nan'))
    contact=torch.full((2,8),float('nan'))
    valid=torch.zeros(2,8,dtype=torch.bool)
    loss=masked_outcome_loss(prediction,motion,contact,valid,valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
