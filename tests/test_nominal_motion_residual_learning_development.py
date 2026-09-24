import numpy as np
import torch
from lewm.nominal_motion_residual_learning_development import (
    nominal_motion,compose_outcomes,NominalResidualTrainer)
from lewm.observation_horizon_learning_development import training_loss
from lewm.pulse_position_scale_learning_development import scaled_outcome_loss


def batch():
    history={k:torch.zeros(s) for k,s in dict(rgb=(2,4,3,96,128),body=(2,4,20,63),control=(2,4,15,7)).items()}
    blocks=torch.zeros(2,8,1,3);blocks[:,:4,0,0]=.2/.3;blocks[:,4:7,0,2]=.45/.5
    valid=torch.ones(2,8,1,dtype=torch.bool);valid[:,7]=False
    active=valid[:,:,0]
    motion=nominal_motion(blocks,valid)
    future={k:torch.zeros((2,8,*v.shape[2:])) for k,v in history.items()}
    return dict(inputs=dict(observation_history=history,known_action_blocks=blocks,known_action_valid=valid),
        targets=dict(motion=motion,motion_valid=active,contact=torch.zeros(2,8),contact_valid=active,
            future_observations=future,future_valid=active,
            target_offsets_ns=torch.where(active,torch.arange(1,9)[None]*100_000_000,0)))


def test_analytic_nominal_mask_and_prefix_causality():
    b=batch()['inputs'];blocks=b['known_action_blocks'];valid=b['known_action_valid']
    predicted=nominal_motion(blocks,valid)
    np.testing.assert_allclose(predicted[0,:4,0],np.arange(1,5)*.02,atol=1e-8)
    np.testing.assert_allclose(predicted[0,4:7,2],np.arange(1,4)*.045,atol=1e-8)
    assert torch.count_nonzero(predicted[:,7])==0
    changed=blocks.clone();changed[:,4:7,0,2]*=-1
    assert torch.equal(nominal_motion(changed,valid)[:,:4],predicted[:,:4])
    residual=torch.zeros(2,8,5);residual[:,:,3]=1;residual[:,:,4]=-.7
    absolute=compose_outcomes(residual,blocks,valid)
    torch.testing.assert_close(absolute[:,:,:2],predicted[:,:,:2])
    torch.testing.assert_close(torch.atan2(absolute[:,:7,2],absolute[:,:7,3]),predicted[:,:7,2])
    assert torch.equal(absolute[:,:7,4],residual[:,:7,4])


def test_all_objectives_train_on_same_absolute_inference_outputs():
    torch.set_num_threads(1);b=batch()
    for condition in ('jepa','direct','supervised_rollout'):
        trainer=NominalResidualTrainer(condition,seed=2026091001)
        output=trainer.model(**b['inputs'])
        _,parts=training_loss(trainer.model,b,condition,outcome_transform=compose_outcomes)
        args=[b['targets'][k] for k in ('motion','contact','motion_valid','contact_valid')]
        assert parts['direct_outcome']==float(scaled_outcome_loss(output['direct_outcomes'],*args).detach())
        if condition!='direct':
            assert parts['rollout_outcome']==float(scaled_outcome_loss(output['rollout_outcomes'],*args).detach())
        record=trainer.step(b)
        assert record['update']==1 and np.isfinite(record['loss'])
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.parameters)


def test_original_loss_default_is_identity_transform():
    from lewm.observation_horizon_learning_development import ObservationHorizonTrainer
    torch.set_num_threads(1);b=batch();model=ObservationHorizonTrainer('jepa',seed=2026091001).model
    original,parts=training_loss(model,b,'jepa')
    identical,other=training_loss(model,b,'jepa',outcome_transform=lambda p,b,v:p)
    assert torch.equal(original,identical) and parts==other
