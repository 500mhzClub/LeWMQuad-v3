from copy import deepcopy
import pytest
import torch
from lewm.observation_horizon_plan_development import plan,validate_plan
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer,training_loss
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.pulse_position_scale_learning_development import scaled_outcome_loss
from lewm.pulse_timed_learning_development import active_parameters,CONDITIONS
from lewm.pulse_timed_training_runner_development import state_digest


def batch():
    generator=torch.Generator().manual_seed(923)
    history={k:torch.rand(shape,generator=generator) for k,shape in
        dict(rgb=(2,4,3,96,128),body=(2,4,20,63),control=(2,4,15,7)).items()}
    pairs=[plan('forward'),plan('hold',offset_ticks=35)]
    blocks,valid=(torch.stack([pair[i] for pair in pairs]) for i in range(2))
    active,offsets=validate_plan(blocks,valid,2)
    future={k:torch.rand((2,8,*v.shape[2:]),generator=generator) for k,v in history.items()}
    for v in future.values():v[~active]=float('nan')
    motion=torch.full((2,8,3),float('nan'));motion[active]=.01
    contact=torch.full((2,8),float('nan'));contact[active]=0.
    return dict(inputs=dict(observation_history=history,known_action_blocks=blocks,known_action_valid=valid),
        targets=dict(motion=motion,motion_valid=active.clone(),contact=contact,contact_valid=active.clone(),
            future_observations=future,future_valid=active.clone(),target_offsets_ns=offsets))


@pytest.mark.parametrize('condition',CONDITIONS)
def test_short_clock_training_and_inference_are_identical_and_gradients_match(condition):
    trainer=ObservationHorizonTrainer(condition,seed=17,latent_dim=16);b=batch()
    out=trainer.model(**b['inputs']);loss,parts=training_loss(trainer.model,b,condition)
    assert out['target_offsets_ns'].tolist()==[[i*100_000_000 for i in range(1,9)],
        [100_000_000,200_000_000,300_000_000,400_000_000,500_000_000,0,0,0]]
    args=[b['targets'][k] for k in ('motion','contact','motion_valid','contact_valid')]
    assert parts['direct_outcome']==float(scaled_outcome_loss(out['direct_outcomes'],*args).detach())
    if condition!='direct':assert parts['rollout_outcome']==float(scaled_outcome_loss(out['rollout_outcomes'],*args).detach())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active_parameters(trainer.model,condition))
    assert all(p.grad is None for p in trainer.model.target_encoder.parameters())
    for head in ('direct_outcomes','rollout_outcomes'):
        assert (torch.diff(out[head][:,:5,4],dim=-1)>=0).all()
        assert not out[head][~out['prediction_valid']].any()
    assert trainer.updates==0


def test_initialization_is_paired_and_old_action_contract_is_rejected():
    trainers=[ObservationHorizonTrainer(c,seed=17,latent_dim=16) for c in CONDITIONS]
    assert len({t.initial_sha256 for t in trainers})==1
    b=batch();b['inputs']['known_action_blocks']=torch.zeros(2,8,5,3)
    b['inputs']['known_action_valid']=torch.ones(2,8,5,dtype=torch.bool)
    with pytest.raises(ValueError,match='B,8,1,3'):trainers[0].model(**b['inputs'])
    checkpoint=trainers[0].checkpoint()
    assert checkpoint['schema']=='observation_horizon_training_development.v1'
    assert checkpoint['target_cadence_ns']==100_000_000 and checkpoint['action_ticks_per_block']==1


def test_later_actions_and_target_data_cannot_change_earlier_forward_predictions():
    b=batch();model=ObservationHorizonRGBBodyJEPA(16).eval()
    with torch.no_grad():before=model(**b['inputs'])
    changed=deepcopy(b)
    for value in changed['targets']['future_observations'].values():value.fill_(123.)
    changed['targets']['motion'].fill_(999.)
    changed['inputs']['known_action_blocks'][0,4:,0,2]=-.9
    with torch.no_grad():after=model(**changed['inputs'])
    for head in ('direct_outcomes','rollout_outcomes','future_latents'):
        torch.testing.assert_close(before[head][:,:4],after[head][:,:4],rtol=0,atol=0)


def test_bad_clock_latches_without_update_and_contact_future_is_rejected():
    trainer=ObservationHorizonTrainer('jepa',seed=17,latent_dim=16);b=batch()
    b['targets']['target_offsets_ns'][0,0]=500_000_000;before=state_digest(trainer.model.state_dict())
    with pytest.raises(ValueError):trainer.step(b)
    assert trainer.failed and trainer.updates==0 and state_digest(trainer.model.state_dict())==before
    with pytest.raises(ValueError,match='latched'):trainer.step(batch())
    b=batch();b['targets']['motion_valid'][0,0]=False;b['targets']['motion'][0,0]=float('nan')
    with pytest.raises(ValueError,match='censoring'):training_loss(ObservationHorizonRGBBodyJEPA(16),b,'jepa')


def test_rgb_removal_preserves_short_clocks_targets_and_original_tensors():
    from lewm.observation_horizon_input_ablation_development import transform_training_batch
    b=batch();changed=transform_training_batch(b,input_variant='no_rgb')
    assert not changed['inputs']['observation_history']['rgb'].any()
    assert not changed['targets']['future_observations']['rgb'].any()
    assert b['inputs']['observation_history']['rgb'].any()
    for k in ('motion','motion_valid','contact','contact_valid','target_offsets_ns','future_valid'):
        assert changed['targets'][k] is b['targets'][k]
    for k in ('known_action_blocks','known_action_valid'):assert changed['inputs'][k] is b['inputs'][k]
    assert torch.isfinite(training_loss(ObservationHorizonRGBBodyJEPA(16),changed,'jepa')[0])
