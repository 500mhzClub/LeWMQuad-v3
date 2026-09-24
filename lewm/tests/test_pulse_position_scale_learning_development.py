from copy import deepcopy
import pytest
import torch
from lewm.pulse_position_scale_learning_development import PositionScaleTrainer,training_loss,scaled_outcome_loss,POSITION_SCALE_M
from lewm.pulse_timed_training_runner_development import PulseTrainer,state_digest
from lewm.pulse_timed_learning_development import training_loss as raw_loss,active_parameters
from lewm.tests.test_pulse_timed_learning_development import batch


@pytest.mark.parametrize('condition',['direct','supervised_rollout','jepa'])
def test_raw_steps_bit_identical_and_scaled_initialization_matched(condition):
    a=PulseTrainer(condition,seed=42,latent_dim=16)
    b=PositionScaleTrainer(condition,objective='raw',seed=42,latent_dim=16)
    c=PositionScaleTrainer(condition,objective='position_6cm',seed=42,latent_dim=16)
    assert a.initial_sha256==b.initial_sha256==c.initial_sha256
    data=batch()
    for _ in range(2):assert a.step(data)==b.step(deepcopy(data))
    assert c.step(data)['model_sha256']!=b.initial_sha256
    assert c.checkpoint()['objective']=='position_6cm'


@pytest.mark.parametrize('condition',['direct','supervised_rollout','jepa'])
def test_unit_scale_reconstructs_frozen_loss_and_gradients(monkeypatch,condition):
    import lewm.pulse_position_scale_learning_development as module
    monkeypatch.setattr(module,'POSITION_SCALE_M',1.)
    model=PulseTrainer(condition,seed=19,latent_dim=16).model;data=batch()
    a,parts=raw_loss(model,data,condition);b,scaled=training_loss(model,data,condition,'position_6cm')
    assert parts==scaled
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    parameters=active_parameters(model,condition)
    ga=torch.autograd.grad(a,parameters);gb=torch.autograd.grad(b,parameters)
    for x,y in zip(ga,gb,strict=True):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_only_position_changes_and_exact_physical_scale():
    model=PulseTrainer('jepa',seed=2,latent_dim=16).model;data=batch()
    _,raw=raw_loss(model,data,'jepa');_,scaled=training_loss(model,data,'jepa','position_6cm')
    for name in ('variance','covariance','latent_prediction'):assert raw[name]==scaled[name]
    p=torch.zeros(1,8,5,requires_grad=True);m=torch.full((1,8,3),float('nan'))
    m[0,0]=torch.tensor([.01,0.,0.]);cv=torch.zeros(1,8,dtype=torch.bool);mv=cv.clone();mv[0,0]=True
    c=torch.full((1,8),float('nan'))
    loss=scaled_outcome_loss(p,m,c,mv,cv)
    assert float(loss.detach())==pytest.approx(.125+(.01/POSITION_SCALE_M)**2/8)
    g,=torch.autograd.grad(loss,p);assert torch.count_nonzero(g[~mv])==0


def test_scaled_ema_and_latched_bad_target():
    trainer=PositionScaleTrainer('jepa',objective='position_6cm',seed=11,latent_dim=16)
    before=deepcopy(trainer.model.state_dict());trainer.step(batch())
    for name,p in trainer.model.encoder.named_parameters():
        expected=before['target_encoder.'+name]*.99+p.detach()*.01
        torch.testing.assert_close(trainer.model.target_encoder.state_dict()[name],expected)
    assert all(p.grad is None for p in trainer.model.target_encoder.parameters())
    data=batch();data['targets']['motion'][0,0]=float('nan')
    identity=state_digest(trainer.model.state_dict())
    with pytest.raises(ValueError):trainer.step(data)
    assert trainer.failed and trainer.updates==1 and state_digest(trainer.model.state_dict())==identity
    with pytest.raises(ValueError,match='latched'):trainer.step(batch())


@pytest.mark.parametrize('problem',['time','unknown','privileged','contact_motion','contact_range'])
def test_scaled_contract_rejection(problem):
    trainer=PositionScaleTrainer('jepa',objective='position_6cm',seed=1,latent_dim=16);data=batch()
    if problem=='time':data['targets']['target_offsets_ns'][0,4]+=100_000_000
    if problem=='unknown':data['targets']['future_valid'][0,7]=True
    if problem=='privileged':data['inputs']['native_pose']=torch.zeros(2,7)
    if problem=='contact_motion':data['targets']['contact'][0,0]=1.
    if problem=='contact_range':data['targets']['contact'][0,0]=.5
    with pytest.raises(ValueError):trainer.step(data)


def test_same_observation_population_for_all_objectives_and_arms():
    trainer=PulseTrainer('jepa',seed=6,latent_dim=16);data=batch();populations=[]
    for objective in ('raw','position_6cm'):
        for arm in ('direct','supervised_rollout','jepa'):
            sizes=[];hook=trainer.model.encoder.register_forward_pre_hook(lambda m,args:sizes.append(len(args[0]['rgb'])))
            training_loss(trainer.model,data,arm,objective);hook.remove();populations.append(sizes)
    assert populations==[[8,10]]*6
