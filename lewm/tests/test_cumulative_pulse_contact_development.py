"""Structural probability semantics and actual training/inference consistency."""
from copy import deepcopy
import pytest
import torch
from torch.nn import functional as F
from lewm.cumulative_pulse_contact_development import cumulative_contact_logits,cumulative_outcomes,CumulativePulseRGBBodyJEPA
from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer,training_loss
from lewm.pulse_timed_training_runner_development import PulseTrainer,state_digest
from lewm.pulse_position_scale_learning_development import scaled_outcome_loss
from lewm.pulse_timed_learning_development import active_parameters,CONDITIONS
from lewm.tests.test_pulse_timed_learning_development import batch


def clocks():
    offsets=torch.tensor([[500_000_000,1_000_000_000,1_500_000_000,2_000_000_000,2_200_000_000,0,0,0],
        [500_000_000,1_000_000_000,1_500_000_000,2_000_000_000,2_500_000_000,0,0,0]])
    return offsets,offsets>0


def test_constant_hazard_matches_exponential_event_law_at_actual_partial_endpoints():
    t,a=clocks();raw=torch.zeros((2,8),dtype=torch.float64,requires_grad=True)
    logits=cumulative_contact_logits(raw,t,a)
    expected=-torch.expm1(-F.softplus(raw)*t/1e9)
    torch.testing.assert_close(logits[a].sigmoid(),expected[a],rtol=1e-12,atol=1e-12)
    assert logits[0,4]<logits[1,4] and not logits[~a].any()


@pytest.mark.parametrize('value',[-1000.,-100.,-20.00001,-20.,0.,20.,1000.])
def test_extreme_finite_hazards_have_finite_logits_and_gradients(value):
    t,a=clocks();raw=torch.full((2,8),value,requires_grad=True)
    logits=cumulative_contact_logits(raw,t,a);F.binary_cross_entropy_with_logits(logits[a],torch.zeros_like(logits[a])).backward()
    assert torch.isfinite(logits).all() and torch.isfinite(raw.grad).all()
    assert (torch.diff(logits[:,:5],dim=-1)>=0).all() and not raw.grad[~a].any()


def test_causal_prefix_and_unknown_padding_cannot_change_earlier_probabilities():
    t,a=clocks();raw=torch.linspace(-4,4,16).reshape(2,8).requires_grad_()
    before=cumulative_contact_logits(raw,t,a);changed=raw.detach().clone();changed[:,3:]=123.
    after=cumulative_contact_logits(changed,t,a)
    torch.testing.assert_close(before[:,:3],after[:,:3],rtol=0,atol=0)
    before[:,2].sum().backward();assert not raw.grad[:,3:].any()


@pytest.mark.parametrize('fault',['time','unknown','hole','dtype','nan','empty'])
def test_invalid_hazard_clock_contract_rejected(fault):
    t,a=clocks();raw=torch.zeros(2,8)
    if fault=='time':t[:,1]=t[:,0]
    elif fault=='unknown':t[:,7]=100
    elif fault=='hole':a[:,2]=False;t[:,2]=0
    elif fault=='dtype':t=t.float()
    elif fault=='nan':raw[0,0]=float('nan')
    else:t=t[:0];a=a[:0];raw=raw[:0]
    with pytest.raises(ValueError):cumulative_contact_logits(raw,t,a)


def test_contact_transform_preserves_motion_exactly():
    t,a=clocks();raw=torch.randn(2,8,5)
    result=cumulative_outcomes(raw,a,t)
    torch.testing.assert_close(result[a][:,:4],raw[a][:,:4],rtol=0,atol=0)
    assert not result[~a].any()


@pytest.mark.parametrize('condition',CONDITIONS)
def test_same_initial_parameters_and_matched_training_inference_contact_semantics(condition):
    old=PulseTrainer(condition,seed=17,latent_dim=16);new=CumulativePulseTrainer(condition,seed=17,latent_dim=16)
    assert old.initial_sha256==new.initial_sha256
    b=batch();out=new.model(**b['inputs']);loss,parts=training_loss(new.model,b,condition)
    args=[b['targets'][k] for k in ('motion','contact','motion_valid','contact_valid')]
    assert parts['direct_outcome']==float(scaled_outcome_loss(out['direct_outcomes'],*args).detach())
    if condition!='direct':
        assert parts['rollout_outcome']==float(scaled_outcome_loss(out['rollout_outcomes'],*args).detach())
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active_parameters(new.model,condition))
    for head in ('direct_outcomes','rollout_outcomes'):
        assert (torch.diff(out[head][:,:5,4],dim=-1)>=0).all()
    new.step(b);checkpoint=new.checkpoint()
    assert checkpoint['schema']=='cumulative_pulse_training_development.v1' and checkpoint['updates']==1
    assert checkpoint['contact_semantics']=='integrated_softplus_hazard_per_second'
    assert all(p.grad is None for p in new.model.target_encoder.parameters())


def test_observed_event_cannot_revert_even_across_censored_slots():
    b=batch();t=b['targets'];t['motion_valid'][0,0]=False;t['motion'][0,0]=float('nan');t['contact'][0,0]=1.
    t['motion_valid'][0,1]=False;t['motion'][0,1]=float('nan');t['contact_valid'][0,1]=False;t['contact'][0,1]=float('nan')
    with pytest.raises(ValueError,match='cannot revert'):training_loss(CumulativePulseRGBBodyJEPA(16),b,'jepa')


def test_bad_target_latches_new_trainer_without_an_optimizer_update():
    trainer=CumulativePulseTrainer('jepa',seed=7,latent_dim=16);before=state_digest(trainer.model.state_dict());b=batch()
    b['targets']['target_offsets_ns'][0,4]+=100_000_000
    with pytest.raises(ValueError):trainer.step(b)
    assert trainer.failed and trainer.updates==0 and state_digest(trainer.model.state_dict())==before
    with pytest.raises(ValueError,match='latched'):trainer.step(batch())


def test_target_only_future_changes_do_not_change_forward_or_past_exposure():
    b=batch();model=CumulativePulseRGBBodyJEPA(16);before=model(**b['inputs']);changed=deepcopy(b)
    changed['targets']['motion'].fill_(999.)
    for v in changed['targets']['future_observations'].values():v.fill_(123.)
    after=model(**changed['inputs'])
    for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
    populations=[]
    for condition in CONDITIONS:
        sizes=[];hook=model.encoder.register_forward_pre_hook(lambda m,args:sizes.append(len(args[0]['rgb'])))
        training_loss(model,b,condition);hook.remove();populations.append(sizes)
    assert populations==[[8,10]]*3
