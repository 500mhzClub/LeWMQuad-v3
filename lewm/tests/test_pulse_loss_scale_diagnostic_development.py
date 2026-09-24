import pytest
import torch
from lewm.pulse_loss_scale_diagnostic_development import diagnose, outcome_terms
from lewm.pulse_timed_training_runner_development import PulseTrainer,state_digest
from lewm.tests.test_pulse_timed_learning_development import batch


@pytest.mark.parametrize('condition',['direct','supervised_rollout','jepa'])
def test_exact_frozen_loss_gradient_and_unchanged_model(condition):
    trainer = PulseTrainer(condition,seed=42,latent_dim=16)
    state = state_digest(trainer.model.state_dict())
    report = diagnose(trainer.model,batch(),condition)
    assert report['total_loss'] == pytest.approx(report['frozen_loss'],rel=2e-6)
    assert state_digest(trainer.model.state_dict()) == state
    assert all(p.grad is None for p in trainer.model.parameters())
    assert not trainer.optimizer.state and trainer.updates == 0
    assert report['terms']['direct_position']['gradient_l2'] > 0
    assert ('latent_prediction' in report['terms']) == (condition=='jepa')


def test_raw_centimetre_position_term_scale_and_censoring():
    p = torch.zeros(1,8,5,requires_grad=True)
    motion = torch.full((1,8,3),float('nan')); motion[0,0] = torch.tensor([.01,0.,0.])
    mask = torch.zeros(1,8,dtype=torch.bool); mask[0,0] = True
    contact = torch.full((1,8),float('nan')); contact[0,0] = 0
    terms = outcome_terms(p,dict(motion=motion,contact=contact,motion_valid=mask,contact_valid=mask))
    assert float(terms['position'].detach()) == pytest.approx(.01**2/8)
    assert float(terms['angle'].detach()) == pytest.approx(.125)
    assert float(terms['contact'].detach()) == pytest.approx(.69314718)
    gradient, = torch.autograd.grad(sum(terms.values()),p)
    assert torch.count_nonzero(gradient[~mask]) == 0
