"""RGB/body-conditioned corrections around a known-command motion reference.

Training compares composed absolute outcomes with the original absolute labels.
Latent prediction, contact semantics and loss weights retain their definitions.
"""
from functools import partial
import torch
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer, training_loss
from lewm.observation_horizon_plan_development import validate_plan
from lewm.pulse_timed_learning_development import active_parameters
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.eligible_floor_registration_development import bind

PARAMETERIZATION = 'command_integrated_xy_yaw_plus_observation_residual_v1'


def nominal_motion(blocks, valid):
    active, _ = validate_plan(blocks, valid, len(blocks))
    command = blocks[:, :, 0]*blocks.new_tensor([.3, 1., .5])
    yaw = (command[:, :, 2]*.1).cumsum(1)
    middle = yaw-command[:, :, 2]*.05
    dx = .1*(command[:, :, 0]*middle.cos()-command[:, :, 1]*middle.sin())
    dy = .1*(command[:, :, 0]*middle.sin()+command[:, :, 1]*middle.cos())
    result = torch.stack((dx.cumsum(1),dy.cumsum(1),yaw),dim=-1)
    return torch.where(active[:, :, None],result,torch.zeros_like(result))


def compose_outcomes(residual, blocks, valid):
    nominal = nominal_motion(blocks,valid)
    active = valid[:, :, 0]
    sine, cosine = nominal[:, :, 2].sin(), nominal[:, :, 2].cos()
    result = torch.cat((residual[:, :, :2]+nominal[:, :, :2],
        (residual[:, :, 2]*cosine+residual[:, :, 3]*sine)[:, :, None],
        (residual[:, :, 3]*cosine-residual[:, :, 2]*sine)[:, :, None],
        residual[:, :, 4:5]),dim=-1)
    return torch.where(active[:, :, None],result,torch.zeros_like(result))


class NominalResidualModel(ObservationHorizonRGBBodyJEPA):
    def __init__(self,latent_dim=32):
        super().__init__(latent_dim)
        with torch.no_grad():
            for head in (self.direct_decode,self.rollout_decode):
                head[-1].weight[:4].zero_()
                head[-1].bias[:4].copy_(head[-1].bias.new_tensor([0.,0.,0.,1.]))

    def forward(self,observation_history,known_action_blocks,known_action_valid):
        result=super().forward(observation_history,known_action_blocks,known_action_valid)
        for key in ('direct_outcomes','rollout_outcomes'):
            result[key]=compose_outcomes(result[key],known_action_blocks,known_action_valid)
        return result


_residual_step=bind(ObservationHorizonTrainer.step,
    training_loss=partial(training_loss,outcome_transform=compose_outcomes))


class NominalResidualTrainer(ObservationHorizonTrainer):
    def __init__(self,condition,*,seed,latent_dim=32,learning_rate=1e-3,ema_momentum=.99):
        super().__init__(condition,seed=seed,latent_dim=latent_dim,
                         learning_rate=learning_rate,ema_momentum=ema_momentum)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); self.model=NominalResidualModel(latent_dim).cpu()
        self.initial_sha256=state_digest(self.model.state_dict())
        self.parameters=active_parameters(self.model,condition)
        self.optimizer=torch.optim.AdamW(self.parameters,lr=learning_rate,weight_decay=0.)
        self.evaluation_only=False

    def step(self,batch):
        if self.evaluation_only:
            raise ValueError('checkpoint loaded for evaluation only')
        return _residual_step(self,batch)

    def checkpoint(self):
        result=super().checkpoint()
        result['schema']='nominal_motion_residual_training_development.v1'
        result['motion_parameterization']=PARAMETERIZATION
        return result
