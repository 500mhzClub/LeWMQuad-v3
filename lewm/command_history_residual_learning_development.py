"""Learn RGB/body corrections around a frozen training-only command fit."""
import numpy as np
import torch
from lewm.eligible_floor_registration_development import bind
from lewm.nominal_motion_residual_learning_development import NominalResidualModel, nominal_motion
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer, training_loss
from lewm.pulse_timed_learning_development import active_parameters
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.navigation_artifact_root_development import BASE

REFERENCE=BASE/'go2_short_pulse_command_control_v1_attempt_001/command_only.npz'
PARAMETERIZATION='frozen_command_history_xy_yaw_plus_observation_residual_v1'


def compose(residual,reference,valid):
    reference=reference.to(residual.dtype)
    sine,cosine=reference[:,:,2].sin(),reference[:,:,2].cos()
    result=torch.cat((residual[:,:,:2]+reference[:,:,:2],
        (residual[:,:,2]*cosine+residual[:,:,3]*sine)[:,:,None],
        (residual[:,:,3]*cosine-residual[:,:,2]*sine)[:,:,None],residual[:,:,4:5]),-1)
    return torch.where(valid,result,torch.zeros_like(result))


class CommandHistoryResidualModel(NominalResidualModel):
    def __init__(self,latent_dim=32):
        super().__init__(latent_dim)
        with np.load(REFERENCE,allow_pickle=False) as archive:
            for name,shape in dict(mean=(8,447),scale=(8,447),coefficient=(8,447,3),bias=(8,3)).items():
                value=archive[name].copy()
                if value.shape!=shape or not np.isfinite(value).all():raise ValueError('invalid frozen command fit')
                if name=='scale' and (value<=0).any():raise ValueError('positive feature scales required')
                self.register_buffer('reference_'+name,torch.as_tensor(value,dtype=torch.float64))

    def reference_motion(self,history,blocks,valid):
        control=history['control'].to(torch.float64)
        if control.shape!=(len(blocks),4,15,7) or not torch.isfinite(control).all():
            raise ValueError('same four public command histories required')
        command=blocks[:,:,0].to(torch.float64)*blocks.new_tensor([.3,1.,.5],dtype=torch.float64)
        # nominal_motion also validates the known prefix and zero unknown padding.
        base=nominal_motion(blocks.to(torch.float64),valid)
        values=[]
        for h in range(8):
            known=torch.zeros_like(command);known[:,:h+1]=command[:,:h+1]
            x=torch.cat((known.flatten(1),base[:,h],control.flatten(1)),dim=1)
            values.append(base[:,h]+((x-self.reference_mean[h])/self.reference_scale[h])@
                          self.reference_coefficient[h]+self.reference_bias[h])
        result=torch.stack(values,dim=1)
        return torch.where(valid,result,torch.zeros_like(result))

    def forward(self,observation_history,known_action_blocks,known_action_valid):
        result=ObservationHorizonRGBBodyJEPA.forward(self,observation_history,known_action_blocks,known_action_valid)
        reference=self.reference_motion(observation_history,known_action_blocks,known_action_valid)
        for key in ('direct_outcomes','rollout_outcomes'):
            result[key]=compose(result[key],reference,known_action_valid)
        return result


def residual_loss(model,batch,condition):
    inputs=batch['inputs']
    reference=model.reference_motion(inputs['observation_history'],inputs['known_action_blocks'],inputs['known_action_valid'])
    return training_loss(model,batch,condition,
        outcome_transform=lambda residual,blocks,valid:compose(residual,reference,valid))


_step=bind(ObservationHorizonTrainer.step,training_loss=residual_loss)


class CommandHistoryResidualTrainer(ObservationHorizonTrainer):
    def __init__(self,condition,*,seed,latent_dim=32,learning_rate=1e-3,ema_momentum=.99):
        super().__init__(condition,seed=seed,latent_dim=latent_dim,learning_rate=learning_rate,ema_momentum=ema_momentum)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.model=CommandHistoryResidualModel(latent_dim).cpu()
        self.initial_sha256=state_digest(self.model.state_dict())
        self.parameters=active_parameters(self.model,condition)
        self.optimizer=torch.optim.AdamW(self.parameters,lr=learning_rate,weight_decay=0.)
        self.evaluation_only=False

    def step(self,batch):
        if self.evaluation_only:raise ValueError('checkpoint loaded for evaluation only')
        return _step(self,batch)

    def checkpoint(self):
        result=super().checkpoint()
        result['schema']='command_history_residual_training_development.v1'
        result['motion_parameterization']=PARAMETERIZATION
        return result
