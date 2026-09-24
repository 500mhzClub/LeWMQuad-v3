"""Untrained JEPA/direct references with truthful partial action-block timing.

This does not modify the launched empirical controller or train a model.
Targets are actual observations at cumulative known-command boundary times;
unknown commands are masked padding, never silently extended braking.
"""
import torch
from torch import nn
from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA
from lewm.coupled_pulse_rollout_development import COMMANDS


def pulse_brake_plan(command,pulse_ticks):
    """A declared pulse followed by its guaranteed20zero-command prefix.

    CPU float32 normalized by [.3,1,.5], as in the existing data contract.
    Subsequent quiet-dependent braking/next actions are UNKNOWN at departure.
    Failure censoring still belongs to the target-side dataset, not this plan.
    """
    if (type(command) is not tuple or command not in COMMANDS
            or type(pulse_ticks) is not int or pulse_ticks not in (2,5)):
        raise ValueError('explicit measured pulse vocabulary required')
    actions=torch.zeros(40,3);valid=torch.zeros(40,dtype=torch.bool)
    actions[:pulse_ticks]=torch.tensor(command)/torch.tensor([.3,1.,.5])
    valid[:pulse_ticks+20]=True
    return actions.reshape(8,5,3),valid.reshape(8,5)


def validate_timed_plan(blocks,valid,batch):
    if blocks.shape!=(batch,8,5,3) or valid.shape!=(batch,8,5) or valid.dtype!=torch.bool:
        raise ValueError('B,8,5,3 actions and boolean B,8,5 validity required')
    if (batch<1 or blocks.device!=valid.device or not blocks.is_floating_point()
            or not torch.isfinite(blocks).all() or torch.count_nonzero(blocks[~valid])
            or torch.count_nonzero(blocks[...,1]) or (blocks.abs()>1+1e-6).any()):
        raise ValueError('finite normalized fixed-axis plan and zero unknown padding required')
    flat=valid.flatten(1)
    if not flat[:,0].all() or (flat[:,1:]&~flat[:,:-1]).any():
        raise ValueError('nonempty contiguous known tick prefix required')
    active=valid.any(-1)
    offsets_ns=valid.sum(-1).cumsum(-1)*100_000_000
    offsets_ns=torch.where(active,offsets_ns,torch.zeros_like(offsets_ns))
    return active,offsets_ns


class PulseTimedRGBBodyJEPA(TemporalRGBBodyJEPA):
    """Same causal four-image/body/control encoder, explicitly timed futures.

    Transition and direct heads consume action values AND tick validity. A
    partial final block predicts its true cumulative time, not the next0.5s
    boundary. No target tensors, native state or future images enter forward.
    These new weights are not compatible with the old transition checkpoint.
    """
    def __init__(self,latent_dim=128):
        super().__init__(latent_dim)
        self.transition=nn.Sequential(nn.Linear(latent_dim+20,256),nn.SiLU(),nn.Linear(256,latent_dim))
        self.direct_plan=nn.GRU(20,128,batch_first=True)
        self.direct_decode=nn.Sequential(nn.Linear(latent_dim+129,256),nn.SiLU(),nn.Linear(256,5))
        del self.direct

    def predict_latents(self,z,blocks,valid):
        active,_=validate_timed_plan(blocks,valid,len(z))
        tokens=torch.cat([blocks.flatten(2),valid.to(z.dtype)],-1)
        if tokens.device!=z.device or tokens.dtype!=z.dtype:raise ValueError('common model/plan dtype and device required')
        future=[]
        for i in range(8):
            selected=active[:,i];updated=z.clone()
            if selected.any():updated[selected]=z[selected]+self.transition(torch.cat([z[selected],tokens[selected,i]],-1))
            z=updated;future.append(torch.where(selected[:,None],z,torch.zeros_like(z)))
        return torch.stack(future,1)

    def decode_rollout(self,z,future,active,offsets_ns):
        seconds=offsets_ns.to(z.dtype).unsqueeze(-1)/1e9
        prediction=self.rollout_decode(torch.cat([z[:,None].expand_as(future),future,seconds],-1))
        return torch.where(active[:,:,None],prediction,torch.zeros_like(prediction))

    def forward(self,observation_history,known_action_blocks,known_action_valid):
        batch=observation_history['rgb'].shape[0]
        active,offsets=validate_timed_plan(known_action_blocks,known_action_valid,batch)
        z,_=self.encode_history(observation_history)
        future=self.predict_latents(z,known_action_blocks,known_action_valid)
        tokens=torch.cat([known_action_blocks.flatten(2),known_action_valid.to(z.dtype)],-1)
        seconds=offsets.to(z.dtype).unsqueeze(-1)/1e9
        planned,_=self.direct_plan(tokens)
        direct=self.direct_decode(torch.cat([z[:,None].expand(-1,8,-1),planned,seconds],-1))
        return dict(latent=z,future_latents=future,prediction_valid=active,target_offsets_ns=offsets,
                    direct_outcomes=torch.where(active[:,:,None],direct,torch.zeros_like(direct)),
                    rollout_outcomes=self.decode_rollout(z,future,active,offsets))
