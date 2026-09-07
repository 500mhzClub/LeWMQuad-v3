"""Four-frame causal RGB/body JEPA reference with explicit known-plan boundaries."""
import copy

import torch
from torch import nn

from lewm.rgb_body_jepa_reference_development import ObservationEncoder,DirectOutcomeHead


def validate_plan(blocks,valid,batch):
    if blocks.shape!=(batch,8,5,3) or valid.shape!=(batch,8,5) or valid.dtype!=torch.bool:
        raise ValueError('plan requires B,8,5,3 commands and B,8,5 boolean validity')
    if blocks.device!=valid.device or not torch.isfinite(blocks).all(): raise ValueError('plan device/finiteness')
    if not torch.equal(valid,valid[:,:,:1].expand_as(valid)): raise ValueError('partial half-second block unsupported')
    active=valid.all(-1)
    if not active[:,0].all() or (active[:,1:]&~active[:,:-1]).any(): raise ValueError('nonempty contiguous known prefix required')
    if torch.count_nonzero(blocks[~valid]): raise ValueError('unknown commands must be masked zero padding')
    if (blocks.abs()>1+1e-6).any() or torch.count_nonzero(blocks[...,1]): raise ValueError('outside normalized fixed-action contract')
    return active


class TemporalRGBBodyJEPA(nn.Module):
    """Predict single future observation embeddings from chronological context.

    Future observations and outcome labels are absent from forward's signature.
    A future target is a single packet, not a fabricated future RGB history.
    """
    def __init__(self,latent_dim=128):
        super().__init__()
        self.encoder=ObservationEncoder(latent_dim)
        self.target_encoder=copy.deepcopy(self.encoder).requires_grad_(False)
        self.history=nn.GRU(latent_dim,latent_dim,batch_first=True)
        self.history_norm=nn.LayerNorm(latent_dim)
        self.direct=DirectOutcomeHead(latent_dim)
        self.transition=nn.Sequential(nn.Linear(latent_dim+15,256),nn.SiLU(),nn.Linear(256,latent_dim))
        self.rollout_decode=nn.Sequential(nn.Linear(2*latent_dim+1,256),nn.SiLU(),nn.Linear(256,5))

    def encode_history(self,history):
        if set(history)!={'rgb','body','control'}: raise ValueError('undeclared history modality')
        batch=history['rgb'].shape[0]
        shapes={'rgb':(batch,4,3,96,128),'body':(batch,4,20,63),'control':(batch,4,15,7)}
        if batch<1 or any(history[k].shape!=shape for k,shape in shapes.items()): raise ValueError('four-frame history shape')
        encoded=self.encoder({k:v.flatten(0,1) for k,v in history.items()}).reshape(batch,4,-1)
        _,hidden=self.history(encoded)
        return self.history_norm(hidden[-1]),encoded

    def predict_latents(self,z,blocks,valid):
        active=validate_plan(blocks,valid,len(z)); future=[]
        for i in range(8):
            indices=active[:,i]
            # Do not execute state transitions for missing commands. Invalid output
            # slots are placeholders, not stop-conditioned latent predictions.
            updated=z.clone()
            if indices.any():
                prior=z[indices]
                updated[indices]=prior+self.transition(torch.cat([prior,blocks[indices,i].flatten(1)],-1))
            z=updated
            future.append(torch.where(indices[:,None],z,torch.zeros_like(z)))
        return torch.stack(future,1)

    def decode_rollout(self,z,future,active):
        horizon=torch.arange(1,9,device=z.device,dtype=z.dtype)[None,:,None]*.5
        prediction=self.rollout_decode(torch.cat([z[:,None].expand_as(future),future,
            horizon.expand(len(z),-1,-1)],-1))
        return torch.where(active[:,:,None],prediction,torch.zeros_like(prediction))

    def forward(self,observation_history,known_action_blocks,known_action_valid):
        z,_=self.encode_history(observation_history)
        active=validate_plan(known_action_blocks,known_action_valid,len(z))
        direct=self.direct(z,known_action_blocks)
        future=self.predict_latents(z,known_action_blocks,known_action_valid)
        return {'latent':z,'future_latents':future,'prediction_valid':active,
            'direct_outcomes':torch.where(active[:,:,None],direct,torch.zeros_like(direct)),
            'rollout_outcomes':self.decode_rollout(z,future,active)}

    @torch.no_grad()
    def target(self,observation): return self.target_encoder(observation)

    @torch.no_grad()
    def update_target(self,momentum=.99):
        if not 0<=momentum<=1: raise ValueError('EMA momentum outside [0,1]')
        for target,online in zip(self.target_encoder.parameters(),self.encoder.parameters(),strict=True):
            target.mul_(momentum).add_(online,alpha=1-momentum)
