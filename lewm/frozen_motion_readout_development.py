"""Fixed ridge readout from frozen action-conditioned motion features."""
import numpy as np
import torch
from lewm.observation_horizon_plan_development import validate_plan


def features(model,inputs):
    z,_=model.encode_history(inputs['observation_history'])
    blocks,valid=inputs['known_action_blocks'],inputs['known_action_valid']
    active,offsets=validate_plan(blocks,valid,len(z))
    future=model.predict_latents(z,blocks,valid)
    seconds=offsets.to(z.dtype).unsqueeze(-1)/1e9
    return model.rollout_decode[:-1](torch.cat((z[:,None].expand_as(future),future,seconds),dim=-1))


def fit_readout(x,y,weights,*,penalty=1.):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if (x.ndim!=2 or y.shape!=(len(x),4) or weights.shape!=(len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all()
            or (weights<=0).any() or penalty<=0):raise ValueError('finite labelled rows and positive fit weights required')
    mean=np.average(x,axis=0,weights=weights)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights));scale[scale<1e-8]=1.
    bias=np.average(y,axis=0,weights=weights);z=(x-mean)/scale
    coefficient=np.linalg.solve(z.T@(z*weights[:,None])+penalty*np.eye(x.shape[1]),
        z.T@((y-bias)*weights[:,None]))
    return dict(mean=mean,scale=scale,bias=bias,coefficient=coefficient)


def predict_readout(features,fit):
    return ((np.asarray(features,dtype=np.float64)-fit['mean'])/fit['scale'])@fit['coefficient']+fit['bias']


class FrozenMotionHead(torch.nn.Module):
    """Keep explicit standardization to avoid cancellation for low-variance features."""
    def __init__(self,original,fit):
        super().__init__();self.original=original
        width=original.weight.shape[1]
        for name,shape in dict(mean=(width,),scale=(width,),coefficient=(width,4),bias=(4,)).items():
            value=np.asarray(fit[name])
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('finite matching fitted head required')
            if name=='scale' and (value<=0).any():raise ValueError('positive feature scales required')
            self.register_buffer(name,torch.as_tensor(value.copy(),dtype=torch.float64))

    def forward(self,x):
        motion=((x.to(torch.float64)-self.mean)/self.scale)@self.coefficient+self.bias
        return torch.cat((motion.to(x.dtype),self.original(x)[...,4:5]),dim=-1)


def install_readout(model,fit):
    """Replace only four motion outputs; preserve the contact head and all features."""
    model.rollout_decode[-1]=FrozenMotionHead(model.rollout_decode[-1],fit)
    model.requires_grad_(False);model.eval()
    return model
