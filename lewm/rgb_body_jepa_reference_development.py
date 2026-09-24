"""Small action-conditioned JEPA/direct references; no data discovery or oracle inputs.

This is a new navigation-oriented reference, not an I-JEPA reproduction. Models
consume only explicit RGB, chronological body/control features and known plans.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F


class ObservationEncoder(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()
        self.visual=nn.Sequential(nn.Conv2d(3,16,5,stride=2,padding=2),nn.SiLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((3,4)),
            nn.Flatten(),nn.Linear(1536,128),nn.SiLU())
        self.body=nn.GRU(63,64,batch_first=True)
        self.controls=nn.GRU(7,32,batch_first=True)
        self.fuse=nn.Sequential(nn.Linear(224,latent_dim),nn.SiLU(),nn.LayerNorm(latent_dim))

    def forward(self,observation):
        if set(observation)!={'rgb','body','control'}: raise ValueError('undeclared encoder inputs')
        rgb,body,control=(observation[k] for k in ('rgb','body','control'))
        batch=rgb.shape[0]
        if rgb.shape!=(batch,3,96,128) or body.shape!=(batch,20,63) or control.shape!=(batch,15,7):
            raise ValueError('encoder tensor shape mismatch')
        if not all(torch.isfinite(v).all() for v in (rgb,body,control)): raise ValueError('nonfinite encoder input')
        _,bh=self.body(body)
        _,ch=self.controls(control)
        return self.fuse(torch.cat([self.visual(rgb),bh[-1],ch[-1]],dim=-1))


class DirectOutcomeHead(nn.Module):
    """Encode the known action prefix; never recurse a predicted world state."""
    def __init__(self,latent_dim=128):
        super().__init__()
        self.plan=nn.GRU(15,128,batch_first=True)
        self.decode=nn.Sequential(nn.Linear(latent_dim+129,256),nn.SiLU(),nn.Linear(256,5))

    def forward(self,z,action_blocks):
        if action_blocks.ndim!=4 or action_blocks.shape[0]!=len(z) or action_blocks.shape[2:]!=(5,3):
            raise ValueError('known action plan must be B,H,5,3')
        plans,_=self.plan(action_blocks.flatten(2))
        horizons=torch.arange(1,plans.shape[1]+1,device=z.device,dtype=z.dtype)[None,:,None]*.5
        return self.decode(torch.cat([z[:,None].expand(-1,plans.shape[1],-1),plans,
            horizons.expand(len(z),-1,-1)],dim=-1))


class RGBBodyJEPAReference(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()
        self.encoder=ObservationEncoder(latent_dim)
        self.target_encoder=copy.deepcopy(self.encoder).requires_grad_(False)
        self.direct=DirectOutcomeHead(latent_dim)
        self.transition=nn.Sequential(nn.Linear(latent_dim+15,256),nn.SiLU(),nn.Linear(256,latent_dim))
        self.rollout_decode=nn.Sequential(nn.Linear(2*latent_dim+1,256),nn.SiLU(),nn.Linear(256,5))

    @torch.no_grad()
    def update_target(self,momentum):
        if not 0<=momentum<=1: raise ValueError('EMA momentum outside [0,1]')
        for target,online in zip(self.target_encoder.parameters(),self.encoder.parameters(),strict=True):
            target.mul_(momentum).add_(online,alpha=1-momentum)

    @torch.no_grad()
    def target(self,observation):
        return self.target_encoder(observation)

    def predict_latents(self,z,action_blocks):
        if action_blocks.ndim!=4 or action_blocks.shape[0]!=len(z) or action_blocks.shape[2:]!=(5,3) or action_blocks.shape[1]<1:
            raise ValueError('known action plan must be nonempty B,H,5,3')
        future=[]
        for action in action_blocks.unbind(1):
            z=z+self.transition(torch.cat([z,action.flatten(1)],dim=-1))
            future.append(z)
        return torch.stack(future,dim=1)

    def forward(self,observation,action_blocks):
        z=self.encoder(observation)
        future=self.predict_latents(z,action_blocks)
        horizon=torch.arange(1,future.shape[1]+1,device=z.device,dtype=z.dtype)[None,:,None]*.5
        rollout=self.rollout_decode(torch.cat([z[:,None].expand_as(future),future,
            horizon.expand(len(z),-1,-1)],dim=-1))
        return {'latent':z,'future_latents':future,'direct_outcomes':self.direct(z,action_blocks),'rollout_outcomes':rollout}

    def direct_prediction(self,observation,action_blocks):
        return self.direct(self.encoder(observation),action_blocks)


def variance_covariance_penalty(z):
    """VICReg-style anti-collapse terms; not a guarantee of task utility."""
    if z.ndim!=2 or len(z)<2: raise ValueError('regularizer requires at least two observations')
    centered=z-z.mean(dim=0)
    std=torch.sqrt(centered.square().sum(0)/(len(z)-1)+1e-4)
    covariance=centered.T@centered/(len(z)-1)
    off_diagonal=covariance-torch.diag_embed(covariance.diagonal())
    return F.relu(1-std).mean(),off_diagonal.square().sum()/z.shape[1]


def masked_outcome_loss(prediction,motion,contact,motion_valid,contact_valid):
    """Prediction channels are dx,dy,sin(dyaw),cos(dyaw),contact logit."""
    if prediction.shape[-1]!=5 or motion.shape!=prediction.shape[:-1]+(3,): raise ValueError('outcome shape')
    shape=prediction.shape[:-1]
    if any(v.shape!=shape for v in (contact,motion_valid,contact_valid)) or motion_valid.dtype!=torch.bool or contact_valid.dtype!=torch.bool:
        raise ValueError('outcome validity shape/type')
    # Index before arithmetic: invalid censored NaNs must never reach a loss.
    result=prediction.sum()*0
    if motion_valid.any():
        measured=motion[motion_valid]
        target=torch.cat([measured[:,:2],torch.sin(measured[:,2:3]),torch.cos(measured[:,2:3])],dim=-1)
        result=result+F.smooth_l1_loss(prediction[motion_valid][:,:4],target)
    if contact_valid.any():
        result=result+F.binary_cross_entropy_with_logits(prediction[...,4][contact_valid],contact[contact_valid])
    return result
