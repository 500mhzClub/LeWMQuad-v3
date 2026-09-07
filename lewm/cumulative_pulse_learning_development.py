"""Matched cumulative-event direct/rollout/JEPA objectives; new fresh fits only."""
from copy import deepcopy
import math
import torch
from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA,cumulative_outcomes
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan
from lewm.pulse_timed_learning_development import CONDITIONS,active_parameters
from lewm.pulse_position_scale_learning_development import scaled_outcome_loss,POSITION_SCALE_M
from lewm.rgb_body_jepa_reference_development import variance_covariance_penalty
from lewm.pulse_timed_training_runner_development import state_digest

def training_loss(model,batch,condition):
    if not isinstance(model,CumulativePulseRGBBodyJEPA):
        raise ValueError('same cumulative contact semantics required in training and inference')
    if condition not in CONDITIONS:raise ValueError('explicit matched condition required')
    if set(batch)!={'inputs','targets'}:raise ValueError('explicit input/target separation required')
    inputs,targets=batch['inputs'],batch['targets']
    if set(inputs)!={'observation_history','known_action_blocks','known_action_valid'}:
        raise ValueError('undeclared inference fields')
    if set(targets)!={'motion','motion_valid','contact','contact_valid','future_observations','future_valid','target_offsets_ns'}:
        raise ValueError('undeclared target fields')
    blocks,mask=inputs['known_action_blocks'],inputs['known_action_valid']
    z,past=model.encode_history(inputs['observation_history'])
    active,offsets=validate_timed_plan(blocks,mask,len(z))
    if not torch.equal(targets['target_offsets_ns'],offsets):raise ValueError('exact partial target timestamps required')
    for name in ('motion_valid','contact_valid','future_valid'):
        v=targets[name]
        if v.shape!=active.shape or v.dtype!=torch.bool or v.device!=active.device or (v&~active).any():
            raise ValueError('target outside known plan')
    mv,cv,fv=(targets[k] for k in ('motion_valid','contact_valid','future_valid'))
    if targets['motion'].shape!=(*active.shape,3) or targets['contact'].shape!=active.shape:
        raise ValueError('outcome shape mismatch')
    if ((mv&~cv).any() or not torch.isfinite(targets['motion'][mv]).all()
            or not torch.isfinite(targets['contact'][cv]).all()
            or ((targets['contact'][cv]!=0)&(targets['contact'][cv]!=1)).any()
            or targets['contact'][mv].any()):raise ValueError('invalid native outcome censoring')
    positive_seen=torch.where(cv,targets['contact'],torch.zeros_like(targets['contact'])).cumsum(-1)>0
    if (positive_seen & cv & (targets['contact']==0)).any():
        raise ValueError('cumulative contact truth cannot revert from observed1 to observed0')
    shapes={'rgb':(len(z),8,3,96,128),'body':(len(z),8,20,63),'control':(len(z),8,15,7)}
    fo=targets['future_observations']
    if set(fo)!=set(shapes) or any(fo[k].shape!=s for k,s in shapes.items()):
        raise ValueError('actual future observation tensor shapes required')
    observed={k:v[fv] for k,v in fo.items()}
    tokens=torch.cat([blocks.flatten(2),mask.to(z.dtype)],-1)
    plans,_=model.direct_plan(tokens);seconds=offsets.to(z.dtype).unsqueeze(-1)/1e9
    direct=model.direct_decode(torch.cat([z[:,None].expand(-1,8,-1),plans,seconds],-1))
    direct=cumulative_outcomes(direct,active,offsets)
    args=[targets[k] for k in ('motion','contact','motion_valid','contact_valid')]
    direct_loss=scaled_outcome_loss(direct,*args)
    future_z=model.encoder(observed) if fv.any() else z[:0]
    variance,covariance=variance_covariance_penalty(torch.cat([z,past.flatten(0,1),future_z]))
    total=direct_loss+.1*variance+.01*covariance
    parts=dict(direct_outcome=direct_loss,variance=variance,covariance=covariance)
    if condition!='direct':
        future=model.predict_latents(z,blocks,mask)
        rollout_loss=scaled_outcome_loss(model.decode_rollout(z,future,active,offsets),*args)
        total=total+rollout_loss;parts['rollout_outcome']=rollout_loss
        if condition=='jepa' and fv.any():
            target=model.target(observed);squared=(future[fv]-target).square().mean(-1)
            counts=fv.sum(-1);groups=torch.repeat_interleave(torch.arange(len(z),device=z.device),counts)
            sums=torch.zeros(len(z),device=z.device,dtype=z.dtype).scatter_add(0,groups,squared)
            prediction=(sums[counts>0]/counts[counts>0]).mean()
            total=total+prediction;parts['latent_prediction']=prediction
    if not torch.isfinite(total):raise ValueError('nonfinite partial-time objective')
    return total,{k:float(v.detach()) for k,v in parts.items()}


class CumulativePulseTrainer:
    def __init__(self,condition,*,seed,latent_dim=32,learning_rate=1e-3,ema_momentum=.99):
        if condition not in CONDITIONS or type(seed) is not int or seed<0:
            raise ValueError('explicit condition and nonnegative seed required')
        if type(latent_dim) is not int or not 8<=latent_dim<=128:
            raise ValueError('bounded latent width required')
        if not math.isfinite(learning_rate) or not 0<learning_rate<=.01 or not 0<=ema_momentum<=1:
            raise ValueError('bounded learning rate and EMA required')
        self.condition=condition;self.seed=seed;self.latent_dim=latent_dim
        self.learning_rate=learning_rate;self.ema_momentum=ema_momentum
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.model=CumulativePulseRGBBodyJEPA(latent_dim).cpu()
        self.initial_sha256=state_digest(self.model.state_dict())
        self.parameters=active_parameters(self.model,condition)
        self.optimizer=torch.optim.AdamW(self.parameters,lr=learning_rate,weight_decay=0.)
        self.updates=0;self.failed=False

    def step(self,batch):
        if self.failed:raise ValueError('training failure latched; no implicit retry')
        try:
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            loss,parts=training_loss(self.model,batch,self.condition);loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.parameters):
                raise ValueError('finite gradients required for every active parameter')
            norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True)
            self.optimizer.step()
            if not all(torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError('nonfinite model after optimizer step')
            # Equal EMA maintenance for every arm; only JEPA consumes the target
            # in its latent-prediction objective. Update strictly after optimizer.
            self.model.update_target(self.ema_momentum)
            if any(p.grad is not None for p in self.model.target_encoder.parameters()):
                raise ValueError('EMA encoder must stay gradient-free')
            self.updates+=1
            return dict(update=self.updates,loss=float(loss.detach()),parts=parts,
                gradient_norm_before_clip=float(norm),model_sha256=state_digest(self.model.state_dict()))
        except Exception:
            self.failed=True;raise

    def checkpoint(self):
        return deepcopy(dict(schema='cumulative_pulse_training_development.v1',
            contact_semantics='integrated_softplus_hazard_per_second',position_scale_m=POSITION_SCALE_M,
            condition=self.condition,seed=self.seed,latent_dim=self.latent_dim,
            learning_rate=self.learning_rate,ema_momentum=self.ema_momentum,
            updates=self.updates,failed=self.failed,initial_sha256=self.initial_sha256,
            model_sha256=state_digest(self.model.state_dict()),model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),navigation_qualified=False))


