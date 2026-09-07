"""Matched partial-time objectives; no optimizer, dataset discovery or training.

All arms expose their online encoder to the same past and available future
observations. RGB-target availability is independent of native outcome masks.
"""
import torch
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan
from lewm.temporal_rgb_body_learning_development import per_window_outcome_loss
from lewm.rgb_body_jepa_reference_development import variance_covariance_penalty

CONDITIONS=('direct','supervised_rollout','jepa')


def join_sample(observation_pair,native_targets):
    """Join already bound target-side products without adding inference inputs."""
    if set(observation_pair)!={'inputs','targets'} or native_targets.get('target_only') is not True:
        raise ValueError('explicit paired observations and target-only native labels required')
    t=observation_pair['targets']
    if not torch.equal(t['target_offsets_ns'],native_targets['target_offsets_ns']):
        raise ValueError('image/native target clock disagreement')
    return dict(inputs=observation_pair['inputs'],targets=t|{k:native_targets[k] for k in
        ('motion','motion_valid','contact','contact_valid')})


def active_parameters(model,condition):
    if condition not in CONDITIONS:raise ValueError('explicit matched condition required')
    modules=[model.encoder,model.history,model.history_norm,model.direct_plan,model.direct_decode]
    if condition!='direct':modules.extend([model.transition,model.rollout_decode])
    return [p for module in modules for p in module.parameters()]


def training_loss(model,batch,condition):
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
    # A target image can exist at contact; a native outcome can exist without
    # a target image. Do not inherit the old future_valid == motion_valid rule.
    shapes={'rgb':(len(z),8,3,96,128),'body':(len(z),8,20,63),'control':(len(z),8,15,7)}
    fo=targets['future_observations']
    if set(fo)!=set(shapes) or any(fo[k].shape!=s for k,s in shapes.items()):
        raise ValueError('actual future observation tensor shapes required')
    observed={k:v[fv] for k,v in fo.items()}
    tokens=torch.cat([blocks.flatten(2),mask.to(z.dtype)],-1)
    plans,_=model.direct_plan(tokens);seconds=offsets.to(z.dtype).unsqueeze(-1)/1e9
    direct=model.direct_decode(torch.cat([z[:,None].expand(-1,8,-1),plans,seconds],-1))
    direct=torch.where(active[:,:,None],direct,torch.zeros_like(direct))
    args=[targets[k] for k in ('motion','contact','motion_valid','contact_valid')]
    direct_loss=per_window_outcome_loss(direct,*args)
    future_z=model.encoder(observed) if fv.any() else z[:0]
    variance,covariance=variance_covariance_penalty(torch.cat([z,past.flatten(0,1),future_z]))
    total=direct_loss+.1*variance+.01*covariance
    parts=dict(direct_outcome=direct_loss,variance=variance,covariance=covariance)
    if condition!='direct':
        future=model.predict_latents(z,blocks,mask)
        rollout_loss=per_window_outcome_loss(model.decode_rollout(z,future,active,offsets),*args)
        total=total+rollout_loss;parts['rollout_outcome']=rollout_loss
        if condition=='jepa' and fv.any():
            # Equal weight per window among windows with actual target images;
            # shorter/censored windows do not silently acquire smaller weight.
            target=model.target(observed);squared=(future[fv]-target).square().mean(-1)
            counts=fv.sum(-1);groups=torch.repeat_interleave(torch.arange(len(z),device=z.device),counts)
            sums=torch.zeros(len(z),device=z.device,dtype=z.dtype).scatter_add(0,groups,squared)
            prediction=(sums[counts>0]/counts[counts>0]).mean()
            total=total+prediction;parts['latent_prediction']=prediction
    if not torch.isfinite(total):raise ValueError('nonfinite partial-time objective')
    return total,{k:float(v.detach()) for k,v in parts.items()}
