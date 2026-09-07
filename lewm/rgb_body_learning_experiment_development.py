"""Fixed development comparisons; target tensors never enter current inference."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from lewm.rgb_body_jepa_reference_development import masked_outcome_loss, variance_covariance_penalty

CONDITIONS=('direct','supervised_rollout','jepa')


def training_loss(model,batch,condition):
    if condition not in CONDITIONS: raise ValueError('unknown training condition')
    observation,plans,targets=batch['observation'],batch['known_action_blocks'],batch['targets']
    z=model.encoder(observation)
    direct=model.direct(z,plans)
    outcome_args=[targets[k] for k in ('motion','contact','motion_valid','contact_valid')]
    direct_loss=masked_outcome_loss(direct,*outcome_args)
    valid=targets['future_valid']
    future_observation={k:v[valid] for k,v in targets['future_observations'].items()}
    # Every condition trains on the identical current AND valid future image/body
    # population through this common regularizer. No condition gets extra samples.
    future_z=model.encoder(future_observation) if valid.any() else z[:0]
    variance,covariance=variance_covariance_penalty(torch.cat([z,future_z]))
    total=direct_loss+.1*variance+.01*covariance
    parts={'direct_outcome':direct_loss,'variance':variance,'covariance':covariance}
    if condition!='direct':
        predicted=model.predict_latents(z,plans)
        horizon=torch.arange(1,plans.shape[1]+1,dtype=z.dtype,device=z.device)[None,:,None]*.5
        decoded=model.rollout_decode(torch.cat([z[:,None].expand_as(predicted),predicted,
            horizon.expand(len(z),-1,-1)],dim=-1))
        rollout_loss=masked_outcome_loss(decoded,*outcome_args)
        total=total+rollout_loss
        parts['rollout_outcome']=rollout_loss
        if condition=='jepa' and valid.any():
            target=model.target(future_observation)
            prediction_loss=F.mse_loss(predicted[valid],target)
            total=total+prediction_loss
            parts['latent_prediction']=prediction_loss
    if not torch.isfinite(total): raise ValueError('nonfinite training objective')
    return total,{k:float(v.detach()) for k,v in parts.items()}


def active_parameters(model,condition):
    if condition not in CONDITIONS: raise ValueError('unknown training condition')
    modules=[model.encoder,model.direct]
    if condition!='direct': modules.extend([model.transition,model.rollout_decode])
    return [p for module in modules for p in module.parameters()]


def layout_batches(metadata,epoch,seed):
    """Each minibatch contains one action from each independent training layout."""
    layouts=sorted({r['layout_id'] for r in metadata})
    members={layout:{r['action_index']:i for i,r in enumerate(metadata) if r['layout_id']==layout} for layout in layouts}
    if len(layouts)!=16 or any(set(v)!=set(range(5)) for v in members.values()):
        raise ValueError('fixed study requires all five branches of sixteen training layouts')
    rng=np.random.default_rng(seed+epoch*1009)
    orders={layout:rng.permutation(5) for layout in layouts}
    for step in range(5):
        yield [members[layout][int(orders[layout][step])] for layout in layouts]


def prediction_metrics(prediction,targets,metadata,include_horizons=True):
    """Reduce within layout first; validity coverage is always reported."""
    prediction=np.asarray(prediction,dtype=np.float64)
    motion=np.asarray(targets['motion'],dtype=np.float64)
    contact=np.asarray(targets['contact'],dtype=np.float64)
    mv=np.asarray(targets['motion_valid'],dtype=bool); cv=np.asarray(targets['contact_valid'],dtype=bool)
    if prediction.shape!=(*mv.shape,5) or not np.isfinite(prediction).all(): raise ValueError('invalid predictions')
    probabilities=1/(1+np.exp(-np.clip(prediction[...,4],-60,60)))
    angle=np.arctan2(prediction[...,2],prediction[...,3])
    rows=[]
    for layout in sorted({r['layout_id'] for r in metadata}):
        indices=np.array([i for i,r in enumerate(metadata) if r['layout_id']==layout])
        p,m=prediction[indices],motion[indices]
        v,c=mv[indices],cv[indices]
        d=angle[indices][v]-m[...,2][v]
        rows.append({'layout_id':layout,'motion_count':int(v.sum()),'contact_count':int(c.sum()),
            'position_error_m':float(np.linalg.norm(p[...,:2][v]-m[...,:2][v],axis=-1).mean()) if v.any() else None,
            'yaw_error_rad':float(np.abs(np.arctan2(np.sin(d),np.cos(d))).mean()) if v.any() else None,
            'contact_brier':float(((probabilities[indices][c]-contact[indices][c])**2).mean()) if c.any() else None,
            'contact_accuracy_at_half':float(((probabilities[indices][c]>=.5)==contact[indices][c]).mean()) if c.any() else None})
    keys=('position_error_m','yaw_error_rad','contact_brier','contact_accuracy_at_half')
    result={'layout_macro':{k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys},
        'layouts':rows,'motion_valid':int(mv.sum()),'contact_valid':int(cv.sum()),'total_horizons':int(mv.size),
        'contact_positives':int(contact[cv].sum())}
    if include_horizons:
        result['by_horizon_seconds']={str((h+1)*.5):prediction_metrics(prediction[:,h:h+1],
            {k:np.asarray(targets[k])[:,h:h+1] for k in ('motion','contact','motion_valid','contact_valid')},metadata,False)
            for h in range(prediction.shape[1])}
        result['cumulative_contact_monotonicity_violation_fraction']=float((np.diff(probabilities,axis=1)<-1e-6).mean()) if prediction.shape[1]>1 else 0.
    return result


def simple_predictions(train,validation):
    """Training-only empirical action means, zero motion, and command integration."""
    n=len(validation['metadata']); shape=(n,8,5)
    action_mean=np.zeros(shape); persistence=np.zeros(shape); kinematic=np.zeros(shape)
    persistence[...,3]=1.; kinematic[...,3]=1.
    # Zero-contact persistence/kinematics are explicit heuristics, not calibrated.
    persistence[...,4]=kinematic[...,4]=-30.
    targets=train['targets']
    for i,meta in enumerate(validation['metadata']):
        selected=np.array([m['action_index']==meta['action_index'] for m in train['metadata']])
        for h in range(8):
            mv=np.asarray(targets['motion_valid'])[:,h]&selected
            cv=np.asarray(targets['contact_valid'])[:,h]&selected
            if not mv.any() or not cv.any(): raise ValueError('missing action-only training target')
            values=np.asarray(targets['motion'])[mv,h]
            action_mean[i,h,:2]=values[:,:2].mean(0)
            action_mean[i,h,2]=np.sin(values[:,2]).mean(); action_mean[i,h,3]=np.cos(values[:,2]).mean()
            p=float(np.asarray(targets['contact'])[cv,h].mean()); p=np.clip(p,1e-6,1-1e-6)
            action_mean[i,h,4]=math.log(p/(1-p))
        xy=np.zeros(2); yaw=0.
        plan=np.asarray(validation['known_action_blocks'][i]).reshape(40,3)*[.3,1.,.5]
        for t,(vx,vy,w) in enumerate(plan):
            # Exact planar constant-twist displacement for each 0.1-s command.
            if abs(w)<1e-8: body_delta=np.array([vx,vy])*.1
            else:
                a=w*.1
                body_delta=np.array([math.sin(a)*vx-(1-math.cos(a))*vy,(1-math.cos(a))*vx+math.sin(a)*vy])/w
            rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            xy+=rotation@body_delta; yaw+=w*.1
            if (t+1)%5==0: kinematic[i,t//5,:4]=[*xy,math.sin(yaw),math.cos(yaw)]
    return {'training_action_mean':action_mean,'zero_motion_no_contact':persistence,'command_kinematics_no_contact':kinematic}
