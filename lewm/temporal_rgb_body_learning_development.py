"""Matched temporal objectives and layout-balanced schedules; no filesystem access."""
import numpy as np
import torch
import torch.nn.functional as F

from lewm.rgb_body_jepa_reference_development import masked_outcome_loss,variance_covariance_penalty
from lewm.temporal_rgb_body_jepa_development import validate_plan

CONDITIONS=('direct','supervised_rollout','jepa')


def per_window_outcome_loss(prediction,motion,contact,motion_valid,contact_valid):
    # With one sampled window per layout, this gives each layout equal outcome
    # weight despite variable remaining duration. Missing motion contributes no
    # motion loss, while an observed contact still contributes its own loss.
    return torch.stack([masked_outcome_loss(prediction[i:i+1],motion[i:i+1],contact[i:i+1],
        motion_valid[i:i+1],contact_valid[i:i+1]) for i in range(len(prediction))]).mean()


def active_parameters(model,condition):
    if condition not in CONDITIONS: raise ValueError('unknown condition')
    modules=[model.encoder,model.history,model.history_norm,model.direct]
    if condition!='direct': modules.extend([model.transition,model.rollout_decode])
    return [p for module in modules for p in module.parameters()]


def training_loss(model,batch,condition):
    if condition not in CONDITIONS: raise ValueError('unknown condition')
    if set(batch)!={'observation_history','known_action_blocks','known_action_valid','targets','metadata'}:
        raise ValueError('undeclared temporal training fields')
    blocks,mask,targets=batch['known_action_blocks'],batch['known_action_valid'],batch['targets']
    z,past=model.encode_history(batch['observation_history'])
    active=validate_plan(blocks,mask,len(z)); valid=targets['future_valid']
    for name in ('motion_valid','contact_valid','future_valid'):
        v=targets[name]
        if v.shape!=active.shape or v.dtype!=torch.bool or (v&~active).any(): raise ValueError('target outside known plan')
    if not torch.equal(valid,targets['motion_valid']): raise ValueError('future/motion censoring mismatch')
    if (targets['motion_valid']&~targets['contact_valid']).any(): raise ValueError('motion without observed contact state')
    if targets['contact'][valid].any(): raise ValueError('future or motion at/after contact')
    args=[targets[k] for k in ('motion','contact','motion_valid','contact_valid')]
    direct=model.direct(z,blocks); direct_loss=per_window_outcome_loss(direct,*args)
    future_observation={k:v[valid] for k,v in targets['future_observations'].items()}
    # The same actual past and future image population trains every condition's
    # encoder. EMA targets get no gradients and add no new image exposure.
    observed=model.encoder(future_observation) if valid.any() else z[:0]
    variance,covariance=variance_covariance_penalty(torch.cat([z,past.flatten(0,1),observed]))
    total=direct_loss+.1*variance+.01*covariance
    parts={'direct_outcome':direct_loss,'variance':variance,'covariance':covariance}
    if condition!='direct':
        future=model.predict_latents(z,blocks,mask)
        rollout_loss=per_window_outcome_loss(model.decode_rollout(z,future,active),*args)
        total=total+rollout_loss; parts['rollout_outcome']=rollout_loss
        if condition=='jepa' and valid.any():
            prediction=F.mse_loss(future[valid],model.target(future_observation))
            total=total+prediction; parts['latent_prediction']=prediction
    if not torch.isfinite(total): raise ValueError('nonfinite temporal objective')
    return total,{k:float(v.detach()) for k,v in parts.items()}


def layout_batches(metadata,epoch,seed):
    """Five steps: each layout contributes each action once, with a sampled offset.

    Offsets are sampled uniformly among that branch's actually observed windows;
    no absent post-contact state is invented. Identical schedule across conditions.
    """
    if isinstance(epoch,bool) or not isinstance(epoch,int) or epoch<0: raise ValueError('nonnegative epoch required')
    if any(r['data_role']!='train' for r in metadata): raise ValueError('training role required')
    layouts=sorted({r['layout_id'] for r in metadata})
    members={layout:{a:[] for a in range(5)} for layout in layouts}
    identities=set()
    for i,row in enumerate(metadata):
        identity=(row['layout_id'],row['action_index'],row['offset_ns'])
        if (identity in identities or row['action_index'] not in range(5)
                or row['offset_ns'] not in range(0,4_000_000_000,500_000_000)):
            raise ValueError('invalid/duplicate temporal window')
        identities.add(identity); members[row['layout_id']][row['action_index']].append(i)
    if len(layouts)!=16 or any(not rows for actions in members.values() for rows in actions.values()):
        raise ValueError('all five actions of sixteen training layouts required')
    rng=np.random.default_rng(seed+epoch*1009)
    orders={layout:rng.permutation(5) for layout in layouts}
    for step in range(5):
        yield [int(rng.choice(members[layout][int(orders[layout][step])])) for layout in layouts]
