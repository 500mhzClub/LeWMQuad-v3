"""Read-only decomposition of the frozen pulse objective, not a new objective."""
import torch
from torch.nn import functional as F
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan
from lewm.pulse_timed_learning_development import active_parameters, training_loss
from lewm.rgb_body_jepa_reference_development import variance_covariance_penalty


def outcome_terms(prediction, targets):
    terms = {'position':[], 'angle':[], 'contact':[]}
    for i,p in enumerate(prediction):
        mv,cv = targets['motion_valid'][i], targets['contact_valid'][i]
        motion,contact = targets['motion'][i],targets['contact'][i]
        zero = p.sum()*0
        xy,angle,risk = zero,zero,zero
        if mv.any():
            xy = .5*F.smooth_l1_loss(p[mv,:2],motion[mv,:2])
            yaw = motion[mv,2]
            angle = .5*F.smooth_l1_loss(p[mv,2:4],torch.stack([yaw.sin(),yaw.cos()],-1))
        if cv.any(): risk = F.binary_cross_entropy_with_logits(p[cv,4],contact[cv])
        for k,x in zip(terms,(xy,angle,risk),strict=True): terms[k].append(x)
    return {k:torch.stack(x).mean() for k,x in terms.items()}


def loss_terms(model, batch, condition):
    inputs,targets = batch['inputs'],batch['targets']
    z,past = model.encode_history(inputs['observation_history'])
    blocks,mask = inputs['known_action_blocks'],inputs['known_action_valid']
    active,offsets = validate_timed_plan(blocks,mask,len(z))
    tokens = torch.cat([blocks.flatten(2),mask.to(z.dtype)],-1)
    plans,_ = model.direct_plan(tokens)
    seconds = offsets.to(z.dtype).unsqueeze(-1)/1e9
    direct = model.direct_decode(torch.cat([z[:,None].expand(-1,8,-1),plans,seconds],-1))
    direct = torch.where(active[:,:,None],direct,torch.zeros_like(direct))
    terms = {'direct_'+k:v for k,v in outcome_terms(direct,targets).items()}
    fv = targets['future_valid']; observed = {k:v[fv] for k,v in targets['future_observations'].items()}
    future_z = model.encoder(observed) if fv.any() else z[:0]
    variance,covariance = variance_covariance_penalty(torch.cat([z,past.flatten(0,1),future_z]))
    terms.update(variance_weighted=.1*variance,covariance_weighted=.01*covariance)
    if condition != 'direct':
        future = model.predict_latents(z,blocks,mask)
        rollout = model.decode_rollout(z,future,active,offsets)
        terms.update({'rollout_'+k:v for k,v in outcome_terms(rollout,targets).items()})
        if condition == 'jepa' and fv.any():
            squared = (future[fv]-model.target(observed)).square().mean(-1)
            counts = fv.sum(-1)
            groups = torch.repeat_interleave(torch.arange(len(z),device=z.device),counts)
            sums = torch.zeros(len(z),device=z.device,dtype=z.dtype).scatter_add(0,groups,squared)
            terms['latent_prediction'] = (sums[counts>0]/counts[counts>0]).mean()
    return terms


def diagnose(model, batch, condition):
    """Check against frozen loss/gradient; no optimizer/EMA or .grad mutation."""
    parameters = active_parameters(model,condition)
    names = {id(p):n for n,p in model.named_parameters()}
    terms = loss_terms(model,batch,condition)
    total = sum(terms.values())
    reference,_ = training_loss(model,batch,condition)
    torch.testing.assert_close(total,reference,rtol=2e-6,atol=2e-7)
    overall = torch.autograd.grad(total,parameters,retain_graph=True,allow_unused=True)
    expected = torch.autograd.grad(reference,parameters,allow_unused=True)
    error = 0.
    for p,a,b in zip(parameters,overall,expected,strict=True):
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-6)
        error = max(error,float((a-b).abs().max()))
    def norm(grads, prefix=None):
        return sum(float(g.detach().double().square().sum()) for p,g in zip(parameters,grads,strict=True)
            if g is not None and (prefix is None or names[id(p)].startswith(prefix)))**.5
    total_norm = norm(overall); report = {}
    for name,term in terms.items():
        grads = torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
        length = norm(grads)
        dot = sum(float((a.detach().double()*b.detach().double()).sum())
            for a,b in zip(grads,overall,strict=True) if a is not None and b is not None)
        report[name] = dict(weighted_loss=float(term.detach()),gradient_l2=length,
            encoder_gradient_l2=norm(grads,'encoder.'),
            cosine_to_total=None if length*total_norm==0 else dot/(length*total_norm))
    return dict(total_loss=float(total.detach()),total_gradient_l2=total_norm,
        frozen_loss=float(reference.detach()),max_gradient_decomposition_error=error,
        terms=report,optimizer_step=False,ema_update=False)
