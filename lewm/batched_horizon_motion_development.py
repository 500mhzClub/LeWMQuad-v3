"""Pack independent horizon forecasts into one predictor/readout batch."""
import torch
import torch.nn.functional as F
from lewm.dense_visual_motion_readout_development import pool_tokens


@torch.inference_mode()
def batched_horizon_motion(predictor, readout, context, control, actions, *, action_blind=False):
    if context.shape!=(1,3,768,1024) or control.shape!=(1,3,5,2) or actions.shape!=(6,8,2):
        raise ValueError('one native context and six prospective applied action tapes required')
    device = context.device
    tapes, times, expansions, counts = [], [], [], []
    offset = 0
    for horizon in range(1,9):
        _, inverse = torch.unique(actions[:,:horizon].reshape(6,-1), dim=0, return_inverse=True)
        if action_blind:
            chosen = actions[:1]
            expansion = torch.zeros(6,dtype=torch.long,device=device)
        else:
            indices = torch.stack([(inverse==j).nonzero()[0,0] for j in range(int(inverse.max())+1)])
            chosen, expansion = actions[indices], inverse
        count = len(chosen)
        tapes.append(chosen)
        times.append(torch.full((count,),horizon,dtype=torch.long,device=device))
        expansions.append(expansion+offset)
        counts.append(count)
        offset += count
    predicted = predictor(context.expand(offset,-1,-1,-1),torch.cat(tapes),torch.cat(times),
        torch.ones(offset,768,dtype=torch.bool,device=device),control=control.expand(offset,-1,-1,-1))
    predicted = F.layer_norm(predicted.float(),(1024,))
    decoded = readout(pool_tokens(context[:,-1]).expand(offset,-1,-1),pool_tokens(predicted))
    motion = torch.stack([decoded[index] for index in expansions],dim=1)
    if not torch.isfinite(motion).all():
        raise ValueError('finite batched physical forecasts required')
    return motion, counts
