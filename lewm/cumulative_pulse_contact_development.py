"""Cumulative event semantics on actual pulse boundaries, not calibrated risk.

Each raw head scalar parameterizes a nonnegative per-second conditional hazard
over its known interval. Integrating hazards makes event probability monotone.
The20cm sensor limit and native contact labels are unrelated to this transform.
"""
import torch
from torch.nn import functional as F
from lewm.pulse_timed_rgb_body_jepa_development import PulseTimedRGBBodyJEPA


def cumulative_contact_logits(raw,offsets_ns,active):
    if (raw.ndim!=2 or raw.shape[1]!=8 or raw.dtype not in (torch.float32,torch.float64)
            or offsets_ns.shape!=raw.shape or offsets_ns.dtype!=torch.int64
            or active.shape!=raw.shape or active.dtype!=torch.bool
            or raw.device!=active.device or raw.device!=offsets_ns.device or not torch.isfinite(raw).all()):
        raise ValueError('finite floating eight-slot hazards and co-device int64 clocks/boolean masks required')
    if (len(raw)==0 or not active[:,0].all() or (active[:,1:]&~active[:,:-1]).any()
            or (offsets_ns[~active]!=0).any()):
        raise ValueError('nonempty known prefix with zero unknown timestamps required')
    delta=torch.diff(torch.cat([torch.zeros_like(offsets_ns[:,:1]),offsets_ns],-1),dim=-1)
    if (delta[active]<=0).any():raise ValueError('strictly increasing actual known event horizons required')
    seconds=torch.where(active,delta,torch.ones_like(delta)).to(raw.dtype)/1e9
    # Stable log(softplus(x)); below-20 asymptote has <1.1e-9 absolute
    # log-rate error. Clamp the unused branch to avoid NaN/Inf gradients.
    log_rate=torch.where(raw < -20.,raw,F.softplus(raw.clamp_min(-20.)).log())
    log_increments=torch.where(active,log_rate+seconds.log(),torch.full_like(raw,-torch.inf))
    log_hazard=torch.logcumsumexp(log_increments,dim=-1)
    # logit(1-exp(-H)) = H + log(-expm1(-H)). The small-H asymptote
    # log(H) keeps finite logits/gradients even for raw scores near-1000.
    safe_hazard=log_hazard.clamp_min(-20.).exp()
    logits=torch.where(log_hazard < -20.,log_hazard,safe_hazard+(-torch.expm1(-safe_hazard)).log())
    if not torch.isfinite(logits[active]).all():raise ValueError('cumulative hazard overflow; no silent clipping')
    return torch.where(active,logits,torch.zeros_like(logits))


def cumulative_outcomes(raw,active,offsets_ns):
    if raw.shape!=(*active.shape,5):raise ValueError('four motion components and one event-rate component required')
    contact=cumulative_contact_logits(raw[...,4],offsets_ns,active)
    result=torch.cat([raw[...,:4],contact[...,None]],-1)
    return torch.where(active[...,None],result,torch.zeros_like(result))


class CumulativePulseRGBBodyJEPA(PulseTimedRGBBodyJEPA):
    """Same parameters/latent dynamics; different declared contact interpretation.

    Old outcome checkpoints are not scientifically interchangeable with this
    variant. Direct and recursive heads both use identical cumulative semantics.
    Targets are not accepted by inference. This does not learn a control policy.
    """
    def decode_rollout(self,z,future,active,offsets_ns):
        raw=super().decode_rollout(z,future,active,offsets_ns)
        return cumulative_outcomes(raw,active,offsets_ns)

    def forward(self,observation_history,known_action_blocks,known_action_valid):
        output=super().forward(observation_history,known_action_blocks,known_action_valid)
        # Parent forward calls the overridden rollout decoder exactly once.
        output['direct_outcomes']=cumulative_outcomes(output['direct_outcomes'],output['prediction_valid'],output['target_offsets_ns'])
        return output
