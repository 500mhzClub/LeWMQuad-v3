"""Explicit 500-ms dense-world-model endpoint for a navigation consumer.

No intermediate path or contact prediction is inferred from the endpoint.
Commands are the applied tape, after the platform limiter, in native units.
"""
import torch
from torch import nn
from torch.nn import functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens


class DenseEndpointMotion(nn.Module):
    def __init__(self, predictor, readout, control_mean, control_std, *, action_blind=False):
        super().__init__()
        self.predictor=predictor.eval().requires_grad_(False)
        self.readout=readout.eval().requires_grad_(False)
        self.register_buffer('control_mean',torch.as_tensor(control_mean,dtype=torch.float32))
        self.register_buffer('control_std',torch.as_tensor(control_std,dtype=torch.float32))
        assert self.control_mean.shape==self.control_std.shape==(2,)
        assert (self.control_std>0).all()
        self.action_blind=action_blind
        self.eval()

    @torch.inference_mode()
    def forward(self, *, context_tokens, context_times_ns, past_applied_commands,
                past_measured_ns, past_available_ns, future_applied_commands):
        if context_tokens.shape!=(3,768,1024):
            raise ValueError('three native dense token frames required')
        times=torch.as_tensor(context_times_ns,dtype=torch.int64)
        measured=torch.as_tensor(past_measured_ns,dtype=torch.int64)
        available=torch.as_tensor(past_available_ns,dtype=torch.int64)
        if (times.shape!=(3,) or not torch.equal(times.diff(),torch.tensor([500_000_000]*2,device=times.device))
                or measured.shape!=(15,) or available.shape!=(15,)
                or not torch.equal(measured[[4,9,14]],times)
                or not bool((measured<=times[-1]).all() and (available<=times[-1]).all())
                or not torch.equal(measured.diff(),torch.full((14,),100_000_000,dtype=torch.int64,device=measured.device))):
            raise ValueError('exact one-second visual context and causal 100-ms command history required')
        past=torch.as_tensor(past_applied_commands,dtype=torch.float32,device=context_tokens.device)
        future=torch.as_tensor(future_applied_commands,dtype=torch.float32,device=context_tokens.device)
        if past.shape!=(15,3) or future.ndim!=3 or future.shape[1:]!=(5,3) or len(future)<1:
            raise ValueError('15 past and five future applied XYZ command rows required')
        if not bool(torch.isfinite(past).all() and torch.isfinite(future).all() and torch.isfinite(context_tokens).all()):
            raise ValueError('finite commands and features required')
        if bool((past[:,1]!=0).any() or (future[:,:,1]!=0).any()):
            raise ValueError('predictor was fitted without lateral command input')
        control=(past[:,[0,2]].reshape(3,5,2)-self.control_mean)/self.control_std
        action=future[:,:,[0,2]].reshape(len(future),10)
        n=1 if self.action_blind else len(future)
        x=context_tokens[None].expand(n,-1,-1,-1)
        c=control[None].expand(n,-1,-1,-1)
        a=torch.zeros_like(action[:1]) if self.action_blind else action
        mask=torch.ones(n,768,dtype=torch.bool,device=x.device)
        predicted=F.layer_norm(self.predictor(x,a,mask,control=c).float(),(1024,))
        current=pool_tokens(context_tokens[-1:]).expand(n,-1,-1)
        motion=self.readout(current,pool_tokens(predicted))
        if not bool(torch.isfinite(motion).all()):raise ValueError('finite endpoint motion required')
        # Broadcast after decoding, preserving exact action blindness despite
        # batched GEMM roundoff when identical feature rows are repeated.
        motion=motion.expand(len(future),-1);predicted=predicted.expand(len(future),-1,-1)
        return dict(endpoint_motion_body_xy_yaw=motion,predicted_tokens=predicted,
                    target_offset_ns=500_000_000,intermediate_motion_available=False,
                    contact_prediction_available=False,action_blind=self.action_blind)


def endpoint_waypoint_scores(motion, goal_body_xy, *, scan_error=None):
    """Candidate utility only; feasibility must come from shared observed geometry."""
    if motion.ndim!=2 or motion.shape[1]!=3 or not torch.isfinite(motion).all():
        raise ValueError('finite candidate XY/yaw endpoints required')
    goal=torch.as_tensor(goal_body_xy,dtype=motion.dtype,device=motion.device)
    if goal.shape!=(2,) or not torch.isfinite(goal).all():raise ValueError('finite observed-frame waypoint required')
    if scan_error is None:
        return torch.linalg.vector_norm(goal)-torch.linalg.vector_norm(goal[None]-motion[:,:2],dim=-1)
    angle=torch.as_tensor(scan_error,dtype=motion.dtype,device=motion.device)
    if angle.ndim or not torch.isfinite(angle):raise ValueError('finite scalar scan error required')
    remaining=torch.atan2(torch.sin(angle-motion[:,2]),torch.cos(angle-motion[:,2]))
    return .35*(angle.abs()-remaining.abs())
