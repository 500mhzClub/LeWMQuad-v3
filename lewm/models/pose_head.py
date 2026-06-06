"""Relative-pose auxiliary head — the metric objective for world-model navigation.

The LeJEPA latent is recognition-strong but metric-poor: it keeps heading
(yaw R^2 ~0.81) but discards position (distance decodability ~0.05). A planning
cost needs a smooth distance-to-goal gradient, which the bare latent can't supply.

This head ``P(z_a, z_b) -> (dx, dy, dyaw)`` predicts the pose of frame ``b`` in
frame ``a``'s body frame. Trained with smooth-L1 against true geometry, its loss
backprops into the ENCODER, forcing the latent to retain metric structure. At
planning time the navigation cost becomes ``||predicted dxy(z_state, z_goal)||`` —
the world model's own distance-to-goal, with no privileged runtime info.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPoseHead(nn.Module):
    """Predicts SE(2) relative pose (dx, dy, dyaw) between two latents.

    Input featurisation mirrors GoalEnergyHead: ``[z_a, z_b, z_a-z_b, z_a*z_b]``.
    Output is in frame ``a``'s body frame: dx forward, dy left, dyaw heading change.
    """

    def __init__(self, latent_dim: int = 192, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_a, z_b, z_a - z_b, z_a * z_b], dim=-1)
        return self.net(x)


def integrate_world_poses(cmd_seq: torch.Tensor, dt: float) -> torch.Tensor:
    """Kinematic world pose (x, y, yaw) at each frame from active-block commands.

    cmd_seq: (B, T, A) active blocks (A = ticks*3, rows [vx, vy, yaw_rate]).
    Returns (B, T, 3); frame 0 at origin/zero-yaw, frame t after executing steps
    0..t-1. Matches the kinematic model the nav benchmark integrates, so the head's
    target is exactly the geometry it will be evaluated under. (Label only — the
    caller wraps this in no_grad.)
    """
    b, t, a = cmd_seq.shape
    # Active block is channel-major ([vx*K, vy*K, yaw_rate*K]); decode to per-tick
    # rows [vx, vy, yaw_rate] to match lewm.actions.active_block_to_matrix.
    ticks = cmd_seq.reshape(b, t, 3, a // 3).transpose(-1, -2)  # (B, T, K, 3)
    x = cmd_seq.new_zeros(b)
    y = cmd_seq.new_zeros(b)
    yaw = cmd_seq.new_zeros(b)
    poses = [torch.stack([x, y, yaw], dim=-1)]
    for step in range(t - 1):
        for k in range(ticks.shape[2]):
            vx, vy, wz = ticks[:, step, k, 0], ticks[:, step, k, 1], ticks[:, step, k, 2]
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            x = x + (vx * cy - vy * sy) * dt
            y = y + (vx * sy + vy * cy) * dt
            yaw = yaw + wz * dt
        poses.append(torch.stack([x, y, yaw], dim=-1))
    return torch.stack(poses, dim=1)  # (B, T, 3)


def body_relative(poses: torch.Tensor, a_idx: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    """Pose of frame b in frame a's body frame. poses (B,T,3) -> (B,P,3)."""
    pa = poses[:, a_idx]  # (B, P, 3)
    pb = poses[:, b_idx]
    dx, dy = pb[..., 0] - pa[..., 0], pb[..., 1] - pa[..., 1]
    ca, sa = torch.cos(pa[..., 2]), torch.sin(pa[..., 2])
    bx = ca * dx + sa * dy   # R(-yaw_a) * world-delta
    by = -sa * dx + ca * dy
    dyaw = torch.atan2(torch.sin(pb[..., 2] - pa[..., 2]), torch.cos(pb[..., 2] - pa[..., 2]))
    return torch.stack([bx, by, dyaw], dim=-1)


def ordered_pair_indices(t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """All non-identical ordered frame pairs.

    Bidirectional pairs prevent a relative-pose head from learning the corpus's
    dominant forward-time motion direction as a shortcut.
    """
    idx = torch.arange(t, device=device)
    a_idx = idx[:, None].expand(t, t).reshape(-1)
    b_idx = idx[None, :].expand(t, t).reshape(-1)
    keep = a_idx != b_idx
    return a_idx[keep], b_idx[keep]


def _horizon_balanced_mean(per_pair: torch.Tensor, a_idx: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    """Average pair loss with equal total weight per absolute frame gap."""
    gaps = (b_idx - a_idx).abs()
    counts = torch.bincount(gaps, minlength=int(gaps.max().item()) + 1).clamp(min=1)
    weights = counts[gaps].to(per_pair.dtype).reciprocal()
    weights = weights / weights.mean()
    return (per_pair * weights[None, :]).mean()


def pose_aux_loss(
    head: RelPoseHead,
    z_lat: torch.Tensor,
    cmd_seq: torch.Tensor,
    dt: float,
    yaw_weight: float = 0.5,
    poses: torch.Tensor | None = None,
):
    """Relative-pose loss over bidirectional, horizon-balanced encoded pairs.

    ``poses`` should contain aligned physical ``(x, y, yaw)`` labels when
    available. Command integration remains an explicit ablation fallback.
    """
    b, t, d = z_lat.shape
    if poses is None:
        with torch.no_grad():
            poses = integrate_world_poses(cmd_seq, dt)  # (B, T, 3)
    a_idx, b_idx = ordered_pair_indices(t, z_lat.device)
    tgt = body_relative(poses, a_idx, b_idx)                       # (B, P, 3)
    za = z_lat[:, a_idx].reshape(-1, d)
    zb = z_lat[:, b_idx].reshape(-1, d)
    pred = head(za, zb).reshape(b, -1, 3)                          # (B, P, 3)
    xy_per_pair = F.smooth_l1_loss(pred[..., :2], tgt[..., :2], reduction="none").mean(dim=-1)
    yaw_delta = torch.atan2(
        torch.sin(pred[..., 2] - tgt[..., 2]),
        torch.cos(pred[..., 2] - tgt[..., 2]),
    )
    yaw_per_pair = F.smooth_l1_loss(yaw_delta, torch.zeros_like(yaw_delta), reduction="none")
    xy_loss = _horizon_balanced_mean(xy_per_pair, a_idx, b_idx)
    yaw_loss = _horizon_balanced_mean(yaw_per_pair, a_idx, b_idx)
    loss = xy_loss + yaw_weight * yaw_loss
    with torch.no_grad():
        stats = {
            "pose_xy_err_m": float((pred[..., :2] - tgt[..., :2]).norm(dim=-1).mean()),
            "pose_yaw_err_rad": float(yaw_delta.abs().mean()),
            "pose_dist_span_m": float(tgt[..., :2].norm(dim=-1).max()),
        }
    return loss, stats


def predicted_pose_aux_loss(
    head: RelPoseHead,
    model: nn.Module,
    z_raw: torch.Tensor,
    z_proj: torch.Tensor,
    cmd_seq: torch.Tensor,
    dt: float,
    yaw_weight: float = 0.5,
    poses: torch.Tensor | None = None,
):
    """Deployment-aligned pose loss: predicted endpoint -> encoded final goal.

    The planner applies ``RelPoseHead`` to predictor-generated candidate endpoints
    and an encoded goal image. This term trains that exact input contract instead
    of evaluating only encoded-to-encoded pairs.
    """
    b, t, _ = z_raw.shape
    if t < 3:
        zero = z_raw.new_zeros(())
        return zero, {
            "pose_pred_xy_err_m": 0.0,
            "pose_pred_yaw_err_rad": 0.0,
            "pose_pred_dist_span_m": 0.0,
        }
    if poses is None:
        with torch.no_grad():
            poses = integrate_world_poses(cmd_seq, dt)

    endpoint_indices = torch.arange(1, t - 1, device=z_raw.device)
    predicted = []
    for endpoint in endpoint_indices.tolist():
        predicted.append(model.plan_rollout(z_raw[:, 0], cmd_seq[:, :endpoint])[:, -1])
    z_pred = torch.stack(predicted, dim=1)  # (B, T-2, D), predictor projection space
    z_goal = z_proj[:, -1:, :].expand(-1, z_pred.shape[1], -1)
    pred = head(z_pred.reshape(-1, z_pred.shape[-1]), z_goal.reshape(-1, z_goal.shape[-1]))
    pred = pred.reshape(b, -1, 3)

    goal_idx = torch.full_like(endpoint_indices, t - 1)
    tgt = body_relative(poses, endpoint_indices, goal_idx)
    xy_loss = F.smooth_l1_loss(pred[..., :2], tgt[..., :2])
    yaw_delta = torch.atan2(
        torch.sin(pred[..., 2] - tgt[..., 2]),
        torch.cos(pred[..., 2] - tgt[..., 2]),
    )
    yaw_loss = F.smooth_l1_loss(yaw_delta, torch.zeros_like(yaw_delta))
    loss = xy_loss + yaw_weight * yaw_loss
    with torch.no_grad():
        stats = {
            "pose_pred_xy_err_m": float((pred[..., :2] - tgt[..., :2]).norm(dim=-1).mean()),
            "pose_pred_yaw_err_rad": float(yaw_delta.abs().mean()),
            "pose_pred_dist_span_m": float(tgt[..., :2].norm(dim=-1).max()),
        }
    return loss, stats
