"""Inverse-dynamics auxiliary head — the action-sensitivity objective.

PLDM-style: from a consecutive latent pair ``(z_t, z_{t+1})`` predict the action
that carried the encoder between them. The loss backprops into the ENCODER, forcing
the latent to be action-sensitive and the transitions action-legible.

Why this and not more pose supervision (see
``docs/lewm_pose_aux_literature_and_options_2026-06-06.md``): the working
navigation-JEPAs (PLDM, DINO-WM) obtain plannability via inverse-dynamics +
action-conditioning, not pose labels. Unlike ``RelPoseHead`` this needs NO pose
labels (the active-block commands are already in the batch), and it shapes
*conditional* geometry (how actions move you through the latent), which coexists
with the SIGReg isotropic-Gaussian marginal instead of fighting an absolute metric
layout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseDynamicsHead(nn.Module):
    """Predicts the active-block action between two consecutive latents.

    Input featurisation mirrors ``RelPoseHead``: ``[z_a, z_b, z_a-z_b, z_a*z_b]``.
    Output is the ``cmd_dim`` active block applied between frame ``a`` and ``b``.
    """

    def __init__(self, latent_dim: int = 192, cmd_dim: int = 30, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.cmd_dim = cmd_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, cmd_dim),
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_a, z_b, z_a - z_b, z_a * z_b], dim=-1)
        return self.net(x)


def idm_loss(head: InverseDynamicsHead, z_lat: torch.Tensor, cmd_seq: torch.Tensor):
    """Inverse-dynamics loss over consecutive latent pairs.

    Args:
        head: the inverse-dynamics head.
        z_lat:   (B, T, D) graph-connected latents (``z_proj`` by default).
        cmd_seq: (B, T, A) active blocks; ``cmd_seq[:, t]`` is the action applied
            between frame ``t`` and ``t+1`` (matches ``integrate_world_poses``).

    Predicts ``cmd_seq[:, t]`` from ``(z_lat[:, t], z_lat[:, t+1])`` for
    ``t = 0..T-2``. Returns ``(loss, stats)``; ``idm_action_r2`` is the pooled
    coefficient of determination (1 = perfectly decodable, 0 = no better than the
    action mean) so we can read whether the latent actually encodes the action.
    """
    b, t, d = z_lat.shape
    if t < 2:
        zero = z_lat.new_zeros(())
        return zero, {"idm_action_err": 0.0, "idm_action_r2": 0.0}
    za = z_lat[:, :-1].reshape(-1, d)
    zb = z_lat[:, 1:].reshape(-1, d)
    tgt = cmd_seq[:, :-1].reshape(-1, cmd_seq.shape[-1])
    pred = head(za, zb)
    loss = F.smooth_l1_loss(pred, tgt)
    with torch.no_grad():
        err = (pred - tgt).float()
        tgt_f = tgt.float()
        sse = (err ** 2).sum()
        sst = ((tgt_f - tgt_f.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-8)
        stats = {
            "idm_action_err": float(err.abs().mean()),
            "idm_action_r2": float(1.0 - sse / sst),
        }
        # Per-channel-group (vx, vy, yaw_rate) error if the block is channel-major 3*K,
        # so we can catch the head ignoring a low-magnitude action component.
        a = tgt.shape[-1]
        if a % 3 == 0:
            k = a // 3
            tg = tgt_f.reshape(-1, 3, k)
            pg = pred.float().reshape(-1, 3, k)
            for i, name in enumerate(("vx", "vy", "wz")):
                stats[f"idm_err_{name}"] = float((pg[:, i] - tg[:, i]).abs().mean())
    return loss, stats
