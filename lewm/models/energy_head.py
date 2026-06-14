"""Learned scalar goal-energy head for latent-space planning.

Ported from TinyQuadJEPA-v2 (`tqjepa/models/energy_head.py`). LeWM's bare
``plan_cost`` is squared-L2 in projected space, which fails as a navigation cost
because the LeJEPA latent is a good place-*recognition* code but a poor *metric*
code (Phase-A: rho ~ 0.03). This head replaces that L2 with a small MLP trained
contrastively to rank the true goal below wrong goals — a learned "does my
predicted view match the goal view" score that does not need the latent to be
metric.

Both inputs live in the projected planning space (D = latent_dim): ``z_pred`` is a
rolled-out predicted latent (``pred_projector`` output via ``plan_rollout``) and
``z_goal`` is an encoded goal observation (``enc_projector`` output, i.e. cached
``z_proj``). The rollout objective trains those two projectors into the same
space, so comparing them is consistent with how the model was trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalEnergyHead(nn.Module):
    """Scores compatibility between a predicted latent and a goal latent.

    Input: concatenation of ``[z_pred, z_goal, z_pred - z_goal, z_pred * z_goal]``.
    Output: scalar energy (lower = more compatible / closer to goal).
    """

    def __init__(self, latent_dim: int = 192, hidden: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = latent_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_pred, z_goal, z_pred - z_goal, z_pred * z_goal], dim=-1)
        return self.net(x).squeeze(-1)


def sample_negative_goals(z_goal: torch.Tensor, num_negatives: int) -> torch.Tensor:
    """Vectorised in-batch negative sampling: shuffled goals as wrong targets.

    Returns ``[B, K, D]``. Each negative is another sample's goal from the same
    batch (almost always a different place); offset-shuffling guarantees the
    negative is never the sample's own goal.
    """
    bsz = z_goal.shape[0]
    if bsz <= 1:
        return z_goal.unsqueeze(1).expand(-1, num_negatives, -1)
    k = min(num_negatives, bsz - 1)
    base = torch.arange(bsz, device=z_goal.device).unsqueeze(1)
    offsets = torch.randint(1, bsz, (bsz, k), device=z_goal.device)
    neg_idx = (base + offsets) % bsz
    return z_goal[neg_idx]  # [B, K, D]


def energy_ranking_loss(
    head: GoalEnergyHead,
    z_pred: torch.Tensor,
    z_goal: torch.Tensor,
    z_neg: torch.Tensor,
    margin: float = 1.0,
    reg_weight: float = 1e-3,
):
    """Contrastive ranking: push E(pred, true_goal) below E(pred, wrong_goal)+margin.

    Returns ``(loss, stats)`` where stats includes ranking accuracy (fraction of
    negatives whose energy exceeds the positive's).
    """
    bsz, k_neg, dim = z_neg.shape
    pos_energy = head(z_pred, z_goal)                                  # [B]
    z_pred_rep = z_pred.unsqueeze(1).expand(-1, k_neg, -1).reshape(bsz * k_neg, dim)
    neg_energy = head(z_pred_rep, z_neg.reshape(bsz * k_neg, dim)).view(bsz, k_neg)
    rank_loss = F.softplus(pos_energy[:, None] - neg_energy + margin).mean()
    reg_loss = reg_weight * (pos_energy.square().mean() + neg_energy.square().mean())
    loss = rank_loss + reg_loss
    stats = {
        "loss": float(loss.detach()),
        "pos_energy": float(pos_energy.mean().detach()),
        "neg_energy": float(neg_energy.mean().detach()),
        "gap": float((neg_energy.mean() - pos_energy.mean()).detach()),
        "ranking_acc": float((pos_energy[:, None] < neg_energy).float().mean().detach()),
    }
    return loss, stats
