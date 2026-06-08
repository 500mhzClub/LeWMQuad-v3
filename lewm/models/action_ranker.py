"""Goal-conditioned first-action ranker for frozen LeWM features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalActionRanker(nn.Module):
    """Predict a lower-is-better cost for one primitive from start/goal latents."""

    def __init__(
        self,
        latent_dim: int = 192,
        cmd_dim: int = 15,
        hidden: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cmd_dim = cmd_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 4 + cmd_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (z_start, z_goal, z_start - z_goal, z_start * z_goal, action),
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class TaskAlignedCandidateScorer(nn.Module):
    """Predict separate task outcomes for one candidate action."""

    def __init__(
        self,
        latent_dim: int = 192,
        cmd_dim: int = 15,
        hidden: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cmd_dim = cmd_dim
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim * 4 + cmd_dim + 1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.collision_head = nn.Linear(hidden // 2, 1)
        self.progress_head = nn.Linear(hidden // 2, 1)
        self.heading_head = nn.Linear(hidden // 2, 1)
        self.clearance_head = nn.Linear(hidden // 2, 1)

    def forward(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        goal_present: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if goal_present.shape != z_start.shape[:-1]:
            raise ValueError(
                "goal_present must match the leading start-latent dimensions, "
                f"got {tuple(goal_present.shape)} and {tuple(z_start.shape)}"
            )
        features = torch.cat(
            (
                z_start,
                z_goal,
                z_start - z_goal,
                z_start * z_goal,
                goal_present.unsqueeze(-1).to(z_start.dtype),
                action,
            ),
            dim=-1,
        )
        hidden = self.trunk(features)
        return {
            "collision_logit": self.collision_head(hidden).squeeze(-1),
            "progress": self.progress_head(hidden).squeeze(-1),
            "heading": self.heading_head(hidden).squeeze(-1),
            "clearance": self.clearance_head(hidden).squeeze(-1),
        }


def task_aligned_candidate_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    goal_present: torch.Tensor,
    collision_pos_weight: torch.Tensor | None = None,
    collision_weight: float = 1.0,
    progress_weight: float = 1.0,
    heading_weight: float = 0.5,
    clearance_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-task loss with goal-dependent progress and heading regression."""
    collision = F.binary_cross_entropy_with_logits(
        predictions["collision_logit"],
        targets["collision"].to(predictions["collision_logit"].dtype),
        pos_weight=collision_pos_weight,
    )
    goal_mask = goal_present.unsqueeze(-1).expand_as(targets["progress"]).bool()
    if goal_mask.any():
        progress = F.smooth_l1_loss(
            predictions["progress"][goal_mask],
            targets["progress"][goal_mask],
        )
        heading = F.smooth_l1_loss(
            predictions["heading"][goal_mask],
            targets["heading"][goal_mask],
        )
    else:
        progress = predictions["progress"].new_zeros(())
        heading = predictions["heading"].new_zeros(())
    clearance = F.smooth_l1_loss(
        predictions["clearance"],
        targets["clearance"],
    )
    total = (
        collision_weight * collision
        + progress_weight * progress
        + heading_weight * heading
        + clearance_weight * clearance
    )
    return total, {
        "collision": collision.detach(),
        "progress": progress.detach(),
        "heading": heading.detach(),
        "clearance": clearance.detach(),
    }


def action_ranker_loss(
    scores: torch.Tensor,
    target_cost: torch.Tensor,
    *,
    temperature: float = 0.1,
    regression_weight: float = 0.25,
) -> torch.Tensor:
    """Listwise ranking loss with a small absolute-cost regression anchor."""
    target_probability = torch.softmax(-target_cost / temperature, dim=-1)
    log_probability = torch.log_softmax(-scores / temperature, dim=-1)
    listwise = -(target_probability * log_probability).sum(dim=-1).mean()
    regression = F.smooth_l1_loss(scores, target_cost)
    return listwise + regression_weight * regression


@torch.no_grad()
def first_action_metrics(
    scores: torch.Tensor,
    true_distance: torch.Tensor,
    collision: torch.Tensor | None = None,
) -> dict[str, float]:
    """Report the deployed argmin decision against oracle and random controls."""
    selected_index = scores.argmin(dim=-1, keepdim=True)
    selected_distance = true_distance.gather(1, selected_index).squeeze(1)
    oracle_distance = true_distance.min(dim=-1).values
    random_distance = true_distance.mean(dim=-1)
    regret = selected_distance - oracle_distance
    random_regret = random_distance - oracle_distance

    score_rank = torch.argsort(torch.argsort(scores, dim=-1), dim=-1).float()
    distance_rank = torch.argsort(torch.argsort(true_distance, dim=-1), dim=-1).float()
    score_rank -= score_rank.mean(dim=-1, keepdim=True)
    distance_rank -= distance_rank.mean(dim=-1, keepdim=True)
    denominator = torch.sqrt(
        score_rank.square().sum(dim=-1) * distance_rank.square().sum(dim=-1)
    ).clamp_min(1e-8)
    spearman = (score_rank * distance_rank).sum(dim=-1) / denominator

    mean_random_regret = random_regret.mean()
    metrics = {
        "mean_first_regret_m": float(regret.mean()),
        "mean_random_regret_m": float(mean_random_regret),
        "regret_ratio_vs_random": float(regret.mean() / mean_random_regret.clamp_min(1e-8)),
        "mean_first_spearman": float(spearman.mean()),
        "mean_selected_distance_m": float(selected_distance.mean()),
        "mean_oracle_distance_m": float(oracle_distance.mean()),
    }
    if collision is not None:
        metrics["selected_collision_rate"] = float(
            collision.gather(1, selected_index).float().mean()
        )
    return metrics
