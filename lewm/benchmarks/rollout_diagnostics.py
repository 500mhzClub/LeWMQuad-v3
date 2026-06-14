"""Shared collapse-aware diagnostics for learned latent rollouts."""
from __future__ import annotations

import torch


def _mse_by_step(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim < 3:
        raise ValueError(
            "rollout tensors must have equal shape (B, H, ...), got "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )
    return (left - right).square().mean(dim=tuple(range(2, left.ndim)))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("inf")


def _normalized_change(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else float("inf")


def summarize_rollout_controls(
    *,
    rollout: torch.Tensor,
    targets: torch.Tensor,
    persistence: torch.Tensor,
    zero_action: torch.Tensor,
    shuffled_action: torch.Tensor,
    previous_targets: torch.Tensor,
) -> dict:
    """Report prediction, action-sensitivity, and representation-collapse controls."""

    metrics = {
        "rollout": _mse_by_step(rollout, targets),
        "persistence": _mse_by_step(persistence, targets),
        "zero_action": _mse_by_step(zero_action, targets),
        "shuffled_action": _mse_by_step(shuffled_action, targets),
        "target_step_delta": _mse_by_step(previous_targets, targets),
    }
    per_horizon = []
    for step in range(targets.shape[1]):
        values = {
            name: float(metric[:, step].mean().detach().cpu())
            for name, metric in metrics.items()
        }
        zero_advantage = values["zero_action"] - values["rollout"]
        shuffled_advantage = values["shuffled_action"] - values["rollout"]
        zero_normalized = _normalized_change(
            zero_advantage,
            values["target_step_delta"],
        )
        shuffled_normalized = _normalized_change(
            shuffled_advantage,
            values["target_step_delta"],
        )
        per_horizon.append(
            {
                "step": step + 1,
                **{f"{name}_token_mse": value for name, value in values.items()},
                "free_running_vs_persistence_mse_ratio": _ratio(
                    values["rollout"],
                    values["persistence"],
                ),
                "zero_minus_free_running_mse": zero_advantage,
                "shuffled_minus_free_running_mse": shuffled_advantage,
                "zero_action_advantage_over_target_change": zero_normalized,
                "shuffled_action_advantage_over_target_change": shuffled_normalized,
                "free_running_beats_persistence": (
                    values["rollout"] < values["persistence"]
                ),
                "real_action_beats_zero": values["rollout"] < values["zero_action"],
                "real_action_beats_shuffled": (
                    values["rollout"] < values["shuffled_action"]
                ),
                "meaningful_real_action_beats_zero": zero_normalized >= 0.1,
                "meaningful_real_action_beats_shuffled": shuffled_normalized >= 0.1,
            }
        )

    feature_dim = targets.shape[-1]
    flattened = targets.float().reshape(-1, feature_dim)
    mean_feature_std = float(
        flattened.std(dim=0, unbiased=False).mean().detach().cpu()
    )
    mean_target_change = float(metrics["persistence"].mean().detach().cpu())
    return {
        "per_horizon_step": per_horizon,
        "representation": {
            "mean_feature_std": mean_feature_std,
            "mean_target_change_mse": mean_target_change,
            "target_change_over_feature_variance": _normalized_change(
                mean_target_change,
                mean_feature_std**2,
            ),
            "collapse_warning": mean_feature_std < 0.05,
            "near_static_target_warning": (
                _normalized_change(mean_target_change, mean_feature_std**2) < 0.01
            ),
        },
    }
