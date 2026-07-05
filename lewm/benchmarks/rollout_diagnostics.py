"""Shared collapse-aware diagnostics for learned latent rollouts."""
from __future__ import annotations

import torch

DEFAULT_MAX_PAIRWISE_STATES = 1024


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


def effective_rank(features: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Return covariance effective rank for flattened feature observations."""

    if features.ndim != 2:
        raise ValueError("features must have shape (samples, dimensions)")
    if features.shape[0] < 2:
        return 0.0
    if not torch.isfinite(features).all():
        return 0.0
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / features.shape[0]
    try:
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0.0)
    except torch.linalg.LinAlgError:
        return 0.0
    if not torch.isfinite(eigenvalues).all():
        return 0.0
    total = eigenvalues.sum()
    if float(total) <= eps:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp(min=eps).log()).sum()
    return float(entropy.exp().detach().cpu())


def _deterministic_sample_rows(
    rows: torch.Tensor,
    *,
    max_rows: int,
) -> torch.Tensor:
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if rows.shape[0] <= max_rows:
        return rows
    indices = (
        torch.arange(max_rows, device=rows.device, dtype=torch.long)
        * rows.shape[0]
        // max_rows
    )
    return rows.index_select(0, indices)


def _mean_off_diagonal_pairwise_mse(states: torch.Tensor) -> float:
    if states.ndim != 2:
        raise ValueError("states must have shape (samples, dimensions)")
    if states.shape[0] < 2:
        return 0.0
    states = states.float()
    squared_norms = states.square().sum(dim=1)
    squared_distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (states @ states.T)
    ).clamp_min_(0.0)
    mse = squared_distances / states.shape[1]
    off_diagonal = ~torch.eye(states.shape[0], dtype=torch.bool, device=states.device)
    return float(mse[off_diagonal].mean().cpu())


def summarize_spatial_stability(
    *,
    pre_normalized_targets: torch.Tensor,
    normalized_targets: torch.Tensor,
    previous_normalized_targets: torch.Tensor | None = None,
    max_pairwise_states: int = DEFAULT_MAX_PAIRWISE_STATES,
) -> dict:
    """Report the registered Phase 2D scale, rank, and discrimination diagnostics."""

    if pre_normalized_targets.shape != normalized_targets.shape:
        raise ValueError("pre-normalized and normalized targets must align")
    if normalized_targets.ndim != 4:
        raise ValueError("spatial targets must have shape (B, T, N, D)")
    feature_dim = normalized_targets.shape[-1]
    nonfinite_feature_warning = not (
        torch.isfinite(pre_normalized_targets).all()
        and torch.isfinite(normalized_targets).all()
    )
    normalized_float = torch.nan_to_num(
        normalized_targets.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    flattened = normalized_float.reshape(-1, feature_dim)
    pre_normalized_float = torch.nan_to_num(
        pre_normalized_targets.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    pre_norms = pre_normalized_float.norm(dim=-1).reshape(-1)
    mean_feature_std = float(flattened.std(dim=0, unbiased=False).mean().cpu())
    rank = effective_rank(flattened)
    result = {
        "nonfinite_feature_warning": bool(nonfinite_feature_warning),
        "mean_feature_std": mean_feature_std,
        "effective_rank": rank,
        "effective_rank_fraction": rank / feature_dim,
        "pre_normalized_token_norm_mean": float(pre_norms.mean().cpu()),
        "pre_normalized_token_norm_p05": float(torch.quantile(pre_norms, 0.05).cpu()),
        "pre_normalized_token_norm_median": float(pre_norms.median().cpu()),
        "pre_normalized_token_norm_p95": float(torch.quantile(pre_norms, 0.95).cpu()),
        "normalized_token_norm_mean": float(
            normalized_float.norm(dim=-1).mean().cpu()
        ),
        "collapse_warning": bool(nonfinite_feature_warning) or mean_feature_std < 0.05,
        "effective_rank_warning": bool(nonfinite_feature_warning)
        or rank < 0.10 * feature_dim,
    }
    states = normalized_float.reshape(
        normalized_float.shape[0] * normalized_float.shape[1],
        -1,
    )
    pairwise_states = _deterministic_sample_rows(
        states,
        max_rows=max_pairwise_states,
    )
    result["pairwise_state_population"] = int(states.shape[0])
    result["pairwise_state_sample_size"] = int(pairwise_states.shape[0])
    result["pairwise_state_sampling"] = (
        "full_population"
        if pairwise_states.shape[0] == states.shape[0]
        else "deterministic_stride"
    )
    if states.shape[0] > 1:
        result["mean_pairwise_state_mse"] = _mean_off_diagonal_pairwise_mse(
            pairwise_states
        )
    else:
        result["mean_pairwise_state_mse"] = 0.0
    if previous_normalized_targets is not None:
        if previous_normalized_targets.shape != normalized_targets.shape:
            raise ValueError("previous_normalized_targets must align with targets")
        target_change = (
            torch.nan_to_num(
                previous_normalized_targets.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            - normalized_float
        ).square().mean()
        result["mean_target_change_mse"] = float(target_change.cpu())
        result["target_change_over_feature_variance"] = _normalized_change(
            float(target_change.cpu()),
            mean_feature_std**2,
        )
        result["near_static_target_warning"] = bool(nonfinite_feature_warning) or (
            result["target_change_over_feature_variance"] < 0.01
        )
    return result


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
