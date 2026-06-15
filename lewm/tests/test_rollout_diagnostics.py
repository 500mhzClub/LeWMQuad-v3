from __future__ import annotations

import pytest
import torch

from lewm.benchmarks.rollout_diagnostics import (
    effective_rank,
    summarize_rollout_controls,
    summarize_spatial_stability,
)


def test_rollout_diagnostics_reports_action_controls_and_collapse() -> None:
    targets = torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]]]])
    persistence = torch.zeros_like(targets)
    rollout = targets.clone()
    zero = persistence.clone()
    shuffled = persistence.clone()
    previous = torch.cat([persistence[:, :1], targets[:, :1]], dim=1)

    report = summarize_rollout_controls(
        rollout=rollout,
        targets=targets,
        persistence=persistence,
        zero_action=zero,
        shuffled_action=shuffled,
        previous_targets=previous,
    )

    assert report["per_horizon_step"][0]["free_running_beats_persistence"]
    assert report["per_horizon_step"][0]["real_action_beats_zero"]
    assert report["per_horizon_step"][0]["meaningful_real_action_beats_zero"]
    assert (
        report["per_horizon_step"][0]["zero_action_advantage_over_target_change"]
        == 1.0
    )
    assert report["per_horizon_step"][1]["target_step_delta_token_mse"] == 0.5
    assert not report["representation"]["collapse_warning"]


def test_rollout_diagnostics_warns_on_collapsed_targets() -> None:
    targets = torch.zeros(2, 2, 3, 4)

    report = summarize_rollout_controls(
        rollout=targets,
        targets=targets,
        persistence=targets,
        zero_action=targets,
        shuffled_action=targets,
        previous_targets=targets,
    )

    assert report["representation"]["collapse_warning"]
    assert report["representation"]["near_static_target_warning"]


def test_effective_rank_distinguishes_collapsed_and_full_rank_features() -> None:
    collapsed = torch.ones(16, 4)
    full_rank = torch.cat([torch.eye(4), -torch.eye(4)], dim=0)

    assert effective_rank(collapsed) == 0.0
    assert effective_rank(full_rank) == 4.0


def test_effective_rank_returns_zero_for_nonfinite_features() -> None:
    features = torch.eye(4)
    features[0, 0] = float("nan")

    assert effective_rank(features) == 0.0


def test_spatial_stability_reports_norm_rank_and_target_change() -> None:
    normalized = torch.cat([torch.eye(4), -torch.eye(4)], dim=0).reshape(2, 2, 2, 4)
    pre_normalized = normalized * 3.0
    previous = normalized.roll(shifts=1, dims=1)

    report = summarize_spatial_stability(
        pre_normalized_targets=pre_normalized,
        normalized_targets=normalized,
        previous_normalized_targets=previous,
    )

    assert report["normalized_token_norm_mean"] == 1.0
    assert report["pre_normalized_token_norm_mean"] == 3.0
    assert report["effective_rank"] == 4.0
    assert not report["collapse_warning"]
    assert not report["effective_rank_warning"]
    assert report["pairwise_state_population"] == 4
    assert report["pairwise_state_sample_size"] == 4
    assert report["pairwise_state_sampling"] == "full_population"
    assert report["mean_target_change_mse"] > 0.0


def test_spatial_stability_bounds_pairwise_state_sample() -> None:
    normalized = torch.arange(20 * 2 * 3, dtype=torch.float32).reshape(5, 4, 2, 3)

    report = summarize_spatial_stability(
        pre_normalized_targets=normalized,
        normalized_targets=normalized,
        max_pairwise_states=6,
    )

    assert report["pairwise_state_population"] == 20
    assert report["pairwise_state_sample_size"] == 6
    assert report["pairwise_state_sampling"] == "deterministic_stride"
    assert report["mean_pairwise_state_mse"] > 0.0


def test_spatial_stability_rejects_invalid_pairwise_sample_size() -> None:
    targets = torch.zeros(2, 2, 3, 4)

    with pytest.raises(ValueError, match="max_rows must be positive"):
        summarize_spatial_stability(
            pre_normalized_targets=targets,
            normalized_targets=targets,
            max_pairwise_states=0,
        )


def test_spatial_stability_warns_on_collapsed_static_targets() -> None:
    targets = torch.zeros(2, 2, 3, 4)

    report = summarize_spatial_stability(
        pre_normalized_targets=targets,
        normalized_targets=targets,
        previous_normalized_targets=targets,
    )

    assert report["collapse_warning"]
    assert report["effective_rank_warning"]
    assert report["near_static_target_warning"]


def test_spatial_stability_reports_nonfinite_targets_without_linalg_failure() -> None:
    targets = torch.zeros(2, 2, 3, 4)
    targets[0, 0, 0, 0] = float("nan")

    report = summarize_spatial_stability(
        pre_normalized_targets=targets,
        normalized_targets=targets,
        previous_normalized_targets=targets,
    )

    assert report["nonfinite_feature_warning"]
    assert report["effective_rank"] == 0.0
    assert report["collapse_warning"]
    assert report["effective_rank_warning"]
    assert report["near_static_target_warning"]
