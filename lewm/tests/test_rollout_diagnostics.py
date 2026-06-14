from __future__ import annotations

import torch

from lewm.benchmarks.rollout_diagnostics import summarize_rollout_controls


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
