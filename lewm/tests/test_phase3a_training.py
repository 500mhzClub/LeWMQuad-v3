from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch

from lewm.benchmarks.phase3a_explore_claim import egocentric_explore_claim_predictions
from lewm.benchmarks.phase3a_marker_memory import egocentric_marker_memory_predictions
from lewm.benchmarks.phase3a_positive_control import (
    ACTION_NAMES,
    action_vector,
    generate_phase3a_rows,
)
from lewm.benchmarks.phase3a_training import (
    CONSEQUENCE_TARGET_NAMES,
    Phase3AMaterializedDataset,
    materialize_phase3a_batch,
    materialize_phase3a_batch_uncached,
    primitive_selection_summary,
    source_grouped_batches,
)
from lewm.models.phase3a_jepa import Phase3AJepaModel


def _load_gate_module():
    path = Path(__file__).resolve().parents[2] / "scripts/check_jepa_phase3a_gate.py"
    spec = importlib.util.spec_from_file_location("check_jepa_phase3a_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=101,
    )
    return rows


def test_materialize_phase3a_batch_shapes_and_hard_negatives() -> None:
    rows = _rows()
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)

    assert batch.vision.shape == (len(ACTION_NAMES) ** 2, 3, 3, 9, 9)
    assert batch.history_vision.shape == (len(ACTION_NAMES) ** 2, 0, 3, 9, 9)
    assert batch.history_actions.shape == (len(ACTION_NAMES) ** 2, 0, len(ACTION_NAMES))
    assert batch.actions.shape == (len(ACTION_NAMES) ** 2, 2, len(ACTION_NAMES))
    assert batch.consequence_targets.shape == (
        len(ACTION_NAMES) ** 2,
        2,
        len(CONSEQUENCE_TARGET_NAMES),
    )
    assert batch.utility_group_ids.shape == (len(ACTION_NAMES) ** 2,)
    assert batch.utility_group_ids.unique().numel() == 1
    assert batch.utility_mask.all()
    assert batch.wrong_actions.shape == (
        len(ACTION_NAMES) ** 2,
        2,
        1,
        len(ACTION_NAMES),
    )
    assert batch.wrong_mask.all()
    assert not batch.non_hold_mask.all()
    assert batch.marker_memory_valid_mask.shape == (len(ACTION_NAMES) ** 2,)
    assert batch.marker_memory_delta_targets.shape == (len(ACTION_NAMES) ** 2, 2)
    assert batch.marker_memory_claim_targets.shape == (len(ACTION_NAMES) ** 2,)
    assert batch.marker_memory_score_targets.shape == (len(ACTION_NAMES) ** 2,)
    assert batch.marker_memory_start_valid_mask.shape == (len(ACTION_NAMES) ** 2,)
    assert batch.marker_memory_start_delta_targets.shape == (
        len(ACTION_NAMES) ** 2,
        2,
    )
    assert batch.marker_memory_start_cell_valid_mask.shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert batch.marker_memory_start_cell_targets.shape == (len(ACTION_NAMES) ** 2,)
    assert batch.spatial_frontier_history_observation_targets.shape == (
        len(ACTION_NAMES) ** 2,
        0,
        4,
        9,
        9,
    )
    assert batch.spatial_frontier_vision_observation_targets.shape == (
        len(ACTION_NAMES) ** 2,
        3,
        4,
        9,
        9,
    )


def test_phase3a_materialized_dataset_matches_uncached_batch() -> None:
    rows = _rows()
    indices = source_grouped_batches(rows, source_states_per_batch=2, shuffle=False)[0]

    cached = Phase3AMaterializedDataset(rows).materialize_batch(indices)
    uncached = materialize_phase3a_batch_uncached(rows, indices)

    tensor_fields = (
        "vision",
        "history_vision",
        "history_actions",
        "actions",
        "utility_targets",
        "consequence_targets",
        "utility_group_ids",
        "utility_mask",
        "wrong_actions",
        "wrong_mask",
        "non_hold_mask",
        "marker_memory_valid_mask",
        "marker_memory_delta_targets",
        "marker_memory_claim_targets",
        "marker_memory_score_targets",
        "marker_memory_start_valid_mask",
        "marker_memory_start_delta_targets",
        "marker_memory_start_cell_valid_mask",
        "marker_memory_start_cell_targets",
        "spatial_frontier_history_observation_targets",
        "spatial_frontier_vision_observation_targets",
    )
    for field in tensor_fields:
        assert torch.equal(getattr(cached, field), getattr(uncached, field))
    assert cached.source_keys == uncached.source_keys
    assert cached.first_primitives == uncached.first_primitives


def test_phase3a_spatial_frontier_targets_follow_randomized_marker_palette() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=4,
        seed=201,
        view_size=7,
        history_steps=3,
        history_policy="explore",
        utility_mode="novelty_then_claim",
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=True,
        history_goal_marker=True,
        future_goal_marker=True,
        color_palette_mode="scene_random",
    )
    assert rows[0]["render_palette"]["goal"] != [0.10, 0.85, 0.18]

    batch = materialize_phase3a_batch(rows, list(range(len(rows))))
    marker_pixels = (
        batch.spatial_frontier_history_observation_targets[:, :, 0].sum()
        + batch.spatial_frontier_vision_observation_targets[:, :, 0].sum()
    )
    assert marker_pixels.item() > 0


def test_phase3a_model_forward_produces_finite_metrics() -> None:
    rows = _rows()
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert torch.isfinite(output["loss"])
    assert output["rollout"].shape[:3] == (len(ACTION_NAMES) ** 2, 2, 81)
    assert output["targets"].shape == output["rollout"].shape
    assert output["target_delta"].shape == output["rollout"].shape
    assert output["rollout_delta"].shape == output["rollout"].shape
    assert output["decision_rollout"].shape[:3] == (
        len(ACTION_NAMES) ** 2,
        2,
        4,
    )
    assert output["memory_context"].shape == (len(ACTION_NAMES) ** 2, 16)
    assert output["decision_teacher_forced_prediction"].shape == (
        output["decision_rollout"].shape
    )
    assert output["decision_targets"].shape == output["decision_rollout"].shape
    assert output["decision_previous_targets"].shape == output["decision_rollout"].shape
    assert output["utility_prediction"].shape == (len(ACTION_NAMES) ** 2,)
    assert output["candidate_score_prediction"].shape == (len(ACTION_NAMES) ** 2,)
    assert output["candidate_claim_logit"].shape == (len(ACTION_NAMES) ** 2,)
    assert output["online_marker_memory_score_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["online_frontier_marker_score_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["candidate_marker_memory_score_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["candidate_marker_memory_delta_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
        2,
    )
    assert output["candidate_marker_memory_claim_logit"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["structured_marker_memory_score_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["structured_marker_memory_start_delta_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
        2,
    )
    assert output["structured_marker_memory_delta_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
        2,
    )
    assert output["categorical_marker_memory_score_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["categorical_marker_memory_logits"].shape == (
        len(ACTION_NAMES) ** 2,
        25,
    )
    assert output["spatial_marker_memory_score_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["spatial_marker_memory_start_belief"].shape == (
        len(ACTION_NAMES) ** 2,
        81,
    )
    assert output["spatial_marker_memory_mass"].shape == (len(ACTION_NAMES) ** 2,)
    assert output["spatial_frontier_memory_score_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["spatial_frontier_marker_belief"].shape == (
        len(ACTION_NAMES) ** 2,
        81,
    )
    assert output["spatial_frontier_marker_mass"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["spatial_frontier_observed_map"].shape == (
        len(ACTION_NAMES) ** 2,
        9,
        9,
    )
    assert output["spatial_frontier_free_map"].shape == (
        len(ACTION_NAMES) ** 2,
        9,
        9,
    )
    assert output["spatial_frontier_blocked_map"].shape == (
        len(ACTION_NAMES) ** 2,
        9,
        9,
    )
    assert output["consequence_utility_prediction"].shape == (len(ACTION_NAMES) ** 2,)
    assert output["free_running_wrong_predictions"].shape[:3] == (
        len(ACTION_NAMES) ** 2,
        2,
        1,
    )
    assert output["decision_wrong_predictions"].shape[:3] == (
        len(ACTION_NAMES) ** 2,
        2,
        1,
    )
    assert torch.isfinite(output["free_running_action_contrast_loss"])
    assert torch.isfinite(output["free_running_zero_contrast_loss"])
    assert torch.isfinite(output["rollout_delta_loss"])
    assert torch.isfinite(output["teacher_forced_delta_loss"])
    assert torch.isfinite(output["decision_prediction_loss"])
    assert torch.isfinite(output["decision_delta_loss"])
    assert torch.isfinite(output["decision_teacher_forced_prediction_loss"])
    assert torch.isfinite(output["decision_teacher_forced_delta_loss"])
    assert torch.isfinite(output["decision_action_contrast_loss"])
    assert torch.isfinite(output["decision_zero_contrast_loss"])
    assert torch.isfinite(output["decision_teacher_forced_action_contrast_loss"])
    assert torch.isfinite(output["decision_teacher_forced_zero_contrast_loss"])
    assert torch.isfinite(output["utility_head_loss"])
    assert torch.isfinite(output["candidate_score_loss"])
    assert torch.isfinite(output["candidate_claim_loss"])
    assert torch.isfinite(output["candidate_marker_memory_loss"])
    assert torch.isfinite(output["candidate_marker_memory_delta_loss"])
    assert torch.isfinite(output["candidate_marker_memory_claim_loss"])
    assert torch.isfinite(output["candidate_marker_memory_ranking_loss"])
    assert torch.isfinite(output["candidate_marker_memory_ranking_ce_loss"])
    assert torch.isfinite(output["structured_marker_memory_loss"])
    assert torch.isfinite(output["structured_marker_memory_start_delta_loss"])
    assert torch.isfinite(output["structured_marker_memory_final_delta_loss"])
    assert torch.isfinite(output["structured_marker_memory_ranking_loss"])
    assert torch.isfinite(output["structured_marker_memory_ranking_ce_loss"])
    assert torch.isfinite(output["categorical_marker_memory_loss"])
    assert torch.isfinite(output["categorical_marker_memory_ranking_loss"])
    assert torch.isfinite(output["categorical_marker_memory_ranking_ce_loss"])
    assert torch.isfinite(output["spatial_frontier_memory_loss"])
    assert torch.isfinite(output["spatial_frontier_memory_occupancy_loss"])
    assert torch.isfinite(output["spatial_frontier_memory_marker_loss"])
    assert torch.isfinite(output["spatial_frontier_memory_ranking_loss"])
    assert torch.isfinite(output["spatial_frontier_memory_ranking_ce_loss"])
    assert torch.isfinite(output["candidate_score_ranking_loss"])
    assert torch.isfinite(output["candidate_score_ranking_ce_loss"])
    assert torch.isfinite(output["candidate_score_ranking_regression_loss"])
    assert output["candidate_score_ranking_group_count"].item() == 1.0
    assert torch.isfinite(output["utility_ranking_loss"])
    assert output["utility_ranking_group_count"].item() == 1.0
    assert output["consequence_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
        2,
        len(CONSEQUENCE_TARGET_NAMES),
    )
    assert torch.isfinite(output["consequence_loss"])


def test_phase3a_online_frontier_marker_score_matches_reference() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=139,
        horizon=2,
        view_size=7,
        history_steps=3,
        history_policy="explore",
        utility_mode="novelty_then_claim",
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        future_goal_marker=True,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=7,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )
    expected = torch.tensor(
        egocentric_explore_claim_predictions([rows[index] for index in indices]),
        dtype=output["online_frontier_marker_score_prediction"].dtype,
    )

    assert torch.allclose(
        output["online_frontier_marker_score_prediction"],
        expected,
        atol=1e-5,
    )


def test_phase3a_online_frontier_marker_score_clips_to_finite_crop() -> None:
    free = [[[0.72 for _ in range(7)] for _ in range(7)] for _ in range(3)]
    history_sequence = (
        "turn_left",
        "hold",
        "turn_right",
        "forward",
        "turn_left",
        "forward",
    )
    candidate_sequence = ("forward", "forward", "turn_left", "forward")
    row = {
        "history_observations_rgb": [free for _ in history_sequence],
        "history_actions": [list(action_vector(action)) for action in history_sequence],
        "history_goal_beacon": False,
        "current_goal_beacon": False,
        "start_observation_rgb": free,
        "active_blocks": [list(action_vector(action)) for action in candidate_sequence],
    }
    model = Phase3AJepaModel(
        view_size=7,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
    )
    actual = model.online_frontier_marker_score(
        torch.tensor([row["history_observations_rgb"]], dtype=torch.float32),
        torch.tensor([row["history_actions"]], dtype=torch.float32),
        torch.tensor([row["active_blocks"]], dtype=torch.float32),
        torch.tensor([row["start_observation_rgb"]], dtype=torch.float32),
    )
    expected = torch.tensor(
        egocentric_explore_claim_predictions([row]),
        dtype=actual.dtype,
    )

    assert torch.allclose(actual, expected, atol=1e-5)


def test_phase3a_spatial_frontier_memory_forward_is_valid() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=151,
        horizon=2,
        view_size=7,
        history_steps=3,
        history_policy="explore",
        utility_mode="novelty_then_claim",
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        future_goal_marker=True,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=7,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        spatial_frontier_memory_loss_lambda=1.0,
        spatial_frontier_observation_loss_lambda=1.0,
        spatial_frontier_memory_score_loss_lambda=1.0,
        spatial_frontier_memory_ranking_loss_lambda=1.0,
        spatial_frontier_memory_score_weight=1.0,
        spatial_frontier_memory_detector_init="neutral",
        spatial_frontier_memory_detector_arch="mlp",
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
        candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
        candidate_marker_memory_claim_targets=batch.marker_memory_claim_targets,
        candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
        structured_marker_memory_valid_mask=batch.marker_memory_start_valid_mask,
        structured_marker_memory_start_delta_targets=(
            batch.marker_memory_start_delta_targets
        ),
        spatial_frontier_history_observation_targets=(
            batch.spatial_frontier_history_observation_targets
        ),
        spatial_frontier_vision_observation_targets=(
            batch.spatial_frontier_vision_observation_targets
        ),
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert output["spatial_frontier_memory_score_prediction"].shape == (
        len(ACTION_NAMES) ** 2,
    )
    assert output["spatial_frontier_observed_map"].shape == (
        len(ACTION_NAMES) ** 2,
        7,
        7,
    )
    assert torch.isfinite(output["spatial_frontier_memory_loss"])
    assert torch.isfinite(output["spatial_frontier_memory_score_loss"])
    assert torch.isfinite(output["spatial_frontier_memory_occupancy_loss"])
    assert torch.isfinite(output["spatial_frontier_memory_marker_loss"])
    assert torch.isfinite(output["spatial_frontier_observation_loss"])
    assert output["spatial_frontier_observation_frame_count"].item() > 0.0
    assert torch.isfinite(output["spatial_frontier_memory_ranking_loss"])
    assert output["spatial_frontier_memory_ranking_group_count"].item() == 1.0
    assert output["spatial_frontier_observed_map"].min().item() >= 0.0
    assert output["spatial_frontier_observed_map"].max().item() <= 1.0
    assert output["spatial_frontier_blocked_map"].min().item() >= 0.0
    assert output["spatial_frontier_blocked_map"].max().item() <= 1.0


def test_phase3a_spatial_frontier_memory_can_exceed_view_size() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=153,
        horizon=2,
        view_size=7,
        history_steps=3,
        history_policy="explore",
        utility_mode="novelty_then_claim",
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        future_goal_marker=True,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=7,
        spatial_memory_size=15,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        spatial_frontier_memory_loss_lambda=1.0,
        spatial_frontier_observation_loss_lambda=1.0,
        spatial_frontier_memory_score_loss_lambda=1.0,
        spatial_frontier_memory_ranking_loss_lambda=1.0,
        spatial_frontier_memory_score_weight=1.0,
        spatial_frontier_memory_detector_init="neutral",
        spatial_frontier_memory_detector_arch="mlp",
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
        candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
        candidate_marker_memory_claim_targets=batch.marker_memory_claim_targets,
        candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
        structured_marker_memory_valid_mask=batch.marker_memory_start_valid_mask,
        structured_marker_memory_start_delta_targets=(
            batch.marker_memory_start_delta_targets
        ),
        spatial_frontier_history_observation_targets=(
            batch.spatial_frontier_history_observation_targets
        ),
        spatial_frontier_vision_observation_targets=(
            batch.spatial_frontier_vision_observation_targets
        ),
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert output["spatial_frontier_marker_belief"].shape == (
        len(ACTION_NAMES) ** 2,
        225,
    )
    assert output["spatial_marker_memory_start_belief"].shape == (
        len(ACTION_NAMES) ** 2,
        225,
    )
    assert output["spatial_frontier_observed_map"].shape == (
        len(ACTION_NAMES) ** 2,
        15,
        15,
    )
    assert output["spatial_frontier_free_map"].shape == (
        len(ACTION_NAMES) ** 2,
        15,
        15,
    )
    assert output["spatial_frontier_blocked_map"].shape == (
        len(ACTION_NAMES) ** 2,
        15,
        15,
    )
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["spatial_frontier_memory_score_loss"])
    assert torch.isfinite(output["spatial_frontier_observation_loss"])

    stepped = model.step_spatial_frontier_memory(
        output["spatial_frontier_marker_belief"],
        output["spatial_frontier_marker_mass"],
        output["spatial_frontier_observed_map"],
        output["spatial_frontier_free_map"],
        output["spatial_frontier_blocked_map"],
        batch.actions[:, :1],
        batch.vision[:, 0],
    )
    assert stepped[0].shape == (len(ACTION_NAMES) ** 2, 225)
    assert stepped[1].shape == (len(ACTION_NAMES) ** 2,)
    assert stepped[2].shape == (len(ACTION_NAMES) ** 2, 15, 15)
    assert torch.allclose(
        stepped[0].sum(dim=-1),
        torch.ones(len(ACTION_NAMES) ** 2),
    )


def test_phase3a_spatial_frontier_marker_update_threshold() -> None:
    model = Phase3AJepaModel(
        view_size=7,
        latent_dim=16,
        spatial_frontier_memory_loss_lambda=1.0,
        spatial_frontier_marker_update_threshold=0.5,
        spatial_frontier_marker_update_width=0.25,
    )

    weights = model.spatial_frontier_marker_update_weight(
        torch.tensor([0.25, 0.5, 0.625, 0.75, 0.9])
    )

    assert torch.allclose(weights, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_phase3a_candidate_score_start_only_mode_is_valid() -> None:
    rows = _rows()
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
    )

    assert model.candidate_score_gradient_mode == "start_only"
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["candidate_score_ranking_loss"])


def test_phase3a_candidate_score_sequence_action_summary_is_valid() -> None:
    rows = _rows()
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        candidate_score_action_summary="sequence",
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
    )

    assert model.candidate_score_action_summary == "sequence"
    assert model.candidate_action_feature_dim == 16
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["candidate_score_ranking_loss"])


def test_phase3a_candidate_claim_head_forward_is_valid() -> None:
    rows = _rows()
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        candidate_claim_loss_lambda=1.0,
        candidate_score_claim_logit_weight=0.5,
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert model.candidate_claim_head is not None
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["candidate_claim_loss"])
    assert output["candidate_claim_logit"].shape == (len(ACTION_NAMES) ** 2,)


def test_phase3a_candidate_marker_memory_head_forward_is_valid() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=109,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        goal_variants_per_source=2,
        minimum_goal_variant_distance=1,
        maximum_goal_variant_distance=2,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        candidate_score_action_summary="sequence",
        use_memory_context=True,
        memory_frame_summary="spatial",
        candidate_marker_memory_loss_lambda=1.0,
        candidate_marker_memory_score_weight=0.5,
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
        candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
        candidate_marker_memory_claim_targets=batch.marker_memory_claim_targets,
        candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert model.candidate_marker_memory_delta_head is not None
    assert model.candidate_marker_memory_claim_head is not None
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["candidate_marker_memory_loss"])
    assert output["candidate_marker_memory_delta_prediction"].shape == (
        len(indices),
        2,
    )
    assert output["candidate_marker_memory_claim_logit"].shape == (len(indices),)


def test_phase3a_candidate_marker_memory_ranking_loss_is_valid() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=117,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        goal_variants_per_source=2,
        minimum_goal_variant_distance=1,
        maximum_goal_variant_distance=2,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        candidate_score_action_summary="sequence",
        use_memory_context=True,
        memory_frame_summary="spatial",
        candidate_marker_memory_ranking_loss_lambda=1.0,
        candidate_marker_memory_score_weight=1.0,
        candidate_marker_memory_score_mode="distance",
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
        candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
        candidate_marker_memory_claim_targets=batch.marker_memory_claim_targets,
        candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
    )

    assert model.candidate_marker_memory_delta_head is not None
    assert model.candidate_marker_memory_score_mode == "distance"
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["candidate_marker_memory_ranking_loss"])
    assert output["candidate_marker_memory_ranking_group_count"].item() == 1.0


def test_phase3a_structured_marker_memory_head_rolls_candidate_actions() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=119,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        goal_variants_per_source=2,
        minimum_goal_variant_distance=1,
        maximum_goal_variant_distance=2,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        candidate_score_action_summary="sequence",
        use_memory_context=True,
        memory_frame_summary="spatial",
        structured_marker_memory_loss_lambda=1.0,
        structured_marker_memory_score_weight=1.0,
        structured_marker_memory_ranking_loss_lambda=1.0,
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
        candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
        candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
        structured_marker_memory_valid_mask=batch.marker_memory_start_valid_mask,
        structured_marker_memory_start_delta_targets=(
            batch.marker_memory_start_delta_targets
        ),
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert model.structured_marker_memory_start_head is not None
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["structured_marker_memory_loss"])
    assert torch.isfinite(output["structured_marker_memory_ranking_loss"])
    assert output["structured_marker_memory_ranking_group_count"].item() == 1.0
    assert output["structured_marker_memory_start_delta_prediction"].shape == (
        len(indices),
        2,
    )
    assert output["structured_marker_memory_delta_prediction"].shape == (
        len(indices),
        2,
    )


def test_phase3a_categorical_marker_memory_head_scores_candidate_claims() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=121,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        goal_variants_per_source=2,
        minimum_goal_variant_distance=1,
        maximum_goal_variant_distance=2,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        candidate_score_action_summary="sequence",
        use_memory_context=True,
        memory_frame_summary="spatial",
        memory_marker_features=True,
        categorical_marker_memory_loss_lambda=1.0,
        categorical_marker_memory_score_weight=1.0,
        categorical_marker_memory_ranking_loss_lambda=1.0,
        categorical_marker_memory_radius=2,
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
        candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
        candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
        categorical_marker_memory_valid_mask=(
            batch.marker_memory_start_cell_valid_mask
        ),
        categorical_marker_memory_cell_targets=batch.marker_memory_start_cell_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert model.categorical_marker_memory_logits_head is not None
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["categorical_marker_memory_loss"])
    assert torch.isfinite(output["categorical_marker_memory_ranking_loss"])
    assert output["categorical_marker_memory_ranking_group_count"].item() == 1.0
    assert output["categorical_marker_memory_logits"].shape == (len(indices), 25)
    assert output["categorical_marker_memory_score_prediction"].shape == (
        len(indices),
    )


def test_phase3a_spatial_marker_memory_head_scores_candidate_claims() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=123,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        goal_variants_per_source=2,
        minimum_goal_variant_distance=1,
        maximum_goal_variant_distance=2,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        candidate_score_action_summary="sequence",
        spatial_marker_memory_loss_lambda=1.0,
        spatial_marker_memory_score_weight=1.0,
        spatial_marker_memory_ranking_loss_lambda=1.0,
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
        candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
        structured_marker_memory_valid_mask=batch.marker_memory_start_valid_mask,
        structured_marker_memory_start_delta_targets=(
            batch.marker_memory_start_delta_targets
        ),
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert model.spatial_marker_memory_detector is not None
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["spatial_marker_memory_loss"])
    assert torch.isfinite(output["spatial_marker_memory_cell_loss"])
    assert torch.isfinite(output["spatial_marker_memory_mass_loss"])
    assert torch.isfinite(output["spatial_marker_memory_ranking_loss"])
    assert output["spatial_marker_memory_ranking_group_count"].item() == 1.0
    assert output["spatial_marker_memory_score_prediction"].shape == (len(indices),)
    assert output["spatial_marker_memory_start_belief"].shape == (len(indices), 81)
    assert output["spatial_marker_memory_mass"].shape == (len(indices),)


def test_phase3a_memory_context_forward_is_valid() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=113,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=True,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        use_memory_context=True,
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert batch.history_vision.shape == (len(ACTION_NAMES) ** 2, 3, 3, 9, 9)
    assert batch.history_actions.shape == (len(ACTION_NAMES) ** 2, 3, len(ACTION_NAMES))
    assert model.use_memory_context
    assert torch.isfinite(output["loss"])
    assert output["memory_context"].shape == (len(ACTION_NAMES) ** 2, 16)
    assert output["memory_context"].abs().sum() > 0.0


def test_phase3a_online_marker_memory_score_matches_baseline() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=211,
        history_steps=4,
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        history_goal_marker=True,
        future_goal_marker=True,
        goal_variants_per_source=2,
        minimum_goal_variant_distance=1,
        maximum_goal_variant_distance=2,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    selected_rows = [rows[index] for index in indices]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    expected = torch.tensor(
        egocentric_marker_memory_predictions(selected_rows),
        dtype=output["online_marker_memory_score_prediction"].dtype,
    )
    assert torch.allclose(
        output["online_marker_memory_score_prediction"],
        expected,
        atol=1e-4,
    )


def test_phase3a_spatial_memory_context_forward_is_valid() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=119,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=False,
        current_goal_marker=False,
        goal_variants_per_source=2,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        use_memory_context=True,
        memory_frame_summary="spatial",
    )

    output = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert model.memory_frame_summary == "spatial"
    assert torch.isfinite(output["loss"])
    assert output["memory_context"].shape == (len(ACTION_NAMES) ** 2, 16)
    assert output["memory_context"].abs().sum() > 0.0


def test_phase3a_memory_context_changes_scoring_heads() -> None:
    rows, _ = generate_phase3a_rows(
        split="train",
        scene_count=1,
        source_states_per_scene=2,
        seed=127,
        history_steps=3,
        current_goal_beacon=False,
        history_goal_beacon=True,
        current_goal_marker=False,
        history_goal_marker=False,
        future_goal_marker=False,
        goal_variants_per_source=2,
    )
    indices = source_grouped_batches(rows, source_states_per_batch=1, shuffle=False)[0]
    batch = materialize_phase3a_batch(rows, indices)
    model = Phase3AJepaModel(
        view_size=9,
        latent_dim=16,
        pred_layers=1,
        pred_heads=2,
        pred_dim_head=8,
        pred_mlp_dim=32,
        candidate_score_gradient_mode="start_only",
        use_memory_context=True,
    )

    with_memory = model(
        vision=batch.vision,
        history_vision=batch.history_vision,
        history_actions=batch.history_actions,
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )
    without_memory = model(
        vision=batch.vision,
        history_vision=batch.history_vision[:, :0],
        history_actions=batch.history_actions[:, :0],
        actions=batch.actions,
        utility_targets=batch.utility_targets,
        consequence_targets=batch.consequence_targets,
        utility_group_ids=batch.utility_group_ids,
        utility_mask=batch.utility_mask,
        wrong_actions=batch.wrong_actions,
        wrong_mask=batch.wrong_mask,
        non_hold_mask=batch.non_hold_mask,
        return_latents=True,
    )

    assert with_memory["memory_context"].abs().sum() > 0.0
    assert without_memory["memory_context"].abs().sum() == 0.0
    assert not torch.allclose(
        with_memory["utility_prediction"],
        without_memory["utility_prediction"],
    )
    assert not torch.allclose(
        with_memory["candidate_score_prediction"],
        without_memory["candidate_score_prediction"],
    )


def test_primitive_selection_summary_scores_source_groups() -> None:
    rows = _rows()
    predictions = [
        float(row["consequence_labels"]["target_utility"])
        for row in rows
    ]

    summary = primitive_selection_summary(rows, predictions)

    assert summary["source_states"] == 2
    assert summary["primitive_match_rate"] == 1.0
    assert summary["mean_target_utility_regret"] == 0.0


def test_phase3a_gate_passes_and_fails_expected_reports() -> None:
    gate_module = _load_gate_module()
    report = {
        "action_only_prior": {
            "primitive_match_rate": 0.25,
            "mean_target_utility_regret": 1.0,
        },
        "final_validation": {
            "primitive_selection": {
                "primitive_match_rate": 0.75,
                "mean_target_utility_regret": 0.25,
            },
            "rollout_controls": {
                "representation": {"collapse_warning": False},
                "per_horizon_step": [
                    {
                        "step": 1,
                        "free_running_beats_persistence": True,
                        "meaningful_real_action_beats_zero": True,
                        "meaningful_real_action_beats_shuffled": True,
                    }
                ],
            },
            "decision_rollout_controls": {
                "representation": {"collapse_warning": False},
                "per_horizon_step": [
                    {
                        "step": 1,
                        "free_running_beats_persistence": True,
                        "meaningful_real_action_beats_zero": True,
                        "meaningful_real_action_beats_shuffled": True,
                    },
                    {
                        "step": 2,
                        "free_running_beats_persistence": False,
                        "meaningful_real_action_beats_zero": True,
                        "meaningful_real_action_beats_shuffled": True,
                    }
                ],
            },
        },
    }

    passed = gate_module.evaluate_gate(report)
    assert passed["passed"]
    assert passed["schema"] == "jepa_phase3a_positive_control_gate_v1"
    assert passed["observed"]["control_surface"] == "decision_rollout_controls"
    assert passed["observed"]["per_horizon_gate_checks"][1] == {
        "step": 2,
        "first_horizon_requires_persistence": False,
        "later_horizon_persistence_or_action_advantage": True,
        "persistence_or_action_advantage_passed": True,
    }

    collapsed_report = copy.deepcopy(report)
    collapsed_report["final_validation"]["decision_rollout_controls"]["representation"][
        "collapse_warning"
    ] = True
    failed = gate_module.evaluate_gate(collapsed_report)
    assert not failed["passed"]
    assert "collapse_warning" in failed["failure_reasons"]

    weak_later_horizon_report = copy.deepcopy(report)
    weak_later_horizon_report["final_validation"]["decision_rollout_controls"][
        "per_horizon_step"
    ][1]["meaningful_real_action_beats_shuffled"] = False
    failed = gate_module.evaluate_gate(weak_later_horizon_report)
    assert not failed["passed"]
    assert (
        "step_2_persistence_or_action_advantage_below_threshold"
        in failed["failure_reasons"]
    )
    assert "step_2_hard_negative_advantage_below_threshold" in failed[
        "failure_reasons"
    ]
