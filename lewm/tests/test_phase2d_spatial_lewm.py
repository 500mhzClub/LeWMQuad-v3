from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from lewm.models.phase2d_spatial_lewm import (
    LearnedSlotGeometry,
    Phase2DSpatialLeWorldModel,
    action_identifiability_losses,
    action_utility_losses,
    normalize_spatial_tokens,
)


def _model(
    *,
    target_ema_momentum: float | None = None,
    action_identifiability_lambda: float = 0.0,
    zero_action_lambda: float = 0.0,
    consequence_dim: int = 0,
    consequence_loss_lambda: float = 0.0,
    action_utility_loss_lambda: float = 0.0,
    action_utility_regression_weight: float = 0.1,
) -> Phase2DSpatialLeWorldModel:
    return Phase2DSpatialLeWorldModel(
        latent_dim=12,
        cmd_dim=6,
        pred_layers=1,
        pred_heads=3,
        pred_dim_head=4,
        pred_mlp_dim=24,
        pred_dropout=0.0,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        appearance_sigreg_lambda=0.0,
        spatial_variance_lambda=1.0,
        action_identifiability_lambda=action_identifiability_lambda,
        zero_action_lambda=zero_action_lambda,
        consequence_dim=consequence_dim,
        consequence_loss_lambda=consequence_loss_lambda,
        action_utility_loss_lambda=action_utility_loss_lambda,
        action_utility_regression_weight=action_utility_regression_weight,
        sigreg_projections=8,
        sigreg_knots=5,
        target_ema_momentum=target_ema_momentum,
    )


def test_normalize_spatial_tokens_has_unit_norm() -> None:
    normalized = normalize_spatial_tokens(torch.randn(2, 3, 4, 12))

    assert torch.allclose(
        normalized.norm(dim=-1),
        torch.ones(2, 3, 4),
        atol=1e-6,
    )


def test_action_identifiability_losses_use_exhaustive_masks_and_dynamic_margin() -> None:
    targets = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    previous = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    real = targets.clone()
    wrong = torch.stack(
        [
            targets + 0.01,
            targets + 1.0,
            targets + 0.02,
        ],
        dim=2,
    )
    wrong_mask = torch.tensor([[[True, True, False], [True, False, False]]])
    non_hold = torch.tensor([[True, False]])
    valid = torch.tensor([[True, True]])

    losses = action_identifiability_losses(
        real_prediction=real,
        targets=targets,
        previous_targets=previous,
        wrong_predictions=wrong,
        wrong_mask=wrong_mask,
        zero_prediction=previous,
        non_hold_mask=non_hold,
        transition_mask=valid,
    )

    assert torch.allclose(losses["margin"], torch.full((1, 2), 0.1))
    assert losses["action_identifiability_loss"] > 0.0
    assert losses["zero_action_loss"] == 0.0
    assert losses["eligible_wrong_mask"].tolist() == [[True, True]]
    assert losses["eligible_zero_mask"].tolist() == [[True, False]]


def test_phase2d_model_separates_non_batchnorm_spatial_projection_paths() -> None:
    model = _model(target_ema_momentum=0.99)

    assert model.online_target_projector is not model.prediction_projector
    assert model.online_target_projector is not model.target_projector
    assert not any(
        isinstance(module, nn.BatchNorm1d)
        for projector in (
            model.online_target_projector,
            model.prediction_projector,
            model.target_projector,
        )
        for module in projector.modules()
    )
    assert model.spatial_target_std == pytest.approx(1.0 / math.sqrt(12))


def test_phase2d_online_control_backpropagates_and_normalizes_outputs() -> None:
    torch.manual_seed(31)
    model = _model()
    vision = torch.randn(2, 3, 3, 28, 28)
    actions = torch.randn(2, 2, 6)

    output = model(
        vision,
        actions,
        transition_mask=torch.tensor([[True, True], [True, False]]),
        return_latents=True,
    )
    output["loss"].backward()

    assert output["real_prediction"].shape == (2, 2, 4, 12)
    assert torch.allclose(
        output["real_prediction"].norm(dim=-1),
        torch.ones(2, 2, 4),
        atol=1e-5,
    )
    assert torch.allclose(
        output["target_normalized_all"].norm(dim=-1),
        torch.ones(2, 3, 4),
        atol=1e-5,
    )
    assert output["valid_transition_count"] == 3
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.online_target_projector.linear.weight.grad is not None
    assert model.prediction_projector.linear.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())


def test_phase2d_c2_requires_and_emits_registered_action_controls() -> None:
    torch.manual_seed(37)
    model = _model(
        target_ema_momentum=0.99,
        action_identifiability_lambda=1.0,
        zero_action_lambda=1.0,
    )
    vision = torch.randn(2, 3, 3, 28, 28)
    actions = torch.randn(2, 2, 6)

    with pytest.raises(ValueError, match="wrong_actions"):
        model(vision, actions)
    with pytest.raises(ValueError, match="non_hold_mask"):
        model(
            vision,
            actions,
            wrong_actions=torch.randn(2, 2, 2, 6),
            wrong_mask=torch.ones(2, 2, 2, dtype=torch.bool),
        )

    output = model(
        vision,
        actions,
        transition_mask=torch.tensor([[True, True], [True, False]]),
        wrong_actions=torch.randn(2, 2, 2, 6),
        wrong_mask=torch.tensor(
            [
                [[True, True], [True, False]],
                [[True, False], [True, True]],
            ]
        ),
        non_hold_mask=torch.tensor([[True, False], [True, True]]),
        return_latents=True,
    )

    assert output["eligible_wrong_transition_count"] == 3
    assert output["eligible_wrong_pair_count"] == 4
    assert output["eligible_zero_count"] == 2
    assert output["wrong_pair_mask"].sum() == 4
    assert output["wrong_predictions"].shape == (2, 2, 2, 4, 12)
    assert output["zero_prediction"].shape == (2, 2, 4, 12)
    assert torch.isfinite(output["action_identifiability_loss"])
    assert torch.isfinite(output["zero_action_loss"])
    assert torch.isfinite(output["hard_negative_mse"])
    assert torch.isfinite(output["zero_action_mse"])
    assert output["mean_target_change_mse"].ndim == 0
    assert output["target_change_mse"].shape == (2, 2)


def test_phase2f_consequence_head_requires_targets_when_weighted() -> None:
    model = _model(consequence_dim=5, consequence_loss_lambda=1.0)

    with pytest.raises(ValueError, match="consequence_targets"):
        model(torch.randn(2, 3, 3, 28, 28), torch.randn(2, 2, 6))


def test_phase2f_consequence_head_predicts_masked_sequence_targets() -> None:
    torch.manual_seed(39)
    model = _model(consequence_dim=5, consequence_loss_lambda=0.5)
    targets = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, True],
        ]
    )

    output = model(
        torch.randn(2, 3, 3, 28, 28),
        torch.randn(2, 2, 6),
        consequence_targets=targets,
        consequence_mask=mask,
        return_latents=True,
    )
    output["loss"].backward()

    assert output["consequence_prediction"].shape == (2, 5)
    assert output["valid_consequence_field_count"] == 7
    assert torch.isfinite(output["consequence_loss"])
    assert output["consequence_loss"] > 0.0
    assert model.consequence_head[-1].weight.grad is not None


def test_phase2g_action_utility_losses_rank_within_source_groups() -> None:
    predictions = torch.tensor([0.0, 2.0, -1.0, 5.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.5, 0.2])
    mask = torch.tensor([True, True, True, False])
    group_ids = torch.tensor([0, 0, 0, 1])

    losses = action_utility_losses(
        utility_prediction=predictions,
        utility_targets=targets,
        utility_mask=mask,
        utility_group_ids=group_ids,
        regression_weight=0.1,
    )
    losses["action_utility_loss"].backward()

    assert losses["action_utility_valid_count"] == 3
    assert losses["action_utility_group_count"] == 1
    assert torch.isfinite(losses["action_utility_loss"])
    assert predictions.grad is not None
    assert predictions.grad[3] == 0.0


def test_phase2g_action_utility_losses_support_soft_target_distribution() -> None:
    predictions = torch.tensor([0.0, 0.2, -0.1, 9.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.5, 0.2])
    mask = torch.tensor([True, True, True, False])
    group_ids = torch.tensor([0, 0, 0, 1])

    losses = action_utility_losses(
        utility_prediction=predictions,
        utility_targets=targets,
        utility_mask=mask,
        utility_group_ids=group_ids,
        regression_weight=1.0,
        ranking_loss="soft_ce",
        softmax_temperature=0.25,
    )
    losses["action_utility_loss"].backward()

    assert losses["action_utility_valid_count"] == 3
    assert losses["action_utility_group_count"] == 1
    assert torch.isfinite(losses["action_utility_loss"])
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad[:3]) == 3
    assert predictions.grad[3] == 0.0


def test_phase2g_action_utility_head_requires_targets_and_backpropagates() -> None:
    torch.manual_seed(59)
    model = _model(action_utility_loss_lambda=0.5)

    with pytest.raises(ValueError, match="action_utility_targets"):
        model(torch.randn(2, 3, 3, 28, 28), torch.randn(2, 2, 6))

    output = model(
        torch.randn(3, 3, 3, 28, 28),
        torch.randn(3, 2, 6),
        action_utility_targets=torch.tensor([0.0, 1.0, 0.5]),
        action_utility_mask=torch.tensor([True, True, True]),
        action_utility_group_ids=torch.tensor([0, 0, 0]),
        return_latents=True,
    )
    output["loss"].backward()

    assert output["action_utility_prediction"].shape == (3,)
    assert output["action_utility_valid_count"] == 3
    assert output["action_utility_group_count"] == 1
    assert torch.isfinite(output["action_utility_loss"])
    assert model.action_utility_head[-1].weight.grad is not None


def test_phase2d_ema_target_is_frozen_and_updates_separately() -> None:
    torch.manual_seed(41)
    model = _model(target_ema_momentum=0.5)
    model.train()
    encoder_before = model.target_encoder.patch_embed.weight.detach().clone()
    projector_before = model.target_projector.linear.weight.detach().clone()
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(2.0)
        model.online_target_projector.linear.weight.add_(4.0)

    model.update_target_encoder()

    assert not model.target_encoder.training
    assert not model.target_projector.training
    assert torch.allclose(
        model.target_encoder.patch_embed.weight,
        encoder_before + 1.0,
    )
    assert torch.allclose(
        model.target_projector.linear.weight,
        projector_before + 2.0,
    )
    assert all(
        not parameter.requires_grad
        for module in (model.target_encoder, model.target_projector)
        for parameter in module.parameters()
    )


def test_phase2d_ema_target_receives_no_gradient() -> None:
    model = _model(target_ema_momentum=0.99)

    output = model(torch.randn(2, 3, 3, 28, 28), torch.randn(2, 2, 6))
    output["loss"].backward()

    assert model.encoder.patch_embed.weight.grad is not None
    assert all(
        parameter.grad is None
        for module in (model.target_encoder, model.target_projector)
        for parameter in module.parameters()
    )


def test_phase2d_rollout_returns_normalized_spatial_tokens() -> None:
    model = _model()
    _appearance, spatial = model.encode_seq(torch.randn(2, 2, 3, 28, 28))

    rollout = model.rollout_spatial(spatial[:, 0], torch.randn(2, 3, 6))

    assert rollout.shape == (2, 3, 4, 12)
    assert torch.allclose(
        rollout.norm(dim=-1),
        torch.ones(2, 3, 4),
        atol=1e-5,
    )


def test_phase2d_state_only_control_ignores_actions() -> None:
    model = _model()
    model.prediction_input_mode = "state_only"
    model.eval()
    _appearance, spatial = model.encode_seq(torch.randn(2, 2, 3, 28, 28))
    actions = torch.randn(2, 2, 6)

    left = model.rollout_spatial(spatial[:, 0], actions)
    right = model.rollout_spatial(spatial[:, 0], actions + 10.0)

    assert torch.allclose(left, right)


def test_phase2d_action_only_control_ignores_observation_state() -> None:
    model = Phase2DSpatialLeWorldModel(
        latent_dim=12,
        cmd_dim=6,
        pred_layers=1,
        pred_heads=3,
        pred_dim_head=4,
        pred_mlp_dim=24,
        pred_dropout=0.0,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        appearance_sigreg_lambda=0.0,
        sigreg_projections=8,
        sigreg_knots=5,
        target_ema_momentum=0.99,
        prediction_input_mode="action_only",
    )
    model.eval()
    actions = torch.randn(2, 2, 6)

    left = model.rollout_spatial(torch.randn(2, 4, 12), actions)
    right = model.rollout_spatial(torch.randn(2, 4, 12), actions)

    assert model.action_only_state is not None
    assert torch.allclose(left, right)


def test_phase2d_diagnostic_controls_reject_action_losses() -> None:
    with pytest.raises(ValueError, match="diagnostic input controls"):
        Phase2DSpatialLeWorldModel(
            action_identifiability_lambda=1.0,
            prediction_input_mode="state_only",
        )


def test_phase2d_model_records_detached_action_control_state_amendment() -> None:
    model = Phase2DSpatialLeWorldModel(
        action_identifiability_lambda=1.0,
        zero_action_lambda=1.0,
        detach_action_control_state=True,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        pred_layers=1,
        pred_heads=3,
        pred_dim_head=4,
        pred_mlp_dim=24,
        pred_dropout=0.0,
    )

    assert model.detach_action_control_state


def test_learned_slot_geometry_pools_patch_tokens_to_fixed_slots() -> None:
    torch.manual_seed(43)
    geometry = LearnedSlotGeometry(latent_dim=12, num_slots=3)

    slots = geometry(torch.randn(2, 5, 4, 12))

    assert slots.shape == (2, 5, 3, 12)
    assert torch.isfinite(slots).all()


def test_phase2e_slot_geometry_changes_state_token_count_and_normalizes() -> None:
    torch.manual_seed(47)
    model = Phase2DSpatialLeWorldModel(
        latent_dim=12,
        cmd_dim=6,
        pred_layers=1,
        pred_heads=3,
        pred_dim_head=4,
        pred_mlp_dim=24,
        pred_dropout=0.0,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        appearance_sigreg_lambda=0.0,
        spatial_variance_lambda=1.0,
        sigreg_projections=8,
        sigreg_knots=5,
        target_ema_momentum=0.99,
        target_geometry="slot",
        num_target_slots=3,
    )

    output = model(
        torch.randn(2, 3, 3, 28, 28),
        torch.randn(2, 2, 6),
        return_latents=True,
    )

    assert model.target_geometry == "slot"
    assert model.num_state_tokens == 3
    assert output["state_raw"].shape == (2, 3, 3, 12)
    assert output["real_prediction"].shape == (2, 2, 3, 12)
    assert output["target_normalized_all"].shape == (2, 3, 3, 12)
    assert torch.allclose(
        output["real_prediction"].norm(dim=-1),
        torch.ones(2, 2, 3),
        atol=1e-5,
    )


def test_phase2e_slot_geometry_ema_updates_separate_pooler() -> None:
    torch.manual_seed(53)
    model = Phase2DSpatialLeWorldModel(
        latent_dim=12,
        cmd_dim=6,
        pred_layers=1,
        pred_heads=3,
        pred_dim_head=4,
        pred_mlp_dim=24,
        pred_dropout=0.0,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        target_ema_momentum=0.5,
        target_geometry="slot",
        num_target_slots=3,
    )
    before = model.target_geometry_module.slot_queries.detach().clone()
    with torch.no_grad():
        model.online_geometry.slot_queries.add_(2.0)

    model.update_target_encoder()

    assert not model.target_geometry_module.training
    assert torch.allclose(
        model.target_geometry_module.slot_queries,
        before + 1.0,
    )
