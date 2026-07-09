from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from lewm.models.egomotion_bev_jepa import (
    BevDecoder,
    EgomotionBevJepa,
    bev_variance_floor_loss,
    warp_bev_current_to_next,
)


def _model() -> EgomotionBevJepa:
    return EgomotionBevJepa(
        image_size=28,
        patch_size=14,
        encoder_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        bev_dim=8,
        bev_size=(8, 8),
        forward_range_m=(-0.5, 0.5),
        left_range_m=(-0.5, 0.5),
        action_dim=4,
        predictor_hidden_dim=12,
        target_ema_momentum=0.5,
        variance_target_std=0.2,
    )


def test_bev_warp_identity_and_forward_translation() -> None:
    current = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    identity, identity_mask = warp_bev_current_to_next(
        current,
        torch.zeros(1, 3),
        forward_range_m=(-0.2, 0.2),
        left_range_m=(-0.2, 0.2),
    )
    translated, translated_mask = warp_bev_current_to_next(
        current,
        torch.tensor([[0.1, 0.0, 0.0]]),
        forward_range_m=(-0.2, 0.2),
        left_range_m=(-0.2, 0.2),
    )

    assert torch.allclose(identity, current)
    assert identity_mask.all()
    assert torch.allclose(translated[:, :, :-1], current[:, :, 1:])
    assert not translated_mask[:, :, -1].any()


def test_bev_warp_lateral_and_positive_yaw_conventions() -> None:
    current = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
    translated, translated_mask = warp_bev_current_to_next(
        current,
        torch.tensor([[0.0, 0.1, 0.0]]),
        forward_range_m=(-0.2, 0.2),
        left_range_m=(-0.2, 0.2),
    )
    assert torch.allclose(translated[:, :, :, :-1], current[:, :, :, 1:])
    assert not translated_mask[:, :, :, -1].any()

    impulse = torch.zeros(1, 1, 5, 5)
    impulse[0, 0, 3, 2] = 1.0  # 0.1 m forward in the current frame.
    rotated, _mask = warp_bev_current_to_next(
        impulse,
        torch.tensor([[0.0, 0.0, math.pi / 2.0]]),
        forward_range_m=(-0.2, 0.2),
        left_range_m=(-0.2, 0.2),
    )
    # A current-frame point ahead is to the right in the next frame after a
    # positive (left) 90-degree base rotation.
    assert rotated[0, 0, 2, 1] == pytest.approx(1.0, abs=1e-6)


def test_variance_floor_rejects_collapse() -> None:
    collapsed = torch.zeros(2, 4, 3, 3)
    varied = torch.randn(32, 4, 3, 3)
    assert bev_variance_floor_loss(collapsed) > 0.9
    assert bev_variance_floor_loss(varied) < 0.2


def test_variance_floor_rejects_input_independent_spatial_template() -> None:
    torch.manual_seed(3)
    spatial_template = torch.randn(1, 4, 3, 3).expand(16, -1, -1, -1)
    assert bev_variance_floor_loss(spatial_template) > 0.9


def test_bev_decoder_each_metric_query_can_use_every_image_token() -> None:
    torch.manual_seed(5)
    decoder = BevDecoder(
        token_dim=6,
        bev_dim=8,
        token_side=2,
        bev_size=(4, 4),
        forward_range_m=(-0.1, 0.2),
        left_range_m=(-0.2, 0.1),
        attention_heads=2,
    )
    tokens = torch.randn(1, 4, 6, requires_grad=True)
    decoder(tokens)[0, :, -1, -1].sum().backward()

    token_gradient = tokens.grad.abs().sum(dim=-1)
    assert torch.all(token_gradient > 0)


def test_model_trains_predictive_encoder_and_occupancy_head() -> None:
    torch.manual_seed(7)
    model = _model()
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    delta = torch.tensor([[0.05, 0.0, 0.1], [0.0, -0.05, -0.1]])
    labels = torch.randint(0, 3, (2, 8, 8))
    mask = torch.ones(2, 8, 8, dtype=torch.bool)

    output = model(
        current,
        nxt,
        action,
        delta,
        commanded_delta_pose_current=delta,
        current_occupancy=labels,
        next_occupancy=labels,
        current_occupancy_mask=mask,
        next_occupancy_mask=mask,
        occupancy_unknown_known_weights=torch.tensor([1.0, 2.0]),
        occupancy_free_occupied_weights=torch.tensor([1.0, 3.0]),
    )
    output["loss"].backward()

    assert output["current_occupancy_logits"].shape == (2, 3, 8, 8)
    assert output["prediction_overlap_mask"].dtype == torch.bool
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.occupancy_head.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(
        parameter.grad is None for parameter in model.target_bev_decoder.parameters()
    )
    assert not output["target_next_bev"].requires_grad


def test_occupancy_mask_is_boolean_and_single_target_is_not_halved() -> None:
    torch.manual_seed(11)
    model = _model()
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    delta = torch.zeros(2, 3)
    labels = torch.randint(0, 3, (2, 8, 8))
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    output = model(
        current,
        nxt,
        action,
        delta,
        commanded_delta_pose_current=delta,
        current_occupancy=labels,
        current_occupancy_mask=mask,
    )
    expected = model._occupancy_loss(
        output["current_occupancy_logits"], labels, mask, None
    )
    assert torch.allclose(output["occupancy_loss"], expected)

    with pytest.raises(ValueError, match="boolean"):
        model(
            current,
            nxt,
            action,
            delta,
            commanded_delta_pose_current=delta,
            current_occupancy=labels,
            current_occupancy_mask=mask.float(),
        )


def test_decomposed_loss_balances_binary_classes_under_imbalance() -> None:
    labels = torch.tensor([[[0, 0, 0, 0, 0, 0, 1, 1, 1, 2]]])
    logits = torch.linspace(-1.4, 1.7, 30).reshape(1, 3, 1, 10)
    unknown_known_weights = torch.tensor([1.0 / 6.0, 1.0 / 4.0])
    free_occupied_weights = torch.tensor([1.0 / 3.0, 1.0])

    actual = EgomotionBevJepa._occupancy_loss(
        logits,
        labels,
        None,
        None,
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
    )

    known_logits = torch.logsumexp(logits[:, 1:], dim=1)
    unknown_known_logits = torch.stack((logits[:, 0], known_logits), dim=1)
    unknown_known_labels = (labels != 0).long()
    unknown_known_cell_loss = F.cross_entropy(
        unknown_known_logits,
        unknown_known_labels,
        reduction="none",
    )
    unknown_known_expected = 0.5 * (
        unknown_known_cell_loss[labels == 0].mean()
        + unknown_known_cell_loss[labels != 0].mean()
    )
    free_occupied_cell_loss = F.cross_entropy(
        logits[:, 1:],
        (labels - 1).clamp_min(0),
        reduction="none",
    )
    free_occupied_expected = 0.5 * (
        free_occupied_cell_loss[labels == 1].mean()
        + free_occupied_cell_loss[labels == 2].mean()
    )
    expected = 0.5 * unknown_known_expected + 0.5 * free_occupied_expected

    assert torch.allclose(actual, expected)


def test_decomposed_loss_masks_before_weight_normalization() -> None:
    labels = torch.tensor([[[0, 1, 2, 0]]])
    mask = torch.tensor([[[True, True, True, False]]])
    logits = torch.tensor(
        [
            [
                [[1.0, -0.2, 0.1, 0.0]],
                [[0.0, 1.2, -0.4, 0.0]],
                [[-1.0, 0.1, 1.4, 0.0]],
            ]
        ]
    )
    changed_masked_logits = logits.clone()
    changed_masked_logits[:, :, 0, -1] = torch.tensor([100.0, -100.0, 50.0])
    kwargs = {
        "unknown_known_weights": torch.tensor([7.0, 2.0]),
        "free_occupied_weights": torch.tensor([3.0, 11.0]),
    }

    reference = EgomotionBevJepa._occupancy_loss(
        logits,
        labels,
        mask,
        None,
        **kwargs,
    )
    changed = EgomotionBevJepa._occupancy_loss(
        changed_masked_logits,
        labels,
        mask,
        None,
        **kwargs,
    )
    empty = EgomotionBevJepa._occupancy_loss(
        logits,
        labels,
        torch.zeros_like(mask),
        None,
        **kwargs,
    )

    assert torch.allclose(reference, changed)
    assert empty == 0.0


def test_decomposed_occupancy_loss_without_known_cells_is_finite() -> None:
    logits = torch.tensor(
        [[[[0.2, -0.5]], [[0.1, 0.4]], [[-0.3, 0.7]]]],
        requires_grad=True,
    )
    labels = torch.zeros((1, 1, 2), dtype=torch.long)
    actual = EgomotionBevJepa._occupancy_loss(
        logits,
        labels,
        None,
        None,
        unknown_known_weights=torch.tensor([4.0, 1.0]),
        free_occupied_weights=torch.tensor([2.0, 9.0]),
    )
    unknown_known_logits = torch.stack(
        (logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)),
        dim=1,
    )
    expected = 0.5 * F.cross_entropy(
        unknown_known_logits,
        torch.zeros_like(labels),
    )

    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_decomposed_occupancy_weight_validation() -> None:
    logits = torch.zeros(1, 3, 1, 1)
    labels = torch.zeros(1, 1, 1, dtype=torch.long)
    valid = torch.ones(2)

    with pytest.raises(ValueError, match="supplied together"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=valid,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            torch.ones(3),
            unknown_known_weights=valid,
            free_occupied_weights=valid,
        )
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=torch.ones(3),
            free_occupied_weights=valid,
        )
    with pytest.raises(ValueError, match="floating point"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=torch.ones(2, dtype=torch.long),
            free_occupied_weights=valid,
        )
    with pytest.raises(ValueError, match="finite"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=torch.tensor([1.0, float("nan")]),
            free_occupied_weights=valid,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        EgomotionBevJepa._occupancy_loss(
            logits,
            labels,
            None,
            None,
            unknown_known_weights=torch.tensor([1.0, -1.0]),
            free_occupied_weights=valid,
        )


def test_prediction_diagnostics_use_geometric_matched_masks() -> None:
    torch.manual_seed(13)
    model = _model()
    with torch.no_grad():
        torch.nn.init.normal_(model.predictor.net[-1].weight, std=0.05)
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    wrong_action = torch.roll(action, shifts=1, dims=1)
    delta = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.1, 0.1]])
    wrong_delta = torch.tensor([[-0.4, 0.0, 0.0], [0.0, 0.4, -0.2]])
    prediction_mask = torch.ones(2, 8, 8, dtype=torch.bool)
    prediction_mask[:, 0] = False

    output = model(
        current,
        nxt,
        action,
        delta,
        commanded_delta_pose_current=delta,
        next_prediction_mask=prediction_mask,
        diagnostic_wrong_action=wrong_action,
        diagnostic_wrong_action_delta_pose_current=wrong_delta,
        diagnostic_wrong_commanded_delta_pose_current=wrong_delta,
    )

    assert output["prediction_valid_mask"].dtype == torch.bool
    assert output["wrong_delta_matched_mask"].dtype == torch.bool
    assert torch.all(
        ~output["wrong_delta_matched_mask"] | output["prediction_valid_mask"]
    )
    assert output["wrong_delta_valid_cells"] < output["prediction_valid_cells"]
    assert output["wrong_action_prediction_sensitivity"] > 0
    assert output["wrong_delta_prediction_sensitivity"] > 0
    assert output["action_contrast_loss"] >= 0
    assert output["wrong_action_contrast_loss"] >= 0
    assert output["zero_action_contrast_loss"] >= 0
    assert torch.all(
        ~output["zero_action_matched_mask"] | output["prediction_valid_mask"]
    )
    for name in (
        "prediction_to_persistence_ratio",
        "wrong_action_advantage_over_target_change",
        "wrong_delta_advantage_over_target_change",
    ):
        assert torch.isfinite(output[name])


def test_promoted_prediction_does_not_consume_realized_future_odometry() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.predictor.net[-1].weight, std=0.05)
    current = torch.randn(2, 3, 28, 28)
    nxt = torch.randn(2, 3, 28, 28)
    action = torch.eye(4)[:2]
    commanded_delta = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.1]])
    first = model(
        current,
        nxt,
        action,
        torch.zeros(2, 3),
        commanded_delta_pose_current=commanded_delta,
    )
    second = model(
        current,
        nxt,
        action,
        torch.tensor([[0.4, 0.0, 0.2], [-0.3, 0.2, -0.4]]),
        commanded_delta_pose_current=commanded_delta,
    )

    assert torch.allclose(first["predicted_next_bev"], second["predicted_next_bev"])
    assert torch.allclose(
        first["commanded_warped_current_bev"],
        second["commanded_warped_current_bev"],
    )
    assert not torch.allclose(
        first["realized_warped_current_bev"],
        second["realized_warped_current_bev"],
    )


def test_ema_target_updates_and_stays_in_eval_mode() -> None:
    model = _model()
    before = model.target_encoder.patch_embed.weight.detach().clone()
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(2.0)
    model.train()
    model.update_target_encoder()

    assert not model.target_encoder.training
    assert torch.allclose(model.target_encoder.patch_embed.weight, before + 1.0)
