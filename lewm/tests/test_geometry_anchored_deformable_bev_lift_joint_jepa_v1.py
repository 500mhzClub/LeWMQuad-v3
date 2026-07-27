from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    GeometryAnchoredDeformableBevLiftJointJepaV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredDeformableBevLiftV1,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(9917)
        encoder = VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
        assert sum(parameter.numel() for parameter in encoder.parameters()) == 2_747_520
        return {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def model(
    n320_encoder_state: dict[str, torch.Tensor],
) -> GeometryAnchoredDeformableBevLiftJointJepaV1:
    value = GeometryAnchoredDeformableBevLiftJointJepaV1(n320_encoder_state)
    value.eval()
    return value


def _fresh_lift() -> GeometryAnchoredDeformableBevLiftV1:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(20260712)
        return GeometryAnchoredDeformableBevLiftV1(
            GeometryAnchoredDeformableBevLiftJointJepaV1Config()
        )
    finally:
        torch.random.set_rng_state(caller_rng)


def test_fixed_camera_anchor_and_local_sampling_shapes_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lift = _fresh_lift().eval()
    assert lift.anchor_in_frustum.shape == (64, 64)
    assert lift.anchor_in_frustum.dtype == torch.bool
    assert bool(lift.anchor_in_frustum.any())
    assert bool((~lift.anchor_in_frustum).any())
    assert lift.anchor_grid_xy.shape == (64, 64, 2)
    assert torch.equal(
        lift.camera_origin_xyz_m, torch.tensor((0.326, 0.0, 0.043))
    )
    assert torch.equal(
        lift.camera_basis_forward_right_up,
        torch.tensor(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))),
    )
    assert lift.ground_z_m.item() == pytest.approx(-0.333)
    assert lift.horizontal_fov_degrees.item() == pytest.approx(78.323)
    assert lift.vertical_fov_degrees.item() == pytest.approx(62.8370386364)
    assert lift.bev_ground_xyz_m[0, 0, 0].item() == pytest.approx(-0.95)
    assert lift.bev_ground_xyz_m[-1, -1, 0].item() == pytest.approx(5.35)
    assert lift.bev_ground_xyz_m[0, 0, 1].item() == pytest.approx(-3.15)
    assert lift.bev_ground_xyz_m[-1, -1, 1].item() == pytest.approx(3.15)
    assert torch.equal(
        lift.bev_ground_xyz_m[..., 2],
        torch.full((64, 64), -0.333),
    )

    assert lift.raw_offsets.shape == (64, 64, 4, 2)
    assert lift.weight_logits.shape == (64, 64, 4)
    assert not hasattr(lift, "sampling_head")
    grid_sample_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_grid_sample = torch.nn.functional.grid_sample

    def counted_grid_sample(
        input_value: torch.Tensor,
        grid: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        grid_sample_calls.append((tuple(input_value.shape), tuple(grid.shape)))
        return original_grid_sample(input_value, grid, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "grid_sample", counted_grid_sample)
    tokens = torch.randn(1, 256, 192)
    result = lift.forward_with_sampling(tokens)
    assert grid_sample_calls == [((1, 64, 16, 16), (1, 64, 256, 2))]
    assert result.latent.shape == (1, 64, 64, 64)
    assert result.anchor_in_frustum.shape == (1, 64, 64)
    assert result.sample_valid_mask.shape == (1, 64, 64, 4)
    assert result.cell_valid_mask.shape == (1, 64, 64)
    assert result.sample_grid_xy.shape == (1, 64, 64, 4, 2)
    assert result.offsets_token_cells.shape == (1, 64, 64, 4, 2)
    assert result.sample_weights.shape == (1, 64, 64, 4)
    assert float(result.offsets_token_cells.detach().abs().max()) <= 2.0
    assert torch.allclose(
        result.offsets_token_cells.detach().abs(),
        torch.full_like(result.offsets_token_cells, 0.5),
        atol=1e-7,
        rtol=0.0,
    )
    assert bool(torch.isfinite(result.latent).all())
    assert bool(torch.isfinite(result.sample_weights).all())

    valid_weight_sums = result.sample_weights.sum(dim=-1)
    assert torch.allclose(
        valid_weight_sums[result.cell_valid_mask],
        torch.ones_like(valid_weight_sums[result.cell_valid_mask]),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.equal(
        valid_weight_sums[~result.cell_valid_mask],
        torch.zeros_like(valid_weight_sums[~result.cell_valid_mask]),
    )
    invalid_samples = ~result.sample_valid_mask
    assert torch.equal(
        result.sample_grid_xy[invalid_samples],
        torch.full_like(result.sample_grid_xy[invalid_samples], 2.0),
    )
    assert not bool(result.sample_valid_mask[:, ~lift.anchor_in_frustum].any())


def test_out_of_frustum_cells_are_null_and_semantically_unknown(
    model: GeometryAnchoredDeformableBevLiftJointJepaV1,
) -> None:
    invalid = ~model.bev_lift.anchor_in_frustum
    first = torch.randn(1, 256, 192)
    second = torch.randn(1, 256, 192) * 10.0 + 7.0
    first_result = model.bev_lift.forward_with_sampling(first)
    second_result = model.bev_lift.forward_with_sampling(second)
    null = model.bev_lift.null_evidence[None, :, None]
    assert torch.equal(
        first_result.latent[:, :, invalid], null.expand(1, 64, int(invalid.sum()))
    )
    assert torch.equal(
        second_result.latent[:, :, invalid], null.expand(1, 64, int(invalid.sum()))
    )
    assert torch.equal(
        first_result.latent[:, :, invalid], second_result.latent[:, :, invalid]
    )

    latent_a = torch.randn(2, 64, 64, 64)
    latent_b = torch.randn(2, 64, 64, 64) * 4.0
    logits_a = model.semantic_logits_from_latent(latent_a)
    logits_b = model.semantic_logits_from_latent(latent_b)
    expected = torch.tensor((0.0, -20.0, -20.0))[None, :, None]
    expected = expected.expand(2, 3, int(invalid.sum()))
    assert torch.equal(logits_a[:, :, invalid], expected)
    assert torch.equal(logits_b[:, :, invalid], expected)
    assert torch.equal(
        logits_a[:, :, invalid].argmax(dim=1),
        torch.zeros((2, int(invalid.sum())), dtype=torch.long),
    )


def test_lift_and_predictor_are_strictly_local_without_global_bypass(
    model: GeometryAnchoredDeformableBevLiftJointJepaV1,
) -> None:
    forbidden = (
        nn.MultiheadAttention,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
    )
    for component in (model.bev_lift, model.predictor):
        assert not any(isinstance(module, forbidden) for module in component.modules())
        for module in component.modules():
            if isinstance(module, nn.Conv2d):
                assert module.kernel_size in ((1, 1), (3, 3))
                assert module.stride == (1, 1)
        assert not any(
            parameter.ndim >= 2 and tuple(parameter.shape[-2:]) == (64, 64)
            for parameter in component.parameters()
        )
    assert len(model.bev_lift.refinement_blocks) == 2
    assert len(model.predictor.residual_blocks) == 2
    assert not hasattr(model.predictor, "coordinate_features")
    assert not hasattr(model.predictor, "pose")


def test_constructor_preserves_caller_rng_and_fresh_heads_are_deterministic(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    torch.random.default_generator.manual_seed(421)
    caller_rng = torch.random.get_rng_state().clone()
    first = GeometryAnchoredDeformableBevLiftJointJepaV1(n320_encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    second = GeometryAnchoredDeformableBevLiftJointJepaV1(n320_encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    for first_module, second_module in (
        (first.bev_lift, second.bev_lift),
        (first.semantic_head, second.semantic_head),
        (first.predictor, second.predictor),
    ):
        first_state = first_module.state_dict()
        second_state = second_module.state_dict()
        assert first_state.keys() == second_state.keys()
        assert all(
            torch.equal(first_state[name], second_state[name]) for name in first_state
        )


def test_clean_online_target_semantic_and_predictor_apis(
    model: GeometryAnchoredDeformableBevLiftJointJepaV1,
) -> None:
    rgb = torch.rand(1, 3, 112, 112)
    # The update-zero observer compares both branches without autograd, which
    # also keeps the ViT attention kernel path identical across the hard sync.
    with torch.no_grad():
        online = model.encode_online(rgb)
    target = model.encode_target(rgb)
    assert online.shape == (1, 64, 64, 64)
    assert target.shape == online.shape
    assert not target.requires_grad
    assert torch.equal(online, target)
    state = model.online_state(rgb)
    assert state.shape == (1, 3, 64, 64)
    invalid = ~model.bev_lift.anchor_in_frustum
    assert torch.equal(
        state[:, :, invalid].argmax(dim=1),
        torch.zeros((1, int(invalid.sum())), dtype=torch.long),
    )

    action_index = 3
    action = torch.eye(9)[action_index : action_index + 1]
    selected = model.predict(online, action)
    all_actions = model.predict_all_actions(online)
    assert selected.shape == online.shape
    assert all_actions.shape == (1, 9, 64, 64, 64)
    assert torch.equal(selected, all_actions[:, action_index])


def test_semantic_gradient_reaches_tokens_and_deformable_lift() -> None:
    lift = _fresh_lift().train()
    head = nn.Conv2d(64, 3, kernel_size=1)
    tokens = torch.randn(1, 256, 192, requires_grad=True)
    labels = torch.arange(64 * 64, dtype=torch.long).reshape(1, 64, 64) % 3
    logits = head(lift(tokens))
    loss = final_class_macro_nll_per_row(logits, labels).mean()
    loss.backward()
    assert tokens.grad is not None and bool(torch.isfinite(tokens.grad).all())
    assert float(tokens.grad.abs().sum()) > 0.0
    for parameter in (
        lift.token_projection.weight,
        lift.raw_offsets,
        lift.weight_logits,
        lift.refinement_blocks[0].conv1.weight,
        head.weight,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0.0


def test_target_is_encoder_and_lift_only_frozen_and_ema_updated(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredDeformableBevLiftJointJepaV1(n320_encoder_state)
    assert int(model.target_hard_sync_count) == 1
    assert int(model.ema_update_count) == 0
    assert tuple(model.iter_online_target_modules()) == (model.encoder, model.bev_lift)
    assert tuple(model.iter_target_modules()) == (
        model.target_encoder,
        model.target_bev_lift,
    )
    assert all(not parameter.requires_grad for module in model.target_modules() for parameter in module.parameters())
    model.train()
    assert not model.target_encoder.training
    assert not model.target_bev_lift.training
    assert not hasattr(model, "target_semantic_head")
    assert not hasattr(model, "target_predictor")

    online_parameter = next(model.encoder.parameters())
    target_parameter = next(model.target_encoder.parameters())
    old_target = target_parameter.detach().clone()
    with torch.no_grad():
        online_parameter.add_(1.0)
    expected = old_target * 0.996 + online_parameter.detach() * 0.004
    model.update_target_ema_after_optimizer_step()
    assert int(model.ema_update_count) == 1
    assert torch.allclose(target_parameter, expected, atol=1e-7, rtol=0.0)
    assert all(not parameter.requires_grad for module in model.target_modules() for parameter in module.parameters())

    model.hard_sync_target_from_online()
    assert int(model.target_hard_sync_count) == 2
    assert int(model.ema_update_count) == 0
    assert torch.equal(target_parameter, online_parameter)


def test_final_class_macro_nll_is_equal_over_present_classes_per_row() -> None:
    logits = torch.tensor(
        [
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[-1.0, -1.0], [-1.0, -1.0]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ]
    )
    labels = torch.tensor([[[0, 0], [0, 1]], [[2, 2], [2, 2]]])
    observed = final_class_macro_nll_per_row(logits, labels)
    per_cell = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected_first = 0.5 * (
        per_cell[0][labels[0] == 0].mean() + per_cell[0][labels[0] == 1].mean()
    )
    assert torch.allclose(observed[0], expected_first)
    assert observed[1].item() == pytest.approx(math.log(3.0))


def test_latent_energy_supports_nchw_and_bnchw_and_detaches_target() -> None:
    predicted = torch.randn(2, 64, 3, 4, requires_grad=True)
    target = torch.randn(2, 64, 3, 4, requires_grad=True)
    energy = latent_energy_per_row(predicted, target)
    assert energy.shape == (2,)
    assert bool((energy > 0.0).all())
    energy.sum().backward()
    assert predicted.grad is not None
    assert target.grad is None
    assert torch.equal(latent_energy_per_row(target.detach(), target.detach()), torch.zeros(2))

    predicted_all = torch.randn(2, 9, 64, 2, 3)
    target_all = torch.randn(2, 9, 64, 2, 3)
    energy_all = latent_energy_per_row(predicted_all, target_all)
    assert energy_all.shape == (2, 9)
    assert bool(torch.isfinite(energy_all).all())
