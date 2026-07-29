from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift import (
    HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11,
    GeometryAnchoredSweptProgressSurvivalJointJepaV11,
    HeightRoleOccupiedPrioritySemanticDecoderV11,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV12,
    HeightRoleNeutralDisjointTernarySemanticDecoderV12,
    neutral_disjoint_ternary_log_probabilities_v12,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller = torch.random.get_rng_state().clone()
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
        return {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
    finally:
        torch.random.set_rng_state(caller)


@pytest.fixture(scope="module")
def matched_v11_v12(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[
    GeometryAnchoredSweptProgressSurvivalJointJepaV11,
    GeometryAnchoredSweptProgressSurvivalJointJepaV12,
]:
    original = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(8173)
        caller_v11 = torch.random.get_rng_state().clone()
        v11 = GeometryAnchoredSweptProgressSurvivalJointJepaV11(
            n320_encoder_state, _sweep_masks()
        ).eval()
        assert torch.equal(torch.random.get_rng_state(), caller_v11)

        torch.random.default_generator.manual_seed(9127)
        caller_v12 = torch.random.get_rng_state().clone()
        v12 = GeometryAnchoredSweptProgressSurvivalJointJepaV12(
            n320_encoder_state, _sweep_masks()
        ).eval()
        assert torch.equal(torch.random.get_rng_state(), caller_v12)
    finally:
        torch.random.set_rng_state(original)
    return v11, v12


def test_wrapper_reuses_axes_and_v12_state_is_bitwise_v11(
    matched_v11_v12: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV11,
        GeometryAnchoredSweptProgressSurvivalJointJepaV12,
    ],
) -> None:
    v11, v12 = matched_v11_v12
    state_v11 = v11.state_dict()
    state_v12 = v12.state_dict()
    assert state_v12.keys() == state_v11.keys()
    assert all(torch.equal(state_v12[name], value) for name, value in state_v11.items())
    parameters_v11 = dict(v11.named_parameters())
    parameters_v12 = dict(v12.named_parameters())
    assert parameters_v12.keys() == parameters_v11.keys()
    assert sum(value.numel() for value in parameters_v12.values()) == sum(
        value.numel() for value in parameters_v11.values()
    )
    assert tuple(dict(v12.semantic_head.named_parameters())) == (
        HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11
    )
    assert isinstance(
        v12.semantic_head, HeightRoleNeutralDisjointTernarySemanticDecoderV12
    )
    assert all(
        not parameter.requires_grad
        for parameter in v12.target_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in v12.target_bev_lift.parameters()
    )

    tokens = torch.randn(
        (1, 256, 192), generator=torch.Generator().manual_seed(41)
    )
    shared = torch.randn(
        (1, 64, 64, 64), generator=torch.Generator().manual_seed(43)
    )
    with torch.no_grad():
        lift_v11 = v11.bev_lift.forward_with_sampling(tokens)
        lift_v12 = v12.bev_lift.forward_with_sampling(tokens)
        prediction_v11 = v11.predict_all_actions_with_survival(shared)
        prediction_v12 = v12.predict_all_actions_with_survival(shared)
    assert all(
        torch.equal(left, right)
        for left, right in zip(lift_v11, lift_v12, strict=True)
    )
    assert torch.equal(
        prediction_v11.predicted_latents, prediction_v12.predicted_latents
    )
    assert torch.equal(
        prediction_v11.survival_logits, prediction_v12.survival_logits
    )

    source = HeightRoleOccupiedPrioritySemanticDecoderV11()
    original_parameters = tuple(source.named_parameters())
    caller = torch.random.get_rng_state().clone()
    wrapper = HeightRoleNeutralDisjointTernarySemanticDecoderV12(source)
    assert torch.equal(torch.random.get_rng_state(), caller)
    assert wrapper.free_axis is source.free_axis
    assert wrapper.occupied_axis is source.occupied_axis
    assert all(
        wrapped is original
        for (_, wrapped), (_, original) in zip(
            wrapper.named_parameters(), original_parameters, strict=True
        )
    )


def test_neutral_ternary_algebra_normalization_and_auxiliary_logit_identity() -> None:
    free = torch.tensor([[[-10.0, 10.0, -5.0, 10.0, 9.0]]])
    occupied = torch.tensor([[[-10.0, -5.0, 10.0, 9.0, 10.0]]])
    result = neutral_disjoint_ternary_log_probabilities_v12(free, occupied)
    raw = torch.stack((torch.zeros_like(free), free, occupied), dim=1)
    assert torch.equal(result, F.log_softmax(raw, dim=1))
    assert result.argmax(dim=1).tolist() == [[[0, 1, 2, 1, 2]]]
    torch.testing.assert_close(
        torch.logsumexp(result, dim=1),
        torch.zeros_like(free),
        rtol=0.0,
        atol=1e-6,
    )

    normalized_occupied_logit = result[:, 2] - torch.logsumexp(
        result[:, :2], dim=1
    )
    raw_occupied_logit = occupied - torch.logsumexp(raw[:, :2], dim=1)
    torch.testing.assert_close(
        normalized_occupied_logit,
        raw_occupied_logit,
        rtol=0.0,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="matching shape"):
        neutral_disjoint_ternary_log_probabilities_v12(
            torch.zeros((1, 2, 2)), torch.zeros((1, 2, 3))
        )
    nonfinite = torch.zeros((1, 2, 2))
    nonfinite[0, 0, 0] = math.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        neutral_disjoint_ternary_log_probabilities_v12(
            nonfinite, torch.zeros_like(nonfinite)
        )
    extreme = neutral_disjoint_ternary_log_probabilities_v12(
        torch.tensor([[[10_000.0, -10_000.0]]]),
        torch.tensor([[[-10_000.0, 10_000.0]]]),
    )
    assert bool(torch.isfinite(extreme).all())
    assert extreme.argmax(dim=1).tolist() == [[[1, 2]]]


def test_disjoint_axis_routing_and_exact_semantic_validity_masks(
    matched_v11_v12: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV11,
        GeometryAnchoredSweptProgressSurvivalJointJepaV12,
    ],
) -> None:
    _v11, model = matched_v11_v12
    latent = torch.randn(
        (1, 64, 64, 64), generator=torch.Generator().manual_seed(53)
    )
    with torch.no_grad():
        free, occupied = model.semantic_head.evidence_logits(latent)
        head_result = model.semantic_head(latent)
        result = model.semantic_logits_from_latent(latent)
        elevated_changed = latent.clone()
        elevated_changed[:, 32:] += 2.0
        free_elevated, occupied_elevated = model.semantic_head.evidence_logits(
            elevated_changed
        )
        floor_changed = latent.clone()
        floor_changed[:, :32] += 2.0
        free_floor, occupied_floor = model.semantic_head.evidence_logits(
            floor_changed
        )
    assert torch.equal(
        head_result,
        neutral_disjoint_ternary_log_probabilities_v12(free, occupied),
    )
    assert torch.equal(free, free_elevated)
    assert not torch.equal(occupied, occupied_elevated)
    assert not torch.equal(free, free_floor)
    assert torch.equal(occupied, occupied_floor)

    floor_valid = model.bev_lift.floor_cell_valid_mask[None]
    elevated_valid = model.bev_lift.elevated_cell_valid_mask[None]
    masked_free = torch.where(floor_valid, free, torch.full_like(free, -20.0))
    masked_occupied = torch.where(
        elevated_valid, occupied, torch.full_like(occupied, -20.0)
    )
    expected = neutral_disjoint_ternary_log_probabilities_v12(
        masked_free, masked_occupied
    )
    valid = model.bev_lift.cell_valid_mask[None, None]
    invalid_value = expected.new_tensor((0.0, -20.0, -20.0))[
        None, :, None, None
    ]
    expected = torch.where(valid, expected, invalid_value)
    assert torch.equal(result, expected)

    invalid = ~model.bev_lift.cell_valid_mask
    inherited_invalid = torch.tensor((0.0, -20.0, -20.0))[
        None, :, None
    ].expand(1, -1, int(invalid.sum()))
    assert torch.equal(result[:, :, invalid], inherited_invalid)
    elevated_only = (
        model.bev_lift.elevated_cell_valid_mask
        & ~model.bev_lift.floor_cell_valid_mask
    )
    assert int(elevated_only.sum()) == 38
    assert not bool((result[:, :, elevated_only].argmax(dim=1) == 1).any())
    torch.testing.assert_close(
        torch.logsumexp(result[:, :, ~invalid], dim=1),
        torch.zeros_like(result[:, 0, ~invalid]),
        rtol=0.0,
        atol=1e-6,
    )


def test_neutral_axes_train_by_step_two_and_predictor_ema_remain_inherited(
    matched_v11_v12: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV11,
        GeometryAnchoredSweptProgressSurvivalJointJepaV12,
    ],
) -> None:
    _v11, model = matched_v11_v12
    decoder = copy.deepcopy(model.semantic_head).train()
    semantic = dict(decoder.named_parameters())
    optimizer = torch.optim.SGD(semantic.values(), lr=0.05)
    latent = torch.randn(
        (1, 64, 64, 64), generator=torch.Generator().manual_seed(61)
    )
    labels = torch.arange(64 * 64).reshape(1, 64, 64).remainder(3)
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        F.nll_loss(decoder(latent), labels).backward()
        if step == 2:
            for name, parameter in semantic.items():
                assert parameter.grad is not None, name
                assert bool(torch.isfinite(parameter.grad).all()), name
                assert torch.count_nonzero(parameter.grad) > 0, name
        optimizer.step()

    shared = torch.randn(
        (1, 64, 64, 64), generator=torch.Generator().manual_seed(67)
    )
    with torch.no_grad():
        prediction = model.predict_all_actions_with_survival(shared)
    expected = shared[:, None].expand(-1, 9, -1, -1, -1)
    assert torch.equal(prediction.predicted_latents, expected)
    assert prediction.survival_logits.shape == (1, 9, 16)

    ema_model = copy.deepcopy(model).train()
    online = ema_model.bev_lift.floor_query_projection.weight
    target = ema_model.target_bev_lift.floor_query_projection.weight
    target_before = target.detach().clone()
    with torch.no_grad():
        online.add_(0.125)
    ema_model.update_target_ema_after_optimizer_step()
    expected_target = target_before.mul(ema_model.config.target_ema_momentum).add(
        online, alpha=1.0 - ema_model.config.target_ema_momentum
    )
    torch.testing.assert_close(target, expected_target, rtol=0.0, atol=0.0)
    assert target.grad is None
    assert not target.requires_grad
    assert int(ema_model.ema_update_count.item()) == 1
