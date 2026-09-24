from __future__ import annotations

import hashlib

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV10,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift import (
    CELL_VOLUME_VALID_MASK_SHA256_V10,
    ELEVATED_ONLY_VALID_CELL_COUNT_V11,
    ELEVATED_SUPPORT_INDICES_V11,
    ELEVATED_VALID_CELL_COUNT_V11,
    ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11,
    FLOOR_SUPPORT_INDICES_V11,
    FLOOR_VALID_CELL_COUNT_V11,
    FLOOR_VALID_MASK_SHA256_V11,
    FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
    GeometryAnchoredSweptProgressSurvivalJointJepaV11,
    HeightRoleFactorizedEvidenceLiftSamplingV11,
    occupied_priority_log_probabilities_v11,
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
def v10_v11(
    n320_encoder_state: dict[str, torch.Tensor],
) -> tuple[
    GeometryAnchoredSweptProgressSurvivalJointJepaV10,
    GeometryAnchoredSweptProgressSurvivalJointJepaV11,
]:
    return (
        GeometryAnchoredSweptProgressSurvivalJointJepaV10(
            n320_encoder_state, _sweep_masks()
        ).eval(),
        GeometryAnchoredSweptProgressSurvivalJointJepaV11(
            n320_encoder_state, _sweep_masks()
        ).eval(),
    )


def _hash_mask(mask: torch.Tensor) -> str:
    return hashlib.sha256(
        bytes(mask.to(torch.uint8).reshape(-1).tolist())
    ).hexdigest()


def _new_attention(lift: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in lift.named_parameters()
        if name in HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11
    }


def test_v11_preserves_v10_common_state_and_fresh_parameter_contract(
    n320_encoder_state: dict[str, torch.Tensor],
    v10_v11: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
        GeometryAnchoredSweptProgressSurvivalJointJepaV11,
    ],
) -> None:
    v10, model = v10_v11
    v10_parameters = dict(v10.named_parameters())
    v11_parameters = dict(model.named_parameters())
    for name in v10_parameters.keys() & v11_parameters.keys():
        assert torch.equal(v10_parameters[name], v11_parameters[name]), name
    for name in dict(v10.named_buffers()).keys() & dict(model.named_buffers()).keys():
        assert torch.equal(dict(v10.named_buffers())[name], dict(model.named_buffers())[name]), name

    online = _new_attention(model.bev_lift)
    target = _new_attention(model.target_bev_lift)
    semantic = dict(model.semantic_head.named_parameters())
    assert tuple(online) == HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11
    assert tuple(target) == HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11
    assert tuple(semantic) == HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11
    assert (len(online), sum(value.numel() for value in online.values())) == (
        HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
    )
    assert (len(semantic), sum(value.numel() for value in semantic.values())) == (
        HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
        HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
    )
    assert all(torch.equal(value, target[name]) for name, value in online.items())
    assert all(value.requires_grad for value in online.values())
    assert not any(value.requires_grad for value in target.values())
    assert not hasattr(model, "target_semantic_head")

    torch.random.default_generator.manual_seed(8173)
    caller = torch.random.get_rng_state().clone()
    GeometryAnchoredSweptProgressSurvivalJointJepaV11(
        n320_encoder_state, _sweep_masks()
    )
    assert torch.equal(torch.random.get_rng_state(), caller)


def test_exact_role_masks_counts_hashes_sampling_and_attention(
    v10_v11: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
        GeometryAnchoredSweptProgressSurvivalJointJepaV11,
    ],
) -> None:
    _v10, model = v10_v11
    lift = model.bev_lift
    assert tuple(torch.nonzero(lift.floor_support_role_mask).flatten().tolist()) == (
        FLOOR_SUPPORT_INDICES_V11
    )
    assert tuple(
        torch.nonzero(lift.elevated_support_role_mask).flatten().tolist()
    ) == ELEVATED_SUPPORT_INDICES_V11
    assert not bool(
        (lift.floor_support_role_mask & lift.elevated_support_role_mask).any()
    )
    assert bool(
        (lift.floor_support_role_mask | lift.elevated_support_role_mask).all()
    )
    floor_valid = lift.floor_cell_valid_mask
    elevated_valid = lift.elevated_cell_valid_mask
    assert int(floor_valid.sum()) == FLOOR_VALID_CELL_COUNT_V11
    assert int(elevated_valid.sum()) == ELEVATED_VALID_CELL_COUNT_V11
    assert _hash_mask(floor_valid) == FLOOR_VALID_MASK_SHA256_V11
    assert _hash_mask(elevated_valid) == CELL_VOLUME_VALID_MASK_SHA256_V10
    assert torch.equal(elevated_valid, lift.cell_valid_mask)
    assert int((elevated_valid & ~floor_valid).sum()) == (
        ELEVATED_ONLY_VALID_CELL_COUNT_V11
    )
    near = lift.bev_ground_xyz_m[..., :2].square().sum(-1) <= 4.0
    assert int(near.sum()) == 1_016
    assert int((near & floor_valid).sum()) == (
        FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11
    )
    assert int((near & elevated_valid).sum()) == (
        ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11
    )

    tokens = torch.randn((1, 256, 192), generator=torch.Generator().manual_seed(31))
    with torch.no_grad():
        state = lift.forward_with_sampling(tokens)
    assert isinstance(state, HeightRoleFactorizedEvidenceLiftSamplingV11)
    assert state.latent.shape == (1, 64, 64, 64)
    assert state.floor_masked_mean.shape == (1, 64, 64, 64)
    assert state.elevated_masked_mean.shape == (1, 64, 64, 64)
    assert state.floor_attention_weights.shape == (1, 64, 64, 2, 25)
    assert state.elevated_attention_weights.shape == (1, 64, 64, 2, 25)
    for weights, valid_support, valid_cell in (
        (
            state.floor_attention_weights,
            state.floor_support_valid_mask,
            state.floor_cell_valid_mask,
        ),
        (
            state.elevated_attention_weights,
            state.elevated_support_valid_mask,
            state.elevated_cell_valid_mask,
        ),
    ):
        invalid = (~valid_support)[..., None, :].expand_as(weights)
        assert torch.count_nonzero(weights.masked_select(invalid)) == 0
        sums = weights.sum(-1)
        torch.testing.assert_close(
            sums[valid_cell[..., None].expand_as(sums)],
            torch.ones_like(sums[valid_cell[..., None].expand_as(sums)]),
            rtol=0.0,
            atol=1e-6,
        )
    channel_last = state.latent.permute(0, 2, 3, 1)
    expected_null = lift.null_evidence[None].expand(
        int((~state.cell_valid_mask).sum()), -1
    )
    assert torch.equal(channel_last[~state.cell_valid_mask], expected_null)


def test_role_attention_and_semantic_axes_are_structurally_separate(
    v10_v11: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
        GeometryAnchoredSweptProgressSurvivalJointJepaV11,
    ],
) -> None:
    _v10, model = v10_v11
    lift = model.bev_lift
    samples = torch.randn(
        (1, 1, 1, 25, 64), generator=torch.Generator().manual_seed(43)
    )
    floor_valid = lift.floor_support_role_mask[None, None, None]
    mean = (samples * floor_valid[..., None]).sum(-2) / floor_valid.sum()
    first, weights = lift._attend_role(
        mean,
        samples,
        floor_valid,
        floor_valid.any(-1),
        query_projection=lift.floor_query_projection,
        key_projection=lift.floor_key_projection,
        value_projection=lift.floor_value_projection,
        output_projection=lift.floor_output_projection,
    )
    changed = samples.clone()
    changed[..., lift.elevated_support_role_mask, :] += 100.0
    second, changed_weights = lift._attend_role(
        mean,
        changed,
        floor_valid,
        floor_valid.any(-1),
        query_projection=lift.floor_query_projection,
        key_projection=lift.floor_key_projection,
        value_projection=lift.floor_value_projection,
        output_projection=lift.floor_output_projection,
    )
    assert torch.equal(first, second)
    assert torch.equal(weights, changed_weights)

    latent = torch.randn(
        (1, 64, 64, 64), generator=torch.Generator().manual_seed(47)
    )
    free, occupied = model.semantic_head.evidence_logits(latent)
    elevated_changed = latent.clone()
    elevated_changed[:, 32:] += 2.0
    free_elevated_changed, occupied_elevated_changed = (
        model.semantic_head.evidence_logits(elevated_changed)
    )
    assert torch.equal(free, free_elevated_changed)
    assert not torch.equal(occupied, occupied_elevated_changed)
    floor_changed = latent.clone()
    floor_changed[:, :32] += 2.0
    free_floor_changed, occupied_floor_changed = (
        model.semantic_head.evidence_logits(floor_changed)
    )
    assert not torch.equal(free, free_floor_changed)
    assert torch.equal(occupied, occupied_floor_changed)


def test_occupied_priority_algebra_and_exact_validity_masks(
    v10_v11: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV10,
        GeometryAnchoredSweptProgressSurvivalJointJepaV11,
    ],
) -> None:
    _v10, model = v10_v11
    free = torch.tensor([[[-10.0, 10.0, 10.0, -10.0]]])
    occupied = torch.tensor([[[-10.0, -10.0, 10.0, 10.0]]])
    log_probs = occupied_priority_log_probabilities_v11(free, occupied)
    assert log_probs.argmax(1).tolist() == [[[0, 1, 2, 2]]]
    torch.testing.assert_close(
        torch.logsumexp(log_probs, dim=1),
        torch.zeros_like(free),
        rtol=0.0,
        atol=1e-6,
    )

    latent = torch.randn(
        (1, 64, 64, 64), generator=torch.Generator().manual_seed(53)
    )
    with torch.no_grad():
        logits = model.semantic_logits_from_latent(latent)
        _raw_free, raw_occupied = model.semantic_head.evidence_logits(latent)
    invalid = ~model.bev_lift.cell_valid_mask
    expected = torch.tensor((0.0, -20.0, -20.0))[None, :, None].expand(
        1, -1, int(invalid.sum())
    )
    assert torch.equal(logits[:, :, invalid], expected)
    elevated_only = (
        model.bev_lift.elevated_cell_valid_mask
        & ~model.bev_lift.floor_cell_valid_mask
    )
    expected_elevated_only = occupied_priority_log_probabilities_v11(
        torch.full_like(raw_occupied[:, None, elevated_only], -20.0),
        raw_occupied[:, None, elevated_only],
    ).squeeze(2)
    assert torch.equal(logits[:, :, elevated_only], expected_elevated_only)
    torch.testing.assert_close(
        torch.logsumexp(logits[:, :, ~invalid], dim=1),
        torch.zeros_like(logits[:, 0, ~invalid]),
        rtol=0.0,
        atol=1e-6,
    )


def test_new_online_tensors_train_by_second_step_target_stays_frozen_and_predictor_uses_latent(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = GeometryAnchoredSweptProgressSurvivalJointJepaV11(
        n320_encoder_state, _sweep_masks()
    ).train()
    online_attention = _new_attention(model.bev_lift)
    target_attention = _new_attention(model.target_bev_lift)
    semantic = dict(model.semantic_head.named_parameters())
    optimizer = torch.optim.SGD(
        [*online_attention.values(), *semantic.values()], lr=0.05
    )
    tokens = torch.randn(
        (1, 256, 192), generator=torch.Generator().manual_seed(61)
    )
    latent_coefficients = torch.randn(
        (1, 64, 64, 64), generator=torch.Generator().manual_seed(67)
    )
    semantic_coefficients = torch.randn(
        (1, 3, 64, 64), generator=torch.Generator().manual_seed(71)
    )
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        latent = model.bev_lift(tokens)
        logits = model.semantic_logits_from_latent(latent)
        loss = (latent * latent_coefficients).mean() + (
            logits * semantic_coefficients
        ).mean()
        loss.backward()
        if step == 2:
            for name, parameter in {**online_attention, **semantic}.items():
                assert parameter.grad is not None, name
                assert bool(torch.isfinite(parameter.grad).all()), name
                assert torch.count_nonzero(parameter.grad) > 0, name
        assert all(value.grad is None for value in target_attention.values())
        optimizer.step()

    with torch.no_grad():
        latent = model.bev_lift(tokens)
        prediction = model.predict_all_actions_with_survival(latent)
    assert prediction.predicted_latents.shape == (1, 9, 64, 64, 64)
    assert prediction.survival_logits.shape == (1, 9, 16)
    before = {
        name: value.detach().clone() for name, value in target_attention.items()
    }
    model.update_target_ema_after_optimizer_step()
    momentum = model.config.target_ema_momentum
    for name, target in target_attention.items():
        expected_target = before[name].clone().mul_(momentum).add_(
            online_attention[name], alpha=1.0 - momentum
        )
        torch.testing.assert_close(target, expected_target, rtol=0.0, atol=0.0)
        assert target.grad is None
    assert int(model.ema_update_count.item()) == 1
