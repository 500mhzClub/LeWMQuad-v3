from __future__ import annotations

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v1 import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV1,
    SweptProgressSurvivalHeadV1,
    SweptProgressSurvivalPredictionV1,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


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
        return {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def model(
    n320_encoder_state: dict[str, torch.Tensor],
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV1:
    value = GeometryAnchoredSweptProgressSurvivalJointJepaV1(
        n320_encoder_state, _sweep_masks()
    )
    value.eval()
    return value


def test_head_is_shared_under_predictor_and_masks_are_fixed(
    model: GeometryAnchoredSweptProgressSurvivalJointJepaV1,
) -> None:
    parameter_names = set(dict(model.named_parameters()))
    assert "predictor.swept_progress_head.output.weight" in parameter_names
    assert "predictor.swept_progress_head.output.bias" in parameter_names
    assert not any(
        name.startswith("swept_progress_head.") for name in parameter_names
    )
    buffers = dict(model.named_buffers())
    masks = buffers["predictor.swept_progress_head.sweep_masks"]
    assert masks.shape == (9, 16, 64, 64)
    assert masks.dtype == torch.bool
    assert masks.requires_grad is False
    assert bool(masks.flatten(start_dim=2).any(dim=2).all())


def test_all_action_survival_api_shapes_and_backpropagates_through_joint_graph(
    model: GeometryAnchoredSweptProgressSurvivalJointJepaV1,
) -> None:
    model.zero_grad(set_to_none=True)
    rgb = torch.rand((1, 3, 112, 112), dtype=torch.float32)
    current = model.encode_online(rgb)
    result = model.predict_all_actions_with_survival(current)
    assert isinstance(result, SweptProgressSurvivalPredictionV1)
    assert result.predicted_latents.shape == (1, 9, 64, 64, 64)
    assert result.survival_logits.shape == (1, 9, 16)
    assert bool(torch.isfinite(result.survival_logits).all())

    result.survival_logits.sum().backward()
    parameters = (
        model.encoder.patch_embed.weight,
        model.bev_lift.token_projection.weight,
        model.predictor.residual_head.weight,
        model.predictor.swept_progress_head.output.weight,
        model.predictor.swept_progress_head.output.bias,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0.0


def test_pooling_is_mask_normalized_and_strictly_local() -> None:
    masks = _sweep_masks()
    masks[0, 0].zero_()
    masks[0, 0, 10, 10] = True
    masks[0, 1].zero_()
    masks[0, 1, 10, 10:12] = True
    head = SweptProgressSurvivalHeadV1(masks)
    latent = torch.zeros((1, 9, 64, 64, 64), dtype=torch.float32)
    channel = int(head.output.weight.detach().abs().argmax())
    latent[0, 0, channel, 10, 10:12] = 3.0
    logits = head(latent)
    torch.testing.assert_close(logits[0, 0, 0], logits[0, 0, 1])

    changed_outside = latent.clone()
    changed_outside[0, 0, channel, 20, 20] = 1_000.0
    assert torch.equal(head(changed_outside)[0, 0, :2], logits[0, 0, :2])

    changed_inside = latent.clone()
    changed_inside[0, 0, channel, 10, 10] += 1.0
    assert not torch.equal(head(changed_inside)[0, 0, 0], logits[0, 0, 0])


def test_invalid_masks_and_latents_fail_closed() -> None:
    masks = _sweep_masks()
    with pytest.raises(ValueError, match="shape"):
        SweptProgressSurvivalHeadV1(masks[:, :-1])
    with pytest.raises(TypeError, match="bool"):
        SweptProgressSurvivalHeadV1(masks.float())
    empty = masks.clone()
    empty[2, 4].zero_()
    with pytest.raises(ValueError, match="nonempty"):
        SweptProgressSurvivalHeadV1(empty)

    head = SweptProgressSurvivalHeadV1(masks)
    valid = torch.zeros((1, 9, 64, 64, 64), dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        head(valid[:, :-1])
    with pytest.raises(ValueError, match="at least one"):
        head(valid[:0])
    with pytest.raises(TypeError, match="float32"):
        head(valid.double())
    nonfinite = valid.clone()
    nonfinite[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        head(nonfinite)
