from __future__ import annotations

import torch
import pytest

from lewm.models.shared_v5_target_observation_head_v1 import (
    CANONICAL_TARGET_COLORS_V1,
    FourColorTargetObservationOutputV1,
    SharedV5TargetObservationHeadConfigV1,
    SharedV5TargetObservationHeadV1,
    SharedV5TargetObservationHeadV1Error,
    initialize_deterministic_mock_weights_v1,
)


def _head() -> SharedV5TargetObservationHeadV1:
    head = SharedV5TargetObservationHeadV1(
        SharedV5TargetObservationHeadConfigV1(
            patch_feature_dim=8,
            bev_feature_dim=6,
            hidden_dim=16,
            color_embedding_dim=4,
            maximum_range_m=10.0,
        )
    )
    initialize_deterministic_mock_weights_v1(head, seed=17)
    head.eval()
    return head


def _features() -> tuple[torch.Tensor, torch.Tensor]:
    patch = torch.linspace(-1.0, 1.0, 2 * 5 * 8).reshape(2, 5, 8)
    bev = torch.linspace(-0.5, 0.5, 2 * 6 * 3 * 4).reshape(2, 6, 3, 4)
    return patch, bev


def test_target_head_is_one_four_colour_batch_with_finite_distributions() -> None:
    head = _head()
    patch, bev = _features()
    patch_version = patch._version
    bev_version = bev._version
    calls = []
    hook = head.register_forward_hook(lambda *_: calls.append(1))
    with torch.no_grad():
        output = head(patch, bev)
    hook.remove()
    assert type(output) is FourColorTargetObservationOutputV1
    assert output.colors == CANONICAL_TARGET_COLORS_V1
    assert output.batch_size == 2
    assert len(calls) == 1
    assert patch._version == patch_version
    assert bev._version == bev_version
    for value in (
        output.presence_probability,
        output.bearing_mean_rad,
        output.bearing_scale_rad,
        output.range_mean_m,
        output.range_scale_m,
        output.uncertainty,
        output.quality,
    ):
        assert tuple(value.shape) == (2, 4)
        assert torch.isfinite(value).all()
    assert ((0.0 <= output.presence_probability) & (output.presence_probability <= 1.0)).all()
    assert ((0.0 <= output.quality) & (output.quality <= 1.0)).all()
    assert (output.range_mean_m > 0.0).all()
    assert (output.bearing_scale_rad > 0.0).all()


def test_target_head_mock_weights_are_deterministic() -> None:
    first = _head()
    second = _head()
    patch, bev = _features()
    with torch.no_grad():
        first_output = first(patch, bev)
        second_output = second(patch, bev)
    assert torch.equal(first_output.presence_logit, second_output.presence_logit)
    assert torch.equal(first_output.range_mean_m, second_output.range_mean_m)
    assert first.architecture_config_sha256 == second.architecture_config_sha256


def test_target_head_rejects_attached_wrong_shape_and_nonfinite_features() -> None:
    head = _head()
    patch, bev = _features()
    with pytest.raises(SharedV5TargetObservationHeadV1Error):
        head(patch.requires_grad_(), bev)
    patch = patch.detach()
    with pytest.raises(SharedV5TargetObservationHeadV1Error):
        head(patch[:, :, :-1], bev)
    bad_bev = bev.clone()
    bad_bev[0, 0, 0, 0] = float("nan")
    with pytest.raises(SharedV5TargetObservationHeadV1Error):
        head(patch, bad_bev)


def test_target_head_owns_no_encoder_or_preprocessor() -> None:
    head = _head()
    assert head.owned_encoder_count == 0
    assert head.owned_rgb_preprocessor_count == 0
    assert all("encoder" not in name.lower() for name, _ in head.named_modules())
    assert all("preprocess" not in name.lower() for name, _ in head.named_modules())
