from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lewm.benchmarks.go2_categorical_radial_factorization import (
    gather_polar_logits_to_cartesian,
)
from lewm.benchmarks.go2_physical_micro_overfit import TRAINING_WEIGHTS
from lewm.benchmarks.go2_physical_spatial_grounding import loss_accumulator_for_batch
from lewm.models.categorical_radial_perception import IMAGE_SIZE, POLAR_SHAPE
from lewm.models.categorical_radial_perception_full_ray import (
    RADIAL_DILATIONS,
    REGISTERED_PARAMETER_COUNT as V2_PARAMETER_COUNT,
)
from lewm.models.categorical_radial_perception_full_ray_hierarchical import (
    CategoricalRadialPerceptionFullRayHierarchical,
    EXECUTION_BINDING_SHA256,
    REGISTERED_CONTEXT_DIM,
    REGISTERED_FACTOR_NAMES,
    REGISTERED_PARAMETER_COUNT,
    REGISTERED_SAME_SHAPE_STATE_ENTRY_COUNT,
    REGISTERED_SHAPE_CHANGED_STATE_KEYS,
    REGISTERED_SHAPE_CHANGES,
    REGISTERED_STATE_ENTRY_COUNT,
    REGISTERED_TOKEN_FEATURE_DIM,
    build_comparable_width24_and_hierarchical_models,
    hierarchical_factors_to_joint_log_probabilities,
)
from scripts.run_go2_categorical_radial_ladder import hierarchical_occupancy_loss


def _weighted_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weights: tuple[float, float],
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(
        logits,
        labels.to(dtype=logits.dtype),
        reduction="none",
    )
    class_weights = logits.new_tensor(weights)[labels]
    applied = class_weights * mask.to(dtype=logits.dtype)
    return (losses * applied).sum() / applied.sum().clamp_min(
        torch.finfo(logits.dtype).tiny
    )


def _direct_hierarchical_binary_loss(
    factors: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    known_labels = (labels != 0).long()
    known_loss = _weighted_binary_loss(
        factors[:, 0],
        known_labels,
        mask,
        TRAINING_WEIGHTS["unknown_known"],
    )
    known_mask = mask & (labels != 0)
    occupied_labels = (labels == 2).long()
    occupied_loss = _weighted_binary_loss(
        factors[:, 1],
        occupied_labels,
        known_mask,
        TRAINING_WEIGHTS["free_occupied"],
    )
    return 0.5 * known_loss + 0.5 * occupied_loss


def test_binding_and_registered_architecture() -> None:
    torch.manual_seed(17)
    model = CategoricalRadialPerceptionFullRayHierarchical()

    assert EXECUTION_BINDING_SHA256 == (
        "bb691c787af0b90f813ced4e5e521f1b15b70b75c836147cd69275c50df6b5d3"
    )
    assert REGISTERED_FACTOR_NAMES == ("known", "occupied_given_known")
    assert model.token_feature_dim == REGISTERED_TOKEN_FEATURE_DIM == 24
    assert model.context_dim == REGISTERED_CONTEXT_DIM == 64
    assert tuple(model.polar_head.weight.shape) == (2, 64, 1, 1)
    assert tuple(model.polar_head.bias.shape) == (2,)
    assert tuple(block.dilation for block in model.radial_context) == RADIAL_DILATIONS
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        REGISTERED_PARAMETER_COUNT
    )
    assert REGISTERED_PARAMETER_COUNT == V2_PARAMETER_COUNT - 65 == 2_887_002
    assert len(model.state_dict()) == REGISTERED_STATE_ENTRY_COUNT == 133


def test_joint_formula_normalizes_and_is_finite_at_extremes() -> None:
    factors = torch.tensor(
        [
            [
                [[-1.0e4, -1.0e4], [0.0, 1.0e4]],
                [[-1.0e4, 1.0e4], [1.0e4, 0.0]],
            ]
        ],
        dtype=torch.float64,
    )

    joint = hierarchical_factors_to_joint_log_probabilities(factors)

    assert joint.shape == (1, 3, 2, 2)
    assert torch.isfinite(joint).all()
    assert torch.allclose(
        torch.logsumexp(joint, dim=1),
        torch.zeros_like(joint[:, 0]),
        atol=1e-14,
        rtol=0.0,
    )
    assert joint.argmax(dim=1).tolist() == [[[0, 0], [0, 1]]]


def test_arbitrary_three_class_logits_round_trip_through_factors() -> None:
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(4, 3, 5, 7, generator=generator, dtype=torch.float64)
    known = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
    occupied_given_known = logits[:, 2] - logits[:, 1]
    factors = torch.stack((known, occupied_given_known), dim=1)

    reconstructed = hierarchical_factors_to_joint_log_probabilities(factors)

    assert torch.allclose(
        reconstructed,
        torch.log_softmax(logits, dim=1),
        atol=2e-15,
        rtol=1e-14,
    )


def test_existing_numpy_evaluator_matches_reconstructed_scores_and_ties() -> None:
    generator = torch.Generator().manual_seed(27)
    logits = torch.randn(2, 3, 6, 7, generator=generator, dtype=torch.float64)
    logits[:, :, 0, 0] = 0.0
    labels = torch.randint(0, 3, (2, 6, 7), generator=generator)
    mask = torch.rand(2, 6, 7, generator=generator) > 0.15
    known = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
    occupied_given_known = logits[:, 2] - logits[:, 1]
    reconstructed = hierarchical_factors_to_joint_log_probabilities(
        torch.stack((known, occupied_given_known), dim=1)
    )

    original = loss_accumulator_for_batch(
        logits.numpy(),
        labels.numpy(),
        mask.numpy(),
        unknown_known_weights=TRAINING_WEIGHTS["unknown_known"],
        free_occupied_weights=TRAINING_WEIGHTS["free_occupied"],
    )
    factored = loss_accumulator_for_batch(
        reconstructed.detach().numpy(),
        labels.numpy(),
        mask.numpy(),
        unknown_known_weights=TRAINING_WEIGHTS["unknown_known"],
        free_occupied_weights=TRAINING_WEIGHTS["free_occupied"],
    )

    assert np.array_equal(
        logits.argmax(dim=1).numpy(),
        reconstructed.argmax(dim=1).numpy(),
    )
    assert reconstructed.argmax(dim=1)[:, 0, 0].tolist() == [0, 0]
    for name in original:
        if isinstance(original[name], int):
            assert factored[name] == original[name], name
        else:
            assert factored[name] == pytest.approx(
                original[name],
                rel=1e-14,
                abs=1e-13,
            ), name


def test_existing_loss_matches_direct_weighted_binary_value_and_gradient() -> None:
    generator = torch.Generator().manual_seed(29)
    raw = torch.randn(2, 2, 64, 64, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 3, (2, 64, 64), generator=generator)
    mask = torch.rand(2, 64, 64, generator=generator) > 0.2
    factors_existing = raw.clone().requires_grad_(True)
    factors_direct = raw.clone().requires_grad_(True)

    existing = hierarchical_occupancy_loss(
        hierarchical_factors_to_joint_log_probabilities(factors_existing),
        labels,
        mask,
    )
    direct = _direct_hierarchical_binary_loss(factors_direct, labels, mask)
    existing_gradient = torch.autograd.grad(existing, factors_existing)[0]
    direct_gradient = torch.autograd.grad(direct, factors_direct)[0]

    assert torch.allclose(existing, direct, atol=2e-15, rtol=1e-14)
    assert torch.allclose(
        existing_gradient,
        direct_gradient,
        atol=2e-15,
        rtol=1e-12,
    )


def test_existing_loss_matches_direct_binary_loss_with_no_known_cells() -> None:
    factors = torch.randn(1, 2, 64, 64, dtype=torch.float64, requires_grad=True)
    labels = torch.zeros(1, 64, 64, dtype=torch.long)
    mask = torch.ones_like(labels, dtype=torch.bool)

    existing = hierarchical_occupancy_loss(
        hierarchical_factors_to_joint_log_probabilities(factors),
        labels,
        mask,
    )
    direct = _direct_hierarchical_binary_loss(factors, labels, mask)

    assert torch.allclose(existing, direct, atol=2e-15, rtol=1e-14)
    assert torch.autograd.grad(existing, factors)[0][:, 1].abs().max() <= 1e-12


@pytest.mark.parametrize(
    ("dtype", "cross_gradient_tolerance"),
    ((torch.float64, 1e-12), (torch.float32, 1e-7)),
)
def test_hierarchical_terms_have_zero_cross_gradients(
    dtype: torch.dtype,
    cross_gradient_tolerance: float,
) -> None:
    generator = torch.Generator().manual_seed(31)
    factors = torch.randn(
        2,
        2,
        4,
        5,
        generator=generator,
        dtype=dtype,
        requires_grad=True,
    )
    labels = torch.randint(0, 3, (2, 4, 5), generator=generator)
    mask = torch.ones_like(labels, dtype=torch.bool)
    joint = hierarchical_factors_to_joint_log_probabilities(factors)

    known_score = torch.logsumexp(joint[:, 1:], dim=1)
    known_logits = torch.stack((joint[:, 0], known_score), dim=1)
    known_loss = F.cross_entropy(known_logits, (labels != 0).long())
    known_gradient = torch.autograd.grad(
        known_loss,
        factors,
        retain_graph=True,
    )[0]
    occupied_loss = F.cross_entropy(
        joint[:, 1:],
        (labels - 1).clamp_min(0),
        reduction="none",
    )
    occupied_loss = occupied_loss[labels != 0].mean()
    occupied_gradient = torch.autograd.grad(occupied_loss, factors)[0]

    assert known_gradient[:, 1].abs().max() <= cross_gradient_tolerance
    assert occupied_gradient[:, 0].abs().max() <= cross_gradient_tolerance
    assert known_gradient[:, 0].abs().sum() > 0
    assert occupied_gradient[:, 1].abs().sum() > 0


def test_comparable_builder_changes_two_shapes_and_copies_131_entries() -> None:
    torch.manual_seed(20260710)
    initial_rng_state = torch.get_rng_state().clone()

    torch.set_rng_state(initial_rng_state)
    raw_hierarchical = CategoricalRadialPerceptionFullRayHierarchical()
    raw_hierarchical_state = {
        name: value.detach().clone()
        for name, value in raw_hierarchical.state_dict().items()
    }
    expected_final_rng_state = torch.get_rng_state().clone()

    width24, hierarchical = build_comparable_width24_and_hierarchical_models(
        initial_rng_state
    )
    width24_state = width24.state_dict()
    hierarchical_state = hierarchical.state_dict()
    changed = {
        name: (tuple(width24_state[name].shape), tuple(hierarchical_state[name].shape))
        for name in width24_state
        if width24_state[name].shape != hierarchical_state[name].shape
    }

    assert changed == REGISTERED_SHAPE_CHANGES
    assert tuple(changed) == REGISTERED_SHAPE_CHANGED_STATE_KEYS
    assert tuple(width24_state) == tuple(hierarchical_state)
    assert torch.equal(torch.get_rng_state(), expected_final_rng_state)
    same_shape_names = [name for name in width24_state if name not in changed]
    assert len(same_shape_names) == REGISTERED_SAME_SHAPE_STATE_ENTRY_COUNT == 131
    for name in width24_state:
        if name in changed:
            assert torch.equal(hierarchical_state[name], raw_hierarchical_state[name])
        else:
            assert width24_state[name].dtype == hierarchical_state[name].dtype
            assert torch.equal(width24_state[name], hierarchical_state[name])


def test_geometry_joint_outputs_and_single_encoder_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(37)
    model = CategoricalRadialPerceptionFullRayHierarchical().eval()
    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    calls = 0
    original = model.encoder.forward_tokens

    def counted_forward_tokens(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(model.encoder, "forward_tokens", counted_forward_tokens)
    with torch.no_grad():
        cartesian = model(image)
    assert calls == 1

    with torch.no_grad():
        raw = model.raw_hierarchical_polar_logits(image)
        polar = hierarchical_factors_to_joint_log_probabilities(raw)
        gathered = gather_polar_logits_to_cartesian(
            polar,
            factorization=model.factorization,
        )

    assert raw.shape == (1, 2, *POLAR_SHAPE)
    assert polar.shape == (1, 3, *POLAR_SHAPE)
    assert cartesian.shape == (1, 3, 64, 64)
    assert torch.isfinite(raw).all()
    assert torch.isfinite(polar).all()
    assert torch.isfinite(cartesian).all()
    assert torch.allclose(
        torch.logsumexp(polar, dim=1),
        torch.zeros_like(polar[:, 0]),
        atol=2e-6,
        rtol=0.0,
    )
    assert torch.equal(cartesian, gathered)
    support = model.cartesian_support_mask
    assert torch.allclose(
        torch.logsumexp(cartesian[:, :, support], dim=1),
        torch.zeros_like(cartesian[:, 0, support]),
        atol=2e-6,
        rtol=0.0,
    )
    outside = ~support
    minimum = torch.finfo(cartesian.dtype).min
    assert torch.equal(
        cartesian[0, 0][outside],
        torch.zeros_like(cartesian[0, 0][outside]),
    )
    assert torch.equal(
        cartesian[0, 1][outside],
        torch.full_like(cartesian[0, 1][outside], minimum),
    )
    assert torch.equal(
        cartesian[0, 2][outside],
        torch.full_like(cartesian[0, 2][outside], minimum),
    )


def test_backward_reaches_encoder_context_and_both_factor_rows() -> None:
    torch.manual_seed(41)
    model = CategoricalRadialPerceptionFullRayHierarchical().train()
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True)

    raw = model.raw_hierarchical_polar_logits(image)
    loss = raw[:, 0].square().mean() + raw[:, 1].square().mean()
    loss.backward()

    gradients = {
        "image": image.grad,
        "encoder": model.encoder.patch_embed.weight.grad,
        "token_projection": model.token_projection.weight.grad,
        "context_stem": model.context_stem[0].weight.grad,
        "full_ray_first": model.radial_context[0].radial_conv.weight.grad,
        "full_ray_last": model.radial_context[-1].pointwise.weight.grad,
        "known_factor": model.polar_head.weight.grad[0],
        "occupied_factor": model.polar_head.weight.grad[1],
    }
    for name, gradient in gradients.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


@pytest.mark.parametrize(
    "bad_factors",
    (
        torch.zeros(2),
        torch.zeros(1, 3, 2, 2),
        torch.zeros(1, 2, 2, 2, dtype=torch.int64),
    ),
)
def test_joint_conversion_rejects_invalid_factors(bad_factors: torch.Tensor) -> None:
    with pytest.raises((ValueError, TypeError), match="factors"):
        hierarchical_factors_to_joint_log_probabilities(bad_factors)


@pytest.mark.parametrize(
    "bad_state",
    (
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(2, 2, dtype=torch.uint8),
    ),
)
def test_comparable_builder_rejects_invalid_rng_state(bad_state: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="CPU uint8"):
        build_comparable_width24_and_hierarchical_models(bad_state)


def test_comparable_builder_rejects_non_tensor_rng_state() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        build_comparable_width24_and_hierarchical_models(  # type: ignore[arg-type]
            b"not a tensor"
        )
