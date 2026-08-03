from __future__ import annotations

from collections import OrderedDict
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.go2_dinov2_dense_shared_spatial_readout_calibration_v1 import (
    CONDITION_WIDTH,
    DenseSharedSpatialReadoutV1,
    INITIALIZATION_ORDER,
    PARAMETER_COUNT,
    PARAMETER_TENSOR_COUNT,
    PATCH_COUNT,
    RELATIONAL_WIDTH,
    dense_shared_state_identity_v1,
    initialize_dense_shared_spatial_readout_v1,
)


SEED = 2_026_080_303


def _random_inputs(
    batch: int = 3,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(91_703)
    relational = torch.randn(
        (batch, PATCH_COUNT, RELATIONAL_WIDTH),
        generator=generator,
        dtype=torch.float32,
    ).requires_grad_(requires_grad)
    condition = torch.randn(
        (batch, CONDITION_WIDTH),
        generator=generator,
        dtype=torch.float32,
    ).requires_grad_(requires_grad)
    return relational, condition


def test_parameter_inventory_positions_and_exact_xavier_draw_order() -> None:
    caller_rng = torch.random.get_rng_state().clone()
    model = initialize_dense_shared_spatial_readout_v1(SEED)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    parameters = dict(model.named_parameters())
    assert len(parameters) == PARAMETER_TENSOR_COUNT == 9
    assert PARAMETER_COUNT == 245
    assert (
        sum(parameter.numel() for parameter in parameters.values())
        == PARAMETER_COUNT
    )
    assert all(
        parameter.dtype == torch.float32 and parameter.requires_grad
        for parameter in parameters.values()
    )

    assert tuple(model.patch_positions.shape) == (256, 2)
    assert torch.equal(
        model.patch_positions[0],
        torch.tensor((-0.9375, -0.9375), dtype=torch.float32),
    )
    assert torch.equal(
        model.patch_positions[15],
        torch.tensor((0.9375, -0.9375), dtype=torch.float32),
    )
    assert torch.equal(
        model.patch_positions[240],
        torch.tensor((-0.9375, 0.9375), dtype=torch.float32),
    )
    assert torch.equal(
        model.patch_positions[-1],
        torch.tensor((0.9375, 0.9375), dtype=torch.float32),
    )
    assert not model.patch_positions.requires_grad

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for name in INITIALIZATION_ORDER:
        actual = getattr(model, name)
        expected_shape = actual.shape if actual.ndim == 2 else (1, actual.numel())
        expected = torch.empty(expected_shape, dtype=torch.float32)
        nn.init.xavier_uniform_(expected, gain=1.0, generator=generator)
        assert torch.equal(actual, expected.reshape(actual.shape))
    assert torch.count_nonzero(model.b_h) == 0
    assert torch.count_nonzero(model.b_score) == 0


def test_forward_matches_frozen_equations_and_is_repeatable() -> None:
    model = initialize_dense_shared_spatial_readout_v1(SEED)
    relational, condition = _random_inputs()

    output = model.forward_with_attention(relational, condition)
    positions = model.patch_positions.unsqueeze(0).expand(relational.shape[0], -1, -1)
    hidden = torch.tanh(
        F.linear(relational, model.W_r)
        + F.linear(positions, model.W_p)
        + F.linear(condition, model.W_q).unsqueeze(1)
        + model.b_h
    )
    expected_attention = torch.softmax(
        torch.einsum("bih,h->bi", hidden, model.w_alpha),
        dim=1,
    )
    expected_values = F.linear(relational, model.W_v)
    expected_pooled = torch.sum(
        expected_attention.unsqueeze(-1) * expected_values,
        dim=1,
    )
    expected_score = (
        torch.einsum("bi,i->b", expected_pooled, model.w_z)
        + torch.einsum("bi,ij,bj->b", expected_pooled, model.B, condition)
        + model.b_score
    )

    assert torch.equal(output.attention, expected_attention)
    assert torch.equal(output.pooled_value, expected_pooled)
    assert torch.equal(output.score, expected_score)
    assert torch.equal(model(relational, condition), expected_score)
    assert torch.equal(
        output.attention.sum(dim=1),
        torch.ones(relational.shape[0], dtype=torch.float32),
    )
    assert torch.equal(
        model(relational, condition),
        model(relational.clone(), condition.clone()),
    )


def test_dense_spatial_successor_and_condition_routes_are_sensitive() -> None:
    model = DenseSharedSpatialReadoutV1()
    with torch.no_grad():
        # Positional attention along u, with the first relational channel as value.
        model.W_p[0, 0] = 2.0
        model.w_alpha[0] = 2.0
        model.W_v[0, 0] = 1.0
        model.w_z[0] = 1.0
        # Explicit goal and requested-action bilinear routes.
        model.B[0, 0] = 1.0
        model.B[0, 2] = 1.0
        # A direct successor-token value route.
        model.W_v[1, 8] = 1.0
        model.w_z[1] = 1.0

    relational = torch.zeros((1, PATCH_COUNT, RELATIONAL_WIDTH))
    condition = torch.zeros((1, CONDITION_WIDTH))
    relational[0, 0, 0] = 1.0
    left_score = model(relational, condition)
    relational[0, 0, 0] = 0.0
    relational[0, 15, 0] = 1.0
    right_score = model(relational, condition)
    assert right_score.item() > left_score.item()

    relational.zero_()
    relational[0, 7, 8] = 1.0
    successor_score = model(relational, condition)
    relational[0, 7, 8] = 0.0
    assert successor_score.item() > model(relational, condition).item()

    relational[0, 15, 0] = 1.0
    base_score = model(relational, condition)
    goal_condition = condition.clone()
    goal_condition[0, 0] = 1.0
    action_condition = condition.clone()
    action_condition[0, 2] = 1.0
    assert model(relational, goal_condition).item() > base_score.item()
    assert model(relational, action_condition).item() > base_score.item()


def test_every_parameter_and_both_inputs_receive_finite_gradient() -> None:
    model = initialize_dense_shared_spatial_readout_v1(SEED)
    relational, condition = _random_inputs(requires_grad=True)
    output = model.forward_with_attention(relational, condition)
    loss = output.score.square().mean() + 0.01 * output.attention.square().mean()
    loss.backward()

    assert relational.grad is not None
    assert condition.grad is not None
    assert bool(torch.isfinite(relational.grad).all())
    assert bool(torch.isfinite(condition.grad).all())
    assert torch.count_nonzero(relational.grad) > 0
    assert torch.count_nonzero(condition.grad) > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_seed_and_state_identity_are_exact_and_mapping_order_independent() -> None:
    first = initialize_dense_shared_spatial_readout_v1(SEED)
    replay = initialize_dense_shared_spatial_readout_v1(SEED)
    different = initialize_dense_shared_spatial_readout_v1(SEED + 1)

    first_identity = dense_shared_state_identity_v1(first)
    assert len(first_identity) == 64
    assert first_identity == dense_shared_state_identity_v1(replay)
    assert first_identity != dense_shared_state_identity_v1(different)
    assert all(
        torch.equal(value, replay.state_dict()[name])
        for name, value in first.state_dict().items()
    )

    reversed_state = OrderedDict(reversed(tuple(first.state_dict().items())))
    assert dense_shared_state_identity_v1(reversed_state) == first_identity

    mutated_state = copy.deepcopy(first.state_dict())
    mutated_state["W_r"][0, 0] += 1.0
    assert dense_shared_state_identity_v1(mutated_state) != first_identity


@pytest.mark.parametrize(
    ("relational", "condition", "error"),
    (
        (torch.zeros((1, 255, 24)), torch.zeros((1, 4)), ValueError),
        (torch.zeros((1, 256, 23)), torch.zeros((1, 4)), ValueError),
        (torch.zeros((0, 256, 24)), torch.zeros((0, 4)), ValueError),
        (torch.zeros((2, 256, 24)), torch.zeros((1, 4)), ValueError),
        (
            torch.zeros((1, 256, 24), dtype=torch.float64),
            torch.zeros((1, 4)),
            TypeError,
        ),
        (
            torch.zeros((1, 256, 24)),
            torch.zeros((1, 4), dtype=torch.float64),
            TypeError,
        ),
    ),
)
def test_input_contract_rejects_wrong_shapes_and_dtypes(
    relational: torch.Tensor,
    condition: torch.Tensor,
    error: type[Exception],
) -> None:
    model = initialize_dense_shared_spatial_readout_v1(SEED)
    with pytest.raises(error):
        model(relational, condition)


def test_nonfinite_inputs_and_invalid_states_fail_closed() -> None:
    model = initialize_dense_shared_spatial_readout_v1(SEED)
    relational, condition = _random_inputs(batch=1)

    bad_relational = relational.clone()
    bad_relational[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        model(bad_relational, condition)
    bad_condition = condition.clone()
    bad_condition[0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        model(relational, bad_condition)

    missing_state = dict(model.state_dict())
    missing_state.pop("B")
    with pytest.raises(ValueError):
        dense_shared_state_identity_v1(missing_state)
    wrong_position_state = copy.deepcopy(model.state_dict())
    wrong_position_state["patch_positions"][0, 0] = 0.0
    with pytest.raises(ValueError):
        dense_shared_state_identity_v1(wrong_position_state)
    nonfinite_state = copy.deepcopy(model.state_dict())
    nonfinite_state["w_z"][0] = float("nan")
    with pytest.raises(FloatingPointError):
        dense_shared_state_identity_v1(nonfinite_state)


def test_seed_validation_is_strict() -> None:
    with pytest.raises(TypeError):
        initialize_dense_shared_spatial_readout_v1(True)
    with pytest.raises(TypeError):
        initialize_dense_shared_spatial_readout_v1(1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        initialize_dense_shared_spatial_readout_v1(-1)
    with pytest.raises(ValueError):
        initialize_dense_shared_spatial_readout_v1(2**63)
